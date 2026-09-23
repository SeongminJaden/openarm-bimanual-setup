# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Find a human hand with the wrist camera and publish where it is.

Deliberately NOT a copy of pen_detector with a different prompt.  That node
finds a pen by shape at true scale - long, thin, lying on the work surface -
and every one of those assumptions is wrong for a hand:

  * a hand is not long and thin, so the minimum-area-rectangle size test that
    identifies a pen rejects a hand outright;
  * a hand is usually held in the air, so the plane segmentation that keeps
    only what stands 3-40 mm proud of the bench finds nothing at all.

The robot's own gripper is the obvious false positive: it is black, roughly
hand shaped, and permanently in the bottom of the frame.  Appearance will not
separate it reliably, but geometry will - the fingers are bolted to the same
hand as the camera, so TF knows exactly where they are in the camera's own
frame.  Any candidate that lands within `self_exclude_m` of a link listed in
`self_links` is the robot, and is dropped.  That test does not care whether
the human is wearing a glove, and it keeps working as the gripper opens and
closes.

What is true of a hand is that it is a compact blob NEARER the camera than
whatever is behind it.  So the pipeline here is:

  1. YOLO-World says where the hand is, from a text prompt, with no training.
  2. Inside that box, the depth histogram is split: the hand is the near
     cluster, the bench or the wall behind it is the far one.  A low
     percentile picks the near cluster without assuming any absolute
     distance.
  3. Pixels within `depth_band_m` of that are the hand.  Their centroid, back
     projected through the intrinsics, is the point published.

Publishes the same topics as pen_detector, so pen_center can drive the arm at
a hand instead of a pen with nothing but a topic remap:

    ros2 run openarm_vision_pick pen_center --ros-args \\
      --params-file <config>/hand.yaml \\
      -p target_pose_topic:=/hand_detector/pose

Two ways to propose candidates, chosen with `detector_mode`:

  'depth'  no neural network at all.  Beyond the gripper cut-off the hand is
           simply the nearest hand-sized thing reaching into the workspace, so
           the near depth band is segmented and its blobs are measured.  Costs
           a couple of milliseconds, needs nothing installed, and cannot be
           fooled by what a thing LOOKS like - only by something else
           hand-sized being nearer.
  'yolo'   YOLO-World proposes boxes from a text prompt.  Slower and needs
           ultralytics plus a checkpoint, but it can tell a hand from a mug.
  'both'   YOLO boxes when it returns any, the depth blobs otherwise.

'depth' is the default because on this bench the only thing that reaches in
past the gripper IS a hand.  Switch to 'yolo' when that stops being true.

mediapipe would be the natural third option and is not offered: on this
machine it resolves to numpy 2.x and opencv 5.x, and the ROS 2 Humble
extension modules here are built against numpy 1.x, so installing it breaks
cv_bridge and the rest of the vision stack.
"""

import math
import os

# Pin the numeric libraries to one thread each, BEFORE numpy and OpenCV load -
# they read these at import time.  Same reasoning as pen_detector: every array
# here is small, and splitting that work across sixteen cores costs more in
# dispatch than it saves, while starving the camera driver of cores.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import cv2                                                      # noqa: E402
import numpy as np                                              # noqa: E402
import rclpy                                                    # noqa: E402
import tf2_ros                                                  # noqa: E402
from cv_bridge import CvBridge                                  # noqa: E402
from geometry_msgs.msg import PoseStamped                       # noqa: E402
from message_filters import (ApproximateTimeSynchronizer,       # noqa: E402
                             Subscriber)
from rcl_interfaces.msg import SetParametersResult              # noqa: E402
from rclpy.node import Node                                     # noqa: E402
from rclpy.qos import qos_profile_sensor_data                   # noqa: E402
from sensor_msgs.msg import CameraInfo, Image                   # noqa: E402
from visualization_msgs.msg import Marker                       # noqa: E402


class HandDetector(Node):

    def __init__(self):
        super().__init__('hand_detector')

        p = self.declare_parameter
        p('color_topic', '/camera/camera/color/image_raw')
        p('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        p('info_topic', '/camera/camera/color/camera_info')
        # The frame the images are in - always a camera optical frame.
        p('camera_frame', 'openarm_left_camera_color_optical_frame')
        # The frame poses are PUBLISHED in.  Leave it equal to camera_frame for
        # an eye-in-hand setup; set it to 'world' for a body-mounted camera,
        # where a camera-frame point is useless to anything that moves the arm.
        p('target_frame', 'openarm_left_camera_color_optical_frame')

        p('depth_scale', 0.001)
        p('min_depth_m', 0.07)
        p('max_depth_m', 1.20)

        # Open-vocabulary detection.  Several phrases are cheap - they are one
        # batched text embedding - and a hand is described differently
        # depending on whether it is open, gripping, or seen edge on.
        # 'depth' (no network), 'yolo', or 'both'.
        p('detector_mode', 'depth')
        # How many depth bands to walk outwards from the nearest surface
        # before giving up.  The first band that holds a hand-sized blob
        # which is not the robot itself wins.
        p('depth_band_steps', 5)
        p('near_floor_percentile', 2.0)

        p('yolo_model', '~/models/yolov8s-worldv2.pt')
        p('yolo_prompt', ['hand', 'human hand', 'person'])
        # Which prompt indices count as a hand.  'person' is in the list
        # because YOLO-World often scores an arm entering the frame far
        # higher as a person than as a hand; set this to [0, 1] to be strict.
        p('hand_class_ids', [0, 1, 2])
        p('yolo_conf', 0.05)
        p('yolo_period_s', 0.1)
        p('yolo_box_pad_px', 4)

        # The hand is the NEAR cluster inside the box.  This percentile of the
        # valid depths there picks it without assuming a distance; everything
        # within depth_band_m of that is taken as hand.
        p('near_percentile', 20.0)
        p('depth_band_m', 0.06)
        p('min_hand_px', 400)

        # Sanity on the blob's real size, longest side in metres.  A hand is
        # 70-220 mm across; anything outside that is a forearm, a torso, or a
        # bad box.
        p('hand_min_m', 0.060)
        p('hand_max_m', 0.250)

        # The robot's own gripper, in the camera's frame, straight from TF.
        # A detection within self_exclude_m of any of these is the robot
        # looking at itself.
        p('self_links', ['openarm_left_left_finger',
                         'openarm_left_right_finger',
                         'openarm_left_hand_tcp'])
        p('self_exclude_m', 0.12)

        p('ema_alpha', 0.4)
        p('stable_count', 4)
        p('stable_tol_m', 0.015)
        p('publish_debug_image', True)

        g = self.get_parameter
        self.cam_frame = g('camera_frame').value
        self.target_frame = g('target_frame').value
        self.depth_scale = g('depth_scale').value
        self.min_d = g('min_depth_m').value
        self.max_d = g('max_depth_m').value
        self.mode = str(g('detector_mode').value)
        self.band_steps = int(g('depth_band_steps').value)
        self.floor_pct = g('near_floor_percentile').value
        self.yolo_conf = g('yolo_conf').value
        self.yolo_period = g('yolo_period_s').value
        self.box_pad = int(g('yolo_box_pad_px').value)
        self.hand_ids = set(int(v) for v in g('hand_class_ids').value)
        self.near_pct = g('near_percentile').value
        self.band = g('depth_band_m').value
        self.min_px = int(g('min_hand_px').value)
        self.size_min = g('hand_min_m').value
        self.size_max = g('hand_max_m').value
        self.self_links = list(g('self_links').value)
        self.self_r = g('self_exclude_m').value
        self.alpha = g('ema_alpha').value
        self.stable_count = int(g('stable_count').value)
        self.stable_tol = g('stable_tol_m').value
        self.want_debug = g('publish_debug_image').value

        cv2.setNumThreads(4)
        self.bridge = CvBridge()
        self.K = None
        self.ema = None
        self.n_stable = 0
        self.boxes = []
        self._yolo_last = 0.0
        self._warned_align = False
        self._warned_self_tf = False
        self._last_tf_warn = 0.0

        self.prompt = list(g('yolo_prompt').value)
        weights = os.path.expanduser(g('yolo_model').value)
        # Ultralytics will pip-install missing extras at import time if it is
        # allowed to.  On a ROS box that is destructive - it has pulled in a
        # setuptools new enough to break colcon, mid-run.  Pin it shut.
        os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
        self.yolo = None
        try:
            if self.mode == 'depth':
                raise RuntimeError('detector_mode is depth; not loading it')
            from ultralytics import YOLOWorld
            self.yolo = YOLOWorld(weights)
            # Once only, and before any predict(): set_classes moves the text
            # features onto the model's current device, and calling it again
            # after a predict has pushed the model to CUDA raises
            # 'Expected all tensors to be on the same device'.
            self.yolo.set_classes(self.prompt)
            self.get_logger().info(
                "YOLO-World '{}' prompted with {}".format(weights,
                                                          self.prompt))
        except Exception as exc:
            if self.mode == 'depth':
                self.get_logger().info(
                    'detector_mode=depth: no network loaded, candidates come '
                    'from the depth image alone')
            else:
                self.get_logger().error(
                    'YOLO-World unavailable ({}); falling back to '
                    'detector_mode=depth.'.format(exc))
                self.mode = 'depth'

        self.add_on_set_parameters_callback(self._on_param_set)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_pose = self.create_publisher(PoseStamped, '~/pose', 10)
        self.pub_debug = self.create_publisher(Image, '~/debug_image', 2)
        self.pub_marker = self.create_publisher(Marker, '~/marker', 2)

        self.create_subscription(CameraInfo, g('info_topic').value,
                                 self._on_info, 10)
        sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, g('color_topic').value,
                        qos_profile=qos_profile_sensor_data),
             Subscriber(self, Image, g('depth_topic').value,
                        qos_profile=qos_profile_sensor_data)],
            queue_size=10, slop=0.15)
        sync.registerCallback(self._on_frame)

        self.get_logger().info(
            'colour=' + g('color_topic').value +
            '\n  depth gate {:.2f}..{:.2f} m'.format(self.min_d, self.max_d) +
            '\n  mode: ' + self.mode +
            '\n  hand size {:.0f}-{:.0f} mm'.format(self.size_min * 1000,
                                                    self.size_max * 1000) +
            '\n  images in ' + self.cam_frame +
            "\n  publishing poses in '" + self.target_frame + "'")

    # --------------------------------------------------------- live retuning
    _LIVE = {
        'min_depth_m': 'min_d', 'max_depth_m': 'max_d',
        'depth_scale': 'depth_scale',
        'yolo_conf': 'yolo_conf', 'yolo_period_s': 'yolo_period',
        'yolo_box_pad_px': 'box_pad',
        'near_percentile': 'near_pct', 'depth_band_m': 'band',
        'min_hand_px': 'min_px', 'self_exclude_m': 'self_r',
        'depth_band_steps': 'band_steps',
        'near_floor_percentile': 'floor_pct',
        'hand_min_m': 'size_min', 'hand_max_m': 'size_max',
        'ema_alpha': 'alpha', 'stable_count': 'stable_count',
        'stable_tol_m': 'stable_tol', 'publish_debug_image': 'want_debug',
    }
    _LIVE_INT = ('box_pad', 'min_px', 'stable_count', 'band_steps')

    def _on_param_set(self, params):
        for prm in params:
            if prm.name == 'yolo_prompt':
                # set_classes() cannot be called again once the model is on
                # CUDA, so refuse rather than appear to have changed it.
                return SetParametersResult(
                    successful=False,
                    reason='the prompt is fixed at start-up; restart the '
                           'node with -p yolo_prompt:=...')
            if prm.name == 'hand_class_ids':
                self.hand_ids = set(int(v) for v in prm.value)
                self.get_logger().info('hand_class_ids = {}'.format(
                    sorted(self.hand_ids)))
                continue
            attr = self._LIVE.get(prm.name)
            if attr is None:
                continue
            value = int(prm.value) if attr in self._LIVE_INT else prm.value
            setattr(self, attr, value)
            self.get_logger().info('{} = {}'.format(prm.name, value))
        self.ema = None
        self.n_stable = 0
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ input
    def _on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(
                'intrinsics fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}'.format(
                    self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]))

    def _to_target(self, p_cam):
        """Carry a camera-frame point into target_frame.

        Identity when the two are the same, which is the eye-in-hand case.
        Otherwise this is the step that makes the published pose mean
        anything to a planner: a point expressed in a moving camera's own
        frame is not a place in the world.
        """
        if self.target_frame == self.cam_frame:
            return p_cam
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, self.cam_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
        except Exception as exc:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_tf_warn > 5.0:
                self._last_tf_warn = now
                self.get_logger().warn(
                    'no TF {} <- {} ({}); not publishing'.format(
                        self.target_frame, self.cam_frame, exc))
            return None
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        t = tf.transform.translation
        return R @ p_cam + np.array([t.x, t.y, t.z])

    def _depth_regions(self, d_m, valid):
        """Candidate boxes from the depth image, nearest band first.

        No appearance model: past the gripper cut-off, a hand is whatever
        hand-sized thing is closest to the camera.  Bands are walked outwards
        so a hand in front of the bench is found before the bench is.
        """
        if np.count_nonzero(valid) < self.min_px:
            return []
        zs = d_m[valid]
        z0 = float(np.percentile(zs, self.floor_pct))
        h, w = d_m.shape
        out = []
        for k in range(self.band_steps):
            lo = z0 + k * 0.5 * self.band
            band = valid & (d_m >= lo) & (d_m < lo + self.band)
            if np.count_nonzero(band) < self.min_px:
                continue
            m = (band.astype(np.uint8)) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area < self.min_px:
                    continue
                out.append((int(x), int(y), int(min(w, x + bw)),
                            int(min(h, y + bh)), 1.0 - 0.05 * k, -1))
            if out:
                break          # nearest band that produced anything wins
        return out

    def _self_points(self, stamp):
        """The robot's own gripper links, as points in the camera frame.

        Returns [] when the robot is not running - then there is no TF, and
        the test is simply skipped rather than rejecting everything.
        """
        out = []
        for link in self.self_links:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.cam_frame, link, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05))
            except Exception:
                continue
            t = tf.transform.translation
            out.append(np.array([t.x, t.y, t.z]))
        if not out and not self._warned_self_tf:
            self._warned_self_tf = True
            self.get_logger().warn(
                'no TF from {} to any of {} - the robot\'s own gripper will '
                'NOT be excluded. Start the robot description (move_group or '
                'robot_state_publisher) to enable it.'.format(
                    self.cam_frame, self.self_links))
        return out

    def _yolo_rois(self, color):
        """Boxes YOLO-World calls a hand, refreshed on a timer."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.boxes and (now - self._yolo_last) < self.yolo_period:
            return self.boxes
        self._yolo_last = now

        h, w = color.shape[:2]
        pad = self.box_pad
        out = []
        try:
            res = self.yolo.predict(color, conf=self.yolo_conf, verbose=False)
        except Exception as exc:
            self.get_logger().warn('YOLO inference failed: {}'.format(exc))
            return self.boxes

        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls = int(b.cls[0]) if b.cls is not None else 0
                if cls not in self.hand_ids:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                out.append((max(0, int(x1) - pad), max(0, int(y1) - pad),
                            min(w, int(x2) + pad), min(h, int(y2) + pad),
                            float(b.conf[0]), cls))
        return out

    # ----------------------------------------------------------------- detect
    def _on_frame(self, color_msg, depth_msg):
        # Only the intrinsics are mandatory.  self.yolo is None whenever
        # detector_mode is 'depth', which is the default - guarding on it
        # here made the callback return on every frame and the node publish
        # nothing at all while looking perfectly healthy in the log.
        if self.K is None:
            return
        if self.mode == 'yolo' and self.yolo is None:
            return
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg,
                                          desired_encoding='passthrough')
        if depth.shape[:2] != color.shape[:2]:
            if not self._warned_align:
                self._warned_align = True
                self.get_logger().warn(
                    'depth {} does not match colour {} - start the camera '
                    'with align_depth.enable:=true'.format(depth.shape[:2],
                                                           color.shape[:2]))
            return

        d_m = depth.astype(np.float32)
        if depth.dtype == np.uint16:
            d_m = d_m * self.depth_scale
        valid = (d_m > self.min_d) & (d_m < self.max_d) & np.isfinite(d_m)

        if self.mode == 'depth' or self.yolo is None:
            self.boxes = self._depth_regions(d_m, valid)
        else:
            self.boxes = self._yolo_rois(color)
            if not self.boxes and self.mode == 'both':
                self.boxes = self._depth_regions(d_m, valid)
        self_pts = self._self_points(color_msg.header.stamp)
        best = None            # (score, mask, box, size_m, z)
        rejects = []
        for (x1, y1, x2, y2, conf, cls) in self.boxes:
            sub_valid = valid[y1:y2, x1:x2]
            if np.count_nonzero(sub_valid) < self.min_px:
                rejects.append((x1, y1, x2, y2, conf, 'no depth'))
                continue

            # The hand is the near cluster inside the box; a low percentile
            # finds it without assuming how far away it is.
            zz = d_m[y1:y2, x1:x2][sub_valid]
            z_near = float(np.percentile(zz, self.near_pct))
            near = sub_valid & (np.abs(d_m[y1:y2, x1:x2] - z_near) < self.band)
            if np.count_nonzero(near) < self.min_px:
                rejects.append((x1, y1, x2, y2, conf, 'near cluster tiny'))
                continue

            m = np.zeros(d_m.shape, np.uint8)
            m[y1:y2, x1:x2] = near.astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                rejects.append((x1, y1, x2, y2, conf, 'no blob'))
                continue
            cnts = [c for c in cnts if cv2.contourArea(c) >= self.min_px]
            if not cnts:
                rejects.append((x1, y1, x2, y2, conf, 'blob tiny'))
                continue
            # Largest first, but every one is judged: a box can hold the hand
            # and a piece of bench, and the bench is often the bigger blob.
            cnts.sort(key=cv2.contourArea, reverse=True)
            c = cnts[0]

            blob = np.zeros(d_m.shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 255, -1)
            sel = (blob > 0) & valid
            if np.count_nonzero(sel) < self.min_px:
                rejects.append((x1, y1, x2, y2, conf, 'blob has no depth'))
                continue

            z = float(np.median(d_m[sel]))
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            size_m = max(rw, rh) * z / self.K[0, 0]
            if not (self.size_min <= size_m <= self.size_max):
                rejects.append((x1, y1, x2, y2, conf,
                                '{:.0f} mm'.format(size_m * 1000)))
                continue

            # Is this the robot's own gripper?  Compare in 3-D, not in the
            # image: a human hand can sit at the same range as the fingers
            # and still be nowhere near them in space.
            ys_c, xs_c = np.nonzero(sel)
            u_c, v_c = float(xs_c.mean()), float(ys_c.mean())
            cand = np.array([(u_c - self.K[0, 2]) * z / self.K[0, 0],
                             (v_c - self.K[1, 2]) * z / self.K[1, 1], z])
            # self_pts are in the camera frame too, so this stays a
            # like-for-like comparison whatever target_frame is set to.
            d_self = min((float(np.linalg.norm(cand - sp))
                          for sp in self_pts), default=None)
            if d_self is not None and d_self < self.self_r:
                rejects.append((x1, y1, x2, y2, conf,
                                'self {:.0f} mm'.format(d_self * 1000)))
                continue

            # Nearest hand wins: with two in view the arm should go for the
            # one it can actually reach.
            if best is None or z < best[0]:
                best = (z, sel, (x1, y1, x2, y2, conf, cls), size_m, c)

        if best is None:
            self.n_stable = 0
            self._debug(color, None, rejects, None, self_pts)
            return

        z, sel, box, size_m, contour = best
        ys, xs = np.nonzero(sel)
        u, v = float(xs.mean()), float(ys.mean())
        x = (u - self.K[0, 2]) * z / self.K[0, 0]
        y = (v - self.K[1, 2]) * z / self.K[1, 1]
        p = self._to_target(np.array([x, y, z]))
        if p is None:
            self.n_stable = 0
            return

        # Smooth, then require the reading to settle before publishing, so a
        # single bad frame cannot yank the arm.
        self.ema = p if self.ema is None else \
            self.alpha * p + (1.0 - self.alpha) * self.ema
        if np.linalg.norm(p - self.ema) < self.stable_tol:
            self.n_stable += 1
        else:
            self.n_stable = 0

        self._debug(color, (box, size_m, contour, u, v, z), rejects,
                    self.ema, self_pts)

        if self.n_stable < self.stable_count:
            return

        msg = PoseStamped()
        msg.header.stamp = color_msg.header.stamp
        msg.header.frame_id = self.target_frame
        msg.pose.position.x = float(self.ema[0])
        msg.pose.position.y = float(self.ema[1])
        msg.pose.position.z = float(self.ema[2])
        msg.pose.orientation.w = 1.0
        self.pub_pose.publish(msg)

        mk = Marker()
        mk.header = msg.header
        mk.ns, mk.id, mk.type, mk.action = 'hand', 0, Marker.SPHERE, Marker.ADD
        mk.pose = msg.pose
        mk.scale.x = mk.scale.y = mk.scale.z = 0.05
        mk.color.g, mk.color.a = 1.0, 0.8
        self.pub_marker.publish(mk)

    # ------------------------------------------------------------------ debug
    def _debug(self, color, hit, rejects, ema, self_pts=()):
        if not self.want_debug or self.pub_debug.get_subscription_count() == 0:
            return
        dbg = color.copy()

        for x1, y1, x2, y2, conf, why in rejects:
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(dbg, '{:.2f} {}'.format(conf, why),
                        (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

        if hit is not None:
            (x1, y1, x2, y2, conf, cls), size_m, contour, u, v, z = hit
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.drawContours(dbg, [contour], -1, (0, 255, 0), 2)
            cv2.circle(dbg, (int(u), int(v)), 7, (0, 0, 255), -1)
            cv2.putText(dbg, '{} {:.2f}  {:.0f} mm  z={:.3f} m'.format(
                self.prompt[cls] if cls < len(self.prompt) else '?',
                conf, size_m * 1000, z),
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if ema is not None:
                cv2.putText(dbg, '[{:+.3f} {:+.3f} {:+.3f}]  stable {}/{}'
                            .format(ema[0], ema[1], ema[2], self.n_stable,
                                    self.stable_count),
                            (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
        else:
            cv2.putText(dbg, 'no hand', (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        # Where TF says the robot's own gripper is: anything inside these
        # circles is rejected as self.
        for sp in self_pts:
            if sp[2] <= 0.01:
                continue
            uu = int(sp[0] * self.K[0, 0] / sp[2] + self.K[0, 2])
            vv = int(sp[1] * self.K[1, 1] / sp[2] + self.K[1, 2])
            rr = int(self.self_r * self.K[0, 0] / sp[2])
            cv2.circle(dbg, (uu, vv), max(3, rr), (0, 128, 255), 1)
            cv2.drawMarker(dbg, (uu, vv), (0, 128, 255), cv2.MARKER_TILTED_CROSS,
                           10, 1)

        # The image centre, which is what pen_center drives the hand towards.
        h, w = dbg.shape[:2]
        cx, cy = int(self.K[0, 2]), int(self.K[1, 2])
        cv2.drawMarker(dbg, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 24, 1)

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, 'bgr8'))


def main():
    rclpy.init()
    node = HandDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
