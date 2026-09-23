# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Shape-based pen detector for the OpenArm wrist camera.

Colour alone cannot find a black pen on this robot: the gripper fingers and
the base plate are black too, so a brightness threshold returns all three.
What separates a pen from both of them is its shape at true scale - long,
thin, and lying on the work surface.

So the pipeline is:

  1. depth gate    keep only pixels in a distance window.  The fingers sit far
                   closer to the camera than the work surface does, which
                   removes them outright.
  2. dark cue      a loose brightness mask, used as one cue and not as the
                   decision.  Set use_dark_mask false for a light pen.
  3. metric shape  every contour is fitted with a minimum-area rectangle and
                   its pixel size converted to millimetres through the depth
                   at that contour and the camera intrinsics.  Only pen-sized
                   boxes survive, which is what rejects the base plate.
  4. pose          the rectangle's long axis gives the heading the gripper has
                   to close across.  A blob centroid cannot provide that, and
                   without it a pen cannot be grasped at all.

Publishes the same topics as object_detector, so pick_and_place can consume it
unchanged by pointing target_pose_topic at this node.
"""

import math
import os

# Pin the numeric libraries to one thread each, BEFORE numpy and OpenCV load
# - they read these at import time and a later change is ignored.
#
# Every array this node touches is small.  The plane fit multiplies a
# 24000x3 matrix by a 3-vector, eighty times per frame; splitting work that
# size across sixteen cores costs far more in dispatch and spin-wait than it
# saves.  Measured on this machine: the same fit takes 47.5 ms with the
# default thread pool and 11.5 ms pinned to one thread, and the process as a
# whole was burning 900% CPU - enough to starve the RealSense driver of the
# cores it needs, dragging the colour stream from 30 Hz down to 5.8 Hz.
#
# YOLO is unaffected because it runs on the GPU: measured 16 ms per frame at
# 1, 4 and 8 torch threads alike.
#
# Set the variable yourself before launching to override any of these.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker


def _quat_from_matrix(m):
    """Rotation matrix (3x3) -> (x, y, z, w). Shepperd's method."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s,
                (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
            0.25 * s, (m[1, 0] - m[0, 1]) / s)


class PenDetector(Node):

    def __init__(self):
        super().__init__('pen_detector')
        p = self.declare_parameter
        cam = '/camera/camera'
        p('color_topic', cam + '/color/image_raw')
        p('depth_topic', cam + '/aligned_depth_to_color/image_raw')
        p('info_topic', cam + '/color/camera_info')
        # realsense2_camera derives its frame ids from camera_name even when
        # the topics stay in the default namespace, so this is usually
        # openarm_left_camera_color_optical_frame, not camera_*.
        p('target_frame', 'openarm_left_camera_color_optical_frame')

        # Depth gate.  The near limit is what removes the gripper: the fingers
        # sit within ~120 mm of the wrist camera, the work surface is further.
        p('min_depth_m', 0.15)
        p('max_depth_m', 0.60)
        p('depth_scale', 0.001)
        p('depth_patch', 7)

        # Plane segmentation - the colour-free way to pick candidates.  A
        # plane is fitted to the depth cloud by RANSAC and only points sitting
        # a few millimetres proud of it are kept.  That separates the three
        # black things in the scene without ever looking at a pixel value: the
        # pen stands ~10 mm off the surface, the base plate lies flat on it
        # (below height_min), and the fingers float far above (above
        # height_max, and usually already gone at the depth gate).
        p('use_plane_segmentation', True)
        p('plane_tol_m', 0.004)      # RANSAC inlier band
        p('plane_iters', 80)
        p('plane_stride', 4)         # subsample for the fit, for speed
        p('height_min_m', 0.003)
        p('height_max_m', 0.040)

        # A pen, in metres.  The ranges are generous because the camera sees
        # the pen obliquely, which foreshortens its length.
        p('pen_len_min_m', 0.100)
        p('pen_len_max_m', 0.180)
        p('pen_wid_min_m', 0.005)
        p('pen_wid_max_m', 0.020)
        # Long/short ratio.  A finger is stubby, the base plate is square-ish.
        p('min_aspect', 5.0)

        # Brightness cue, off by default.  It narrows the search when the pen
        # really is darker than the surface, but it assumes that, and the
        # assumption is exactly what fails here - the fingers and the base
        # plate are the same black.  Plane segmentation needs no such
        # assumption, so leave this off unless it earns its keep.
        p('use_dark_mask', False)
        p('dark_v_max', 90)
        p('dark_s_max', 120)

        # YOLO-World: open-vocabulary detection from a text prompt, no
        # training.  It answers "which of these is the pen", which geometry
        # alone cannot when several objects are the same size and height.
        #
        # It does NOT answer "at what angle": YOLO-World emits axis-aligned
        # boxes only, and a diagonal pen's axis-aligned box is a big square.
        # So the box is used as a region of interest and the silhouette inside
        # it still comes from the plane cut, which is what the minimum-area
        # rectangle is then fitted to.
        p('use_yolo', False)
        p('yolo_model', 'yolov8s-worldv2.pt')
        p('yolo_prompt', ['pen'])
        p('yolo_conf', 0.05)
        # Inference is ~0.3 s on this CPU, so it runs on a timer and the boxes
        # are reused between runs.  The scene is static; the arm is not.
        p('yolo_period_s', 0.5)
        p('yolo_roi_pad_px', 8)
        # When the plane cut finds nothing inside the box - a pen lying on a
        # surface the camera barely sees - fall back to thresholding the box
        # interior, which still yields a silhouette to fit.
        p('yolo_fallback_otsu', True)

        # Image-space crop, in pixels, -1 for the image edge.  The gripper
        # fingers are bolted to the same hand as the camera, so they never
        # leave their corner of the frame no matter where the arm goes.  A
        # depth gate can only push them out when they happen to be nearer than
        # everything else; a crop removes them unconditionally.  Watch the
        # debug image, note where the fingers sit, and cut them off here.
        p('roi_x0', -1)
        p('roi_y0', -1)
        p('roi_x1', -1)
        p('roi_y1', -1)

        p('min_area_px', 150)
        p('max_area_px', 60000)

        p('ema_alpha', 0.4)
        p('stable_count', 6)
        p('stable_tol_m', 0.008)
        p('publish_debug_image', True)

        g = self.get_parameter
        self.target_frame = g('target_frame').value
        self.min_d = g('min_depth_m').value
        self.max_d = g('max_depth_m').value
        self.depth_scale = g('depth_scale').value
        self.patch = int(g('depth_patch').value)
        self.len_min = g('pen_len_min_m').value
        self.len_max = g('pen_len_max_m').value
        self.wid_min = g('pen_wid_min_m').value
        self.wid_max = g('pen_wid_max_m').value
        self.min_aspect = g('min_aspect').value
        self.use_plane = g('use_plane_segmentation').value
        self.plane_tol = g('plane_tol_m').value
        self.plane_iters = int(g('plane_iters').value)
        self.plane_stride = int(g('plane_stride').value)
        self.h_min = g('height_min_m').value
        self.h_max = g('height_max_m').value
        self.use_yolo = g('use_yolo').value
        self.yolo_conf = g('yolo_conf').value
        self.yolo_period = g('yolo_period_s').value
        self.yolo_pad = int(g('yolo_roi_pad_px').value)
        self.yolo_otsu = g('yolo_fallback_otsu').value
        self.use_dark = g('use_dark_mask').value
        self.dark_v = int(g('dark_v_max').value)
        self.dark_s = int(g('dark_s_max').value)
        self.roi_x0 = int(g('roi_x0').value)
        self.roi_y0 = int(g('roi_y0').value)
        self.roi_x1 = int(g('roi_x1').value)
        self.roi_y1 = int(g('roi_y1').value)
        self.min_area = g('min_area_px').value
        self.max_area = g('max_area_px').value
        self.alpha = g('ema_alpha').value
        self.stable_count = int(g('stable_count').value)
        self.stable_tol = g('stable_tol_m').value
        self.want_debug = g('publish_debug_image').value

        # OpenCV keeps its own pool, set after import.  Its work here is
        # sub-millisecond either way, but four threads measured slightly
        # ahead of one and of sixteen, and it keeps cv2 from contending with
        # the camera driver for every core on the machine.
        cv2.setNumThreads(4)

        self.bridge = CvBridge()
        self.K = None
        self.ema = None
        self.n_stable = 0
        self._warned_depth = False
        self._warned_align = False
        self._last_tf_warn = 0.0
        self.plane_inliers = 0

        # YOLO-World, loaded once.  Kept optional on purpose: ultralytics and
        # torch are a couple of gigabytes, and the plane path works without
        # them, so a missing import degrades instead of killing the node.
        self.yolo = None
        self.yolo_boxes = []
        self._yolo_last = 0.0
        if self.use_yolo:
            prompt = list(g('yolo_prompt').value)
            weights = os.path.expanduser(g('yolo_model').value)
            # Ultralytics will pip-install missing extras at import time if it
            # is allowed to.  On a ROS box that is destructive: it pulled in a
            # setuptools new enough to break colcon, mid-run.  Pin it shut.
            os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
            try:
                from ultralytics import YOLOWorld
                self.yolo = YOLOWorld(weights)
                self.yolo.set_classes(prompt)   # this is the zero-shot part
                self.get_logger().info(
                    "YOLO-World '{}' prompted with {}".format(weights, prompt))
            except Exception as exc:
                self.get_logger().error(
                    'YOLO-World unavailable ({}); falling back to plane '
                    'segmentation alone. Install with: '
                    'pip install ultralytics'.format(exc))
                self.use_yolo = False

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
            '\n  depth=' + g('depth_topic').value +
            '\n  depth gate {:.2f}..{:.2f} m'.format(self.min_d, self.max_d) +
            '\n  pen {:.0f}-{:.0f} mm long, {:.0f}-{:.0f} mm wide, aspect >= {}'
            .format(self.len_min * 1000, self.len_max * 1000,
                    self.wid_min * 1000, self.wid_max * 1000, self.min_aspect) +
            '\n  segmentation: ' +
            ('plane, keeping {:.0f}-{:.0f} mm above the surface'.format(
                self.h_min * 1000, self.h_max * 1000)
             if self.use_plane else 'depth gate only') +
            '\n  yolo gate: ' + ('on' if self.use_yolo else 'off') +
            '\n  dark cue ' + ('on' if self.use_dark else 'off') +
            "\n  publishing poses in '" + self.target_frame + "'")

    def _roi(self, shape):
        """ROI clamped to the image, as (x0, y0, x1, y1); -1 means the edge."""
        h, w = shape[:2]
        x0 = 0 if self.roi_x0 < 0 else min(self.roi_x0, w - 1)
        y0 = 0 if self.roi_y0 < 0 else min(self.roi_y0, h - 1)
        x1 = w if self.roi_x1 < 0 else min(self.roi_x1, w)
        y1 = h if self.roi_y1 < 0 else min(self.roi_y1, h)
        if x1 <= x0 or y1 <= y0:      # nonsense crop: ignore it rather than
            return 0, 0, w, h         # blank the frame and report nothing
        return x0, y0, x1, y1

    # Tuning values that may be changed while the node runs.  Every one of
    # these was read into an attribute at start-up, so `ros2 param set` alone
    # had no effect - which made bench tuning one restart per attempt.
    _LIVE = {
        'min_depth_m': 'min_d', 'max_depth_m': 'max_d',
        'depth_scale': 'depth_scale', 'depth_patch': 'patch',
        'pen_len_min_m': 'len_min', 'pen_len_max_m': 'len_max',
        'pen_wid_min_m': 'wid_min', 'pen_wid_max_m': 'wid_max',
        'min_aspect': 'min_aspect',
        'use_plane_segmentation': 'use_plane', 'plane_tol_m': 'plane_tol',
        'plane_iters': 'plane_iters', 'plane_stride': 'plane_stride',
        'height_min_m': 'h_min', 'height_max_m': 'h_max',
        'use_dark_mask': 'use_dark', 'dark_v_max': 'dark_v',
        'dark_s_max': 'dark_s',
        'use_yolo': 'use_yolo', 'yolo_conf': 'yolo_conf',
        'yolo_period_s': 'yolo_period', 'yolo_roi_pad_px': 'yolo_pad',
        'yolo_fallback_otsu': 'yolo_otsu',
        'roi_x0': 'roi_x0', 'roi_y0': 'roi_y0',
        'roi_x1': 'roi_x1', 'roi_y1': 'roi_y1',
        'min_area_px': 'min_area', 'max_area_px': 'max_area',
        'ema_alpha': 'alpha', 'stable_count': 'stable_count',
        'stable_tol_m': 'stable_tol', 'publish_debug_image': 'want_debug',
    }
    _LIVE_INT = ('patch', 'plane_iters', 'plane_stride', 'dark_v', 'dark_s',
                 'yolo_pad', 'stable_count',
                 'roi_x0', 'roi_y0', 'roi_x1', 'roi_y1')

    def _on_param_set(self, params):
        for prm in params:
            attr = self._LIVE.get(prm.name)
            if attr is None:
                continue
            # The weights load once, at start-up.  Switching YOLO on afterwards
            # would find self.yolo empty and quietly gate every mask to
            # nothing, so refuse rather than pretend it worked.
            if prm.name == 'use_yolo' and prm.value and self.yolo is None:
                return SetParametersResult(
                    successful=False,
                    reason='YOLO-World was not loaded at start-up; restart '
                           'the node with use_yolo:=true')
            value = int(prm.value) if attr in self._LIVE_INT else prm.value
            setattr(self, attr, value)
            self.get_logger().info('{} = {}'.format(prm.name, value))
        # The EMA was accumulated under the old settings; drop it so a retune
        # shows up at once instead of being dragged back by stale history.
        self.ema = None
        self.n_stable = 0
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ inputs
    def _on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(
                'intrinsics fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}'.format(
                    self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]))

    def _depth_m(self, depth, mask):
        """Median depth over a mask, in metres, holes discarded."""
        vals = depth[mask > 0].astype(np.float64)
        if depth.dtype == np.uint16:
            vals = vals * self.depth_scale
        good = vals[np.isfinite(vals) & (vals > self.min_d) & (vals < self.max_d)]
        return float(np.median(good)) if good.size >= 5 else None

    def _yolo_rois(self, color):
        """Boxes YOLO-World calls a pen, refreshed on a timer.

        Inference costs a few hundred milliseconds on a CPU-only machine,
        which is far slower than the camera.  The scene the boxes describe
        barely moves, so they are recomputed periodically and reused in
        between rather than blocking every frame.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.yolo_boxes and (now - self._yolo_last) < self.yolo_period:
            return self.yolo_boxes
        self._yolo_last = now

        h, w = color.shape[:2]
        pad = self.yolo_pad
        out = []
        try:
            res = self.yolo.predict(color, conf=self.yolo_conf, verbose=False)
        except Exception as exc:
            self.get_logger().warn('YOLO inference failed: {}'.format(exc))
            return self.yolo_boxes

        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                conf = float(b.conf[0])
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                out.append((max(0, int(x1) - pad), max(0, int(y1) - pad),
                            min(w, int(x2) + pad), min(h, int(y2) + pad), conf))
        return out

    def _height_above_plane(self, d_m, valid):
        """Per-pixel height above the dominant plane, in metres.

        RANSAC rather than a least-squares fit, because the objects sitting on
        the surface are exactly the outliers a least-squares fit would let
        drag the plane upwards - and dragging it up by even 3 mm hides a pen.

        Returns None when the surface is not visible enough to trust.
        """
        h, w = d_m.shape
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        X = (us - cx) * d_m / fx
        Y = (vs - cy) * d_m / fy
        pts_all = np.stack((X, Y, d_m), axis=-1)

        s = max(1, self.plane_stride)
        sub = pts_all[::s, ::s][valid[::s, ::s]]
        if sub.shape[0] < 200:
            return None

        rng = np.random.default_rng(0)
        best_n, best_d, best_cnt = None, 0.0, 0
        for _ in range(self.plane_iters):
            idx = rng.integers(0, sub.shape[0], 3)
            a, b, c = sub[idx]
            n = np.cross(b - a, c - a)
            ln = np.linalg.norm(n)
            if ln < 1e-9:
                continue
            n = n / ln
            d = -float(n @ a)
            cnt = int(np.count_nonzero(np.abs(sub @ n + d) < self.plane_tol))
            if cnt > best_cnt:
                best_n, best_d, best_cnt = n, d, cnt

        # A plane needs to explain most of the view to be the work surface.
        if best_n is None or best_cnt < 0.25 * sub.shape[0]:
            return None

        # Refit on the inliers: three random points set the hypothesis, but
        # the whole inlier set gives a far steadier normal, and the height
        # band below is only a few millimetres wide.
        inl = sub[np.abs(sub @ best_n + best_d) < self.plane_tol]
        centre = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - centre, full_matrices=False)
        best_n = vt[2]
        best_d = -float(best_n @ centre)

        # Point the normal at the camera, so "above the surface" is positive.
        if best_d < 0.0:
            best_n, best_d = -best_n, -best_d

        self.plane_inliers = int(inl.shape[0])
        return pts_all @ best_n + best_d

    # ------------------------------------------------------------------ detect
    def _on_frame(self, color_msg, depth_msg):
        if self.K is None:
            return
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        if depth.shape[:2] != color.shape[:2]:
            if not self._warned_align:
                self._warned_align = True
                self.get_logger().warn(
                    'depth {} does not match colour {} - start the camera with '
                    'align_depth.enable:=true'.format(depth.shape[:2],
                                                      color.shape[:2]))
            return
        if depth.dtype == np.uint16 and not self._warned_depth:
            self._warned_depth = True
            self.get_logger().info(
                'depth is 16UC1, applying depth_scale={} m/unit'.format(
                    self.depth_scale))

        d_m = depth.astype(np.float32)
        if depth.dtype == np.uint16:
            d_m = d_m * self.depth_scale

        # 1. depth gate.
        valid = (d_m > self.min_d) & (d_m < self.max_d) & np.isfinite(d_m)

        # 1b. image crop.  Applied to `valid`, not just to the finished mask,
        #     so the fingers stay out of the plane fit below as well -
        #     otherwise RANSAC can lock onto the finger faces instead of the
        #     work surface when they fill much of the frame.
        x0, y0, x1, y1 = self._roi(valid.shape)
        if (x0, y0, x1, y1) != (0, 0, valid.shape[1], valid.shape[0]):
            keep = np.zeros(valid.shape, bool)
            keep[y0:y1, x0:x1] = True
            valid &= keep

        mask = valid.astype(np.uint8) * 255
        self.plane_inliers = 0

        # 2. plane segmentation: keep only what stands proud of the surface.
        #    No pixel value is consulted, so the pen, the base plate and the
        #    fingers are separated by geometry alone.
        if self.use_plane:
            hgt = self._height_above_plane(d_m, valid)
            if hgt is None:
                self._debug(color, mask, None, [], None)
                return
            mask = ((hgt > self.h_min) & (hgt < self.h_max) &
                    valid).astype(np.uint8) * 255

        # 3. brightness cue, only if explicitly asked for.
        if self.use_dark:
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            dark = ((hsv[:, :, 2] <= self.dark_v) &
                    (hsv[:, :, 1] <= self.dark_s)).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, dark)

        # 4. YOLO-World: keep only what falls inside a box it called a pen.
        #    This is identification, not localisation - the silhouette that
        #    the rectangle gets fitted to is still the geometric mask above.
        if self.use_yolo:
            self.yolo_boxes = self._yolo_rois(color)
            roi = np.zeros(mask.shape, np.uint8)
            for x1, y1, x2, y2, _ in self.yolo_boxes:
                cv2.rectangle(roi, (x1, y1), (x2, y2), 255, -1)
            if self.yolo_boxes:
                inside = cv2.bitwise_and(mask, roi)
                if np.count_nonzero(inside) >= self.min_area:
                    mask = inside
                elif self.yolo_otsu:
                    # The box is trustworthy but the plane cut came back empty
                    # inside it.  Threshold the box interior instead, so there
                    # is still a silhouette to fit a rectangle to.
                    grey = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                    inv = np.zeros(mask.shape, np.uint8)
                    for x1, y1, x2, y2, _ in self.yolo_boxes:
                        sub = grey[y1:y2, x1:x2]
                        if sub.size < self.min_area:
                            continue
                        _, th = cv2.threshold(
                            sub, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        inv[y1:y2, x1:x2] = th
                    mask = cv2.bitwise_and(inv, valid.astype(np.uint8) * 255)
                else:
                    mask = inside
            else:
                mask = np.zeros_like(mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        fx, fy = self.K[0, 0], self.K[1, 1]
        best = None
        rejects = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), ang = rect
            if rw < 1e-6 or rh < 1e-6:
                continue
            long_px, short_px = max(rw, rh), min(rw, rh)
            aspect = long_px / short_px

            cmask = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(cmask, [c], -1, 255, -1)
            z = self._depth_m(depth, cmask)
            if z is None:
                continue

            # pixels -> metres at that range
            long_m = long_px * z / fx
            short_m = short_px * z / fy
            entry = (long_m, short_m, aspect, z, rect)

            if (self.len_min <= long_m <= self.len_max and
                    self.wid_min <= short_m <= self.wid_max and
                    aspect >= self.min_aspect):
                # closest to the middle of the accepted length band wins
                mid = 0.5 * (self.len_min + self.len_max)
                score = abs(long_m - mid)
                if best is None or score < best[0]:
                    best = (score, entry)
            else:
                rejects.append(entry)

        if best is None:
            self.n_stable = 0
            self._debug(color, mask, None, rejects, None)
            return

        long_m, short_m, aspect, z, rect = best[1]
        (cx, cy), (rw, rh), ang = rect

        # 3. back-project the centre through the intrinsics.
        x = (cx - self.K[0, 2]) * z / fx
        y = (cy - self.K[1, 2]) * z / fy
        p_cam = np.array([x, y, z])

        # 4. the long axis as a yaw in the image plane.  minAreaRect reports
        #    the angle of the width edge, so rotate when height is the longer.
        yaw_img = math.radians(ang if rw >= rh else ang + 90.0)

        p_out, quat = self._to_target(p_cam, yaw_img, color_msg.header.frame_id)
        if p_out is None:
            self._debug(color, mask, best[1], rejects, None)
            return

        # 5. smooth, and only publish once it has settled.
        if self.ema is None:
            self.ema = p_out
        else:
            moved = float(np.linalg.norm(p_out - self.ema))
            self.ema = self.alpha * p_out + (1.0 - self.alpha) * self.ema
            self.n_stable = self.n_stable + 1 if moved < self.stable_tol else 0

        self._debug(color, mask, best[1], rejects,
                    (int(round(cx)), int(round(cy)), z, self.ema))

        if self.n_stable < self.stable_count:
            return

        msg = PoseStamped()
        msg.header.stamp = color_msg.header.stamp
        msg.header.frame_id = self.target_frame
        msg.pose.position.x = float(self.ema[0])
        msg.pose.position.y = float(self.ema[1])
        msg.pose.position.z = float(self.ema[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.pub_pose.publish(msg)

        mk = Marker()
        mk.header = msg.header
        mk.ns, mk.id = 'pen', 0
        mk.type, mk.action = Marker.CUBE, Marker.ADD
        mk.pose = msg.pose
        mk.scale.x, mk.scale.y, mk.scale.z = float(long_m), float(short_m), 0.01
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1.0, 0.6, 0.0, 0.9
        self.pub_marker.publish(mk)

    def _to_target(self, p_cam, yaw_img, cam_frame):
        """Camera point + image-plane yaw -> target frame point and quaternion."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, cam_frame, rclpy.time.Time())
        except Exception as exc:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_tf_warn > 5.0:
                self._last_tf_warn = now
                self.get_logger().warn(
                    'TF {} -> {} unavailable: {}'.format(
                        cam_frame, self.target_frame, exc))
            return None, None

        t, q = tf.transform.translation, tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        p_out = R @ p_cam + np.array([t.x, t.y, t.z])

        # The pen's long axis, carried into the target frame and flattened:
        # what the gripper needs is the pen's heading on the work surface, not
        # its full 3-D direction.
        axis_out = R @ np.array([math.cos(yaw_img), math.sin(yaw_img), 0.0])
        yaw = math.atan2(axis_out[1], axis_out[0])
        cz, sz = math.cos(yaw), math.sin(yaw)
        M = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        return p_out, _quat_from_matrix(M)

    # ------------------------------------------------------------------- debug
    def _debug(self, color, mask, best, rejects, hit):
        if not self.want_debug or self.pub_debug.get_subscription_count() == 0:
            return
        dbg = color.copy()
        dbg[mask > 0] = (0.5 * dbg[mask > 0] +
                         0.5 * np.array([0, 255, 255])).astype(np.uint8)

        # The crop in orange, everything outside it dimmed - so one glance
        # says whether the fingers fell outside.
        x0, y0, x1, y1 = self._roi(dbg.shape)
        if (x0, y0, x1, y1) != (0, 0, dbg.shape[1], dbg.shape[0]):
            out = np.ones(dbg.shape[:2], bool)
            out[y0:y1, x0:x1] = False
            dbg[out] = (0.35 * dbg[out]).astype(np.uint8)
            cv2.rectangle(dbg, (x0, y0), (x1 - 1, y1 - 1), (0, 165, 255), 1)

        if self.use_plane:
            # Zero inliers means no surface was found, and then the cyan area
            # means nothing - worth seeing at a glance.
            cv2.putText(dbg, 'plane inliers {}'.format(self.plane_inliers),
                        (10, dbg.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 0), 1)

        # What YOLO-World thought was a pen, in magenta.  These are the boxes
        # that gated the mask; the green rectangle below is what was actually
        # fitted, and only that one carries an angle.
        for x1, y1, x2, y2, conf in self.yolo_boxes:
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.putText(dbg, 'yolo pen {:.2f}'.format(conf), (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Everything the shape test threw away, in red, labelled with the real
        # size it measured - that label is what you tune the ranges against.
        for long_m, short_m, aspect, z, rect in rejects[:12]:
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(dbg, [box], 0, (0, 0, 255), 1)
            cx, cy = rect[0]
            cv2.putText(dbg, '{:.0f}x{:.0f}'.format(long_m * 1000, short_m * 1000),
                        (int(cx) - 30, int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1)

        if best is not None:
            long_m, short_m, aspect, z, rect = best
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(dbg, [box], 0, (0, 255, 0), 2)
            cv2.putText(dbg, 'PEN {:.0f}x{:.0f}mm a={:.1f} z={:.3f}m'.format(
                long_m * 1000, short_m * 1000, aspect, z),
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if hit is not None:
            u, v, z, p = hit
            cv2.circle(dbg, (u, v), 6, (0, 0, 255), -1)
            cv2.putText(dbg, '[{:+.3f} {:+.3f} {:+.3f}]  stable {}/{}'.format(
                p[0], p[1], p[2], self.n_stable, self.stable_count),
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, 'bgr8'))


def main():
    rclpy.init()
    node = PenDetector()
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
