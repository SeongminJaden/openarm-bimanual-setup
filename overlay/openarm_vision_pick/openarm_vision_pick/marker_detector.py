# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Find a fiducial marker and publish where to grasp the object it is on.

Unlike the blob detectors in this package, a square fiducial gives a full
6-DOF pose from a single view: the four corners of a square of known size are
enough to solve for the plane it lies in.  That matters for picking, because
a grasp needs the object's YAW - which way to turn the jaws - and a centroid
cannot supply it.

  corners -> solvePnP(IPPE_SQUARE) -> marker pose in the camera frame
          -> grasp_offset applied IN THE MARKER FRAME
          -> TF -> target_frame

The offset is applied in the marker's own frame on purpose.  A marker stuck
on the top of a box is not where you grasp; the grasp point is some distance
below it, along the marker's -Z.  Expressing that in the marker frame means
it stays correct when the object is rotated or moved, which a world-frame
offset would not.

ONE NUMBER MATTERS MORE THAN ANY OTHER: marker_size_m, the side of the black
square in metres.  Pose distance scales linearly with it, so a marker printed
5% small reports the object 5% nearer than it is.  Measure the print; do not
trust the printer.  make_markers.py puts a 100 mm ruler on every sheet for
exactly this.

MEASURING THE OBJECT

A tag gives the object's pose but not its shape, and a grasp needs both: how
wide to open, which way to turn the jaws, and how far down to go.  So the
depth image is read in a box anchored ON the marker - the tag says where to
look, which is the part that made blob segmentation so fragile before.
Inside that box the points are expressed in the marker's own frame, the
footprint is fitted with a minimum-area rectangle, and the result is the
object's length, width and height in millimetres.

That measurement then decides the grasp:

  * the jaws close across the SHORT side, because that is the one that fits;
  * the yaw comes from the measured long axis, not from the tag's own
    rotation - a tag stuck on skew still yields the right grasp;
  * the grasp height is the top surface less half the object's height, the
    same sign error that cost a whole session with the pen detector;
  * an object wider than the gripper can open is REFUSED rather than
    attempted.

Set measure_object false to fall back to the tag pose plus a fixed offset.

Runs on either camera - every topic and frame is a parameter - so the same
node serves the chest camera looking across the bench and the wrist camera
closing in.

Publishes the same PoseStamped interface as the other detectors here, so
pick_and_place can consume it with nothing but a topic name.
"""

import math
import os

# Pin the numeric libraries to one thread each, before numpy and OpenCV load.
# Same reasoning as the other detectors: the arrays are small, and splitting
# that work across every core costs more in dispatch than it saves.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402
import rclpy                                                  # noqa: E402
import tf2_ros                                                # noqa: E402
from cv_bridge import CvBridge                                # noqa: E402
from geometry_msgs.msg import (Pose, PoseArray,              # noqa: E402
                               PoseStamped, Vector3)
from rcl_interfaces.msg import (ParameterDescriptor,          # noqa: E402
                                ParameterType,
                                SetParametersResult)
from rclpy.executors import ExternalShutdownException         # noqa: E402
from rclpy.node import Node                                   # noqa: E402
from rclpy.qos import qos_profile_sensor_data                 # noqa: E402
from sensor_msgs.msg import CameraInfo, Image                 # noqa: E402
from std_msgs.msg import Float32, Int32MultiArray             # noqa: E402
from visualization_msgs.msg import Marker                     # noqa: E402


# Sentinel for "accept any marker id".  See the accept_ids parameter.
ANY_ID = -1


def _wants_any(ids):
    return (not ids) or (ANY_ID in ids)


def quat_from_matrix(m):
    """Rotation matrix -> (x, y, z, w).  Shepperd's method."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
                (m[2, 1] - m[1, 2]) / s)
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
                (m[0, 2] - m[2, 0]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
            (m[1, 0] - m[0, 1]) / s)


def rot_from_quat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class MarkerDetector(Node):

    def __init__(self):
        super().__init__('marker_detector')

        p = self.declare_parameter
        p('color_topic', '/chest/openarm_chest_camera/color/image_raw')
        p('info_topic', '/chest/openarm_chest_camera/color/camera_info')
        p('depth_topic',
          '/chest/openarm_chest_camera/aligned_depth_to_color/image_raw')
        p('camera_frame', 'openarm_chest_camera_color_optical_frame')
        p('target_frame', 'world')

        # 'aruco', 'qr', or 'both'.  ArUco is the better tag: it survives
        # being small, tilted and half-lit, and its id tells one object from
        # another.  QR is here because objects sometimes already carry one.
        p('marker_family', 'aruco')
        p('aruco_dict', 'DICT_4X4_50')

        # THE side of the black square, in metres.  Everything scales with
        # it: PnP range is proportional to this number, so a size that is
        # wrong by 5% puts the object 5% nearer or further than it is.
        #
        # This is the fallback for ids not listed in size_ids below.
        p('marker_size_m', 0.060)

        # Different tags may be different sizes - a small one on the object
        # and a big one on the destination, say - and using one size for both
        # would misplace whichever it did not describe.  These two arrays are
        # a map: size_ids[k] is that many metres across.
        p('size_ids', [ANY_ID],
          ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY,
                              description='ids that have their own size'))
        p('size_values', [0.0])
        # QR codes vary; if the family is 'qr', this is the side of its black
        # module area.  Zero means "use marker_size_m".
        p('qr_size_m', 0.0)

        # Which ids to accept.  [-1] means any - a sentinel rather than an
        # empty list, because rclpy infers a parameter's type from its default
        # VALUE and an empty list has no type to infer: `p('accept_ids', [])`
        # declares it uninitialized and the first read throws
        # ParameterUninitializedException.  A ParameterDescriptor does not
        # rescue it either; the descriptor describes the type, it does not
        # supply one.  Marker ids are never negative, so -1 is unambiguous.
        p('accept_ids', [ANY_ID],
          ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY,
                              description='marker ids to accept; [-1] = any'))

        # Where the grasp point is relative to the marker, IN THE MARKER'S OWN
        # FRAME (x right, y up, z out of the surface).  A tag on the lid of a
        # 40 mm tall box wants roughly [0, 0, -0.02] to aim at its middle.
        p('grasp_offset_xyz', [0.0, 0.0, 0.0])

        # 'nearest' or 'lowest_id'.  Nearest is the sane default when the arm
        # should go for what it can actually reach.
        p('select', 'nearest')

        # Which tag the GRASP is computed for.  -1 keeps the rule above.
        #
        # With two tags on the bench - one on the object, one on where it goes
        # - 'nearest' is a coin toss, and measuring the destination as if it
        # were the thing to pick up is worse than useless.  Naming the id
        # settles it.  Every tag still appears on ~/poses either way.
        p('select_id', ANY_ID)

        # How many pixels the tag's side must span before it is used.
        #
        # This was 45 when the measurement was expressed in the tag's frame
        # and inherited every degree of error in its plane.  The measurement
        # no longer leans on the tag's plane at all: the support surface is
        # fitted from the depth data and the object is the connected blob at
        # the tag's POSITION, which is far better determined than its
        # orientation.  So the gate only has to keep out tags too small for
        # their centre to be trusted, and 25 px is ample for that - a tag
        # that small still locates its centre to a millimetre or two.
        #
        # Detection itself gives out at about 20 px.
        p('min_marker_px', 25.0)

        # Planar tags have TWO poses that project almost identically - the
        # classic square-marker ambiguity - and solvers flip between them
        # frame to frame when the tag is small or seen face on.  A pose whose
        # runner-up fits nearly as well is not a pose to measure against.
        # This is the ratio of the second solution's reprojection error to
        # the first's; below it, the pose is called ambiguous.
        p('ambiguity_ratio', 2.5)

        # Sanity band on the reported range.  A tag resolved at 5 m from a
        # wrist camera is a bad corner fit, not an object.
        p('min_range_m', 0.05)
        p('max_range_m', 3.00)

        # Take the RANGE from the depth camera and keep only the DIRECTION
        # and ORIENTATION from the tag.
        #
        # This is the answer to "what if the printed marker is not exactly the
        # size we told it".  A wrong marker_size_m is not noise to be
        # tolerated - it is a scale factor, and it biases every range by the
        # same proportion in the same direction.  Widening a tolerance does
        # not correct a bias; it only makes the wrong answer harder to
        # notice.  The depth camera measures that range directly and knows
        # nothing about the tag's size, so using it removes the error rather
        # than accommodating it.
        #
        # The tag is still what supplies the orientation, which depth cannot.
        p('range_from_depth', True)

        # Watch the PnP range against the depth range and report what marker
        # size would reconcile them.  Free diagnosis of a mis-printed tag.
        p('calibrate_size', True)
        p('calibrate_frames', 30)

        # Cross-check the PnP range against the depth camera.  They should
        # agree; a persistent gap means marker_size_m is wrong, and that is
        # worth being told rather than silently grasping short.
        p('depth_cross_check', True)
        p('depth_scale', 0.001)
        p('depth_warn_m', 0.03)

        # --- measuring the object around the marker -----------------------
        p('measure_object', True)
        # How far from the marker centre to look, in the marker's plane.
        # Big enough to contain the object, small enough to exclude its
        # neighbours.
        p('search_radius_m', 0.15)
        # How far BEHIND the marker face the object may extend.  With the tag
        # on the lid of a box this is the box's height.
        p('object_max_height_m', 0.20)
        # Points within this of the marker's plane are the top surface (and
        # the tag itself); anything further back is the body of the object.
        p('surface_margin_m', 0.006)
        p('min_object_points', 150)
        # The tag is on the object's TOP face and the object hangs below it
        # along the marker's -Z.  False for a tag on a vertical face.
        p('marker_on_top', True)

        # What the gripper can actually span.  An object wider than this
        # cannot be grasped, and finding that out before moving is cheaper
        # than finding out afterwards.
        p('gripper_max_m', 0.040)
        # Leave this much of the opening spare when reporting a grasp width.
        p('grasp_clearance_m', 0.006)

        # 'object_axis' publishes a yaw about world Z taken from the measured
        # long axis - which is what pick_and_place expects, since it adds 90
        # degrees itself to close across the object.  'marker' publishes the
        # tag's full orientation instead, for anything that wants 6-DOF.
        p('orientation_mode', 'object_axis')

        p('ema_alpha', 0.5)
        p('stable_count', 3)
        p('stable_tol_m', 0.010)
        p('publish_debug_image', True)

        g = self.get_parameter
        self.cam_frame = g('camera_frame').value
        self.target_frame = g('target_frame').value
        self.family = str(g('marker_family').value)
        self.size = g('marker_size_m').value
        self.qr_size = g('qr_size_m').value or self.size
        self.accept = self._id_filter(g('accept_ids').value)
        self.sizes = self._size_map(g('size_ids').value,
                                    g('size_values').value)
        self.offset = np.array(list(g('grasp_offset_xyz').value), dtype=float)
        self.select = str(g('select').value)
        self.select_id = int(g('select_id').value)
        self.min_px_side = g('min_marker_px').value
        self.ambig_ratio = g('ambiguity_ratio').value
        self.min_r = g('min_range_m').value
        self.max_r = g('max_range_m').value
        self.use_depth_range = g('range_from_depth').value
        self.calibrate = g('calibrate_size').value
        self.cal_n = int(g('calibrate_frames').value)
        self._cal = {}
        self.depth_check = g('depth_cross_check').value
        self.depth_scale = g('depth_scale').value
        self.depth_warn = g('depth_warn_m').value
        self.measure = g('measure_object').value
        self.search_r = g('search_radius_m').value
        self.max_h = g('object_max_height_m').value
        self.surf_margin = g('surface_margin_m').value
        self.min_pts = int(g('min_object_points').value)
        self.on_top = g('marker_on_top').value
        self.grip_max = g('gripper_max_m').value
        self.grip_clear = g('grasp_clearance_m').value
        self.ori_mode = str(g('orientation_mode').value)
        self.alpha = g('ema_alpha').value
        self.stable_count = int(g('stable_count').value)
        self.stable_tol = g('stable_tol_m').value
        self.want_debug = g('publish_debug_image').value

        cv2.setNumThreads(4)
        self._warned = {}
        self.bridge = CvBridge()
        self.K = None
        self.D = None
        self.depth = None
        self.ema = None
        self.n_stable = 0
        self._last_tf_warn = 0.0
        self._warned_size = False

        dict_name = str(g('aruco_dict').value)
        self.detector = None
        if self.family in ('aruco', 'both'):
            if not hasattr(cv2.aruco, dict_name):
                self.get_logger().error(
                    "unknown aruco_dict '{}'".format(dict_name))
            else:
                d = cv2.aruco.getPredefinedDictionary(
                    getattr(cv2.aruco, dict_name))
                params = cv2.aruco.DetectorParameters()
                # Sub-pixel corner refinement.  The pose is built entirely
                # from corner positions, so this is not a nicety: without it
                # the orientation jitters by degrees at any distance.
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                self.detector = cv2.aruco.ArucoDetector(d, params)
                self.get_logger().info(
                    'ArUco {}, marker {:.0f} mm'.format(dict_name,
                                                        self.size * 1000))
        self.qr = cv2.QRCodeDetector() if self.family in ('qr', 'both') \
            else None

        self.add_on_set_parameters_callback(self._on_param_set)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_pose = self.create_publisher(PoseStamped, '~/pose', 10)
        self.pub_debug = self.create_publisher(Image, '~/debug_image', 2)
        self.pub_marker = self.create_publisher(Marker, '~/marker', 2)
        self.pub_ids = self.create_publisher(Int32MultiArray, '~/ids', 2)
        # Every marker in view, in the same order as ~/ids and with the same
        # stamp.  A task that needs two tags at once - one on the object, one
        # on where it goes - cannot work from a single 'best' pose.
        self.pub_poses = self.create_publisher(PoseArray, '~/poses', 2)
        # How wide to open for this object.  A picker that ignores it can
        # still work from the pose alone; one that reads it does not have to
        # guess.
        self.pub_width = self.create_publisher(Float32, '~/grasp_width', 2)
        self.pub_box = self.create_publisher(Marker, '~/object_box', 2)
        # long, short, height in metres.  Placing needs the height: the grasp
        # is at the object's middle, so setting it down means putting that
        # point half a height above the destination surface.
        self.pub_dims = self.create_publisher(Vector3, '~/object_dims', 2)

        self.create_subscription(CameraInfo, g('info_topic').value,
                                 self._on_info, 10)
        self.create_subscription(Image, g('color_topic').value,
                                 self._on_color, qos_profile_sensor_data)
        if self.depth_check:
            self.create_subscription(Image, g('depth_topic').value,
                                     self._on_depth, qos_profile_sensor_data)

        self.get_logger().info(
            'colour=' + g('color_topic').value +
            '\n  family: ' + self.family +
            '\n  marker {:.1f} mm, accepting ids {}'.format(
                self.size * 1000,
                sorted(self.accept) if self.accept else 'any') +
            '\n  grasp offset in the marker frame [{:+.3f} {:+.3f} {:+.3f}]'
            .format(*self.offset) +
            '\n  images in ' + self.cam_frame +
            "\n  publishing poses in '" + self.target_frame + "'")

    def _size_map(self, ids, values):
        """id -> side length.  Unlisted ids fall back to marker_size_m."""
        ids = [int(v) for v in ids]
        values = [float(v) for v in values]
        if len(ids) != len(values):
            self.get_logger().error(
                'size_ids has {} entries and size_values {} - they must '
                'match; ignoring both'.format(len(ids), len(values)))
            return {}
        out = {i: v for i, v in zip(ids, values) if i != ANY_ID and v > 0.0}
        if out:
            self.get_logger().info(
                'per-id marker sizes: ' + ', '.join(
                    'id {} = {:.0f} mm'.format(i, v * 1000)
                    for i, v in sorted(out.items())))
        return out

    def _size_for(self, mid):
        return self.sizes.get(int(mid), self.size)

    @staticmethod
    def _id_filter(value):
        """An empty set means accept everything."""
        ids = set(int(v) for v in value)
        return set() if _wants_any(ids) else ids

    # --------------------------------------------------------- live retuning
    _LIVE = {
        'marker_size_m': 'size', 'qr_size_m': 'qr_size',
        'min_range_m': 'min_r', 'max_range_m': 'max_r',
        'depth_warn_m': 'depth_warn', 'ema_alpha': 'alpha',
        'stable_count': 'stable_count', 'stable_tol_m': 'stable_tol',
        'publish_debug_image': 'want_debug', 'select': 'select',
        'marker_size_m': 'size', 'select_id': 'select_id',
        'min_marker_px': 'min_px_side', 'ambiguity_ratio': 'ambig_ratio',
        'range_from_depth': 'use_depth_range', 'calibrate_size': 'calibrate',
        'measure_object': 'measure', 'search_radius_m': 'search_r',
        'object_max_height_m': 'max_h', 'surface_margin_m': 'surf_margin',
        'min_object_points': 'min_pts', 'marker_on_top': 'on_top',
        'gripper_max_m': 'grip_max', 'grasp_clearance_m': 'grip_clear',
        'orientation_mode': 'ori_mode',
    }

    def _on_param_set(self, params):
        for prm in params:
            if prm.name == 'accept_ids':
                self.accept = self._id_filter(prm.value)
                self.get_logger().info('accept_ids = {}'.format(
                    sorted(self.accept) if self.accept else 'any'))
                continue
            if prm.name == 'grasp_offset_xyz':
                v = np.array(list(prm.value), dtype=float)
                if v.shape != (3,):
                    return SetParametersResult(
                        successful=False, reason='needs three numbers')
                self.offset = v
                self.get_logger().info('grasp_offset_xyz = {}'.format(list(v)))
                continue
            attr = self._LIVE.get(prm.name)
            if attr is None:
                continue
            value = int(prm.value) if attr == 'stable_count' else prm.value
            setattr(self, attr, value)
            self.get_logger().info('{} = {}'.format(prm.name, value))
        self.ema = None
        self.n_stable = 0
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ input
    def _on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
            self.get_logger().info(
                'intrinsics fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}, '
                '{} distortion coefficients'.format(
                    self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2],
                    self.D.size))

    def _on_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg,
                                               desired_encoding='passthrough')

    # ----------------------------------------------------------------- detect
    def _object_points(self, side):
        """The marker's own corners, in its own frame, in ArUco's order.

        ArUco returns corners clockwise from the top-left as seen on the tag,
        so the model points have to follow that order or the pose comes out
        rotated by a multiple of 90 degrees - which looks almost right and
        turns the gripper the wrong way.
        """
        h = side / 2.0
        return np.array([[-h, h, 0.0], [h, h, 0.0],
                         [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)

    def _warn_once(self, key, text, period=5.0, level='warn'):
        """Say it at most once every `period` seconds, not once a frame."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._warned.get(key, 0.0) < period:
            return
        self._warned[key] = now
        # Separate call sites on purpose: rclpy caches a logger's severity
        # against the source location of the call, so severity is fixed per
        # call site and choosing between info and warn at one of them raises
        # "Logger severity cannot be changed between calls."
        if level == 'info':
            self.get_logger().info(text)
        else:
            self.get_logger().warn(text)

    @staticmethod
    def _apparent_px(corners):
        """Mean side length of the tag in the image, in pixels."""
        d = [np.linalg.norm(corners[(k + 1) % 4] - corners[k])
             for k in range(4)]
        return float(np.mean(d))

    def _pose_from_corners(self, corners, side):
        """Pose, plus how trustworthy its PLANE is.

        Returns (R, t, quality) where quality is a dict carrying the tag's
        apparent size and the ambiguity ratio.  Both are needed downstream:
        the position is usable long before the orientation is.
        """
        obj = self._object_points(side)
        img = corners.astype(np.float64)
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj, img, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except Exception:
            ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.D,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                return None, None, None
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec.reshape(3), {
                'px': self._apparent_px(corners), 'ratio': float('inf')}
        if n < 1:
            return None, None, None

        # IPPE_SQUARE returns the two planar solutions, best first.  How much
        # worse the runner-up is IS the confidence in the plane.
        e = [float(x) for x in np.array(errs).ravel()] if errs is not None \
            else [1.0]
        ratio = (e[1] / e[0]) if len(e) > 1 and e[0] > 1e-9 else float('inf')

        R, _ = cv2.Rodrigues(rvecs[0])
        return R, np.array(tvecs[0]).reshape(3), {
            'px': self._apparent_px(corners), 'ratio': ratio}

    def _detections(self, color):
        """Every marker in the frame, as (id, corners, side, label)."""
        out = []
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(color)
            if ids is not None:
                for c, i in zip(corners, ids.flatten()):
                    if self.accept and int(i) not in self.accept:
                        continue
                    out.append((int(i), c.reshape(4, 2),
                                self._size_for(int(i)),
                                'aruco {}'.format(int(i))))
        if self.qr is not None:
            try:
                ok, infos, pts, _ = self.qr.detectAndDecodeMulti(color)
            except Exception:
                ok, infos, pts = False, None, None
            if ok and pts is not None:
                for k, quad in enumerate(pts):
                    txt = infos[k] if infos is not None and k < len(infos) \
                        else ''
                    out.append((-1 - k, quad.reshape(4, 2), self.qr_size,
                                'qr {}'.format(txt[:16] or '?')))
        return out

    def _depth_at(self, uv):
        if self.depth is None:
            return None
        u, v = int(round(uv[0])), int(round(uv[1]))
        h, w = self.depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None
        patch = self.depth[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
        vals = patch.astype(np.float64).ravel()
        if self.depth.dtype == np.uint16:
            vals = vals * self.depth_scale
        good = vals[np.isfinite(vals) & (vals > 0.02)]
        return float(np.median(good)) if good.size >= 5 else None

    def _measure(self, R, t):
        """Length, width, height and long-axis direction of the object.

        Anchored on the tag's POSITION and on the support surface fitted from
        the depth data - deliberately not on the tag's orientation.  See
        _fit_support for why.

        Returns a dict in CAMERA-frame terms: the grasp centre, the object's
        extents, and a unit vector along its long axis.
        """
        if self.depth is None or not self.measure:
            return None
        h_img, w_img = self.depth.shape[:2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        if t[2] <= 1e-6:
            return None

        # Only deproject the part of the image the object can be in.
        half = int(self.search_r * fx / t[2]) + 10
        u_c = int(t[0] * fx / t[2] + cx)
        v_c = int(t[1] * fy / t[2] + cy)
        u0, u1 = max(0, u_c - half), min(w_img, u_c + half)
        v0, v1 = max(0, v_c - half), min(h_img, v_c + half)
        if u1 - u0 < 8 or v1 - v0 < 8:
            return None

        d_m = self.depth[v0:v1, u0:u1].astype(np.float32)
        if self.depth.dtype == np.uint16:
            d_m = d_m * self.depth_scale
        vs, us = np.nonzero(np.isfinite(d_m) & (d_m > 0.02))
        if vs.size < self.min_pts:
            return None
        zz = d_m[vs, us]
        pts = np.stack([((us + u0) - cx) * zz / fx,
                        ((vs + v0) - cy) * zz / fy, zz], 1)

        # Everything within the search radius of the tag, in space.
        near = pts[np.linalg.norm(pts - t, axis=1) <
                   self.search_r * 1.6]
        if near.shape[0] < self.min_pts:
            return None

        n, d0 = self._fit_support(near)
        if n is None:
            return None

        # Everything standing proud of the bench, as a MASK in the image, so
        # that connectivity is available.
        pts_all = np.stack([((us + u0) - cx) * zz / fx,
                            ((vs + v0) - cy) * zz / fy, zz], 1)
        above = (pts_all @ n + d0)
        proud = (above > self.surf_margin) & (above < self.max_h)

        mask = np.zeros(d_m.shape, np.uint8)
        mask[vs[proud], us[proud]] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))

        # Keep only the piece the TAG IS ON.
        #
        # Without this the measurement is "everything standing on the bench
        # within the search radius", which is a different thing entirely: a
        # neighbouring object, the sheet the tag is printed on, or the
        # robot's own finger all join in, and the answer comes out as
        # 250 x 105 mm for something far smaller, varying frame to frame as
        # the extra pieces come and go.  The tag says which lump is the
        # object; connectivity is how that gets used.
        ncomp, labels = cv2.connectedComponents(mask, 8)
        tu = int(round(t[0] * fx / t[2] + cx)) - u0
        tv = int(round(t[1] * fy / t[2] + cy)) - v0
        lab = 0
        if 0 <= tu < labels.shape[1] and 0 <= tv < labels.shape[0]:
            lab = int(labels[tv, tu])
        if lab == 0:
            # The tag's own centre did not land on a proud pixel - it can sit
            # in a depth hole.  Take the nearest labelled pixel instead.
            ys_l, xs_l = np.nonzero(labels > 0)
            if ys_l.size == 0:
                return None
            k = int(np.argmin((xs_l - tu) ** 2 + (ys_l - tv) ** 2))
            if (xs_l[k] - tu) ** 2 + (ys_l[k] - tv) ** 2 > (0.25 * half) ** 2:
                return None
            lab = int(labels[ys_l[k], xs_l[k]])

        sel = labels[vs, us] == lab
        obj = pts_all[sel & proud]
        if obj.shape[0] < self.min_pts:
            return None

        # In-plane axes, so the footprint can be fitted in two dimensions.
        e1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n, np.array([0.0, 1.0, 0.0]))
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)

        origin = obj.mean(axis=0)
        flat = np.stack([(obj - origin) @ e1, (obj - origin) @ e2],
                        1).astype(np.float32)
        (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(flat)
        long_m, short_m = max(rw, rh), min(rw, rh)
        if long_m < 1e-4:
            return None
        ang_long = math.radians(ang if rw >= rh else ang + 90.0)

        top = float(np.percentile(obj @ n + d0, 97.0))
        height = max(0.0, top)

        # The grasp: centre of the footprint, half the object's height above
        # the bench.  The depth camera only ever sees the TOP, and closing
        # the jaws there grips nothing.
        centre_plane = origin + rcx * e1 + rcy * e2
        on_plane = centre_plane - n * float(centre_plane @ n + d0)
        centre = on_plane + n * (height / 2.0)

        axis = math.cos(ang_long) * e1 + math.sin(ang_long) * e2

        return dict(centre=centre, axis=axis, normal=n,
                    long=float(long_m), short=float(short_m),
                    height=float(height), n=int(obj.shape[0]))

    def _calibrate(self, mid, side, t, corners):
        """Collect what size would reconcile this tag with the depth camera.

        Done for EVERY tag in view, not just the one being grasped: a
        destination tag that is never measured is still a tag whose printed
        size decides where the object is put down.
        """
        if t[2] <= 1e-6:
            return
        d = self._depth_at(corners.mean(axis=0))
        if d is None:
            return
        hist = self._cal.setdefault(mid, [])
        hist.append(side * d / t[2])
        if len(hist) < self.cal_n:
            return
        implied = float(np.median(hist))
        self._cal[mid] = []
        err = (implied - side) / side * 100.0
        if abs(err) > 2.0:
            self._warn_once(
                'cal{}'.format(mid),
                'tag {}: configured {:.1f} mm, but the depth camera implies '
                '{:.1f} mm ({:+.1f}%). Measure the printed square; set '
                'size_values for this id to {:.4f}.'.format(
                    mid, side * 1000, implied * 1000, err, implied))
        else:
            # Say it once.  A confirmation repeated every couple of seconds
            # is noise, and noise is what hides the next real message.
            self._warn_once(
                'calok{}'.format(mid),
                'tag {}: configured {:.1f} mm agrees with depth to {:+.1f}% '
                '(implied {:.1f} mm)'.format(mid, side * 1000, err,
                                             implied * 1000),
                period=300.0, level='info')

    def _fit_support(self, pts):
        """RANSAC the dominant plane in a patch of points: the bench.

        The measurement used to express everything in the MARKER's frame and
        assume the support surface was parallel to it.  It is not, quite -
        and it does not need to be wrong by much.  A few degrees of tilt error
        in the tag's pose puts a point 150 mm away out by 150*sin(theta):
        26 mm at ten degrees, which is more than the objects being measured
        are tall.  The bench then reads as part of the object and the
        footprint comes out as garbage - 163 x 153 mm for something that is
        neither.

        Fitting the surface from the depth data instead makes the measurement
        depend on the tag's POSITION, which is well determined, and not on
        its ORIENTATION, which at 50-odd pixels is not.
        """
        if pts.shape[0] < self.min_pts:
            return None, None
        sub = pts[::max(1, pts.shape[0] // 4000)]
        rng = np.random.default_rng(0)
        best_n, best_d, best_cnt = None, 0.0, 0
        for _ in range(120):
            idx = rng.choice(sub.shape[0], 3, replace=False)
            a, b, c = sub[idx]
            nv = np.cross(b - a, c - a)
            nn = np.linalg.norm(nv)
            if nn < 1e-9:
                continue
            nv = nv / nn
            dd = -float(nv @ a)
            cnt = int(np.count_nonzero(np.abs(sub @ nv + dd) < 0.006))
            if cnt > best_cnt:
                best_n, best_d, best_cnt = nv, dd, cnt
        if best_n is None or best_cnt < 0.2 * sub.shape[0]:
            return None, None
        inl = sub[np.abs(sub @ best_n + best_d) < 0.006]
        centre = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - centre, full_matrices=False)
        n = vt[2]
        if n[2] > 0:            # point it back towards the camera
            n = -n
        return n, -float(n @ centre)

    def _to_target(self, p_cam, R_cam):
        if self.target_frame == self.cam_frame:
            return p_cam, R_cam
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
            return None, None
        R = rot_from_quat(tf.transform.rotation)
        t = tf.transform.translation
        return R @ p_cam + np.array([t.x, t.y, t.z]), R @ R_cam

    def _on_color(self, msg):
        if self.K is None:
            return
        color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        found = []
        for mid, corners, side, label in self._detections(color):
            R, t, qual = self._pose_from_corners(corners, side)
            if R is None:
                continue
            rng = float(np.linalg.norm(t))
            if not (self.min_r <= rng <= self.max_r):
                continue
            found.append((mid, corners, side, label, R, t, rng, qual))

        if not found:
            self.n_stable = 0
            self._debug(color, [], None)
            return

        if self.select_id != ANY_ID:
            named = [e for e in found if e[0] == self.select_id]
            if not named:
                # The named tag is not in view.  Everything else still goes
                # out on ~/poses above; there is simply no grasp to publish.
                self.n_stable = 0
                self._debug(color, found, None)
                self._publish_all(msg, found)
                return
            found = named + [e for e in found if e[0] != self.select_id]
        elif self.select == 'lowest_id':
            found.sort(key=lambda e: (e[0] < 0, e[0]))
        else:
            found.sort(key=lambda e: e[6])
        mid, corners, side, label, R, t, rng, qual = found[0]

        # Does the depth camera agree about how far away it is?  If it does
        # not, marker_size_m is the first thing to suspect - the PnP range is
        # proportional to it, and nothing else in this pipeline is.
        centre = corners.mean(axis=0)
        d_meas = self._depth_at(centre) \
            if (self.depth_check or self.use_depth_range
                or self.calibrate) else None

        # What size WOULD reconcile the tag with the depth camera?  Collected
        # per id, because the two tags here are different sizes and a single
        # running figure would average them into a number describing neither.
        if self.calibrate:
            for e in found:
                self._calibrate(e[0], e[2], e[5], e[1])
        if False and self.calibrate and d_meas is not None and t[2] > 1e-6:
            hist = self._cal.setdefault(mid, [])
            hist.append(side * d_meas / t[2])
            if len(hist) >= self.cal_n:
                implied = float(np.median(hist))
                self._cal[mid] = []
                err = (implied - side) / side * 100.0
                if abs(err) > 2.0:
                    self.get_logger().warn(
                        'tag {}: configured {:.1f} mm, but the depth camera '
                        'implies {:.1f} mm ({:+.1f}%). Measure the printed '
                        'square; set size_values for this id to {:.4f}.'
                        .format(mid, side * 1000, implied * 1000, err,
                                implied))
                else:
                    self.get_logger().info(
                        'tag {}: configured {:.1f} mm agrees with depth to '
                        '{:+.1f}% (implied {:.1f} mm)'.format(
                            mid, side * 1000, err, implied * 1000))

        # Rescale the translation so its range is the measured one.  The
        # direction the tag lies in is well determined even when its size is
        # not, so only the magnitude needs correcting.
        if self.use_depth_range and d_meas is not None and t[2] > 1e-6:
            t = t * (d_meas / t[2])
            rng = float(np.linalg.norm(t))
        if d_meas is not None and abs(d_meas - t[2]) > self.depth_warn \
                and not self._warned_size:
            self._warned_size = True
            implied = self.size * d_meas / t[2] if t[2] > 1e-6 else 0.0
            self.get_logger().warn(
                'marker solves at {:.3f} m but the depth camera says {:.3f} m'
                ' ({:+.0f} mm). If that gap is steady, marker_size_m is '
                'wrong: {:.3f} m would fit the depth. Measure the printed '
                'square.'.format(t[2], d_meas, (t[2] - d_meas) * 1000,
                                 implied))

        # Measure the object the tag is stuck to, and let that decide the
        # grasp where it can.
        # Only measure against a plane worth measuring against.
        meas = None
        if qual is not None and qual['px'] < self.min_px_side:
            self._warn_once('too_small',
                            'tag {} spans {:.0f} px; below min_marker_px '
                            '{:.0f} the plane is not reliable enough to '
                            'measure a shape against, so the object is not '
                            'being measured. Move the camera closer, print a '
                            'bigger tag, or measure from the wrist camera.'
                            .format(mid, qual['px'], self.min_px_side))
        elif qual is not None and qual['ratio'] < self.ambig_ratio:
            self._warn_once('ambiguous',
                            'tag {} pose is ambiguous (runner-up fits {:.1f}x '
                            'as well, needs {:.1f}x) - the two planar '
                            'solutions are too close to tell apart, so the '
                            'shape is not being measured'
                            .format(mid, qual['ratio'], self.ambig_ratio))
        else:
            meas = self._measure(R, t)
        grasp_width = None
        if meas is not None:
            width = meas['short']
            if width > self.grip_max:
                # Publish the markers anyway: refusing to grasp THIS object
                # is no reason to hide the destination tag from a task that
                # is waiting to see both.
                self._publish_all(msg, found)
                self._warn_once(
                    'too_wide',
                    'object measures {:.0f} x {:.0f} mm; its short side is '
                    'wider than the gripper can open ({:.0f} mm). The tag '
                    'spans {:.0f} px at {:.2f} m - if that measurement looks '
                    'nothing like the real object, the tag is too small in '
                    'frame for its plane to be trusted rather than the '
                    'object being too big.'
                    .format(meas['long'] * 1000, width * 1000,
                            self.grip_max * 1000,
                            qual['px'] if qual else 0.0, rng))
                self.n_stable = 0
                self._debug(color, found, None, meas)
                return
            grasp_width = min(self.grip_max,
                              width + 2.0 * self.grip_clear)
            # Centre of the measured footprint, half the object's height
            # below the marker face.  The sign matters: the depth camera sees
            # the TOP, and closing the jaws at the top of an object grips
            # nothing.
            # The measurement already reports the grasp centre in the
            # camera frame; the offset is still applied in the tag's frame,
            # so a hand-set correction keeps its meaning.
            p_cam = meas['centre'] + R @ self.offset
        else:
            # The grasp point lives in the marker's frame, so it rotates
            # with it.
            p_cam = t + R @ self.offset

        # Yaw from the measured long axis when there is one: a tag stuck on
        # skew should not turn the jaws with it.
        R_use = R
        if meas is not None and self.ori_mode == 'object_axis':
            # Build a frame whose X is the object's measured long axis and
            # whose Z is the surface normal: the grasp is defined against the
            # object and the bench, not against however the tag was stuck on.
            zc = meas['normal']
            xc = meas['axis'] - zc * float(meas['axis'] @ zc)
            nx = np.linalg.norm(xc)
            if nx > 1e-6:
                xc = xc / nx
                yc = np.cross(zc, xc)
                R_use = np.column_stack((xc, yc, zc))

        p_out, R_out = self._to_target(p_cam, R_use)
        if p_out is None:
            self.n_stable = 0
            self._debug(color, found, None, meas)
            return

        self.ema = p_out if self.ema is None else \
            self.alpha * p_out + (1.0 - self.alpha) * self.ema
        if np.linalg.norm(p_out - self.ema) < self.stable_tol:
            self.n_stable += 1
        else:
            self.n_stable = 0

        self._debug(color, found, (mid, label, rng, d_meas), meas)

        self._publish_all(msg, found)

        if self.n_stable < self.stable_count:
            return

        if self.ori_mode == 'object_axis':
            # pick_and_place reads a yaw about world Z and adds 90 degrees to
            # close across the object, so give it a pure Z rotation carrying
            # the long axis - the same convention pen_detector used.
            axis = R_out @ np.array([1.0, 0.0, 0.0])
            yaw = math.atan2(axis[1], axis[0])
            cz, sz = math.cos(yaw), math.sin(yaw)
            R_pub = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        else:
            R_pub = R_out
        q = quat_from_matrix(R_pub)
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.target_frame
        out.pose.position.x = float(self.ema[0])
        out.pose.position.y = float(self.ema[1])
        out.pose.position.z = float(self.ema[2])
        out.pose.orientation.x, out.pose.orientation.y, \
            out.pose.orientation.z, out.pose.orientation.w = \
            [float(v) for v in q]
        self.pub_pose.publish(out)

        mk = Marker()
        mk.header = out.header
        mk.ns, mk.id = 'marker', 0
        mk.type, mk.action = Marker.CUBE, Marker.ADD
        mk.pose = out.pose
        mk.scale.x = mk.scale.y = float(side)
        mk.scale.z = 0.004
        mk.color.r, mk.color.g, mk.color.a = 1.0, 0.6, 0.9
        self.pub_marker.publish(mk)

        if grasp_width is not None:
            self.pub_width.publish(Float32(data=float(grasp_width)))
            box = Marker()
            box.header = out.header
            box.ns, box.id = 'object', 1
            box.type, box.action = Marker.CUBE, Marker.ADD
            box.pose = out.pose
            box.scale.x = float(meas['long'])
            box.scale.y = float(meas['short'])
            box.scale.z = float(max(0.005, meas['height']))
            box.color.b, box.color.g, box.color.a = 1.0, 0.4, 0.5
            self.pub_box.publish(box)

            self.pub_dims.publish(Vector3(x=float(meas['long']),
                                          y=float(meas['short']),
                                          z=float(meas['height'])))

    def _publish_all(self, msg, found):
        """Every marker in view, ids and poses in the same order.

        Published before the stability gate on purpose: a consumer waiting
        for the destination tag should not be held up by the object tag
        settling, or the other way round.
        """
        ids = Int32MultiArray()
        ids.data = [int(e[0]) for e in found]

        arr = PoseArray()
        arr.header.stamp = msg.header.stamp
        arr.header.frame_id = self.target_frame
        keep = []
        for e in found:
            p_o, R_o = self._to_target(e[5], e[4])
            if p_o is None:
                continue
            pose = Pose()
            pose.position.x = float(p_o[0])
            pose.position.y = float(p_o[1])
            pose.position.z = float(p_o[2])
            qq = quat_from_matrix(R_o)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, \
                pose.orientation.w = [float(v) for v in qq]
            arr.poses.append(pose)
            keep.append(int(e[0]))
        ids.data = keep
        self.pub_ids.publish(ids)
        self.pub_poses.publish(arr)

    # ------------------------------------------------------------------ debug
    def _debug(self, color, found, hit, meas=None):
        if not self.want_debug or self.pub_debug.get_subscription_count() == 0:
            return
        dbg = color.copy()

        for mid, corners, side, label, R, t, rng, qual in found:
            pts = corners.astype(np.int32)
            cv2.polylines(dbg, [pts], True, (0, 255, 0), 2)
            # The marker's own axes: red X, green Y, blue Z out of the face.
            # If Z does not point away from the surface, the corner order is
            # wrong and so is every yaw downstream.
            try:
                axis = np.float32([[0, 0, 0], [side / 2, 0, 0],
                                   [0, side / 2, 0], [0, 0, side / 2]])
                rvec, _ = cv2.Rodrigues(R)
                proj, _ = cv2.projectPoints(axis, rvec,
                                            t.reshape(3, 1), self.K, self.D)
                o, x, y, z = [tuple(np.int32(q).ravel()) for q in proj]
                cv2.line(dbg, o, x, (0, 0, 255), 2)
                cv2.line(dbg, o, y, (0, 255, 0), 2)
                cv2.line(dbg, o, z, (255, 0, 0), 2)
            except Exception:
                pass
            c = tuple(np.int32(corners.mean(axis=0)))
            good = (qual is not None and qual['px'] >= self.min_px_side
                    and qual['ratio'] >= self.ambig_ratio)
            cv2.putText(dbg, '{}  {:.0f} mm  {:.0f}px{}'.format(
                label, rng * 1000, qual['px'] if qual else 0.0,
                '' if good else '  POSE WEAK'),
                (c[0] - 40, c[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255) if good else (0, 0, 255), 1)

        if hit is not None:
            mid, label, rng, d_meas = hit
            txt = 'grasping {}   pnp {:.0f} mm'.format(label, rng * 1000)
            if d_meas is not None:
                txt += '   depth {:.0f} mm'.format(d_meas * 1000)
            cv2.putText(dbg, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            if meas is not None:
                cv2.putText(dbg,
                            'object {:.0f} x {:.0f} x {:.0f} mm  '
                            '({} pts)  grasp across {:.0f} mm'.format(
                                meas['long'] * 1000, meas['short'] * 1000,
                                meas['height'] * 1000, meas['n'],
                                meas['short'] * 1000),
                            (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)
            if self.ema is not None:
                cv2.putText(dbg, '[{:+.3f} {:+.3f} {:+.3f}] {} stable {}/{}'
                            .format(self.ema[0], self.ema[1], self.ema[2],
                                    self.target_frame, self.n_stable,
                                    self.stable_count),
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
        else:
            cv2.putText(dbg, 'no marker', (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, 'bgr8'))


def main():
    rclpy.init()
    node = MarkerDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
