# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Colour + depth object detector for the OpenArm wrist camera.

Segments one coloured object in the RGB image, reads its range out of the
aligned depth image, back-projects it through the camera intrinsics and
republishes it as a pose in a fixed frame.

The topic names default to the ones realsense2_camera uses, and the Gazebo
model publishes the same names, so the identical node runs against simulation
and hardware.
"""

import math

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
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
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        p = self.declare_parameter
        cam = '/oa/openarm_left_camera'
        p('color_topic', f'{cam}/color/image_raw')
        p('depth_topic', f'{cam}/aligned_depth_to_color/image_raw')
        p('info_topic', f'{cam}/color/camera_info')
        p('target_frame', 'world')

        # HSV window, OpenCV convention: H 0-179, S/V 0-255.
        p('hsv_lower', [100, 120, 60])
        p('hsv_upper', [130, 255, 255])
        # Red straddles H=0, so it needs two windows.  When this is true the
        # mask is (H >= hue_lo) OR (H <= hue_hi) instead of a single band.
        p('hue_wrap', False)

        p('min_area_px', 300)
        p('max_area_px', 200000)
        # Metres per raw depth unit.  32FC1 images are already in metres and
        # this is ignored; 16UC1 images are raw sensor units - 0.001 for the
        # D435, but the D405 defaults to 0.0001, which is the single most
        # common reason a vision pick lands ten times too far away.
        p('depth_scale', 0.001)
        p('depth_patch', 7)
        p('min_depth_m', 0.05)
        p('max_depth_m', 2.5)

        p('ema_alpha', 0.4)
        p('stable_count', 5)
        p('stable_tol_m', 0.01)
        p('publish_debug_image', True)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.target_frame = g('target_frame')
        self.hsv_lo = np.array(g('hsv_lower'), dtype=np.uint8)
        self.hsv_hi = np.array(g('hsv_upper'), dtype=np.uint8)
        self.hue_wrap = g('hue_wrap')
        self.min_area = g('min_area_px')
        self.max_area = g('max_area_px')
        self.depth_scale = g('depth_scale')
        self.patch = int(g('depth_patch'))
        self.min_d = g('min_depth_m')
        self.max_d = g('max_depth_m')
        self.alpha = g('ema_alpha')
        self.stable_count = g('stable_count')
        self.stable_tol = g('stable_tol_m')
        self.want_debug = g('publish_debug_image')

        self.bridge = CvBridge()
        self.K = None
        self.filtered = None
        self.n_stable = 0
        self._warned_depth = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.pub_pose = self.create_publisher(PoseStamped, '~/pose', 10)
        self.pub_raw = self.create_publisher(PoseStamped, '~/pose_raw', 10)
        self.pub_marker = self.create_publisher(Marker, '~/marker', 10)
        self.pub_debug = self.create_publisher(Image, '~/debug_image', 2)

        self.create_subscription(CameraInfo, g('info_topic'), self._on_info,
                                 qos_profile_sensor_data)
        sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, g('color_topic'), qos_profile=qos_profile_sensor_data),
             Subscriber(self, Image, g('depth_topic'), qos_profile=qos_profile_sensor_data)],
            queue_size=10, slop=0.15)
        sync.registerCallback(self._on_frame)

        self.get_logger().info(
            f"colour={g('color_topic')}\n  depth={g('depth_topic')}\n"
            f"  HSV {list(self.hsv_lo)}..{list(self.hsv_hi)} wrap={self.hue_wrap}\n"
            f"  publishing poses in '{self.target_frame}'")

    # ------------------------------------------------------------------ input
    def _on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)
            self.get_logger().info(
                f'intrinsics fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} '
                f'cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}')

    def _mask(self, hsv):
        if not self.hue_wrap:
            return cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)
        lo, hi = self.hsv_lo.copy(), self.hsv_hi.copy()
        a = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]], np.uint8),
                        np.array([179, hi[1], hi[2]], np.uint8))
        b = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8),
                        np.array([hi[0], hi[1], hi[2]], np.uint8))
        return cv2.bitwise_or(a, b)

    def _depth_metres(self, depth_img, u, v):
        """Median depth in a small patch, ignoring the holes."""
        h, w = depth_img.shape[:2]
        r = self.patch
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        patch = depth_img[y0:y1, x0:x1].astype(np.float64)
        if depth_img.dtype == np.uint16:
            patch = patch * self.depth_scale
        good = patch[np.isfinite(patch) & (patch > self.min_d) & (patch < self.max_d)]
        return float(np.median(good)) if good.size >= 3 else None

    # ------------------------------------------------------------- processing
    def _on_frame(self, color_msg, depth_msg):
        if self.K is None:
            return

        color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        if depth.dtype == np.uint16 and not self._warned_depth:
            self._warned_depth = True
            self.get_logger().info(
                f'depth is 16UC1, applying depth_scale={self.depth_scale} m/unit')

        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = self._mask(hsv)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            if self.min_area <= cv2.contourArea(c) <= self.max_area:
                blob = c

        if blob is None:
            self.n_stable = 0
            self._publish_debug(color, mask, None, None)
            return

        m = cv2.moments(blob)
        u = int(round(m['m10'] / m['m00']))
        v = int(round(m['m01'] / m['m00']))
        z = self._depth_metres(depth, u, v)
        if z is None:
            self.n_stable = 0
            self._publish_debug(color, mask, blob, None)
            return

        # Camera optical frame: +X right, +Y down, +Z forward.
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        p_cam = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])

        # Long axis of the blob, so the jaws can be set across it rather than
        # along it.  Both endpoints are back-projected at the same range, which
        # is right for a roughly flat object seen from above.
        rect = cv2.minAreaRect(blob)
        (rw, rh), ang = rect[1], rect[2]
        if rw < rh:
            ang += 90.0
        th = math.radians(ang)
        half = max(rw, rh) / 2.0
        e1 = np.array([(u + half * math.cos(th) - cx) / fx * z,
                       (v + half * math.sin(th) - cy) / fy * z, z])
        e2 = np.array([(u - half * math.cos(th) - cx) / fx * z,
                       (v - half * math.sin(th) - cy) / fy * z, z])

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, color_msg.header.frame_id,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.2))
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f'TF {color_msg.header.frame_id} -> '
                                   f'{self.target_frame} unavailable: {exc}',
                                   throttle_duration_sec=5.0)
            return

        R, t = self._tf_to_rt(tf)
        p_world = R @ p_cam + t
        a_world = (R @ e1 + t) - (R @ e2 + t)
        yaw = math.atan2(a_world[1], a_world[0])
        # A two-jaw gripper is symmetric under a half turn.
        yaw = math.atan2(math.sin(2 * yaw), math.cos(2 * yaw)) / 2.0

        self._publish(p_world, yaw, color_msg.header.stamp)
        self._publish_debug(color, mask, blob, (u, v, z, p_world))

    @staticmethod
    def _tf_to_rt(tf: TransformStamped):
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        tr = tf.transform.translation
        return R, np.array([tr.x, tr.y, tr.z])

    # ----------------------------------------------------------------- output
    def _publish(self, p_world, yaw, stamp):
        raw = PoseStamped()
        raw.header.stamp = stamp
        raw.header.frame_id = self.target_frame
        raw.pose.position.x, raw.pose.position.y, raw.pose.position.z = p_world
        raw.pose.orientation.z = math.sin(yaw / 2.0)
        raw.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub_raw.publish(raw)

        if self.filtered is None:
            self.filtered = p_world.copy()
            self.n_stable = 1
        else:
            moved = float(np.linalg.norm(p_world - self.filtered))
            self.filtered = self.alpha * p_world + (1 - self.alpha) * self.filtered
            self.n_stable = self.n_stable + 1 if moved < self.stable_tol else 0

        if self.n_stable < self.stable_count:
            return

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_frame
        msg.pose.position.x = float(self.filtered[0])
        msg.pose.position.y = float(self.filtered[1])
        msg.pose.position.z = float(self.filtered[2])
        msg.pose.orientation = raw.pose.orientation
        self.pub_pose.publish(msg)

        tf = TransformStamped()
        tf.header = msg.header
        tf.child_frame_id = 'target_object'
        tf.transform.translation.x = msg.pose.position.x
        tf.transform.translation.y = msg.pose.position.y
        tf.transform.translation.z = msg.pose.position.z
        tf.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

        mk = Marker()
        mk.header = msg.header
        mk.ns, mk.id, mk.type, mk.action = 'target', 0, Marker.SPHERE, Marker.ADD
        mk.pose = msg.pose
        mk.scale.x = mk.scale.y = mk.scale.z = 0.04
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1.0, 0.6, 0.0, 0.9
        self.pub_marker.publish(mk)

    def _publish_debug(self, color, mask, blob, hit):
        if not self.want_debug or self.pub_debug.get_subscription_count() == 0:
            return
        dbg = color.copy()
        dbg[mask > 0] = (0.5 * dbg[mask > 0] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
        if blob is not None:
            cv2.drawContours(dbg, [blob], -1, (0, 255, 0), 2)
        if hit is not None:
            u, v, z, p = hit
            cv2.circle(dbg, (u, v), 6, (0, 0, 255), -1)
            txt = f'{z:.3f}m  [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]'
            cv2.putText(dbg, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(dbg, f'stable {self.n_stable}/{self.stable_count}', (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, 'bgr8'))


def main():
    rclpy.init()
    node = ObjectDetector()
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
