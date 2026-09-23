# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Show, on the camera image, whether each marker is somewhere the arm can
reach - live, so the object can be slid into place while watching.

  ros2 run openarm_vision_pick reach_overlay --ros-args \\
    --params-file <config>/marker.yaml
  ros2 run rqt_image_view rqt_image_view /reach_overlay/image

The reachable region is worked out ONCE at start-up, by sweeping /compute_ik
over a grid at the grasp and place heights, and cached.  A marker is then
looked up in that grid every frame - a few microseconds - instead of asking
IK each time, which would run at a fraction of the camera rate and lag the
picture behind the hand moving the object.

What it draws:

  * the reachable region itself, projected onto the bench as a tinted patch,
    so you can see where to aim before the marker gets there;
  * each marker in GREEN if the arm can grasp there at some jaw angle, RED if
    not, with the angles that work written beside it;
  * the object tag and the destination tag judged at their OWN heights - the
    grasp is half an object-height above the bench, the place is that plus
    the object's height again, and a spot that works for one can fail for
    the other.

A cell is judged reachable only if the pose AND the approach above it both
solve, because that is what the pick sequence actually needs.
"""

import math
import os
import time

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402
import rclpy                                                  # noqa: E402
import tf2_ros                                                # noqa: E402
from cv_bridge import CvBridge                                # noqa: E402
from geometry_msgs.msg import PoseArray, PoseStamped          # noqa: E402
from moveit_msgs.srv import GetPositionIK                     # noqa: E402
from rclpy.executors import ExternalShutdownException         # noqa: E402
from rclpy.node import Node                                   # noqa: E402
from rclpy.qos import qos_profile_sensor_data                 # noqa: E402
from sensor_msgs.msg import CameraInfo, Image                 # noqa: E402
from std_msgs.msg import Int32MultiArray                      # noqa: E402

from openarm_vision_pick.pick_and_place import (  # noqa: E402
    grasp_orientation_candidates)


class ReachOverlay(Node):

    def __init__(self):
        super().__init__('reach_overlay')

        p = self.declare_parameter
        p('color_topic', '/chest/openarm_chest_camera/color/image_raw')
        p('info_topic', '/chest/openarm_chest_camera/color/camera_info')
        p('camera_frame', 'openarm_chest_camera_color_optical_frame')
        p('poses_topic', '/chest_marker_detector/poses')
        p('ids_topic', '/chest_marker_detector/ids')
        p('planning_frame', 'world')

        p('group', 'left_arm')
        p('tcp_link', 'openarm_left_hand_tcp')
        p('pick_id', 3)
        p('place_id', 0)

        # The grid.  Coarse enough to sweep in under a minute, fine enough
        # that a marker between two cells is judged sensibly.
        p('grid_x', [0.05, 0.45, 0.025])
        p('grid_y', [-0.10, 0.55, 0.025])
        p('yaws_deg', [0.0, 45.0, 90.0, 135.0])
        p('approach_height_m', 0.10)
        p('ik_timeout_s', 0.02)
        # Leans to try after straight down fails, in this order.  Every extra
        # lean multiplies the sweep time for cells that fail, so the grid
        # step is coarser than it was to compensate.
        p('tilt_deg', [0.0, 15.0, 30.0])

        # Heights.  The bench z has to be right or the whole map is judged
        # at the wrong height; measure it, or read it off a marker lying flat
        # on the bench (the detector publishes that pose in world).
        p('bench_z', 0.20)
        p('object_height_m', 0.040)
        # Override to judge at exactly these heights instead.  Zero = derive
        # from bench_z and object_height_m.
        p('grasp_z', 0.0)
        p('place_z', 0.0)

        g = self.get_parameter
        self.cam_frame = g('camera_frame').value
        self.frame = g('planning_frame').value
        self.group = g('group').value
        self.tcp = g('tcp_link').value
        self.pick_id = int(g('pick_id').value)
        self.place_id = int(g('place_id').value)
        gx, gy = g('grid_x').value, g('grid_y').value
        self.xs = np.arange(gx[0], gx[1] + 1e-9, gx[2])
        self.ys = np.arange(gy[0], gy[1] + 1e-9, gy[2])
        self.yaws = [math.radians(v) for v in g('yaws_deg').value]
        self.approach = g('approach_height_m').value
        self.ik_timeout = g('ik_timeout_s').value
        self.tilts = [math.radians(v) for v in g('tilt_deg').value if v > 0.0]
        bench = g('bench_z').value
        oh = g('object_height_m').value
        self.grasp_z = g('grasp_z').value or (bench + oh / 2.0)
        self.place_z = g('place_z').value or (bench + oh / 2.0 + 0.004)
        self.object_h = oh

        self.bridge = CvBridge()
        self.K = None
        self.ids = []
        self.poses = None
        self.maps = {}          # z -> bool grid [len(xs), len(ys), len(yaws)]
        self._sweeping = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ik = self.create_client(GetPositionIK, '/compute_ik')

        self.pub = self.create_publisher(Image, '~/image', 2)
        self.create_subscription(CameraInfo, g('info_topic').value,
                                 self._on_info, 10)
        self.create_subscription(Image, g('color_topic').value,
                                 self._on_color, qos_profile_sensor_data)
        self.create_subscription(Int32MultiArray, g('ids_topic').value,
                                 self._on_ids, 10)
        self.create_subscription(PoseArray, g('poses_topic').value,
                                 self._on_poses, 10)

        self.get_logger().info(
            'judging the object tag {} at z={:.3f} and the destination tag '
            '{} at z={:.3f}; grid {}x{} cells x {} angles'.format(
                self.pick_id, self.grasp_z, self.place_id, self.place_z,
                len(self.xs), len(self.ys), len(self.yaws)))

    # ------------------------------------------------------------------ input
    def _on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_ids(self, msg):
        self.ids = list(msg.data)

    def _on_poses(self, msg):
        self.poses = msg

    # ---------------------------------------------------------------- the map
    def _ik_ok(self, xyz, q, avoid=True):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group
        req.ik_request.ik_link_name = self.tcp
        req.ik_request.robot_state.is_diff = True
        req.ik_request.avoid_collisions = bool(avoid)
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = int(self.ik_timeout * 1e9)
        ps = PoseStamped()
        ps.header.frame_id = self.frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = \
            [float(v) for v in xyz]
        ps.pose.orientation.x, ps.pose.orientation.y, \
            ps.pose.orientation.z, ps.pose.orientation.w = \
            [float(v) for v in q]
        req.ik_request.pose_stamped = ps
        fut = self.ik.call_async(req)
        end = time.time() + 3.0
        while rclpy.ok() and not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.005)
        return fut.done() and fut.result().error_code.val == 1

    def sweep(self):
        """Build the reachability grids.  Blocks; run once before spinning."""
        if not self.ik.wait_for_service(timeout_sec=120.0):
            self.get_logger().error(
                '/compute_ik was not found in 120 s - is move_group up?')
            return False
        # The grid stores, per (cell, jaw angle), the index of the FIRST
        # orientation that solved: 0 = straight down, higher = more lean,
        # -1 = nothing.  The lean is kept so the overlay can say "reachable,
        # but leaned 30 deg" rather than just "reachable".
        zs = sorted({round(self.grasp_z, 4), round(self.place_z, 4)})
        done = {}
        for z in zs:
            # Two heights a few millimetres apart do not need two sweeps.
            near = [zz for zz in done if abs(zz - z) < 0.02]
            if near:
                self.maps[z] = done[near[0]]
                self.get_logger().info(
                    'z={:.3f}: reusing the sweep at z={:.3f}'.format(z, near[0]))
                continue
            grid = np.full((len(self.xs), len(self.ys), len(self.yaws)), -1,
                           np.int8)
            t0 = time.time()
            n_ok = 0
            for i, x in enumerate(self.xs):
                for j, y in enumerate(self.ys):
                    for k, yaw in enumerate(self.yaws):
                        # Reachable means the grasp AND the approach above
                        # it, at the same orientation, because that is what
                        # the sequence needs.  Stop at the first lean that
                        # works: the order is the preference.
                        for idx, (q, tdeg, lab) in enumerate(
                                grasp_orientation_candidates(yaw, self.tilts)):
                            # The cell itself is a contact pose - with the
                            # octomap on, the object's own cell would read
                            # red otherwise - so reach only; the pose above
                            # it is checked against the scene.
                            if self._ik_ok((x, y, z), q, avoid=False) and \
                                    self._ik_ok((x, y, z + self.approach), q):
                                grid[i, j, k] = idx
                                n_ok += 1
                                break
                self.get_logger().info(
                    'z={:.3f}: row {}/{} (x={:.2f}), {} reachable so far, '
                    '{:.0f} s'.format(z, i + 1, len(self.xs), x, n_ok,
                                      time.time() - t0))
            self.maps[z] = grid
            done[z] = grid
            self.get_logger().info(
                'z={:.3f}: {} of {} (cell, angle) pairs reachable, swept in '
                '{:.0f} s'.format(z, n_ok, grid.size, time.time() - t0))
        return True

    def _lookup(self, xyz, z_key):
        """Which jaw angles work at the cell nearest to xyz."""
        grid = self.maps.get(round(z_key, 4))
        if grid is None:
            return None
        i = int(np.argmin(np.abs(self.xs - xyz[0])))
        j = int(np.argmin(np.abs(self.ys - xyz[1])))
        # Outside the grid entirely: say so rather than snap to the edge.
        if abs(self.xs[i] - xyz[0]) > 0.75 * (self.xs[1] - self.xs[0]) or \
                abs(self.ys[j] - xyz[1]) > 0.75 * (self.ys[1] - self.ys[0]):
            return []
        return [(k, int(grid[i, j, k])) for k in range(len(self.yaws))
                if grid[i, j, k] >= 0]

    # --------------------------------------------------------------- drawing
    def _world_to_px(self, pts_world, R_cw, t_cw):
        """world points (N,3) -> pixel (N,2), or None for points behind."""
        pc = (pts_world - t_cw) @ R_cw          # into the camera frame
        out = np.full((pc.shape[0], 2), np.nan)
        ok = pc[:, 2] > 0.01
        out[ok, 0] = pc[ok, 0] * self.K[0, 0] / pc[ok, 2] + self.K[0, 2]
        out[ok, 1] = pc[ok, 1] * self.K[1, 1] / pc[ok, 2] + self.K[1, 2]
        return out

    def _cam_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame, self.cam_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception:
            return None, None
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        t = tf.transform.translation
        return R, np.array([t.x, t.y, t.z])

    def _draw_region(self, img, R_cw, t_cw, z_key, colour):
        """Tint every reachable cell onto the bench plane."""
        grid = self.maps.get(round(z_key, 4))
        if grid is None:
            return
        any_ok = (grid >= 0).any(axis=2)
        hx = (self.xs[1] - self.xs[0]) / 2.0
        hy = (self.ys[1] - self.ys[0]) / 2.0
        overlay = img.copy()
        for i, x in enumerate(self.xs):
            for j, y in enumerate(self.ys):
                if not any_ok[i, j]:
                    continue
                corners = np.array([[x - hx, y - hy, z_key],
                                    [x + hx, y - hy, z_key],
                                    [x + hx, y + hy, z_key],
                                    [x - hx, y + hy, z_key]])
                px = self._world_to_px(corners, R_cw, t_cw)
                if np.any(np.isnan(px)):
                    continue
                cv2.fillPoly(overlay, [px.astype(np.int32)], colour)
        cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

    def _on_color(self, msg):
        if self.K is None or not self.maps:
            return
        if self.pub.get_subscription_count() == 0:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        R_cw, t_cw = self._cam_pose()
        if R_cw is None:
            cv2.putText(img, 'no TF world <- camera', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))
            return

        # The reachable patch for the OBJECT, in green, at the grasp height.
        self._draw_region(img, R_cw, t_cw, self.grasp_z, (0, 200, 0))

        verdicts = {}
        if self.poses is not None and self.ids:
            for mid, pose in zip(self.ids, self.poses.poses):
                if mid not in (self.pick_id, self.place_id):
                    continue
                p = np.array([pose.position.x, pose.position.y,
                              pose.position.z])
                z_key = self.grasp_z if mid == self.pick_id else self.place_z
                # Judge at the working height, not the tag's own z - the tag
                # is on TOP of the object, the grasp is in its middle.
                ok = self._lookup([p[0], p[1], z_key], z_key)
                px = self._world_to_px(p.reshape(1, 3), R_cw, t_cw)[0]
                if np.any(np.isnan(px)):
                    continue
                u, v = int(px[0]), int(px[1])
                role = 'OBJECT' if mid == self.pick_id else 'PLACE'
                if ok is None:
                    col, txt = (0, 200, 255), '?'
                elif ok:
                    # Amber rather than green when it only works leaned
                    # over, so a marginal spot looks marginal.
                    leaned = all(idx > 0 for _, idx in ok)
                    col = (0, 200, 255) if leaned else (0, 255, 0)
                    txt = 'OK  ' + ' '.join(
                        '{:.0f}{}'.format(math.degrees(self.yaws[k]),
                                          '' if idx == 0 else '(lean)')
                        for k, idx in ok)
                else:
                    col, txt = (0, 0, 255), 'NOT REACHABLE'
                verdicts[mid] = bool(ok)
                cv2.circle(img, (u, v), 22, col, 3)
                cv2.putText(img, '{} id{}: {}'.format(role, mid, txt),
                            (u + 28, v + 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, col, 2)
                cv2.putText(img, '[{:+.2f} {:+.2f}]'.format(p[0], p[1]),
                            (u + 28, v + 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 1)

        # A one-line verdict at the top, big.
        seen = [m for m in (self.pick_id, self.place_id) if m in verdicts]
        if len(seen) == 2 and all(verdicts.values()):
            banner, col = 'BOTH REACHABLE - go', (0, 255, 0)
        elif not seen:
            banner, col = 'no tags in view', (0, 200, 255)
        else:
            missing = [m for m in (self.pick_id, self.place_id)
                       if m not in verdicts]
            bad = [m for m, v in verdicts.items() if not v]
            parts = []
            if missing:
                parts.append('id {} not seen'.format(
                    ', '.join(str(m) for m in missing)))
            if bad:
                parts.append('id {} out of reach'.format(
                    ', '.join(str(m) for m in bad)))
            banner, col = '; '.join(parts), (0, 0, 255)
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, banner, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    col, 2)
        cv2.putText(img, 'green patch = where the object can be grasped '
                    '(z={:.2f})'.format(self.grasp_z),
                    (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 0), 1)

        self.pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))


def main():
    rclpy.init()
    node = ReachOverlay()
    try:
        node.get_logger().info('sweeping IK - this takes up to a minute...')
        if not node.sweep():
            return
        node.get_logger().info(
            'ready. view with: ros2 run rqt_image_view rqt_image_view '
            '/reach_overlay/image')
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
