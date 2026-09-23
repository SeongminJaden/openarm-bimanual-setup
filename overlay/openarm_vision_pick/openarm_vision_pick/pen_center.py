# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Move the left arm until the pen sits in the middle of the wrist camera.

This is position-based visual servoing, and it works because pen_detector
already publishes the pen as a 3-D point in the CAMERA's own optical frame.
In that frame the image centre is the +Z axis, so "the pen is centred" is
simply x = 0, y = 0 - no image-plane Jacobian, no hand-eye calibration beyond
the TF the URDF and the RealSense driver already publish.

One iteration:

  1. read the pen at (x, y, z) in the camera optical frame
  2. translating the camera by (x, y, 0) in that same frame would put the pen
     at (0, 0, z) - dead centre
  3. rotate that step into the planning frame and add it to the CURRENT TCP
     position, leaving the orientation alone
  4. drive there with a Cartesian move, look again

Step 3 is why the loop tolerates a mis-measured camera mount: the step is only
ever a correction, and a correction that lands short is fixed by the next one.
It converges geometrically at `gain` per iteration.

By default the world Z of the TCP is held fixed (`lock_z`), so the arm slides
over the bench at a constant height instead of diving along the camera's
tilted optical axis.  That is a constrained solve, not a truncation: the
horizontal step is chosen so the pen still moves by the full requested amount
in the image, which keeps the convergence rate at `gain` per iteration.

  ros2 run openarm_vision_pick pen_center --ros-args \\
    --params-file <install>/openarm_vision_pick/config/pen.yaml

Runs in dry_run by default: every step is planned and displayed, nothing
moves.  Set dry_run:=false once the planned motion looks right in RViz.

Needs move_group up, the arm controllers up, and pen_detector publishing.
"""

import math
import time

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Pose, PoseStamped, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (Constraints, MotionPlanRequest,
                             OrientationConstraint, PositionConstraint)
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from shape_msgs.msg import SolidPrimitive

from openarm_vision_pick.pick_and_place import top_down_grasp_quat


def rot_from_quat(q):
    """geometry_msgs Quaternion -> 3x3 rotation matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PenCenter(Node):

    def __init__(self):
        super().__init__('pen_center')

        p = self.declare_parameter
        p('group', 'left_arm')
        p('tcp_link', 'openarm_left_hand_tcp')
        p('planning_frame', 'world')
        # The frame pen_detector publishes its poses in.  Poses arriving in
        # any other frame are transformed into this one first, so a detector
        # configured with target_frame: world still works.
        p('camera_frame', 'openarm_left_camera_color_optical_frame')
        p('target_pose_topic', '/pen_detector/pose')
        # Only used to print the error in pixels, which is what you can
        # actually check against the debug image.  Optional.
        p('info_topic', '/camera/camera/color/camera_info')

        # Servo loop.
        p('center_tol_m', 0.006)
        p('gain', 0.6)
        p('max_step_m', 0.05)
        p('max_iters', 12)
        # After a move the detector's EMA is still catching up, and its own
        # stable_count has to be met again before it publishes at all.
        p('settle_s', 1.0)
        p('fresh_msgs', 3)
        p('detection_timeout', 20.0)
        # Hold the TCP height constant and solve for the horizontal step
        # that still delivers the full image-plane correction.
        p('lock_z', True)
        # A 50 mm sidestep is well within the Cartesian planner, but if the
        # straight line clips a joint limit, fall back to a planned motion
        # rather than abandoning the loop.
        p('fallback_to_planner', True)

        # Sanity band on the pen's distance.  A detection outside it is a
        # misfire, and chasing one moves the arm somewhere it has no business
        # being - so stop instead.
        p('depth_min_m', 0.10)
        p('depth_max_m', 0.70)

        # Optionally park at the observe pose first.  Off by default: the
        # observe point was tuned in simulation, so the safe default is to
        # centre from wherever the arm already is.
        p('start_at_observe', False)
        p('observe_xyz', [0.257, 0.20, 0.55])
        p('observe_jaw_yaw', 1.5708)

        p('vel_scale', 0.08)
        p('acc_scale', 0.08)
        p('planning_time', 10.0)
        p('planning_attempts', 16)
        p('pos_tolerance', 0.008)
        p('ori_tolerance', 0.10)
        p('cartesian_step', 0.004)
        p('cartesian_min_fraction', 0.9)

        p('dry_run', True)
        p('loop', False)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.group = g('group')
        self.tcp = g('tcp_link')
        self.frame = g('planning_frame')
        self.cam_frame = g('camera_frame')
        self.tol = g('center_tol_m')
        self.gain = g('gain')
        self.max_step = g('max_step_m')
        self.max_iters = int(g('max_iters'))
        self.settle = g('settle_s')
        self.fresh_msgs = int(g('fresh_msgs'))
        self.det_timeout = g('detection_timeout')
        self.lock_z = g('lock_z')
        self.fallback = g('fallback_to_planner')
        self.depth_min = g('depth_min_m')
        self.depth_max = g('depth_max_m')
        self.start_observe = g('start_at_observe')
        self.observe_xyz = list(g('observe_xyz'))
        self.observe_yaw = float(g('observe_jaw_yaw'))
        self.vel, self.acc = g('vel_scale'), g('acc_scale')
        self.plan_time = g('planning_time')
        self.plan_tries = g('planning_attempts')
        self.pos_tol, self.ori_tol = g('pos_tolerance'), g('ori_tolerance')
        self.cart_step = g('cartesian_step')
        self.cart_min = g('cartesian_min_fraction')
        self.dry_run = g('dry_run')
        self.do_loop = g('loop')

        self.K = None
        self.msgs = []
        self._warned_amp = False

        self.create_subscription(PoseStamped, g('target_pose_topic'),
                                 self._on_target, 10)
        self.create_subscription(CameraInfo, g('info_topic'),
                                 self._on_info, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.exec_client = ActionClient(self, ExecuteTrajectory,
                                        '/execute_trajectory')
        self.cart_client = self.create_client(GetCartesianPath,
                                              '/compute_cartesian_path')

        if self.dry_run:
            self.get_logger().warn(
                'DRY RUN - the first correction will be planned and displayed '
                'but not executed, and the loop then stops, because with the '
                'arm still the error cannot change. Set dry_run:=false to '
                'actually centre the pen.')

    # ----------------------------------------------------------------- plumbing
    def _on_target(self, msg):
        self.msgs.append(msg)

    def _on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _spin(self, future, timeout=60.0):
        end = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result() if future.done() else None

    def _sleep(self, seconds):
        """Sleep while still servicing callbacks, so poses keep arriving."""
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_servers(self, timeout=30.0):
        ok = True
        for name, client in (('/move_action', self.move_client),
                             ('/execute_trajectory', self.exec_client)):
            if not client.wait_for_server(timeout_sec=timeout):
                self.get_logger().error(f'action server {name} not available')
                ok = False
        if not self.cart_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                'service /compute_cartesian_path not available')
            ok = False
        return ok

    # --------------------------------------------------------------------- TF
    def _tcp_pose(self):
        """Current TCP pose in the planning frame, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame, self.tcp, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as exc:
            self.get_logger().error(
                f'no TF {self.frame} <- {self.tcp}: {exc}')
            return None
        pose = Pose()
        pose.position.x = tf.transform.translation.x
        pose.position.y = tf.transform.translation.y
        pose.position.z = tf.transform.translation.z
        pose.orientation = tf.transform.rotation
        return pose

    def _rot_planning_from_camera(self):
        """Rotation that carries a camera-frame vector into the planning frame."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame, self.cam_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as exc:
            self.get_logger().error(
                f'no TF {self.frame} <- {self.cam_frame}: {exc}')
            return None
        return rot_from_quat(tf.transform.rotation)

    def _to_camera(self, msg):
        """The detection as (x, y, z) in the camera optical frame."""
        pos = msg.pose.position
        p = np.array([pos.x, pos.y, pos.z])
        src = msg.header.frame_id
        if src == self.cam_frame:
            return p
        try:
            tf = self.tf_buffer.lookup_transform(
                self.cam_frame, src, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as exc:
            self.get_logger().error(
                f'no TF {self.cam_frame} <- {src}: {exc}')
            return None
        t = tf.transform.translation
        return rot_from_quat(tf.transform.rotation) @ p + \
            np.array([t.x, t.y, t.z])

    # ------------------------------------------------------------------ motion
    def _base_request(self):
        req = MotionPlanRequest()
        req.group_name = self.group
        req.num_planning_attempts = int(self.plan_tries)
        req.allowed_planning_time = float(self.plan_time)
        req.max_velocity_scaling_factor = float(self.vel)
        req.max_acceleration_scaling_factor = float(self.acc)
        req.workspace_parameters.header.frame_id = self.frame
        req.workspace_parameters.min_corner = Vector3(x=-2.0, y=-2.0, z=-2.0)
        req.workspace_parameters.max_corner = Vector3(x=2.0, y=2.0, z=2.0)
        return req

    def _send_move(self, req, label):
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = bool(self.dry_run)
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        handle = self._spin(self.move_client.send_goal_async(goal), 20.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: goal rejected')
            return False
        result = self._spin(handle.get_result_async(), self.plan_time + 60.0)
        if result is None:
            self.get_logger().error(f'{label}: timed out')
            return False
        code = result.result.error_code.val
        if code != 1:
            self.get_logger().error(f'{label}: MoveItErrorCode {code}')
            return False
        self.get_logger().info(
            f'{label}: {"planned" if self.dry_run else "done"}')
        return True

    def move_to_pose(self, pose, label):
        req = self._base_request()
        c = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = self.frame
        pc.link_name = self.tcp
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(self.pos_tol)]
        pc.constraint_region.primitives.append(sphere)
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0
        c.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = self.frame
        oc.link_name = self.tcp
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = float(self.ori_tol)
        oc.absolute_y_axis_tolerance = float(self.ori_tol)
        oc.absolute_z_axis_tolerance = float(self.ori_tol)
        oc.weight = 1.0
        c.orientation_constraints.append(oc)

        req.goal_constraints.append(c)
        return self._send_move(req, label)

    def cartesian_to(self, pose, label):
        req = GetCartesianPath.Request()
        req.header.frame_id = self.frame
        req.start_state.is_diff = True
        req.group_name = self.group
        req.link_name = self.tcp
        req.waypoints = [pose]
        req.max_step = float(self.cart_step)
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        for attr, val in (('max_velocity_scaling_factor', self.vel),
                          ('max_acceleration_scaling_factor', self.acc)):
            if hasattr(req, attr):
                setattr(req, attr, float(val))

        res = self._spin(self.cart_client.call_async(req), 30.0)
        if res is None:
            self.get_logger().error(f'{label}: Cartesian service timed out')
            return False
        if res.fraction < self.cart_min:
            self.get_logger().warn(
                f'{label}: only {res.fraction*100:.0f}% of the straight line '
                f'is reachable (need {self.cart_min*100:.0f}%)')
            if self.fallback:
                self.get_logger().info(f'{label}: falling back to the planner')
                return self.move_to_pose(pose, label + ' (planned)')
            return False
        self.get_logger().info(f'{label}: path {res.fraction*100:.0f}% solved')
        if self.dry_run:
            return True

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        handle = self._spin(self.exec_client.send_goal_async(goal), 20.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: execution rejected')
            return False
        result = self._spin(handle.get_result_async(), 120.0)
        if result is None or result.result.error_code.val != 1:
            code = 'timeout' if result is None else result.result.error_code.val
            self.get_logger().error(f'{label}: execution failed ({code})')
            return False
        return True

    # ------------------------------------------------------------------- vision
    def _fresh_detection(self):
        """Discard what is queued and wait for `fresh_msgs` new poses."""
        self.msgs = []
        end = time.time() + self.det_timeout
        while rclpy.ok() and len(self.msgs) < self.fresh_msgs \
                and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if len(self.msgs) < self.fresh_msgs:
            self.get_logger().error(
                f'only {len(self.msgs)} detection(s) in '
                f'{self.det_timeout:.0f} s - is the pen in view and is '
                'pen_detector publishing?')
            return None
        return self.msgs[-1]

    def _pixel_error(self, p_cam):
        """(du, dv) from the principal point, or None without intrinsics."""
        if self.K is None or p_cam[2] <= 0.0:
            return None
        return (self.K[0, 0] * p_cam[0] / p_cam[2],
                self.K[1, 1] * p_cam[1] / p_cam[2])

    # --------------------------------------------------------------------- loop
    def goto_observe(self):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = \
            [float(v) for v in self.observe_xyz]
        q = top_down_grasp_quat(self.observe_yaw)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, \
            pose.orientation.w = [float(v) for v in q]
        return self.move_to_pose(pose, 'observe')

    def center_once(self):
        if self.start_observe and not self.goto_observe():
            return False

        for i in range(1, self.max_iters + 1):
            msg = self._fresh_detection()
            if msg is None:
                return False
            p_cam = self._to_camera(msg)
            if p_cam is None:
                return False

            if not (self.depth_min < p_cam[2] < self.depth_max):
                self.get_logger().error(
                    'pen reported at {:.3f} m, outside the sanity band '
                    '{:.2f}..{:.2f} m - refusing to chase it'.format(
                        p_cam[2], self.depth_min, self.depth_max))
                return False

            err = math.hypot(p_cam[0], p_cam[1])
            px = self._pixel_error(p_cam)
            self.get_logger().info(
                'iter {}/{}: offset [{:+.1f} {:+.1f}] mm, |e| = {:.1f} mm, '
                'depth {:.0f} mm{}'.format(
                    i, self.max_iters, p_cam[0] * 1000, p_cam[1] * 1000,
                    err * 1000, p_cam[2] * 1000,
                    '' if px is None else
                    '  ({:+.0f} {:+.0f}) px from centre'.format(*px)))

            if err <= self.tol:
                self.get_logger().info(
                    'centred: {:.1f} mm <= {:.1f} mm tolerance'.format(
                        err * 1000, self.tol * 1000))
                return True

            # Translating the camera by (x, y, 0) in its own frame lands the
            # pen on the optical axis.  Damped by `gain` so a bad depth
            # reading cannot throw the arm across the bench in one go.
            want = np.array([p_cam[0], p_cam[1]]) * self.gain
            R = self._rot_planning_from_camera()
            if R is None:
                return False

            if self.lock_z:
                # Hold the height and still deliver the whole correction.
                #
                # Simply dropping the Z component of the free-space step would
                # be wrong: it throws away however much of the correction
                # pointed along world Z, so a camera tilted 53 degrees loses
                # most of it and the loop crawls (measured: an effective gain
                # of 0.22 instead of 0.6, more than 20 iterations to converge).
                #
                # Solve for the horizontal step instead.  Moving the camera by
                # d shifts the pen to p_cam - R^T d, so the two image-plane
                # components we care about are A @ (dx, dy) with A the
                # horizontal part of R^T's first two rows.  det(A) is R[2][2],
                # the world-Z component of the optical axis: it vanishes only
                # for a camera looking horizontally, which no horizontal move
                # can ever shift vertically in the image.
                A = np.array([[R[0, 0], R[1, 0]],
                              [R[0, 1], R[1, 1]]])
                if abs(R[2, 2]) < 0.05:
                    self.get_logger().error(
                        'the optical axis is within 3 degrees of horizontal, '
                        'so no level move can centre the pen vertically; set '
                        'lock_z:=false')
                    return False
                # 1/|R[2][2]| is how much longer the level step has to be
                # than the correction it buys.  Near 1 for a camera looking
                # down; it grows without bound as the optical axis levels out,
                # and past a few times the step spends its whole budget on the
                # max_step clamp and the loop crawls.  Say so once.
                amp = 1.0 / abs(R[2, 2])
                if amp > 3.0 and not self._warned_amp:
                    self._warned_amp = True
                    self.get_logger().warn(
                        'the optical axis is {:.0f} degrees off vertical, so a '
                        'level step must be {:.1f}x the correction it buys; '
                        'expect slow convergence, and set lock_z:=false to '
                        'move perpendicular to the axis instead'.format(
                            math.degrees(math.acos(min(1.0, abs(R[2, 2])))),
                            amp))
                dxy = np.linalg.solve(A, want)
                step = np.array([dxy[0], dxy[1], 0.0])
            else:
                step = R @ np.array([want[0], want[1], 0.0])

            n = float(np.linalg.norm(step))
            if n < 1e-9:
                self.get_logger().error('degenerate correction; giving up')
                return False
            if n > self.max_step:
                step *= self.max_step / n
                n = self.max_step

            cur = self._tcp_pose()
            if cur is None:
                return False
            goal = Pose()
            goal.position.x = cur.position.x + float(step[0])
            goal.position.y = cur.position.y + float(step[1])
            goal.position.z = cur.position.z + float(step[2])
            goal.orientation = cur.orientation

            self.get_logger().info(
                '  step [{:+.1f} {:+.1f} {:+.1f}] mm in {}'.format(
                    step[0] * 1000, step[1] * 1000, step[2] * 1000,
                    self.frame))
            if not self.cartesian_to(goal, f'centre {i}'):
                return False

            if self.dry_run:
                self.get_logger().warn(
                    'dry run: the arm did not move, so the error would not '
                    'change - stopping after one step')
                return True
            self._sleep(self.settle)

        self.get_logger().warn(
            f'gave up after {self.max_iters} iterations; raise max_iters or '
            'gain, or loosen center_tol_m')
        return False


def main():
    rclpy.init()
    node = PenCenter()
    try:
        if not node.wait_for_servers():
            node.get_logger().error('MoveIt is not up; is move_group running?')
            return
        while rclpy.ok():
            ok = node.center_once()
            if not node.do_loop:
                break
            if not ok:
                node.get_logger().warn('retrying in 3 s')
                node._sleep(3.0)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C, or a SIGTERM from whatever started this.  Both are how the
        # tracking loop is meant to be stopped, so neither deserves a
        # traceback - rclpy raises ExternalShutdownException out of
        # spin_once() the moment the context goes down, and an unhandled one
        # buries any real error that came before it.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
