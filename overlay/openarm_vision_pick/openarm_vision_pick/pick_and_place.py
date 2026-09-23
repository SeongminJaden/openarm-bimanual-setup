# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Vision-guided pick and place for the OpenArm left arm.

Drives MoveIt through its action/service interface directly, because Humble
ships no Python MoveIt binding (moveit_py arrived in Iron).

  observe pose -> wait for a stable detection -> pre-grasp -> Cartesian
  approach -> close -> Cartesian lift -> transfer -> Cartesian place -> open
  -> Cartesian retreat -> observe pose

Run with dry_run:=true to plan every step and display it without moving
anything; that is the only mode that is safe to point at real hardware before
the trajectories have been eyeballed.
"""

import math
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (Constraints, JointConstraint, MotionPlanRequest,
                             OrientationConstraint, PositionConstraint)
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def quat_from_matrix(m):
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


def top_down_grasp_quat(jaw_yaw):
    """Orientation for a straight-down grasp with the jaws across `jaw_yaw`.

    The hand frame's +Z is the approach direction and its +Y is the axis the
    fingers travel along, so +Z is pointed at the floor and +Y is laid across
    the object.
    """
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(jaw_yaw), math.sin(jaw_yaw), 0.0])
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return quat_from_matrix(np.column_stack((x, y, z)))



def tilted_grasp_quat(jaw_yaw, tilt=0.0, about='pitch', sign=1.0):
    """top_down_grasp_quat, then leaned over by `tilt` radians.

    A grasp does not have to come in dead vertical, and insisting that it
    does is what made the reachable region so small: with the base 0.7 m up
    and the bench 0.25 m up, holding the hand exactly plumb at the far end of
    the reach folds the wrist to its limit.  Positions the arm can plainly
    get to - it has been put there by hand - came back "no IK" for want of
    twenty degrees of lean.

    about='pitch' rotates about the jaw axis, so the jaws still close along
    a horizontal line and only the approach leans; that is the one to prefer
    for a box.  'roll' rotates about the other in-plane axis and lowers one
    jaw below the other - still a grip, just a less tidy one, and worth
    having when pitch alone will not solve.
    """
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(jaw_yaw), math.sin(jaw_yaw), 0.0])
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    if abs(tilt) > 1e-9:
        k = y if about == 'pitch' else x
        K = np.array([[0.0, -k[2], k[1]],
                      [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]])
        t = sign * tilt
        R = np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * (K @ K)
        x, y, z = R @ x, R @ y, R @ z
    return quat_from_matrix(np.column_stack((x, y, z)))


def grasp_orientation_candidates(jaw_yaw, tilts_rad):
    """Every orientation worth trying for one jaw angle, best first.

    Yields (quat, tilt_deg, label).  Straight down comes first; then each
    tilt magnitude in the order given, pitching before rolling, both signs.
    Callers stop at the first that solves, so the order IS the preference.
    """
    yield top_down_grasp_quat(jaw_yaw), 0.0, 'vertical'
    for t in tilts_rad:
        if t <= 1e-9:
            continue
        for about, sign, name in (('pitch', 1.0, 'pitch+'),
                                  ('pitch', -1.0, 'pitch-'),
                                  ('roll', 1.0, 'roll+'),
                                  ('roll', -1.0, 'roll-')):
            yield (tilted_grasp_quat(jaw_yaw, t, about, sign),
                   math.degrees(t), '{} {:.0f}deg'.format(name, math.degrees(t)))

class PickAndPlace(Node):

    def __init__(self):
        super().__init__('pick_and_place')

        p = self.declare_parameter
        p('group', 'left_arm')
        p('tcp_link', 'openarm_left_hand_tcp')
        p('planning_frame', 'world')
        p('target_pose_topic', '/object_detector/pose')

        p('gripper_action', '/left_gripper_controller/follow_joint_trajectory')
        # Two entries in simulation (Gazebo ignores <mimic>), one on hardware.
        p('gripper_joints', ['openarm_left_finger_joint1',
                             'openarm_left_finger_joint2'])
        p('gripper_open', 0.040)
        p('gripper_closed', 0.012)
        p('gripper_time', 1.0)

        # Where to park the TCP while looking for the object.  The wrist camera
        # does not look straight down: in hand coordinates its optical axis is
        # roughly (-0.6, 0, +0.8), i.e. along the approach direction but tilted
        # towards hand -X.  So from a height h the camera's ray lands about
        # 0.75*h away in that direction, and the observe point has to be offset
        # from the object by that much for the object to sit near the centre of
        # the frame rather than at its edge.
        p('observe_xyz', [0.33, 0.406, 0.60])
        p('observe_jaw_yaw', 0.0)
        p('home_joints', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # TCP sits this far above the reported object centre when grasping.
        p('grasp_z_offset', 0.005)
        p('approach_height', 0.12)
        p('lift_height', 0.15)
        p('place_xyz', [0.33, 0.02, 0.4175])

        p('vel_scale', 0.15)
        p('acc_scale', 0.15)
        p('planning_time', 8.0)
        p('planning_attempts', 12)
        p('pos_tolerance', 0.008)
        p('ori_tolerance', 0.10)
        p('cartesian_step', 0.005)
        p('cartesian_min_fraction', 0.85)

        p('detection_timeout', 60.0)
        p('dry_run', True)
        p('loop', False)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.group = g('group')
        self.tcp = g('tcp_link')
        self.frame = g('planning_frame')
        self.gripper_joints = list(g('gripper_joints'))
        self.g_open, self.g_closed = g('gripper_open'), g('gripper_closed')
        self.g_time = g('gripper_time')
        self.observe_xyz = list(g('observe_xyz'))
        self.observe_yaw = float(g('observe_jaw_yaw'))
        self.home_q = list(g('home_joints'))
        self.grasp_dz = g('grasp_z_offset')
        self.approach_h = g('approach_height')
        self.lift_h = g('lift_height')
        self.place_xyz = list(g('place_xyz'))
        self.vel, self.acc = g('vel_scale'), g('acc_scale')
        self.plan_time = g('planning_time')
        self.plan_tries = g('planning_attempts')
        self.pos_tol, self.ori_tol = g('pos_tolerance'), g('ori_tolerance')
        self.cart_step = g('cartesian_step')
        self.cart_min = g('cartesian_min_fraction')
        self.det_timeout = g('detection_timeout')
        self.dry_run = g('dry_run')
        self.do_loop = g('loop')

        self.joint_names = [f'openarm_left_joint{i}' for i in range(1, 8)] \
            if self.group == 'left_arm' else \
            [f'openarm_right_joint{i}' for i in range(1, 8)]

        self.latest = None
        self.create_subscription(PoseStamped, g('target_pose_topic'),
                                 self._on_target, 10)

        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.exec_client = ActionClient(self, ExecuteTrajectory,
                                        '/execute_trajectory')
        self.grip_client = ActionClient(self, FollowJointTrajectory,
                                        g('gripper_action'))
        self.cart_client = self.create_client(GetCartesianPath,
                                              '/compute_cartesian_path')

        if self.dry_run:
            self.get_logger().warn(
                'DRY RUN - every motion will be planned and displayed but not '
                'executed. Set dry_run:=false to actually move.')

    # ----------------------------------------------------------------- plumbing
    def _on_target(self, msg):
        self.latest = msg

    def _spin(self, future, timeout=60.0):
        end = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result() if future.done() else None

    def wait_for_servers(self, timeout=30.0):
        ok = True
        for name, client in (('/move_action', self.move_client),
                             ('/execute_trajectory', self.exec_client),
                             ('gripper', self.grip_client)):
            if not client.wait_for_server(timeout_sec=timeout):
                self.get_logger().error(f'action server {name} not available')
                ok = False
        if not self.cart_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error('service /compute_cartesian_path not available')
            ok = False
        return ok

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
        result = self._spin(handle.get_result_async(),
                            self.plan_time + 60.0)
        if result is None:
            self.get_logger().error(f'{label}: timed out')
            return False
        code = result.result.error_code.val
        if code != 1:
            self.get_logger().error(f'{label}: MoveItErrorCode {code}')
            return False
        n = len(result.result.planned_trajectory.joint_trajectory.points)
        self.get_logger().info(
            f'{label}: {"planned" if self.dry_run else "done"} ({n} points)')
        return True

    def move_to_joints(self, q, label):
        req = self._base_request()
        c = Constraints()
        for name, value in zip(self.joint_names, q):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        return self._send_move(req, label)

    def move_to_pose(self, pose: Pose, label):
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

    def cartesian_to(self, pose: Pose, label):
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
            self.get_logger().error(
                f'{label}: only {res.fraction*100:.0f}% of the straight-line '
                f'path is reachable (need {self.cart_min*100:.0f}%)')
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

    def gripper(self, width, label):
        if self.dry_run:
            self.get_logger().info(f'{label}: would move gripper to {width:.3f} m')
            return True
        pt = JointTrajectoryPoint()
        pt.positions = [float(width)] * len(self.gripper_joints)
        pt.time_from_start.sec = int(self.g_time)
        pt.time_from_start.nanosec = int((self.g_time % 1.0) * 1e9)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.gripper_joints
        goal.trajectory.points = [pt]

        handle = self._spin(self.grip_client.send_goal_async(goal), 15.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: gripper goal rejected')
            return False
        self._spin(handle.get_result_async(), self.g_time + 10.0)
        self.get_logger().info(f'{label}: gripper at {width:.3f} m')
        return True

    # ------------------------------------------------------------------- vision
    def wait_for_target(self):
        self.latest = None
        self.get_logger().info('waiting for a stable detection...')
        end = time.time() + self.det_timeout
        while rclpy.ok() and self.latest is None and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.latest is None:
            self.get_logger().error(
                f'no object detected within {self.det_timeout:.0f} s')
            return None
        p = self.latest.pose.position
        self.get_logger().info(
            f'target at [{p.x:+.3f} {p.y:+.3f} {p.z:+.3f}] in {self.latest.header.frame_id}')
        return self.latest

    # -------------------------------------------------------------------- cycle
    @staticmethod
    def _pose(xyz, quat):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [float(v) for v in xyz]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = \
            [float(v) for v in quat]
        return pose

    def goto_observe(self, label='observe'):
        pose = self._pose(self.observe_xyz, top_down_grasp_quat(self.observe_yaw))
        return self.move_to_pose(pose, label)

    def run_once(self):
        if not self.goto_observe():
            return False
        if not self.gripper(self.g_open, 'open'):
            return False

        target = self.wait_for_target()
        if target is None:
            return False

        q = target.pose.orientation
        jaw_yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z)) + math.pi / 2.0
        grasp_quat = top_down_grasp_quat(jaw_yaw)

        t = target.pose.position
        grasp = [t.x, t.y, t.z + self.grasp_dz]
        pre = [grasp[0], grasp[1], grasp[2] + self.approach_h]
        lift = [grasp[0], grasp[1], grasp[2] + self.lift_h]
        place = list(self.place_xyz)
        place_pre = [place[0], place[1], place[2] + self.approach_h]

        steps = [
            ('pre-grasp', lambda: self.move_to_pose(self._pose(pre, grasp_quat), 'pre-grasp')),
            ('approach', lambda: self.cartesian_to(self._pose(grasp, grasp_quat), 'approach')),
            ('close', lambda: self.gripper(self.g_closed, 'close')),
            ('lift', lambda: self.cartesian_to(self._pose(lift, grasp_quat), 'lift')),
            ('transfer', lambda: self.move_to_pose(self._pose(place_pre, grasp_quat), 'transfer')),
            ('lower', lambda: self.cartesian_to(self._pose(place, grasp_quat), 'lower')),
            ('release', lambda: self.gripper(self.g_open, 'release')),
            ('retreat', lambda: self.cartesian_to(self._pose(place_pre, grasp_quat), 'retreat')),
            ('home', lambda: self.goto_observe('home')),
        ]
        for name, step in steps:
            if not step():
                self.get_logger().error(f'aborting at step "{name}"')
                return False
        self.get_logger().info('cycle complete')
        return True


def main():
    rclpy.init()
    node = PickAndPlace()
    try:
        if not node.wait_for_servers():
            node.get_logger().error('MoveIt is not up; is move_group running?')
            return
        while rclpy.ok():
            ok = node.run_once()
            if not node.do_loop:
                break
            if not ok:
                node.get_logger().warn('retrying in 3 s')
                time.sleep(3.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
