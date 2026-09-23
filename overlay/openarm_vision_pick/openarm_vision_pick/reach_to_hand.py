# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Move the arm towards a hand seen by a body-mounted camera.

This is the eye-to-hand counterpart of pen_center, and the difference is not
cosmetic.  pen_center servos a camera that RIDES on the arm: moving the arm
changes what the camera sees, so "put the target in the middle of the frame"
is a goal the arm can act on.  A chest camera is bolted to the body - moving
the arm changes nothing it sees - so that goal is unreachable by construction.

What a body-mounted camera gives instead is the hand's position in a fixed
frame, and what the arm can do with it is go there.  So:

  1. hand_detector publishes the hand in `world` (it must: a point in a
     camera's own frame is not a place, and this node refuses anything that
     does not arrive in planning_frame).
  2. Stop short of it.  The goal is pulled back along the line from the robot
     base to the hand by `standoff_m`, because the point of this is to reach
     TOWARDS a person, not into them.
  3. Clamp that into a workspace box and refuse anything outside it, so a
     misdetection at the far wall cannot become a full-extension lunge.
  4. Plan and, unless dry_run, execute.

Safety, in the order it bites:

  * dry_run defaults to true - it plans and displays, and moves nothing.
  * the workspace box is checked BEFORE planning, and a violation aborts.
  * standoff_m keeps the TCP away from the hand itself.
  * max_step_m limits how far any single move may travel from where the arm
    is now, so a jump in the detection cannot become a fast long throw.
  * vel_scale/acc_scale are deliberately low.

  ros2 run openarm_vision_pick reach_to_hand --ros-args \\
    --params-file <config>/chest.yaml

Needs move_group, the controllers, and hand_detector publishing in world.
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
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from openarm_vision_pick.pick_and_place import top_down_grasp_quat


class ReachToHand(Node):

    def __init__(self):
        super().__init__('reach_to_hand')

        p = self.declare_parameter
        p('group', 'left_arm')
        p('tcp_link', 'openarm_left_hand_tcp')
        p('planning_frame', 'world')
        p('base_link', 'openarm_body_link0')
        p('target_pose_topic', '/hand_detector/pose')

        # How far short of the hand to stop, along the base->hand line.
        p('standoff_m', 0.15)
        # How far the TCP may travel in one commanded move.  This is a cap on
        # TRAVEL, not a sanity check on the target - the arm starts from rest
        # hanging straight down, so the very first move to anything useful is
        # most of the reach, and a tight value here refuses the one move that
        # was always going to be long.  The workspace box is what bounds
        # where the arm may go; this only bounds how far it goes at once.
        p('max_step_m', 0.80)

        # The anti-jump guard, expressed as a SPEED rather than a distance.
        #
        # A fixed distance does not survive contact with tracking: one cycle
        # here is a full plan-and-execute, a second or two, and a hand moves
        # a long way in a second.  A 0.20 m cap therefore rejected ordinary
        # human movement as if it were a misdetection.  What actually
        # separates a moving hand from a bad frame is how fast the target
        # would have had to travel, so the allowance grows with the time
        # since the last goal.
        #
        # 1.5 m/s is brisk for a hand being deliberately followed; a
        # detection that flickers to the far side of the bench between two
        # frames still exceeds it.  jump_floor_m keeps a small allowance so
        # quick successive goals are not refused by arithmetic alone.
        p('max_target_speed_m_s', 1.5)
        p('jump_floor_m', 0.15)

        # Workspace box in planning_frame.  Nothing outside this is ever
        # planned to, whatever the camera says.  MEASURE THESE on the bench
        # before running with dry_run:=false.
        p('ws_min', [0.10, -0.10, 0.25])
        p('ws_max', [0.60, 0.60, 0.80])

        # Orientation to hold while reaching.  jaw_yaw 0 has no IK solution
        # anywhere over this table - swept through /compute_ik - so 90 deg is
        # the value that works.
        p('jaw_yaw', 1.5708)

        # Re-plan when the hand has moved this far since the last goal.
        p('retarget_m', 0.05)
        p('settle_s', 0.5)
        p('detection_timeout', 20.0)
        # While tracking, a lost hand should not stall the loop for the full
        # detection_timeout - it should notice quickly and keep watching.
        p('loop_detection_timeout', 3.0)
        p('fresh_msgs', 2)

        p('vel_scale', 0.08)
        p('acc_scale', 0.08)
        p('planning_time', 10.0)
        p('planning_attempts', 16)
        p('pos_tolerance', 0.015)
        p('ori_tolerance', 0.20)

        # Discovery of an action can take far longer than discovery
        # of its node, especially on a busy graph.  30 s was not
        # enough here.
        p('server_timeout_s', 90.0)

        p('dry_run', True)
        p('loop', False)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.group = g('group')
        self.tcp = g('tcp_link')
        self.frame = g('planning_frame')
        self.base = g('base_link')
        self.standoff = g('standoff_m')
        self.max_step = g('max_step_m')
        self.max_speed = g('max_target_speed_m_s')
        self.jump_floor = g('jump_floor_m')
        self.ws_min = np.array(g('ws_min'), dtype=float)
        self.ws_max = np.array(g('ws_max'), dtype=float)
        self.jaw_yaw = float(g('jaw_yaw'))
        self.retarget = g('retarget_m')
        self.settle = g('settle_s')
        self.det_timeout = g('detection_timeout')
        self.loop_timeout = g('loop_detection_timeout')
        self.fresh_msgs = int(g('fresh_msgs'))
        self.vel, self.acc = g('vel_scale'), g('acc_scale')
        self.plan_time = g('planning_time')
        self.plan_tries = g('planning_attempts')
        self.pos_tol, self.ori_tol = g('pos_tolerance'), g('ori_tolerance')
        self.server_timeout = g('server_timeout_s')
        self.dry_run = g('dry_run')
        self.do_loop = g('loop')

        self.msgs = []
        self.create_subscription(PoseStamped, g('target_pose_topic'),
                                 self._on_target, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.add_on_set_parameters_callback(self._on_param_set)

        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.exec_client = ActionClient(self, ExecuteTrajectory,
                                        '/execute_trajectory')
        self.cart_client = self.create_client(GetCartesianPath,
                                              '/compute_cartesian_path')

        self.get_logger().info(
            'reaching towards {} in {}\n  standoff {:.0f} mm, max step '
            '{:.0f} mm\n  workspace x {:.2f}..{:.2f}  y {:.2f}..{:.2f}  '
            'z {:.2f}..{:.2f}'.format(
                g('target_pose_topic'), self.frame, self.standoff * 1000,
                self.max_step * 1000, self.ws_min[0], self.ws_max[0],
                self.ws_min[1], self.ws_max[1], self.ws_min[2],
                self.ws_max[2]))
        if self.dry_run:
            self.get_logger().warn(
                'DRY RUN - goals are planned and displayed, the arm does not '
                'move. Set dry_run:=false only once the workspace box above '
                'is right for your bench.')

    # ------------------------------------------------------------ live tuning
    # The workspace box is the thing most likely to need a nudge on the
    # bench, and restarting the node to move a boundary by a centimetre is
    # absurd - especially since a refusal prints the goal it refused, so the
    # number you want is right there in the log.
    #
    # dry_run is settable too, on purpose: `ros2 param set /reach_to_hand
    # dry_run true` stops the arm being commanded any further without having
    # to find and kill the process.
    _LIVE = {
        'standoff_m': 'standoff', 'max_step_m': 'max_step',
        'retarget_m': 'retarget', 'settle_s': 'settle',
        'max_target_speed_m_s': 'max_speed', 'jump_floor_m': 'jump_floor',
        'loop_detection_timeout': 'loop_timeout',
        'vel_scale': 'vel', 'acc_scale': 'acc',
        'jaw_yaw': 'jaw_yaw', 'dry_run': 'dry_run',
        'detection_timeout': 'det_timeout',
    }

    def _on_param_set(self, params):
        for prm in params:
            if prm.name in ('ws_min', 'ws_max'):
                v = np.array(list(prm.value), dtype=float)
                if v.shape != (3,):
                    return SetParametersResult(
                        successful=False,
                        reason='{} needs three numbers'.format(prm.name))
                other = self.ws_max if prm.name == 'ws_min' else self.ws_min
                lo = v if prm.name == 'ws_min' else other
                hi = other if prm.name == 'ws_min' else v
                if np.any(lo >= hi):
                    return SetParametersResult(
                        successful=False,
                        reason='ws_min must be below ws_max on every axis')
                setattr(self, 'ws_min' if prm.name == 'ws_min' else 'ws_max',
                        v)
                self.get_logger().info('{} = {}'.format(prm.name, list(v)))
                continue
            attr = self._LIVE.get(prm.name)
            if attr is None:
                continue
            setattr(self, attr, prm.value)
            self.get_logger().info('{} = {}'.format(prm.name, prm.value))
            if prm.name == 'dry_run' and prm.value:
                self.get_logger().warn(
                    'dry_run is on again - nothing further will be executed')
        return SetParametersResult(successful=True)

    # ----------------------------------------------------------------- plumbing
    def _on_target(self, msg):
        self.msgs.append(msg)

    def _spin(self, future, timeout=60.0):
        end = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result() if future.done() else None

    def _sleep(self, seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_servers(self, timeout=None):
        """Wait for MoveIt, and say which of the two problems it is.

        "action server not available" reads like "move_group is not running",
        and it usually is not that.  Discovery of an ACTION - a bundle of
        three services and two topics - can take appreciably longer than
        discovery of the node that owns it, and longer again on a busy graph:
        move_group here answered a node listing immediately but took most of
        a minute to be found as an action server, while a bare graph found it
        in twelve seconds.

        So this retries, and looks at the node list before complaining.  If
        move_group is there, the message says so, because the fix is patience
        or a longer timeout, not restarting anything.
        """
        timeout = self.server_timeout if timeout is None else timeout
        wanted = [('/move_action', self.move_client),
                  ('/execute_trajectory', self.exec_client)]
        if getattr(self, 'grip_client', None) is not None:
            wanted.append(('gripper', self.grip_client))

        deadline = time.time() + timeout
        pending = list(wanted)
        said = 0.0
        while pending and time.time() < deadline:
            pending = [(n, c) for n, c in pending
                       if not c.wait_for_server(timeout_sec=2.0)]
            if pending and time.time() - said > 10.0:
                said = time.time()
                self.get_logger().info(
                    'waiting for {} ({:.0f}s left)'.format(
                        ', '.join(n for n, _ in pending),
                        deadline - time.time()))

        if not self.cart_client.wait_for_service(
                timeout_sec=max(1.0, deadline - time.time())):
            pending.append(('/compute_cartesian_path', None))

        if not pending:
            return True

        names = [n for n, _ in pending]
        seen = [x[0] for x in self.get_node_names_and_namespaces()]
        self.get_logger().error(
            'gave up waiting for: {}'.format(', '.join(names)))
        if 'move_group' in seen:
            self.get_logger().error(
                'move_group IS on the graph, so it is running - its action '
                'server just was not discovered in {:.0f} s. Raise '
                'server_timeout, or give the launch longer to settle before '
                'starting this node.'.format(timeout))
        else:
            self.get_logger().error(
                'move_group is not on the graph at all. Start the robot: '
                'ros2 launch openarm_vision_pick hand_demo.launch.py')
        return False

    def _tcp_now(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame, self.tcp, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as exc:
            self.get_logger().error(f'no TF {self.frame} <- {self.tcp}: {exc}')
            return None
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z])

    def _base_origin(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame, self.base, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception:
            return np.zeros(3)
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z])

    # ------------------------------------------------------------------ motion
    def move_to(self, xyz, quat, label):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = \
            [float(v) for v in xyz]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, \
            pose.orientation.w = [float(v) for v in quat]

        req = MotionPlanRequest()
        req.group_name = self.group
        req.num_planning_attempts = int(self.plan_tries)
        req.allowed_planning_time = float(self.plan_time)
        req.max_velocity_scaling_factor = float(self.vel)
        req.max_acceleration_scaling_factor = float(self.acc)
        req.workspace_parameters.header.frame_id = self.frame
        req.workspace_parameters.min_corner = Vector3(x=-2.0, y=-2.0, z=-2.0)
        req.workspace_parameters.max_corner = Vector3(x=2.0, y=2.0, z=2.0)

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

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = bool(self.dry_run)
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        handle = self._spin(self.move_client.send_goal_async(goal), 20.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: goal rejected')
            return False
        res = self._spin(handle.get_result_async(), self.plan_time + 60.0)
        if res is None:
            self.get_logger().error(f'{label}: timed out')
            return False
        code = res.result.error_code.val
        if code != 1:
            self.get_logger().error(f'{label}: MoveItErrorCode {code}')
            return False
        self.get_logger().info(
            f'{label}: {"planned" if self.dry_run else "done"}')
        return True

    # ------------------------------------------------------------------- vision
    def _fresh_hand(self, timeout=None, quiet=False):
        """Wait for fresh detections.  `quiet` is for the tracking loop,
        where a hand out of view is normal and not worth an error a second."""
        timeout = self.det_timeout if timeout is None else timeout
        self.msgs = []
        end = time.time() + timeout
        while rclpy.ok() and len(self.msgs) < self.fresh_msgs \
                and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if len(self.msgs) < self.fresh_msgs:
            if not quiet:
                self.get_logger().error(
                    'no hand seen in {:.0f} s'.format(timeout))
            return None
        msg = self.msgs[-1]
        if msg.header.frame_id != self.frame:
            self.get_logger().error(
                "hand pose arrived in '{}' but this node plans in '{}'. A "
                'point in a camera frame is not a place - set '
                "hand_detector's target_frame to '{}'.".format(
                    msg.header.frame_id, self.frame, self.frame))
            return None
        p = msg.pose.position
        return np.array([p.x, p.y, p.z])

    # --------------------------------------------------------------------- goal
    def _goal_for(self, hand, previous=None, prev_time=None):
        """Stop short of the hand, along the line from the base to it."""
        base = self._base_origin()
        v = hand - base
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            self.get_logger().error('hand is at the base origin; ignoring')
            return None
        goal = hand - (v / n) * self.standoff

        low = goal < self.ws_min
        high = goal > self.ws_max
        if np.any(low) or np.any(high):
            # Name the axis and the margin: "outside the box" on its own
            # sends you hunting, and the miss is often a centimetre.
            bits = []
            for i, ax in enumerate('xyz'):
                if low[i]:
                    bits.append('{} {:+.3f} is {:.0f} mm below ws_min {:+.3f}'
                                .format(ax, goal[i],
                                        (self.ws_min[i] - goal[i]) * 1000,
                                        self.ws_min[i]))
                elif high[i]:
                    bits.append('{} {:+.3f} is {:.0f} mm above ws_max {:+.3f}'
                                .format(ax, goal[i],
                                        (goal[i] - self.ws_max[i]) * 1000,
                                        self.ws_max[i]))
            self.get_logger().error(
                'goal [{:+.3f} {:+.3f} {:+.3f}] refused: {}. Widen it live '
                'with: ros2 param set /reach_to_hand ws_min '
                '"[x, y, z]"'.format(goal[0], goal[1], goal[2],
                                     '; '.join(bits)))
            return None

        # Has the TARGET jumped further than a hand could have moved in the
        # time available?  Only meaningful once there is a previous goal.
        if previous is not None and prev_time is not None:
            dt = max(0.0, time.time() - prev_time)
            allowed = max(self.jump_floor, self.max_speed * dt)
            jump = float(np.linalg.norm(goal - previous))
            if jump > allowed:
                self.get_logger().warn(
                    'target moved {:.0f} mm in {:.1f} s ({:.1f} m/s) - more '
                    'than max_target_speed_m_s {:.1f} allows ({:.0f} mm); '
                    'ignoring as a probable misdetection'
                    .format(jump * 1000, dt, jump / dt if dt > 0 else 0.0,
                            self.max_speed, allowed * 1000))
                return None

        now = self._tcp_now()
        if now is None:
            return None
        step = float(np.linalg.norm(goal - now))
        if step > self.max_step:
            self.get_logger().error(
                'the move would travel {:.0f} mm, more than max_step_m '
                '{:.0f} mm - refusing. The arm rests hanging down, so the '
                'first reach is long; raise it live with: ros2 param set '
                '/reach_to_hand max_step_m {:.2f}'
                .format(step * 1000, self.max_step * 1000, step + 0.05))
            return None
        self.get_logger().info('  travel {:.0f} mm from where the arm is now'
                               .format(step * 1000))

        self.get_logger().info(
            'hand [{:+.3f} {:+.3f} {:+.3f}] -> goal [{:+.3f} {:+.3f} {:+.3f}]'
            '  ({:.0f} mm move, {:.0f} mm standoff)'.format(
                hand[0], hand[1], hand[2], goal[0], goal[1], goal[2],
                step * 1000, self.standoff * 1000))
        return goal

    def run_once(self):
        hand = self._fresh_hand()
        if hand is None:
            return False
        goal = self._goal_for(hand)
        if goal is None:
            return False
        return self.move_to(goal, top_down_grasp_quat(self.jaw_yaw), 'reach')

    def run_tracking(self):
        """Follow the hand: re-plan whenever it has moved far enough.

        This is move-wait-look, not continuous servoing.  move_to blocks
        until the trajectory finishes, so the arm completes one motion before
        it looks again - the hand may have moved meanwhile, and the next
        iteration simply picks that up.  Smooth continuous following would
        need moveit_servo streaming velocity commands, which is a different
        controller and a different set of safety questions.

        retarget_m is what stops it thrashing: a hand held still still jitters
        by a centimetre or two in the depth image, and re-planning for that
        would keep the arm permanently in motion for no gain.
        """
        last = None
        last_goal = None
        last_time = None
        misses = 0
        n = 0
        self.get_logger().info(
            'tracking: re-planning whenever the hand moves more than '
            '{:.0f} mm. Ctrl-C to stop, or: ros2 param set /reach_to_hand '
            'dry_run true'.format(self.retarget * 1000))

        while rclpy.ok():
            hand = self._fresh_hand(timeout=self.loop_timeout, quiet=True)
            if hand is None:
                misses += 1
                if misses in (1, 10, 50):
                    self.get_logger().info(
                        'no hand in view - holding position (miss {})'
                        .format(misses))
                self._sleep(0.2)
                continue
            if misses:
                self.get_logger().info('hand back in view')
                misses = 0

            if last is not None:
                moved = float(np.linalg.norm(hand - last))
                if moved < self.retarget:
                    self._sleep(self.settle)
                    continue
                self.get_logger().info(
                    'hand moved {:.0f} mm - re-planning'.format(moved * 1000))

            goal = self._goal_for(hand, previous=last_goal,
                                  prev_time=last_time)
            if goal is None:
                self._sleep(0.3)
                continue

            n += 1
            if self.move_to(goal, top_down_grasp_quat(self.jaw_yaw),
                            'reach {}'.format(n)):
                last = hand
                last_goal = goal
                last_time = time.time()
            if self.dry_run:
                self.get_logger().warn(
                    'dry run: stopping after one goal rather than replanning '
                    'the same one forever. Run with dry_run:=false to follow '
                    'the hand for real.')
                return True
            self._sleep(self.settle)
        return True


def main():
    rclpy.init()
    node = ReachToHand()
    try:
        if not node.wait_for_servers():
            node.get_logger().error('MoveIt is not up; is move_group running?')
            return
        if node.do_loop:
            node.run_tracking()
        else:
            node.run_once()
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
