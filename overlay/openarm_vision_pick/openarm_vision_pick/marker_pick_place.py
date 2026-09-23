# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Pick the object carrying one tag and set it down centred on another.

  tag on the object  (id 3, 60 mm)  ->  where to grasp
  tag on the bench   (id 0, 80 mm)  ->  where to put it

Everything that makes this different from a hard-coded pick comes from the
two tags:

  * the GRASP POSE is the object tag's, corrected by the depth measurement
    around it - the jaws close across the object's short side, at half its
    height, not at the surface the camera can see;
  * the PLACE POSE is the destination tag's centre.  The object is set down
    so its BOTTOM rests on that surface, which means putting the grasp point
    - the object's middle - half an object-height above the tag's plane;
  * the grasp WIDTH comes from the measurement, so the gripper closes to fit
    the object rather than to a number typed in a config file.

Sequence, with a Cartesian move wherever a straight line matters:

  observe -> open -> pre-grasp above -> straight down -> close -> straight up
          -> above the destination -> straight down -> open -> straight up

Safety, in the order it bites:

  * dry_run defaults to true: every step is planned and displayed, nothing
    moves;
  * both tags must be seen and settled before anything is planned;
  * a workspace box is checked BEFORE planning, for the grasp and the place
    alike, so a misread tag cannot send the arm across the room;
  * the object's short side must fit the gripper, or the run is refused;
  * a Cartesian segment that cannot be followed to `cartesian_min_fraction`
    aborts rather than being approximated.

  ros2 run openarm_vision_pick marker_pick_place --ros-args \\
    --params-file <config>/marker_pick.yaml

Needs move_group, the controllers, and marker_detector publishing.
"""

import math
import time

import numpy as np
import rclpy
import tf2_ros
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject,
                             Constraints, JointConstraint, MotionPlanRequest,
                             OrientationConstraint, PlanningScene,
                             PositionConstraint, RobotState)
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath,
                             GetPlanningScene, GetPositionIK)
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float32, Int32MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from openarm_vision_pick.pick_and_place import (
    grasp_orientation_candidates, top_down_grasp_quat)


class MarkerPickPlace(Node):

    def __init__(self):
        super().__init__('marker_pick_place')

        p = self.declare_parameter
        p('group', 'left_arm')
        p('tcp_link', 'openarm_left_hand_tcp')
        p('planning_frame', 'world')
        # The arm's own root, so a refused grasp can be reported as a
        # distance rather than as bare coordinates.
        p('arm_base_link', 'openarm_left_link0')

        p('poses_topic', '/chest_marker_detector/poses')
        p('ids_topic', '/chest_marker_detector/ids')
        p('grasp_pose_topic', '/chest_marker_detector/pose')
        p('grasp_width_topic', '/chest_marker_detector/grasp_width')
        p('object_dims_topic', '/chest_marker_detector/object_dims')

        p('pick_id', 3)
        p('place_id', 0)

        p('gripper_action', '/left_gripper_controller/follow_joint_trajectory')
        # One joint on hardware: finger_joint2 really is a mimic there.
        p('gripper_joints', ['openarm_left_finger_joint1'])
        # THE GRIPPER COMMAND IS A JOINT POSITION, NOT AN OPENING.
        #
        # From the URDF: finger_joint1 is prismatic with its origin at
        # y = -0.005 along axis (0,-1,0), and finger_joint2 mirrors it at
        # y = +0.005 along (0,+1,0), mimicking joint1.  So at joint value q
        # the gap between the fingers is
        #
        #     gap = finger_gap_closed_m + 2 * q
        #
        # and the travel limit of 0.044 puts the widest opening at 98 mm -
        # not the 40 mm an earlier config's `gripper_open: 0.040` was
        # mistaken for.  That number was a joint position all along.
        #
        # Getting this backwards is worse than getting it wrong: commanding
        # the measured width directly as a joint value asks a 62 mm object's
        # gripper to open to 134 mm, which releases instead of grips.
        p('finger_gap_closed_m', 0.010)
        p('finger_travel_max_m', 0.044)

        # The opening to rest at while approaching, as a SPAN.
        p('gripper_open_span_m', 0.090)
        # How much narrower than the measured object to close, so the jaws
        # actually load up rather than just touching.  Also a span.
        p('grip_squeeze_m', 0.006)
        p('gripper_time', 1.5)

        p('approach_height_m', 0.10)
        p('lift_height_m', 0.12)
        # Set the object down this far above the destination surface and let
        # it drop the last millimetre, rather than pressing it into the bench.
        p('place_clearance_m', 0.004)

        # Fallback when the detector reports no measurement: how tall to
        # assume the object is, for the place height.
        p('assumed_height_m', 0.040)
        p('assumed_width_m', 0.050)

        p('ws_min', [0.05, -0.55, 0.10])
        p('ws_max', [0.65, 0.65, 0.90])

        p('vel_scale', 0.08)
        p('acc_scale', 0.08)
        p('planning_time', 10.0)
        p('planning_attempts', 16)
        p('pos_tolerance', 0.010)
        p('ori_tolerance', 0.15)
        p('cartesian_step', 0.004)
        p('cartesian_min_fraction', 0.9)

        p('detection_timeout', 30.0)
        # After the grasp pose arrives, how long to wait for the measurement
        # that the detector publishes right behind it.  They are separate
        # topics, so they land in separate callbacks, and returning on the
        # pose alone raced the dims: one run measured 62 mm, the next assumed
        # 40 - eleven millimetres lower at the place, which at the edge of
        # reach was the difference between a plan and none.
        p('dims_grace_s', 1.5)

        # Where to go when the job is done, or abandoned: the zero pose the
        # hardware interface itself drives to on activation, so "home" is
        # the same place whichever way the arm got there.
        # Where to go when the job is done.  'initial' means the pose the
        # arm was actually in when this run began - recorded from TF at the
        # start - and is the default.  'home_joints' is the fixed joint
        # vector below, kept as a fallback.
        # Pick the object up and stop, holding it, with no destination.
        # For trying out the grasp itself: no place tag needed, the object
        # stays in the gripper at the lift height so you can see it took.
        # It is not released or returned - Ctrl-C when done.
        p('lift_only', False)
        # Stop after the straight-down approach with the jaws still OPEN around the object -
        # nothing is closed or lifted.  For checking the grasp position by eye.
        p('stop_before_close', False)
        # Stop even earlier: open, pre-grasp above the object, wrist refine (hand plumb over the
        # centre) - and hold there.  No descent.
        p('stop_after_align', False)
        # From the hanging home pose the hand sits at bench level right next to the box, and a
        # direct plan to the pre-grasp pose (with padded collision geometry) tends to fail.  Raise
        # the arm first through a joint-space goal - arm out to the side, elbow folded - which keeps
        # every link well above a box in front of the robot (checked with URDF FK, ready_via_search).
        p('raise_joints', [0.0, 1.2, 0.0, 1.8, 0.0, 0.0, 0.0])
        p('raise_if_tcp_below_m', 0.30)
        # At the pre-grasp pose, re-measure the object's tag with the WRIST camera (it looks
        # straight down at it from 10-15 cm, far more accurately than the chest camera from
        # half a metre), slide the hand horizontally onto that centre, and only then descend
        # vertically.  Needs wrist_camera:=true wrist_detector:=true.
        p('refine_with_wrist', True)
        p('refine_timeout_s', 4.0)
        p('refine_min_samples', 3)
        p('refine_max_shift_m', 0.06)   # a bigger disagreement with the chest camera is a misdetection
        p('refine_iterations', 4)       # measure -> slide -> measure again, until the residual < 3 mm
        p('sag_correct', True)          # command = target + (commanded - actual) observed on the real arm
        # The detectors' depth footprint merges whatever touches the object (the neighbouring box, the
        # sheet it sits on) and over-reads; with a known object the configured assumed_* size is used
        # and only the YAW is taken from the measurement.
        p('trust_measured_dims', False)
        # The grasp point: the tag centre, this far straight down.  The TCP (fingertips) ends here
        # with the hand's x axis along the tag's x axis.
        p('grasp_below_tag_m', 0.02)
        p('yaw_from_wrist_tag', True)   # refine also aligns the jaws to the wrist-seen tag axes
        p('refine_max_samples', 8)
        p('approach_vel_scale', 0.05)   # the straight-down descent, slower than vel_scale
        # Where to look from for the wrist refine: the hand is parked so the D405 sits this far from
        # the tag ALONG ITS OWN LINE OF SIGHT, with the tag in the middle of the image.  Farther =
        # a wider patch of bench in view (the 87 x 58 deg field covers 76 x 44 cm at 0.40 m, 47 x 28
        # at 0.25) at the price of a smaller tag (57 mm = 65 px at 0.40 m, plenty).  0 = refine
        # straight above the tag at the pre-grasp height, as before.
        p('view_distance_m', 0.40)
        # The camera on the hand (URDF, hand frame: z towards the fingertips): position and the
        # direction it looks along.  Used to place the viewing pose.
        p('wrist_cam_xyz_in_hand', [0.0605, 0.0, -0.0149])
        p('wrist_cam_axis_in_hand', [-0.6, 0.0, 0.8])
        p('tcp_z_in_hand', 0.0835)
        p('approach_check_iters', 8)    # servo iterations per stage (hover, then the last 3 cm)
        p('servo_gain', 0.7)            # fraction of the TF error applied to the command per iteration
        p('hover_clear_m', 0.01)        # fingertips this far above the tag face while converging xy
        p('wrist_image_topic', '/camera/camera/color/image_raw')
        # realign_only: skip the chest camera and the plan to the pre-grasp - the hand is already
        # over the object (e.g. held after `align`); just do the wrist-camera alignment from here.
        p('realign_only', False)
        p('return_to', 'initial')
        # The last stretch down into the initial pose is an unchecked
        # straight line from this height above it, like the descent into a
        # grasp: the resting spot is known to be fine (the arm just left
        # it), but the octomap says otherwise once the hand is gone - the
        # padding that masks the hand had been masking the bench beside it,
        # and with the hand away those voxels are filled in.
        p('initial_lift_m', 0.08)
        p('home_joints', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        p('return_home', True)
        # On an abort, open the gripper and go home rather than stop with
        # the arm stretched out over the bench.
        p('return_home_on_abort', True)

        # The link the picked object is attached to, and the links allowed to
        # touch it while it is held.  Attaching is what tells the planner
        # the object now moves with the hand - without it the transfer would
        # swing the object through the box it was lifted from.
        p('attach_link', 'openarm_left_hand_tcp')
        p('touch_links', ['openarm_left_hand', 'openarm_left_hand_tcp',
                          'openarm_left_left_finger',
                          'openarm_left_right_finger'])
        # Fallback footprint when the detector reported no measurement.
        p('assumed_length_m', 0.100)

        # Before a collision-checked plan that follows a contact move, give
        # the octomap time to clear the voxels the arm itself left behind:
        # the updater only frees a voxel when a camera ray passes through
        # it, and right after the hand pulls back those rays have not yet
        # arrived.  If the start state is still in collision with the map
        # after that, back straight up by clear_step_m (unchecked, like any
        # contact move) and try again, up to clear_tries times.
        # Did the grasp actually take?  The wrist camera looks straight at
        # the gripper, so after the lift the object's own tag should still
        # be sitting a few centimetres from the TCP.  If it is still down on
        # the bench, or nowhere to be seen, the hand is empty - and carrying
        # an empty hand across the bench and opening it over the destination
        # is a waste at best.  Fail here instead, with the object undamaged
        # where it was.
        p('verify_grasp', True)
        p('verify_poses_topic', '/wrist_marker_detector/poses')
        p('verify_ids_topic', '/wrist_marker_detector/ids')
        # How close the tag must be to the TCP to count as "in the hand".
        # The tag rides on top of the object and the grasp is at its
        # middle, so half the object's height plus a margin.
        p('held_radius_m', 0.08)
        p('verify_timeout_s', 3.0)
        # If the wrist detector is not running the check cannot be made.
        # True aborts anyway; False warns and carries on unverified.
        p('verify_required', False)

        p('octomap_settle_s', 1.0)
        p('clear_step_m', 0.08)
        p('clear_tries', 2)
        p('settle_s', 0.5)
        # Discovery of an action can take far longer than discovery
        # of its node, especially on a busy graph.  30 s was not
        # enough here.
        # A parallel gripper closing on a box is the same grasp turned 180
        # degrees, so both are always worth trying.  For a nearly square
        # footprint the quarter turns are legitimate too - the jaws still
        # close across a side, just the other one.
        p('try_yaw_offsets_deg', [0.0, 180.0, 90.0, 270.0])
        # Below this short/long ratio the object has a clear long axis and
        # the quarter turns would grip across the wrong side, so they are
        # dropped.
        p('square_ratio', 0.8)
        p('ik_timeout_s', 0.05)
        # How far the hand may lean from vertical, tried in this order after
        # straight down fails.  Zero disables leaning.  See tilted_grasp_quat
        # for why insisting on plumb was refusing reachable grasps.
        p('tilt_deg', [0.0, 15.0, 30.0])

        # Jaw angles to try at the PLACE, relative to the grasp's, in order.
        # The place is free to turn: the object rides in a fixed grip, but
        # the hand itself can be turned about vertical by the wrist on the
        # way over, and the destination asks only for a position.  No turn
        # is tried first, so the object lands the way it was picked up when
        # that works, and quarter turns before eighths.
        p('place_yaw_offsets_deg', [0.0, 180.0, 90.0, 270.0,
                                    45.0, 135.0, 225.0, 315.0])

        p('server_timeout_s', 90.0)

        p('dry_run', True)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.group = g('group')
        self.tcp = g('tcp_link')
        self.frame = g('planning_frame')
        self.arm_base = g('arm_base_link')
        self.pick_id = int(g('pick_id'))
        self.place_id = int(g('place_id'))
        self.gripper_joints = list(g('gripper_joints'))
        self.gap0 = g('finger_gap_closed_m')
        self.travel = g('finger_travel_max_m')
        self.open_span = g('gripper_open_span_m')
        self.squeeze = g('grip_squeeze_m')
        self.g_time = g('gripper_time')
        self.approach = g('approach_height_m')
        self.lift = g('lift_height_m')
        self.place_clear = g('place_clearance_m')
        self.assumed_h = g('assumed_height_m')
        self.assumed_w = g('assumed_width_m')
        self.ws_min = np.array(g('ws_min'), dtype=float)
        self.ws_max = np.array(g('ws_max'), dtype=float)
        self.vel, self.acc = g('vel_scale'), g('acc_scale')
        self.plan_time = g('planning_time')
        self.plan_tries = g('planning_attempts')
        self.pos_tol, self.ori_tol = g('pos_tolerance'), g('ori_tolerance')
        self.cart_step = g('cartesian_step')
        self.cart_min = g('cartesian_min_fraction')
        self.det_timeout = g('detection_timeout')
        self.dims_grace = g('dims_grace_s')
        self.lift_only = g('lift_only')
        self.stop_before_close = g('stop_before_close')
        self.stop_after_align = g('stop_after_align')
        self.raise_joints = [float(v) for v in g('raise_joints')]
        self.raise_below = g('raise_if_tcp_below_m')
        self.refine = g('refine_with_wrist')
        self.refine_timeout = g('refine_timeout_s')
        self.refine_min = int(g('refine_min_samples'))
        self.refine_max = g('refine_max_shift_m')
        self.refine_iters = int(g('refine_iterations'))
        self.sag_correct = g('sag_correct')
        self.trust_dims = g('trust_measured_dims')
        self.below_tag = g('grasp_below_tag_m')
        self.yaw_from_tag = g('yaw_from_wrist_tag')
        self.refine_max_n = int(g('refine_max_samples'))
        self.approach_vel = g('approach_vel_scale')
        self.view_dist = float(g('view_distance_m'))
        self.cam_xyz = np.array([float(v) for v in g('wrist_cam_xyz_in_hand')])
        self.cam_axis = np.array([float(v) for v in g('wrist_cam_axis_in_hand')])
        self.cam_axis /= max(1e-9, np.linalg.norm(self.cam_axis))
        self.tcp_z_hand = float(g('tcp_z_in_hand'))
        self.approach_iters = int(g('approach_check_iters'))
        self.servo_gain = float(g('servo_gain'))
        self.hover_clear = float(g('hover_clear_m'))
        self._wrist_img = None
        try:
            from sensor_msgs.msg import Image
            from rclpy.qos import qos_profile_sensor_data
            self.create_subscription(Image, g('wrist_image_topic'), self._on_wrist_img, qos_profile_sensor_data)
        except Exception as e:  # diagnostics only
            self.get_logger().warn('no wrist image subscription: {}'.format(e))
        self.realign_only = g('realign_only')
        self.return_to = str(g('return_to'))
        self.initial_lift = g('initial_lift_m')
        self._initial_tcp = None
        self.home_q = [float(v) for v in g('home_joints')]
        self.go_home = g('return_home')
        self.home_on_abort = g('return_home_on_abort')
        self.attach_link = g('attach_link')
        self.touch_links = list(g('touch_links'))
        self.assumed_l = g('assumed_length_m')
        self.verify = g('verify_grasp')
        self.held_r = g('held_radius_m')
        self.verify_timeout = g('verify_timeout_s')
        self.verify_required = g('verify_required')
        self.wrist_ids = []
        self.wrist_poses = None
        self.wrist_stamp = 0.0
        self.settle_map = g('octomap_settle_s')
        self.clear_step = g('clear_step_m')
        self.clear_tries = int(g('clear_tries'))
        self.joint_names = ['openarm_left_joint{}'.format(i)
                            for i in range(1, 8)] if self.group == 'left_arm' \
            else ['openarm_right_joint{}'.format(i) for i in range(1, 8)]
        self._attached = False
        self._in_scene = False
        self._ctx = None
        self.settle = g('settle_s')
        self.yaw_offsets = [float(v) for v in g('try_yaw_offsets_deg')]
        self.square_ratio = g('square_ratio')
        self.ik_timeout = g('ik_timeout_s')
        self.tilts = [math.radians(v) for v in g('tilt_deg') if v > 0.0]
        self.place_yaw_offsets = [float(v) for v in g('place_yaw_offsets_deg')]
        self.server_timeout = g('server_timeout_s')
        self.dry_run = g('dry_run')

        # Where the arm WOULD be, in a dry run.
        #
        # move_to_pose with plan_only leaves the robot where it was, so the
        # next Cartesian segment - which starts from the live state - was
        # being asked for a straight line from the rest pose to the grasp,
        # and reported 0% reachable.  That made dry_run useless as a check
        # of the sequence.  So each planned trajectory's final joints are
        # carried forward as the start of the next request.  Live runs never
        # touch this: there the robot really is where the last step left it.
        self._dry_state = None

        self.grasp = None            # PoseStamped for the object
        self.width = None            # measured grasp width
        self.dims = None             # long, short, height
        self.ids = []
        self.poses = None

        self.create_subscription(PoseStamped, g('grasp_pose_topic'),
                                 self._on_grasp, 10)
        self.create_subscription(Float32, g('grasp_width_topic'),
                                 self._on_width, 10)
        self.create_subscription(Vector3, g('object_dims_topic'),
                                 self._on_dims, 10)
        self.create_subscription(Int32MultiArray, g('ids_topic'),
                                 self._on_ids, 10)
        self.create_subscription(PoseArray, g('poses_topic'),
                                 self._on_poses, 10)
        self.create_subscription(Int32MultiArray, g('verify_ids_topic'),
                                 self._on_wrist_ids, 10)
        self.create_subscription(PoseArray, g('verify_poses_topic'),
                                 self._on_wrist_poses, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.exec_client = ActionClient(self, ExecuteTrajectory,
                                        '/execute_trajectory')
        self.grip_client = ActionClient(self, FollowJointTrajectory,
                                        g('gripper_action'))
        self.cart_client = self.create_client(GetCartesianPath,
                                              '/compute_cartesian_path')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.scene_client = self.create_client(ApplyPlanningScene,
                                               '/apply_planning_scene')
        self.get_scene_client = self.create_client(GetPlanningScene,
                                                   '/get_planning_scene')

        self.add_on_set_parameters_callback(self._on_param_set)

        self.get_logger().info(
            'pick id {} -> place centred on id {}, in {}'.format(
                self.pick_id, self.place_id, self.frame))
        if self.dry_run:
            self.get_logger().warn(
                'DRY RUN - every step is planned and displayed, nothing '
                'moves and the gripper does not close. Set dry_run:=false '
                'once the plan looks right in RViz.')

    # ------------------------------------------------------------ live tuning
    _LIVE = {
        'gripper_open_span_m': 'open_span', 'grip_squeeze_m': 'squeeze',
        'finger_gap_closed_m': 'gap0', 'finger_travel_max_m': 'travel',
        'approach_height_m': 'approach', 'lift_height_m': 'lift',
        'place_clearance_m': 'place_clear', 'assumed_height_m': 'assumed_h',
        'vel_scale': 'vel', 'acc_scale': 'acc', 'dry_run': 'dry_run',
        'detection_timeout': 'det_timeout', 'settle_s': 'settle',
        'pick_id': 'pick_id', 'place_id': 'place_id',
        'return_home': 'go_home', 'return_home_on_abort': 'home_on_abort',
        'lift_only': 'lift_only',
        'verify_grasp': 'verify', 'held_radius_m': 'held_r',
        'verify_required': 'verify_required',
    }

    def _on_param_set(self, params):
        for prm in params:
            if prm.name in ('ws_min', 'ws_max'):
                v = np.array(list(prm.value), dtype=float)
                if v.shape != (3,):
                    return SetParametersResult(successful=False,
                                               reason='needs three numbers')
                setattr(self, prm.name, v)
                self.get_logger().info('{} = {}'.format(prm.name, list(v)))
                continue
            attr = self._LIVE.get(prm.name)
            if attr is None:
                continue
            value = int(prm.value) if attr in ('pick_id', 'place_id') \
                else prm.value
            setattr(self, attr, value)
            self.get_logger().info('{} = {}'.format(prm.name, value))
            if prm.name == 'dry_run' and prm.value:
                self.get_logger().warn('dry_run is on again - nothing '
                                       'further will be executed')
        return SetParametersResult(successful=True)

    # ----------------------------------------------------------------- inputs
    def _on_grasp(self, msg):
        self.grasp = msg
        self._grasp_lowered = False

    def _on_width(self, msg):
        self.width = float(msg.data)

    def _on_dims(self, msg):
        self.dims = np.array([msg.x, msg.y, msg.z])

    def _on_ids(self, msg):
        self.ids = list(msg.data)

    def _on_poses(self, msg):
        self.poses = msg

    def _on_wrist_ids(self, msg):
        self.wrist_ids = list(msg.data)

    def _on_wrist_poses(self, msg):
        self.wrist_poses = msg
        self.wrist_stamp = time.time()

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

    def _place_pose(self):
        """The destination tag's centre, from the all-markers stream."""
        if self.poses is None or not self.ids:
            return None
        if self.place_id not in self.ids:
            return None
        k = self.ids.index(self.place_id)
        if k >= len(self.poses.poses):
            return None
        if self.poses.header.frame_id != self.frame:
            self.get_logger().error(
                "markers arrive in '{}' but this node plans in '{}'".format(
                    self.poses.header.frame_id, self.frame))
            return None
        q = self.poses.poses[k].position
        return np.array([q.x, q.y, q.z])

    def move_to_joints(self, q, label):
        """A joint-space goal, collision-checked like any other plan."""
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
        return self._send_move(req, label, joints=list(q))

    def remember_initial(self):
        """Record where the arm is now: this is where it goes back to."""
        self._initial_tcp = self._tcp_pose_now()
        if self._initial_tcp is None:
            self.get_logger().warn(
                'could not read the initial TCP pose from TF; the run will '
                'end at home_joints instead')
        else:
            p = self._initial_tcp.position
            self.get_logger().info(
                'initial position recorded: [{:+.3f} {:+.3f} {:+.3f}]'.format(
                    p.x, p.y, p.z))

    def return_to_initial(self, label='initial'):
        """Back to where the run started.

        A checked plan to a point above the initial pose, then an unchecked
        straight descent into it - the same shape as the approach to a
        grasp, and for the same reason: the last few centimetres end
        somewhere the map calls occupied but the arm knows is clear.
        """
        if self.return_to != 'initial' or self._initial_tcp is None:
            self._ensure_clear_start(label)
            return self.move_to_joints(self.home_q, label)
        self._ensure_clear_start(label)
        tgt = self._initial_tcp
        above = Pose()
        above.position.x = tgt.position.x
        above.position.y = tgt.position.y
        above.position.z = tgt.position.z + self.initial_lift
        above.orientation = tgt.orientation
        if not self.move_to_pose(above, label + ': above'):
            self.get_logger().warn(
                '{}: cannot plan to above the initial position; trying the '
                'joint home instead'.format(label))
            return self.move_to_joints(self.home_q, label + ': joints')
        if not self.cartesian_to(tgt, label + ': settle', avoid=False,
                                 min_fraction=0.5):
            self.get_logger().warn(
                '{}: stopped short of the initial position'.format(label))
            return True          # above it is close enough to count
        self.get_logger().info('{}: back at the initial position'
                               .format(label))
        return True

    # kept for anything still calling the old name
    def go_home_now(self, label='initial'):
        return self.return_to_initial(label)

    def verify_grasp(self, label='verify'):
        """Is the object in the hand?  Ask the wrist camera.

        The object's tag, as the wrist detector sees it, is compared with
        the TCP in the planning frame.  Held: the tag is within held_radius
        of the TCP and stays there.  Not held: it is further away - still on
        the bench - or not seen at all.

        Not a matter of geometry alone: fresh detections are required, so a
        stale pose from before the lift cannot pass for a current one.
        """
        if not self.verify or self.dry_run:
            return True
        self.wrist_poses = None
        self.wrist_ids = []
        t_lift = time.time()
        end = t_lift + self.verify_timeout
        seen_near = seen_far = 0
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.wrist_poses is None or self.wrist_stamp < t_lift:
                continue
            if self.pick_id not in self.wrist_ids:
                continue
            k = self.wrist_ids.index(self.pick_id)
            if k >= len(self.wrist_poses.poses):
                continue
            if self.wrist_poses.header.frame_id != self.frame:
                self.get_logger().warn(
                    '{}: wrist poses arrive in {!r}, need {!r} - set the '
                    "wrist detector's target_frame".format(
                        label, self.wrist_poses.header.frame_id, self.frame))
                return not self.verify_required
            tcp = self._tcp_pose_now()
            if tcp is None:
                continue
            p = self.wrist_poses.poses[k].position
            d = math.sqrt((p.x - tcp.position.x) ** 2 +
                          (p.y - tcp.position.y) ** 2 +
                          (p.z - tcp.position.z) ** 2)
            if d <= self.held_r:
                seen_near += 1
            else:
                seen_far += 1
            # Two agreeing sightings is enough either way.
            if seen_near >= 2:
                self.get_logger().info(
                    '{}: object in hand - tag {:.0f} mm from the TCP'.format(
                        label, d * 1000))
                return True
            if seen_far >= 2:
                self.get_logger().error(
                    '{}: GRASP FAILED - tag {} is {:.0f} mm from the TCP, '
                    'still on the bench'.format(label, self.pick_id,
                                                d * 1000))
                return False
            self.wrist_poses = None

        if seen_near or seen_far:
            # One sighting only; be conservative in the direction of the
            # majority.
            ok = seen_near > seen_far
            (self.get_logger().info if ok else self.get_logger().error)(
                '{}: {} on one sighting'.format(
                    label, 'held' if ok else 'GRASP FAILED'))
            return ok
        self.get_logger().warn(
            '{}: the wrist detector reported nothing in {:.0f} s - is it '
            'running, and is wrist_camera:=true?  {}'.format(
                label, self.verify_timeout,
                'aborting (verify_required)' if self.verify_required
                else 'carrying on unverified'))
        return not self.verify_required

    def refine_with_wrist(self, ctx):
        """Wrist-camera measurement at the pre-grasp pose -> centred grasp point, iterated."""
        if not self.refine or self.dry_run:
            return True
        for i in range(max(1, self.refine_iters)):
            ctx['refine_nodata'] = False
            moved = self._refine_once(ctx, i + 1)
            if moved is None:
                return False          # the sideways move failed
            if not moved:
                if ctx.get('refine_nodata') and ctx.get('view_len', 0.0) > 1e-6:
                    # nothing seen from the viewing pose: try from straight above the tag, closer
                    self.get_logger().warn(
                        'refine: no tag from the viewing pose - moving straight above the tag at the '
                        'pre-grasp height and looking again')
                    ctx['view_len'] = 0.0
                    ctx.pop('cmd_pre', None)
                    if not self.cartesian_to(self._pose(ctx['pre'], ctx['quat_g']), 'refine: to pre-grasp', avoid=True):
                        if not self.move_to_pose(self._pose(ctx['pre'], ctx['quat_g']), 'refine: to pre-grasp'):
                            return False
                    self._sleep(self.settle)
                    continue
                return True           # residual under 3 mm (or nothing usable to refine with)
        return True

    def _refine_once(self, ctx, it):
        """One measure-and-slide. Returns True if the hand moved, False if done/no data, None on failure."""
        self._sleep(self.settle)        # let the arm stop swinging before trusting the camera
        t0 = time.time()
        self.wrist_poses = None
        samples = []
        quats = []
        while rclpy.ok() and time.time() < t0 + self.refine_timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.wrist_poses is None or self.wrist_stamp < t0:
                continue
            if self.pick_id not in self.wrist_ids:
                self.wrist_poses = None
                continue
            k = self.wrist_ids.index(self.pick_id)
            if k >= len(self.wrist_poses.poses):
                self.wrist_poses = None
                continue
            if self.wrist_poses.header.frame_id != self.frame:
                self.get_logger().warn(
                    "refine: wrist poses arrive in {!r}, need {!r} - keeping the chest estimate".format(
                        self.wrist_poses.header.frame_id, self.frame))
                return True
            q = self.wrist_poses.poses[k].position
            o = self.wrist_poses.poses[k].orientation
            samples.append([q.x, q.y, q.z])
            quats.append([o.x, o.y, o.z, o.w])
            self.wrist_poses = None
            if len(samples) >= max(self.refine_min, self.refine_max_n):
                break
        if len(samples) < self.refine_min:
            self.get_logger().warn(
                'refine #{}: the wrist camera gave {} detection(s) of tag {} in {:.0f} s - keeping the '
                'current estimate'.format(it, len(samples), self.pick_id, self.refine_timeout))
            self._save_wrist_img('refine_fail_{}'.format(it))
            ctx['refine_nodata'] = True
            return False
        med = np.median(np.array(samples), axis=0)
        act = self._actual_tcp()
        if act is None:
            self.get_logger().warn('refine #{}: no TF for the TCP - cannot verify the alignment'.format(it))
            return False
        spread = np.ptp(np.array(samples), axis=0)
        if float(np.max(spread)) > 0.012:
            self.get_logger().warn(
                'refine #{}: the {} wrist samples disagree by [{:.0f} {:.0f} {:.0f}] mm - the arm or the '
                'object is still moving; measuring again'.format(it, len(samples), *(spread * 1000)))
            return True
        # the grasp point: the tag centre, below_tag straight down
        new = np.array([med[0], med[1], med[2] - self.below_tag])
        # the jaws: hand x along the tag's x axis (the jaws close across it), on the 180-degree
        # branch the IK already chose
        if self.yaw_from_tag and 'jaw_yaw' in ctx and quats:
            Q = np.array(quats)
            xs = np.stack([1.0 - 2.0 * (Q[:, 1] ** 2 + Q[:, 2] ** 2), 2.0 * (Q[:, 0] * Q[:, 1] + Q[:, 2] * Q[:, 3])], axis=1)
            xm = xs.mean(axis=0)
            tag_yaw = math.atan2(xm[1], xm[0])
            want = tag_yaw + math.pi / 2.0
            cur = ctx['jaw_yaw']
            if math.cos(want - cur) < 0.0:
                want += math.pi
            dyaw = math.atan2(math.sin(want - cur), math.cos(want - cur))
            if abs(dyaw) > math.radians(0.5):
                ctx['jaw_yaw'] = want
                ctx['quat_g'] = top_down_grasp_quat(want)
                self.get_logger().info(
                    'refine #{}: tag x axis at {:+.1f} deg -> jaws turned {:+.1f} deg to {:.1f} deg'.format(
                        it, math.degrees(tag_yaw), math.degrees(dyaw), math.degrees(want) % 360.0))
        if abs(new[2] - ctx['grasp'][2]) > 0.001:
            self.get_logger().info(
                'refine #{}: grasp height {:.3f} -> {:.3f} from the wrist tag (face at z={:.3f})'.format(
                    it, ctx['grasp'][2], new[2], med[2]))
        if float(np.linalg.norm(new[:2] - ctx['grasp'][:2])) > self.refine_max:
            self.get_logger().warn(
                'refine #{}: wrist tag at [{:+.3f} {:+.3f}] is {:.0f} mm from the chest estimate - more than '
                'refine_max_shift_m, keeping the chest estimate'.format(
                    it, new[0], new[1], float(np.linalg.norm(new[:2] - ctx['grasp'][:2])) * 1000))
            return False
        ctx['grasp'] = new
        ctx['pre'] = new + [0, 0, self.approach]
        # residual = where the tag is minus where the camera's AIM POINT actually is (TF: the hand,
        # less the viewing offset), not where it was told to go
        off = self._view_off(ctx)
        shift = (new[:2] + off[:2]) - act[:2]
        d = float(np.linalg.norm(shift))
        cmd = ctx.get('cmd_pre', ctx['pre'] + off)
        sag = cmd - act
        self.get_logger().info(
            'refine #{}: tag centre [{:+.3f} {:+.3f}], hand actually at [{:+.3f} {:+.3f} {:+.3f}] -> {:.0f} mm off; '
            'arm sits [{:+.0f} {:+.0f} {:+.0f}] mm from its command'.format(
                it, new[0], new[1], act[0], act[1], act[2], d * 1000, *(sag * 1000)))
        ctx['sag'] = sag
        if d < 0.003:
            self.get_logger().info('refine #{}: aligned - the camera aim point is {:.0f} mm from the tag centre'.format(it, d * 1000))
            return False
        # integral correction: move the COMMAND by the residual (the arm will again stop short of the
        # command by roughly the same sag, which is exactly what brings the actual TCP onto the tag)
        cmd = cmd + np.array([shift[0], shift[1], 0.0])
        if self.sag_correct:
            cmd[2] = ctx['pre'][2] + off[2] + sag[2] if abs(sag[2]) < 0.10 else cmd[2]
        ctx['cmd_pre'] = cmd
        ctx['sag'] = sag
        # unchecked: a few centimetres sideways at the pre-grasp height, well above the box.  The
        # checked version was refused by stale octomap voxels of the arm's own fingertip.
        ok = self.cartesian_to(self._pose(cmd, ctx['quat_g']), 'refine #{}'.format(it), avoid=False)
        if not ok:
            return None
        self._sleep(self.settle)
        return True

    def maybe_raise(self):
        """Joint-space move to the raised side pose when the hand starts low (hanging home)."""
        if not self.raise_joints or len(self.raise_joints) != len(self.joint_names):
            return True
        tcp = self._tcp_pose_now()
        if tcp is not None and tcp.position.z >= self.raise_below:
            self.get_logger().info('raise: hand already at z={:.2f} m, skipping'.format(tcp.position.z))
            return True
        self.get_logger().info('raise: hand low (z={}), lifting the arm to the side first'.format(
            '?' if tcp is None else '{:.2f}'.format(tcp.position.z)))
        return self.move_to_joints(self.raise_joints, 'raise')

    @staticmethod
    def _hand_x(quat):
        """Horizontal unit vector of the hand's x axis for a quaternion (x, y, z, w)."""
        x, y, z, w = quat
        v = np.array([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 0.0])
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.zeros(3)

    def _view_off(self, ctx):
        """World-frame offset of the viewing pose from the pre-grasp (rotates with the jaw yaw).

        Camera at distance D from the tag along its line of sight: in the hand frame the TCP is then
        at (cam - D*axis) + (tcp - cam) relative to the tag; hand z points DOWN at the tag, so the
        hand-frame z becomes -world z, and hand x is horizontal (top-down grasp)."""
        D = ctx.get('view_len', 0.0)
        if D <= 1e-6:
            return np.zeros(3)
        tag = self.cam_xyz + D * self.cam_axis                    # the tag, in the hand frame
        rel = np.array([0.0, 0.0, self.tcp_z_hand]) - tag          # tcp relative to the tag, hand frame
        dx, dz_hand = float(rel[0]), float(rel[2])
        # the hand's x axis is horizontal; the hand's +z points down: height above the tag = -dz_hand
        height = -dz_hand
        return dx * self._hand_x(ctx['quat_g']) + np.array([0.0, 0.0, height - self.approach])

    def goto_pregrasp(self, ctx):
        """Checked plan to the viewing pose (tag in the middle of the wrist image); if that has no
        plan, to the plain pre-grasp straight above the tag."""
        self._ensure_clear_start('pre-grasp')
        if ctx.get('view_len', 0.0) > 1e-6:
            off = self._view_off(ctx)
            self.get_logger().info(
                'pre-grasp: viewing pose with the wrist camera {:.0f} mm from the tag: TCP {:.0f} mm above the '
                'tag face, {:.0f} mm to the hand +x'.format(
                    ctx['view_len'] * 1000, (off[2] + self.approach + self.below_tag) * 1000,
                    float(np.linalg.norm(off[:2])) * 1000))
            if self.move_to_pose(self._pose(ctx['pre'] + off, ctx['quat_g']), 'pre-grasp (viewing pose)'):
                return True
            self.get_logger().warn(
                'pre-grasp: no plan to the viewing pose; trying straight above the tag at the pre-grasp height')
            ctx['view_len'] = 0.0
        return self.move_to_pose(self._pose(ctx['pre'], ctx['quat_g']), 'pre-grasp')

    def centre_over_tag(self, ctx):
        """From the viewing pose: slide the TCP over the tag centre at the pre-grasp height and verify
        it on TF (up to approach_check_iters corrections)."""
        if ctx.get('view_len', 0.0) <= 1e-6 or self.dry_run:
            return True
        target = np.array(ctx['pre'], dtype=float)
        sag = np.array(ctx.get('sag', [0.0, 0.0, 0.0]), dtype=float) if self.sag_correct else np.zeros(3)
        if np.linalg.norm(sag) > 0.10:
            sag = np.zeros(3)
        cmd = target + sag
        self.get_logger().info('centre: down and across onto the tag centre at the pre-grasp height')
        if not self.cartesian_to(self._pose(cmd, ctx['quat_g']), 'centre', avoid=True):
            self.get_logger().warn('centre: the checked slide was refused - sliding unchecked')
            if not self.cartesian_to(self._pose(cmd, ctx['quat_g']), 'centre', avoid=False):
                return False
        self._sleep(self.settle)
        for i in range(max(0, self.approach_iters)):
            act = self._actual_tcp()
            if act is None:
                return True
            err = target - act
            self.get_logger().info(
                'centre check #{}: hand at [{:+.3f} {:+.3f} {:+.3f}], tag centre [{:+.3f} {:+.3f}], off by '
                '[{:+.0f} {:+.0f} {:+.0f}] mm'.format(i + 1, *act, target[0], target[1], *(err * 1000)))
            ctx['sag'] = cmd - act
            if float(np.linalg.norm(err[:2])) < 0.003 and abs(err[2]) < 0.01:
                return True
            cmd = cmd + err
            if not self.cartesian_to(self._pose(cmd, ctx['quat_g']), 'centre fix #{}'.format(i + 1), avoid=False):
                return False
            self._sleep(self.settle)
        return True

    def _on_wrist_img(self, msg):
        self._wrist_img = msg

    def _save_wrist_img(self, name):
        """Write the latest wrist colour frame to ~/<name>.png (diagnostics)."""
        m = self._wrist_img
        if m is None:
            return
        try:
            import os
            import cv2
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
            if m.encoding == 'rgb8':
                img = img[:, :, ::-1]
            path = os.path.expanduser('~/{}.png'.format(name))
            cv2.imwrite(path, img)
            self.get_logger().info('wrist frame saved to {}'.format(path))
        except Exception as e:
            self.get_logger().warn('could not save the wrist frame: {}'.format(e))

    def _servo_to(self, target, quat, cmd, label, tol_xy, tol_z, guard_z=None):
        """Move to cmd, read the actual TCP on TF, nudge the command by servo_gain x error, repeat.

        Returns (converged, cmd, act).  guard_z: if the hand is below this height with more than
        12 mm of xy error, it is lifted back above it before the next correction (the fingers may
        be pressing on the box)."""
        target = np.array(target, dtype=float)
        cmd = np.array(cmd, dtype=float)
        vel, acc = self.vel, self.acc
        self.vel = self.acc = float(self.approach_vel)
        try:
            act = None
            for i in range(max(1, self.approach_iters)):
                if not self.cartesian_to(self._pose(cmd, quat), '{} #{}'.format(label, i + 1), avoid=False):
                    return False, cmd, act
                self._sleep(self.settle)
                act = self._actual_tcp()
                if act is None or self.dry_run:
                    return True, cmd, act
                err = target - act
                exy = float(np.linalg.norm(err[:2]))
                self.get_logger().info(
                    '{} check #{}: hand at [{:+.3f} {:+.3f} {:+.3f}], target [{:+.3f} {:+.3f} {:+.3f}], '
                    'off by [{:+.0f} {:+.0f} {:+.0f}] mm'.format(label, i + 1, *act, *target, *(err * 1000)))
                if exy < tol_xy and abs(err[2]) < tol_z:
                    return True, cmd, act
                if guard_z is not None and act[2] < guard_z and exy > 0.012:
                    self.get_logger().warn(
                        '{}: {:.0f} mm off sideways with the fingers below the tag face - lifting them '
                        'clear before correcting'.format(label, exy * 1000))
                    cmd[2] += (guard_z - act[2]) + 0.01
                    continue
                cmd = cmd + self.servo_gain * err
            self.get_logger().error(
                '{}: not converged after {} corrections (still [{:+.0f} {:+.0f} {:+.0f}] mm off)'.format(
                    label, self.approach_iters, *(err * 1000)))
            return False, cmd, act
        finally:
            self.vel, self.acc = vel, acc

    def approach_checked(self, ctx):
        """Vertical descent as a servo on the actual TCP: hover with the fingertips just above the
        tag face until the xy error is gone, then the last few centimetres, re-converged.  A failed
        convergence lifts the hand clear and fails the step."""
        target = np.array(ctx['grasp'], dtype=float)
        quat = ctx['quat_g']
        face_z = target[2] + self.below_tag
        hover = target + [0.0, 0.0, self.below_tag + self.hover_clear]
        sag = np.array(ctx.get('sag', [0.0, 0.0, 0.0]), dtype=float) if self.sag_correct else np.zeros(3)
        if np.linalg.norm(sag) > 0.10:
            sag = np.zeros(3)
        self.get_logger().info(
            'approach: hover with the fingertips {:.0f} mm above the tag face, then {:.0f} mm below it, '
            'at {:.0f}% speed'.format(self.hover_clear * 1000, self.below_tag * 1000, self.approach_vel * 100))
        ok, cmd, act = self._servo_to(hover, quat, hover + sag, 'hover', tol_xy=0.003, tol_z=0.006)
        if not ok:
            self._lift_clear(cmd, quat, hover[2] + 0.05)
            return False
        if self.dry_run or act is None:
            return True
        # the last stretch: keep the command offset (sag) the hover converged to
        ok, cmd, act = self._servo_to(target, quat, target + (cmd - act), 'approach', tol_xy=0.004,
                                      tol_z=0.005, guard_z=face_z)
        if not ok:
            self._lift_clear(cmd, quat, hover[2] + 0.05)
            return False
        ctx['sag'] = cmd - act
        self.get_logger().info(
            'approach: fingertips at [{:+.3f} {:+.3f} {:+.3f}], {:.0f} mm below the tag face'.format(
                *act, (face_z - act[2]) * 1000))
        return True

    def _lift_clear(self, cmd, quat, z):
        """After a failed descent: straight up to z (unchecked), so the fingers are off the object."""
        up = np.array(cmd, dtype=float)
        up[2] = max(up[2], z)
        self.get_logger().warn('approach failed: lifting the hand clear to z={:.3f}'.format(z))
        self.cartesian_to(self._pose(up, quat), 'approach: lift clear', avoid=False, min_fraction=0.2)
        self._sleep(self.settle)

    def _actual_tcp(self):
        tcp = self._tcp_pose_now()
        return None if tcp is None else np.array([tcp.position.x, tcp.position.y, tcp.position.z])

    def _tcp_pose_now(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame, self.tcp, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception:
            return None
        pose = Pose()
        pose.position.x = tf.transform.translation.x
        pose.position.y = tf.transform.translation.y
        pose.position.z = tf.transform.translation.z
        pose.orientation = tf.transform.rotation
        return pose

    def _ensure_clear_start(self, label):
        """Get the arm out of any octomap voxels it is starting in.

        A checked plan cannot begin from a colliding state, and after a
        contact move the map often still holds the voxels the arm itself
        occupied a moment ago.  Wait for the map to catch up; if the start
        is still in collision with it, back straight up a little - an
        unchecked move, like any contact move - and look again.
        """
        if self.dry_run:
            return True
        self._sleep(self.settle_map)
        cleared_map = False
        for attempt in range(self.clear_tries + 1):
            v = self._state_valid(self._start_state())
            if v is None or v[0]:
                return True
            octo_only = all('<octomap>' in c for c in v[1])
            own = octo_only and all(any(k in c for k in ('hand', 'finger', 'camera')) for c in v[1])
            if own and not cleared_map:
                # Voxels touching only the hand's own links are the arm's own points left in the
                # hand's shadow (the chest camera cannot free what it cannot see).  Drop the map;
                # it rebuilds from the live cloud within a frame.
                cleared_map = True
                self.get_logger().warn(
                    '{}: start state touches the octomap only at the hand ({}); clearing the '
                    'octomap and looking again'.format(label, '; '.join(v[1])))
                self._clear_octomap()
                self._sleep(self.settle_map)
                continue
            self.get_logger().warn(
                '{}: start state in collision ({}); {}'.format(
                    label, '; '.join(v[1]),
                    'backing up {:.0f} mm'.format(self.clear_step * 1000)
                    if attempt < self.clear_tries and octo_only
                    else 'giving up'))
            if attempt >= self.clear_tries or not octo_only:
                return False
            cur = self._tcp_pose_now()
            if cur is None:
                return False
            up = Pose()
            up.position.x = cur.position.x
            up.position.y = cur.position.y
            up.position.z = cur.position.z + self.clear_step
            up.orientation = cur.orientation
            # a partial back-up is still a back-up: re-check rather than give up
            self.cartesian_to(up, '{}: clear'.format(label), avoid=False, min_fraction=0.25)
            self._sleep(self.settle_map)
        return False

    def _clear_octomap(self):
        from std_srvs.srv import Empty
        if not hasattr(self, 'clear_map_client'):
            self.clear_map_client = self.create_client(Empty, '/clear_octomap')
        if not self.clear_map_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('/clear_octomap not available')
            return False
        return self._spin(self.clear_map_client.call_async(Empty.Request()), 5.0) is not None

    # ------------------------------------------------------------------- scene
    def _scene_apply(self, scene, label, quiet=False):
        if not self.scene_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                '{}: /apply_planning_scene not available - the planner will '
                'not know about the object'.format(label))
            return False
        scene.is_diff = True
        scene.robot_state.is_diff = True
        req = ApplyPlanningScene.Request(scene=scene)
        res = self._spin(self.scene_client.call_async(req), 10.0)
        ok = res is not None and res.success
        if not ok and not quiet:
            self.get_logger().warn('{}: planning scene update failed'
                                   .format(label))
        return ok

    def _target_object(self, xyz, quat, dims, op):
        """The picked object as a box, in the planning frame."""
        co = CollisionObject()
        co.id = 'target'
        co.header.frame_id = self.frame
        co.operation = op
        if op != CollisionObject.REMOVE:
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [float(dims[0]), float(dims[1]),
                              float(dims[2])]
            co.primitives.append(box)
            co.primitive_poses.append(self._pose(xyz, quat))
        return co

    def scene_add_target(self, xyz, quat, dims):
        """Put the object into the scene so its voxels are excluded from the
        octomap and the planner reasons about a box, not a cloud."""
        sc = PlanningScene()
        sc.world.collision_objects.append(
            self._target_object(xyz, quat, dims, CollisionObject.ADD))
        if self._scene_apply(sc, 'scene: add target'):
            self._in_scene = True
            self.get_logger().info(
                'scene: target box {:.0f} x {:.0f} x {:.0f} mm added'.format(
                    dims[0] * 1000, dims[1] * 1000, dims[2] * 1000))

    def scene_attach_target(self, xyz, quat, dims):
        """Hand the object to the gripper.  From here the planner carries it
        with the hand, and lets the fingers touch it.

        The attach travels ALONE in its diff.  It used to share one with a
        world REMOVE of the same id "as a guard" - and MoveIt ANDs the
        results of everything in a diff, so removing a world object that
        was not there marked the whole diff failed.  The attach had in fact
        gone through; this node believed it had not; the detach at the end
        was skipped for want of a flag; and the planner carried a phantom
        box to the end of the run.  Now the scene is asked what is attached,
        and the flag follows the answer rather than the return code.
        """
        sc = PlanningScene()
        aco = AttachedCollisionObject()
        aco.link_name = self.attach_link
        aco.touch_links = list(self.touch_links)
        aco.object = self._target_object(xyz, quat, dims,
                                         CollisionObject.ADD)
        sc.robot_state.attached_collision_objects.append(aco)
        self._scene_apply(sc, 'scene: attach')
        ids = self._attached_ids()
        self._attached = bool(ids) and 'target' in ids
        self._in_scene = False
        if self._attached:
            self.get_logger().info('scene: target attached to {}'.format(
                self.attach_link))
        else:
            self.get_logger().warn(
                'scene: attach did not take (attached now: {}) - the planner '
                'will not know the object is in hand'.format(ids))

    def _attached_ids(self):
        """What is attached to the robot right now, according to MoveIt -
        not according to a flag this node set."""
        if not self.get_scene_client.wait_for_service(timeout_sec=3.0):
            return None
        req = GetPlanningScene.Request()
        req.components.components = \
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        res = self._spin(self.get_scene_client.call_async(req), 5.0)
        if res is None:
            return None
        return [a.object.id
                for a in res.scene.robot_state.attached_collision_objects]

    def _detach(self, object_id):
        """Detach one object by id, or everything on the link if id is
        empty.  MoveIt puts a detached object back into the world."""
        sc = PlanningScene()
        aco = AttachedCollisionObject()
        aco.link_name = self.attach_link
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        sc.robot_state.attached_collision_objects.append(aco)
        return self._scene_apply(sc, 'scene: detach {}'.format(
            object_id or 'ALL'))

    def scene_release_target(self):
        """Let go: detach, then drop it from the scene entirely - the octomap
        sees it lying there from now on, which is all the planner needs.

        Verified against the scene, not assumed.  The first version sent
        one detach and trusted it; the planner then reported the start state
        in collision with 'target(attached)' - the box was still on the hand
        after every following step, and no plan home could start.
        """
        # Trust the scene, not the flag: if a 'target' is attached, detach
        # it whatever this node believes happened earlier.
        ids = self._attached_ids() or []
        if self._attached or 'target' in ids:
            self._detach('target')
            still = self._attached_ids()
            if still and 'target' in still:
                self.get_logger().warn(
                    "scene: 'target' is STILL attached after a detach by id; "
                    'detaching everything on {}'.format(self.attach_link))
                self._detach('')
                still = self._attached_ids()
            if still and 'target' in still:
                self.get_logger().error(
                    "scene: could not detach 'target' - planning will treat "
                    'the hand as still holding it')
            else:
                self._attached = False
                self._in_scene = True    # MoveIt put it back in the world
        if self._in_scene:
            sc = PlanningScene()
            co = CollisionObject()
            co.id = 'target'
            co.header.frame_id = self.frame
            co.operation = CollisionObject.REMOVE
            sc.world.collision_objects.append(co)
            self._scene_apply(sc, 'scene: remove target')
            self._in_scene = False

    def scene_cleanup(self):
        """Leave nothing behind, whatever happened - and pick up nothing
        left behind by a run before this one."""
        try:
            ids = self._attached_ids()
            if ids:
                self.get_logger().warn(
                    'scene: found attached {} - detaching'.format(ids))
                self._attached = True
            self.scene_release_target()
            # Belt and braces: whatever the flags say, nothing named target
            # may remain anywhere.
            sc = PlanningScene()
            co = CollisionObject()
            co.id = 'target'
            co.header.frame_id = self.frame
            co.operation = CollisionObject.REMOVE
            sc.world.collision_objects.append(co)
            # MoveIt reports "failed" for removing what is not there, which
            # is exactly the usual case here.
            self._scene_apply(sc, 'scene: sweep', quiet=True)
            self._attached = False
            self._in_scene = False
        except Exception as exc:
            self.get_logger().warn('scene cleanup: {}'.format(exc))

    def _grasp_from_wrist(self):
        """Object grasp pose synthesised from a fresh WRIST detection of the object tag.

        Used when the chest camera cannot see the tag (typically because the arm is over it).
        Position: tag centre lowered by half the object height; orientation: the tag's."""
        if self.wrist_poses is None or time.time() - self.wrist_stamp > 1.0:
            return None
        if self.pick_id not in self.wrist_ids:
            return None
        k = self.wrist_ids.index(self.pick_id)
        if k >= len(self.wrist_poses.poses) or self.wrist_poses.header.frame_id != self.frame:
            return None
        src = self.wrist_poses.poses[k]
        h = 2.0 * float(self.below_tag)
        msg = PoseStamped()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = src.position.x
        msg.pose.position.y = src.position.y
        msg.pose.position.z = src.position.z - h / 2.0
        msg.pose.orientation = src.orientation
        self._grasp_lowered = True      # already half a height below the tag face
        return msg

    def wait_for_object(self):
        """Just the object's grasp, settled - no destination tag."""
        self.grasp = None
        self.dims = None
        self.width = None
        end = time.time() + self.det_timeout
        said_wrist = False
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.grasp is None:
                g = self._grasp_from_wrist()
                if g is not None:
                    self.grasp = g
                    if not said_wrist:
                        said_wrist = True
                        self.get_logger().info('object tag {} seen by the WRIST camera (chest view blocked) - using it'.format(self.pick_id))
            if self.grasp is not None:
                grace = time.time() + self.dims_grace
                while rclpy.ok() and self.dims is None and \
                        time.time() < grace:
                    rclpy.spin_once(self, timeout_sec=0.05)
                return True
        self.get_logger().error(
            'did not see the object tag {} within {:.0f} s'.format(
                self.pick_id, self.det_timeout))
        return None

    def wait_for_both(self):
        """Both tags in view, the object's grasp settled - and its size."""
        self.grasp = None
        self.dims = None
        self.width = None
        end = time.time() + self.det_timeout
        said = False
        said_wrist = False
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            place = self._place_pose()
            if self.grasp is None:
                g = self._grasp_from_wrist()
                if g is not None:
                    self.grasp = g
                    if not said_wrist:
                        said_wrist = True
                        self.get_logger().info('object tag {} seen by the WRIST camera (chest view blocked) - using it'.format(self.pick_id))
            if self.grasp is not None and place is not None:
                # The measurement follows the pose by microseconds on the
                # wire but by a whole callback here; give it a moment.
                grace = time.time() + self.dims_grace
                while rclpy.ok() and self.dims is None and \
                        time.time() < grace:
                    rclpy.spin_once(self, timeout_sec=0.05)
                return place
            if not said and time.time() > end - self.det_timeout + 5.0:
                said = True
                self.get_logger().info(
                    'waiting: object tag {} {}, destination tag {} {}'.format(
                        self.pick_id,
                        'seen' if self.grasp is not None else 'NOT seen',
                        self.place_id,
                        'seen' if place is not None else 'NOT seen'))
        self.get_logger().error(
            'did not see both tags within {:.0f} s'.format(self.det_timeout))
        return None

    def _inside(self, xyz, what):
        low, high = xyz < self.ws_min, xyz > self.ws_max
        if not (np.any(low) or np.any(high)):
            return True
        bits = []
        for i, ax in enumerate('xyz'):
            if low[i]:
                bits.append('{} {:+.3f} below {:+.3f}'.format(
                    ax, xyz[i], self.ws_min[i]))
            elif high[i]:
                bits.append('{} {:+.3f} above {:+.3f}'.format(
                    ax, xyz[i], self.ws_max[i]))
        self.get_logger().error(
            '{} [{:+.3f} {:+.3f} {:+.3f}] is outside the workspace box: {}'
            .format(what, xyz[0], xyz[1], xyz[2], '; '.join(bits)))
        return False

    # --------------------------------------------------------------- reach
    def _ik_ok(self, xyz, quat, avoid=True):
        """Is there any arm posture that puts the TCP here?

        avoid=False for the CONTACT poses - the grasp and the place.  With
        the octomap on, the grasp point is by definition inside the object
        and the place point is where the held object meets the surface, so a
        collision-checked IK at either always fails: it did, at every jaw
        angle and every lean, the moment the map went live.  The poses ABOVE
        them keep the check on, and a straight descent from a clear pose is
        safe by construction.
        """
        if not self.ik_client.service_is_ready():
            return None                      # unknown; do not pretend
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group
        req.ik_request.ik_link_name = self.tcp
        req.ik_request.robot_state.is_diff = True
        req.ik_request.avoid_collisions = bool(avoid)
        req.ik_request.timeout.sec = int(self.ik_timeout)
        req.ik_request.timeout.nanosec = int((self.ik_timeout % 1.0) * 1e9)
        ps = PoseStamped()
        ps.header.frame_id = self.frame
        ps.pose = self._pose(xyz, quat)
        req.ik_request.pose_stamped = ps
        res = self._spin(self.ik_client.call_async(req), 5.0)
        if res is None:
            return None
        return res.error_code.val == 1

    def _pick_reachable_yaw(self, grasp_xyz, pre_xyz, base_yaw, ratio,
                            place_xyz=None, place_pre=None):
        """Choose a jaw angle the arm can actually reach.

        OMPL's "Unable to sample any valid states for goal tree" means the
        goal had no reachable posture at all - not that planning was hard.
        On this arm whole bands of jaw angle are like that: a top-down grasp
        at yaw 0 has no IK solution anywhere over the bench, while 90 degrees
        opens the workspace up.

        The object's long axis decides the angle, so whether a grasp is
        reachable is decided by how the object happens to be lying - which is
        not something to leave to luck when the same grasp turned 180 degrees
        is identical for a parallel gripper.  Each candidate is tried against
        /compute_ik first, which costs milliseconds instead of a planning
        attempt.
        """
        offsets = list(self.yaw_offsets)
        if ratio is not None and ratio < self.square_ratio:
            # A clearly oblong object: quarter turns would close the jaws
            # across the long side, which does not fit.
            offsets = [o for o in offsets if abs(o % 180.0) < 1e-6]

        tried = []
        for off in offsets:
            yaw = base_yaw + math.radians(off)

            # First an orientation that works for the GRASP and the approach
            # above it, leaning only as far as it has to.
            quat_g = None
            for q, tdeg, lab in grasp_orientation_candidates(yaw, self.tilts):
                ok = self._ik_ok(grasp_xyz, q, avoid=False)
                if ok is None:
                    self.get_logger().warn(
                        '/compute_ik is not available; taking the first jaw '
                        'angle, straight down, on trust')
                    return q, q, off
                if ok and self._ik_ok(pre_xyz, q):
                    quat_g, lab_g = q, lab
                    break
            if quat_g is None:
                tried.append('{:+.0f} deg: no IK at the grasp at any lean'
                             .format(off))
                continue

            # Then the PLACE.  Its jaw angle is NOT tied to the grasp's: the
            # object rides in a fixed grip, but the hand can be turned about
            # vertical by the wrist on the way over, and the destination
            # asks only for a position.  Tying the two together refused
            # places the arm could reach at a different turn - which is
            # exactly the case the overlay was calling green, because the
            # overlay tries every angle.  So: every turn, every lean, in
            # order of preference, and the first that solves.
            #
            # Checked here, before anything is picked up: the alternative is
            # finding out with the object already in the air.
            if place_xyz is None:
                if lab_g != 'vertical':
                    self.get_logger().info(
                        'grasp with the hand leaned {}'.format(lab_g))
                return quat_g, quat_g, off
            quat_p = None
            for poff in self.place_yaw_offsets:
                pyaw = yaw + math.radians(poff)
                for q, tdeg, lab in grasp_orientation_candidates(
                        pyaw, self.tilts):
                    if self._ik_ok(place_xyz, q, avoid=False) and \
                            self._ik_ok(place_pre, q):
                        quat_p = q
                        lab_p = lab + ('' if poff == 0.0 else
                                       ', turned {:+.0f} deg'.format(poff))
                        break
                if quat_p is not None:
                    break
            if quat_p is None:
                tried.append('{:+.0f} deg: grasp ok ({}) but no IK at the '
                             'place at any turn or lean'.format(off, lab_g))
                continue

            note = []
            if off != 0.0:
                note.append('jaw {:+.0f} deg from the object axis'.format(off))
            if lab_g != 'vertical':
                note.append('grasp leaned {}'.format(lab_g))
            if lab_p != 'vertical':
                note.append('place leaned {}'.format(lab_p))
            if note:
                self.get_logger().info('; '.join(note))
            return quat_g, quat_p, off

        # Say WHERE it was and how far - "out of reach" is a guess until the
        # numbers are on the screen beside it.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame, self.arm_base, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            tr = tf.transform.translation
            base = np.array([tr.x, tr.y, tr.z])
            d = float(np.linalg.norm(np.array(grasp_xyz) - base))
            where = (' The grasp was [{:+.3f} {:+.3f} {:+.3f}], {:.0f} mm '
                     'from the arm base at [{:+.3f} {:+.3f} {:+.3f}].'
                     .format(grasp_xyz[0], grasp_xyz[1], grasp_xyz[2],
                             d * 1000, base[0], base[1], base[2]))
        except Exception:
            where = ' The grasp was [{:+.3f} {:+.3f} {:+.3f}].'.format(
                grasp_xyz[0], grasp_xyz[1], grasp_xyz[2])

        if place_xyz is not None:
            where += ' The place was [{:+.3f} {:+.3f} {:+.3f}].'.format(
                place_xyz[0], place_xyz[1], place_xyz[2])
        self.get_logger().error(
            'no jaw angle works for this grasp: {}.{}'.format(
                ', '.join(tried), where))
        self.get_logger().error(
            'The contact poses are checked for reach only; the poses 10 cm '
            'above them are also checked against the scene (octomap and '
            'all). To see what IS reachable at this height: python3 -m '
            'openarm_vision_pick.reach_map --z {:.2f}'.format(grasp_xyz[2]))
        return None, None, None

    # ------------------------------------------------------------------ motion
    def _start_state(self):
        """RobotState for a request: live, or the dry-run carry-over."""
        st = RobotState()
        st.is_diff = True
        if self.dry_run and self._dry_state is not None:
            names, positions = self._dry_state
            st.joint_state.name = list(names)
            st.joint_state.position = [float(v) for v in positions]
        return st

    def _carry_forward(self, trajectory):
        """Remember where a planned trajectory ends, for the dry run."""
        if not self.dry_run:
            return
        jt = trajectory.joint_trajectory
        if not jt.points:
            return
        self._dry_state = (list(jt.joint_names),
                           list(jt.points[-1].positions))

    def _base_request(self):
        req = MotionPlanRequest()
        req.start_state = self._start_state()
        req.group_name = self.group
        req.num_planning_attempts = int(self.plan_tries)
        req.allowed_planning_time = float(self.plan_time)
        req.max_velocity_scaling_factor = float(self.vel)
        req.max_acceleration_scaling_factor = float(self.acc)
        req.workspace_parameters.header.frame_id = self.frame
        req.workspace_parameters.min_corner = Vector3(x=-2.0, y=-2.0, z=-2.0)
        req.workspace_parameters.max_corner = Vector3(x=2.0, y=2.0, z=2.0)
        return req

    @staticmethod
    def _pose(xyz, quat):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = \
            [float(v) for v in xyz]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, \
            pose.orientation.w = [float(v) for v in quat]
        return pose

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
        return self._send_move(req, label, pose=pose)

    # moveit_msgs/MoveItErrorCodes as shipped in Humble.  An earlier table
    # here was off by one on most of these - -2 is INVALID_MOTION_PLAN, not
    # PLANNING_FAILED - and 99999 is plain FAILURE.
    _ERR = {
        1: 'SUCCESS', 99999: 'FAILURE',
        -1: 'PLANNING_FAILED', -2: 'INVALID_MOTION_PLAN',
        -3: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
        -4: 'CONTROL_FAILED', -5: 'UNABLE_TO_AQUIRE_SENSOR_DATA',
        -6: 'TIMED_OUT', -7: 'PREEMPTED',
        -10: 'START_STATE_IN_COLLISION',
        -11: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
        -12: 'GOAL_IN_COLLISION', -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
        -14: 'GOAL_CONSTRAINTS_VIOLATED', -15: 'INVALID_GROUP_NAME',
        -16: 'INVALID_GOAL_CONSTRAINTS', -17: 'INVALID_ROBOT_STATE',
        -18: 'INVALID_LINK_NAME', -19: 'INVALID_OBJECT_NAME',
        -21: 'FRAME_TRANSFORM_FAILURE', -22: 'COLLISION_CHECKING_UNAVAILABLE',
        -23: 'ROBOT_STATE_STALE', -24: 'SENSOR_INFO_STALE',
        -25: 'COMMUNICATION_FAILURE', -26: 'START_STATE_INVALID',
        -27: 'GOAL_STATE_INVALID', -28: 'UNRECOGNIZED_GOAL_TYPE',
        -31: 'NO_IK_SOLUTION', -32: 'KINEMATIC_STATE_NOT_INITIALIZED',
    }

    # A plan can be found and then invalidated between planning and the
    # start of execution when the octomap updates underneath it (error -3),
    # and OMPL occasionally just misses on one attempt (-1).  Both are
    # transient: the next attempt sees a fresh scene.  Retry a couple of
    # times before treating it as a real failure.
    _ENV_CHANGE_RETRIES = 3

    def _send_move(self, req, label, pose=None, joints=None):
        """Plan (and, unless dry_run, execute) one MoveGroup goal."""
        for attempt in range(self._ENV_CHANGE_RETRIES + 1):
            ok, code = self._send_move_once(req, label)
            if ok:
                return True
            if code in (-3, -1) and attempt < self._ENV_CHANGE_RETRIES:
                self.get_logger().warn(
                    '{}: MoveItErrorCode {} (transient - the scene changed '
                    'under the plan); retrying ({}/{})'.format(
                        label, code, attempt + 1, self._ENV_CHANGE_RETRIES))
                self._sleep(0.4)
                continue
            self._diagnose_plan_failure(label, pose, joints)
            return False
        return False

    def _send_move_once(self, req, label):
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = bool(self.dry_run)
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        handle = self._spin(self.move_client.send_goal_async(goal), 20.0)
        if handle is None or not handle.accepted:
            self.get_logger().error('{}: goal rejected'.format(label))
            return False, 99999
        res = self._spin(handle.get_result_async(), self.plan_time + 60.0)
        if res is None:
            self.get_logger().error('{}: timed out'.format(label))
            return False, -6
        code = res.result.error_code.val
        if code != 1:
            self.get_logger().error('{}: MoveItErrorCode {} ({})'.format(
                label, code, self._ERR.get(code, '?')))
            return False, code
        self._carry_forward(res.result.planned_trajectory)
        self.get_logger().info('{}: {}'.format(
            label, 'planned' if self.dry_run else 'done'))
        return True, 1

    def _diagnose_plan_failure(self, label, pose, joints=None):
        """Say WHY a plan failed, instead of just that it did.

        PLANNING_FAILED covers two very different situations: the arm is
        starting from a state the scene says is in collision, or the goal has
        no collision-free posture.  With an octomap live, both become
        possible for the first time - the resting arm may be within a voxel
        of the bench edge, or a freshly added collision box may sit where the
        goal wants the fingers - and the fix is different for each.  So ask
        the scene directly.
        """
        # 1. the start state
        v = self._state_valid(self._start_state())
        if v is None:
            self.get_logger().error(
                '  (could not query /check_state_validity)')
        elif v[0]:
            self.get_logger().error('  start state: clear')
        else:
            self.get_logger().error(
                '  start state IN COLLISION: {}'.format(
                    '; '.join(v[1]) or 'no contact detail'))
            self.get_logger().error(
                '  -> the planner cannot leave a colliding state. If the '
                'arm is resting near the bench, the octomap padding may be '
                'touching it: raise it clear first, or start further from '
                'the bench.')
        # 2. the goal - a joint goal is a state, so ask the scene about it
        #    directly; the home pose can be in collision with the map just
        #    as any other can, and "start clear, plan failed" said nothing.
        if joints is not None:
            st = RobotState()
            st.is_diff = True
            st.joint_state.name = list(self.joint_names)
            st.joint_state.position = [float(v) for v in joints]
            v = self._state_valid(st)
            if v is None:
                self.get_logger().error('  goal state: could not query')
            elif v[0]:
                self.get_logger().error(
                    '  goal state: clear - so the planner failed to find a '
                    'path between two valid states; more planning time, or '
                    'something in the way')
            else:
                self.get_logger().error(
                    '  goal state IN COLLISION: {}'.format(
                        '; '.join(v[1]) or 'no contact detail'))
        if pose is not None:
            quat = (pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w)
            xyz = (pose.position.x, pose.position.y, pose.position.z)
            reach = self._ik_ok(xyz, quat, avoid=False)
            clear = self._ik_ok(xyz, quat, avoid=True) if reach else False
            self.get_logger().error(
                '  goal: {}'.format(
                    'reachable and clear of the scene' if clear else
                    ('reachable but EVERY posture collides with the scene '
                     '- something in the octomap or a scene object is where '
                     'the hand needs to be' if reach else
                     'not reachable at all')))

    def _state_valid(self, state):
        """(valid, [contact descriptions]) from /check_state_validity."""
        if not hasattr(self, 'valid_client'):
            from moveit_msgs.srv import GetStateValidity
            self.valid_client = self.create_client(GetStateValidity,
                                                   '/check_state_validity')
        if not self.valid_client.wait_for_service(timeout_sec=3.0):
            return None
        from moveit_msgs.srv import GetStateValidity
        req = GetStateValidity.Request()
        req.robot_state = state
        req.group_name = self.group
        res = self._spin(self.valid_client.call_async(req), 10.0)
        if res is None:
            return None
        # moveit_msgs/ContactInformation: ROBOT_LINK=0, WORLD_OBJECT=1,
        # ROBOT_ATTACHED=2.  An earlier table had the last two swapped and
        # read a phantom attached box as a world one.
        kind = {0: 'link', 1: 'world', 2: 'attached'}
        contacts = ['{}({}) <-> {}({})'.format(
            c.contact_body_1, kind.get(c.body_type_1, '?'),
            c.contact_body_2, kind.get(c.body_type_2, '?'))
            for c in res.contacts]
        return bool(res.valid), sorted(set(contacts))

    def cartesian_to(self, pose, label, avoid=True, min_fraction=None):
        """A straight-line move.

        avoid=False for the last few centimetres into and out of a grasp:
        those segments END in contact, or start there, and a collision check
        that forbids contact forbids the grasp.  Everything longer keeps the
        check on.
        """
        req = GetCartesianPath.Request()
        req.header.frame_id = self.frame
        req.start_state = self._start_state()
        req.group_name = self.group
        req.link_name = self.tcp
        req.waypoints = [pose]
        req.max_step = float(self.cart_step)
        req.jump_threshold = 0.0
        req.avoid_collisions = bool(avoid)
        for attr, val in (('max_velocity_scaling_factor', self.vel),
                          ('max_acceleration_scaling_factor', self.acc)):
            if hasattr(req, attr):
                setattr(req, attr, float(val))

        res = self._spin(self.cart_client.call_async(req), 30.0)
        if res is None:
            self.get_logger().error(
                '{}: Cartesian service timed out'.format(label))
            return False
        need_frac = self.cart_min if min_fraction is None else min_fraction
        if res.fraction < need_frac:
            # Approximating a straight line into a grasp is how a gripper
            # ends up somewhere other than where it was told.
            self.get_logger().error(
                '{}: only {:.0f}% of the straight line is reachable (need '
                '{:.0f}%)'.format(label, res.fraction * 100, need_frac * 100))
            return False
        self.get_logger().info('{}: path {:.0f}% solved'.format(
            label, res.fraction * 100))
        if self.dry_run:
            self._carry_forward(res.solution)
            return True

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        handle = self._spin(self.exec_client.send_goal_async(goal), 20.0)
        if handle is None or not handle.accepted:
            self.get_logger().error('{}: execution rejected'.format(label))
            return False
        res = self._spin(handle.get_result_async(), 120.0)
        if res is None or res.result.error_code.val != 1:
            self.get_logger().error('{}: execution failed'.format(label))
            return False
        return True

    def _joint_for_span(self, span):
        """Opening in metres -> finger joint position, clamped to travel."""
        q = (span - self.gap0) / 2.0
        return float(min(max(q, 0.0), self.travel))

    def _max_span(self):
        return self.gap0 + 2.0 * self.travel

    def gripper(self, span, label):
        """Command an OPENING, converting to the joint position it needs."""
        q = self._joint_for_span(span)
        if span > self._max_span() + 1e-6:
            self.get_logger().warn(
                '{}: asked for a {:.0f} mm opening but the gripper only '
                'reaches {:.0f} mm; commanding the maximum'.format(
                    label, span * 1000, self._max_span() * 1000))
        self.get_logger().info(
            '{}: opening {:.0f} mm -> finger joint {:.4f}'.format(
                label, span * 1000, q))
        return self._gripper_joint(q, label)

    def _gripper_joint(self, width, label):
        if self.dry_run:
            self.get_logger().info(
                '{}: would move the gripper to {:.3f} m'.format(label, width))
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
            self.get_logger().error('{}: gripper goal rejected'.format(label))
            return False
        self._spin(handle.get_result_async(), self.g_time + 10.0)
        self.get_logger().info('{}: gripper at {:.3f} m'.format(label, width))
        return True

    # ------------------------------------------------------------------- cycle
    def realign_now(self):
        """From the current hand pose: wrist-camera alignment over the tag, then hold."""
        tcp = self._tcp_pose_now()
        if tcp is None:
            self.get_logger().error('realign: no TF for the TCP')
            return False
        q = tcp.orientation
        here = np.array([tcp.position.x, tcp.position.y, tcp.position.z])
        ctx = dict(grasp=here - [0, 0, self.approach], pre=here.copy(),
                   quat_g=(q.x, q.y, q.z, q.w))
        self.get_logger().info(
            'REALIGN from [{:+.3f} {:+.3f} {:+.3f}], keeping the current hand orientation'.format(*here))
        if not self.refine_with_wrist(ctx):
            self.get_logger().error('realign: the sideways move failed')
            return False
        self.get_logger().info(
            'REALIGNED: hand over [{:+.3f} {:+.3f}] at z={:.3f}, jaws untouched. Ctrl-C to finish '
            '(the arm stays).'.format(ctx['pre'][0], ctx['pre'][1], ctx['pre'][2]))
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.5)
        return True

    def run_once(self):
        if self.realign_only:
            return self.realign_now()
        self.scene_cleanup()
        self.remember_initial()
        if self.lift_only:
            self.get_logger().info(
                'LIFT-ONLY: pick the object up and hold it, no destination')
            if self.wait_for_object() is None:
                return False
            place_xy = None
        else:
            place_xy = self.wait_for_both()
            if place_xy is None:
                return False

        gp = self.grasp.pose.position
        grasp_xyz = np.array([gp.x, gp.y, gp.z])
        q = self.grasp.pose.orientation
        quat_obj = (q.x, q.y, q.z, q.w)
        # The detector publishes the object's long axis as a yaw about world
        # Z; the jaws have to close across it, hence the quarter turn.
        yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))
        base_yaw = yaw + math.pi / 2.0

        measured = ([float(v) for v in self.dims]
                    if self.dims is not None and self.dims[2] > 0.005 else None)
        if measured is not None and self.trust_dims:
            dims = measured
            self.get_logger().info(
                'object measures {:.0f} x {:.0f} x {:.0f} mm'.format(
                    dims[0] * 1000, dims[1] * 1000, dims[2] * 1000))
        elif measured is not None:
            dims = [self.assumed_l, self.assumed_w, self.assumed_h]
            self.get_logger().info(
                'detector measures {:.0f} x {:.0f} x {:.0f} mm; using the configured {:.0f} x {:.0f} x {:.0f} mm '
                '(trust_measured_dims is off)'.format(*[v * 1000 for v in measured + dims]))
            if not getattr(self, '_grasp_lowered', False):
                # the detector put the grasp half of ITS height below the top face; wanted: below_tag
                dz = measured[2] / 2.0 - self.below_tag
                grasp_xyz = grasp_xyz + [0.0, 0.0, dz]
                self.get_logger().info('grasp height corrected by {:+.0f} mm: {:.0f} mm below the tag face'.format(
                    dz * 1000, self.below_tag * 1000))
        else:
            dims = [self.assumed_l, self.assumed_w, self.assumed_h]
            self.get_logger().warn(
                'no measurement from the detector; assuming the object is '
                '{:.0f} x {:.0f} x {:.0f} mm'.format(
                    dims[0] * 1000, dims[1] * 1000, dims[2] * 1000))
        height = dims[2]

        # Close on the object's measured SHORT SIDE, less the squeeze.  Not
        # on the detector's grasp_width: that figure is an OPENING - the
        # object plus clearance - and treating it as the object's width
        # closed the jaws 6 mm clear of a 63 mm box and gripped nothing.
        if measured is None and not getattr(self, '_grasp_lowered', False):
            # Without a depth measurement the detector publishes the TAG CENTRE, i.e. the top
            # face; the jaws have to close half a height below it.
            grasp_xyz = grasp_xyz - [0.0, 0.0, self.below_tag]
            self.get_logger().info(
                'no measured height: grasping {:.0f} mm below the tag face, at z={:.3f}'.format(
                    self.below_tag * 1000, grasp_xyz[2]))
        close_to = max(self.gap0, dims[1] - self.squeeze)
        self.get_logger().info(
            'object short side {:.0f} mm; closing to a {:.0f} mm opening'
            .format(dims[1] * 1000, close_to * 1000))

        # Set the object down so its BOTTOM rests on the destination surface:
        # the grasp holds it at its middle, so the grasp point goes half a
        # height above the tag's plane.
        if not self._inside(grasp_xyz, 'the grasp'):
            return False
        if self.lift_only or self.stop_after_align or self.stop_before_close:
            # no place in these modes: do not let an unreachable destination veto the grasp
            place_xyz = None
        else:
            place_xyz = np.array([place_xy[0], place_xy[1],
                                  place_xy[2] + height / 2.0 + self.place_clear])
            if not self._inside(place_xyz, 'the place'):
                return False

        # The object is NOT put into the planning scene as a world object.
        #
        # It was, and the very first plan then failed with the start state
        # reported "in collision: target <-> target" - the box tangled with
        # itself the moment it existed, and every plan was refused until the
        # abort handler removed it, whereupon the plan home went through at
        # once.  The world object was never needed: the contact segments run
        # with collision checking off anyway, the pre-grasp sits well above
        # the object's voxels, and the only thing the transfer needs is the
        # ATTACHED box, which is created directly at the grasp.
        #
        # The camera keeps seeing the object where it lies; the octomap says
        # so; that is enough.

        ratio = dims[1] / dims[0] if dims[0] > 1e-6 else None
        quat_g, quat_p, off = self._pick_reachable_yaw(
            grasp_xyz, grasp_xyz + [0, 0, self.approach], base_yaw, ratio,
            place_xyz,
            None if place_xyz is None else place_xyz + [0, 0, self.approach])
        if quat_g is None:
            return False

        jaw_deg = (math.degrees(base_yaw) + off) % 360.0
        if place_xyz is None:
            self.get_logger().info(
                'grasp [{:+.3f} {:+.3f} {:+.3f}] with the jaws at {:.0f} deg (object axis {:.0f} deg), then lift and hold'.format(
                    *grasp_xyz, jaw_deg, math.degrees(yaw) % 360.0))
        else:
            self.get_logger().info(
                'grasp [{:+.3f} {:+.3f} {:+.3f}] -> place [{:+.3f} {:+.3f} '
                '{:+.3f}]'.format(*np.concatenate([grasp_xyz, place_xyz])))

        pre = grasp_xyz + [0, 0, self.approach]
        lifted = grasp_xyz + [0, 0, self.lift]
        above = (None if place_xyz is None
                 else place_xyz + [0, 0, self.approach])

        # Everything the recovery needs to put the object back where it was,
        # and the one value a later step may revise (the place orientation).
        ctx = dict(grasp=grasp_xyz, pre=pre, quat_g=quat_g, quat_p=quat_p, jaw_yaw=math.radians(jaw_deg),
                   view_len=max(0.0, self.view_dist) if self.refine else 0.0,
                   place=place_xyz, above=above, dims=dims,
                   quat_obj=quat_obj, base_yaw=base_yaw)
        self._ctx = ctx

        def replan_place():
            """With the object now attached, is the place still reachable?

            The place orientation was chosen before anything was in the
            hand.  Once the box is attached it takes part in every collision
            check, and a pose that was clear for the bare gripper can be
            blocked for the gripper plus a 146 mm box.  So ask again, with
            the object in hand, and take a different turn or lean if the
            first choice no longer solves.
            """
            q = ctx['quat_p']
            if self._ik_ok(above, q) and self._ik_ok(place_xyz, q,
                                                       avoid=False):
                return True
            self.get_logger().warn(
                'the place chosen before the grasp is blocked now the object '
                'is in hand; choosing again')
            yaw = base_yaw + math.radians(off)
            for poff in self.place_yaw_offsets:
                for q2, tdeg, lab in grasp_orientation_candidates(
                        yaw + math.radians(poff), self.tilts):
                    if self._ik_ok(above, q2) and \
                            self._ik_ok(place_xyz, q2, avoid=False):
                        ctx['quat_p'] = q2
                        self.get_logger().info(
                            'place re-chosen: {}{}'.format(
                                lab, '' if poff == 0.0 else
                                ', turned {:+.0f} deg'.format(poff)))
                        return True
            self.get_logger().error(
                'no place orientation works with the object in hand')
            return False

        # Long moves are planned with every collision check on.  The short
        # straight segments into and out of a grasp are not - they end in
        # contact by design.
        def steps_index(name):
            return [i for i, st in enumerate(steps) if st[0] == name][0]

        steps = [
            ('open', lambda: self.gripper(self.open_span, 'open')),
            ('raise', lambda: self.maybe_raise()),
            # a checked plan cannot leave a start the map calls occupied (stale self-voxels after an
            # aborted run): clear that first, as the transfer does
            ('pre-grasp', lambda: self.goto_pregrasp(ctx)),
            ('refine', lambda: self.refine_with_wrist(ctx)),
            ('centre', lambda: self.centre_over_tag(ctx)),
            ('approach', lambda: self.approach_checked(ctx)),
            ('close', lambda: self.gripper(close_to, 'close')),
            ('attach', lambda: self.scene_attach_target(
                ctx['grasp'], quat_obj, dims) or True),
            ('lift', lambda: self.cartesian_to(
                self._pose(ctx['grasp'] + [0, 0, self.lift], quat_g), 'lift', avoid=False)),
            ('verify', lambda: self.verify_grasp('verify')),
            ('replan-place', replan_place),
            ('transfer', lambda: self._ensure_clear_start('transfer') and
                self.move_to_pose(self._pose(above, ctx['quat_p']),
                                  'transfer')),
            ('lower', lambda: self.cartesian_to(
                self._pose(place_xyz, ctx['quat_p']), 'lower', avoid=False)),
            ('release', lambda: self.gripper(self.open_span, 'release')),
            ('detach', lambda: self.scene_release_target() or True),
            ('retreat', lambda: self.cartesian_to(
                self._pose(above, ctx['quat_p']), 'retreat', avoid=False)),
        ]
        if self.stop_after_align:
            # open -> pre-grasp -> wrist refine: hand vertical over the object's centre, and stop
            steps = steps[:steps_index('approach')]
        elif self.stop_before_close:
            # open -> pre-grasp -> straight down, then stop with the jaws open around the object
            steps = steps[:steps_index('close')]
        elif self.lift_only:
            # Pick, lift, confirm it took - and stop there, holding it.
            steps = steps[:steps_index('verify') + 1]
        elif self.go_home:
            steps.append(('initial', lambda: self.return_to_initial('initial')))

        for name, step in steps:
            if not step():
                self.get_logger().error('aborting at step "{}"'.format(name))
                self._recover(name)
                return False
            self._sleep(self.settle)

        if self.stop_after_align:
            act = self._actual_tcp()
            self.get_logger().info(
                'ALIGNED: hand actually at {} vs tag centre [{:+.3f} {:+.3f}], jaws open. Ctrl-C to finish '
                '(the arm stays where it is).'.format(
                    'unknown' if act is None else '[{:+.3f} {:+.3f} {:+.3f}]'.format(*act),
                    ctx['grasp'][0], ctx['grasp'][1]))
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.5)
        elif self.stop_before_close:
            self.get_logger().info(
                'STOPPED BEFORE CLOSING: the jaws are open around the object at the grasp pose. '
                'Check the fit by eye; Ctrl-C to finish (the arm stays where it is).')
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.5)
        elif self.lift_only:
            self.get_logger().info(
                'LIFT-ONLY done: the object is held at the lift height. '
                'Ctrl-C to finish; it is left in the gripper.')
        else:
            self.get_logger().info('done')
        return True

    # Steps during which the object is physically in the gripper.
    _HOLDING = ('attach', 'lift', 'verify', 'replan-place', 'transfer',
                'lower')

    def _recover(self, failed_at):
        """After an abort: put the object back if it is held, tidy the scene,
        and bring the arm home rather than leave it over the bench.

        The first version opened the gripper wherever the abort happened.
        With a transfer failing after the lift, that was twelve centimetres
        up - the object was dropped.  Holding something means putting it
        down before letting go, and the safest place to put it is where it
        was picked up: that pose is known, it was reachable a minute ago,
        and nothing has moved there since.
        """
        if not self.home_on_abort:
            self.scene_cleanup()
            return
        holding = (not self.dry_run) and failed_at in self._HOLDING
        if holding:
            self.get_logger().warn(
                'recovering while HOLDING the object: putting it back')
            if not self._put_back():
                self.get_logger().error(
                    'could not put the object back - it is still in the '
                    'gripper. The arm is where it stopped; nothing more '
                    'will be commanded.')
                return
        elif not self.dry_run and failed_at not in ('open', 'pre-grasp'):
            self.gripper(self.open_span, 'recover: open')
        self.scene_cleanup()
        if not self.return_to_initial('recover: initial'):
            self.get_logger().error(
                'could not get back to the initial position - the arm is '
                'where it stopped')

    def _put_back(self):
        """Return the held object to its pick-up point and release it there.

        Plan (checked, with the attached object counted) to the pre-grasp,
        descend the same unchecked ten centimetres that picked it up, open,
        rise, detach.  If even that plan fails, descend as far as a straight
        line allows from wherever the arm is and release there - low is
        better than high - and say so.
        """
        c = self._ctx
        if c is None:
            return False
        if self._ensure_clear_start('put-back') and \
                self.move_to_pose(self._pose(c['pre'], c['quat_g']),
                                  'put-back: to pre-grasp'):
            ok = self.cartesian_to(self._pose(c['grasp'], c['quat_g']),
                                   'put-back: lower', avoid=False,
                                   min_fraction=0.5)
            self.gripper(self.open_span, 'put-back: release')
            self.scene_release_target()
            self.cartesian_to(self._pose(c['pre'], c['quat_g']),
                              'put-back: rise', avoid=False, min_fraction=0.3)
            if ok:
                self.get_logger().info('put-back: object returned to where '
                                       'it was picked up')
            else:
                self.get_logger().warn('put-back: released short of the '
                                       'pick-up point')
            return True

        # Last resort: straight down from here, as far as it goes.
        self.get_logger().warn(
            'put-back: cannot reach the pick-up point; lowering straight '
            'down from here before releasing')
        cur = self._tcp_pose_now()
        if cur is None:
            return False
        down = Pose()
        down.position.x = cur.position.x
        down.position.y = cur.position.y
        down.position.z = max(self.ws_min[2], cur.position.z - 0.15)
        down.orientation = cur.orientation
        self.cartesian_to(down, 'put-back: descend', avoid=False,
                          min_fraction=0.1)
        self.gripper(self.open_span, 'put-back: release')
        self.scene_release_target()
        return True

def main():
    rclpy.init()
    node = MarkerPickPlace()
    try:
        if not node.wait_for_servers():
            node.get_logger().error(
                'MoveIt is not up; is move_group running?')
            return
        node.run_once()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # A target left attached or lying in the scene would be planned
        # around by every later run.
        try:
            node.scene_cleanup()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
