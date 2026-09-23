# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Everything the hand demo needs, in one command, in one RViz window.

  ros2 launch openarm_vision_pick hand_demo.launch.py                  # mock
  ros2 launch openarm_vision_pick hand_demo.launch.py use_fake_hardware:=false

Brings up:

  robot_state_publisher + ros2_control + the five controllers
  move_group
  the chest D455f
  hand_detector
  RViz, already showing the robot, the TF frames, the chest point cloud,
  the hand marker and the detector's debug image

WHY THE ROBOT BRING-UP IS SPELLED OUT HERE INSTEAD OF INCLUDED

Both of the obvious things to include start an RViz of their own that cannot
be switched off - demo.launch.py opens MoveIt's, and
openarm_bringup/openarm.bimanual.launch.py opens the description package's.
Including either would leave two or three RViz windows fighting over the
screen, so the handful of nodes they wrap are created directly below.  They
are the same nodes with the same parameters; only the RViz is ours.

A TRAP IN THE MOVEIT PACKAGE

demo.launch.py accepts arm_type in any spelling ("v1.0", "v10",
"openarm_v1.0") and maps it to a config directory.  move_group.launch.py and
moveit_rviz.launch.py do NOT: they paste arm_type straight into
config/<arm_type>/openarm_bimanual.srdf, so they need the directory name
exactly - "openarm_v1.0".  Passing them "v1.0" fails with

    File .../config/v1.0/openarm_bimanual.srdf doesn't exist

This file takes the friendly spelling and converts it, so arm_type:=v1.0
works everywhere.

OTHER DEFAULTS WORTH KNOWING

  arm_type is v1.0 here.  The MoveIt package's own default is v2.0, which is
  the wrong robot for this bench and, more quietly, the wrong URDF for the
  wrist camera - only the v1.0 model carries one.

  The CAN interfaces default to the FOLLOWER pair, can0 and can1.

  Do not trust those numbers blindly.  can* numbering follows USB enumeration
  order, so it is a property of this machine and of what happened to be
  plugged in first - not of OpenArm.  What is stable is the USB PORT: the
  follower adapter is on port 1-4.  With both adapters connected it came up
  as can2/can3 and the leader took can0/can1; with the leader unplugged the
  follower moved down to can0/can1, which is the case these defaults are set
  for.  ~/openarm_link_check.sh reports the mapping by port and says which
  pair is the follower, so run it after any re-plug rather than guessing.

  The camera's IMU is off.  With it on the D455f dropped off the USB bus 18
  times in 60 seconds and delivered no usable frames, because /dev/hidraw* is
  root-only here and the IMU arrives over hidraw.  Nothing in this pipeline
  needs it; 99-openarm-realsense.rules fixes the permissions if you ever do.
"""

import os

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# The chest camera's mounting, corrected.
#
# openarm_description ships chest_camera_rpy="3.141593 0.785398 0.0" and marks
# the camera origins in that file as placeholders.  The leading 3.141593 is a
# 180-degree roll, and in a RealSense URDF camera_link's +x IS the viewing
# direction, so a roll there is a rotation about the optical axis - it turns
# the camera upside down.
#
# Measured on the real robot, by fitting the floor in the chest camera's own
# depth image and comparing with what TF claimed:
#
#     floor normal, measured, in the camera frame : [-0.004 -0.730 -0.684]
#     world +Z per TF,        in the camera frame : [+0.001 +0.707 -0.707]
#
# The tilt magnitude agreed (46.9 measured against 45.0 claimed - the 45 deg
# pitch is right), but the y component came out with the opposite sign, which
# is exactly the 180-degree roll.  With the old value every robot link
# projected 220 pixels ABOVE the top of the image and openarm_left_link0 came
# out behind the camera.
#
# Pass chest_camera_rpy:="..." to override, e.g. while re-measuring.
# z is 20 mm below the description package's 0.653484: first taken down
# 40 mm, then back up 20, against the real robot (2026-08-28).  The
# projected-link check (tf_project.py) is the way to see whether this is
# right - the arm's links should land on the arm in the camera image.
CHEST_CAMERA_XYZ = '0.0290 0.0000 0.6274'
CHEST_CAMERA_RPY = '0.0 0.785398 0.0'

# Every spelling the MoveIt package tolerates, mapped to the one that its
# config directories actually use.
_V10 = ('v1.0', 'v10', 'v1_0', 'openarm_v1.0', 'openarm_v10', 'openarm_v1_0')


def _config_dir(arm_type_str):
    return 'openarm_v1.0' if arm_type_str in _V10 else 'openarm_v2.0'


def _xacro_file(arm_type_str):
    if arm_type_str in _V10:
        return 'openarm_v1.0', 'openarm_v10.urdf.xacro'
    return 'openarm_v2.0', 'openarm_v20.urdf.xacro'


def robot_nodes(context, arm_type, use_fake, right_can, left_can,
                chest_xyz, chest_rpy, chest_model):
    """robot_state_publisher and ros2_control, as demo.launch.py builds them."""
    arm_type_str = context.perform_substitution(arm_type)
    folder, xacro_name = _xacro_file(arm_type_str)

    xacro_path = os.path.join(
        get_package_share_directory('openarm_description'),
        'assets', 'robot', folder, 'urdf', xacro_name)

    robot_description = xacro.process_file(xacro_path, mappings={
        'arm_type': arm_type_str,
        'bimanual': 'true',
        'ros2_control': 'true',
        'use_fake_hardware': context.perform_substitution(use_fake),
        'right_can_interface': context.perform_substitution(right_can),
        'left_can_interface': context.perform_substitution(left_can),
        'chest_camera_xyz': context.perform_substitution(chest_xyz),
        'chest_camera_rpy': context.perform_substitution(chest_rpy),
        'chest_camera_model': context.perform_substitution(chest_model),
    }).toprettyxml(indent='  ')

    rd = {'robot_description': robot_description}
    controllers = os.path.join(
        get_package_share_directory('openarm_bringup'), 'config',
        'controllers', 'openarm_bimanual_moveit_controllers.yaml')

    return [
        Node(package='robot_state_publisher',
             executable='robot_state_publisher',
             name='robot_state_publisher', output='screen', parameters=[rd]),
        Node(package='controller_manager', executable='ros2_control_node',
             output='both', parameters=[rd, controllers]),
    ]


def _moveit_params(arm_type_str, use_fake_str, chest_xyz, chest_rpy,
                   chest_model='d455',
                   solver='pick_ik', ik_timeout=0.05,
                   octomap=True, cloud_topic='', wrist_cloud_topic=''):
    """The MoveIt parameter set, built the way demo.launch.py builds it.

    move_group.launch.py is NOT used, and not because of the arm_type
    spelling alone: it lets MoveItConfigsBuilder pull in its default pipeline
    list, which includes Pilz, and Pilz then demands
    config/pilz_cartesian_limits.yaml at the package root - a file this
    package does not have.  It dies with

        File .../config/pilz_cartesian_limits.yaml doesn't exist

    demo.launch.py sidesteps that by asking for OMPL only and merging the
    per-arm Pilz limits itself if they happen to exist.  Same here.
    """
    cfg = _config_dir(arm_type_str)
    folder, xacro_name = _xacro_file(arm_type_str)
    xacro_path = os.path.join(
        get_package_share_directory('openarm_description'),
        'assets', 'robot', folder, 'urdf', xacro_name)

    moveit_config = (
        MoveItConfigsBuilder(
            'openarm', package_name='openarm_bimanual_moveit_config')
        .robot_description(file_path=xacro_path, mappings={
            'arm_type': arm_type_str,
            'bimanual': 'true',
            'ros2_control': 'true',
            'use_fake_hardware': use_fake_str,
            # The same correction as robot_state_publisher gets: if the two
            # descriptions disagree, the planning scene and TF disagree too.
            'chest_camera_xyz': chest_xyz,
            'chest_camera_rpy': chest_rpy,
            'chest_camera_model': chest_model,
        })
        .robot_description_semantic(
            file_path='config/{}/openarm_bimanual.srdf'.format(cfg))
        .robot_description_kinematics(
            file_path='config/{}/kinematics.yaml'.format(cfg))
        .joint_limits(file_path='config/{}/joint_limits.yaml'.format(cfg))
        .trajectory_execution(
            file_path='config/{}/moveit_controllers.yaml'.format(cfg))
        .planning_pipelines(pipelines=['ompl'],
                            default_planning_pipeline='ompl')
        .to_moveit_configs()
    )
    params = moveit_config.to_dict()

    # --- the IK solver -----------------------------------------------------
    #
    # The MoveIt package ships KDL with a 5 ms timeout.  KDL is a local
    # Newton-Raphson solver seeded from wherever the arm is now, and on a
    # seven-joint arm that has a whole redundant degree of freedom to get lost
    # in, five milliseconds finds a solution only when one is close by.  The
    # reach map that came out of it - a strip 20 cm deep - was the solver's
    # reach, not the arm's.  pick_ik is a global optimiser that does not care
    # where the arm started, and it is what MoveIt 2 recommends.
    kin = params.setdefault('robot_description_kinematics', {})
    for grp in ('left_arm', 'right_arm'):
        g = kin.setdefault(grp, {})
        if solver == 'pick_ik':
            g.clear()
            g.update({
                'kinematics_solver': 'pick_ik/PickIkPlugin',
                'kinematics_solver_timeout': float(ik_timeout),
                'kinematics_solver_attempts': 3,
                'mode': 'global',
                'stop_optimization_on_valid_solution': True,
                'position_scale': 1.0,
                'rotation_scale': 0.5,
                'position_threshold': 0.001,
                'orientation_threshold': 0.01,
                'cost_threshold': 0.001,
                'minimal_displacement_weight': 0.0,
                'gd_step_size': 0.0001,
            })
        else:
            g['kinematics_solver'] = 'kdl_kinematics_plugin/KDLKinematicsPlugin'
            g['kinematics_solver_timeout'] = float(ik_timeout)
            g['kinematics_solver_search_resolution'] = 0.005

    # --- the octomap -------------------------------------------------------
    #
    # Without this, move_group checks the arm against ITSELF and nothing
    # else - "No 3D sensor plugin(s) defined for octomap updates" in the log
    # was the warning - and the hand went straight through a box the camera
    # could plainly see.  The chest camera's point cloud is turned into an
    # occupancy map that every plan is checked against.  The arm's own links
    # are masked out of the cloud by the updater (padding_*), and anything
    # added to the scene as a collision object - the object being picked -
    # has its voxels excluded too, which is what lets the gripper close on
    # it without the planner objecting.
    if octomap and cloud_topic:
        params['sensors'] = ['chest_cloud']
        params['chest_cloud'] = {
            'sensor_plugin': 'occupancy_map_monitor/PointCloudOctomapUpdater',
            'point_cloud_topic': cloud_topic,
            'max_range': 2.0,
            'point_subsample': 2,
            # Self-filter margin.  It MUST exceed default_robot_padding (0.03) plus half a voxel
            # (0.01), or points the chest camera sees 3-5 cm off the hand - the D405 body, its
            # USB plug and cable, the hand covers - survive the filter and then "collide" with the
            # padded hand: 2026-09-09 every collision-checked IK at the pre-grasp failed and the
            # recovery could not leave the "colliding" start (octomap <-> camera_mount/hand/finger).
            # 0.10 (not 0.07): the wrist camera's cable hangs to the fingertips and its points survived a 7 cm mask.
            'padding_offset': 0.10,
            'padding_scale': 1.2,
            # Faster clearing of the voxels the arm leaves behind it: a
            # voxel is only freed when a ray passes through, and at 5 Hz
            # the hand had pulled back before the map noticed.
            'max_update_rate': 10.0,
            'filtered_cloud_topic': '/chest/filtered_cloud',
        }
        params['octomap_frame'] = 'world'
        params['octomap_resolution'] = 0.02

    # The wrist camera as a second source.  It rides on the arm, so it looks
    # at exactly where the hand is going, from close up - which is the one
    # view the chest camera cannot have: behind the object, in the arm's
    # own shadow, under the hand.  Both clouds update the same map; the
    # updater transforms each into octomap_frame through TF, so a moving
    # sensor is no different from a fixed one.
    #
    # max_range is the D405's: beyond half a metre its depth is noise, and
    # noise in an occupancy map is phantom obstacles.  The attached object,
    # which this camera sees filling its frame while held, is excluded from
    # the map by MoveIt (attached bodies are), and the fingers are masked
    # as robot links.
    if octomap and wrist_cloud_topic:
        params.setdefault('sensors', [])
        params['sensors'] = list(params['sensors']) + ['wrist_cloud']
        params['wrist_cloud'] = {
            'sensor_plugin': 'occupancy_map_monitor/PointCloudOctomapUpdater',
            'point_cloud_topic': wrist_cloud_topic,
            'max_range': 0.6,
            'point_subsample': 2,
            'padding_offset': 0.10,
            'padding_scale': 1.2,
            'max_update_rate': 10.0,
            'filtered_cloud_topic': '/camera/filtered_cloud',
        }
        params.setdefault('octomap_frame', 'world')
        params.setdefault('octomap_resolution', 0.02)
    # Collision margin for PLANNED moves (the Cartesian grasp segments run unchecked anyway): the
    # robot's collision geometry is inflated by this much against the octomap / scene objects, so a
    # planned transfer no longer grazes the box or the bench (2026-09-09: "살짝씩 건드는게 있어").
    params['robot_description_planning.default_robot_padding'] = 0.03
    params['robot_description_planning.default_object_padding'] = 0.03
    params['robot_description_planning.default_attached_padding'] = 0.03

    pilz = os.path.join(
        get_package_share_directory('openarm_bimanual_moveit_config'),
        'config', cfg, 'pilz_cartesian_limits.yaml')
    if os.path.exists(pilz):
        with open(pilz) as f:
            data = yaml.safe_load(f)
        if data and 'cartesian_limits' in data:
            params.setdefault('robot_description_planning', {}).update(data)
    return params


def moveit_nodes(context, arm_type, use_fake, moveit_rviz, chest_xyz,
                 chest_rpy, solver, ik_timeout, octomap, cloud_topic,
                 wrist_octo, wrist_cloud_topic, chest_model):
    """move_group, and optionally MoveIt's own RViz alongside ours."""
    arm_type_str = context.perform_substitution(arm_type)
    params = _moveit_params(
        arm_type_str, context.perform_substitution(use_fake),
        context.perform_substitution(chest_xyz),
        context.perform_substitution(chest_rpy),
        chest_model=context.perform_substitution(chest_model),
        solver=context.perform_substitution(solver),
        ik_timeout=float(context.perform_substitution(ik_timeout)),
        octomap=context.perform_substitution(octomap) == 'true',
        cloud_topic=context.perform_substitution(cloud_topic),
        # Only when the wrist camera is actually started, so the updater
        # is not left waiting on a topic nobody publishes.
        # The wrist camera feeds the octomap ONLY when explicitly asked.
        # It rides on the hand, so during a pick it stares at the object it
        # is about to grasp and at the gripper itself, filling the approach
        # with obstacle voxels and churning the map every time the arm moves
        # - which invalidates plans mid-execution
        # (MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE).  It stays valuable
        # for grasp verification; that is a separate role.
        wrist_cloud_topic=(context.perform_substitution(wrist_cloud_topic)
                           if context.perform_substitution(wrist_octo) == 'true'
                           else ''))

    out = [Node(package='moveit_ros_move_group', executable='move_group',
                output='screen', parameters=[params])]

    if context.perform_substitution(moveit_rviz) == 'true':
        cfg = _config_dir(arm_type_str)
        rviz_cfg = os.path.join(
            get_package_share_directory('openarm_bimanual_moveit_config'),
            'config', cfg, 'moveit.rviz')
        out.append(Node(package='rviz2', executable='rviz2',
                        name='moveit_rviz', output='log',
                        arguments=['-d', rviz_cfg], parameters=[params]))
    return out


def demo_rviz(context, arm_type, use_fake, chest_xyz, chest_rpy, solver,
              ik_timeout, chest_model):
    """Our RViz, given the MoveIt parameters too.

    A bare rviz2 can show the robot, TF, clouds and markers, but the MoveIt
    displays - MotionPlanning, and the Trajectory preview that makes dry_run
    worth watching - need robot_description_semantic and the kinematics
    solver config.  Handing it the same parameter set costs nothing and lets
    those displays be added by hand without restarting anything.
    """
    # RViz needs the kinematics to match move_group's, so the MotionPlanning
    # panel drags the same solver; it has no use for the octomap keys.
    params = _moveit_params(
        context.perform_substitution(arm_type),
        context.perform_substitution(use_fake),
        context.perform_substitution(chest_xyz),
        context.perform_substitution(chest_rpy),
        chest_model=context.perform_substitution(chest_model),
        solver=context.perform_substitution(solver),
        ik_timeout=float(context.perform_substitution(ik_timeout)),
        octomap=False)
    return [Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', os.path.join(
            get_package_share_directory('openarm_vision_pick'), 'rviz',
            'openarm_hand.rviz')],
        parameters=[params])]


def generate_launch_description():
    arm_type = LaunchConfiguration('arm_type')
    use_fake = LaunchConfiguration('use_fake_hardware')
    left_can = LaunchConfiguration('left_can_interface')
    right_can = LaunchConfiguration('right_can_interface')
    moveit_rviz = LaunchConfiguration('moveit_rviz')
    solver = LaunchConfiguration('kinematics_solver')
    ik_timeout = LaunchConfiguration('ik_timeout')
    octomap = LaunchConfiguration('octomap')
    cloud_topic = LaunchConfiguration('pointcloud_topic')
    wrist_octo = LaunchConfiguration('wrist_octomap')
    wrist_cloud_topic = LaunchConfiguration('wrist_pointcloud_topic')
    chest_model = LaunchConfiguration('chest_camera_model')
    chest_xyz = LaunchConfiguration('chest_camera_xyz')
    chest_rpy = LaunchConfiguration('chest_camera_rpy')
    w = LaunchConfiguration('camera_width')
    h = LaunchConfiguration('camera_height')
    fps = LaunchConfiguration('camera_fps')

    args = [
        DeclareLaunchArgument(
            'arm_type', default_value='v1.0',
            description='v1.0 is the robot on this bench; the MoveIt '
                        "package's own default of v2.0 has no wrist camera."),
        DeclareLaunchArgument(
            'use_fake_hardware', default_value='true',
            description='true drives mock_components - no CAN needed and the '
                        'arm does not move. Set false for the real arm.'),
        DeclareLaunchArgument(
            'right_can_interface', default_value='can0',
            description='FOLLOWER right. Check with ~/openarm_link_check.sh: '
                        'the numbering moves when adapters are re-plugged.'),
        DeclareLaunchArgument(
            'left_can_interface', default_value='can1',
            description='FOLLOWER left. Check with ~/openarm_link_check.sh: '
                        'the numbering moves when adapters are re-plugged.'),
        DeclareLaunchArgument(
            'robot_controller', default_value='joint_trajectory_controller',
            choices=['joint_trajectory_controller',
                     'forward_position_controller']),

        DeclareLaunchArgument('camera', default_value='true',
                              description='start the chest D455f'),
        DeclareLaunchArgument('wrist_camera', default_value='false',
                              description='also start the wrist D405'),
        DeclareLaunchArgument('wrist_side', default_value='left',
                              description='which wrist carries the D405 being started: left | right'),
        DeclareLaunchArgument('wrist_serial', default_value='',
                              description="serial_no of that D405 as a string with a leading underscore, e.g. _260322270526 (empty = first free D405)"),
        DeclareLaunchArgument('marker_detector', default_value='false',
                              description='start chest_marker_detector (marker.yaml) here, for marker_pick_place'),
        DeclareLaunchArgument('detector', default_value='true',
                              description='start hand_detector'),
        DeclareLaunchArgument(
            'wrist_detector', default_value='false',
            description='start a marker_detector on the wrist camera, as '
                        '/wrist_marker_detector - what marker_pick_place '
                        'uses to confirm the object is in the hand. Needs '
                        'wrist_camera:=true.'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='open RViz with the demo config'),
        DeclareLaunchArgument(
            'moveit_rviz', default_value='false',
            description="also open MoveIt's own RViz with its planning panel"),

        DeclareLaunchArgument(
            'kinematics_solver', default_value='pick_ik',
            choices=['pick_ik', 'kdl'],
            description="pick_ik (MoveIt 2's recommended global solver) or "
                        'the KDL the package shipped with. KDL at 5 ms found '
                        'a 20 cm strip of the bench reachable; the arm can do '
                        'far better.'),
        DeclareLaunchArgument('ik_timeout', default_value='0.05'),
        DeclareLaunchArgument(
            'octomap', default_value='true',
            description='Build an occupancy map from the chest camera so '
                        'plans avoid the bench, boxes, and anything else it '
                        'sees. Without it only self-collision is checked.'),
        DeclareLaunchArgument(
            'pointcloud_topic',
            default_value='/chest/openarm_chest_camera/depth/color/points'),
        DeclareLaunchArgument(
            'wrist_octomap', default_value='false',
            description='Feed the wrist D405 cloud into the octomap. OFF by '
                        'default: an eye-in-hand camera sees the object it is '
                        'grasping and the gripper as obstacles, and churns '
                        'the map as the arm moves, aborting plans with '
                        'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE. The '
                        'wrist camera is still used for grasp verification '
                        'independently of this.'),
        DeclareLaunchArgument(
            'wrist_pointcloud_topic',
            default_value='/camera/camera/depth/color/points'),
        DeclareLaunchArgument(
            'chest_camera_model', default_value='d455',
            choices=['d455', 'd435'],
            description='The chest RealSense. The description package '
                        'assumes a D435; this robot carries a D455, whose '
                        'colour optical frame sits 74 mm from where the '
                        'D435 macro would put it.'),
        DeclareLaunchArgument(
            'chest_camera_xyz', default_value=CHEST_CAMERA_XYZ,
            description='chest camera mount position on openarm_body_link0'),
        DeclareLaunchArgument(
            'chest_camera_rpy', default_value=CHEST_CAMERA_RPY,
            description='chest camera mount orientation. The description '
                        "package's 3.141593 roll is a 180 deg flip about the "
                        'optical axis and is wrong on this robot - measured '
                        'against the floor plane. See the note at the top.'),

        DeclareLaunchArgument('camera_width', default_value='640'),
        DeclareLaunchArgument('camera_height', default_value='480'),
        DeclareLaunchArgument('camera_fps', default_value='30'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('openarm_vision_pick'), 'config',
                 'chest.yaml']),
            description='hand_detector and reach_to_hand parameters'),
    ]

    robot = OpaqueFunction(
        function=robot_nodes,
        args=[arm_type, use_fake, right_can, left_can, chest_xyz, chest_rpy,
              chest_model])

    move_group = TimerAction(period=4.0, actions=[
        OpaqueFunction(function=moveit_nodes,
                       args=[arm_type, use_fake, moveit_rviz, chest_xyz,
                             chest_rpy, solver, ik_timeout, octomap,
                             cloud_topic, wrist_octo, wrist_cloud_topic,
                             chest_model])])

    jsb = TimerAction(period=3.0, actions=[Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '-c', '/controller_manager'])])

    arms = TimerAction(period=4.0, actions=[Node(
        package='controller_manager', executable='spawner',
        arguments=['left_joint_trajectory_controller',
                   'right_joint_trajectory_controller',
                   '-c', '/controller_manager'])])

    grippers = TimerAction(period=4.0, actions=[Node(
        package='controller_manager', executable='spawner',
        arguments=['left_gripper_controller', 'right_gripper_controller',
                   '-c', '/controller_manager'])])

    # namespace and node name are set here rather than through
    # camera_namespace, which is a launch-file argument in the RealSense
    # package and is ignored when passed as a parameter.  camera_name stays a
    # parameter: it is what names the TF frames that tie the images into the
    # robot model.
    chest = Node(
        package='realsense2_camera', executable='realsense2_camera_node',
        namespace='chest', name='openarm_chest_camera', output='screen',
        condition=IfCondition(LaunchConfiguration('camera')),
        parameters=[{
            # One D455f and one D405 on this robot, so the model is the only
            # key that survives a re-plug into any port.  The two serials
            # disagree - librealsense wants 419122302733, the USB descriptor
            # says 254643066593 - and the D405 has no USB serial at all.
            'device_type': 'd455',
            'camera_name': 'openarm_chest_camera',
            'align_depth.enable': True,
            'depth_module.depth_units': 0.001,
            'rgb_camera.color_profile': [w, 'x', h, 'x', fps],
            'depth_module.depth_profile': [w, 'x', h, 'x', fps],
            'enable_infra1': False, 'enable_infra2': False,
            'enable_gyro': False, 'enable_accel': False,
            'pointcloud.enable': True,
        }])

    # The wrist D405, off by default.  It sits on a USB 2 port where 848x480
    # is capped at 10 fps, so it runs at 640x480 where the link allows 30.
    # Its colour profile lives under depth_module, not rgb_camera: the D405
    # has one sensor module that produces colour and depth both, and there is
    # no rgb_camera group at all - passing rgb_camera.profile is ignored in
    # silence, which is what kept it at 10 fps.
    wrist_side = LaunchConfiguration('wrist_side')
    is_right = PythonExpression(["'", wrist_side, "' == 'right'"])
    is_left = PythonExpression(["'", wrist_side, "' != 'right'"])
    wrist = Node(
        package='realsense2_camera', executable='realsense2_camera_node',
        namespace='camera', name='camera', output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('wrist_camera'), "' == 'true' and '", wrist_side, "' != 'right'"])),
        parameters=[{
            'device_type': 'd405',
            'serial_no': LaunchConfiguration('wrist_serial'),
            'camera_name': 'openarm_left_camera',
            'align_depth.enable': True,
            'depth_module.depth_units': 0.001,
            'depth_module.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
            'enable_infra1': False, 'enable_infra2': False,
            # For the octomap.  Costs some CPU on the camera node; on the
            # USB 2 port this camera is already running at 20 Hz rather
            # than 30, so watch the rate if it matters.
            'pointcloud.enable': True,
        }])

    # RIGHT wrist twin: camera_name names the TF frames, so it must be the right camera's.
    wrist_right = Node(
        package='realsense2_camera', executable='realsense2_camera_node',
        namespace='camera', name='camera', output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('wrist_camera'), "' == 'true' and '", wrist_side, "' == 'right'"])),
        parameters=[{
            'device_type': 'd405',
            'serial_no': LaunchConfiguration('wrist_serial'),
            'camera_name': 'openarm_right_camera',
            'align_depth.enable': True,
            'depth_module.depth_units': 0.001,
            # USB 2.1 port: 640x480x30 colour+depth+pointcloud starved the link and NO frames came
            # out (2026-09-09); 15 fps without the point cloud is all the wrist detector needs
            'depth_module.color_profile': '640x480x15',
            'depth_module.depth_profile': '640x480x15',
            'enable_infra1': False, 'enable_infra2': False,
            'pointcloud.enable': False,
        }])
    # marker.yaml's wrist section names the LEFT camera's topics/frame; override for the right
    # The realsense node names its topics after the NODE name (/camera/camera/...), not after
    # camera_name (which only names the TF frames) - so the topics are marker.yaml's defaults and
    # only the optical frame differs for the right wrist.
    wrist_right_overrides = {
        'color_topic': '/camera/camera/color/image_raw',
        'info_topic': '/camera/camera/color/camera_info',
        'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
        'camera_frame': 'openarm_right_camera_color_optical_frame',
    }
    chest_marker = TimerAction(period=8.0, actions=[Node(
        package='openarm_vision_pick', executable='marker_detector',
        name='chest_marker_detector', output='screen',
        condition=IfCondition(LaunchConfiguration('marker_detector')),
        parameters=[PathJoinSubstitution(
            [FindPackageShare('openarm_vision_pick'), 'config',
             'marker.yaml'])])])

    # Started late: it needs the camera streaming and, because it publishes in
    # `world`, robot_state_publisher up as well.  Without that TF it says so
    # and stays quiet rather than publishing camera-frame numbers under a
    # world label.
    detector = TimerAction(period=8.0, actions=[Node(
        package='openarm_vision_pick', executable='hand_detector',
        name='hand_detector', output='screen',
        condition=IfCondition(LaunchConfiguration('detector')),
        parameters=[LaunchConfiguration('params_file')])])

    # The wrist detector publishes the object's tag as the WRIST camera sees
    # it - the check that the grasp took.  Same node as the chest detector,
    # the wrist profile from marker.yaml, poses in world so they can be
    # compared with the TCP.
    wrist_detector = TimerAction(period=8.0, actions=[
        Node(package='openarm_vision_pick', executable='marker_detector',
             name='wrist_marker_detector', output='screen',
             condition=IfCondition(PythonExpression(["'", LaunchConfiguration('wrist_detector'), "' == 'true' and '", wrist_side, "' != 'right'"])),
             parameters=[PathJoinSubstitution(
                 [FindPackageShare('openarm_vision_pick'), 'config',
                  'marker.yaml'])]),
        Node(package='openarm_vision_pick', executable='marker_detector',
             name='wrist_marker_detector', output='screen',
             condition=IfCondition(PythonExpression(["'", LaunchConfiguration('wrist_detector'), "' == 'true' and '", wrist_side, "' == 'right'"])),
             parameters=[PathJoinSubstitution(
                 [FindPackageShare('openarm_vision_pick'), 'config',
                  'marker.yaml']), wrist_right_overrides]),
    ])

    rviz = TimerAction(period=6.0, actions=[
        OpaqueFunction(function=demo_rviz,
                       args=[arm_type, use_fake, chest_xyz, chest_rpy, solver,
                             ik_timeout, chest_model],
                       condition=IfCondition(LaunchConfiguration('rviz')))])

    return LaunchDescription(
        args + [robot, jsb, arms, grippers, move_group, chest, wrist, wrist_right,
                chest_marker, detector, wrist_detector, rviz])
