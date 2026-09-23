"""The marker pick in Ignition Gazebo (Fortress), on the same ROS wiring as the real robot.

Brings up: the world (stand, white 50x140x50 box with ArUco tag 1 on top, tag 0 as the destination),
the bimanual robot under gz_ros2_control with the real controller names, simulated RGB-D on the
chest and the right wrist bridged onto the real camera topics, TF for the camera optical frames,
move_group with the hand_demo parameter set (pick_ik, octomap from the chest cloud, 3 cm padding),
and both marker detectors.  The pick itself is started separately, exactly as on the robot:

    ros2 launch openarm_vision_pick gazebo_pick.launch.py gui:=true
    ros2 run openarm_vision_pick marker_pick_place --ros-args --params-file <share>/config/marker_pick_gz.yaml ...
"""
import importlib.util
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

CHEST_XYZ = '0.0290 0.0000 0.6274'
CHEST_RPY = '0.0 0.785398 0.0'


def _hand_demo():
    """hand_demo.launch.py as a module, for its MoveIt parameter builder."""
    path = os.path.join(get_package_share_directory('openarm_vision_pick'), 'launch', 'hand_demo.launch.py')
    spec = importlib.util.spec_from_file_location('hand_demo_launch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world_file(context):
    share = get_package_share_directory('openarm_vision_pick')
    tex = os.path.join(share, 'textures')
    src = os.path.join(share, 'worlds', 'openarm_pick.sdf.in')
    out = '/tmp/openarm_pick.sdf'
    text = open(src).read().replace('@TEX@', 'file://' + tex)
    text = text.replace('@TAG_YAW@', context.perform_substitution(LaunchConfiguration('tag_yaw')))
    open(out, 'w').write(text)
    return out


def setup(context, *args, **kwargs):
    gui = context.perform_substitution(LaunchConfiguration('gui')) == 'true'
    rviz = context.perform_substitution(LaunchConfiguration('rviz')) == 'true'
    world = _world_file(context)

    rl_share = get_package_share_directory('openarm_rl_deploy')
    xacro_file = os.path.join(rl_share, 'urdf', 'openarm_gz.urdf.xacro')
    controllers = os.path.join(rl_share, 'config', 'gz_controllers.yaml')
    robot_description = xacro.process_file(xacro_file, mappings={
        'bimanual': 'true', 'ros2_control': 'false', 'controllers_file': controllers,
        'camera_width': '640', 'camera_height': '480', 'camera_rate': '15.0',
        # the chest like the D455 on the robot: ~87 deg horizontal at 1280x720 (69 deg missed a tag at y=-0.25)
        'chest_hfov': '1.518', 'chest_width': '1280', 'chest_height': '720',
        'chest_camera_xyz': CHEST_XYZ, 'chest_camera_rpy': CHEST_RPY,
    }).toprettyxml(indent='  ')
    from openarm_rl_deploy.mirror_meshes import fix_negative_scales
    robot_description, n_fixed = fix_negative_scales(robot_description)

    sim = {'use_sim_time': True}
    gz_args = [world, ' -r -v2'] if gui else [world, ' -r -s --headless-rendering -v2']
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments={'gz_args': gz_args}.items())

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher', output='log',
               parameters=[{'robot_description': robot_description}, sim])
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-topic', 'robot_description', '-name', 'openarm', '-z', '0.0'])

    # Gazebo -> ROS.  The rgbd_camera publishes <topic>/image, /depth_image (32FC1 metres, which the
    # detectors accept), /camera_info and /points.
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/chest_cam/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/chest_cam/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/chest_cam/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/chest_cam/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
            '/right_wrist_cam/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/right_wrist_cam/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/right_wrist_cam/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        remappings=[
            ('/chest_cam/image', '/chest/openarm_chest_camera/color/image_raw'),
            ('/chest_cam/depth_image', '/chest/openarm_chest_camera/aligned_depth_to_color/image_raw'),
            ('/chest_cam/camera_info', '/chest/openarm_chest_camera/color/camera_info'),
            ('/chest_cam/points', '/gz/chest_points'),
            ('/right_wrist_cam/image', '/camera/camera/color/image_raw'),
            ('/right_wrist_cam/depth_image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('/right_wrist_cam/camera_info', '/camera/camera/color/camera_info'),
        ],
        parameters=[sim])

    # The Gazebo cloud carries Gazebo's own frame name; give it the robot's camera link for the octomap.
    relay = Node(package='openarm_vision_pick', executable='frame_relay', name='chest_cloud_relay',
                 output='screen',
                 parameters=[{'in_topic': '/gz/chest_points',
                              'out_topic': '/chest/openarm_chest_camera/depth/color/points',
                              'frame_id': LaunchConfiguration('cloud_frame')}, sim])

    # The RealSense driver publishes the *_color_optical_frame TFs on the real robot; here the images
    # are rendered from *_camera_link (x forward), so the optical frame is that link turned optical.
    def optical(parent, child):
        return Node(package='tf2_ros', executable='static_transform_publisher', output='log',
                    arguments=['--x', '0', '--y', '0', '--z', '0', '--roll', '-1.5707963',
                               '--pitch', '0', '--yaw', '-1.5707963',
                               '--frame-id', parent, '--child-frame-id', child],
                    parameters=[sim])
    tfs = [optical('openarm_chest_camera_link', 'openarm_chest_camera_color_optical_frame'),
           optical('openarm_right_camera_link', 'openarm_right_camera_color_optical_frame')]

    spawners = [TimerAction(period=8.0, actions=[
        Node(package='controller_manager', executable='spawner', output='screen', arguments=[name])])
        for name in ('joint_state_broadcaster', 'right_joint_trajectory_controller',
                     'left_joint_trajectory_controller', 'right_gripper_controller',
                     'left_gripper_controller')]

    hd = _hand_demo()
    params = hd._moveit_params('v1.0', 'true', CHEST_XYZ, CHEST_RPY, chest_model='d455',
                               solver='pick_ik', ik_timeout=0.05, octomap=True,
                               cloud_topic='/chest/openarm_chest_camera/depth/color/points',
                               wrist_cloud_topic='')
    params['use_sim_time'] = True
    move_group = TimerAction(period=10.0, actions=[
        Node(package='moveit_ros_move_group', executable='move_group', output='screen',
             parameters=[params])])

    marker_yaml = os.path.join(get_package_share_directory('openarm_vision_pick'), 'config', 'marker.yaml')
    wrist_overrides = {
        'color_topic': '/camera/camera/color/image_raw',
        'info_topic': '/camera/camera/color/camera_info',
        'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
        'camera_frame': 'openarm_right_camera_color_optical_frame',
    }
    detectors = TimerAction(period=12.0, actions=[
        Node(package='openarm_vision_pick', executable='marker_detector', name='chest_marker_detector',
             output='screen', parameters=[marker_yaml, sim]),
        Node(package='openarm_vision_pick', executable='marker_detector', name='wrist_marker_detector',
             output='screen', parameters=[marker_yaml, wrist_overrides, sim]),
    ])

    out = [gz, rsp, spawn, bridge, relay] + tfs + spawners + [move_group, detectors]
    if rviz:
        cfg = os.path.join(get_package_share_directory('openarm_bimanual_moveit_config'),
                           'config', 'openarm_v1.0', 'moveit.rviz')
        out.append(TimerAction(period=14.0, actions=[
            Node(package='rviz2', executable='rviz2', name='moveit_rviz', output='log',
                 arguments=['-d', cfg], parameters=[params])]))
    return out


def generate_launch_description():
    desc_parent = os.path.dirname(get_package_share_directory('openarm_description'))
    resource_path = SetEnvironmentVariable(
        'IGN_GAZEBO_RESOURCE_PATH',
        os.pathsep.join([desc_parent, os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')]).rstrip(os.pathsep))
    args = [
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo GUI (false = headless)'),
        DeclareLaunchArgument('rviz', default_value='false', description="MoveIt's RViz"),
        DeclareLaunchArgument('tag_yaw', default_value='0', description='yaw of the tag plate on the box (rad)'),
        DeclareLaunchArgument('cloud_frame', default_value='openarm_chest_camera_link',
                              description='TF frame the Gazebo chest cloud is expressed in'),
    ]
    return LaunchDescription([resource_path] + args + [OpaqueFunction(function=setup)])
