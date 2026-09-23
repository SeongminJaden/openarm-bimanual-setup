# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Detector plus pick-and-place, parameterised for simulation or hardware.

  ros2 launch openarm_vision_pick vision_pick.launch.py            # sim, dry run
  ros2 launch openarm_vision_pick vision_pick.launch.py dry_run:=false
  ros2 launch openarm_vision_pick vision_pick.launch.py profile:=hardware

Assumes move_group and the controllers are already up.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('openarm_vision_pick')

    profile = LaunchConfiguration('profile')
    dry_run = LaunchConfiguration('dry_run')
    use_sim_time = LaunchConfiguration('use_sim_time')
    run_pick = LaunchConfiguration('run_pick')

    args = [
        DeclareLaunchArgument('profile', default_value='sim',
                              choices=['sim', 'hardware', 'black_bench'],
                              description='Which parameter file to load.'),
        DeclareLaunchArgument('dry_run', default_value='true',
                              description='Plan and display only; do not move.'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('run_pick', default_value='true',
                              description='Set false to run the detector alone.'),
    ]

    params = PathJoinSubstitution([pkg, 'config', [profile, '.yaml']])

    detector = Node(
        package='openarm_vision_pick',
        executable='object_detector',
        name='object_detector',
        output='screen',
        parameters=[params, {'use_sim_time': use_sim_time}],
    )

    pick = Node(
        package='openarm_vision_pick',
        executable='pick_and_place',
        name='pick_and_place',
        output='screen',
        condition=IfCondition(run_pick),
        parameters=[params, {'use_sim_time': use_sim_time,
                             'dry_run': dry_run}],
    )

    return LaunchDescription(args + [detector, pick])
