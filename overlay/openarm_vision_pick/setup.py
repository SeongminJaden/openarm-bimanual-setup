from glob import glob

from setuptools import setup

package_name = 'openarm_vision_pick'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name + '/assets', glob('assets/aruco*')),

        ('share/' + package_name + '/docs', glob('docs/*.md')),

        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf*')),
        ('share/' + package_name + '/textures', glob('textures/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Seongmin',
    maintainer_email='roboticsmaster@naver.com',
    description='Wrist-camera object detection and MoveIt pick-and-place for OpenArm.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'joint_state_readonly = openarm_vision_pick.joint_state_readonly:main',
            'chest_calibration = openarm_vision_pick.chest_calibration:main',

            'object_detector = openarm_vision_pick.object_detector:main',
            'pen_detector = openarm_vision_pick.pen_detector:main',
            'pen_center = openarm_vision_pick.pen_center:main',
            'hand_detector = openarm_vision_pick.hand_detector:main',
            'reach_to_hand = openarm_vision_pick.reach_to_hand:main',
            'make_markers = openarm_vision_pick.make_markers:main',
            'marker_detector = openarm_vision_pick.marker_detector:main',
            'camera_adjust = openarm_vision_pick.camera_adjust:main',
            'marker_pick_place = openarm_vision_pick.marker_pick_place:main',
            'reach_map = openarm_vision_pick.reach_map:main',
            'reach_overlay = openarm_vision_pick.reach_overlay:main',
            'pick_and_place = openarm_vision_pick.pick_and_place:main',
            'frame_relay = openarm_vision_pick.frame_relay:main',
        ],
    },
)
