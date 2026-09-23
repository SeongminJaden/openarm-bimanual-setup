"""Run from package root: python3 test/test_chest_calibration.py (no hardware)."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_vision_pick.chest_calibration import pose, solve, marker_pose


def main():
    rng = np.random.default_rng(20260923)
    camera = pose(Rotation.from_euler('xyz', [2.9, .8, .04]).as_matrix(), [.06, -.01, .65])
    marker = pose(Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix(), [.02, .03, .08])
    internal = pose(Rotation.from_euler('xyz', [-1.57, 0, -1.57]).as_matrix(), [.01, .02, .012])
    samples = []
    for i in range(25):
        hand = pose(Rotation.from_rotvec(rng.normal(0, .6, 3)).as_matrix(), rng.uniform(-.3, .3, 3))
        observed = np.linalg.inv(camera) @ hand @ marker
        samples.append(dict(base_hand=hand.tolist(), camera_marker=observed.tolist()))
    result = solve(samples, internal)
    assert result['accepted'], result
    assert np.allclose(result['base_camera'], camera, atol=1e-7)
    assert np.allclose(result['hand_marker'], marker, atol=1e-7)
    assert np.allclose(result['base_mount'], camera @ np.linalg.inv(internal), atol=1e-7)
    assert result['validation']['max_position_m'] < 1e-7
    # A bad independent observation must not pass the quality gate.
    samples[4]['camera_marker'][0][3] += .1
    assert not solve(samples, internal)['accepted']
    try:
        solve([samples[0]] * 15, internal)
    except ValueError:
        pass
    else:
        raise AssertionError('Repeated poses must be rejected')
    # Exercise actual planar PnP with the same corner ordering as ArUco.
    import cv2
    k = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
    corners3 = np.array([[-.03, .03, 0], [.03, .03, 0], [.03, -.03, 0], [-.03, -.03, 0]])
    rotation = np.array([2.7, .3, .15])
    corners2, _ = cv2.projectPoints(corners3, rotation, np.array([.04, .02, .5]), k, np.zeros(5))
    found, error = marker_pose(corners2, .06, k, np.zeros(5), 1., .05)
    assert np.allclose(found, pose(cv2.Rodrigues(rotation)[0], [.04, .02, .5]), atol=1e-6)
    assert error < 1e-5
    print('PASS: eye-to-hand, mount conversion, held-out rejection, degenerate poses, planar PnP')


def ros_check():
    """Exercise ROS callbacks with synthetic image/TF, without a DDS network."""
    import os
    import tempfile
    import json
    import cv2
    os.environ['ROS_DOMAIN_ID'] = '213'
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CameraInfo
    from std_srvs.srv import Trigger
    from rclpy.duration import Duration
    from openarm_vision_pick.chest_calibration import create_node
    with tempfile.TemporaryDirectory() as folder:
        rclpy.init(args=['--ros-args', '-p', 'marker_size_m:=0.06', '-p', f'output_dir:={folder}'])
        node = create_node()
        try:
            stamp = node.get_clock().now() - Duration(seconds=.1)
            def tf(parent, child, when):
                msg = TransformStamped()
                msg.header.stamp = when.to_msg()
                msg.header.frame_id, msg.child_frame_id = parent, child
                msg.transform.rotation.w = 1.
                return msg
            node.buffer.set_transform_static(tf('openarm_body_link0', 'openarm_chest_camera_bottom_screw_frame', stamp), 'test')
            node.buffer.set_transform_static(tf('openarm_chest_camera_bottom_screw_frame', 'openarm_chest_camera_color_optical_frame', stamp), 'test')
            for when in (stamp - Duration(seconds=1.), stamp):
                node.buffer.set_transform(tf('openarm_body_link0', 'openarm_right_hand', when), 'test')
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            marker = (cv2.aruco.generateImageMarker(dictionary, 10, 300) if hasattr(cv2.aruco, 'generateImageMarker')
                      else cv2.aruco.drawMarker(dictionary, 10, 300))
            k = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
            obj = np.array([[-.03, .03, 0], [.03, .03, 0], [.03, -.03, 0], [-.03, -.03, 0]])
            corners, _ = cv2.projectPoints(obj, np.array([2.7, .3, .15]), np.array([.04, .02, .5]), k, np.zeros(5))
            warp = cv2.getPerspectiveTransform(np.float32([[0, 0], [299, 0], [299, 299], [0, 299]]), corners.reshape(4, 2).astype(np.float32))
            frame = cv2.warpPerspective(marker, warp, (640, 480), borderValue=255)
            msg = CvBridge().cv2_to_imgmsg(frame, encoding='mono8')
            msg.header.stamp = stamp.to_msg()
            msg.header.frame_id = 'openarm_chest_camera_color_optical_frame'
            info = CameraInfo()
            info.header = msg.header
            info.width, info.height = 640, 480
            info.k, info.d, info.distortion_model = k.ravel().tolist(), [0.] * 5, 'plumb_bob'
            node.on_info(info)
            node.on_image(msg)
            response = node.capture(Trigger.Request(), Trigger.Response())
            assert response.success, response.message
            response = node.capture(Trigger.Request(), Trigger.Response())
            assert not response.success and 'Already captured' in response.message
            saved = json.loads(next(Path(folder).glob('*/samples.json')).read_text())
            assert len(saved['samples']) == 1
            response = node.finish(Trigger.Request(), Trigger.Response())
            assert not response.success and '15' in response.message
            msg.header.frame_id = 'wrong_camera'
            response = node.capture(Trigger.Request(), Trigger.Response())
            assert not response.success and 'frame' in response.message
            print('PASS: ROS callbacks, image/TF capture, persistence, duplicate/frame/insufficient-data rejection')
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
    if '--ros' in sys.argv:
        ros_check()
