"""Fixed chest camera / moving arm marker calibration. T_a_b maps b into a.

base_hand @ hand_marker = base_camera @ camera_marker.
No robot commands, TF publication, or URDF writes are performed.
"""
import json
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def pose(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def checked_pose(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1])
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-5)):
        raise ValueError('Invalid rigid transform')
    return value


def difference(a, b):
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())))


def solve(samples, mount_camera, max_position_m=.01, max_angle_deg=2.):
    """Reserve every fifth sample for validation; never refit on those samples."""
    if not all(np.isfinite(v) and v > 0 for v in (max_position_m, max_angle_deg)):
        raise ValueError('Error limits must be positive and finite')
    if len(samples) < 15:
        raise ValueError('Collect at least 15 distinct poses (20-30 recommended)')
    hands = [checked_pose(s['base_hand']) for s in samples]
    markers = [checked_pose(s['camera_marker']) for s in samples]
    train = [i for i in range(len(samples)) if i % 5 != 4]
    validation = [i for i in range(len(samples)) if i % 5 == 4]
    motions = [Rotation.from_matrix(hands[train[0]][:3, :3].T @ hands[i][:3, :3]).as_rotvec()
               for i in train[1:]]
    # ponytail: simple excitation gate; use an uncertainty model for metrology-grade work.
    singular = np.linalg.svd(motions, compute_uv=False)
    if singular[1] < .25 or singular[1] / max(singular[0], 1e-12) < .1:
        raise ValueError('Insufficient rotation diversity: tilt the hand around at least two axes')
    inverse_hands = [np.linalg.inv(hands[i]) for i in train]
    r, t = cv2.calibrateHandEye(
        [h[:3, :3] for h in inverse_hands], [h[:3, 3] for h in inverse_hands],
        [markers[i][:3, :3] for i in train], [markers[i][:3, 3] for i in train],
        method=cv2.CALIB_HAND_EYE_PARK)
    camera = checked_pose(pose(r, t))
    attached = [np.linalg.inv(hands[i]) @ camera @ markers[i] for i in train]
    marker = pose(Rotation.from_matrix(np.array([a[:3, :3] for a in attached])).mean().as_matrix(),
                  np.mean([a[:3, 3] for a in attached], axis=0))
    mount = camera @ np.linalg.inv(checked_pose(mount_camera))

    def errors(indices):
        values = np.array([difference(hands[i] @ marker, camera @ markers[i]) for i in indices])
        return dict(count=len(indices), rms_position_m=float(np.sqrt(np.mean(values[:, 0] ** 2))),
                    max_position_m=float(values[:, 0].max()), max_angle_deg=float(values[:, 1].max()))

    fit, check = errors(train), errors(validation)
    accepted = all(e['max_position_m'] <= max_position_m and e['max_angle_deg'] <= max_angle_deg
                   for e in (fit, check))
    return dict(accepted=accepted, base_camera=camera.tolist(), hand_marker=marker.tolist(),
                base_mount=mount.tolist(), chest_camera_xyz=mount[:3, 3].tolist(),
                chest_camera_rpy=Rotation.from_matrix(mount[:3, :3]).as_euler('xyz').tolist(),
                training=fit, validation=check,
                limits=dict(max_position_m=max_position_m, max_angle_deg=max_angle_deg))


def marker_pose(corners, size, k, distortion, max_error, ambiguity_margin):
    """IPPE square corners: top-left, top-right, bottom-right, bottom-left."""
    half = size / 2
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    corners = np.asarray(corners, dtype=float).reshape(4, 2)
    candidates = cv2.solvePnPGeneric(obj, corners, k, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    choices = []
    for r, t in zip(candidates[1], candidates[2]):
        transform = pose(cv2.Rodrigues(r)[0], t)
        if np.min((obj @ transform[:3, :3].T + transform[:3, 3])[:, 2]) <= 0:
            continue
        projected, _ = cv2.projectPoints(obj, r, t, k, distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1))))
        if np.isfinite(transform).all() and np.isfinite(error):
            choices.append((error, transform))
    choices.sort(key=lambda item: item[0])
    if not choices or choices[0][0] > max_error:
        raise ValueError('Marker pose reprojection error too large / no valid pose')
    if len(choices) > 1 and choices[1][0] - choices[0][0] < ambiguity_margin:
        raise ValueError('Planar pose is ambiguous; tilt the marker toward another angle')
    return choices[0][1], choices[0][0]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def create_node():
    import tf2_ros
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.duration import Duration
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from std_srvs.srv import Trigger
    from cv_bridge import CvBridge
    from cv_bridge import CvBridgeError

    class Calibration(Node):
        def __init__(self):
            super().__init__('chest_calibration')
            defaults = dict(
                color_topic='/chest/openarm_chest_camera/color/image_raw',
                info_topic='/chest/openarm_chest_camera/color/camera_info',
                base_frame='openarm_body_link0', hand_frame='openarm_right_hand',
                mount_frame='openarm_chest_camera_bottom_screw_frame',
                optical_frame='openarm_chest_camera_color_optical_frame',
                dictionary='DICT_4X4_50', marker_id=10, marker_size_m=0.,
                output_dir='~/chest_calibration', max_image_age_s=1.,
                max_reprojection_px=1., ambiguity_margin_px=.15,
                max_position_error_m=.01, max_angle_error_deg=2.)
            self.p = {key: self.declare_parameter(key, value).value for key, value in defaults.items()}
            if not np.isfinite(self.p['marker_size_m']) or self.p['marker_size_m'] <= 0:
                raise ValueError('Set marker_size_m to the measured BLACK square width in metres')
            for name in ('max_image_age_s', 'max_reprojection_px', 'ambiguity_margin_px',
                         'max_position_error_m', 'max_angle_error_deg'):
                if not np.isfinite(self.p[name]) or self.p[name] <= 0:
                    raise ValueError(f'{name} must be positive and finite')
            if not self.p['dictionary'].startswith('DICT_') or not hasattr(cv2.aruco, self.p['dictionary']):
                raise ValueError('Unknown ArUco dictionary')
            self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.p['dictionary']))
            if not 0 <= self.p['marker_id'] < len(self.dictionary.bytesList):
                raise ValueError('marker_id outside dictionary range')
            self.parameters = (cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, 'DetectorParameters_create')
                               else cv2.aruco.DetectorParameters())
            self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = (cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
                             if hasattr(cv2.aruco, 'ArucoDetector') else None)
            self.folder = Path(self.p['output_dir']).expanduser() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.folder.mkdir(parents=True, exist_ok=False)
            self.bridge, self.image, self.info = CvBridge(), None, None
            self.images = deque(maxlen=60)
            self.samples, self.internal, self.original = [], None, None
            self.buffer = tf2_ros.Buffer()
            self.listener = tf2_ros.TransformListener(self.buffer, self)
            self.create_subscription(Image, self.p['color_topic'], self.on_image, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, self.p['info_topic'], self.on_info, qos_profile_sensor_data)
            self.create_service(Trigger, '~/capture', self.capture)
            self.create_service(Trigger, '~/solve', self.finish)
            self.get_logger().info(f'Stop arm before capture. Output: {self.folder}')

        def on_image(self, message):
            if self.image is None:
                self.get_logger().info('Receiving color images')
            self.image = message
            self.images.append(message)

        def on_info(self, message):
            self.info = message

        def transform(self, target, source, stamp):
            t = self.buffer.lookup_transform(target, source, stamp).transform
            q = t.rotation
            return pose(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                        [t.translation.x, t.translation.y, t.translation.z])

        def capture(self, request, response):
            self.get_logger().info('Capture requested')
            try:
                image, info, p = self.image, self.info, self.p
                if image is None or info is None:
                    raise ValueError('Waiting for color image and CameraInfo; check configured topics')
                # Allow encoder TF to arrive before looking up the image's exact timestamp.
                now = self.get_clock().now()
                candidates = [m for m in self.images
                              if (now - Time.from_msg(m.header.stamp)).nanoseconds >= 80_000_000]
                if not candidates:
                    raise ValueError('Waiting for 80 ms of image/TF history; try capture again')
                image = candidates[-1]
                if image.header.frame_id != p['optical_frame'] or info.header.frame_id != p['optical_frame']:
                    raise ValueError('Image/CameraInfo optical frame does not match optical_frame parameter')
                stamp = Time.from_msg(image.header.stamp)
                age = (self.get_clock().now() - stamp).nanoseconds / 1e9
                if stamp.nanoseconds == 0 or not 0 <= age <= p['max_image_age_s']:
                    raise ValueError('Image timestamp is zero, stale, or in the future')
                if self.samples and self.samples[-1]['stamp_ns'] == stamp.nanoseconds:
                    raise ValueError('Already captured this image')
                if (image.width, image.height) != (info.width, info.height):
                    raise ValueError('Image and CameraInfo resolutions differ')
                if info.distortion_model not in ('plumb_bob', 'rational_polynomial'):
                    raise ValueError('Use raw color with OpenCV-compatible distortion (plumb_bob/rational_polynomial)')
                k, d = np.asarray(info.k).reshape(3, 3), np.asarray(info.d)
                if (not np.isfinite(k).all() or not np.isfinite(d).all() or k[0, 0] <= 0
                        or k[1, 1] <= 0 or len(d) not in (4, 5, 8, 12, 14)):
                    raise ValueError('Invalid CameraInfo intrinsics/distortion')
                hand = self.transform(p['base_frame'], p['hand_frame'], stamp)
                earlier = self.transform(p['base_frame'], p['hand_frame'], stamp - Duration(seconds=.5))
                distance, angle = difference(hand, earlier)
                if distance > .001 or angle > .5:
                    raise ValueError('Arm moved over the last 0.5 s; stop and wait before capture')
                for sample in self.samples:
                    distance, angle = difference(hand, np.asarray(sample['base_hand']))
                    if distance < .005 and angle < 3.:
                        raise ValueError('Pose too similar to an existing sample; move/rotate the arm')
                gray = self.bridge.imgmsg_to_cv2(image, desired_encoding='mono8')
                corners, ids, _ = (self.detector.detectMarkers(gray) if self.detector else
                                   cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters))
                matches = [] if ids is None else np.flatnonzero(ids.ravel() == p['marker_id'])
                if len(matches) != 1:
                    raise ValueError(f'Need exactly one visible marker ID {p["marker_id"]}')
                camera_marker, error = marker_pose(corners[matches[0]], p['marker_size_m'], k, d,
                                                  p['max_reprojection_px'], p['ambiguity_margin_px'])
                internal = self.transform(p['mount_frame'], p['optical_frame'], stamp)
                original = self.transform(p['base_frame'], p['mount_frame'], stamp)
                if self.internal is not None and (not np.allclose(internal, self.internal, atol=1e-6)
                                                  or not np.allclose(original, self.original, atol=1e-6)):
                    raise ValueError('Camera TF changed during this session; start a new session')
                sample = dict(stamp_ns=stamp.nanoseconds, base_hand=hand.tolist(),
                              camera_marker=camera_marker.tolist(), reprojection_px=error,
                              corners_px=np.asarray(corners[matches[0]]).reshape(4, 2).tolist(),
                              camera_matrix=k.tolist(), distortion=d.tolist())
                # Write before changing memory: a failed disk write cannot silently add a sample.
                write_json(self.folder / 'samples.json', dict(parameters=p, mount_camera=internal.tolist(),
                           original_base_mount=original.tolist(), samples=self.samples + [sample]))
                self.internal, self.original = internal, original
                self.samples.append(sample)
                response.success = True
                response.message = f'Captured {len(self.samples)} poses; reprojection {error:.3f} px'
            except (ValueError, RuntimeError, CvBridgeError, cv2.error, tf2_ros.TransformException, OSError) as exc:
                response.message = str(exc)
            return response

        def finish(self, request, response):
            try:
                snippet = self.folder / 'chest_camera_origin.xacro.txt'
                snippet.unlink(missing_ok=True)
                result = solve(self.samples, self.internal, self.p['max_position_error_m'],
                               self.p['max_angle_error_deg'])
                result['parameters'] = self.p
                result['sample_count'] = len(self.samples)
                result['last_sample_stamp_ns'] = self.samples[-1]['stamp_ns']
                result['change_from_urdf_m_deg'] = difference(self.original, np.asarray(result['base_mount']))
                write_json(self.folder / 'result.json', result)
                if result['accepted']:
                    snippet.write_text('\n'.join(
                        f'<xacro:arg name="{key}" default="' + ' '.join(f'{v:.9f}' for v in result[key]) + '" />'
                        for key in ('chest_camera_xyz', 'chest_camera_rpy')) + '\n', encoding='utf-8')
                response.success = result['accepted']
                response.message = f'{"PASS" if result["accepted"] else "REJECTED"}: {self.folder}; validation={result["validation"]}'
            except (ValueError, RuntimeError, cv2.error, OSError) as exc:
                response.message = str(exc)
            return response

    return Calibration()


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    node = None
    try:
        node = create_node()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
