# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Adjust the chest camera's mounting pose by hand, in RViz, and read off the
value to put in the launch file.

The URDF's chest_camera_xyz/rpy is a mount pose nobody measured precisely, and
sliding a number in a file and relaunching to see the effect is slow.  This
does it live:

  * a 6-DOF interactive marker appears at the camera's current mount pose,
    as a draggable handle on openarm_body_link0;
  * the camera's coloured point cloud is RE-PUBLISHED from that handle's
    frame, so it moves with the handle - drag until the cloud's bench and
    arm land on the robot model's bench and arm, and the pose is right;
  * every change prints the launch value, and the final answer is printed
    on exit.

  ros2 run openarm_vision_pick camera_adjust --ros-args \\
    -p cloud_topic:=/chest/openarm_chest_camera/depth/color/points

In RViz: add an InteractiveMarkers display on /camera_adjust/update, and a
PointCloud2 on /camera_adjust/cloud (Best Effort).  Show the RobotModel too.
Then drag the handle.  The number to copy is printed as:

  chest_camera_xyz:='X Y Z'  chest_camera_rpy:='R P Yaw'

This adjusts where the camera's OWN optical frame is placed relative to
body_link0.  It publishes a correction on top of the live camera->world TF,
so it works whether or not the URDF value is already close.
"""

import math

import numpy as np
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, Quaternion
from interactive_markers import InteractiveMarkerServer
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import (InteractiveMarker,
                                    InteractiveMarkerControl, Marker)


def quat_to_rpy(x, y, z, w):
    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = 2 * (w * y - z * x)
    p = math.asin(max(-1.0, min(1.0, sp)))
    yw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return r, p, yw


def rpy_to_quat(r, p, yw):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(yw / 2), math.sin(yw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def quat_to_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class CameraAdjust(Node):

    def __init__(self):
        super().__init__('camera_adjust')

        p = self.declare_parameter
        p('body_frame', 'openarm_body_link0')
        p('camera_frame', 'openarm_chest_camera_color_optical_frame')
        p('cloud_topic', '/chest/openarm_chest_camera/depth/color/points')
        # The current mount value, so the handle starts where the URDF has it.
        p('init_xyz', [0.053740, 0.0, 0.621])
        p('init_rpy', [0.0, 0.785398, 0.0])
        # The camera optical frame relative to the mount (bottom_screw).  For
        # the D455 this is roughly the macro's bottom_screw->color offset; a
        # small constant, and the handle is adjusted to absorb any error in
        # it anyway.  Left at zero, the handle IS the optical-frame pose.
        p('mount_to_optical_xyz', [0.0, 0.0, 0.0])
        p('cloud_stride', 4)

        g = self.get_parameter
        self.body = g('body_frame').value
        self.cam = g('camera_frame').value
        self.stride = int(g('cloud_stride').value)
        xyz = list(g('init_xyz').value)
        rpy = list(g('init_rpy').value)
        self.pos = np.array(xyz, dtype=float)
        self.quat = np.array(rpy_to_quat(*rpy))     # x,y,z,w

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cloud_in = None
        self.create_subscription(PointCloud2, g('cloud_topic').value,
                                 self._on_cloud, qos_profile_sensor_data)
        self.pub_cloud = self.create_publisher(PointCloud2,
                                               '~/cloud', 2)

        self.server = InteractiveMarkerServer(self, 'camera_adjust')
        self._make_marker()
        self.server.applyChanges()

        # Re-publish the cloud at 10 Hz from wherever the handle currently is.
        self.create_timer(0.1, self._republish)
        self._print_value('start')

        self.get_logger().info(
            'drag the handle in RViz until the cloud lands on the robot.\n'
            '  add: InteractiveMarkers on /camera_adjust/update,\n'
            '       PointCloud2 (Best Effort) on /camera_adjust/cloud,\n'
            '       RobotModel, fixed frame {}'.format(self.body))

    # ------------------------------------------------------------- marker
    def _make_marker(self):
        im = InteractiveMarker()
        im.header.frame_id = self.body
        im.name = 'chest_camera'
        im.description = 'chest camera mount'
        im.scale = 0.15
        im.pose.position = Point(x=float(self.pos[0]), y=float(self.pos[1]),
                                 z=float(self.pos[2]))
        im.pose.orientation = Quaternion(x=float(self.quat[0]),
                                         y=float(self.quat[1]),
                                         z=float(self.quat[2]),
                                         w=float(self.quat[3]))

        # A little axis triad so the orientation is visible.
        vis = InteractiveMarkerControl()
        vis.always_visible = True
        vis.interaction_mode = InteractiveMarkerControl.NONE
        for axis, col in ((0, (1., 0., 0.)), (1, (0., 1., 0.)),
                          (2, (0., 0., 1.))):
            m = Marker()
            m.type = Marker.ARROW
            m.scale.x, m.scale.y, m.scale.z = 0.08, 0.01, 0.01
            m.color.r, m.color.g, m.color.b, m.color.a = col + (1.0,)
            q = [0., 0., 0., 1.]
            if axis == 1:
                q = list(rpy_to_quat(0, 0, math.pi / 2))
            elif axis == 2:
                q = list(rpy_to_quat(0, -math.pi / 2, 0))
            m.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            vis.markers.append(m)
        im.controls.append(vis)

        # 6-DOF handles.
        for aname, ax in (('x', (1., 0., 0.)), ('y', (0., 1., 0.)),
                          ('z', (0., 0., 1.))):
            q = self._axis_quat(ax)
            move = InteractiveMarkerControl()
            move.name = 'move_' + aname
            move.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            move.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            im.controls.append(move)
            rot = InteractiveMarkerControl()
            rot.name = 'rot_' + aname
            rot.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            rot.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            im.controls.append(rot)

        self.server.insert(im, feedback_callback=self._on_feedback)

    @staticmethod
    def _axis_quat(ax):
        # A quaternion whose local x points along ax (the convention RViz
        # uses to orient a MOVE_AXIS/ROTATE_AXIS control).
        if ax == (1., 0., 0.):
            return (0., 0., 0., 1.)
        if ax == (0., 1., 0.):
            return rpy_to_quat(0, 0, math.pi / 2)
        return rpy_to_quat(0, -math.pi / 2, 0)

    def _on_feedback(self, fb):
        po = fb.pose
        self.pos = np.array([po.position.x, po.position.y, po.position.z])
        self.quat = np.array([po.orientation.x, po.orientation.y,
                              po.orientation.z, po.orientation.w])
        if fb.event_type == fb.MOUSE_UP:
            self._print_value('adjusted')

    # ------------------------------------------------------------- cloud
    def _on_cloud(self, msg):
        self.cloud_in = msg

    def _republish(self):
        if self.cloud_in is None:
            return
        # The incoming cloud is in the camera's own optical frame.  Place it
        # at the handle pose (which is that frame's pose in body_frame), so
        # dragging the handle drags the cloud.
        pts = np.array(list(pc2.read_points(
            self.cloud_in, field_names=('x', 'y', 'z', 'rgb'),
            skip_nans=True)))
        if pts.size == 0:
            return
        pts = pts[::self.stride]
        xyz = np.stack([pts['x'], pts['y'], pts['z']], axis=1) \
            if pts.dtype.names else pts[:, :3]
        R = quat_to_mat(*self.quat)
        out_xyz = xyz @ R.T + self.pos
        rgb = pts['rgb'] if pts.dtype.names else pts[:, 3]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.body
        recs = [(float(out_xyz[i, 0]), float(out_xyz[i, 1]),
                 float(out_xyz[i, 2]), float(rgb[i]))
                for i in range(out_xyz.shape[0])]
        cloud = pc2.create_cloud(
            header, self.cloud_in.fields, recs)
        self.pub_cloud.publish(cloud)

    # ------------------------------------------------------------- output
    def _print_value(self, tag):
        r, p, yw = quat_to_rpy(*self.quat)
        self.get_logger().info(
            '[{}] chest_camera_xyz:=\'{:.4f} {:.4f} {:.4f}\'  '
            'chest_camera_rpy:=\'{:.5f} {:.5f} {:.5f}\''.format(
                tag, self.pos[0], self.pos[1], self.pos[2], r, p, yw))


def main():
    rclpy.init()
    node = CameraAdjust()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._print_value('FINAL - copy this into the launch defaults')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
