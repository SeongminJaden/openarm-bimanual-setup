"""Republish a PointCloud2 with a different header frame_id (and optionally re-stamped).

The Gazebo bridge hands over clouds stamped with Gazebo's own scoped frame name, which is not a TF
frame, so MoveIt's octomap updater cannot place them.  This puts the robot's camera link name on.

    ros2 run openarm_vision_pick frame_relay --ros-args -p in_topic:=... -p out_topic:=... -p frame_id:=...
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2


class FrameRelay(Node):
    def __init__(self):
        super().__init__('frame_relay')
        self.declare_parameter('in_topic', '/gz/points')
        self.declare_parameter('out_topic', '/points')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('restamp', True)
        self.frame = self.get_parameter('frame_id').value
        self.restamp = self.get_parameter('restamp').value
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(PointCloud2, self.get_parameter('out_topic').value, qos)
        self.create_subscription(PointCloud2, self.get_parameter('in_topic').value, self.cb, qos)
        self.n = 0

    def cb(self, msg):
        msg.header.frame_id = self.frame
        if self.restamp:
            msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.n += 1
        if self.n == 1:
            self.get_logger().info('relaying clouds as frame {!r}'.format(self.frame))


def main():
    rclpy.init()
    n = FrameRelay()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
