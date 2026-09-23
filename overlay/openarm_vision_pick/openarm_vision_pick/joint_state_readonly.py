"""OpenArm v1 encoder polling only: never enable/disable/control/zero motors.

Packet format and 12.5-rad range follow openarm_can's dm_motor_control.cpp
and dm_motor_constants.hpp (DM8009/DM4340/DM4310). SocketCAN, Linux only.
"""
import select
import math
import socket
import struct
import time


def query_frame(motor_id):
    if not 1 <= motor_id <= 8:
        raise ValueError('Expected arm/gripper motor ID 1..8')
    return struct.pack('=IBBBB64s', 0x7ff, 8, 1, 0, 0,
                       bytes([motor_id, 0, 0xcc, 0, 0, 0, 0, 0]))


def decode_frame(frame):
    if len(frame) not in (16, 72) or frame[4] != 8:
        return None
    can_id = struct.unpack_from('=I', frame)[0]
    if not 0x11 <= can_id <= 0x18:
        return None  # Also rejects extended/RTR/error frames.
    motor_id = can_id - 0x10
    data = frame[8:16]
    if data[0] != motor_id:
        return None  # Require disabled state (high nibble 0), correct motor ID.
    position = ((data[1] << 8) | data[2]) * 25.0 / 65535.0 - 12.5
    return motor_id, position


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from sensor_msgs.msg import JointState

    class Reader(Node):
        def __init__(self):
            super().__init__('joint_state_readonly')
            self.sockets = {}
            self.pub = self.create_publisher(JointState, '/joint_states', 10)
            self.last_warning = 0.
            # Same approximate parallel-link conversion as OpenArmHW; tune if calibrated.
            self.gripper_scale = self.declare_parameter('gripper_m_per_rad', .044 / -1.0472).value
            if not isinstance(self.gripper_scale, (int, float)) or not math.isfinite(self.gripper_scale):
                raise ValueError('gripper_m_per_rad must be finite')
            try:
                for side, default in [('right', 'can0'), ('left', 'can1')]:
                    interface = self.declare_parameter(side + '_can_interface', default).value
                    if not interface:
                        continue
                    s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                    self.sockets[s] = side
                    s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
                    s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                                 b''.join(struct.pack('=II', i, socket.CAN_EFF_FLAG | socket.CAN_RTR_FLAG | socket.CAN_SFF_MASK)
                                          for i in range(0x11, 0x19)))
                    s.bind((interface,))
                    s.setblocking(False)
                    self.get_logger().info(f'{side}: {interface}; state queries ONLY')
                if not self.sockets:
                    raise ValueError('Configure at least one CAN interface')
            except Exception:
                for s in self.sockets:
                    s.close()
                raise
            self.create_timer(.05, self.poll)

        def poll(self):
            received = {s: {} for s in self.sockets}
            started = self.get_clock().now()
            try:
                for s in self.sockets:
                    # Reject queued old observations before this request batch.
                    for _ in range(1000):
                        try:
                            s.recv(72)
                        except BlockingIOError:
                            break
                    else:
                        raise RuntimeError('CAN receive queue did not drain')
                    for motor_id in range(1, 9):
                        s.send(query_frame(motor_id))
                deadline = time.monotonic() + .025
                while time.monotonic() < deadline and any(len(v) < 8 for v in received.values()):
                    ready, _, _ = select.select(list(self.sockets), [], [], max(0., deadline-time.monotonic()))
                    for s in ready:
                        decoded = decode_frame(s.recv(72))
                        if decoded:
                            received[s][decoded[0]] = decoded[1]
                missing = []
                for s, values in received.items():
                    side = self.sockets[s]
                    if set(range(1, 8)) - values.keys():
                        missing.append(f'{side}: missing/invalid {sorted(set(range(1,8))-values.keys())}')
                        continue
                    msg = JointState()
                    # Measurement uncertainty is bounded by the 25 ms polling window.
                    msg.header.stamp = started.to_msg()
                    msg.name = [f'openarm_{side}_joint{i}' for i in range(1, 8)]
                    msg.position = [values[i] for i in range(1, 8)]
                    if 8 in values:
                        msg.name.append(f'openarm_{side}_finger_joint1')
                        msg.position.append(values[8] * self.gripper_scale)
                    else:
                        missing.append(f'{side}: gripper response missing')
                    self.pub.publish(msg)
                if missing:
                    self.warn('; '.join(missing) + '; no stale or zero-filled states published')
            except (OSError, RuntimeError) as exc:
                self.warn(str(exc))

        def warn(self, message):
            if time.monotonic() - self.last_warning > 3.:
                self.get_logger().warning(message)
                self.last_warning = time.monotonic()

        def destroy_node(self):
            for s in self.sockets:
                s.close()
            return super().destroy_node()

    rclpy.init(args=args)
    node = None
    try:
        node = Reader()
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
