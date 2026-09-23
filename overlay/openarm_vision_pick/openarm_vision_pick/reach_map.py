"""Where can this arm actually take a top-down grasp, and at what jaw angle?

"no jaw angle works" says the grasp was unreachable but not what would be.
This sweeps /compute_ik over a grid of positions and jaw angles and prints
the answer as a map, so the object can be put somewhere the arm can reach
instead of guessed at.

  python3 reach_map.py                  # a slice at the grasp height
  python3 reach_map.py --z 0.25 0.35    # several heights
  python3 reach_map.py --group right_arm --tcp openarm_right_hand_tcp

Needs move_group running.  Nothing moves: /compute_ik only asks whether a
posture exists.
"""
import argparse
import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node

sys.argv_backup = list(sys.argv)
from openarm_vision_pick.pick_and_place import (  # noqa: E402
    grasp_orientation_candidates)


class Sweeper(Node):
    def __init__(self, group, tcp, frame, approach=0.10):
        super().__init__('reach_map')
        self.group, self.tcp, self.frame = group, tcp, frame
        self.approach = approach
        self.cli = self.create_client(GetPositionIK, '/compute_ik')

    def ready(self, timeout=120.0):
        return self.cli.wait_for_service(timeout_sec=timeout)

    def ik(self, xyz, yaw, timeout=0.02, tilts=()):
        """0 if reachable straight down, the lean index if only leaned,
        None if not at all."""
        above = (xyz[0], xyz[1], xyz[2] + self.approach)
        for idx, (q, tdeg, lab) in enumerate(
                grasp_orientation_candidates(yaw, tilts)):
            # Contact pose: reach only.  The pose above it: reach AND clear
            # of the scene.  Same rule as the pick itself.
            if self._ik_one(xyz, q, timeout, avoid=False) and \
                    self._ik_one(above, q, timeout, avoid=True):
                return idx
        return None

    def _ik_one(self, xyz, q, timeout, avoid=True):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group
        req.ik_request.ik_link_name = self.tcp
        req.ik_request.robot_state.is_diff = True
        req.ik_request.avoid_collisions = bool(avoid)
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = int(timeout * 1e9)
        ps = PoseStamped()
        ps.header.frame_id = self.frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = \
            [float(v) for v in xyz]
        ps.pose.orientation.x, ps.pose.orientation.y, \
            ps.pose.orientation.z, ps.pose.orientation.w = \
            [float(v) for v in q]
        req.ik_request.pose_stamped = ps
        fut = self.cli.call_async(req)
        end = time.time() + 3.0
        while rclpy.ok() and not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
        if not fut.done():
            return False
        return fut.result().error_code.val == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='left_arm')
    ap.add_argument('--tcp', default='openarm_left_hand_tcp')
    ap.add_argument('--frame', default='world')
    ap.add_argument('--x', type=float, nargs=3, default=[0.10, 0.60, 0.05],
                    metavar=('MIN', 'MAX', 'STEP'))
    ap.add_argument('--y', type=float, nargs=3, default=[-0.40, 0.50, 0.05],
                    metavar=('MIN', 'MAX', 'STEP'))
    ap.add_argument('--z', type=float, nargs='+', default=[0.25])
    ap.add_argument('--yaws', type=float, nargs='+',
                    default=[0, 45, 90, 135])
    ap.add_argument('--approach', type=float, default=0.10,
                    help='a cell counts only if the pose this far above it '
                         'is also reachable and clear of the scene')
    ap.add_argument('--tilts', type=float, nargs='*', default=[15, 30],
                    help='leans from vertical to try, in degrees; give '
                         'none to insist on plumb')
    a = ap.parse_args()

    rclpy.init()
    n = Sweeper(a.group, a.tcp, a.frame, a.approach)
    if not n.ready():
        print('/compute_ik was not discovered in 120 s. move_group can be '
          'running and still take that long to be found on a busy graph.')
        return 1

    xs = np.arange(a.x[0], a.x[1] + 1e-9, a.x[2])
    ys = np.arange(a.y[0], a.y[1] + 1e-9, a.y[2])
    yaws = [math.radians(v) for v in a.yaws]
    tilts = [math.radians(v) for v in a.tilts if v > 0]

    print('group {}, tcp {}, frame {}'.format(a.group, a.tcp, a.frame))
    print('a cell shows which of the jaw angles {} have an IK solution;'
          .format(', '.join('{:.0f}'.format(v) for v in a.yaws)))
    print('. = none; a digit is that angle\'s index; a trailing \' means it '
          'only solves with the hand leaned (tilts {})'.format(
              ', '.join('{:.0f}'.format(v) for v in a.tilts) or 'none'))
    print()

    for z in a.z:
        print('=== z = {:.3f} m '.format(z) + '=' * 40)
        header = '       ' + ''.join('{:>7.2f}'.format(y) for y in ys)
        print(header + '   <- y')
        total = 0
        for x in xs:
            row = '{:6.2f} '.format(x)
            for y in ys:
                cell = ''
                for k, w in enumerate(yaws):
                    r = n.ik((x, y, z), w, tilts=tilts)
                    if r is None:
                        continue
                    total += 1
                    # a digit = reachable plumb; digit' = only when leaned
                    cell += str(k) + ('' if r == 0 else "'")
                row += '{:>7s}'.format(cell if cell else '.')
            print(row)
        print()
        if total == 0:
            print('  nothing on this slice is reachable at any of those '
                  'angles')
        print()

    print('x is forward from the robot base, y is to its left.')
    print('Put the object on a cell with a digit, and note which angle')
    print('index worked - that is the jaw angle the grasp will need.')
    n.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
