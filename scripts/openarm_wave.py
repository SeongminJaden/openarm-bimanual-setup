#!/usr/bin/env python3
"""OpenArm bimanual named-pose sequence player.

Plays a sequence of SRDF group_states (group "both_arms") as ONE continuous
trajectory per arm.  Both arms are given an identical absolute start stamp so
they stay in sync.

  default sequence:  wave_ready -> wave_in -> wave_out -> home  (repeated)

Poses are read from the installed SRDF, so editing the SRDF is enough to
change the motion - no edit to this file required.
"""

import argparse
import math
import os
import sys
import xml.dom.minidom as md

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = [f"joint{i}" for i in range(1, 8)]
SIDES = ("left", "right")
GROUP = "both_arms"
DEFAULT_SEQUENCE = "wave_ready,wave_in,wave_out,home"


def dur(seconds: float) -> Duration:
    return Duration(sec=int(seconds), nanosec=int(round((seconds % 1.0) * 1e9)))


def load_poses(srdf_path):
    """-> {state_name: {side: {joint: value}}}"""
    doc = md.parse(srdf_path)
    poses = {}
    for gs in doc.getElementsByTagName("group_state"):
        if gs.getAttribute("group") != GROUP:
            continue
        name = gs.getAttribute("name")
        entry = {s: {} for s in SIDES}
        for j in gs.getElementsByTagName("joint"):
            jn = j.getAttribute("name")
            for side in SIDES:
                pre = f"openarm_{side}_"
                if jn.startswith(pre) and jn[len(pre):] in JOINTS:
                    entry[side][jn[len(pre):]] = float(j.getAttribute("value"))
        if all(len(entry[s]) == len(JOINTS) for s in SIDES):
            poses[name] = entry
    return poses


class SeqNode(Node):
    def __init__(self):
        super().__init__("openarm_sequence")
        self.current = {}
        self.urdf = None
        latched = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._on_urdf, latched)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.act = {
            side: ActionClient(
                self, FollowJointTrajectory,
                f"/{side}_joint_trajectory_controller/follow_joint_trajectory")
            for side in SIDES
        }
        # RViz MotionPlanning "Planned Path" listens here (see moveit.rviz)
        self.preview_pub = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", 1)

    def _on_urdf(self, msg):
        if self.urdf is None:
            self.urdf = msg.data

    def _on_js(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.current[n] = p

    def wait_for_inputs(self, timeout=10.0):
        need = {f"openarm_{s}_{j}" for s in SIDES for j in JOINTS}
        end = self.get_clock().now().nanoseconds + timeout * 1e9
        while self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if need.issubset(self.current) and self.urdf is not None:
                return True
        return need.issubset(self.current)

    def limits(self):
        """-> {side: {joint: (lower, upper)}} or None."""
        if not self.urdf:
            return None
        out = {s: {} for s in SIDES}
        robot = md.parseString(self.urdf).documentElement
        for j in robot.childNodes:
            if getattr(j, "tagName", None) != "joint":
                continue
            name = j.getAttribute("name")
            lim = j.getElementsByTagName("limit")
            if not lim:
                continue
            for side in SIDES:
                pre = f"openarm_{side}_"
                if name.startswith(pre) and name[len(pre):] in JOINTS:
                    out[side][name[len(pre):]] = (
                        float(lim[0].getAttribute("lower")),
                        float(lim[0].getAttribute("upper")))
        return out if all(len(out[s]) == len(JOINTS) for s in SIDES) else None

    def start_pose(self, side):
        return {j: self.current.get(f"openarm_{side}_{j}", 0.0) for j in JOINTS}


def build_all(sides, seq_names, poses, starts, args):
    """Build one trajectory per arm on a SHARED timeline.

    Each segment's duration is sized by the largest joint motion across BOTH
    arms, so every arm reaches every pose at the same instant.  Sizing each
    arm independently would let them drift apart pose by pose.

    -> ({side: JointTrajectory}, [(name, t, dt)])
    """
    trajs = {}
    for side in sides:
        tr = JointTrajectory()
        tr.joint_names = [f"openarm_{side}_{j}" for j in JOINTS]
        trajs[side] = tr

    prev = {side: dict(starts[side]) for side in sides}
    t, log = 0.0, []

    for name in seq_names:
        delta = max(abs(poses[name][side][j] - prev[side][j])
                    for side in sides for j in JOINTS)
        dt = max(args.min_segment, delta / args.speed)
        t += dt

        for side in sides:
            target = poses[name][side]
            p = JointTrajectoryPoint()
            p.positions = [target[j] for j in JOINTS]
            p.velocities = [0.0] * len(JOINTS)
            p.time_from_start = dur(t)
            trajs[side].points.append(p)
        log.append((name, t, dt))

        if args.hold > 0.0:
            t += args.hold
            for side in sides:
                h = JointTrajectoryPoint()
                h.positions = list(trajs[side].points[-1].positions)
                h.velocities = [0.0] * len(JOINTS)
                h.time_from_start = dur(t)
                trajs[side].points.append(h)

        prev = {side: dict(poses[name][side]) for side in sides}

    return trajs, log


def main():
    ap = argparse.ArgumentParser(
        description="Play a sequence of SRDF both_arms poses on OpenArm.")
    ap.add_argument("--sequence", default=DEFAULT_SEQUENCE,
                    help=f"comma separated pose names (default: {DEFAULT_SEQUENCE})")
    ap.add_argument("--cycles", type=int, default=3,
                    help="how many times to repeat the whole sequence")
    ap.add_argument("--speed", type=float, default=0.6,
                    help="rad/s used to size each segment's duration")
    ap.add_argument("--min-segment", type=float, default=1.0,
                    help="minimum seconds per segment")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="seconds to pause at each pose")
    ap.add_argument("--arms", choices=["both", "left", "right"], default="both")
    ap.add_argument("--start-delay", type=float, default=0.5,
                    help="common start offset so both arms begin together")
    ap.add_argument("--srdf", default=None, help="override SRDF path")
    ap.add_argument("--arm-type", default="openarm_v1.0",
                    help="which moveit config folder to read poses from "
                         "(accepts v10 / v1.0 / openarm_v1.0 / v20 / ...)")
    ap.add_argument("--preview-dt", type=float, default=0.05,
                    help="preview sampling step; match RViz 'State Display "
                         "Time' so the animation plays at real speed")
    ap.add_argument("--list", action="store_true", help="list poses and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the trajectory to the terminal, publish nothing")
    ap.add_argument("--preview", action="store_true",
                    help="animate the motion in RViz (MoveIt 'Planned Path'); "
                         "the robot does NOT move")
    args = ap.parse_args()

    ARM_ALIASES = {
        "v1.0": "openarm_v1.0", "v10": "openarm_v1.0", "v1_0": "openarm_v1.0",
        "openarm_v1.0": "openarm_v1.0", "openarm_v10": "openarm_v1.0",
        "v2.0": "openarm_v2.0", "v20": "openarm_v2.0", "v2_0": "openarm_v2.0",
        "openarm_v2.0": "openarm_v2.0", "openarm_v20": "openarm_v2.0",
    }
    arm_dir = ARM_ALIASES.get(args.arm_type, args.arm_type)
    srdf = args.srdf or os.path.join(
        get_package_share_directory("openarm_bimanual_moveit_config"),
        "config", arm_dir, "openarm_bimanual.srdf")
    if not os.path.exists(srdf):
        print(f"ERROR: SRDF not found: {srdf}", file=sys.stderr)
        return 1
    poses = load_poses(srdf)
    if not poses:
        print(f"ERROR: no '{GROUP}' group_states in {srdf}", file=sys.stderr)
        return 1

    if args.list:
        print(f"SRDF: {srdf}\nposes for group '{GROUP}':")
        for n, v in poses.items():
            l = "  ".join(f"{j}={v['left'][j]:+.3f}" for j in JOINTS)
            r = "  ".join(f"{j}={v['right'][j]:+.3f}" for j in JOINTS)
            print(f"\n  {n}\n    left : {l}\n    right: {r}")
        return 0

    seq = [s.strip() for s in args.sequence.split(",") if s.strip()]
    missing = [s for s in seq if s not in poses]
    if missing:
        print(f"ERROR: unknown pose(s): {', '.join(missing)}", file=sys.stderr)
        print(f"available: {', '.join(poses)}", file=sys.stderr)
        return 1
    full_seq = seq * args.cycles

    rclpy.init()
    node = SeqNode()
    have = node.wait_for_inputs()
    if not have and not args.dry_run:
        node.get_logger().error("no /joint_states - is the controller running?")
        rclpy.shutdown()
        return 1
    if not have:
        node.get_logger().warn("no /joint_states - assuming start = all zeros (dry-run)")

    lims = node.limits()
    if lims is None:
        node.get_logger().warn("/robot_description unavailable - limit check SKIPPED")

    sides = list(SIDES) if args.arms == "both" else [args.arms]
    starts = {side: node.start_pose(side) for side in sides}
    trajs, log = build_all(sides, full_seq, poses, starts, args)

    if lims:
        bad = []
        for side in sides:
            for p in trajs[side].points:
                for j, v in zip(JOINTS, p.positions):
                    lo, up = lims[side][j]
                    if not (lo <= v <= up):
                        bad.append(f"{side}_{j}={v:+.4f} outside "
                                   f"[{lo:+.4f},{up:+.4f}]")
        if bad:
            node.get_logger().error("joint limit violation, aborting")
            for b in sorted(set(bad)):
                node.get_logger().error("  " + b)
            rclpy.shutdown()
            return 1

    built = {side: (trajs[side], log) for side in sides}
    total = trajs[sides[0]].points[-1].time_from_start
    node.get_logger().info(
        f"{len(full_seq)} poses on a shared timeline, "
        f"{len(trajs[sides[0]].points)} points/arm, "
        f"total {total.sec + total.nanosec/1e9:.1f}s")

    if args.dry_run:
        for side, (traj, log) in built.items():
            print(f"\n=== {side} ===")
            print(f"{'t(s)':>7}  {'seg':>6}  {'pose':<12} " +
                  " ".join(f"{j:>7}" for j in JOINTS))
            i = 0
            for name, t, dt in log:
                pos = traj.points[i].positions
                print(f"{t:7.2f}  {dt:6.2f}  {name:<12} " +
                      " ".join(f"{v:+7.3f}" for v in pos))
                i += 2 if args.hold > 0 else 1
        rclpy.shutdown()
        return 0

    if args.preview:
        # Merge both arms into ONE 14-joint trajectory. They already share a
        # timeline, so RViz animates the two arms moving together.
        merged = JointTrajectory()
        names, npts = [], len(trajs[sides[0]].points)
        for side in sides:
            names += list(trajs[side].joint_names)
        merged.joint_names = names

        # t=0 anchor at the current pose, so the approach is animated too
        p0 = JointTrajectoryPoint()
        p0.positions = [starts[side][j] for side in sides for j in JOINTS]
        p0.velocities = [0.0] * len(names)
        p0.time_from_start = dur(0.0)
        merged.points.append(p0)

        for i in range(npts):
            pos, vel = [], []
            for side in sides:
                pos += list(trajs[side].points[i].positions)
                vel += list(trajs[side].points[i].velocities)
            p = JointTrajectoryPoint()
            p.positions = pos
            p.velocities = vel
            p.time_from_start = trajs[sides[0]].points[i].time_from_start
            merged.points.append(p)

        # RViz ignores time_from_start and shows each waypoint for a fixed
        # "State Display Time" (0.05 s in this moveit.rviz).  A sparse
        # trajectory therefore replays far faster than it will actually run.
        # Resampling at that same step makes the preview run at real speed.
        if args.preview_dt > 0:
            def secs(d):
                return d.sec + d.nanosec / 1e9

            src = merged.points
            total = secs(src[-1].time_from_start)
            dense = JointTrajectory()
            dense.joint_names = list(merged.joint_names)
            k, t = 0, 0.0
            while t <= total + 1e-9:
                while k + 1 < len(src) - 1 and secs(src[k + 1].time_from_start) < t:
                    k += 1
                t0, t1 = secs(src[k].time_from_start), secs(src[k + 1].time_from_start)
                a = 0.0 if t1 <= t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
                q = JointTrajectoryPoint()
                q.positions = [
                    v0 + a * (v1 - v0)
                    for v0, v1 in zip(src[k].positions, src[k + 1].positions)]
                q.velocities = [0.0] * len(dense.joint_names)
                q.time_from_start = dur(t)
                dense.points.append(q)
                t += args.preview_dt
            node.get_logger().info(
                f"preview resampled: {len(src)} -> {len(dense.points)} points "
                f"@ {args.preview_dt}s (plays in ~{total:.1f}s in RViz)")
            merged = dense

        rt = RobotTrajectory()
        rt.joint_trajectory = merged

        js = JointState()
        jn, jp = [], []
        for side in sides:
            for j in JOINTS:
                jn.append(f"openarm_{side}_{j}")
                jp.append(starts[side][j])
        js.name, js.position = jn, jp
        rs = RobotState()
        rs.joint_state = js

        msg = DisplayTrajectory()
        msg.model_id = "openarm_v20"
        msg.trajectory_start = rs
        msg.trajectory.append(rt)

        deadline = node.get_clock().now().nanoseconds + 5e9
        while (node.preview_pub.get_subscription_count() == 0
               and node.get_clock().now().nanoseconds < deadline):
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.preview_pub.get_subscription_count() == 0:
            node.get_logger().warn(
                "nobody is subscribed to /display_planned_path - "
                "is RViz (MotionPlanning display) running?")
        node.preview_pub.publish(msg)
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)
        node.get_logger().info(
            "preview published to /display_planned_path - "
            "watch RViz. The robot was NOT commanded.")
        rclpy.shutdown()
        return 0

    for side in sides:
        if not node.act[side].wait_for_server(timeout_sec=5.0):
            node.get_logger().error(f"{side}: controller action server missing")
            rclpy.shutdown()
            return 1

    start = node.get_clock().now() + RclDuration(seconds=args.start_delay)
    futures = {}
    for side in sides:
        traj = built[side][0]
        traj.header.stamp = start.to_msg()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        futures[side] = node.act[side].send_goal_async(goal)

    handles = {}
    for side, fut in futures.items():
        rclpy.spin_until_future_complete(node, fut)
        gh = fut.result()
        if not gh.accepted:
            node.get_logger().error(f"{side}: goal REJECTED")
            for other in handles.values():
                other.cancel_goal_async()
            rclpy.spin_once(node, timeout_sec=1.0)
            rclpy.shutdown()
            return 1
        handles[side] = gh
    node.get_logger().info("goals accepted - moving")

    rc = 0
    try:
        for side, gh in handles.items():
            rf = gh.get_result_async()
            rclpy.spin_until_future_complete(node, rf)
            res = rf.result().result
            if res.error_code != 0:
                rc = 1
                node.get_logger().error(
                    f"{side}: error_code={res.error_code} {res.error_string}")
            else:
                node.get_logger().info(f"{side}: done")
    except KeyboardInterrupt:
        node.get_logger().warn("interrupted - cancelling")
        for gh in handles.values():
            gh.cancel_goal_async()
        rclpy.spin_once(node, timeout_sec=1.0)
        rc = 130

    rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
