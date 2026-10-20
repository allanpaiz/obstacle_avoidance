#!/usr/bin/env python3
"""Detect one lidar obstacle, plan a straight-path detour around it, and drive it.

The sim counterpart of the hardware detect_plan_execute_straight.py:

  hold     stand still for HOLD_S
  detect   cluster every scan for DETECT_S with the hardware DBSCAN and
           circle fit (lidar_clustering.py), pick the
           corridor obstacle the way the hardware detector does, and take the
           median over the window
  plan     a straight path to GOAL_DISTANCE that swings out past the inflated
           obstacle and back (the hardware's trajectory optimiser is not in
           this repository, so this is a geometric stand-in)
  execute  pure pursuit on wheel odometry, in the start frame
  done     stopped at the goal

Writes avoid_events.jsonl (phase changes), detections.jsonl (every processed
scan), and plan.json to --output. All times are simulation seconds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.odometry_pb2 import Odometry
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node

from . import spec as run
from .lidar_clustering import ObstacleCircle, detect_obstacles_from_scan
from .sim_clock import SimClock, stamp_seconds, yaw_from_quaternion


def log(message: str) -> None:
    print(f"[avoid-controller] {message}", flush=True)


# --- detection, as in the hardware lidar_obstacle_detector.py ---

def selection_score(obstacle: ObstacleCircle, corridor_width: float) -> float:
    half_width = 0.5 * corridor_width
    misses = 0.0 if obstacle.intersects_corridor(corridor_width) else 1.0
    lateral_gap = max(0.0, abs(obstacle.center_y) - obstacle.radius - half_width)
    return (1000.0 * misses + abs(obstacle.center_y) + 0.25 * obstacle.forward_clearance
            + 0.10 * obstacle.radius + 0.25 * lateral_gap)


def robust_obstacle(history: list[ObstacleCircle]) -> dict:
    """Median centre bearing and near-surface range, 90th-percentile radius."""
    cx = float(np.median([o.center_x for o in history]))
    cy = float(np.median([o.center_y for o in history]))
    bearing = math.atan2(cy, cx)
    near = float(np.median([o.min_range for o in history]))
    radius = max(run.PLANNER_RADIUS_MIN, float(np.percentile([o.radius for o in history], 90.0)))
    return {"center": [(near + radius) * math.cos(bearing), (near + radius) * math.sin(bearing)],
            "radius": radius, "min_range": near, "bearing": bearing, "scans": len(history)}


# --- planning ---

def plan_path(obstacle: dict) -> dict:
    """Straight line to the goal with a raised-cosine swing past the obstacle.

    The path holds its full offset while within `clearance` of the obstacle's
    x, so the robot centre stays at least obstacle radius + robot radius +
    safety margin from the obstacle centre.
    """
    ox, oy = obstacle["center"]
    clearance = obstacle["radius"] + run.ROBOT_RADIUS + run.SAFETY_MARGIN
    side = -1.0 if oy >= 0 else 1.0               # pass on the side away from the obstacle
    offset = oy + side * clearance
    hold_from, hold_to = ox - clearance, ox + clearance
    ramp_in, ramp_out = hold_from - run.RAMP_LENGTH, hold_to + run.RAMP_LENGTH
    xs = np.arange(0.0, run.GOAL_DISTANCE + 1e-9, 0.02)

    def lateral(x):
        if x <= ramp_in or x >= ramp_out:
            return 0.0
        if hold_from <= x <= hold_to:
            return offset
        s = (x - ramp_in) / run.RAMP_LENGTH if x < hold_from else (ramp_out - x) / run.RAMP_LENGTH
        return offset * 0.5 * (1.0 - math.cos(math.pi * s))

    path = [[round(float(x), 4), round(lateral(x), 4)] for x in xs]
    return {"path": path, "goal": [run.GOAL_DISTANCE, 0.0], "clearance": clearance,
            "offset": offset, "side": "right" if side < 0 else "left"}


# --- the run ---

class Controller:
    def __init__(self, node: Node, output: Path):
        self.output = output
        self.lock = threading.Lock()
        self.odom = None                          # (sim time, x, y, yaw) in the odometry frame
        self.scan = None
        self.clock = SimClock(node, run.CLOCK_TOPIC)
        self.cmd = node.advertise(run.CMD_TOPIC, Twist())
        self.events = (output / "avoid_events.jsonl").open("w", encoding="utf-8")
        self.detections = (output / "detections.jsonl").open("w", encoding="utf-8")
        self._odom_cb, self._scan_cb = self.on_odom, self.on_scan
        if not node.subscribe(Odometry, run.ODOM_TOPIC, self._odom_cb):
            raise RuntimeError(f"Could not subscribe to {run.ODOM_TOPIC}")
        if not node.subscribe(LaserScan, run.LIDAR_TOPIC, self._scan_cb):
            raise RuntimeError(f"Could not subscribe to {run.LIDAR_TOPIC}")

    def on_odom(self, message):
        p = message.pose.position
        with self.lock:
            self.odom = (stamp_seconds(message.header.stamp), p.x, p.y, yaw_from_quaternion(message.pose.orientation))

    def on_scan(self, message):
        with self.lock:
            self.scan = (stamp_seconds(message.header.stamp), list(message.ranges), message.angle_min,
                         message.angle_step, message.range_min, message.range_max)

    def event(self, phase, **fields):
        row = {"sim_time_s": self.clock.now(), "phase": phase, **fields}
        self.events.write(json.dumps(row) + "\n")
        self.events.flush()
        log(f"{phase} at sim t {row['sim_time_s']:.2f} s {fields if fields else ''}")

    def drive(self, v, w):
        command = Twist()
        command.linear.x, command.angular.z = v, w
        self.cmd.publish(command)

    def wait(self, seconds, stop):
        if self.clock.wait_until(0.0, stop) is None:      # the first clock tick
            raise RuntimeError("interrupted")
        target = self.clock.now() + seconds
        if self.clock.wait_until(target, stop) is None:
            raise RuntimeError("interrupted")

    def detect(self, stop) -> dict:
        history, last_stamp = [], None
        end = self.clock.now() + run.DETECT_S
        while self.clock.now() < end:
            with self.lock:
                scan = self.scan
            if scan is None or scan[0] == last_stamp:
                self.wait(0.01, stop)
                continue
            last_stamp = scan[0]
            stamp, ranges, angle_min, step, range_min, range_max = scan
            obstacles, _, _ = detect_obstacles_from_scan(
                ranges, angle_min, step, range_min, range_max,
                max_range=run.MAX_RANGE, front_fov_rad=math.radians(run.FRONT_FOV_DEG), min_x=run.MIN_X,
                cluster_eps=run.CLUSTER_EPS, min_samples=run.MIN_SAMPLES, radius_padding=run.RADIUS_PADDING,
                max_obstacle_radius=run.MAX_OBSTACLE_RADIUS)
            scores = [selection_score(o, run.CORRIDOR_WIDTH) for o in obstacles]
            selected = min(range(len(obstacles)), key=scores.__getitem__) if obstacles else None
            if selected is not None and scores[selected] >= 1000.0:
                selected = None                    # nothing in the corridor
            if selected is not None:
                history.append(obstacles[selected])
            self.detections.write(json.dumps({
                "sim_time_s": stamp, "selected": selected,
                "candidates": [{"x": o.center_x, "y": o.center_y, "r": o.radius, "points": o.point_count}
                               for o in obstacles]}) + "\n")
        self.detections.flush()
        if not history:
            raise RuntimeError("no obstacle in the corridor during the detection window")
        return robust_obstacle(history)

    def execute(self, path, start, stop):
        """Pure pursuit along `path` (start frame), on odometry relative to `start`."""
        sx, sy, syaw = start
        c0, s0 = math.cos(syaw), math.sin(syaw)
        points = np.asarray(path)
        goal = points[-1]
        progress = 0
        period = 1.0 / run.CONTROL_RATE
        while True:
            with self.lock:
                _, ox, oy, oyaw = self.odom
            dx, dy = ox - sx, oy - sy
            x, y, yaw = c0 * dx + s0 * dy, -s0 * dx + c0 * dy, oyaw - syaw
            remaining = math.hypot(goal[0] - x, goal[1] - y)
            if remaining < run.GOAL_TOLERANCE or x >= goal[0]:
                self.drive(0.0, 0.0)
                return x, y, yaw
            window = points[progress:progress + 60]
            progress += int(np.argmin(np.hypot(window[:, 0] - x, window[:, 1] - y)))
            ahead = np.hypot(points[progress:, 0] - x, points[progress:, 1] - y)
            beyond = np.flatnonzero(ahead >= run.LOOKAHEAD)
            tx, ty = points[progress + beyond[0]] if len(beyond) else goal
            alpha = math.atan2(ty - y, tx - x) - yaw
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            v = min(run.CRUISE_SPEED, 0.6 * remaining + 0.04)
            w = max(-0.9, min(0.9, 2.0 * v * math.sin(alpha) / run.LOOKAHEAD))
            self.drive(v, w)
            self.wait(period, stop)

    def run(self, stop):
        self.wait_for_inputs(stop)
        self.drive(0.0, 0.0)
        self.event("hold")
        self.wait(run.HOLD_S, stop)
        with self.lock:
            start = self.odom[1:]
        self.event("detect", start_odom=start)
        obstacle = self.detect(stop)
        planned = plan_path(obstacle)
        plan = {"frame": "start (x forward, y left)", "start_odom": start, "obstacle": obstacle,
                "robot_radius": run.ROBOT_RADIUS, "safety_margin": run.SAFETY_MARGIN, **planned}
        (self.output / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
        self.event("plan", obstacle=obstacle["center"], radius=round(obstacle["radius"], 3),
                   offset=round(planned["offset"], 3), side=planned["side"])
        self.event("execute")
        x, y, yaw = self.execute(planned["path"], start, stop)
        self.event("done", final=[round(x, 3), round(y, 3), round(yaw, 3)],
                   goal_error=round(math.hypot(run.GOAL_DISTANCE - x, y), 3))
        for _ in range(10):
            self.drive(0.0, 0.0)
            self.wait(0.05, stop)

    def wait_for_inputs(self, stop):
        while True:
            with self.lock:
                if self.odom is not None and self.scan is not None:
                    return
            self.wait(0.05, stop)

    def close(self):
        self.events.close()
        self.detections.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("/outputs"))
    args = parser.parse_args()
    node = Node()
    stop = threading.Event()
    controller = Controller(node, args.output)
    status = 0
    try:
        controller.run(stop)
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        controller.drive(0.0, 0.0)
        status = 1
    controller.close()
    sys.stdout.flush()
    os._exit(status)


if __name__ == "__main__":
    raise SystemExit(main())
