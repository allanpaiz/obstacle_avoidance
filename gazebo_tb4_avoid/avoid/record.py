#!/usr/bin/env python3
"""Record the avoidance run: overview camera, turtlebot4_1 camera and lidar, ground-truth pose.

The window is measured on the simulation clock, and every sample keeps the
simulation time it was produced at, so the recordings and the videos made from
them show the run exactly as it happened however slowly the host stepped it.
The window opens before the controller starts, so the video includes the robot
standing still while it detects, and closes TAIL_S after it reaches its goal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node
from PIL import Image as PilImage

from . import spec as run
from .sim_clock import SimClock, stamp_seconds, yaw_from_quaternion


POSE_INTERVAL_S = 0.05      # simulation seconds between recorded pose samples
STALL_LIMIT_S = 60.0        # wall seconds without a clock tick before giving up
DRAIN_TIMEOUT_S = 15.0      # wall seconds to wait for in-flight samples after the window
# A sensor must deliver at least this share of the samples its rate implies.
MIN_COVERAGE = 0.9


def log(message: str) -> None:
    print(f"[avoid-record] {message}", flush=True)


def decode(message: Image) -> np.ndarray | None:
    """gz.msgs.Image -> HxWx3 uint8 RGB."""
    height, width = int(message.height), int(message.width)
    if height <= 0 or width <= 0 or not message.data:
        return None
    buffer = np.frombuffer(message.data, dtype=np.uint8)
    channels = buffer.size // (height * width)
    if channels < 1:
        return None
    frame = buffer[: height * width * channels].reshape(height, width, channels)
    if channels == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif channels >= 3 and int(message.pixel_format_type) == 4:  # BGR
        frame = frame[:, :, ::-1]
    return np.ascontiguousarray(frame[:, :, :3])


def json_range(value: float) -> float | str:
    if math.isfinite(value):
        return round(value, 4)
    return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--"
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


class Recorder:
    """Readiness and the simulation-time capture window for one group of topics."""

    def __init__(self, topics):
        self.topics = dict(topics)
        self.lock = threading.Lock()
        self.latest_stamp = {}
        self.counts = dict.fromkeys(self.topics, 0)
        self.first = {}
        self.last = {}
        self.window = None
        self.end = math.inf
        self.closed = False
        self.callbacks = []

    def arrived(self, name, stamp):
        """Note an arrival; return the window start if the sample is inside the window."""
        with self.lock:
            self.latest_stamp[name] = stamp
            if self.window is None or not self.window[0] <= stamp < self.window[1]:
                return None
            return self.window[0]

    def _claim(self, name, stamp) -> int | None:
        """Reserve the next sample index; None once the window has closed. Hold self.lock."""
        if self.window is None:
            return None
        index = self.counts[name]
        self.counts[name] += 1
        self.first.setdefault(name, stamp)
        self.last[name] = stamp
        return index

    def ready(self) -> bool:
        with self.lock:
            return all(name in self.latest_stamp for name in self.topics)

    def missing(self):
        with self.lock:
            return [name for name in self.topics if name not in self.latest_stamp]

    def begin(self, start, end) -> None:
        with self.lock:
            self.window = (start, end)
            self.end = end

    def drained(self) -> bool:
        """True once every topic has published past the end of the window."""
        with self.lock:
            return all(self.latest_stamp.get(name, -math.inf) >= self.end for name in self.topics)

    def stop(self) -> bool:
        """Close the window; False if it was already closed."""
        with self.lock:
            self.window = None
            already, self.closed = self.closed, True
            return not already

    def stats(self, rate_hz, duration, count_key):
        rows = {}
        for name, topic in self.topics.items():
            count = self.counts[name]
            expected = math.floor(duration * rate_hz)
            rows[name] = {
                "topic": topic, count_key: count, "expected": expected,
                "coverage": round(count / expected, 3) if expected else None,
                "first_sim_time_s": self.first.get(name), "last_sim_time_s": self.last.get(name),
            }
        return rows


class CameraRecorder(Recorder):
    def __init__(self, node, output: Path, topics, jsonl_name, frame_directory, pool):
        super().__init__(topics)
        self.output = output
        self.jsonl_name = jsonl_name
        self.frame_directory = frame_directory
        self.pool = pool
        self.pending = []
        self.records = []
        for name, topic in self.topics.items():
            (output / frame_directory / name).mkdir(parents=True, exist_ok=True)

            def callback(message, name=name):
                self.on_image(name, message)

            self.callbacks.append(callback)
            if not node.subscribe(Image, topic, callback):
                raise RuntimeError(f"Could not subscribe to {topic}")

    def on_image(self, name, message):
        stamp = stamp_seconds(message.header.stamp)
        start = self.arrived(name, stamp)
        if start is None:
            return
        frame = decode(message)
        if frame is None:
            return
        with self.lock:
            index = self._claim(name, stamp)
            if index is None:
                return
            relative = Path(self.frame_directory) / name / f"frame_{index:05d}.png"
            record = {
                "robot": name, "topic": self.topics[name],
                "t_s": round(stamp - start, 4), "sim_time_s": stamp, "path": str(relative),
                "width": int(message.width), "height": int(message.height),
                "pixel_format_type": int(message.pixel_format_type),
            }
            self.pending.append(self.pool.submit(self.write_frame, frame, relative, record))

    def write_frame(self, frame, relative, record):
        PilImage.fromarray(frame).save(self.output / relative, compress_level=3)
        with self.lock:
            self.records.append(record)

    def close(self):
        if not self.stop():
            return
        with self.lock:
            pending = list(self.pending)
        for future in pending:
            future.result()
        with (self.output / self.jsonl_name).open("w", encoding="utf-8") as handle:
            for record in sorted(self.records, key=lambda row: (row["sim_time_s"], row["robot"])):
                handle.write(json.dumps(record) + "\n")


class ScanRecorder(Recorder):
    def __init__(self, node, path: Path, topics):
        super().__init__(topics)
        self.handle = path.open("w", encoding="utf-8")
        for name, topic in self.topics.items():
            def callback(message, name=name):
                self.on_scan(name, message)

            self.callbacks.append(callback)
            if not node.subscribe(LaserScan, topic, callback):
                raise RuntimeError(f"Could not subscribe to {topic}")

    def on_scan(self, name, message):
        if not message.ranges or len(message.ranges) != message.count:
            return
        stamp = stamp_seconds(message.header.stamp)
        start = self.arrived(name, stamp)
        if start is None:
            return
        record = {
            "robot": name, "topic": self.topics[name],
            "t_s": round(stamp - start, 4), "sim_time_s": stamp,
            "frame": message.frame, "count": message.count,
            "angle_min": message.angle_min, "angle_max": message.angle_max,
            "angle_step": message.angle_step, "range_min": message.range_min,
            "range_max": message.range_max,
            "ranges": [json_range(r) for r in message.ranges],
            # The simulated lidar reports no return intensity.
            "intensities": [round(v, 4) for v in message.intensities] if any(message.intensities) else [],
        }
        line = json.dumps(record, allow_nan=False) + "\n"
        with self.lock:
            if self._claim(name, stamp) is None:
                return
            self.handle.write(line)

    def close(self):
        if self.stop():
            with self.lock:
                self.handle.close()


class PoseRecorder:
    """Ground-truth poses of the tracked models, one sample per POSE_INTERVAL_S."""

    def __init__(self, node, path: Path, tracked, topic):
        self.tracked = frozenset(tracked)
        self.lock = threading.Lock()
        self.handle = path.open("w", encoding="utf-8")
        self.window = None
        self.last_slot = None
        self.samples = 0
        self.seen = False
        self.callback = self.on_poses
        if not node.subscribe(Pose_V, topic, self.callback):
            raise RuntimeError(f"Could not subscribe to {topic}")

    def on_poses(self, message):
        stamp = stamp_seconds(message.header.stamp)
        poses = {}
        for entry in message.pose:
            if entry.name in self.tracked:
                p = entry.position
                poses[entry.name] = (p.x, p.y, p.z, yaw_from_quaternion(entry.orientation))
        with self.lock:
            if self.tracked <= poses.keys():
                self.seen = True
            if self.window is None or not self.window[0] <= stamp < self.window[1]:
                return
            slot = int((stamp - self.window[0]) // POSE_INTERVAL_S)
            if slot == self.last_slot:
                return
            self.last_slot = slot
            self.samples += 1
            self.handle.write(json.dumps({
                "t_s": round(stamp - self.window[0], 4), "sim_time_s": stamp,
                "poses": {name: [round(value, 4) for value in pose] for name, pose in poses.items()},
            }) + "\n")

    def begin(self, start, end) -> None:
        with self.lock:
            self.window = (start, end)

    def close(self) -> None:
        with self.lock:
            self.window = None
            if not self.handle.closed:
                self.handle.close()


def renderer_name() -> str | None:
    """The OpenGL renderer Gazebo used for sensors, from its Ogre log."""
    log_path = Path.home() / ".gz" / "rendering" / "ogre2.log"
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    names = [line.split("GL_RENDERER =", 1)[1].strip() for line in lines if "GL_RENDERER =" in line]
    return names[-1] if names else None


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for(condition, cancelled, *, wall_timeout, describe):
    deadline = time.monotonic() + wall_timeout
    while not condition():
        if cancelled.wait(0.05):
            raise RuntimeError("Run interrupted")
        if time.monotonic() >= deadline:
            raise RuntimeError(describe())


def run_capture(output: Path, node: Node) -> int:
    output.mkdir(parents=True, exist_ok=True)
    cancelled = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancelled.set())
    signal.signal(signal.SIGTERM, lambda *_: cancelled.set())
    clock = SimClock(node, run.CLOCK_TOPIC)
    pool = ThreadPoolExecutor(max_workers=4)
    cameras = CameraRecorder(node, output, {run.ROBOT: run.CAMERA_TOPIC}, "camera_frames.jsonl", "frames", pool)
    overview = CameraRecorder(node, output, {"overview": run.OVERVIEW_TOPIC},
                              "simulation_frames.jsonl", "simulation_frames", pool)
    scans = ScanRecorder(node, output / "scans.jsonl", {run.ROBOT: run.LIDAR_TOPIC})
    poses = PoseRecorder(node, output / "poses.jsonl", tracked=(run.ROBOT,), topic=run.POSE_TOPIC)
    recorders = (cameras, overview, scans)
    controller = None
    with (output / "controller.log").open("w") as controller_log:
        try:
            log("Waiting for the camera, lidar, overview camera, and ground-truth pose...")
            wait_for(lambda: all(r.ready() for r in recorders) and poses.seen and clock.now() is not None,
                     cancelled, wall_timeout=180.0,
                     describe=lambda: f"Timed out; missing {cameras.missing() + overview.missing() + scans.missing()}")
            start = clock.now()
            end = start + run.MAX_DURATION_S
            for recorder in (*recorders, poses):
                recorder.begin(start, end)
            controller = subprocess.Popen([sys.executable, "-m", "avoid.controller", "--output", str(output)],
                                          stdout=controller_log, stderr=subprocess.STDOUT)
            log(f"Recording from sim t = {start:.2f} s until {run.TAIL_S:g} s after the goal")
            wall_start = last_progress = time.monotonic()
            while clock.now() < end:
                if cancelled.wait(0.1):
                    raise RuntimeError("Run interrupted")
                if clock.stalled_for() > STALL_LIMIT_S:
                    raise RuntimeError("Simulation clock stopped advancing")
                if controller.poll() is not None and end == start + run.MAX_DURATION_S:
                    end = min(end, clock.now() + run.TAIL_S)
                    for recorder in (*recorders, poses):
                        recorder.begin(start, end)
                    log(f"Controller exited ({controller.returncode}); closing the window at "
                        f"{end - start:.1f} s")
                if time.monotonic() - last_progress >= 10.0:
                    log(f"sim {clock.now() - start:5.1f} s")
                    last_progress = time.monotonic()
            wall_s = time.monotonic() - wall_start
            deadline = time.monotonic() + DRAIN_TIMEOUT_S
            while not all(r.drained() for r in recorders) and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            for recorder in (*recorders, poses):
                recorder.close()
            stop_process(controller)
            pool.shutdown(wait=True)

    duration = end - start
    groups = {
        "camera": cameras.stats(run.CAMERA_RATE, duration, "frames"),
        "overview": overview.stats(run.OVERVIEW_RATE, duration, "frames"),
        "lidar": scans.stats(run.LIDAR_RATE, duration, "scans"),
    }
    problems = [f"{name}: {row.get('frames', row.get('scans'))}/{row['expected']} samples"
                for rows in groups.values() for name, row in rows.items()
                if row["coverage"] is None or row["coverage"] < MIN_COVERAGE]
    if controller.returncode != 0:
        problems.append(f"controller exited with {controller.returncode}; see controller.log")
    summary = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "time_base": "simulation",
        "duration_sim_s": round(duration, 3),
        "sim_time_start_s": round(start, 4),
        "sim_time_end_s": round(end, 4),
        "duration_wall_s": round(wall_s, 2),
        "real_time_factor": round(duration / wall_s, 3),
        "renderer": renderer_name(),
        "sensors": groups,
        "poses": {"samples": poses.samples},
        "controller_exit_code": controller.returncode,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"{summary['status']}: {duration:.1f} simulation seconds in {format_duration(wall_s)} "
        f"(real-time factor {summary['real_time_factor']}, renderer {summary['renderer']})")
    for problem in problems:
        log(f"  {problem}")
    return 0 if not problems else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/outputs"))
    args = parser.parse_args()
    node = Node()
    try:
        status = run_capture(args.output, node)
    except (RuntimeError, subprocess.SubprocessError) as exc:
        log(f"ERROR: {exc}")
        status = 1
    # Every output is written and closed by now. Skip interpreter teardown:
    # destroying gz-transport's threads while Gazebo is still publishing to
    # this process can abort it ("terminate called without an active exception").
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    main()
