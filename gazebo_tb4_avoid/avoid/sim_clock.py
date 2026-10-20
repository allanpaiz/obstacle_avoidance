"""Simulation time for the controller and the sensor recorder.

Gazebo publishes the world clock on every physics step. Timing on it instead of
the wall clock keeps behaviour identical at any real-time factor: one second of
holding still is one second of physics whether the host runs the world at 1.0x
or 0.1x.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

from gz.msgs10.clock_pb2 import Clock
from gz.transport13 import Node


def stamp_seconds(stamp) -> float:
    """gz.msgs.Time -> float seconds."""
    return stamp.sec + stamp.nsec * 1e-9


def yaw_from_quaternion(q) -> float:
    w = q.w if (q.x or q.y or q.z or q.w) else 1.0
    return math.atan2(2.0 * (w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SimClock:
    def __init__(self, node: Node, topic: str):
        self._condition = threading.Condition()
        self._now: Optional[float] = None
        self._updated_wall = time.monotonic()
        self._callback = self._on_clock
        if not node.subscribe(Clock, topic, self._callback):
            raise RuntimeError(f"Could not subscribe to {topic}")

    def _on_clock(self, message: Clock) -> None:
        with self._condition:
            self._now = stamp_seconds(message.sim)
            self._updated_wall = time.monotonic()
            self._condition.notify_all()

    def now(self) -> Optional[float]:
        """Latest simulation time in seconds, or None before the first tick."""
        with self._condition:
            return self._now

    def stalled_for(self) -> float:
        """Wall seconds since the clock last advanced."""
        with self._condition:
            return time.monotonic() - self._updated_wall

    def wait_until(self, sim_time: float, stop: threading.Event) -> Optional[float]:
        """Block until simulation time reaches sim_time; None if `stop` was set first."""
        with self._condition:
            while self._now is None or self._now < sim_time:
                if stop.is_set():
                    return None
                self._condition.wait(0.1)
            return self._now
