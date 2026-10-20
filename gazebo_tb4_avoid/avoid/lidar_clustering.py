"""LaserScan clustering for TurtleBot 4 obstacle detection, vendored from tb4_sensor_detection.

The detector uses DBSCAN-style density clustering on 2D LiDAR points. DBSCAN is
a standard open algorithm for grouping points by local density; keeping the
small implementation here avoids adding a scikit-learn dependency to the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ObstacleCircle:
    """Circular approximation of one detected obstacle in the scan frame."""

    center_x: float
    center_y: float
    radius: float
    point_count: int
    min_range: float
    max_range: float
    bearing: float
    width: float
    depth: float
    cluster_id: int

    @property
    def center_distance(self) -> float:
        return float(math.hypot(self.center_x, self.center_y))

    @property
    def clearance_from_lidar(self) -> float:
        return float(max(0.0, self.center_distance - self.radius))

    @property
    def forward_clearance(self) -> float:
        return float(max(0.0, self.center_x - self.radius))

    def intersects_corridor(self, corridor_width: float) -> bool:
        half_width = 0.5 * max(0.0, float(corridor_width))
        return self.center_x + self.radius >= 0.0 and abs(self.center_y) <= half_width + self.radius


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def ranges_to_points(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    *,
    max_range: float | None = None,
    front_fov_rad: float | None = None,
    min_x: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a LaserScan into valid 2D points in the scan frame.

    Returns ``(points_xy, angles, ranges)`` after filtering invalid values,
    ranges outside the sensor limits, optional max range, optional forward
    field-of-view, and optional minimum x coordinate.
    """

    scan_ranges = np.asarray(list(ranges), dtype=float)
    if scan_ranges.size == 0:
        return np.empty((0, 2), dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    angles = float(angle_min) + np.arange(scan_ranges.size, dtype=float) * float(angle_increment)
    finite = np.isfinite(scan_ranges)
    valid = finite & (scan_ranges >= float(range_min)) & (scan_ranges <= float(range_max))
    if max_range is not None and max_range > 0.0:
        valid &= scan_ranges <= float(max_range)
    if front_fov_rad is not None and front_fov_rad > 0.0 and front_fov_rad < 2.0 * math.pi:
        valid &= np.abs(wrap_angle(angles)) <= 0.5 * float(front_fov_rad)

    filtered_ranges = scan_ranges[valid]
    filtered_angles = angles[valid]
    x = filtered_ranges * np.cos(filtered_angles)
    y = filtered_ranges * np.sin(filtered_angles)
    points = np.column_stack((x, y)) if filtered_ranges.size else np.empty((0, 2), dtype=float)

    if min_x is not None and points.size:
        keep = points[:, 0] >= float(min_x)
        points = points[keep]
        filtered_angles = filtered_angles[keep]
        filtered_ranges = filtered_ranges[keep]

    return points, filtered_angles, filtered_ranges


def dbscan_labels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Cluster points with a compact DBSCAN implementation.

    Labels are ``-1`` for noise and ``0..N`` for clusters. This implementation is
    O(n^2), which is acceptable for 2D LiDAR scans and keeps runtime predictable
    without external dependencies.
    """

    pts = np.asarray(points, dtype=float)
    n_points = int(pts.shape[0])
    labels = np.full(n_points, -1, dtype=int)
    if n_points == 0:
        return labels

    eps_sq = float(eps) ** 2
    min_samples = max(1, int(min_samples))
    visited = np.zeros(n_points, dtype=bool)
    cluster_id = 0

    neighbors: list[np.ndarray] = []
    for idx in range(n_points):
        diff = pts - pts[idx]
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        neighbors.append(np.flatnonzero(dist_sq <= eps_sq))

    for idx in range(n_points):
        if visited[idx]:
            continue
        visited[idx] = True
        seed_neighbors = neighbors[idx]
        if seed_neighbors.size < min_samples:
            labels[idx] = -1
            continue

        labels[idx] = cluster_id
        seeds = list(int(v) for v in seed_neighbors if int(v) != idx)
        seed_seen = set(seeds)
        cursor = 0
        while cursor < len(seeds):
            point_idx = seeds[cursor]
            cursor += 1
            if not visited[point_idx]:
                visited[point_idx] = True
                point_neighbors = neighbors[point_idx]
                if point_neighbors.size >= min_samples:
                    for candidate in point_neighbors:
                        candidate_i = int(candidate)
                        if candidate_i not in seed_seen:
                            seeds.append(candidate_i)
                            seed_seen.add(candidate_i)
            if labels[point_idx] == -1:
                labels[point_idx] = cluster_id
        cluster_id += 1

    return labels


def circle_from_cluster(
    points: np.ndarray,
    *,
    cluster_id: int,
    radius_padding: float,
    min_radius: float,
) -> ObstacleCircle:
    """Build a conservative circular planner obstacle from clustered points."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        raise ValueError("Cluster points must be a non-empty Nx2 array")

    xy_min = np.min(pts, axis=0)
    xy_max = np.max(pts, axis=0)
    center = 0.5 * (xy_min + xy_max)
    extents = xy_max - xy_min
    raw_radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
    radius = max(float(min_radius), raw_radius + float(radius_padding))
    ranges = np.linalg.norm(pts, axis=1)
    bearing = math.atan2(float(center[1]), float(center[0]))

    return ObstacleCircle(
        center_x=float(center[0]),
        center_y=float(center[1]),
        radius=float(radius),
        point_count=int(pts.shape[0]),
        min_range=float(np.min(ranges)),
        max_range=float(np.max(ranges)),
        bearing=float(bearing),
        width=float(extents[1]),
        depth=float(extents[0]),
        cluster_id=int(cluster_id),
    )


def detect_obstacles_from_scan(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    *,
    max_range: float = 3.5,
    front_fov_rad: float = math.radians(180.0),
    min_x: float = 0.05,
    cluster_eps: float = 0.12,
    min_samples: int = 4,
    radius_padding: float = 0.04,
    min_radius: float = 0.06,
    max_obstacle_radius: float | None = None,
) -> tuple[list[ObstacleCircle], np.ndarray, np.ndarray]:
    """Detect circular obstacles from one LaserScan.

    The returned obstacles are sorted by nearest estimated clearance from the
    LiDAR. ``points`` and ``labels`` are returned for plotting/debugging.
    """

    points, _, _ = ranges_to_points(
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        max_range=max_range,
        front_fov_rad=front_fov_rad,
        min_x=min_x,
    )
    labels = dbscan_labels(points, cluster_eps, min_samples)
    obstacles: list[ObstacleCircle] = []

    for cluster_id in sorted(int(v) for v in set(labels.tolist()) if int(v) >= 0):
        cluster_points = points[labels == cluster_id]
        if cluster_points.shape[0] < max(1, int(min_samples)):
            continue
        obstacle = circle_from_cluster(
            cluster_points,
            cluster_id=cluster_id,
            radius_padding=radius_padding,
            min_radius=min_radius,
        )
        if max_obstacle_radius is not None and obstacle.radius > float(max_obstacle_radius):
            continue
        obstacles.append(obstacle)

    obstacles.sort(key=lambda item: (item.clearance_from_lidar, abs(item.bearing)))
    return obstacles, points, labels

