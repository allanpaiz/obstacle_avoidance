#!/usr/bin/env python3
"""Render the avoidance run as videos: a 3x1 overview | robot camera | lidar
composite, plus each of the three panels on its own.

The lidar panel is the robot's own frame (forward up), with the detection and
plan drawn over it: candidate clusters while detecting, then the planned path,
the detected obstacle, and its inflated footprint, and the path driven so far.

  python3 -m avoid.render /outputs      # in the container
  ./run_avoid.sh --render RUN_DIR       # from the host
"""

from __future__ import annotations

import argparse
import bisect
import functools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import spec as run
from .video import VideoWriter


BACKGROUND = "#e8edf1"
INK = "#182b40"
MUTED = "#5b6b7c"
ACCENT = "#147d92"
HEADER_HEIGHT = 40
GAP = 8
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


@functools.lru_cache(maxsize=None)
def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(FONT_PATHS[1 if bold else 0], size)
    except OSError:
        return ImageFont.load_default()


def even(value: int) -> int:
    return value + (value % 2)


def rgb(rgba: str) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, round(255 * float(c))))) for c in rgba.split()[:3])


def rgb_tint(color, amount=0.55):
    """Blend a colour towards white, for path trails under the live markers."""
    return tuple(int(c + (255 - c) * amount) for c in color)


def output_times(duration: float, fps: float, speed: float):
    """Simulation times shown by each output frame: `speed` sim seconds per video second."""
    step = speed / fps
    return [index * step for index in range(max(1, math.ceil(duration / step - 1e-9)))]


def draw_header(draw: ImageDraw.ImageDraw, width: int, title: str, t: float,
                duration: float, speed: float) -> None:
    draw.rectangle((0, 0, width, HEADER_HEIGHT - 1), fill="#ffffff")
    draw.text((12, 11), title, fill=INK, font=font(15, bold=True))
    clock = f"sim t = {t:6.1f} / {duration:g} s"
    if speed != 1:
        clock += f"  ({speed:g}x)"
    clock_font = font(14)
    clock_width = draw.textlength(clock, font=clock_font)
    draw.text((width - 12 - clock_width, 12), clock, fill=MUTED, font=clock_font)
    progress = min(1.0, t / duration) if duration > 0 else 0.0
    draw.rectangle((0, HEADER_HEIGHT - 4, width, HEADER_HEIGHT - 1), fill="#d5dde5")
    draw.rectangle((0, HEADER_HEIGHT - 4, round(width * progress), HEADER_HEIGHT - 1), fill=ACCENT)


def label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    label_font = font(12)
    width = draw.textlength(text, font=label_font)
    draw.rectangle((x, y, x + width + 10, y + 18), fill=(24, 43, 64))
    draw.text((x + 5, y + 2), text, fill="#ffffff", font=label_font)


def draw_robot(draw, center, radius_px, heading, color, outline=INK):
    x, y = center
    draw.ellipse((x - radius_px, y - radius_px, x + radius_px, y + radius_px), fill=color, outline=outline, width=2)
    draw.line((x, y, x + math.cos(heading) * radius_px, y - math.sin(heading) * radius_px), fill="#ffffff", width=3)


def plot_points(pixels: np.ndarray, us, vs, color, bounds) -> None:
    """Paint 2x2-pixel dots straight into an RGB array, inside bounds (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = bounds
    us, vs = np.floor(us).astype(int), np.floor(vs).astype(int)
    for du in (0, 1):
        for dv in (0, 1):
            u, v = us + du, vs + dv
            keep = (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
            pixels[v[keep], u[keep]] = color


def read_frames(input_path: Path, robot: str) -> list[tuple[float, Path]]:
    """(t_s, PNG path) for every frame `robot` recorded, in time order."""
    frames = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["robot"] == robot:
                frames.append((float(row["t_s"]), input_path.parent / row["path"]))
    if not frames:
        raise ValueError(f"No camera frames for {robot} in {input_path}")
    return sorted(frames)


def read_scans(input_path: Path, robot: str):
    """(t_s, Nx2 points in the lidar frame, ray count) for every scan of `robot`, in time order."""
    scans = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["robot"] != robot:
            continue
        ranges = np.asarray(row["ranges"], dtype=float)
        angles = float(row["angle_min"]) + np.arange(len(ranges)) * float(row["angle_step"])
        valid = (np.isfinite(ranges) & (ranges >= float(row["range_min"]))
                 & (ranges < float(row["range_max"])))
        points = np.column_stack((ranges[valid] * np.cos(angles[valid]),
                                  ranges[valid] * np.sin(angles[valid])))
        scans.append((float(row["t_s"]), points, len(ranges)))
    if not scans:
        raise ValueError(f"No lidar scans for {robot} in {input_path}")
    scans.sort(key=lambda item: item[0])
    return scans


class PoseTrack:
    """Ground-truth pose samples, interpolated at any time in the window."""

    def __init__(self, input_path: Path):
        times, samples = [], []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                times.append(float(row["t_s"]))
                samples.append(row["poses"])
        if not times:
            raise ValueError(f"No poses in {input_path}")
        self.times = np.asarray(times)
        self.tracks = {}
        for name in set().union(*samples):
            values = np.array([sample.get(name, [np.nan] * 4) for sample in samples], dtype=float)
            known = ~np.isnan(values[:, 0])
            if known.sum() < 1:
                continue
            values = values[known]
            values[:, 3] = np.unwrap(values[:, 3])
            self.tracks[name] = (self.times[known], values)

    def at(self, name: str, t: float):
        """(x, y, z, yaw) at time t, or None for an untracked entity."""
        if name not in self.tracks:
            return None
        times, values = self.tracks[name]
        return tuple(float(np.interp(t, times, values[:, column])) for column in range(4))

    def history(self, name: str, start: float, end: float):
        times, values = self.tracks.get(name, (np.empty(0), np.empty((0, 4))))
        low, high = np.searchsorted(times, start, "left"), np.searchsorted(times, end, "right")
        return values[low:high, :2]


PANEL_H = 480
OVERVIEW_W = even(round(run.OVERVIEW_WIDTH * PANEL_H / run.OVERVIEW_HEIGHT))
CAMERA_W = round(run.CAMERA_WIDTH * PANEL_H / run.CAMERA_HEIGHT)
LIDAR_W = PANEL_H
CAPTION_H = 26
SCALE = 100.0                      # lidar panel pixels per metre
ROBOT_PX = (LIDAR_W / 2, PANEL_H - 140)   # robot sits low in the panel: 3.4 m ahead, 1.4 m behind
COLOR = rgb(run.ROBOT_COLOR)
PATH_COLOR = (20, 125, 146)
OLD_PATH_COLOR = (157, 99, 47)
OBSTACLE_RED = (205, 52, 42)
CANDIDATE_ORANGE = (236, 140, 30)
STEPS = (("hold", "Hold"), ("detect", "Detect"), ("plan", "Plan"), ("execute", "Execute"), ("done", "Goal reached"))
TITLE = "TurtleBot4 lidar obstacle avoidance (Gazebo demo arena)"


def read_jsonl(path: Path, start: float):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    for row in rows:
        row["t"] = row["sim_time_s"] - start
    return rows


def latest(rows, t):
    index = bisect.bisect_right([row["t"] for row in rows], t + 1e-6) - 1
    return rows[index] if index >= 0 else None


class LidarPanel:
    def __init__(self, run_dir: Path, start: float):
        self.scans = read_scans(run_dir / "scans.jsonl", run.ROBOT)
        self.scan_times = [row[0] for row in self.scans]
        self.track = PoseTrack(run_dir / "poses.jsonl")
        history = self.track.history(run.ROBOT, 0.0, math.inf)
        self.old_trajectory = history[[0, -1]] if len(history) > 1 else None
        self.events = read_jsonl(run_dir / "avoid_events.jsonl", start)
        self.detections = read_jsonl(run_dir / "detections.jsonl", start)
        self.plan = None
        plan_path = run_dir / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            detect_t = next(e["t"] for e in self.events if e["phase"] == "detect")
            x0, y0, _, yaw0 = self.track.at(run.ROBOT, detect_t)
            to_world = lambda pts: np.array([(x0 + math.cos(yaw0) * px - math.sin(yaw0) * py,
                                              y0 + math.sin(yaw0) * px + math.cos(yaw0) * py) for px, py in pts])
            self.plan = {
                "t": next(e["t"] for e in self.events if e["phase"] == "plan"),
                "path": to_world(plan["path"]),
                "obstacle": to_world([plan["obstacle"]["center"]])[0],
                "radius": plan["obstacle"]["radius"],
                "clearance": plan["clearance"],
                "ahead": plan["obstacle"]["min_range"],
            }
        self.base = Image.new("RGB", (LIDAR_W, PANEL_H), "#f7f9fb")
        draw = ImageDraw.Draw(self.base)
        cx, cy = ROBOT_PX
        for ring in (1, 2, 3):
            r = ring * SCALE
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="#dde3e9", width=1)
            draw.text((cx + 4, cy - r + 2), f"{ring} m", fill="#9aa7b4", font=font(10))
        draw.line((cx, 0, cx, PANEL_H), fill="#eef1f4")
        draw.line((0, cy, LIDAR_W, cy), fill="#eef1f4")
        self.min_gap = math.inf

    def phase(self, t):
        event = latest(self.events, t)
        return event["phase"] if event else "hold"

    def to_panel(self, pose, world):
        """World points -> panel pixels, in the robot frame at `pose` (forward up)."""
        x, y, _, yaw = pose
        world = np.atleast_2d(world)
        dx, dy = world[:, 0] - x, world[:, 1] - y
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        left = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return np.column_stack((ROBOT_PX[0] - left * SCALE, ROBOT_PX[1] - forward * SCALE))

    def circle(self, draw, center, radius, **style):
        u, v = center
        r = radius * SCALE
        draw.ellipse((u - r, v - r, u + r, v + r), **style)

    def gap(self, pose):
        """Ground-truth free space between the robot body and the barrel."""
        return (math.hypot(pose[0] - run.BARREL_XY[0], pose[1] - run.BARREL_XY[1])
                - run.BARREL_RADIUS - run.ROBOT_RADIUS)

    def frame(self, t):
        index = bisect.bisect_right(self.scan_times, t + 1e-6) - 1
        scan_t, points, count = self.scans[index] if index >= 0 else (t, np.empty((0, 2)), 0)
        pose = self.track.at(run.ROBOT, scan_t)
        phase = self.phase(t)
        canvas = self.base.copy()
        draw = ImageDraw.Draw(canvas)
        cx, cy = ROBOT_PX

        if phase in ("hold", "detect"):
            half = run.CORRIDOR_WIDTH / 2 * SCALE
            draw.rectangle((cx - half, 0, cx + half, cy), fill="#dcf1e2")
        planned = self.plan is not None and t >= self.plan["t"]
        if self.old_trajectory is not None:
            old_path = self.to_panel(pose, self.old_trajectory)
            draw.line([tuple(p) for p in old_path], fill=OLD_PATH_COLOR, width=3)
        if planned:
            obstacle = self.to_panel(pose, self.plan["obstacle"])[0]
            self.circle(draw, obstacle, self.plan["clearance"] - run.ROBOT_RADIUS, fill="#fde6cf", outline="#f0b47a")
            self.circle(draw, obstacle, self.plan["radius"], fill="#f6c3bc", outline=OBSTACLE_RED, width=2)
            path = self.to_panel(pose, self.plan["path"])
            draw.line([tuple(p) for p in path], fill=PATH_COLOR, width=3)
            gu, gv = path[-1]
            draw.ellipse((gu - 7, gv - 7, gu + 7, gv + 7), outline=PATH_COLOR, width=3)
            draw.text((gu + 10, gv - 8), "goal", fill=PATH_COLOR, font=font(12, bold=True))
        trail = self.track.history(run.ROBOT, 0.0, scan_t)
        if len(trail) > 1:
            draw.line([tuple(p) for p in self.to_panel(pose, trail)], fill=rgb_tint(COLOR, 0.45), width=4)

        pixels = np.array(canvas)
        if len(points):
            plot_points(pixels, cx - points[:, 1] * SCALE, cy - points[:, 0] * SCALE, COLOR,
                        (0, 0, LIDAR_W, PANEL_H))
        canvas = Image.fromarray(pixels)
        draw = ImageDraw.Draw(canvas)

        detection = latest(self.detections, t) if phase == "detect" else None
        if detection:
            for i, c in enumerate(detection["candidates"]):
                selected = i == detection["selected"]
                self.circle(draw, (cx - c["y"] * SCALE, cy - c["x"] * SCALE), c["r"],
                            outline=OBSTACLE_RED if selected else CANDIDATE_ORANGE, width=3 if selected else 2)
        draw_robot(draw, (cx, cy), run.ROBOT_RADIUS * SCALE, math.pi / 2, COLOR)

        self.status(draw, t, phase, pose, detection)
        self.legend(draw, planned)
        draw.rectangle((0, 0, LIDAR_W - 1, PANEL_H - 1), outline="#b8c2cc")
        return canvas

    def status(self, draw, t, phase, pose, detection):
        lines = []
        if phase == "hold":
            lines.append("standing still")
        elif phase == "detect":
            n = sum(1 for d in self.detections if d["t"] <= t + 1e-6)
            lines.append(f"clustering scan {n}")
            if detection and detection["selected"] is not None:
                c = detection["candidates"][detection["selected"]]
                lines.append(f"corridor obstacle {math.hypot(c['x'], c['y']):.2f} m")
        if self.plan is not None and phase in ("plan", "execute", "done"):
            lines.append(f"obstacle {self.plan['ahead']:.2f} m ahead, r {self.plan['radius']:.2f} m")
        if phase in ("execute", "done"):
            gap = self.gap(pose)
            self.min_gap = min(self.min_gap, gap)
            lines.append(f"gap to barrel {gap:.2f} m (min {self.min_gap:.2f})")
        step = next(i for i, (key, _) in enumerate(STEPS) if key == phase)
        chip = f"{step + 1}  {STEPS[step][1].upper()}"
        width = draw.textlength(chip, font=font(13, bold=True))
        fill = (19, 128, 72) if phase == "done" else (24, 43, 64)
        draw.rounded_rectangle((8, 8, 8 + width + 16, 32), radius=5, fill=fill)
        draw.text((16, 12), chip, fill="#ffffff", font=font(13, bold=True))
        for i, text in enumerate(lines):
            draw.text((10, 40 + 17 * i), text, fill=INK, font=font(12), stroke_width=2, stroke_fill="#f7f9fb")

    def legend(self, draw, planned):
        items = [("dots", COLOR, "lidar returns"), ("line", rgb_tint(COLOR, 0.45), "path driven"),
                 ("line", OLD_PATH_COLOR, "old trajectory")]
        if planned:
            items += [("line", PATH_COLOR, "planned path"), ("ring", OBSTACLE_RED, "detected obstacle"),
                      ("ring", (240, 180, 122), "+ robot radius & margin")]
        else:
            items += [("ring", OBSTACLE_RED, "selected cluster"), ("ring", CANDIDATE_ORANGE, "other clusters")]
        y = PANEL_H - 10 - 16 * len(items)
        draw.rounded_rectangle((LIDAR_W - 176, y - 6, LIDAR_W - 6, PANEL_H - 8), radius=4,
                               fill="#ffffff", outline="#dde3e9")
        for kind, color, text in items:
            x = LIDAR_W - 168
            if kind == "dots":
                for dx in (2, 8, 14):
                    draw.rectangle((x + dx, y + 6, x + dx + 2, y + 8), fill=color)
            elif kind == "line":
                draw.line((x, y + 7, x + 18, y + 7), fill=color, width=3)
            else:
                draw.ellipse((x + 3, y + 1, x + 15, y + 13), outline=color, width=2)
            draw.text((x + 24, y), text, fill=MUTED, font=font(11))
            y += 16


class YoloOverlay:
    """YOLO boxes from yolo_detect.py, each checked against the lidar.

    A box counts as an obstacle when the ground distance implied by its bottom
    edge (camera height and pitch, assuming the object stands on the floor)
    matches the nearest lidar return inside the box's horizontal span. Anything
    else, such as a floor marking, is drawn grey as unconfirmed.
    """

    def __init__(self, run_dir: Path, lidar: LidarPanel):
        path = run_dir / "yolo_detections.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        self.rows = {row["path"]: row for row in rows}
        self.model = Path(rows[0]["model"]).stem.replace("yolo", "YOLO") if rows else None
        self.lidar = lidar
        cam = [float(v) for v in run.CAMERA_POSE.split()]
        self.cam_x, self.cam_z, self.pitch = cam[0], cam[2], cam[4]
        self.lidar_x = float(run.LIDAR_POSE.split()[0])
        self.focal = run.CAMERA_WIDTH / 2 / math.tan(run.CAMERA_HFOV / 2)
        self.stats = {}

    def fuse(self, box, t):
        """Lidar range to the boxed object if the camera and lidar agree, else None."""
        u0, _, u1, v1 = box["xyxy"]
        cx, cy = run.CAMERA_WIDTH / 2, run.CAMERA_HEIGHT / 2
        left, right = math.atan((cx - u0) / self.focal), math.atan((cx - u1) / self.focal)
        index = max(0, bisect.bisect_right(self.lidar.scan_times, t + 1e-6) - 1)
        points = self.lidar.scans[index][1]
        if not len(points):
            return None
        forward = points[:, 0] + self.lidar_x - self.cam_x      # lidar frame -> camera position
        bearing = np.arctan2(points[:, 1], forward)
        keep = ((np.hypot(points[:, 0], points[:, 1]) > run.MIN_X) & (forward > 0.05)
                & (bearing <= left) & (bearing >= right))
        below = self.pitch + math.atan((v1 - cy) / self.focal)
        if not keep.any() or below <= 0.01:
            return None
        lidar_range = float(forward[keep].min())
        ground = self.cam_z / math.tan(below)
        return lidar_range if abs(ground - lidar_range) <= max(0.25, 0.35 * lidar_range) else None

    def draw(self, image: Image.Image, key: str, t: float) -> None:
        row = self.rows.get(key)
        if row is None:
            return
        draw = ImageDraw.Draw(image)
        sx, sy = image.width / row["width"], image.height / row["height"]
        for box in row["boxes"]:
            lidar_range = self.fuse(box, t)
            tag = f"{box['label']} {box['conf']:.2f}"
            self.stats.setdefault(box["label"], [0, 0])[lidar_range is None] += 1
            x0, y0, x1, y1 = (box["xyxy"][0] * sx, box["xyxy"][1] * sy, box["xyxy"][2] * sx, box["xyxy"][3] * sy)
            if lidar_range is not None:
                draw.rectangle((x0, y0, x1, y1), outline=OBSTACLE_RED, width=3)
                lines = ((f"obstacle  {lidar_range:.2f} m", OBSTACLE_RED, font(13, bold=True)),
                         (f"{self.model}: {tag}, lidar match", (24, 43, 64), font(11)))
            else:
                draw.rectangle((x0, y0, x1, y1), outline="#8a96a3", width=1)
                lines = ((f"{self.model}: {tag}, no lidar match", "#8a96a3", font(11)),)
            y = y0 - 21 * len(lines) - 2 if y0 > 21 * len(lines) + 40 else y1 + 3
            for text, fill, face in lines:
                w = draw.textlength(text, font=face)
                bx = min(max(x0, 0), image.width - w - 10)
                draw.rectangle((bx, y, bx + w + 10, y + 19), fill=fill)
                draw.text((bx + 5, y + 2), text, fill="#ffffff", font=face)
                y += 21


def draw_steps(draw, width, phase):
    """Phase stepper in the header, centred between the title and the clock."""
    current = next(i for i, (key, _) in enumerate(STEPS) if key == phase)
    sizes = [draw.textlength(name, font=font(13, bold=True)) + 20 for _, name in STEPS]
    x = (width - sum(sizes) - 18 * (len(STEPS) - 1)) / 2
    for i, ((_, name), size) in enumerate(zip(STEPS, sizes)):
        if i == current:
            fill, ink = ((19, 128, 72) if i == len(STEPS) - 1 else (20, 125, 146)), "#ffffff"
        elif i < current:
            fill, ink = "#d5e9ee", INK
        else:
            fill, ink = "#eef1f4", "#9aa7b4"
        draw.rounded_rectangle((x, 8, x + size, 30), radius=11, fill=fill)
        draw.text((x + 10, 11), name, fill=ink, font=font(13, bold=True))
        x += size
        if i < len(STEPS) - 1:
            draw.text((x + 5, 10), "›", fill="#9aa7b4", font=font(14, bold=True))
            x += 18


def single_panel_base(panel_width, caption):
    top = HEADER_HEIGHT + GAP
    width = even(panel_width + 2 * GAP)
    height = even(top + PANEL_H + CAPTION_H)
    base = Image.new("RGB", (width, height), BACKGROUND)
    ImageDraw.Draw(base).text((GAP + 2, top + PANEL_H + 6), caption, fill=MUTED, font=font(12))
    return base, top


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    run_dir = args.run_dir
    summary = json.loads((run_dir / "summary.json").read_text())
    duration, start = float(summary["duration_sim_s"]), float(summary["sim_time_start_s"])

    lidar = LidarPanel(run_dir, start)
    yolo = YoloOverlay(run_dir, lidar)
    overview = read_frames(run_dir / "simulation_frames.jsonl", "overview")
    camera = read_frames(run_dir / "camera_frames.jsonl", run.ROBOT)
    feeds = ((overview, (GAP, OVERVIEW_W), "Overview camera",
              f"Fixed overview camera, {run.OVERVIEW_WIDTH}x{run.OVERVIEW_HEIGHT} at "
              f"{run.OVERVIEW_RATE:g} Hz"),
             (camera, (2 * GAP + OVERVIEW_W, CAMERA_W), f"{run.ROBOT} camera",
              f"{run.ROBOT} forward camera, {run.CAMERA_WIDTH}x{run.CAMERA_HEIGHT} at {run.CAMERA_RATE:g} Hz"
              + (f"; {yolo.model} boxes checked against the lidar" if yolo.model else "")))
    lidar_x = 3 * GAP + OVERVIEW_W + CAMERA_W
    width = even(lidar_x + LIDAR_W + GAP)
    top = HEADER_HEIGHT + GAP
    height = even(top + PANEL_H + CAPTION_H)

    base = Image.new("RGB", (width, height), BACKGROUND)
    base_draw = ImageDraw.Draw(base)
    captions = [(x, text) for _, (x, _), _, text in feeds]
    captions.append((lidar_x, f"{run.ROBOT} lidar in the robot frame (forward up), "
                              f"{run.LIDAR_SAMPLES} rays at {run.LIDAR_RATE:g} Hz"))
    for x, text in captions:
        base_draw.text((x + 2, top + PANEL_H + 6), text, fill=MUTED, font=font(12))

    shown = {}
    simulation_base, simulation_top = single_panel_base(OVERVIEW_W, captions[0][1])
    camera_base, camera_top = single_panel_base(CAMERA_W, captions[1][1])
    lidar_base, lidar_top = single_panel_base(LIDAR_W, captions[2][1])
    writer = VideoWriter(run_dir / "avoidance_3x1.mp4", run.LIDAR_RATE)
    simulation_writer = VideoWriter(run_dir / "avoidance_simulation.mp4", run.LIDAR_RATE)
    camera_writer = VideoWriter(run_dir / "avoidance_camera.mp4", run.LIDAR_RATE)
    lidar_writer = VideoWriter(run_dir / "avoidance_lidar.mp4", run.LIDAR_RATE)
    for t in output_times(duration, run.LIDAR_RATE, args.speed):
        canvas = base.copy()
        draw = ImageDraw.Draw(canvas)
        for frames, (x, w), name, _ in feeds:
            index = bisect.bisect_right([row[0] for row in frames], t + 1e-6) - 1
            if index < 0:
                continue
            if shown.get(name, (None,))[0] != index:
                with Image.open(frames[index][1]) as source:
                    panel = source.convert("RGB").resize((w, PANEL_H), Image.LANCZOS)
                if frames is camera:
                    yolo.draw(panel, str(frames[index][1].relative_to(run_dir)), frames[index][0])
                shown[name] = (index, panel)
            canvas.paste(shown[name][1], (x, top))
            label(draw, x + 8, top + 8, name)
        lidar_panel = lidar.frame(t)
        canvas.paste(lidar_panel, (lidar_x, top))
        draw_header(draw, width, TITLE, t, duration, args.speed)
        draw_steps(draw, width, lidar.phase(t))
        writer.write(canvas)
        simulation = simulation_base.copy()
        if "Overview camera" in shown:
            simulation.paste(shown["Overview camera"][1], (GAP, simulation_top))
        simulation_draw = ImageDraw.Draw(simulation)
        draw_header(simulation_draw, simulation.width, "TurtleBot4 simulation overview", t, duration, args.speed)
        draw_steps(simulation_draw, simulation.width, lidar.phase(t))
        simulation_writer.write(simulation)

        camera_video = camera_base.copy()
        if f"{run.ROBOT} camera" in shown:
            camera_video.paste(shown[f"{run.ROBOT} camera"][1], (GAP, camera_top))
        camera_draw = ImageDraw.Draw(camera_video)
        draw_header(camera_draw, camera_video.width, f"{run.ROBOT} camera", t, duration, args.speed)
        draw_steps(camera_draw, camera_video.width, lidar.phase(t))
        camera_writer.write(camera_video)

        lidar_video = lidar_base.copy()
        lidar_video.paste(lidar_panel, (GAP, lidar_top))
        lidar_draw = ImageDraw.Draw(lidar_video)
        draw_header(lidar_draw, lidar_video.width, f"{run.ROBOT} lidar", t, duration, args.speed)
        draw_steps(lidar_draw, lidar_video.width, lidar.phase(t))
        lidar_writer.write(lidar_video)
    path = writer.close()
    outputs = [(path, writer),
               (simulation_writer.close(), simulation_writer),
               (camera_writer.close(), camera_writer),
               (lidar_writer.close(), lidar_writer)]
    for output, output_writer in outputs:
        print(f"[avoid-render] {output} ({output_writer.frames} frames, "
              f"{output.stat().st_size / 1e6:.1f} MB)", flush=True)
    for name, (confirmed, rejected) in sorted(yolo.stats.items()):
        print(f"[avoid-render] {yolo.model} '{name}': {confirmed} boxes matched the lidar, {rejected} did not",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
