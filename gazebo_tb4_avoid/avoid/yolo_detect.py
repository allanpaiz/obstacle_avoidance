#!/usr/bin/env python3
"""Run YOLO on every turtlebot4_1 camera frame and write yolo_detections.jsonl.

This is an offline pass over saved frames and runs inside the simulation image,
with the weights the image was built with. render.py draws the boxes on the
camera panel when the output exists.

The stock models are trained on COCO, which has no barrel class; the barrel
comes back as the nearest COCO class (usually "cup"), and the video shows
every detection as an obstacle with that raw class beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default=os.environ.get("YOLO_MODEL", "yolo26n.pt"),
                        help="Ultralytics weights (default: $YOLO_MODEL, baked into the image)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--robot", default="turtlebot4_1")
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.run_dir / "camera_frames.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["robot"] == args.robot]
    model = YOLO(args.model)
    started = time.perf_counter()
    hits = 0
    with (args.run_dir / "yolo_detections.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            result = model(str(args.run_dir / row["path"]), conf=args.conf, verbose=False)[0]
            boxes = [{"label": result.names[int(box.cls)], "conf": round(float(box.conf), 3),
                      "xyxy": [round(float(v), 1) for v in box.xyxy[0]]} for box in result.boxes]
            hits += bool(boxes)
            handle.write(json.dumps({"path": row["path"], "t_s": row["t_s"], "model": Path(args.model).name,
                                     "width": row["width"], "height": row["height"], "boxes": boxes}) + "\n")
    elapsed = time.perf_counter() - started
    print(f"[yolo] {Path(args.model).name}: {len(rows)} frames, {hits} with detections, "
          f"{1000 * elapsed / max(1, len(rows)):.0f} ms/frame -> {args.run_dir / 'yolo_detections.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
