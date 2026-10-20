"""Stream PIL frames into an H.264 MP4 through ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image


class VideoWriter:
    """Write equally spaced frames at `fps`; ffmpeg starts on the first frame."""

    def __init__(self, path: Path, fps: float):
        self.path = Path(path)
        self.fps = float(fps)
        self.size = None
        self.frames = 0
        self.process = None

    def write(self, image: Image.Image) -> None:
        image = image.convert("RGB")
        if self.size is None:
            self.size = image.size
            width, height = image.size
            self.process = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{self.fps:g}",
                 "-i", "-", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(self.path)],
                stdin=subprocess.PIPE,
            )
        elif image.size != self.size:
            raise ValueError(f"frame size {image.size} differs from {self.size}")
        try:
            self.process.stdin.write(image.tobytes())
        except BrokenPipeError:
            raise RuntimeError(f"ffmpeg exited with code {self.process.wait()} while writing "
                               f"{self.path}; see its message above") from None
        self.frames += 1

    def close(self) -> Path:
        if self.process is None:
            raise RuntimeError(f"no frames were written to {self.path}")
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {self.process.returncode} for {self.path}")
        return self.path
