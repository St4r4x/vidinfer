"""Synthetic H.264 test videos: tests never depend on the 184 MB input and run on CPU in seconds."""

from pathlib import Path

import av
import numpy as np
import pytest

BITS = 10  # frame index encoded as 10 vertical black/white bands (robust to lossy H.264 and to resizing)


def encode_index(i: int, height: int = 180, width: int = 320) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    band = width // BITS
    for b in range(BITS):
        if (i >> b) & 1:
            img[:, b * band : (b + 1) * band] = 255
    return img


def decode_index(rgb: np.ndarray) -> int:
    band = rgb.shape[1] // BITS
    return sum(1 << b for b in range(BITS) if rgb[:, b * band + 2 : (b + 1) * band - 2].mean() > 127)


def write_video(path: Path, frames: list[np.ndarray], fps: int = 25) -> Path:
    """H.264 yuv420p, 1 keyframe per second, no B-frames: same structure as cut.mp4."""
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.height, s.width = frames[0].shape[:2]
        s.pix_fmt = "yuv420p"
        s.options = {"g": str(fps), "bf": "0"}
        for img in frames:
            c.mux(s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
        c.mux(s.encode())
    return path


@pytest.fixture
def index_video(tmp_path: Path) -> Path:
    """10 s at 25 fps (250 frames) -> 100 samples at 10 fps."""
    return write_video(tmp_path / "index.mp4", [encode_index(i) for i in range(250)])


@pytest.fixture
def red_video(tmp_path: Path) -> Path:
    red = np.zeros((180, 320, 3), np.uint8)
    red[..., 0] = 255
    return write_video(tmp_path / "red.mp4", [red] * 5)
