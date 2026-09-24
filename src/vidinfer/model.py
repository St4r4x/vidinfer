"""Detector: official YOLO26 weights pinned by URL + sha256, fed with RGB frames."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np

# Before importing ultralytics: no telemetry / online checks, never pip-install anything during a run.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "0")

from vidinfer.video import sha256_file  # noqa: E402

log = logging.getLogger("vidinfer")

# Pinned release (Ultralytics otherwise downloads from the *latest* release: not reproducible).
_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
WEIGHTS = {
    "yolo26s.pt": (f"{_RELEASE}/yolo26s.pt", "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"),
    "yolo26m.pt": (f"{_RELEASE}/yolo26m.pt", "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"),
}
PERSON, SPORTS_BALL = 0, 32  # COCO class ids


def resolve_weights(model: str) -> tuple[Path, str]:
    """Return (path, sha256) of the weights, downloading pinned ones if needed.

    A pinned name whose file does not match its sha256 is refused BEFORE loading (a .pt is a pickle:
    loading it can execute code). An arbitrary local file is accepted but flagged as unpinned.
    """
    if model not in WEIGHTS:
        path = Path(model)
        if not path.is_file():
            raise FileNotFoundError(f"unknown model {model!r}: not a file and not one of {sorted(WEIGHTS)}")
        digest = sha256_file(path)
        log.warning(f"event=unpinned_weights path={path} sha256={digest}")
        return path, digest

    url, expected = WEIGHTS[model]
    path = Path(os.environ.get("VIDINFER_WEIGHTS_DIR", Path.home() / ".cache" / "vidinfer")) / model
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"event=download_weights url={url}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    digest = sha256_file(path)
    if digest != expected:
        raise RuntimeError(f"sha256 mismatch for {path}: got {digest}, expected {expected}")
    return path, digest


class Detector:
    """Person + ball detector. Takes RGB uint8 frames; returns plain-dict detections and Ultralytics stage times."""

    def __init__(self, weights: Path, device: str, imgsz: int = 1280, conf: float = 0.25) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        # imgsz 1280: keep the 720p pixels (the ball is 8-16 px); FP16 on GPU only.
        self.kwargs = dict(imgsz=imgsz, conf=conf, classes=[PERSON, SPORTS_BALL], device=device, verbose=False)
        if device != "cpu":
            self.kwargs["quantize"] = 16

    def __call__(self, rgb: np.ndarray) -> tuple[list[dict], dict[str, float]]:
        # Ultralytics treats numpy input as BGR (OpenCV convention) and flips it to RGB internally
        # (ultralytics/engine/predictor.py, preprocess). Feeding RGB as-is would silently swap channels.
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        result = self.model.predict(bgr, **self.kwargs)[0]
        boxes = result.boxes
        detections = [
            {
                "bbox_xyxy": [round(v, 1) for v in xyxy],
                "score": round(score, 4),
                "class_id": int(cls),
                "class_name": result.names[int(cls)],
            }
            for xyxy, cls, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=True)
        ]  # .tolist() copies to host, i.e. waits for the GPU
        return detections, result.speed

    def warmup(self, height: int, width: int, n: int = 2) -> None:
        """First calls pay CUDA context / cuDNN autotuning: keep them out of the measured loop."""
        for _ in range(n):
            self(np.zeros((height, width, 3), np.uint8))
