"""The model boundary: Ultralytics must receive BGR, since it treats numpy input as BGR."""

import numpy as np
import pytest

from vidinfer.model import Detector, resolve_weights


class FakeBoxes:
    xyxy = cls = conf = np.zeros((0, 4))


class FakeResult:
    boxes, names, speed = FakeBoxes(), {}, {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}


class FakeYOLO:
    def predict(self, image, **kwargs):
        self.received = image
        return [FakeResult()]


def test_detector_feeds_bgr_to_ultralytics():
    detector = Detector.__new__(Detector)  # skip weight loading
    detector.model, detector.kwargs = FakeYOLO(), {}
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[..., 0] = 255  # pure red in RGB
    detections, speed = detector(rgb)
    assert detector.model.received[..., 2].min() == 255  # red is the LAST channel in BGR
    assert detector.model.received[..., 0].max() == 0
    assert detections == [] and speed["inference"] == 2.0


def test_weights_with_a_wrong_sha256_are_refused_before_loading(monkeypatch, tmp_path):
    (tmp_path / "yolo26s.pt").write_bytes(b"not the pinned weights")  # e.g. a corrupted cache or a swapped file
    monkeypatch.setenv("VIDINFER_WEIGHTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        resolve_weights("yolo26s.pt")
