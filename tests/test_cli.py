"""End to end on a synthetic video, with the detector stubbed (no GPU, no weights download)."""

import json

from vidinfer import __main__ as cli
from vidinfer import model as model_lib
from vidinfer.viz import read_blocks


class StubDetector:
    kwargs = {"imgsz": 1280, "conf": 0.25}

    def __init__(self, weights, device):
        pass

    def warmup(self, height, width):
        pass

    def __call__(self, rgb):
        return [{"bbox_xyxy": [1.0, 2.0, 3.0, 4.0], "score": 0.9, "class_id": 0, "class_name": "person"}], {}


def run_stubbed(monkeypatch, tmp_path, video, *extra):
    monkeypatch.setattr(model_lib, "resolve_weights", lambda name: (tmp_path / "stub.pt", "0" * 64))
    monkeypatch.setattr(model_lib, "Detector", StubDetector)
    out = tmp_path / "out"
    code = cli.main([str(video), "--output-dir", str(out), "--device", "cpu", *extra])
    lines = [json.loads(line) for line in (out / "output.jsonl").read_text().splitlines()]
    return code, lines, (out / "run.log").read_text()


def test_end_to_end_output_and_log(monkeypatch, tmp_path, index_video):
    code, lines, log = run_stubbed(monkeypatch, tmp_path, index_video)
    assert code == 0
    header, frames, summary = lines[0], lines[1:-1], lines[-1]
    assert header["type"] == "header" and header["processing"]["frame_size_hw"] == [720, 1280]
    assert summary["type"] == "summary" and summary["n_inferences"] == summary["expected"] == 100
    assert [f["sample_index"] for f in frames] == list(range(100))
    assert all(abs(f["timestamp_s"] - f["sample_index"] / 10) <= 0.02 + 1e-9 for f in frames)  # half a frame
    blocks = read_blocks(tmp_path / "out" / "run.log")  # the log is a machine-readable data source
    assert len(blocks) == 10  # one performance line per 10 inferences
    assert {"wall_ms", "decode_ms", "preprocess_ms", "inference_ms", "speed", "cpu_util"} <= blocks[0].keys()
    assert "event=timing" in log and "event=fingerprint" in log and "event=video" in log


def test_same_input_gives_same_fingerprints(monkeypatch, tmp_path, index_video):
    _, a, _ = run_stubbed(monkeypatch, tmp_path / "a", index_video)
    _, b, _ = run_stubbed(monkeypatch, tmp_path / "b", index_video)
    for key in ("results_sha256", "frames_crc32"):
        assert a[-1][key] == b[-1][key]


def test_max_frames_stops_early(monkeypatch, tmp_path, index_video):
    code, lines, log = run_stubbed(monkeypatch, tmp_path, index_video, "--max-frames", "25")
    assert code == 0 and lines[-1]["n_inferences"] == 25 and lines[-1]["stopped_early"] is True
    assert log.count("event=block ") == 3  # 10 + 10 + partial 5
    assert "count_mismatch" not in log


def test_missing_input_exits_with_code_2(tmp_path):
    assert cli.main([str(tmp_path / "nope.mp4"), "--output-dir", str(tmp_path / "out")]) == 2
