"""End to end on synthetic videos, with the detector stubbed (no GPU, no weights download)."""

import json

import pytest
from conftest import encode_index, write_video

from vidinfer import __main__ as cli
from vidinfer import model as model_lib
from vidinfer import viz


class StubDetector:
    kwargs = {"imgsz": 1280, "conf": 0.25}

    def __init__(self, weights, device):
        pass

    def warmup(self, rgb):
        pass

    def __call__(self, rgb):
        return [{"bbox_xyxy": [1.0, 2.0, 3.0, 4.0], "score": 0.9, "class_id": 0, "class_name": "person"}], {}


class FailingDetector(StubDetector):
    def __call__(self, rgb):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


def run_stubbed(monkeypatch, tmp_path, video, *extra, detector=StubDetector):
    monkeypatch.setattr(model_lib, "resolve_weights", lambda name: (tmp_path / "stub.pt", "0" * 64))
    monkeypatch.setattr(model_lib, "Detector", detector)
    out = tmp_path / "out"
    code = cli.main([str(video), "--output-dir", str(out), "--device", "cpu", *extra])
    lines = [json.loads(line) for line in (out / "output.jsonl").read_text().splitlines()]
    return code, lines, (out / "run.log").read_text()


def test_end_to_end_output_and_log(monkeypatch, tmp_path, index_video):
    code, lines, log = run_stubbed(monkeypatch, tmp_path, index_video)
    assert code == 0
    header, frames, summary = lines[0], lines[1:-1], lines[-1]
    assert header["type"] == "header" and header["processing"]["frame_size_hw"] == [720, 1280]
    assert summary["status"] == "complete" and summary["n_inferences"] == summary["expected"] == 100
    assert [f["sample_index"] for f in frames] == list(range(100))
    expected_sources = [(5 * k) // 2 for k in range(100)]  # nearest frame to k/10 s at 25 fps, ties -> earlier
    assert [f["source_frame_index"] for f in frames] == expected_sources
    assert [f["timestamp_s"] for f in frames] == [round(i / 25, 6) for i in expected_sources]
    blocks = viz.read_blocks(tmp_path / "out" / "run.log")  # the log is a machine-readable data source
    assert len(blocks) == 10  # one performance line per 10 inferences
    assert {"wall_ms", "decode_ms", "preprocess_ms", "inference_ms", "speed", "cpu_cores"} <= blocks[0].keys()
    assert "event=timing" in log and "event=fingerprint" in log and "event=video" in log
    assert log.startswith("ts=") and "Z level=INFO event=start" in log.splitlines()[0]  # UTC logfmt


def test_expected_count_matches_the_sampler(monkeypatch, tmp_path):
    # 251 frames at 25 fps: the sampler emits 101 samples; int(duration * fps) would say 100.
    video = write_video(tmp_path / "v251.mp4", [encode_index(i) for i in range(251)])
    code, lines, _ = run_stubbed(monkeypatch, tmp_path, video)
    assert code == 0 and lines[-1]["status"] == "complete" and lines[-1]["n_inferences"] == lines[-1]["expected"] == 101


def test_corrupt_video_is_partial_not_success(monkeypatch, tmp_path, corrupt_video):
    # The sampler fills gaps with the nearest decoded frame, so n == expected can still hold:
    # the count alone cannot tell that frames were lost.
    code, lines, log = run_stubbed(monkeypatch, tmp_path, corrupt_video)
    summary = lines[-1]
    assert code == 3 and summary["status"] == "partial"
    assert summary["decode_errors"] > 0 and summary["duplicate_samples"] > 0
    assert "level=ERROR event=summary status=partial" in log


def test_hd_untagged_video_uses_bt709(monkeypatch, tmp_path, hd_video):
    code, lines, log = run_stubbed(monkeypatch, tmp_path, hd_video)
    assert code == 0 and lines[0]["processing"]["yuv_matrix"] == "ITU709"
    assert "event=color_untagged yuv_matrix=ITU709" in log


def test_fingerprints_are_stable_and_sensitive(monkeypatch, tmp_path, index_video, red_video):
    _, a, _ = run_stubbed(monkeypatch, tmp_path / "a", index_video)
    _, b, _ = run_stubbed(monkeypatch, tmp_path / "b", index_video)
    _, c, _ = run_stubbed(monkeypatch, tmp_path / "c", red_video)
    for key in ("results_sha256", "frames_crc32"):
        assert a[-1][key] == b[-1][key]  # same input -> same fingerprint
        assert a[-1][key] != c[-1][key]  # other input -> other fingerprint


def test_max_frames_stops_early(monkeypatch, tmp_path, index_video):
    code, lines, log = run_stubbed(monkeypatch, tmp_path, index_video, "--max-frames", "25")
    assert code == 0 and lines[-1]["n_inferences"] == 25 and lines[-1]["status"] == "stopped_early"
    assert log.count("event=block ") == 3  # 10 + 10 + partial 5


def test_cpu_fallback_is_logged(monkeypatch, tmp_path, index_video):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model_lib, "resolve_weights", lambda name: (tmp_path / "stub.pt", "0" * 64))
    monkeypatch.setattr(model_lib, "Detector", StubDetector)
    assert cli.main([str(index_video), "--output-dir", str(tmp_path / "out"), "--max-frames", "5"]) == 0
    assert "level=WARNING event=device_fallback device=cpu" in (tmp_path / "out" / "run.log").read_text()


@pytest.mark.parametrize("case", ["missing", "not_a_video", "unknown_model"])
def test_invalid_input_exits_2(tmp_path, index_video, case):
    video = {"missing": tmp_path / "nope.mp4", "not_a_video": tmp_path / "notes.txt"}.get(case, index_video)
    (tmp_path / "notes.txt").write_text("not a video")
    model = "nope.pt" if case == "unknown_model" else "yolo26s.pt"
    assert cli.main([str(video), "--output-dir", str(tmp_path / "out"), "--model", model]) == 2


@pytest.mark.parametrize("arg", [["--fps", "0"], ["--fps", "-1"], ["--max-frames", "0"]])
def test_invalid_arguments_exit_2(tmp_path, index_video, arg):
    with pytest.raises(SystemExit) as exc:
        cli.main([str(index_video), "--output-dir", str(tmp_path / "out"), *arg])
    assert exc.value.code == 2


def test_unexpected_error_exits_1(monkeypatch, tmp_path, index_video):
    code, _, log = run_stubbed(monkeypatch, tmp_path, index_video, detector=FailingDetector)
    assert code == 1 and "event=unexpected_error error_type=RuntimeError" in log


def test_viz_rebuilds_from_outputs_and_checks_them(monkeypatch, tmp_path, index_video):
    run_stubbed(monkeypatch, tmp_path, index_video)
    out = tmp_path / "out"
    viz.main([str(out), "--at", "1", "5"])
    assert (out / "viz" / "timeline.png").is_file() and len(list((out / "viz").glob("frame_*.jpg"))) == 2
    lines = (out / "output.jsonl").read_text().splitlines()
    tampered = json.loads(lines[11])
    tampered["source_frame_index"] += 1  # an output that no longer matches the video
    lines[11] = json.dumps(tampered)
    (out / "output.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="re-decode disagrees"):
        viz.main([str(out), "--at", "1"])
