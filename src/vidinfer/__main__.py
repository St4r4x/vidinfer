"""python -m vidinfer VIDEO [--output-dir DIR] [--fps 10] [--model yolo26s.pt] [--device auto] [--max-frames N]

Decodes VIDEO, samples it at --fps inferences per second of VIDEO time, converts each sample to RGB 720x1280,
detects persons + the ball, and writes:
  - OUTPUT_DIR/output.jsonl : header (lineage) / one line per inferred frame / summary (completeness marker)
  - OUTPUT_DIR/run.log      : logfmt lines (human-readable and machine-parsable): environment, video metadata,
                              one performance line per 10 inferences, totals, reproducibility fingerprints
Exit codes: 0 success (summary status "complete" or "stopped_early"), 1 unexpected error (retry may help),
2 invalid input or configuration (do not retry), 3 partial output: the video was not fully processed (review).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

from vidinfer import __version__
from vidinfer import model as model_lib
from vidinfer.video import decode, metadata, sample_stream, to_rgb, yuv_matrix

HEIGHT, WIDTH = 720, 1280  # required by the exercise
BLOCK = 10  # log performance every 10 inferences
SCHEMA_VERSION = "1.1.0"  # 1.1: summary.status + input anomaly counters
log = logging.getLogger("vidinfer")


class InputError(Exception):
    """Invalid input or configuration (exit code 2): retrying will not help, the input must be fixed."""


def _logfmt_value(value: object) -> str:
    text = f"{value:.2f}" if isinstance(value, float) else str(value)
    return json.dumps(text, ensure_ascii=False) if not text or any(c in text for c in ' "=\\\t\n') else text


def kv(event: str, **fields: object) -> str:
    """One logfmt line (key=value, double quotes when needed): readable, and parsable by any logfmt parser."""
    return " ".join([f"event={event}", *(f"{k}={_logfmt_value(v)}" for k, v in fields.items())])


def setup_logging(path: Path) -> None:
    """ts=<UTC ISO 8601> level=<LEVEL> event=... : the whole line is logfmt."""
    formatter = logging.Formatter("ts=%(asctime)s.%(msecs)03dZ level=%(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    handlers = [logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stderr)]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)  # force: tests call main() repeatedly


def git_commit() -> str:
    """Commit of the running code, '-dirty' if src/ or the dependencies differ; Docker passes it as an env var."""
    if commit := os.environ.get("VIDINFER_GIT_COMMIT"):
        return commit
    cwd = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True)
        dirty = subprocess.run(
            # Only what makes the code: outputs written into the repo (results/) must not flag it dirty.
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ":/src", ":/pyproject.toml", ":/uv.lock"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def environment() -> dict:
    """Everything that can change the numbers: library versions, CUDA/cuDNN, GPU and driver."""
    import ultralytics

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics": ultralytics.__version__,
        "av": av.__version__,
        "libavcodec": ".".join(map(str, av.library_versions["libavcodec"])),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        try:
            import pynvml  # shipped by nvidia-ml-py, an ultralytics dependency

            pynvml.nvmlInit()
            env["gpu_driver"] = pynvml.nvmlSystemGetDriverVersion()
        except Exception:  # noqa: BLE001  the driver version is informative only
            env["gpu_driver"] = "unknown"
    return env


def pick_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if requested == "cuda":
        raise InputError("--device cuda requested but CUDA is not available")
    log.warning(kv("device_fallback", device="cpu", reason="cuda unavailable, expect ~2 inferences/s"))
    return "cpu"


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vidinfer", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("video", type=Path, help="input video, e.g. cut.mp4")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--fps", type=positive_int, default=10, help="inferences per second of VIDEO time (default 10)")
    p.add_argument("--model", default="yolo26s.pt", help=f"pinned weights ({', '.join(model_lib.WEIGHTS)}) or a .pt")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--max-frames", type=positive_int, help="stop after N inferences (smoke test)")
    args = p.parse_args(argv)
    args.argv = sys.argv[1:] if argv is None else argv
    return args


def percentile(sorted_ms: list[float], q: float) -> float:
    return sorted_ms[min(len(sorted_ms) - 1, round(q * (len(sorted_ms) - 1)))] if sorted_ms else 0.0


def run(args: argparse.Namespace) -> int:
    t_program, cpu_program = time.perf_counter(), time.process_time()
    commit = git_commit()
    log.info(kv("start", version=__version__, git_commit=commit, argv=" ".join(args.argv)))
    env = environment()
    log.info(kv("environment", **env))

    # ---- input validation + metadata (lineage: the sha256 identifies the exact input)
    if not args.video.is_file():
        raise InputError(f"input not found: {args.video}")
    try:
        meta = metadata(args.video)
    except (av.error.FFmpegError, ValueError) as e:
        raise InputError(f"cannot read a video stream from {args.video}: {e}") from e
    log.info(kv("video", **meta))
    if meta["duration_s"] is None:
        raise InputError(f"cannot determine the duration of {args.video} (raw stream without container?)")
    tagged = meta["color_space"] if meta["color_space"] != "unspecified" else None
    matrix = yuv_matrix(tagged, meta["height"])
    if tagged is None:
        log.warning(kv("color_untagged", yuv_matrix=matrix, note="libraries would default to ITU601"))
    if Fraction(meta["display_aspect_ratio"]) != Fraction(WIDTH, HEIGHT):
        log.warning(kv("aspect_ratio", display=meta["display_aspect_ratio"], note=f"stretched to {WIDTH}x{HEIGHT}"))
    # Same count as the sampler: one target per 1/fps from the first frame, up to the end of the last frame.
    expected = math.ceil(Fraction(meta["duration_s"]).limit_denominator(10**6) * args.fps)
    log.info(
        kv(
            "sampling",
            target_fps=args.fps,
            expected_inferences=expected,
            rule="nearest frame to t0+k/fps, tie->earlier",
        )
    )

    # ---- model: load and warm-up are measured apart from the steady state
    device = pick_device(args.device)
    t = time.perf_counter()
    try:
        weights, weights_sha = model_lib.resolve_weights(args.model)
    except (FileNotFoundError, ValueError) as e:  # unknown model, sha256 mismatch: configuration errors
        raise InputError(str(e)) from e
    detector = model_lib.Detector(weights, device)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    with av.open(str(args.video)) as container:  # warm up on a real frame (a black one leaves NMS cold)
        first = next(decode(container, defaultdict(int)), None)
    detector.warmup(to_rgb(first[1], HEIGHT, WIDTH, matrix) if first else np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    warmup_s = time.perf_counter() - t
    params = {k: v for k, v in detector.kwargs.items() if k not in ("device", "verbose")}
    log.info(
        kv(
            "model",
            weights=weights.name,
            sha256=weights_sha,
            device=device,
            load_s=load_s,
            warmup_s=warmup_s,
            **{k: str(v).replace(" ", "") for k, v in params.items()},
        )
    )
    cuda = device != "cpu"
    sync = torch.cuda.synchronize if cuda else (lambda: None)  # CUDA is async: time the work, not the launch
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    header = {
        "type": "header",
        "schema_version": SCHEMA_VERSION,
        "program": {"version": __version__, "git_commit": commit, "argv": args.argv},
        "environment": env,
        "video": meta,
        "processing": {
            "target_fps": args.fps,
            "sampling": "nearest frame to t0 + k/fps, tie -> earlier",
            "frame_size_hw": [HEIGHT, WIDTH],
            "color": "RGB",
            "yuv_matrix": matrix,
            "resize_interpolation": "area",
        },
        "model": {"weights": weights.name, "sha256": weights_sha, "device": device, **params},
        "coordinates": "bbox_xyxy in pixels of the 1280x720 frame",
    }

    # ---- main loop
    stats: dict[str, int] = defaultdict(int)
    total: dict[str, float] = defaultdict(float)  # seconds per stage, whole run
    block: dict[str, float] = defaultdict(float)  # seconds per stage, current block
    yolo: dict[str, float] = defaultdict(float)  # ms inside Ultralytics (letterbox / network / NMS), current block
    latencies_ms: list[float] = []
    results_hash, frames_crc = hashlib.sha256(), 0
    n, stopped_early = 0, False
    max_sampling_error, duplicate_samples, last_source = Fraction(0), 0, None

    def stage(name: str, seconds: float) -> None:
        total[name] += seconds
        block[name] += seconds

    def log_block(count: int, video_t: float) -> None:
        nonlocal t_block, cpu_block
        now, cpu_now = time.perf_counter(), time.process_time()
        wall = now - t_block
        speed = (count / args.fps) / wall  # seconds of video processed per second of wall time
        stages_ms = {f"{k}_ms": v * 1e3 for k, v in block.items()}
        log.info(
            kv(
                "block",
                block=(n - 1) // BLOCK + 1,
                samples=f"{n - count}-{n - 1}",
                video_t_s=video_t,
                wall_ms=wall * 1e3,
                ms_per_frame=wall * 1e3 / count,
                inf_per_s=count / wall,
                speed=speed,
                **stages_ms,
                other_ms=wall * 1e3 - sum(stages_ms.values()),
                **{f"yolo_{k}_ms": v for k, v in yolo.items()},
                cpu_cores=(cpu_now - cpu_block) / wall,  # CPU seconds per wall second (decoder threads included)
                gpu_mem_mb=torch.cuda.memory_allocated() / 2**20 if cuda else 0.0,
                elapsed_s=now - t_proc,
            )
        )
        if speed < 1:
            log.warning(kv("slow_block", block=(n - 1) // BLOCK + 1, speed=speed, note="slower than real time"))
        block.clear()
        yolo.clear()
        t_block, cpu_block = now, cpu_now

    out_path = args.output_dir / "output.jsonl"
    t_proc = t_block = time.perf_counter()
    cpu_block = time.process_time()
    with out_path.open("w", encoding="utf-8") as out, av.open(str(args.video)) as container:
        out.write(json.dumps(header, separators=(",", ":")) + "\n")
        samples = sample_stream(decode(container, stats), args.fps)
        while True:
            t0 = time.perf_counter()
            item = next(samples, None)  # decodes ~2.5 source frames per sample at 25 -> 10 fps
            t1 = time.perf_counter()
            stage("decode", t1 - t0)
            if item is None:
                break
            sample, frame = item
            max_sampling_error = max(max_sampling_error, abs(sample.frame_s - sample.target_s))
            duplicate_samples += sample.source_index == last_source  # source fps < target fps, or gaps
            last_source = sample.source_index
            rgb = to_rgb(frame, HEIGHT, WIDTH, matrix)
            t2 = time.perf_counter()
            stage("preprocess", t2 - t1)
            frames_crc = zlib.crc32(rgb.data, frames_crc)  # fingerprint of every pixel sent to the model
            t3 = time.perf_counter()
            stage("fingerprint", t3 - t2)
            detections, speed_ms = detector(rgb)
            sync()
            t4 = time.perf_counter()
            stage("inference", t4 - t3)
            for k in ("preprocess", "inference", "postprocess"):
                yolo[k] += speed_ms.get(k, 0.0)
            record = {
                "type": "frame",
                "sample_index": sample.sample_index,
                "source_frame_index": sample.source_index,
                "pts": frame.pts,
                "timestamp_s": round(float(sample.frame_s), 6),
                "inference_ms": round((t4 - t3) * 1e3, 3),
                "detections": detections,
            }
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            # Timings excluded: two runs with the same results give the same digest.
            results_hash.update(json.dumps([sample.source_index, detections], separators=(",", ":")).encode())
            t5 = time.perf_counter()
            stage("write", t5 - t4)
            latencies_ms.append((t5 - t1) * 1e3)
            n += 1
            if n % BLOCK == 0:
                out.flush()  # a crash loses at most one block
                log_block(BLOCK, float(sample.target_s))
            if args.max_frames and n >= args.max_frames:
                stopped_early = True
                break
        if n % BLOCK:
            log_block(n % BLOCK, float(sample.target_s))
        processing_s = time.perf_counter() - t_proc
        lat = sorted(latencies_ms)
        anomalies = {k: stats[k] for k in ("decode_errors", "pts_missing", "pts_non_monotonic")}
        if stopped_early:
            status = "stopped_early"
        elif n > 0 and n == expected and not any(anomalies.values()):
            status = "complete"
        else:
            status = "partial"  # something in the input was not processed: never mark the video as done
        summary = {
            "type": "summary",
            "status": status,
            "n_inferences": n,
            "expected": expected,
            "stopped_early": stopped_early,
            "decoded_frames": stats["decoded_frames"],
            "declared_frames": meta["declared_frames"],
            **anomalies,
            "max_sampling_error_ms": round(float(max_sampling_error) * 1e3, 3),
            "duplicate_samples": duplicate_samples,
            "processing_wall_s": round(processing_s, 3),
            "inferences_per_s": round(n / processing_s, 2),
            "speed_x": round(n / args.fps / processing_s, 2),
            "latency_ms": {
                "p50": round(percentile(lat, 0.50), 2),
                "p95": round(percentile(lat, 0.95), 2),
                "p99": round(percentile(lat, 0.99), 2),
                "max": round(lat[-1] if lat else 0.0, 2),
            },
            "stage_s": {k: round(v, 3) for k, v in total.items()},
            "model_load_s": round(load_s, 3),
            "warmup_s": round(warmup_s, 3),
            "gpu_peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else 0.0,
            "results_sha256": results_hash.hexdigest(),
            "frames_crc32": f"{frames_crc:08x}",
        }
        out.write(json.dumps(summary, separators=(",", ":")) + "\n")  # completion is summary.status, not presence

    # ---- totals
    log.log(
        logging.INFO if status != "partial" else logging.ERROR,
        kv(
            "summary",
            status=status,
            n_inferences=n,
            expected=expected,
            decoded_frames=stats["decoded_frames"],
            declared_frames=meta["declared_frames"],
            **anomalies,
            max_sampling_error_ms=summary["max_sampling_error_ms"],
            duplicate_samples=duplicate_samples,
        ),
    )
    source_fps = Fraction(meta["avg_fps"])
    if source_fps and max_sampling_error > 1 / (2 * source_fps):
        log.warning(kv("sampling_error", max_ms=summary["max_sampling_error_ms"], note="above half a source frame"))
    log.info(
        kv(
            "timing",
            program_wall_s=time.perf_counter() - t_program,
            setup_s=t_proc - t_program,
            model_load_s=load_s,
            warmup_s=warmup_s,
            processing_wall_s=processing_s,
            cpu_s=time.process_time() - cpu_program,
            inf_per_s=n / processing_s,
            speed=n / args.fps / processing_s,
        )
    )
    log.info(
        kv(
            "stages",
            **{f"{k}_s": v for k, v in total.items()},
            **{f"{k}_pct": 100 * v / processing_s for k, v in total.items()},
        )
    )
    log.info(
        kv(
            "latency",
            p50_ms=summary["latency_ms"]["p50"],
            p95_ms=summary["latency_ms"]["p95"],
            p99_ms=summary["latency_ms"]["p99"],
            max_ms=summary["latency_ms"]["max"],
            mean_ms=statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            gpu_peak_mem_mb=summary["gpu_peak_mem_mb"],
        )
    )
    log.info(kv("fingerprint", results_sha256=summary["results_sha256"], frames_crc32=summary["frames_crc32"]))
    log.info(kv("outputs", output=out_path, log=args.output_dir / "run.log"))
    return 3 if status == "partial" else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)  # invalid arguments: argparse exits with code 2
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(args.output_dir / "run.log")
    except OSError as e:
        print(kv("output_error", error=e), file=sys.stderr)
        return 2
    try:
        return run(args)
    except InputError as e:
        log.error(kv("input_error", error=e))
        return 2
    except Exception as e:
        log.exception(kv("unexpected_error", error_type=type(e).__name__, error=e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
