"""Visualize a finished run from its outputs only: annotated frames + a performance / detection timeline.

    python -m vidinfer.viz OUTPUT_DIR [--video PATH] [--at 20 90 125 170 240 285]

The timeline is rebuilt by parsing run.log: the performance log is the data source, not a side channel.
Annotated frames are re-decoded with the same sampler and checked against output.jsonl.
"""

from __future__ import annotations

import argparse
import json
import shlex
from collections import defaultdict
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from vidinfer.video import decode, sample_stream, to_rgb  # noqa: E402

STAGES = ["decode", "preprocess", "fingerprint", "inference", "write", "other"]
COLORS = {"person": (0, 200, 255), "sports ball": (255, 40, 200)}


def read_output(out_dir: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in (out_dir / "output.jsonl").read_text().splitlines()]
    return lines[0], [rec for rec in lines if rec["type"] == "frame"]


def read_blocks(log_path: Path) -> list[dict[str, str]]:
    """Parse the logfmt 'event=block' lines back into dicts."""
    return [
        dict(tok.split("=", 1) for tok in shlex.split(line.split(" event=block ", 1)[1]))
        for line in log_path.read_text().splitlines()
        if " event=block " in line
    ]


def plot_timeline(blocks: list[dict[str, str]], frames: list[dict], path: Path) -> None:
    fig, (perf, dets) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, height_ratios=[3, 2])
    x = [float(b["video_t_s"]) for b in blocks]
    bottom = [0.0] * len(blocks)
    for stage in STAGES:
        y = [float(b.get(f"{stage}_ms", 0)) for b in blocks]
        perf.bar(x, y, width=0.9, bottom=bottom, label=stage)
        bottom = [a + b for a, b in zip(bottom, y, strict=True)]
    perf.axhline(1000, color="red", ls="--", lw=1)
    perf.text(x[0], 1010, "real-time budget: 10 inferences = 1 s of video = 1000 ms", color="red", va="bottom")
    perf.set_ylim(0, 1150)
    perf.set_ylabel("wall time per block of 10 inferences (ms)")
    perf.set_title("Performance per 10-inference block (parsed from run.log)")
    perf.legend(ncol=6, loc="upper right", fontsize=8)

    t = [f["timestamp_s"] for f in frames]
    persons = [sum(d["class_id"] == 0 for d in f["detections"]) for f in frames]
    ball_t = [f["timestamp_s"] for f in frames if any(d["class_id"] == 32 for d in f["detections"])]
    dets.plot(t, persons, lw=0.6, label="persons per frame")
    dets.scatter(ball_t, [-1.5] * len(ball_t), marker="|", s=40, color="magenta", label="ball detected")
    dets.set_xlabel("video time (s)")
    dets.set_ylabel("detections")
    dets.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def annotate(header: dict, frames: list[dict], video: Path, times: list[float], out_dir: Path) -> list[Path]:
    proc = header["processing"]
    height, width = proc["frame_size_hw"]
    by_index = {f["sample_index"]: f for f in frames}
    wanted = {min(by_index, key=lambda k: abs(by_index[k]["timestamp_s"] - t)) for t in times}
    font = ImageFont.load_default(size=22)
    saved, stats = [], defaultdict(int)
    with av.open(str(video)) as container:
        for sample, frame in sample_stream(decode(container, stats), proc["target_fps"]):
            if sample.sample_index > max(wanted):
                break
            if sample.sample_index not in wanted:
                continue
            rec = by_index[sample.sample_index]
            if rec["source_frame_index"] != sample.source_index:
                raise RuntimeError(f"re-decode disagrees with output.jsonl at sample {sample.sample_index}")
            img = Image.fromarray(to_rgb(frame, height, width, proc["yuv_matrix"]))  # RGB end to end, no BGR trap
            draw = ImageDraw.Draw(img)
            for d in rec["detections"]:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                color = COLORS.get(d["class_name"], (255, 255, 255))
                if d["class_id"] == 32:  # the ball is ~10 px: circle it
                    draw.ellipse([x1 - 10, y1 - 10, x2 + 10, y2 + 10], outline=color, width=3)
                    draw.text((x2 + 12, y1 - 12), f"ball {d['score']:.2f}", fill=color, font=font)
                else:
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            n_persons = sum(d["class_id"] == 0 for d in rec["detections"])
            has_ball = any(d["class_id"] == 32 for d in rec["detections"])
            caption = (
                f"t={rec['timestamp_s']:.1f}s  sample {rec['sample_index']}  source frame "
                f"{rec['source_frame_index']}  |  persons {n_persons}  ball {'yes' if has_ball else 'no'}"
            )
            draw.rectangle([0, height - 36, width, height], fill=(0, 0, 0))
            draw.text((10, height - 31), caption, fill=(255, 255, 255), font=font)
            path = out_dir / f"frame_t{rec['timestamp_s']:05.1f}s.jpg"
            img.save(path, quality=90)
            saved.append(path)
    return saved


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vidinfer.viz", description=__doc__)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--video", type=Path, help="input video (default: the path recorded in output.jsonl)")
    p.add_argument("--at", type=float, nargs="+", default=[20, 90, 125, 170, 240, 285], help="video times (s)")
    args = p.parse_args(argv)
    header, frames = read_output(args.output_dir)
    viz_dir = args.output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    plot_timeline(read_blocks(args.output_dir / "run.log"), frames, viz_dir / "timeline.png")
    video = args.video or Path(header["video"]["path"])
    for path in [viz_dir / "timeline.png", *annotate(header, frames, video, args.at, viz_dir)]:
        print(path)


if __name__ == "__main__":
    main()
