# vidinfer

🇫🇷 [Version française](README.fr.md)

Decode a video, run inference at **10 inferences per second of video** on **RGB 720×1280** frames, and log
**performance every 10 inferences**, with results whose reproducibility is checked, not assumed.

Model: person + ball detection (official Ultralytics YOLO26s weights, COCO classes `person` and `sports ball`).

## Reading the statement

- "Each frame" and "10 inferences per second of video" hold together only for the **sampled** frames: 300 s of
  video → **3 000 inferences**, one output line each.
- "Log every 10 frames" → one line per **10 inferences = 1 s of video**, with the time spent in each stage.
- "Ensure RGB" → RGB **by construction** (one conversion call) **and checked** (shape/dtype contract on every frame).
- "720 (h) × 1280 (w)" → same 16:9 ratio as the 1920×1080 input: resize without distortion.

## Results on `cut.mp4` (reference run, `results/reference/`, commit `3600844`)

| | |
|---|---|
| Input | H.264 1920×1080, 25 fps CFR, 7 500 frames, 300 s, untagged colour space |
| Status | `complete`: **3 000 / 3 000** inferences, 7 500 / 7 500 frames decoded, 0 decode error, max sampling error 20 ms |
| Throughput | **64.6 inferences/s = ×6.5 real time** for the processing loop (46.5 s for 300 s); ×5.9 for the whole program, model load and warm-up included |
| Service time per frame | p50 15.0 ms · p95 17.1 ms · p99 18.5 ms (conversion + inference + write; decoding runs in FFmpeg threads) |
| Where time goes | inference 69 % · preprocess 25 % · fingerprint 4 % · decode wait 2 % · write 0.6 % |
| GPU | RTX 4050 Laptop; the network runs 52 % of the wall time; peak 144 MiB allocated by torch |
| Reproducibility | reference, re-run and Docker run at the same commit: identical `results_sha256` and `frames_crc32` |

`results/reference/viz/`: a timeline rebuilt from `run.log`, and 4 annotated frames: two successes (t=20 s, t=240 s)
and two failures (t=8.3 s: advertising boards detected as balls; t=36 s: blurred close-up, nothing detected).

![timeline](results/reference/viz/timeline.png)

## Prerequisites

- **Linux x86_64 + NVIDIA GPU** with a driver supporting CUDA 12.6 (tested with 580). The lockfile targets this platform only:
  on macOS or Windows, read `results/reference/`, or use Docker on a Linux GPU host.
- Without a GPU: `--device cpu` works at ≈ 2 inferences/s (≈ 25 min for the full video; use `--max-frames 100`).
- The video is not in the repository: put it at `data/cut.mp4` (sha256 `4072bd68753d20120615054f9fabfd0d09e67e765e93b9fcd717ab5748405eb7`).
- Disk: ≈ 7 GB for the environment (CUDA wheels) or for the image. The first local run downloads the pinned weights
  (20 MB); the Docker image already contains them.

## Quick start

```bash
uv sync --locked                                              # exact environment from uv.lock
uv run python -m vidinfer data/cut.mp4 --output-dir outputs   # -> outputs/output.jsonl, outputs/run.log
uv run python -m vidinfer.viz outputs                         # -> outputs/viz/*.png|jpg
uv run pytest                                                 # 29 tests, CPU only, ~8 s, synthetic videos
```

Docker (GPU via the NVIDIA Container Toolkit; YOLO26s weights baked in and verified at build time):

```bash
docker build --build-arg GIT_COMMIT=$(git describe --always --dirty) -t vidinfer:0.2.0 .
mkdir -p outputs
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/data:ro" -v "$PWD/outputs:/work/outputs" vidinfer:0.2.0 /data/cut.mp4
```

Options: `--fps 10`, `--model yolo26s.pt|yolo26m.pt|path.pt` (only `s` is baked in the image; mount a weights
directory on `VIDINFER_WEIGHTS_DIR` for `m`), `--device auto|cuda|cpu`, `--max-frames N` (smoke test).

| Exit code | Meaning | Orchestrator action |
|---|---|---|
| 0 | `complete`, or `stopped_early` with `--max-frames` | done |
| 1 | unexpected error (e.g. CUDA failure) | retry |
| 2 | invalid input or configuration (missing/unreadable video, bad argument, unknown model, weights sha256 mismatch) | do not retry, fix the input |
| 3 | `partial`: decode errors or broken timestamps, the video was not fully processed | do not mark as done, review |

## Requirement → implementation → proof

| Requirement | Implementation | Proof |
|---|---|---|
| Python program taking `cut.mp4` | `python -m vidinfer VIDEO` | `tests/test_cli.py`, end to end on synthetic videos |
| Resize to 720×1280 | one swscale call: YUV→RGB + `AREA` resize; a WARNING if the input is not 16:9 | contract checked on every frame |
| Ensure RGB | `rgb24` by construction; untagged HD → BT.709 (libraries default to BT.601), logged as WARNING; BGR only at the Ultralytics call | tests: pure-red video, matrix rule, BT.709 applied on an untagged 1280×720 video, BGR at the model boundary |
| 10 inferences per second of video | nearest frame to `t0 + k/10`, tie → earlier frame, exact fractions | `3000/3000`, max error 20 ms in the summary; `test_sampler.py`; a video whose frame index is written in its pixels |
| Log time every 10 frames | one `event=block` line per 10 inferences, per-stage breakdown | 300 block lines in `run.log`, parsed back by the tests and by `viz` |
| Total time + metadata | `event=timing` (program, setup, model load, warm-up, processing, CPU) and `event=video` (codec, size, fps, duration, frames, colour tags, aspect ratio, sha256) | `run.log`, `output.jsonl` header |
| `output.<format>` per frame | `output.jsonl`: header / one line per inferred frame (even without detection) / summary with `status` | `results/reference/output.jsonl` |
| Images or a visualisation | `python -m vidinfer.viz`: timeline from `run.log`, annotated frames re-decoded and checked against `output.jsonl` | `results/reference/viz/` |

## Method and decisions

Every decision followed the same loop: **read the statement as a spec → look at the data → list what can be silently
wrong → measure the options on the real video → decide with a stated rule → cover it with a test → stop** (what is
not required, not tested or not cheap goes to the roadmap, with the condition that would bring it back).

| Decision | Options studied | Choice | Why |
|---|---|---|---|
| 25 → 10 fps | 1 frame out of N, float timestamps, ffmpeg `fps` filter, floor, nearest | nearest frame, tie → earlier, `Fraction`, grid anchored on the first PTS | 25/10 = 2.5: every odd target is an exact tie. Floats resolve ties by rounding noise; ffmpeg's default `fps=10` picks frames 1, 3, 6, 8… (20 to 40 ms late). Nearest bounds the error to half a frame. |
| Decoder | OpenCV, ffmpeg pipe, PyAV, torchcodec | PyAV, `thread_type="AUTO"` | exact PTS and colour matrix under control; every H.264 frame must be decoded anyway (P-frames) |
| Colour matrix | library default / explicit | BT.709 for untagged HD, logged | an assumption (the usual HD convention), made visible: a wrong matrix moves a pure red from G=0 to G=23 (test suite) |
| Model boundary | pass RGB as-is / convert | `cv2.cvtColor` to BGR at the Ultralytics call only | Ultralytics treats numpy input as BGR (`engine/predictor.py`); RGB as-is is a silent error |
| Model | YOLO26 n/s/m, imgsz 640/1280, FP16/32 | YOLO26s, imgsz 1280, FP16, conf 0.25 | the ball is 8–16 px at 720p, hence imgsz 1280. s vs m is a **cost** trade-off: on a hand-labelled sample, m found every ball s found plus 3 of 27, for 2.4× the GPU time; the sample is too small to conclude (study measurements, scripts not in this repo). m is one flag away. |
| Loop | producer/consumer threads, batching | synchronous loop | ×6.5 real time already; decoding runs in FFmpeg threads (0.26 ms of waiting per frame) |
| Output | JSON, CSV, Parquet, COCO, MOT | JSON Lines | streamable, readable after a crash, keeps frames without detection; the summary `status` marks completion |

## Reproducibility

Every run ends with two fingerprints:

- `results_sha256`: hash of `(source frame index, detections)` for every sample, timings excluded.
- `frames_crc32`: CRC of every pixel sent to the model (0.55 ms/frame, timed as its own stage). When results differ,
  it tells whether the drift comes from decoding/preprocessing or from the model.

`results/reproducibility.log` holds the start, environment, summary, timing and fingerprint lines of three runs at
commit `3600844`: reference and re-run in the host venv (Python 3.12.13, glibc 2.43), and the Docker image
(Python 3.12.14, glibc 2.36). All three give `results_sha256 = 824865c8…a40a9c` and `frames_crc32 = 0fb35010`,
the same values as version 0.1.0: the fixes of 0.2.0 changed no detection.

**Scope**: bit-identical with the same GPU model, driver and wheels. On CPU, `results_sha256` differs (FP32 instead of
FP16, other kernels: boxes within 0.7 px) while `frames_crc32` stays identical; the same is expected on another GPU
model. The pixels are reproducible everywhere, the detections within a hardware class.

| Layer | Pinned by | Recorded in |
|---|---|---|
| Code | git commit (`-dirty` if `src/` or the dependencies differ); image label `org.opencontainers.image.revision` | `event=start`, output header |
| Python dependencies | `uv.lock` (versions + hashes; torch from the cu126 index) | `event=environment` (versions, CUDA, cuDNN, GPU, driver) |
| System image | base image by digest, uv by version (apt packages are not pinned: build once, deploy by digest) | Dockerfile |
| Model weights | exact release URL + sha256, checked **before** loading (a `.pt` is a pickle); Docker `ADD --checksum` | `event=model`, output header |
| Input | sha256 of the video | `event=video`, output header |
| Runtime side effects | `YOLO_OFFLINE=1` (no telemetry), `YOLO_AUTOINSTALL=0` (no pip install mid-run) | — |
| Clean-machine rebuild | CI: `uv sync --locked` + lint + tests on every push | `.github/workflows/ci.yml` |

Ultralytics downloads weights from the *latest* GitHub release by default: this is why the URL is pinned.

## Performance observability

Every log line is **logfmt** with a UTC timestamp, readable by a human and by any logfmt parser (Loki, Vector…):

```text
ts=2026-09-25T07:14:20.940Z level=INFO event=block block=150 samples=1490-1499 video_t_s=149.90 wall_ms=154.67 ms_per_frame=15.47 inf_per_s=64.65 speed=6.47 decode_ms=2.52 preprocess_ms=38.45 fingerprint_ms=5.29 inference_ms=107.17 write_ms=0.97 other_ms=0.26 yolo_preprocess_ms=10.38 yolo_inference_ms=80.82 yolo_postprocess_ms=8.24 cpu_cores=1.81 gpu_mem_mb=50.25 elapsed_s=23.00
```

- `speed` = seconds of video per second of wall time (the real-time SLO); `speed < 1` logs a WARNING.
- Stages add up to `wall_ms`; `other_ms` is what is left unaccounted, so nothing is hidden.
- GPU stages are timed after `torch.cuda.synchronize()`; `yolo_*_ms` are Ultralytics' own timings, kept next to ours.
- `cpu_cores` = CPU seconds per wall second (main thread + FFmpeg decoder threads); `gpu_mem_mb` = memory allocated by
  torch (`nvidia-smi` shows ≈ 320 MiB for the process, CUDA context included).

What the logs showed, and what changed because of them:

1. **Measuring at my boundary exposed my own cost.** The model call measured 12.4 ms per frame against 9.1 ms reported
   by Ultralytics. The 3.3 ms gap was my RGB→BGR conversion (`np.ascontiguousarray(rgb[..., ::-1])`, ≈ 5 ms in
   isolation): `cv2.cvtColor` produces the same pixels in 0.07 ms. Gap now 0.8 ms, throughput ×5.8 → ×6.5, same
   fingerprints.
2. **Decoding is almost free on the main thread** (0.26 ms/frame of waiting): FFmpeg threads decode concurrently.
3. **The GPU runs the network about half of the wall time.** Next target if needed: preprocessing (3.9 ms/frame,
   single-threaded swscale), or feeding the GPU from a decoding thread. Not needed at ×6.5.

```python
import shlex
blocks = [dict(t.split("=", 1) for t in shlex.split(line.split(" event=block ", 1)[1]))
          for line in open("outputs/run.log") if " event=block " in line]
```

## Output format (`output.jsonl`, schema 1.1.0)

```json
{"type":"header","schema_version":"1.1.0","program":{"git_commit":"3600844…"},"environment":{…},"video":{"sha256":"4072bd68…",…},"processing":{"target_fps":10,"yuv_matrix":"ITU709",…},"model":{"weights":"yolo26s.pt","sha256":"646f8bc3…",…},"coordinates":"bbox_xyxy in pixels of the 1280x720 frame"}
{"type":"frame","sample_index":2400,"source_frame_index":6000,"pts":3072000,"timestamp_s":240.0,"inference_ms":…,"detections":[{"bbox_xyxy":[628.0,537.0,641.0,550.0],"score":0.8081,"class_id":32,"class_name":"sports ball"},…]}
{"type":"summary","status":"complete","n_inferences":3000,"expected":3000,"decode_errors":0,"max_sampling_error_ms":20.0,"duplicate_samples":0,"speed_x":6.46,…,"results_sha256":"824865c8…","frames_crc32":"0fb35010"}
```

Completion is `summary.status == "complete"`, not the presence of the line: on a corrupted file the sampler fills
the gaps with the nearest decoded frame, so the count can still match while frames were lost (see
`test_corrupt_video_is_partial_not_success`). Readers ignore unknown keys.

## Limitations

- **License**: Ultralytics is AGPL-3.0. Production use needs an Enterprise license or an Apache-2.0 detector
  (RF-DETR was considered during the study). Only `model.py` would change, plus the class ids used by `viz`.
- **No quality measurement**: the statement asks for none, and there is no labelled data. What the output shows:
  13.9 persons per frame on average; 77 % of frames have at least one `sports ball` detection, but 29 % have two or
  more, i.e. false balls on boards and spare balls (`frame_t008.3s.jpg`). 13 frames have no detection: 9 in the
  opening graphics, 4 in blurred close-ups (`frame_t036.0s.jpg`).
- **BT.709 is an assumption** for this untagged HD stream; the matrix is logged and a tagged stream is respected.
- **Input geometry**: a non-16:9 input is stretched (WARNING); interlacing and rotation metadata are not handled.
- Timings come from one laptop GPU and vary by a few percent between runs (×6.46, ×6.38, ×6.25); results do not.

## Deliberately not done (and what would bring it back)

| Not done | Why not now | Trigger |
|---|---|---|
| Decoding thread, NVDEC, TensorRT, batching | ×6.5 real time; GPU busy half the time is acceptable for one video | several streams per GPU, or cost at scale |
| Tracking, team assignment, pitch homography | not requested, no ground truth to evaluate them | annotated data and a product need |
| Quality evaluation of the detector | no labels; the statement does not ask for it | before any model change: a small labelled set per shot type |
| MLflow | no training; lineage is already in the output header and summary | several runs to compare, or an existing tracking server |
| Docker build and GPU run in CI | hosted runners have no GPU; CI rebuilds `uv.lock` and runs the CPU tests | self-hosted GPU runner |

## Production roadmap (not implemented)

S3 input → one Step Functions execution per match → GPU job on EKS Spot (this image, pinned by digest; model version
resolved once) → outputs to S3 under a deterministic key (`match/model/commit`). The exit code drives the
orchestrator (table above); a match is done only when `summary.status == "complete"`. Block metrics feed Grafana
(throughput, p95, `speed`); detections per frame, confidence and ball ratio per league and broadcaster serve as
label-free drift signals.

## Layout

```text
src/vidinfer/video.py     metadata, decode (+ input anomaly counters), timestamp sampler, RGB conversion + contract
src/vidinfer/model.py     pinned weights (URL + sha256), detector with the BGR boundary
src/vidinfer/__main__.py  CLI, loop, stage timers, logfmt logging, output.jsonl, status and exit codes, fingerprints
src/vidinfer/viz.py       timeline parsed from run.log, annotated frames checked against output.jsonl
tests/                    sampler, pixels/RGB/matrix, model boundary and weights, end to end, exit codes
results/                  reference run + reproducibility.log (fingerprints of the 3 runs)
```
