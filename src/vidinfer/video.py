"""Video side: metadata, full decode, timestamp sampling to a target fps, RGB conversion + resize."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import av
import numpy as np

log = logging.getLogger("vidinfer")

# FFmpeg AVColorSpace codes -> swscale matrix names accepted by PyAV.
_TAGGED_MATRIX = {1: "ITU709", 5: "ITU601", 6: "ITU601", 7: "SMPTE240M", 9: "BT2020", 10: "BT2020"}
UNSPECIFIED = 2


class Sample(NamedTuple):
    sample_index: int  # k in the target-fps timeline: 0, 1, 2, ...
    source_index: int  # index of the chosen frame in decode order
    target_s: Fraction  # t0 + k / fps
    frame_s: Fraction  # presentation time of the chosen frame


def sample_stream[T](items: Iterable[tuple[Fraction, T]], fps: int | Fraction = 10) -> Iterator[tuple[Sample, T]]:
    """For each target time t0 + k/fps, yield the nearest frame; a tie goes to the earlier frame.

    `items` are (presentation time in seconds, payload) in increasing time order, straight from the decoder.
    - Timestamps, not "1 frame out of N": 25 -> 10 fps is a 2.5 ratio.
    - Fractions, not floats: at 25 -> 10 fps every odd target lies exactly between two frames, so the
      tie rule must decide, not float rounding noise.
    - t0 = first frame time: streams do not always start at 0 (e.g. MPEG-TS), anchoring at 0 would
      duplicate the first frame.
    Needs a one-frame lookahead only, so it streams. Targets after the last frame are not emitted.
    """
    period = 1 / Fraction(fps)
    k, t0, prev = 0, None, None
    for i, (t, payload) in enumerate(items):
        if t0 is None:
            t0 = t
        while t0 + k * period <= t:
            target = t0 + k * period
            chosen = prev if prev is not None and target - prev[1] <= t - target else (i, t, payload)
            yield Sample(k, chosen[0], target, chosen[1]), chosen[2]
            k += 1
        prev = (i, t, payload)


def decode(container: av.container.InputContainer, stats: dict[str, int]) -> Iterator[tuple[Fraction, av.VideoFrame]]:
    """Decode EVERY frame of the first video stream (H.264 P-frames depend on the previous ones).

    Corrupt packets are logged, counted and skipped instead of aborting the whole video.
    """
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"  # frame + slice threading: measured x2.4 faster than the default
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError as e:
            stats["decode_errors"] += 1
            log.warning(f"event=decode_error pts={packet.pts} error={str(e)!r}")
            continue
        for frame in frames:
            stats["decoded_frames"] += 1
            if frame.pts is not None:  # ponytail: a frame without pts cannot be placed in time; never seen here
                yield frame.pts * frame.time_base, frame


def yuv_matrix(color_space: int | None, height: int) -> str:
    """Matrix used for YUV -> RGB: the stream tag if any, else the usual convention (HD -> BT.709, SD -> BT.601).

    Libraries default to BT.601 for untagged streams, which is wrong for HD broadcast content.
    """
    if color_space in _TAGGED_MATRIX:
        return _TAGGED_MATRIX[color_space]
    return "ITU709" if height >= 720 else "ITU601"


def to_rgb(frame: av.VideoFrame, height: int, width: int, matrix: str) -> np.ndarray:
    """YUV -> RGB and resize in a single swscale call (AREA = anti-aliased downscale), then check the contract."""
    rgb = frame.to_ndarray(format="rgb24", width=width, height=height, interpolation="AREA", src_colorspace=matrix)
    check_rgb(rgb, height, width)
    return rgb


def check_rgb(rgb: np.ndarray, height: int, width: int) -> None:
    """The 'verify' half of 'ensure RGB'. Channel ORDER cannot be read from an array: it is guaranteed by
    construction (rgb24) and covered by a test on a pure-red video."""
    if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"frame contract violated: got {rgb.shape} {rgb.dtype}, expected ({height}, {width}, 3) uint8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def metadata(path: Path) -> dict:
    """Container + video stream metadata (PyAV, no ffprobe subprocess) and the file sha256 for lineage."""
    with av.open(str(path)) as c:
        if not c.streams.video:
            raise ValueError(f"no video stream in {path}")
        v = c.streams.video[0]
        cc = v.codec_context
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "format": c.format.name,
            "codec": cc.name,
            "profile": cc.profile,
            "width": cc.width,
            "height": cc.height,
            "pix_fmt": cc.pix_fmt,
            "avg_fps": str(v.average_rate),
            "time_base": str(v.time_base),
            "start_time_s": float(v.start_time * v.time_base) if v.start_time is not None else None,
            "duration_s": float(v.duration * v.time_base) if v.duration else c.duration / av.time_base,
            "declared_frames": v.frames,
            "bit_rate": cc.bit_rate,
            "color_space": "unspecified" if cc.colorspace == UNSPECIFIED else cc.colorspace,
            "color_range": cc.color_range,
            "audio_streams": len(c.streams.audio),
        }
