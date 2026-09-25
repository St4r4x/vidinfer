"""The pixels sent to the model: right frames, right size, right channel order."""

from collections import defaultdict

import av
import numpy as np
import pytest
from conftest import decode_index

from vidinfer.video import decode, sample_stream, to_rgb, yuv_matrix


def rgb_samples(path):
    """ITU601: the matrix the synthetic encoder used (decoding it as ITU709 turns pure red into G=23)."""
    stats = defaultdict(int)
    with av.open(str(path)) as c:
        return [(s, to_rgb(f, 720, 1280, "ITU601")) for s, f in sample_stream(decode(c, stats), 10)]


def test_sampled_pixels_come_from_the_expected_source_frames(index_video):
    got = rgb_samples(index_video)
    expected = [(5 * k) // 2 for k in range(100)]  # nearest frame to k/10 s at 25 fps, ties -> earlier
    assert [s.source_index for s, _ in got] == expected  # what the sampler says
    assert [decode_index(rgb) for _, rgb in got] == expected  # what is actually in the pixels (threaded decoder)


def test_frames_are_720x1280_uint8_rgb(red_video):
    _, rgb = rgb_samples(red_video)[0]
    assert rgb.shape == (720, 1280, 3) and rgb.dtype == np.uint8
    r, g, b = rgb.reshape(-1, 3).mean(axis=0)
    assert r > 240 and g < 15 and b < 15  # channel 0 is red: not BGR


@pytest.mark.parametrize(
    ("color_space", "height", "matrix"),
    [(None, 1080, "ITU709"), (None, 720, "ITU709"), (None, 576, "ITU601"), (1, 480, "ITU709"), (5, 1080, "ITU601")],
)
def test_yuv_matrix_tag_first_then_hd_convention(color_space, height, matrix):
    assert yuv_matrix(color_space, height) == matrix


def test_the_matrix_choice_changes_the_pixels(red_video):
    """Red encoded with BT.601 (the encoder default) and decoded as BT.709: the matrix really is applied."""
    stats = defaultdict(int)
    with av.open(str(red_video)) as c:
        _, frame = next(decode(c, stats))
        g601 = to_rgb(frame, 720, 1280, "ITU601")[..., 1].mean()
        g709 = to_rgb(frame, 720, 1280, "ITU709")[..., 1].mean()
    assert g601 < 15 < g709  # G=23 instead of 0: the size of the error a wrong matrix makes
