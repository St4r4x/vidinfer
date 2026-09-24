"""The time arithmetic of 25 -> 10 fps sampling, on synthetic presentation timestamps."""

from fractions import Fraction

from vidinfer.video import sample_stream


def sample(timestamps, fps=10):
    return [s for s, _ in sample_stream(((t, None) for t in timestamps), fps)]


def test_cut_mp4_timeline_gives_3000_samples_with_ties_to_earlier_frame():
    samples = sample(Fraction(512 * i, 12800) for i in range(7500))  # cut.mp4: time_base 1/12800, 25 fps CFR
    idx = [s.source_index for s in samples]
    assert len(samples) == 3000
    assert idx[:8] == [0, 2, 5, 7, 10, 12, 15, 17]  # odd targets are exact ties -> earlier frame
    assert idx[-1] == 7497
    assert max(abs(s.frame_s - s.target_s) for s in samples) == Fraction(1, 50)  # 20 ms = half a frame period


def test_29_97_fps_uses_nearest_not_floor():
    # Here "nearest" and "floor" (ffmpeg fps round=up, torchcodec) disagree on 1 sample out of 2.
    idx = [s.source_index for s in sample(Fraction(1001 * i, 30000) for i in range(300))]
    assert idx[:5] == [0, 3, 6, 9, 12]


def test_stream_not_starting_at_zero_is_anchored_on_first_frame():
    samples = sample(Fraction(7, 5) + Fraction(i, 25) for i in range(7500))  # starts at 1.4 s (MPEG-TS style)
    idx = [s.source_index for s in samples]
    assert len(samples) == 3000
    assert len(set(idx)) == 3000  # no duplicated first frame


def test_gap_in_timestamps_still_picks_nearest_frame():
    kept = [Fraction(i, 25) for i in range(25) if not 10 <= i < 20]  # frames 0.40-0.76 s missing
    samples = sample(kept)
    assert len(samples) == 10
    assert samples[5].frame_s == Fraction(9, 25)  # t=0.5: 0.36 is closer than 0.80
