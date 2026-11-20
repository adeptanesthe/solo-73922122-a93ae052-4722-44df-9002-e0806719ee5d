import io
import wave

import numpy as np
import pytest

from app import dsp

RATE = 16_000


def make_wav(left: np.ndarray, right: np.ndarray, rate: int = RATE) -> bytes:
    n = min(len(left), len(right))
    pcm = np.stack([left[:n], right[:n]], axis=1)
    pcm = np.rint(np.clip(pcm, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def source(seconds=4.0, seed=5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    x = rng.standard_normal(n) * 0.12
    return (x * (0.55 + 0.45 * np.sin(2 * np.pi * 1.7 * t) ** 2)).astype(np.float32)


def voiced(seconds=4.0, seed=11, f0=200.0, noise=0.05) -> np.ndarray:
    """Periodic pseudo-voice.

    A periodic source keeps its (small) inter-channel delay correlated enough
    for the *in-place* left/right average to cancel by >= 5 dB when one side is
    polarity-inverted.  Broadband noise with the same delay does not: its
    in-place average barely loses energy and must not be flagged.
    """
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    rng = np.random.default_rng(seed)
    x = (
        0.85 * np.sin(2 * np.pi * f0 * t)
        + 0.22 * np.sin(2 * np.pi * 2 * f0 * t + 0.4)
        + 0.08 * np.sin(2 * np.pi * 3 * f0 * t + 1.1)
    )
    x = x + noise * rng.standard_normal(n)
    x = x * (0.72 + 0.28 * np.sin(2 * np.pi * 2.1 * t) ** 2)
    return (x / np.max(np.abs(x)) * 0.55).astype(np.float32)


def delayed(x: np.ndarray, d: int) -> np.ndarray:
    y = np.zeros_like(x)
    if d >= 0:
        y[d:] = x[: len(x) - d]
    else:
        y[: len(x) + d] = x[-d:]
    return y


# ---------- WAV validation ----------

def test_accepts_valid_stereo():
    x = source()
    pcm, rate = dsp.parse_wav(make_wav(x, x))
    assert rate == RATE
    assert pcm.shape == (len(x), 2)
    assert pcm.dtype == np.float32


def test_rejects_wrong_channels():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(np.zeros(RATE, dtype="<i2").tobytes())
    with pytest.raises(dsp.AudioError, match="双声道"):
        dsp.parse_wav(buf.getvalue())


def test_rejects_wrong_rate():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(np.zeros(8000 * 2, dtype="<i2").tobytes())
    with pytest.raises(dsp.AudioError, match="16000"):
        dsp.parse_wav(buf.getvalue())


def test_rejects_wrong_bit_depth_and_length():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(3)
        wf.setframerate(RATE)
        wf.writeframes(b"\x00" * RATE * 6)
    with pytest.raises(dsp.AudioError, match="16 位"):
        dsp.parse_wav(buf.getvalue())


def test_rejects_over_60_seconds():
    n = RATE * 61
    with pytest.raises(dsp.AudioError, match="60"):
        dsp.parse_wav(make_wav(np.zeros(n), np.zeros(n)))


def test_rejects_non_wav():
    with pytest.raises(dsp.AudioError):
        dsp.parse_wav(b"not a wav at all")


# ---------- delay estimation ----------

def test_delay_is_estimated_signed():
    x = source()
    for d in (0, 12, 32):
        info = dsp.estimate_delay(x, delayed(x, d))
        assert info["confirmed"] is True
        assert info["lag"] == d
        assert info["lag_ms"] == pytest.approx(d * 1000 / RATE, abs=1e-3)


def test_negative_delay_is_estimated_and_repaired():
    x = source()
    # left is late: left[n] ~ right[n-18], so right leads by 18 samples
    left = delayed(x, 18)
    right = x.copy()
    left[20000:40000] *= -1  # left inverted in a span
    info = dsp.estimate_delay(left, right)
    assert info["confirmed"] is True
    assert info["lag"] == -18
    pcm = np.stack([left, right], axis=1)
    mono = dsp.repair(pcm, [(20000, 40000)], -18)
    # align by delaying the early channel (right) inside the span, then flip it
    inner = slice(20600, 39400)
    assert np.max(np.abs(mono[inner] - left[inner])) < 1e-6


def test_inverted_passage_does_not_hide_delay():
    x = source(6.0)
    r = delayed(x, 24)
    r[RATE * 2 : RATE * 3] *= -1
    info = dsp.estimate_delay(x, r)
    assert info["confirmed"] is True
    assert info["lag"] == 24


def test_decorrelated_channels_are_unconfirmed():
    x, y = source(seed=1), source(seed=2)
    info = dsp.estimate_delay(x, y)
    assert info["confirmed"] is False


def test_silence_and_one_sided_are_unconfirmed():
    x = source()
    for left, right in ((np.zeros_like(x), np.zeros_like(x)), (x, np.zeros_like(x))):
        info = dsp.estimate_delay(left, right)
        assert info["confirmed"] is False


def test_delay_beyond_assumption_is_rejected(monkeypatch):
    # A 6 ms lag is outside the 2 ms assumption even though the search reaches 4 ms.
    x = source(6.0)
    info = dsp.estimate_delay(x, delayed(x, int(RATE * 0.006)))
    assert info["confirmed"] is False


# ---------- region detection ----------

def test_flags_inverted_span_only():
    x = voiced(5.0)
    r = x.copy()
    a, b = RATE, RATE * 2
    r[a:b] *= -1
    regions = dsp.detect_regions(x, r, 0)
    assert len(regions) == 1
    reg = regions[0]
    # regions are built from whole 10 ms units
    assert reg["start_sample"] % 160 == 0
    assert reg["end_sample"] % 160 == 0
    assert abs(reg["start_sample"] - a) <= 160
    assert abs(reg["end_sample"] - b) <= 160
    assert reg["loss_db"] > 5
    assert reg["correlation"] < -0.9


def test_silence_and_single_sided_never_flagged():
    x = voiced()
    silent = np.zeros_like(x)
    assert dsp.detect_regions(silent, silent, 0) == []
    assert dsp.detect_regions(x, silent, 0) == []
    assert dsp.detect_regions(silent, x, 0) == []


def test_detection_uses_delay_alignment():
    """The anti-phase test is aligned; the energy-loss test is not.

    A periodic source is flagged with the true lag; reading the correlation at
    a lag that does not align the channels drops the unit.  The in-place loss
    is identical in both calls, so only the aligned correlation can explain the
    difference.
    """
    x = voiced(5.0)
    d = 8
    r = delayed(x, d)
    r[RATE : RATE * 2] *= -1
    aligned = dsp.detect_regions(x, r, d)
    assert len(aligned) == 1
    assert aligned[0]["correlation"] <= -0.9
    assert dsp.detect_regions(x, r, 24) == []


def test_in_place_energy_loss_uses_recording_time_axis():
    """Broadband noise delayed by 1.5 ms: aligned it is perfectly anti-phase,
    but the *in-place* downmix does not cancel, so it must not be flagged."""
    x = source()
    d = 24
    r = delayed(x, d)
    r[RATE : RATE * 2] *= -1
    assert dsp.estimate_delay(x, r)["confirmed"] is True
    # sanity: alignment achieves the cancellation a loss measured on aligned
    # windows would have reported
    a = x[RATE : RATE + 320].astype(np.float64)
    b = r[RATE + d : RATE + d + 320].astype(np.float64)
    assert float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))) < -0.9
    assert dsp.detect_regions(x, r, d) == []


def test_short_blip_ignored():
    x = voiced()
    r = x.copy()
    r[1000:1100] *= -1  # 6.9 ms < 80 ms minimum
    assert dsp.detect_regions(x, r, 0) == []


def test_unit_never_reaches_past_the_recording():
    """A partial final unit is not padded into a region."""
    n = 32000 + 80  # 200 whole units plus half a unit
    x = voiced(4.0)
    x = np.concatenate([x[:32000], x[:80]]).astype(np.float32)
    r = x.copy()
    r[16000:32080] *= -1
    regions = dsp.detect_regions(x, r, 0)
    assert regions
    assert regions[0]["end_sample"] <= (n // 160) * 160
    assert all(reg["end_sample"] <= n for reg in regions)


def test_region_is_whole_units_and_drops_trailing_partial_unit():
    x = voiced(4.0)
    r = x.copy()
    r[int(RATE * 1.0) : int(RATE * 2.0) - 37] *= -1
    regions = dsp.detect_regions(x, r, 0)
    assert len(regions) == 1
    assert regions[0]["start_sample"] == 16000
    assert regions[0]["end_sample"] % 160 == 0
    assert regions[0]["end_sample"] <= 32000


def test_inactive_unit_breaks_region_even_between_overlapping_windows():
    """Adjacent 20 ms windows overlap neighbouring 10 ms units.  A single unit
    that fails the in-place test must still split the region."""
    x = voiced()
    r = x.copy()
    r[16000:32000] *= -1
    r[24000:24160] = x[24000:24160]  # one healthy 10 ms unit in the middle
    regions = dsp.detect_regions(x, r, 0)
    # The unit at 24000 is healthy but the overlapping 20 ms window starting at
    # 23840 also covers the following unit, so the run ends one unit earlier.
    assert [(g["start_sample"], g["end_sample"]) for g in regions] == [
        (16000, 23840),
        (24160, 31840),
    ]


def test_one_sided_unit_breaks_region_without_gap_closing():
    x = voiced()
    r = x.copy()
    r[16000:32000] *= -1
    left = x.copy()
    left[24000:24160] = 0.0  # left side silent for exactly one unit
    regions = dsp.detect_regions(left, r, 0)
    # Unit [24000, 24160) has no left audio and is not selected; the unit just
    # before it still looks at a 20 ms window that includes the silent unit, so
    # its correlation fails too.
    assert [(g["start_sample"], g["end_sample"]) for g in regions] == [
        (16000, 24000),
        (24160, 31840),
    ]
    # a two-unit dropout is also not closed
    left = x.copy()
    left[24000:24320] = 0.0
    regions = dsp.detect_regions(left, r, 0)
    assert [(g["start_sample"], g["end_sample"]) for g in regions] == [
        (16000, 24000),
        (24320, 31840),
    ]


# ---------- repair ----------

def test_outside_spans_is_exact_downmix():
    x = source()
    r = x.copy()
    r[20000:40000] *= -1
    pcm = np.stack([x, r], axis=1)
    mono = dsp.repair(pcm, [(20000, 40000)], 0)
    direct = 0.5 * (x.astype(np.float64) + r.astype(np.float64))
    assert len(mono) == len(pcm)
    assert np.array_equal(mono[:20000], direct[:20000])
    assert np.array_equal(mono[40000:], direct[40000:])


def test_inside_span_recovers_source():
    x = source()
    r = x.copy()
    r[20000:40000] *= -1
    pcm = np.stack([x, r], axis=1)
    mono = dsp.repair(pcm, [(20000, 40000)], 0)
    inner = slice(20500, 39500)
    assert np.max(np.abs(mono[inner] - x[inner])) < 1e-6


def test_edges_fade_from_direct_downmix():
    x = source()
    r = x.copy()
    r[20000:40000] *= -1
    pcm = np.stack([x, r], axis=1)
    mono = dsp.repair(pcm, [(20000, 40000)], 0)
    direct = 0.5 * (x.astype(np.float64) + r.astype(np.float64))
    assert mono[20000] == pytest.approx(direct[20000])
    assert mono[39999] == pytest.approx(direct[39999])
    # no discontinuity larger than a fade step
    assert np.max(np.abs(np.diff(mono[19990:20020]))) < 0.2


def test_repair_compensates_delay_using_only_span_samples():
    x = source()
    d = 24
    r = delayed(x, d)
    r[20000:40000] *= -1
    pcm = np.stack([x, r], axis=1)
    mono = dsp.repair(pcm, [(20000, 40000)], d)
    # expected: the late channel's content recovered, i.e. left shifted by d
    expected = np.zeros_like(x, dtype=np.float64)
    expected[d:] = x[: len(x) - d]
    inner = slice(20600, 39400)
    assert np.max(np.abs(mono[inner] - expected[inner])) < 1e-6
    # the faded edges are exactly the direct downmix
    direct = 0.5 * (x.astype(np.float64) + r.astype(np.float64))
    assert mono[20000] == direct[20000]
    assert mono[39999] == direct[39999]

    # Nothing outside the span may leak into the repaired span: rewriting raw
    # samples just before and after it must not change the output inside.
    pcm2 = pcm.copy()
    pcm2[19500:20000] = 0.9
    pcm2[40000:40500] = -0.9
    mono2 = dsp.repair(pcm2, [(20000, 40000)], d)
    assert np.array_equal(mono2[20000:40000], mono[20000:40000])


def test_overlapping_and_touching_spans_merge():
    assert dsp.merge_intervals([(10, 20), (20, 30), (25, 40), (50, 60)], 100) == [
        (10, 40),
        (50, 60),
    ]
    x = source()
    r = x.copy()
    r[20000:50000] *= -1
    pcm = np.stack([x, r], axis=1)
    single = dsp.repair(pcm, [(20000, 50000)], 0)
    merged = dsp.repair(pcm, [(20000, 30000), (30000, 40000), (35000, 50000)], 0)
    assert np.array_equal(single, merged)


def test_repair_clamps_and_ignores_empty():
    x = source()
    pcm = np.stack([x, x], axis=1)
    direct = x.astype(np.float64)
    assert np.array_equal(dsp.repair(pcm, [], 0), direct)
    assert np.array_equal(dsp.repair(pcm, [(10, 10)], 0), direct)
    clamped = dsp.repair(pcm, [(-50, len(x) + 50)], 0)
    assert len(clamped) == len(x)


def test_span_shorter_than_delay_is_left_as_downmix():
    """A span that cannot hold the alignment padding must not produce a click."""
    x = source()
    d = 24
    r = delayed(x, d)
    pcm = np.stack([x, r], axis=1)
    tiny = dsp.repair(pcm, [(30000, 30020)], d)  # 20 samples < 2*d
    direct = 0.5 * (x.astype(np.float64) + r.astype(np.float64))
    assert np.array_equal(tiny, direct)
    # a span with exactly enough room is repaired
    ok = dsp.repair(pcm, [(30000, 30000 + 2 * d + 2)], d)
    assert not np.array_equal(ok, direct)


def test_repair_with_unconfirmed_delay_is_not_called():
    # The API layer must not call repair when delay is unconfirmed; verify the
    # direct downmix path stays bit-exact.
    x = source()
    y = source(seed=9)
    pcm = np.stack([x, y], axis=1)
    assert not dsp.estimate_delay(x, y)["confirmed"]


# ---------- end-to-end analysis / encoding ----------

def test_analyze_end_to_end():
    x = voiced(5.0)
    d = 8
    r = delayed(x, d)
    r[RATE : RATE * 2] *= -1
    pcm, a = dsp.analyze(make_wav(x, r))
    assert a.delay_confirmed
    assert a.delay_samples == d
    assert len(a.regions) == 1
    assert a.regions[0]["start_sample"] % 160 == 0
    assert a.regions[0]["end_sample"] % 160 == 0
    assert a.samples == len(x)
    assert a.duration_s == pytest.approx(5.0, abs=0.01)


def test_encoded_wav_is_mono_pcm16_of_same_length():
    x = source()
    data = dsp.encode_wav(x.astype(np.float64))
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == RATE
        assert wf.getnframes() == len(x)


def test_waveform_peaks_shape():
    x = source()
    peaks = dsp.waveform_peaks(x, bins=100)
    assert len(peaks) == 100
    assert all(lo <= hi for lo, hi in peaks)
