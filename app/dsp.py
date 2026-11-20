"""Analysis and repair for near phase-inverted stereo pairs.

The analysis answers two separate questions:

1. *Is a constant inter-channel delay detectable?*  A gated, normalised
   cross-correlation is searched over +/-4 ms.  Because a polarity-inverted
   channel produces a negative correlation, the search maximises |r|; the sign
   at the winning lag is reported separately.  A delay is only "confirmed"
   when the winning lag is an interior peak, |r| clears a high floor, it beats
   lag 0 and the next separated lobe by a margin, and enough jointly-active
   audio exists.  Otherwise the tool offers the direct downmix only.
2. *Where does the direct downmix lose energy?*  Per 20 ms frame with both
   channels active, the direct average is compared with the best coherent
   energy.  A sustained loss plus a negative aligned channel correlation marks
   suspected anti-phase.  Frames with silence or a single active side never
   qualify.  Detection only runs once a delay is confirmed: without alignment
   the cancellation measurement is meaningless, and the contract then offers
   the direct downmix alone.

Repair works on merged selections only.  The base buffer is always exactly
0.5 * (L + R); inside a selection the aligned right/left channel is inverted
and cross-faded in.  Alignment uses only samples from that merged selection.
"""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000
MAX_SECONDS = 60
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS

MAX_LAG_MS = 2.0            # contractual assumption
SEARCH_LAG_MS = 4.0         # search range; peaks beyond MAX_LAG_MS are rejected

FRAME_MS = 20
HOP_MS = 10
MIN_RMS_DBFS = -50.0        # below this a channel counts as silent
SILENCE_GAP_BLOCKS = 2      # close detection gaps up to ~20 ms
MIN_REGION_MS = 80
MIN_LOSS_DB = 5.0           # downmix energy loss vs. the coherent sum
MIN_CORR = 0.45             # aligned channels must be anti-correlated this strongly
CORR_CONFIRM = 0.80         # |r| needed to confirm a constant delay
CORR_DOMINANCE = 0.10       # margin over lag 0 / next lobe
MIN_ACTIVE_MS = 300         # jointly-active audio needed to confirm a delay
FADE_MS = 5.0               # cross-fade at each end of a repaired selection


class AudioError(ValueError):
    """Raised when an upload is not an acceptable stereo PCM16 WAV."""


@dataclass
class Analysis:
    channels: int
    sample_rate: int
    samples: int
    duration_s: float
    delay_samples: int | None
    delay_ms: float | None
    delay_confirmed: bool
    delay_confidence: float
    delay_correlation: float
    delay_reason: str
    regions: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    downmix_rms_dbfs: float = -120.0

    @property
    def repairable(self) -> bool:
        return self.delay_confirmed


def _ms(samples: int | float, rate: int = SAMPLE_RATE) -> float:
    return round(float(samples) * 1000.0 / rate, 2)


def parse_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Return (samples [N, 2] float32 in [-1, 1), sample_rate)."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("文件不是 WAV（缺少 RIFF/WAVE 头）。")
    if len(data) > 96 * 1024 * 1024:
        raise AudioError("文件过大。")
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.getnframes()
            raw = wf.readframes(frames)
    except (wave.Error, EOFError, struct.error) as exc:
        raise AudioError(f"无法解析 WAV：{exc}") from exc

    if channels != 2:
        raise AudioError(f"需要双声道 WAV，当前为 {channels} 声道。")
    if sampwidth != 2:
        raise AudioError(f"需要 16 位 PCM，当前为 {sampwidth * 8} 位。")
    if rate != SAMPLE_RATE:
        raise AudioError(f"需要 {SAMPLE_RATE} Hz 采样率，当前为 {rate} Hz。")
    if frames == 0:
        raise AudioError("WAV 不含音频帧。")
    if frames > MAX_SAMPLES:
        raise AudioError(f"时长超过 {MAX_SECONDS} 秒（{frames / rate:.2f} 秒）。")
    if len(raw) != frames * 4:
        raise AudioError("WAV 数据长度与声道/位深不匹配。")

    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return pcm.reshape(-1, 2).copy(), rate


def encode_wav(mono: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode mono float samples as 16-bit PCM WAV (symmetric clipping)."""
    clipped = np.clip(np.asarray(mono, dtype=np.float64), -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _frame_rms_db(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """RMS in dBFS for frames starting at 0, hop, 2*hop, ..."""
    n = len(x)
    if n == 0:
        return np.zeros(0)
    if n < frame:
        rms = np.sqrt(np.mean(np.square(x, dtype=np.float64)))
        return np.array([20.0 * np.log10(max(rms, 1e-9))])
    count = 1 + (n - frame) // hop
    view = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop][:count]
    energy = np.mean(np.square(view, dtype=np.float64), axis=1)
    return 20.0 * np.log10(np.maximum(np.sqrt(energy), 1e-9))


def _smooth_mask(mask: np.ndarray, gap: int) -> np.ndarray:
    """Close interior gaps of at most ``gap`` blocks."""
    out = mask.astype(bool).copy()
    if gap <= 0 or not out.any():
        return out
    idx = np.flatnonzero(out)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            out[a + 1 : b] = True
    return out


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges where ``mask`` is True."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _next_pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def estimate_delay(left: np.ndarray, right: np.ndarray, rate: int = SAMPLE_RATE) -> dict:
    """Estimate the constant delay of ``right`` relative to ``left``.

    Positive lag means the right channel is late by that many samples.
    ``confirmed`` is False unless the evidence is unambiguous; in that case the
    caller must not offer delay compensation.

    The estimate is block-wise and uses |correlation| per block before
    averaging.  That keeps an inverted passage from cancelling the evidence of
    a non-inverted passage when both share the same physical delay.  Blocks
    where either channel is quiet are excluded, so a silent side cannot bias
    the result toward lag 0.
    """
    n = min(len(left), len(right))
    frame = max(1, int(round(rate * FRAME_MS / 1000)))
    hop = max(1, int(round(rate * HOP_MS / 1000)))
    max_lag = max(1, int(round(rate * SEARCH_LAG_MS / 1000)))

    db_l = _frame_rms_db(left[:n], frame, hop)
    db_r = _frame_rms_db(right[:n], frame, hop)
    joint = _smooth_mask((db_l > MIN_RMS_DBFS) & (db_r > MIN_RMS_DBFS), SILENCE_GAP_BLOCKS)
    active_ms = float(joint.sum()) * HOP_MS
    info = {
        "lag": 0,
        "lag_ms": 0.0,
        "confirmed": False,
        "abs_correlation": 0.0,
        "correlation": 0.0,
        "active_ms": round(active_ms, 1),
        "reason": "",
    }
    if active_ms < MIN_ACTIVE_MS or joint.sum() < 5:
        info["reason"] = "两声道共同活跃片段不足，无法确认固定时差。"
        return info

    # Block-wise GCC-style correlation.  Block length 250 ms is long enough to
    # average the noise down and short enough that a polarity flip usually
    # stays within one block.
    block = min(n, max(16 * max_lag, int(round(rate * 0.25))))
    block_hop = max(max_lag, block // 2)
    nfft = _next_pow2(block + 2 * max_lag + 1)
    lags = np.arange(-max_lag, max_lag + 1)
    idx = np.where(lags >= 0, lags, nfft + lags)

    acc_abs = np.zeros(len(lags))
    acc_signed = np.zeros(len(lags))
    total_weight = 0.0
    win = np.hanning(block)  # tame edge discontinuities between blocks
    for s in range(0, n - block + 1, block_hop):
        x = left[s : s + block].astype(np.float64) * win
        y = right[s : s + block].astype(np.float64) * win
        ex = float(np.dot(x, x))
        ey = float(np.dot(y, y))
        if ex <= 1e-9 or ey <= 1e-9:
            continue
        xr = 20.0 * np.log10(max(np.sqrt(ex / block), 1e-9))
        yr = 20.0 * np.log10(max(np.sqrt(ey / block), 1e-9))
        if xr <= MIN_RMS_DBFS or yr <= MIN_RMS_DBFS:
            continue
        # c[k] = sum y[n] x[n-k]; the FFT circular index maps -k to nfft-k, so
        # a positive k means y (right) is late by k samples.
        cc = np.fft.irfft(np.fft.rfft(y, nfft) * np.conj(np.fft.rfft(x, nfft)), nfft)
        vals = cc[idx] / np.sqrt(ex * ey)
        weight = np.sqrt(ex * ey)
        acc_abs += weight * np.abs(vals)
        acc_signed += weight * vals
        total_weight += weight

    if total_weight <= 0.0:
        info["reason"] = "没有足够的双声道活跃素材，无法确认固定时差。"
        return info

    mag = acc_abs / total_weight
    signed = acc_signed / total_weight
    peak_k = int(np.argmax(mag))
    peak_lag = int(lags[peak_k])
    peak = float(mag[peak_k])
    zero_k = int(np.flatnonzero(lags == 0)[0])
    info["abs_correlation"] = round(peak, 4)
    info["correlation"] = round(float(signed[peak_k]), 4)

    guard = max(1, int(round(rate * 0.001)))  # 1 ms exclusion around the peak
    runner_mask = np.ones(len(lags), dtype=bool)
    runner_mask[max(0, peak_k - guard) : peak_k + guard + 1] = False
    runner = float(mag[runner_mask].max()) if runner_mask.any() else 0.0
    zero_mag = float(mag[zero_k])

    # A real peak has fallen off well before both search limits.  If the
    # correlation at either edge is still close to the peak, the true optimum
    # most likely lies outside the searched range.
    edge = max(float(mag[0]), float(mag[-1]))
    interior = 0 < peak_k < len(lags) - 1 and (peak - edge) >= CORR_DOMINANCE

    if peak_lag == 0:
        dominant = peak - runner >= CORR_DOMINANCE
    else:
        dominant = peak - max(zero_mag, runner) >= CORR_DOMINANCE
    in_range = abs(peak_lag) <= int(round(rate * MAX_LAG_MS / 1000))

    if abs(peak_lag) <= 2:  # within 0.125 ms: report as zero, still valid
        peak_lag = 0
        in_range = True

    if peak < CORR_CONFIRM:
        info["reason"] = "两声道相关性过低，无法确认固定时差。"
    elif not in_range:
        info["reason"] = (
            f"最佳相关出现在约 {_ms(lags[peak_k], rate):g} ms，"
            f"超出 {MAX_LAG_MS:g} ms 的假定范围，按无法确认处理。"
        )
    elif not interior:
        info["reason"] = "相关峰不明确（可能时变或落在搜索边界），无法确认固定时差。"
    elif not dominant:
        info["reason"] = "相关峰与零时差/次峰的差异不足，无法确认固定时差。"
    else:
        info.update(
            {
                "confirmed": True,
                "lag": peak_lag,
                "lag_ms": round(peak_lag * 1000.0 / rate, 4),
            }
        )
    return info


def _make_region(start: int, end: int, loss: float, corr: float, rate: int) -> dict:
    return {
        "start_sample": int(start),
        "end_sample": int(end),
        "start_ms": _ms(start, rate),
        "end_ms": _ms(end, rate),
        "duration_ms": _ms(end - start, rate),
        "loss_db": round(loss, 2),
        "correlation": round(corr, 3),
    }


def detect_regions(left: np.ndarray, right: np.ndarray, delay: int, rate: int = SAMPLE_RATE) -> list[dict]:
    """Find intervals where the direct downmix cancels energy.

    ``delay`` is the right channel's lag in samples (positive: right is late)
    and is applied before comparing the channels.  Frames where either channel
    is silent are never candidates, so silence and single-sided audio are never
    flagged.

    Two independent pieces of evidence must agree: the direct average loses at
    least ``MIN_LOSS_DB`` relative to the best coherent energy, and the aligned
    channels are anti-correlated by at least ``MIN_CORR``.
    """
    n = min(len(left), len(right))
    frame = max(1, int(round(rate * FRAME_MS / 1000)))
    hop = max(1, int(round(rate * HOP_MS / 1000)))
    db_l = _frame_rms_db(left[:n], frame, hop)
    db_r = _frame_rms_db(right[:n], frame, hop)
    active_l = _smooth_mask(db_l > MIN_RMS_DBFS, SILENCE_GAP_BLOCKS)
    active_r = _smooth_mask(db_r > MIN_RMS_DBFS, SILENCE_GAP_BLOCKS)
    joint = active_l & active_r

    n_frames = len(db_l)
    loss_db = np.full(n_frames, np.nan)
    corr = np.full(n_frames, np.nan)
    valid = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        s = i * hop
        ls, le = s + max(0, -delay), s + max(0, -delay) + frame
        rs, re = s + max(0, delay), s + max(0, delay) + frame
        if le > n or re > n:
            continue
        if not joint[i]:
            continue
        a = left[ls:le].astype(np.float64)
        b = right[rs:re].astype(np.float64)
        a2 = float(np.dot(a, a))
        b2 = float(np.dot(b, b))
        if a2 <= 1e-12 or b2 <= 1e-12:
            continue
        dm = 0.5 * (a + b)
        direct = float(np.dot(dm, dm))
        best = 0.25 * (a2 + b2)
        loss_db[i] = 60.0 if direct <= 1e-12 else min(60.0, 10.0 * np.log10(best / direct))
        corr[i] = float(np.dot(a, b) / np.sqrt(a2 * b2))
        valid[i] = True

    flagged = valid & (loss_db >= MIN_LOSS_DB) & (corr <= -MIN_CORR)
    flagged = _smooth_mask(flagged, SILENCE_GAP_BLOCKS)

    min_frames = max(1, int(round(MIN_REGION_MS / HOP_MS)))
    regions: list[dict] = []
    for a, b in _segments(flagged):
        if b - a + 1 < min_frames:
            continue
        # A candidate must stay jointly active for its whole span.  If one side
        # drops out, keep only the largest jointly-active sub-run.
        span = joint[a : b + 1]
        if span.all():
            runs = [(a, b)]
        else:
            runs = [(a + x, a + y) for x, y in _segments(span)]
        for x, y in runs:
            if y - x + 1 < min_frames:
                continue
            # The region is expressed on the file's own (left-channel) time
            # axis; alignment only shifts which samples are compared.
            start = min(n, x * hop)
            end = min(n, y * hop + frame)
            if end <= start:
                continue
            regions.append(
                _make_region(
                    start,
                    end,
                    float(np.nanmean(loss_db[x : y + 1])),
                    float(np.nanmean(corr[x : y + 1])),
                    rate,
                )
            )

    regions.sort(key=lambda r: r["start_sample"])
    merged: list[dict] = []
    for r in regions:
        if merged and r["start_sample"] <= merged[-1]["end_sample"]:
            last = merged[-1]
            last["end_sample"] = max(last["end_sample"], r["end_sample"])
            last["end_ms"] = _ms(last["end_sample"], rate)
            last["duration_ms"] = _ms(last["end_sample"] - last["start_sample"], rate)
            last["loss_db"] = round(max(last["loss_db"], r["loss_db"]), 2)
            last["correlation"] = round(min(last["correlation"], r["correlation"]), 3)
        else:
            merged.append(dict(r))
    return merged


def waveform_peaks(mono: np.ndarray, bins: int = 1400) -> list[list[float]]:
    n = len(mono)
    if n == 0:
        return []
    bins = max(1, min(bins, n))
    bounds = np.linspace(0, n, bins + 1).astype(int)
    out: list[list[float]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            b = min(n, a + 1)
        seg = mono[a:b]
        out.append([round(float(seg.min()), 4), round(float(seg.max()), 4)])
    return out


def analyze(data: bytes) -> tuple[np.ndarray, Analysis]:
    pcm, rate = parse_wav(data)
    left, right = pcm[:, 0], pcm[:, 1]
    n = len(pcm)
    delay_info = estimate_delay(left, right, rate)
    # Without a confirmed delay there is nothing to align, so any "cancellation"
    # measurement would be unreliable.  In that case the contract is to offer
    # the direct downmix only, and no spans are marked.
    regions = detect_regions(left, right, delay_info["lag"], rate) if delay_info["confirmed"] else []
    direct = 0.5 * (left.astype(np.float64) + right.astype(np.float64))
    rms = float(np.sqrt(np.mean(np.square(direct)))) if n else 0.0

    notes: list[str] = []
    if delay_info["confirmed"]:
        lag = delay_info["lag"]
        if lag == 0:
            notes.append("两声道时差为 0（已确认），可直接翻转极性修复。")
        elif lag > 0:
            notes.append(
                f"确认右声道比左声道晚 {_ms(lag, rate):g} ms，修复时仅在该合并区间内对齐。"
            )
        else:
            notes.append(
                f"确认左声道比右声道晚 {_ms(-lag, rate):g} ms，修复时仅在该合并区间内对齐。"
            )
    else:
        notes.append(
            "时差未能确认，按约束只提供左右平均的直接下混，不提供修复输出。"
        )
    if not regions:
        notes.append("未发现疑似反相区间。")

    analysis = Analysis(
        channels=2,
        sample_rate=rate,
        samples=n,
        duration_s=round(n / rate, 3),
        delay_samples=delay_info["lag"] if delay_info["confirmed"] else None,
        delay_ms=delay_info["lag_ms"] if delay_info["confirmed"] else None,
        delay_confirmed=bool(delay_info["confirmed"]),
        delay_confidence=round(float(delay_info["abs_correlation"]), 4),
        delay_correlation=round(float(delay_info["correlation"]), 4),
        delay_reason=delay_info["reason"],
        regions=regions,
        notes=notes,
        downmix_rms_dbfs=round(20.0 * np.log10(max(rms, 1e-9)), 2),
    )
    return pcm, analysis


def merge_intervals(intervals: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Clamp, sort, and merge overlaps and touching intervals."""
    clean: list[tuple[int, int]] = []
    for a, b in intervals:
        a = max(0, min(int(a), n))
        b = max(0, min(int(b), n))
        if b > a:
            clean.append((a, b))
    clean.sort()
    merged: list[tuple[int, int]] = []
    for a, b in clean:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def repair(
    pcm: np.ndarray,
    intervals: list[tuple[int, int]],
    delay: int | None,
    rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return a mono repair with length exactly ``len(pcm)``.

    * Every sample outside the union of merged selections is exactly
      ``0.5 * (L + R)`` (the direct downmix).
    * Each merged selection is repaired on its own: the early channel is
      delayed by ``|delay|`` so it lines up with the late one, the late channel
      is inverted, and the pair is averaged.  Only samples from inside that
      merged selection are used; the |delay| samples exposed at the start of
      the selection are filled with silence rather than borrowed from outside.
    * The repaired signal is faded in and out against the direct downmix, so a
      selection's first and last samples are exactly the downmix.
    """
    n = len(pcm)
    left = pcm[:, 0].astype(np.float64)
    right = pcm[:, 1].astype(np.float64)
    direct = 0.5 * (left + right)
    out = direct.copy()
    merged = merge_intervals(intervals, n)
    if not merged:
        return out

    d = int(delay or 0)
    if abs(d) >= n:
        d = 0
    fade_len = max(1, int(round(rate * FADE_MS / 1000)))

    for a, b in merged:
        length = b - a
        if length < 2:
            continue
        # Aligning inside the span zero-fills its first |d| samples.  Refuse a
        # span too short for that padding to stay inside the fade; short spans
        # are left as the direct downmix rather than producing a click.
        shift = min(abs(d), length - 1)
        if length < 2 * shift + 2:
            continue
        span_l = left[a:b]
        span_r = right[a:b]

        # Delay the early channel by |d| inside this span only.
        if d > 0:
            # Right is late: delay left so left_aligned[t] ~ right[t].
            l_al = np.concatenate((np.zeros(shift), span_l[: length - shift]))
            r_al = span_r
        elif d < 0:
            # Right is early: delay right so right_aligned[t] ~ left[t].
            l_al = span_l
            r_al = np.concatenate((np.zeros(shift), span_r[: length - shift]))
        else:
            l_al, r_al = span_l, span_r

        # Flip the late channel; for a perfectly anti-phase pair this recovers
        # the source at full amplitude.
        flipped = 0.5 * (l_al - r_al)

        f = min(fade_len, max(1, length // 2))
        gain = np.ones(length)
        ramp = _smoothstep(np.linspace(0.0, 1.0, f, endpoint=True))
        gain[:f] = ramp
        gain[-f:] = np.minimum(gain[-f:], ramp[::-1])
        seg = direct[a:b].copy()
        mask = gain > 0.0
        seg[mask] = (1.0 - gain[mask]) * direct[a:b][mask] + gain[mask] * flipped[mask]
        out[a:b] = seg
    return out
