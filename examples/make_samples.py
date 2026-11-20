"""Generate example WAVs that exercise the detector.

Usage (from the repository root, with the app dependencies installed):

    python examples/make_samples.py examples/out

Produces:
  * ``speech_inverted.wav``  - 6 s, right channel 1.5 ms late, two spans with
    polarity inverted, separated by clean audio.
  * ``clean.wav``            - same material, no inversion.
  * ``one_sided.wav``        - left-only speech; must not be flagged.
  * ``decorrelated.wav``     - two unrelated channels; delay must be unconfirmed.
  * ``no_delay_inverted.wav``- inverted with zero delay.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

RATE = 16_000


def speech_like(seconds: float, seed: int) -> np.ndarray:
    """Deterministic band-limited, amplitude-modulated pseudo-speech."""
    rng = np.random.default_rng(seed)
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    noise = rng.standard_normal(n)
    # crude band limit: sum of 3 smoothed noise copies at different scales
    kernel = np.hanning(33)
    kernel /= kernel.sum()
    smooth = np.convolve(noise, kernel, mode="same")
    carrier = 0.6 * np.sin(2 * np.pi * 170 * t) + 0.4 * np.sin(2 * np.pi * 530 * t + 0.7)
    syll = 0.55 + 0.45 * np.sin(2 * np.pi * 2.1 * t) ** 2
    phrase = np.ones(n)
    # pause between phrases
    for a, b in ((0.9, 1.35), (2.8, 3.2), (4.6, 4.95), (5.6, 6.0)):
        ia, ib = int(a * RATE), int(b * RATE)
        phrase[ia:ib] = 0.0
    env = syll * phrase
    x = (0.32 * smooth + 0.18 * carrier) * env
    return np.clip(x, -0.98, 0.98).astype(np.float32)


def write_stereo(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    n = min(len(left), len(right))
    pcm = np.stack([left[:n], right[:n]], axis=1)
    pcm = np.rint(np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/out")
    out.mkdir(parents=True, exist_ok=True)

    src = speech_like(6.0, seed=11)
    n = len(src)

    # 1.5 ms late right channel.
    d = 24
    delayed = np.zeros_like(src)
    delayed[d:] = src[: n - d]

    inverted = delayed.copy()
    inverted[int(1.4 * RATE) : int(2.5 * RATE)] *= -1.0
    inverted[int(3.6 * RATE) : int(4.45 * RATE)] *= -1.0
    write_stereo(out / "speech_inverted.wav", src, inverted)
    write_stereo(out / "clean.wav", src, delayed)

    one_sided = np.zeros_like(src)
    write_stereo(out / "one_sided.wav", src, one_sided)

    rng = np.random.default_rng(99)
    write_stereo(out / "decorrelated.wav", src, (rng.standard_normal(n) * 0.12).astype(np.float32))

    write_stereo(out / "no_delay_inverted.wav", src, -src)

    print(f"wrote examples to {out.resolve()}")
    for p in sorted(out.glob("*.wav")):
        print(f"  {p.name}  {p.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
