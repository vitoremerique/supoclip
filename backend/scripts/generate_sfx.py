"""Synthesize transition SFX (no third-party samples): whoosh, impact, swish.

Usage (inside the worker): .venv/bin/python scripts/generate_sfx.py /app/transitions/sfx
"""
import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
rng = np.random.default_rng(7)


def sweep_filter(noise, f_lo, f_hi):
    """Band-pass white noise whose centre follows f_lo -> f_hi (arrays per sample)."""
    out = np.zeros_like(noise)
    lp = hp_in = hp_out = 0.0
    for i, x in enumerate(noise):
        a_lp = 1.0 - np.exp(-2 * np.pi * f_hi[i] / SR)
        lp += a_lp * (x - lp)
        a_hp = np.exp(-2 * np.pi * f_lo[i] / SR)
        hp_out = a_hp * (hp_out + lp - hp_in)
        hp_in = lp
        out[i] = hp_out
    return out


def envelope(n, peak_at, attack_pow=2.0, release_pow=3.0):
    t = np.linspace(0, 1, n)
    env = np.where(t < peak_at, (t / peak_at) ** attack_pow, ((1 - t) / (1 - peak_at)) ** release_pow)
    return env


def whoosh(duration=0.55, peak_at=0.62, f0=250, f1=4200, f2=700):
    n = int(SR * duration)
    t = np.linspace(0, 1, n)
    centre = np.where(t < peak_at, f0 + (f1 - f0) * (t / peak_at) ** 1.5,
                      f1 + (f2 - f1) * ((t - peak_at) / (1 - peak_at)) ** 0.7)
    left = sweep_filter(rng.standard_normal(n), centre * 0.45, centre * 1.6)
    right = sweep_filter(rng.standard_normal(n), centre * 0.45, centre * 1.6)
    env = envelope(n, peak_at)
    pan = np.clip(t * 1.4 - 0.2, 0, 1)            # moves left -> right
    return np.stack([left * env * (1.1 - pan), right * env * (0.1 + pan)], axis=1)


def impact(duration=0.8):
    n = int(SR * duration)
    t = np.arange(n) / SR
    freq = 45 + 95 * np.exp(-t * 18)               # pitch drop 140 -> 45 Hz
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t * 6.5)
    click_n = int(SR * 0.025)
    click = np.zeros(n)
    click[:click_n] = rng.standard_normal(click_n) * np.exp(-np.linspace(0, 6, click_n))
    air = sweep_filter(rng.standard_normal(n), np.full(n, 1800.0), np.full(n, 7000.0)) * np.exp(-t * 14) * 0.5
    mono = body + 0.6 * click + air
    return np.stack([mono, mono], axis=1)


def save(path, data, peak_db=-3.0):
    data = data / (np.max(np.abs(data)) + 1e-9) * (10 ** (peak_db / 20))
    fade = int(SR * 0.01)
    data[:fade] *= np.linspace(0, 1, fade)[:, None]
    data[-fade:] *= np.linspace(1, 0, fade)[:, None]
    pcm = (data * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(path, f"{len(data) / SR:.2f}s")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    save(out / "whoosh.wav", whoosh())
    save(out / "swish.wav", whoosh(duration=0.32, peak_at=0.55, f0=600, f1=6500, f2=1500), peak_db=-5.0)
    save(out / "impact.wav", impact())
