"""Unit tests for G.711 mu-law <-> PCM16/float32 audio transcoding and framing."""

from __future__ import annotations

import base64

import numpy as np

from app.audio_codec import (
    MULAW_FRAME_BYTES_20MS,
    mulaw_b64_to_float32_16k,
    mulaw_bytes_to_pcm16,
    pcm16_to_mulaw_b64_frames,
    pcm16_to_mulaw_bytes,
    resample_pcm16,
)


def test_mulaw_roundtrip_preserves_sine_wave() -> None:
    t = np.linspace(0.0, 0.1, 800, endpoint=False)
    sine_pcm16 = (np.sin(2.0 * np.pi * 440.0 * t) * 16000).astype(np.int16)

    encoded = pcm16_to_mulaw_bytes(sine_pcm16)
    assert len(encoded) == 800

    decoded = mulaw_bytes_to_pcm16(encoded)
    assert decoded.shape == sine_pcm16.shape
    correlation = np.corrcoef(sine_pcm16.astype(np.float32), decoded.astype(np.float32))[0, 1]
    assert correlation > 0.995


def test_resample_8k_to_16k_and_back() -> None:
    src_8k = (np.sin(np.linspace(0, 10, 160)) * 10000).astype(np.int16)
    up_16k = resample_pcm16(src_8k, 8000, 16000)
    assert up_16k.size == 320

    down_8k = resample_pcm16(up_16k, 16000, 8000)
    assert down_8k.size == 160


def test_mulaw_b64_to_float32_16k_and_framing() -> None:
    t = np.linspace(0.0, 0.04, 640, endpoint=False)
    pcm_16k = (np.sin(2.0 * np.pi * 300.0 * t) * 12000).astype(np.int16)

    frames = list(pcm16_to_mulaw_b64_frames(pcm_16k, sample_rate=16000))
    assert len(frames) == 2
    for frame_b64 in frames:
        raw = base64.b64decode(frame_b64)
        assert len(raw) == MULAW_FRAME_BYTES_20MS
        f32_16k = mulaw_b64_to_float32_16k(frame_b64)
        assert f32_16k.size == 320
        assert float(np.max(np.abs(f32_16k))) <= 1.0
