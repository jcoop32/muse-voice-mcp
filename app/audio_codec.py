"""G.711 mu-law (8000Hz) <-> PCM16 / Float32 (16000Hz) audio transcoding and framing."""

from __future__ import annotations

import base64
from typing import Iterator

import numpy as np

# Precompute ITU-T G.711 mu-law decode lookup table (256 uint8 -> int16)
_MULAW_TO_PCM16_TABLE: np.ndarray = np.empty(256, dtype=np.int16)
for _i in range(256):
    _val = ~_i & 0xFF
    _sign = _val & 0x80
    _exponent = (_val >> 4) & 0x07
    _mantissa = _val & 0x0F
    _sample = (((_mantissa << 3) + 0x84) << _exponent) - 0x84
    _MULAW_TO_PCM16_TABLE[_i] = -_sample if _sign else _sample

# 20ms of 8kHz mu-law audio = 160 bytes (standard Twilio/Telnyx RTP frame size)
MULAW_FRAME_BYTES_20MS = 160
MULAW_SILENCE_BYTE = 0xFF


def mulaw_bytes_to_pcm16(mulaw_bytes: bytes) -> np.ndarray:
    """Decode raw 8000Hz G.711 mu-law bytes into 8000Hz int16 PCM samples."""
    if not mulaw_bytes:
        return np.zeros(0, dtype=np.int16)
    indices = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    return _MULAW_TO_PCM16_TABLE[indices].copy()


def pcm16_to_mulaw_bytes(pcm16: np.ndarray) -> bytes:
    """Encode int16 PCM samples into ITU-T G.711 mu-law bytes."""
    if pcm16.size == 0:
        return b""
    samples = np.asarray(pcm16, dtype=np.int32)
    sign = np.where(samples < 0, 0x80, 0x00)
    abs_samples = np.minimum(np.abs(samples), 32635) + 0x84

    # Find exponent (0..7) based on highest set bit in (abs_samples >> 7)
    shifted = np.maximum(abs_samples >> 7, 1)
    exponent = np.floor(np.log2(shifted)).astype(np.int32)
    exponent = np.clip(exponent, 0, 7)

    mantissa = (abs_samples >> (exponent + 3)) & 0x0F
    mulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return mulaw.astype(np.uint8).tobytes()


def resample_pcm16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample 1D int16 PCM audio from src_rate to dst_rate via linear interpolation."""
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.int16, copy=False)
    if src_rate == 8000 and dst_rate == 16000:
        # Fast 2x upsample with midpoint interpolation
        out = np.empty(samples.size * 2, dtype=np.int32)
        s32 = samples.astype(np.int32)
        out[0::2] = s32
        out[1:-1:2] = (s32[:-1] + s32[1:]) // 2
        out[-1] = s32[-1]
        return np.clip(out, -32768, 32767).astype(np.int16)
    if src_rate == 16000 and dst_rate == 8000:
        # Fast 2x downsample with 2-tap anti-aliasing average
        even_len = (samples.size // 2) * 2
        if even_len == 0:
            return np.zeros(0, dtype=np.int16)
        s32 = samples[:even_len].astype(np.int32)
        avg = (s32[0::2] + s32[1::2]) // 2
        return np.clip(avg, -32768, 32767).astype(np.int16)

    duration = samples.size / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    interp = np.interp(x_new, x_old, samples.astype(np.float32))
    return np.clip(np.round(interp), -32768, 32767).astype(np.int16)


def mulaw_b64_to_float32_16k(payload_b64: str) -> np.ndarray:
    """Convert a Telnyx/Twilio base64 8kHz mu-law chunk into 16kHz float32 [-1.0, 1.0]."""
    raw = base64.b64decode(payload_b64)
    pcm8k = mulaw_bytes_to_pcm16(raw)
    pcm16k = resample_pcm16(pcm8k, 8000, 16000)
    return (pcm16k.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def pcm16_to_mulaw_b64_frames(
    pcm16: np.ndarray, sample_rate: int, frame_bytes: int = MULAW_FRAME_BYTES_20MS
) -> Iterator[str]:
    """Convert PCM16 audio at sample_rate into 20ms base64-encoded 8kHz mu-law frames."""
    pcm8k = resample_pcm16(pcm16, sample_rate, 8000)
    mulaw = pcm16_to_mulaw_bytes(pcm8k)
    for offset in range(0, len(mulaw), frame_bytes):
        chunk = mulaw[offset : offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + bytes([MULAW_SILENCE_BYTE]) * (frame_bytes - len(chunk))
        yield base64.b64encode(chunk).decode("ascii")
