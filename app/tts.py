"""Natural Neural Text-to-Speech engine supporting Edge-TTS (Azure Neural), Piper, and Kokoro-ONNX."""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import socket
import subprocess
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from app.audio_codec import pcm16_to_mulaw_b64_frames

logger = logging.getLogger(__name__)

# Force IPv4 resolution for speech.platform.bing.com to avoid IPv6 happy-eyeballs delays
_ORIG_GETADDRINFO = socket.getaddrinfo


def _fast_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host == "speech.platform.bing.com":
        family = socket.AF_INET
    return _ORIG_GETADDRINFO(host, port, family, type, proto, flags)


socket.getaddrinfo = _fast_getaddrinfo  # type: ignore[assignment]

DEFAULT_FILLER_PHRASES: tuple[str, ...] = (
    "Got it, one moment.",
    "Okay, let me check on that.",
    "Alright, one second.",
)


class LocalTTSEngine:
    """Synthesizes text into 8kHz G.711 mu-law base64 frames with natural prosody and caching."""

    def __init__(
        self,
        engine: str = "edge",
        edge_voice: str = "en-US-AndrewMultilingualNeural",
        piper_model: str = "en_US-lessac-medium",
        kokoro_voice: str = "af_heart",
        models_dir: Path | None = None,
        custom_synth_fn: Callable[[str], tuple[np.ndarray, int]] | None = None,
    ) -> None:
        self.engine = engine
        self.edge_voice = edge_voice
        self.piper_model = piper_model
        self.kokoro_voice = kokoro_voice
        self.models_dir = models_dir
        self._custom_synth_fn = custom_synth_fn
        self._piper_voice = None
        self._kokoro = None
        self._filler_cache: list[list[str]] = []
        self._filler_idx = 0
        self._frame_cache: dict[str, list[str]] = {}

    async def synthesize_mulaw_frames(self, text: str) -> list[str]:
        """Synthesize text into 20ms base64 8kHz mu-law frames (using fast neural Edge-TTS by default)."""
        cleaned = text.strip()
        if not cleaned:
            return []

        cached = self._frame_cache.get(cleaned)
        if cached is not None:
            return list(cached)

        if self._custom_synth_fn is not None:
            pcm16, sample_rate = await asyncio.to_thread(self._custom_synth_fn, cleaned)
            frames = list(pcm16_to_mulaw_b64_frames(pcm16, sample_rate))
            self._store_cache(cleaned, frames)
            return frames

        # 1. Primary: Natural Neural Voice via Edge-TTS (0ms startup, conversational inflection)
        if self.engine == "edge":
            try:
                pcm16, sample_rate = await self._synth_with_edge_async(cleaned)
                if pcm16.size > 0:
                    frames = list(pcm16_to_mulaw_b64_frames(pcm16, sample_rate))
                    self._store_cache(cleaned, frames)
                    return frames
            except Exception as exc:
                logger.warning("Edge-TTS failed (%s), falling back to local Piper TTS.", exc)

        # 2. Fallback: Local synchronous synthesis (Kokoro / Piper / espeak-ng)
        pcm16, sample_rate = await asyncio.to_thread(self.synthesize_pcm16_sync, cleaned)
        if pcm16.size == 0:
            return []
        frames = list(pcm16_to_mulaw_b64_frames(pcm16, sample_rate))
        self._store_cache(cleaned, frames)
        return frames

    def _store_cache(self, text: str, frames: list[str]) -> None:
        if len(self._frame_cache) > 64:
            oldest_key = next(iter(self._frame_cache))
            self._frame_cache.pop(oldest_key, None)
        self._frame_cache[text] = list(frames)

    async def _synth_with_edge_async(self, text: str) -> tuple[np.ndarray, int]:
        """Stream neural speech from edge-tts and decode MP3 -> 8000Hz mono int16 PCM via ffmpeg."""
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            voice=self.edge_voice,
            rate="+5%",
            boundary="SentenceBoundary",
        )
        mp3_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                mp3_chunks.append(chunk["data"])

        if not mp3_chunks:
            raise RuntimeError("edge-tts returned empty audio stream")

        mp3_bytes = b"".join(mp3_chunks)
        pcm8k = await asyncio.to_thread(self._decode_mp3_to_pcm8k, mp3_bytes)
        return pcm8k, 8000

    @staticmethod
    def _decode_mp3_to_pcm8k(mp3_bytes: bytes) -> np.ndarray:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "8000",
                "pipe:1",
            ],
            input=mp3_bytes,
            check=True,
            capture_output=True,
            timeout=10,
        )
        return np.frombuffer(proc.stdout, dtype=np.int16).copy()

    def synthesize_pcm16_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Synchronous local TTS fallback (Kokoro-ONNX -> Piper ONNX -> espeak-ng)."""
        cleaned = text.strip()
        if not cleaned:
            return np.zeros(0, dtype=np.int16), 16000

        if self._custom_synth_fn is not None:
            return self._custom_synth_fn(cleaned)

        if self.engine == "kokoro":
            try:
                return self._synth_with_kokoro(cleaned)
            except Exception as exc:
                logger.warning("Kokoro-ONNX fallback to Piper/espeak (%s)", exc)

        try:
            return self._synth_with_piper(cleaned)
        except Exception as exc:
            logger.debug("Piper TTS unavailable (%s), trying system TTS fallback.", exc)

        if shutil.which("espeak-ng"):
            try:
                return self._synth_with_espeak(cleaned)
            except Exception as exc:
                logger.warning("espeak-ng failed: %s", exc)

        return self._synth_tone_placeholder(cleaned)

    def _synth_with_kokoro(self, text: str) -> tuple[np.ndarray, int]:
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            model_path = (self.models_dir or Path(".")) / "kokoro-v1.0.onnx"
            voices_path = (self.models_dir or Path(".")) / "voices-v1.0.bin"
            self._kokoro = Kokoro(str(model_path), str(voices_path))
        samples_f32, sample_rate = self._kokoro.create(
            text, voice=self.kokoro_voice, speed=1.0, lang="en-us"
        )
        pcm16 = np.clip(np.asarray(samples_f32) * 32767.0, -32768, 32767).astype(np.int16)
        return pcm16, int(sample_rate)

    def _synth_with_piper(self, text: str) -> tuple[np.ndarray, int]:
        models_dir = self.models_dir or Path("/tmp/muse-voice-data/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_onnx = models_dir / f"{self.piper_model}.onnx"
        model_json = models_dir / f"{self.piper_model}.onnx.json"

        piper_bin = shutil.which("piper")
        if piper_bin:
            if not model_onnx.exists() or not model_json.exists():
                self._download_piper_voice(model_onnx, model_json)
            proc = subprocess.run(
                [piper_bin, "--model", str(model_onnx), "--output_file", "-"],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=15,
            )
            return self._decode_wav_bytes(proc.stdout)

        if self._piper_voice is None:
            from piper.voice import PiperVoice

            if not model_onnx.exists():
                raise FileNotFoundError(f"Piper model not found at {model_onnx}")
            self._piper_voice = PiperVoice.load(str(model_onnx))

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self._piper_voice.synthesize(text, wav_file)
        return self._decode_wav_bytes(buf.getvalue())

    @staticmethod
    def _download_piper_voice(model_onnx: Path, model_json: Path) -> None:
        import urllib.request

        base_url = (
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
            "en/en_US/lessac/medium/en_US-lessac-medium"
        )
        logger.info("Downloading Piper neural voice model to %s ...", model_onnx)
        urllib.request.urlretrieve(f"{base_url}.onnx", str(model_onnx))
        urllib.request.urlretrieve(f"{base_url}.onnx.json", str(model_json))

    def _synth_with_espeak(self, text: str) -> tuple[np.ndarray, int]:
        proc = subprocess.run(
            ["espeak-ng", "--stdout", "-s", "165", "-v", "en-us", text],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return self._decode_wav_bytes(proc.stdout)

    @staticmethod
    def _decode_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_in:
            sample_rate = wav_in.getframerate()
            frames = wav_in.readframes(wav_in.getnframes())
            pcm16 = np.frombuffer(frames, dtype=np.int16).copy()
            if wav_in.getnchannels() > 1:
                pcm16 = pcm16[:: wav_in.getnchannels()]
            return pcm16, sample_rate

    @staticmethod
    def _synth_tone_placeholder(text: str) -> tuple[np.ndarray, int]:
        sample_rate = 8000
        word_count = max(1, len(text.split()))
        duration = min(2.5, max(0.25, word_count * 0.08))
        t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
        wave_f32 = 0.15 * np.sin(2.0 * np.pi * 440.0 * t)
        return (wave_f32 * 32767.0).astype(np.int16), sample_rate

    async def get_next_filler_frames(self) -> list[str]:
        """Return pre-cached 20ms mu-law frames for a natural conversational backchannel."""
        if not self._filler_cache:
            for phrase in DEFAULT_FILLER_PHRASES:
                frames = await self.synthesize_mulaw_frames(phrase)
                if frames:
                    self._filler_cache.append(frames)
        if not self._filler_cache:
            return []
        frames = self._filler_cache[self._filler_idx % len(self._filler_cache)]
        self._filler_idx += 1
        return frames
