"""Local Text-to-Speech engine supporting Piper TTS, Kokoro-ONNX, and fallback synthesis."""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from app.audio_codec import pcm16_to_mulaw_b64_frames

logger = logging.getLogger(__name__)

DEFAULT_FILLER_PHRASES: tuple[str, ...] = (
    "Got it, one moment.",
    "Okay, let me check on that.",
    "Alright, one second.",
)


class LocalTTSEngine:
    """Synthesizes text into 8kHz G.711 mu-law base64 frames for Telnyx/Twilio streams."""

    def __init__(
        self,
        engine: str = "piper",
        piper_model: str = "en_US-lessac-medium",
        kokoro_voice: str = "af_heart",
        models_dir: Path | None = None,
        custom_synth_fn: Callable[[str], tuple[np.ndarray, int]] | None = None,
    ) -> None:
        self.engine = engine
        self.piper_model = piper_model
        self.kokoro_voice = kokoro_voice
        self.models_dir = models_dir
        self._custom_synth_fn = custom_synth_fn
        self._piper_voice = None
        self._kokoro = None
        self._filler_cache: list[list[str]] = []
        self._filler_idx = 0

    def synthesize_pcm16_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text into (int16_pcm_samples, sample_rate)."""
        cleaned = text.strip()
        if not cleaned:
            return np.zeros(0, dtype=np.int16), 16000

        if self._custom_synth_fn is not None:
            return self._custom_synth_fn(cleaned)

        # 1. Try Kokoro-ONNX if requested and available
        if self.engine == "kokoro":
            try:
                return self._synth_with_kokoro(cleaned)
            except Exception as exc:
                logger.warning("Kokoro-ONNX fallback to Piper/espeak (%s)", exc)

        # 2. Try Piper Python API or CLI binary
        try:
            return self._synth_with_piper(cleaned)
        except Exception as exc:
            logger.debug("Piper TTS unavailable (%s), trying system TTS fallback.", exc)

        # 3. Try espeak-ng CLI (installed in container image)
        if shutil.which("espeak-ng"):
            try:
                return self._synth_with_espeak(cleaned)
            except Exception as exc:
                logger.warning("espeak-ng failed: %s", exc)

        # 4. Deterministic soft tone envelope fallback (for unit tests / bare environments)
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

        # If piper CLI binary is installed, auto-fetch the ONNX voice model if missing and run piper CLI
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
        # Roughly 60ms per word, clamped between 0.25s and 2.5s
        word_count = max(1, len(text.split()))
        duration = min(2.5, max(0.25, word_count * 0.08))
        t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
        wave_f32 = 0.15 * np.sin(2.0 * np.pi * 440.0 * t)
        return (wave_f32 * 32767.0).astype(np.int16), sample_rate

    async def synthesize_mulaw_frames(self, text: str) -> list[str]:
        """Synthesize text asynchronously into a list of 20ms base64 8kHz mu-law frames."""
        pcm16, sample_rate = await asyncio.to_thread(self.synthesize_pcm16_sync, text)
        if pcm16.size == 0:
            return []
        return list(pcm16_to_mulaw_b64_frames(pcm16, sample_rate))

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
