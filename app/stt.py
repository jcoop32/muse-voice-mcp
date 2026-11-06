"""Local Speech-to-Text engine powered by faster-whisper with low-latency CPU tuning."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Patterns of Whisper noise/ringtone hallucinations to ignore
_HALLUCINATION_RE = re.compile(
    r"^(?:(?:ee|ah|oh|uh|um|mm|hm|la|da)\s*){4,}$", re.IGNORECASE
)
_HALLUCINATION_PHRASES = (
    "thank you for watching",
    "thanks for watching",
    "subtitles by",
    "amara.org",
)


def is_hallucinated_transcript(text: str) -> bool:
    """Return True if Whisper output is a ringtone/silence hallucination (e.g. 'ee ee ee...')."""
    cleaned = text.strip()
    if not cleaned:
        return True
    if _HALLUCINATION_RE.match(cleaned):
        return True
    lower = cleaned.lower()
    if any(p in lower for p in _HALLUCINATION_PHRASES):
        return True
    words = lower.split()
    if len(words) >= 6 and len(set(words)) == 1:
        return True
    return False


class LocalWhisperSTT:
    """Low-latency faster-whisper transcriber cached on Longhorn persistent storage."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Path | None = None,
        custom_transcribe_fn: Callable[[np.ndarray], str] | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self._custom_transcribe_fn = custom_transcribe_fn
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None or self._custom_transcribe_fn is not None:
            return
        from faster_whisper import WhisperModel

        root = str(self.download_root) if self.download_root else None
        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s root=%s",
            self.model_size,
            self.device,
            self.compute_type,
            root,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=4,
            num_workers=1,
            download_root=root,
        )

    def warmup(self) -> None:
        """Pre-load the Whisper model into RAM at server startup so the first call has zero load delay."""
        if self._custom_transcribe_fn is not None:
            return
        try:
            self._ensure_model()
            dummy = np.zeros(8000, dtype=np.float32)
            self.transcribe_sync(dummy)
            logger.info("Pre-warmed faster-whisper (%s) successfully.", self.model_size)
        except Exception as exc:
            logger.warning("Whisper warmup skipped: %s", exc)

    def transcribe_sync(self, audio_16k_f32: np.ndarray) -> str:
        """Synchronously transcribe 16kHz float32 mono audio into text."""
        if audio_16k_f32.size == 0:
            return ""
        if self._custom_transcribe_fn is not None:
            return self._custom_transcribe_fn(audio_16k_f32)

        self._ensure_model()
        segments, _info = self._model.transcribe(
            audio_16k_f32,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            without_timestamps=True,
            max_new_tokens=64,
            no_speech_threshold=0.6,
            repetition_penalty=1.2,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        if is_hallucinated_transcript(text):
            logger.debug("Filtered Whisper noise hallucination: %r", text[:80])
            return ""
        return text

    async def transcribe(self, audio_16k_f32: np.ndarray) -> str:
        """Non-blocking async wrapper around faster-whisper transcription."""
        return await asyncio.to_thread(self.transcribe_sync, audio_16k_f32)
