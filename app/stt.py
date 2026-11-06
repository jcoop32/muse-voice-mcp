"""Local Speech-to-Text engine powered by faster-whisper."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class LocalWhisperSTT:
    """Lazy-loaded faster-whisper transcriber cached on Longhorn persistent storage."""

    def __init__(
        self,
        model_size: str = "base.en",
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
            download_root=root,
        )

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
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return text.strip()

    async def transcribe(self, audio_16k_f32: np.ndarray) -> str:
        """Non-blocking async wrapper around faster-whisper transcription."""
        return await asyncio.to_thread(self.transcribe_sync, audio_16k_f32)
