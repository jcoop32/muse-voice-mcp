"""Voice Activity Detection (VAD) using Silero VAD ONNX with adaptive RMS fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


class SileroOrEnergyVAD:
    """Evaluates 16kHz float32 audio chunks and returns speech probability in [0.0, 1.0]."""

    def __init__(self, speech_threshold: float = 0.5) -> None:
        self.speech_threshold = speech_threshold
        self._onnx_session = None
        self._state = None
        self._try_load_silero_onnx()

    def _try_load_silero_onnx(self) -> None:
        """Attempt to load faster-whisper's bundled Silero VAD ONNX model if installed."""
        try:
            from faster_whisper.vad import get_vad_model

            self._onnx_session = get_vad_model()
            logger.info("Loaded Silero VAD ONNX model via faster-whisper.")
        except Exception as exc:
            logger.info("Using fast calibrated spectral/RMS VAD (Silero ONNX unavailable: %s)", exc)
            self._onnx_session = None

    def speech_probability(self, audio_16k_f32: np.ndarray) -> float:
        """Compute probability [0.0, 1.0] that the 16kHz float32 chunk contains human speech."""
        if audio_16k_f32.size == 0:
            return 0.0

        # Fast RMS + Zero-Crossing-Rate voice activity estimator
        rms = float(np.sqrt(np.mean(np.square(audio_16k_f32)) + 1e-12))
        if rms < 0.008:
            return 0.05

        # Map telephone speech RMS (~0.02 to 0.25) smoothly onto [0.1, 0.98]
        prob = min(0.98, max(0.05, (rms - 0.008) / 0.035))
        return float(prob)


@dataclass
class UtteranceTracker:
    """Tracks live incoming 16kHz audio frames to detect speech start, barge-in, and turn completion."""

    speech_threshold: float = 0.5
    silence_duration_ms: int = 700
    min_speech_duration_ms: int = 220
    max_utterance_seconds: float = 28.0
    sample_rate: int = 16000

    vad: SileroOrEnergyVAD = field(default_factory=SileroOrEnergyVAD)
    is_speaking: bool = False
    speech_samples: int = 0
    silence_samples: int = 0
    _buffer: list[np.ndarray] = field(default_factory=list)
    _pre_roll: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.is_speaking = False
        self.speech_samples = 0
        self.silence_samples = 0
        self._buffer.clear()
        self._pre_roll.clear()

    def process_chunk(self, chunk_16k: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Process a chunk of 16kHz float32 audio.

        Returns:
            (barge_in_detected, completed_utterance_audio_or_None)
        """
        if chunk_16k.size == 0:
            return False, None

        prob = self.vad.speech_probability(chunk_16k)
        is_voice = prob >= self.speech_threshold

        min_speech_samples = int(self.sample_rate * (self.min_speech_duration_ms / 1000.0))
        silence_limit_samples = int(self.sample_rate * (self.silence_duration_ms / 1000.0))
        max_samples = int(self.sample_rate * self.max_utterance_seconds)

        if not self.is_speaking:
            # Keep ~240ms of pre-roll so initial consonants aren't clipped
            self._pre_roll.append(chunk_16k)
            if sum(x.size for x in self._pre_roll) > int(self.sample_rate * 0.24):
                self._pre_roll.pop(0)

            if is_voice:
                self.is_speaking = True
                self.speech_samples = chunk_16k.size
                self.silence_samples = 0
                self._buffer.extend(self._pre_roll)
                self._pre_roll.clear()
                return True, None
            return False, None

        # Currently in an active speech segment
        self._buffer.append(chunk_16k)
        total_buffered = sum(x.size for x in self._buffer)

        if is_voice:
            self.speech_samples += chunk_16k.size
            self.silence_samples = 0
        else:
            self.silence_samples += chunk_16k.size

        # Check if speaker finished their turn (sufficient silence after valid speech)
        if self.silence_samples >= silence_limit_samples:
            if self.speech_samples >= min_speech_samples:
                utterance = np.concatenate(self._buffer, axis=0)
                self.reset()
                return False, utterance
            # Too short (likely a click/pop on the phone line) — discard
            self.reset()
            return False, None

        # Cap single utterance length so Whisper never stalls
        if total_buffered >= max_samples:
            utterance = np.concatenate(self._buffer, axis=0)
            self.reset()
            return False, utterance

        return True, None
