"""Configuration settings for Muse Voice MCP Bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:
    """Application settings loaded from environment variables."""

    # Server & Networking
    app_name: str = "muse-voice-mcp"
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8090")))
    public_domain: str = field(
        default_factory=lambda: os.getenv("PUBLIC_DOMAIN", "voice-mcp.cooperlabs.app")
    )
    mcp_api_key: str = field(default_factory=lambda: os.getenv("MCP_API_KEY", ""))
    webhook_signing_secret: str = field(
        default_factory=lambda: os.getenv("WEBHOOK_SIGNING_SECRET", "")
    )

    # Telephony Provider: "telnyx_sip" (default, zero Cloudflare/webhooks needed), "telnyx", or "twilio"
    telephony_provider: str = field(
        default_factory=lambda: os.getenv("TELEPHONY_PROVIDER", "telnyx_sip").lower().strip()
    )
    outbound_caller_number: str = field(
        default_factory=lambda: os.getenv("OUTBOUND_CALLER_NUMBER", "")
    )

    # Telnyx Direct Outbound SIP Credentials (No public webhook/Cloudflare needed)
    telnyx_sip_host: str = field(
        default_factory=lambda: os.getenv("TELNYX_SIP_HOST", "sip.telnyx.com")
    )
    telnyx_sip_port: int = field(
        default_factory=lambda: int(os.getenv("TELNYX_SIP_PORT", "5060"))
    )
    telnyx_sip_username: str = field(
        default_factory=lambda: os.getenv("TELNYX_SIP_USERNAME", "")
    )
    telnyx_sip_password: str = field(
        default_factory=lambda: os.getenv("TELNYX_SIP_PASSWORD", "")
    )

    # Telnyx Configuration ($1.00/mo number, ~$0.005-$0.007/min outbound)
    telnyx_api_key: str = field(default_factory=lambda: os.getenv("TELNYX_API_KEY", ""))
    telnyx_connection_id: str = field(
        default_factory=lambda: os.getenv("TELNYX_CONNECTION_ID", "")
    )

    # Twilio Configuration ($1.15/mo number, ~$0.018/min outbound + media streaming)
    twilio_account_sid: str = field(
        default_factory=lambda: os.getenv("TWILIO_ACCOUNT_SID", "")
    )
    twilio_auth_token: str = field(
        default_factory=lambda: os.getenv("TWILIO_AUTH_TOKEN", "")
    )

    # Local Speech-to-Text (faster-whisper)
    whisper_model_size: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
    )
    whisper_device: str = field(
        default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu")
    )
    whisper_compute_type: str = field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    )

    # Text-to-Speech (edge neural voice with local piper fallback)
    tts_engine: str = field(
        default_factory=lambda: os.getenv("TTS_ENGINE", "edge").lower().strip()
    )
    edge_tts_voice: str = field(
        default_factory=lambda: os.getenv(
            "EDGE_TTS_VOICE", "en-US-AndrewMultilingualNeural"
        )
    )
    piper_voice_model: str = field(
        default_factory=lambda: os.getenv("PIPER_VOICE_MODEL", "en_US-lessac-medium")
    )
    kokoro_voice: str = field(
        default_factory=lambda: os.getenv("KOKORO_VOICE", "af_heart")
    )

    # VAD & Turn-Taking Parameters
    vad_speech_threshold: float = field(
        default_factory=lambda: float(os.getenv("VAD_SPEECH_THRESHOLD", "0.5"))
    )
    vad_silence_duration_ms: int = field(
        default_factory=lambda: int(os.getenv("VAD_SILENCE_DURATION_MS", "420"))
    )
    vad_min_speech_duration_ms: int = field(
        default_factory=lambda: int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "180"))
    )

    # Filler & Timeout Guards (0.0 disables automatic filler so it never blocks real replies)
    filler_delay_seconds: float = field(
        default_factory=lambda: float(os.getenv("FILLER_DELAY_SECONDS", "0.0"))
    )
    default_answer_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_ANSWER_TIMEOUT_SECONDS", "45.0"))
    )
    default_listen_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_LISTEN_TIMEOUT_SECONDS", "25.0"))
    )

    # Persistent Storage (Longhorn PVC mounted at /data)
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "/tmp/muse-voice-data"))
    )

    @property
    def models_dir(self) -> Path:
        path = self.data_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def calls_dir(self) -> Path:
        path = self.data_dir / "calls"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def fillers_dir(self) -> Path:
        path = self.data_dir / "fillers"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def public_https_base(self) -> str:
        if self.public_domain.startswith(("http://", "https://")):
            return self.public_domain.rstrip("/")
        return f"https://{self.public_domain}"

    @property
    def public_wss_base(self) -> str:
        base = self.public_https_base
        if base.startswith("https://"):
            return "wss://" + base[len("https://") :]
        if base.startswith("http://"):
            return "ws://" + base[len("http://") :]
        return f"wss://{base}"


def get_settings() -> Settings:
    return Settings()
