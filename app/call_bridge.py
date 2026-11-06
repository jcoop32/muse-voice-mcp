"""Synchronous Turn-by-Turn Call Bridge connecting Telnyx/Twilio WebSockets to Muse MCP tools."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from app.audio_codec import mulaw_b64_to_float32_16k
from app.config import Settings
from app.stt import LocalWhisperSTT
from app.telephony import TelephonyDriver
from app.tts import LocalTTSEngine
from app.vad import UtteranceTracker

logger = logging.getLogger(__name__)


@dataclass
class TranscriptEntry:
    speaker: str  # "muse" | "remote" | "system"
    text: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class OutboundAudioItem:
    frames_b64: list[str]
    label: str = "speech"
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


class CallSession:
    """State and synchronization primitives for a single outbound phone call."""

    def __init__(
        self,
        call_id: str,
        phone_number: str,
        call_context: str,
        settings: Settings,
        stt: LocalWhisperSTT,
        tts: LocalTTSEngine,
    ) -> None:
        self.call_id = call_id
        self.phone_number = phone_number
        self.call_context = call_context
        self.settings = settings
        self.stt = stt
        self.tts = tts

        self.provider: str = settings.telephony_provider
        self.provider_call_id: str = ""
        self.stream_sid: str = ""
        self.status: str = "dialing"  # dialing, connected, completed, busy, no-answer, failed
        self.created_at: float = time.time()
        self.ended_at: float | None = None
        self.summary_for_user: str = ""

        self.transcript: list[TranscriptEntry] = []
        self.latest_remote_utterance: str = ""

        # Synchronization events for MCP turn-by-turn blocking
        self.human_finished_speaking = asyncio.Event()
        self.call_terminated = asyncio.Event()
        self.ws_connected = asyncio.Event()

        # Audio pipeline state
        self.outbound_queue: asyncio.Queue[OutboundAudioItem | None] = asyncio.Queue()
        self.is_playing_tts: bool = False
        self.abort_current_playback: bool = False
        self.websocket: WebSocket | None = None
        self._filler_task: asyncio.Task[None] | None = None

        self.utterance_tracker = UtteranceTracker(
            speech_threshold=settings.vad_speech_threshold,
            silence_duration_ms=settings.vad_silence_duration_ms,
            min_speech_duration_ms=settings.vad_min_speech_duration_ms,
        )

    def record_turn(self, speaker: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self.transcript.append(TranscriptEntry(speaker=speaker, text=cleaned))
        if speaker == "remote":
            self.latest_remote_utterance = cleaned
            self.human_finished_speaking.set()

    def cancel_pending_filler(self) -> None:
        if self._filler_task and not self._filler_task.done():
            self._filler_task.cancel()
        self._filler_task = None

    def schedule_delayed_filler(self, delay_seconds: float | None = None) -> None:
        """Schedule a soft filler clip ONLY if Muse takes longer than delay_seconds to reply."""
        self.cancel_pending_filler()
        wait_sec = (
            delay_seconds
            if delay_seconds is not None
            else self.settings.filler_delay_seconds
        )
        if wait_sec <= 0 or self.status != "connected":
            return

        async def _delayed_filler() -> None:
            try:
                await asyncio.sleep(wait_sec)
                if self.status == "connected" and not self.is_playing_tts:
                    frames = await self.tts.get_next_filler_frames()
                    if frames:
                        await self.outbound_queue.put(
                            OutboundAudioItem(frames_b64=frames, label="filler")
                        )
            except asyncio.CancelledError:
                return

        self._filler_task = asyncio.create_task(_delayed_filler())

    async def enqueue_tts_and_wait(self, text: str) -> None:
        """Synthesize text via local TTS, stream to the phone line, and wait until spoken."""
        self.cancel_pending_filler()
        cleaned = text.strip()
        if not cleaned:
            return
        self.record_turn("muse", cleaned)
        frames = await self.tts.synthesize_mulaw_frames(cleaned)
        if not frames or self.websocket is None:
            return
        item = OutboundAudioItem(frames_b64=frames, label="muse_turn")
        await self.outbound_queue.put(item)
        # Wait for the WebSocket sender loop to finish streaming (or call termination)
        done_wait = asyncio.create_task(item.done_event.wait())
        term_wait = asyncio.create_task(self.call_terminated.wait())
        await asyncio.wait(
            {done_wait, term_wait},
            timeout=max(5.0, len(frames) * 0.025 + 3.0),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in (done_wait, term_wait):
            if not task.done():
                task.cancel()

    async def wait_for_remote_reply(
        self, timeout_seconds: float, arm_filler_on_speech: bool = True
    ) -> tuple[str, str]:
        """Block until the remote human speaks and finishes, or timeout/hangup occurs.

        Returns:
            (outcome_status, transcribed_text)
            where outcome_status is one of: "replied", "on_hold_or_silence", "ended"
        """
        if self.call_terminated.is_set() or self.status in (
            "completed",
            "busy",
            "no-answer",
            "failed",
        ):
            return "ended", self.latest_remote_utterance

        speech_task = asyncio.create_task(self.human_finished_speaking.wait())
        term_task = asyncio.create_task(self.call_terminated.wait())

        done, pending = await asyncio.wait(
            {speech_task, term_task},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

        if speech_task in done:
            self.human_finished_speaking.clear()
            text = self.latest_remote_utterance
            if arm_filler_on_speech:
                self.schedule_delayed_filler()
            return "replied", text

        if term_task in done:
            return "ended", self.latest_remote_utterance

        return "on_hold_or_silence", ""

    async def ingest_mulaw_b64(self, payload_b64: str) -> None:
        """Process an incoming 8kHz mu-law base64 frame from Telnyx/Twilio."""
        chunk_16k = mulaw_b64_to_float32_16k(payload_b64)
        barge_in, completed_audio = self.utterance_tracker.process_chunk(chunk_16k)

        if barge_in and self.is_playing_tts:
            # Remote person interrupted while Muse was speaking -> clear playback immediately
            self.abort_current_playback = True
            if self.websocket is not None and self.stream_sid:
                try:
                    await self.websocket.send_json(
                        {"event": "clear", "streamSid": self.stream_sid}
                    )
                except Exception:
                    pass

        if completed_audio is not None:
            transcript_text = await self.stt.transcribe(completed_audio)
            if transcript_text:
                logger.info(
                    "[call_id=%s] Remote party said: %s", self.call_id, transcript_text
                )
                self.record_turn("remote", transcript_text)

    def mark_terminated(self, final_status: str = "completed") -> None:
        self.cancel_pending_filler()
        self.status = final_status
        if self.ended_at is None:
            self.ended_at = time.time()
        self.call_terminated.set()
        try:
            self.outbound_queue.put_nowait(None)
        except Exception:
            pass

    def to_dict(self) -> dict[str, Any]:
        duration = round((self.ended_at or time.time()) - self.created_at, 2)
        return {
            "call_id": self.call_id,
            "phone_number": self.phone_number,
            "call_context": self.call_context,
            "provider": self.provider,
            "provider_call_id": self.provider_call_id,
            "status": self.status,
            "duration_seconds": duration,
            "summary_for_user": self.summary_for_user,
            "latest_remote_utterance": self.latest_remote_utterance,
            "transcript": [
                {
                    "speaker": entry.speaker,
                    "text": entry.text,
                    "timestamp": entry.timestamp,
                }
                for entry in self.transcript
            ],
        }


class CallBridgeManager:
    """Manages active and historical CallSessions, WebSocket loops, and persistence."""

    def __init__(
        self,
        settings: Settings,
        stt: LocalWhisperSTT | None = None,
        tts: LocalTTSEngine | None = None,
        telephony: TelephonyDriver | None = None,
    ) -> None:
        self.settings = settings
        self.stt = stt or LocalWhisperSTT(
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            download_root=settings.models_dir,
        )
        self.tts = tts or LocalTTSEngine(
            engine=settings.tts_engine,
            piper_model=settings.piper_voice_model,
            kokoro_voice=settings.kokoro_voice,
            models_dir=settings.models_dir,
        )
        self.telephony = telephony or TelephonyDriver(settings)
        self.sessions: dict[str, CallSession] = {}

    def get_session(self, call_id: str) -> CallSession:
        session = self.sessions.get(call_id)
        if session is None:
            raise KeyError(f"Unknown call_id '{call_id}'.")
        return session

    async def start_call(
        self,
        phone_number: str,
        call_context: str = "",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Dial an external number, wait for their initial greeting, and return it to Muse."""
        call_id = f"call_{uuid.uuid4().hex[:10]}"
        session = CallSession(
            call_id=call_id,
            phone_number=phone_number,
            call_context=call_context,
            settings=self.settings,
            stt=self.stt,
            tts=self.tts,
        )
        self.sessions[call_id] = session

        dial_info = await self.telephony.dial_outbound(call_id, phone_number)
        session.provider = dial_info["provider"]
        session.provider_call_id = dial_info["provider_call_id"]

        wait_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.settings.default_answer_timeout_seconds
        )

        outcome, greeting = await session.wait_for_remote_reply(
            timeout_seconds=wait_timeout, arm_filler_on_speech=True
        )
        if outcome == "replied":
            session.status = "connected"
            return {
                "call_id": call_id,
                "status": "connected",
                "provider": session.provider,
                "they_said": greeting,
                "next_action_hint": (
                    "Call speak_and_listen(call_id, what_to_say) to respond, "
                    "send_dtmf(call_id, digits) if facing an automated menu, "
                    "or end_phone_call(call_id, parting_words) when done."
                ),
            }

        if outcome == "ended":
            self.persist_session(session)
            return {
                "call_id": call_id,
                "status": session.status,
                "provider": session.provider,
                "they_said": greeting,
                "message": f"Call ended during dialing with status '{session.status}'.",
            }

        return {
            "call_id": call_id,
            "status": "connected_silent",
            "provider": session.provider,
            "they_said": "",
            "next_action_hint": (
                "Line connected or ringing with no greeting after timeout. "
                "Call speak_and_listen(call_id, 'Hello?') or end_phone_call(call_id)."
            ),
        }

    async def speak_and_listen(
        self,
        call_id: str,
        what_to_say: str,
        listen_timeout_seconds: float | None = None,
        play_filler_after_reply: bool = True,
    ) -> dict[str, Any]:
        """Speak Muse's line over the phone, wait for the person to reply, and return their words."""
        session = self.get_session(call_id)
        session.human_finished_speaking.clear()

        await session.enqueue_tts_and_wait(what_to_say)

        wait_timeout = (
            listen_timeout_seconds
            if listen_timeout_seconds is not None
            else self.settings.default_listen_timeout_seconds
        )
        outcome, reply_text = await session.wait_for_remote_reply(
            timeout_seconds=wait_timeout,
            arm_filler_on_speech=play_filler_after_reply,
        )

        if outcome == "replied":
            return {
                "call_id": call_id,
                "status": "replied",
                "they_said": reply_text,
            }
        if outcome == "ended":
            self.persist_session(session)
            return {
                "call_id": call_id,
                "status": "call_ended_by_remote",
                "they_said": reply_text,
            }
        return {
            "call_id": call_id,
            "status": "on_hold_or_silence",
            "they_said": "",
            "hint": (
                f"No speech detected for {wait_timeout:.0f}s (person may be checking or holding). "
                "Call wait_on_hold(call_id) to continue listening, or speak_and_listen to check in."
            ),
        }

    async def wait_on_hold(
        self, call_id: str, timeout_seconds: float = 30.0
    ) -> dict[str, Any]:
        """Keep listening while on hold without speaking."""
        session = self.get_session(call_id)
        session.cancel_pending_filler()
        outcome, reply_text = await session.wait_for_remote_reply(
            timeout_seconds=timeout_seconds, arm_filler_on_speech=True
        )
        if outcome == "replied":
            return {
                "call_id": call_id,
                "status": "replied",
                "they_said": reply_text,
            }
        if outcome == "ended":
            self.persist_session(session)
            return {
                "call_id": call_id,
                "status": "call_ended_by_remote",
                "they_said": reply_text,
            }
        return {
            "call_id": call_id,
            "status": "still_on_hold",
            "they_said": "",
            "hint": "Still silent/on hold. Call wait_on_hold(call_id) again or end_phone_call(call_id).",
        }

    async def send_dtmf(
        self,
        call_id: str,
        digits: str,
        listen_after: bool = True,
        listen_timeout_seconds: float = 20.0,
    ) -> dict[str, Any]:
        """Send keypad tones (e.g. '1', '2', '0#') to navigate automated IVR menus."""
        session = self.get_session(call_id)
        session.cancel_pending_filler()
        session.human_finished_speaking.clear()
        ws_url = f"{self.settings.public_wss_base}/ws/media/{call_id}"

        sent = await self.telephony.send_dtmf(
            session.provider_call_id, digits, ws_url, call_id
        )
        session.record_turn("system", f"[Sent DTMF digits: {digits}]")

        if not listen_after:
            return {"call_id": call_id, "status": "dtmf_sent", "sent": sent, "they_said": ""}

        outcome, reply_text = await session.wait_for_remote_reply(
            timeout_seconds=listen_timeout_seconds, arm_filler_on_speech=False
        )
        return {
            "call_id": call_id,
            "status": outcome,
            "sent": sent,
            "they_said": reply_text,
        }

    async def end_call(
        self, call_id: str, parting_words: str = "", summary_for_user: str = ""
    ) -> dict[str, Any]:
        """Speak optional parting words, hang up the call, and persist the full transcript."""
        session = self.get_session(call_id)
        session.cancel_pending_filler()
        if summary_for_user:
            session.summary_for_user = summary_for_user

        if parting_words.strip() and session.status == "connected":
            await session.enqueue_tts_and_wait(parting_words)

        await self.telephony.hangup(session.provider_call_id)
        session.mark_terminated("completed")
        saved_path = self.persist_session(session)
        result = session.to_dict()
        result["saved_transcript_path"] = str(saved_path)
        return result

    async def handle_media_websocket(self, websocket: WebSocket, call_id: str) -> None:
        """Bidirectional Telnyx/Twilio Media Stream WebSocket loop (8kHz PCMU <-> VAD/STT/TTS)."""
        await websocket.accept()
        session = self.sessions.get(call_id)
        if session is None:
            await websocket.close(code=1008, reason="Unknown call_id")
            return

        session.websocket = websocket
        session.status = "connected"
        session.ws_connected.set()

        sender_task = asyncio.create_task(self._outbound_sender_loop(websocket, session))
        try:
            while True:
                raw_msg = await websocket.receive_text()
                msg = json.loads(raw_msg)
                event_type = msg.get("event")

                if event_type == "start":
                    start_obj = msg.get("start", {})
                    session.stream_sid = (
                        start_obj.get("streamSid")
                        or msg.get("stream_id")
                        or session.stream_sid
                    )
                elif event_type == "media":
                    media_obj = msg.get("media", {})
                    # Ignore outbound echo frames if provider streams both tracks
                    track = media_obj.get("track", "inbound")
                    if track in ("inbound", "inbound_track"):
                        payload_b64 = media_obj.get("payload", "")
                        if payload_b64:
                            await session.ingest_mulaw_b64(payload_b64)
                elif event_type == "stop":
                    break
        except Exception as exc:
            logger.debug("Media WebSocket closed for call_id=%s: %s", call_id, exc)
        finally:
            sender_task.cancel()
            session.mark_terminated(
                session.status if session.status != "connected" else "completed"
            )
            self.persist_session(session)

    async def _outbound_sender_loop(
        self, websocket: WebSocket, session: CallSession
    ) -> None:
        """Pulls 20ms 8kHz mu-law frames from session.outbound_queue and streams them to Telnyx/Twilio."""
        while not session.call_terminated.is_set():
            item = await session.outbound_queue.get()
            if item is None:
                break

            session.is_playing_tts = True
            session.abort_current_playback = False
            try:
                for frame_b64 in item.frames_b64:
                    if session.abort_current_playback or session.call_terminated.is_set():
                        break
                    outbound_msg: dict[str, Any] = {
                        "event": "media",
                        "media": {"payload": frame_b64},
                    }
                    if session.stream_sid:
                        outbound_msg["streamSid"] = session.stream_sid
                    await websocket.send_json(outbound_msg)
                    await asyncio.sleep(0.018)
            except Exception as exc:
                logger.debug("Outbound frame send stopped for %s: %s", session.call_id, exc)
            finally:
                session.is_playing_tts = False
                item.done_event.set()

    def persist_session(self, session: CallSession) -> Path:
        """Write the call session metadata and full transcript to the Longhorn volume."""
        out_path = self.settings.calls_dir / f"{session.call_id}.json"
        out_path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
        return out_path
