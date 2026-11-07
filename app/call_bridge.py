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
from app.security_guard import (
    TrustProfile,
    classify_answering_system,
    classify_call_trust,
    detect_goodbye_intent,
    filter_outbound_speech_dlp,
    sanitize_for_muse_llm,
)
from app.sip_client import SipCallHandle
from app.stt import LocalWhisperSTT
from app.telephony import TelephonyDriver, normalize_phone_number
from app.tts import LocalTTSEngine
from app.vad import UtteranceTracker
from app.voice_brain import VoiceBrain, split_into_speakable_clauses

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

        # Security & Conversational Trust Profile (+12242458826 Owner vs. External Untrusted)
        self.trust_profile: TrustProfile = classify_call_trust(
            phone_number, settings.owner_phone_numbers
        )
        self.trust_level: str = self.trust_profile.trust_level
        self.allow_sensitive_tools: bool = self.trust_profile.allow_sensitive_tools
        self.conversation_mode: str = self.trust_profile.conversation_mode
        self.answering_state: str = "unknown"  # unknown, human, call_screener, voicemail
        self.security_events: list[str] = []

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
        self.sip_handle: SipCallHandle | None = None
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
        """Synthesize text (with Outbound DLP & sentence streaming), stream to the phone line, and wait until spoken."""
        self.cancel_pending_filler()
        cleaned = text.strip()
        if not cleaned:
            return

        # Outbound DLP Guard: block accidental secret/account leakage on external untrusted calls
        safe_text, dlp_redacted = filter_outbound_speech_dlp(
            cleaned, trust_level=self.trust_level, call_context=self.call_context
        )
        if dlp_redacted:
            self.security_events.append("outbound_dlp_redacted")

        self.record_turn("muse", safe_text)
        if self.websocket is None and self.sip_handle is None:
            await self.tts.synthesize_mulaw_frames(safe_text)
            return

        # Drain any stale filler items in the outbound queue and preempt active filler playback
        drained_items: list[OutboundAudioItem | None] = []
        while not self.outbound_queue.empty():
            try:
                queued = self.outbound_queue.get_nowait()
                if queued is not None and queued.label == "filler":
                    queued.done_event.set()
                else:
                    drained_items.append(queued)
            except asyncio.QueueEmpty:
                break
        for kept in drained_items:
            await self.outbound_queue.put(kept)

        if self.is_playing_tts:
            self.abort_current_playback = True

        # Sentence-chunked streaming: synthesize clause 1 and immediately start playback while clause 2+ synthesizes
        clauses = split_into_speakable_clauses(safe_text)
        if not clauses:
            return

        synth_tasks = [
            asyncio.create_task(self.tts.synthesize_mulaw_frames(clause))
            for clause in clauses
        ]
        queued_items: list[OutboundAudioItem] = []
        total_frames = 0
        for task in synth_tasks:
            frames = await task
            if frames:
                total_frames += len(frames)
                item = OutboundAudioItem(frames_b64=frames, label="muse_turn")
                queued_items.append(item)
                await self.outbound_queue.put(item)

        if not queued_items:
            return

        last_item = queued_items[-1]
        done_wait = asyncio.create_task(last_item.done_event.wait())
        term_wait = asyncio.create_task(self.call_terminated.wait())
        await asyncio.wait(
            {done_wait, term_wait},
            timeout=max(5.0, total_frames * 0.025 + 3.0),
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

    async def wait_for_answer_and_greeting(
        self,
        answer_timeout_seconds: float,
        post_pickup_greeting_seconds: float = 3.2,
    ) -> tuple[str, str]:
        """Wait for the call to be answered and for the initial 'Hello?' without dead air.

        Once the line connects (`ws_connected`), waits up to `post_pickup_greeting_seconds`
        for the callee's initial greeting so we never sit in 45s of silence if they answer quietly.
        """
        if self.call_terminated.is_set() or self.status in (
            "completed",
            "busy",
            "no-answer",
            "failed",
        ):
            return "ended", self.latest_remote_utterance

        speech_task = asyncio.create_task(self.human_finished_speaking.wait())
        connect_task = asyncio.create_task(self.ws_connected.wait())
        term_task = asyncio.create_task(self.call_terminated.wait())

        done, pending = await asyncio.wait(
            {speech_task, connect_task, term_task},
            timeout=answer_timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if speech_task in done:
            for t in pending:
                t.cancel()
            self.human_finished_speaking.clear()
            return "replied", self.latest_remote_utterance

        if term_task in done:
            for t in pending:
                t.cancel()
            return "ended", self.latest_remote_utterance

        if connect_task in done:
            # Call was answered! Wait up to post_pickup_greeting_seconds for their "Hello?"
            greeting_wait = min(answer_timeout_seconds, post_pickup_greeting_seconds)
            done2, pending2 = await asyncio.wait(
                {speech_task, term_task},
                timeout=greeting_wait,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # CRITICAL: If the remote speaker / call screener / voicemail is actively talking right now,
            # do NOT cut them off—wait until they finish their current sentence!
            if not done2 and self.utterance_tracker.is_speaking:
                done2, pending2 = await asyncio.wait(
                    {speech_task, term_task},
                    timeout=min(12.0, max(3.0, answer_timeout_seconds - greeting_wait)),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            for t in pending2:
                t.cancel()
            if speech_task in done2:
                self.human_finished_speaking.clear()
                return "replied", self.latest_remote_utterance
            if term_task in done2:
                return "ended", self.latest_remote_utterance
            return "connected_silent", ""

        for t in pending:
            t.cancel()
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

    def security_envelope(self, raw_remote_text: str) -> dict[str, Any]:
        """Build the standardized trust, prompt-injection firewall, and conversation metadata for Muse."""
        safe_text, was_flagged, reason = sanitize_for_muse_llm(
            raw_remote_text, self.trust_level
        )
        if was_flagged and reason:
            self.security_events.append(reason)
        sys_type = classify_answering_system(raw_remote_text)
        if sys_type in ("voicemail", "call_screener_prompt", "call_screener_hold"):
            self.answering_state = sys_type
        elif raw_remote_text.strip() and self.answering_state == "unknown":
            self.answering_state = "human"
        return {
            "they_said": safe_text,
            "trust_level": self.trust_level,
            "conversation_mode": self.conversation_mode,
            "allow_sensitive_tools": self.allow_sensitive_tools,
            "answering_state": self.answering_state,
            "goodbye_detected": detect_goodbye_intent(raw_remote_text),
            "prompt_injection_blocked": was_flagged,
            "security_directive": self.trust_profile.security_directive,
        }

    def to_dict(self) -> dict[str, Any]:
        duration = round((self.ended_at or time.time()) - self.created_at, 2)
        return {
            "call_id": self.call_id,
            "phone_number": self.phone_number,
            "call_context": self.call_context,
            "provider": self.provider,
            "provider_call_id": self.provider_call_id,
            "status": self.status,
            "trust_level": self.trust_level,
            "conversation_mode": self.conversation_mode,
            "allow_sensitive_tools": self.allow_sensitive_tools,
            "answering_state": self.answering_state,
            "security_events": list(self.security_events),
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
        voice_brain: VoiceBrain | None = None,
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
            edge_voice=settings.edge_tts_voice,
            models_dir=settings.models_dir,
        )
        self.telephony = telephony or TelephonyDriver(settings)
        self.voice_brain = voice_brain or VoiceBrain(
            api_key=settings.openrouter_api_key,
            model=settings.voice_brain_model,
        )
        self.sessions: dict[str, CallSession] = {}

    def get_session(self, call_id: str) -> CallSession:
        session = self.sessions.get(call_id)
        if session is None:
            raise KeyError(f"Unknown call_id '{call_id}'.")
        return session

    async def _navigate_call_screener_if_needed(
        self, session: CallSession, initial_greeting: str
    ) -> tuple[str, str]:
        """Automatically handle iOS/Android Call Screening ('record your name' -> 'please stay on the line')."""
        current_text = initial_greeting
        for _ in range(3):
            sys_state = classify_answering_system(current_text)
            if sys_state == "call_screener_prompt":
                session.answering_state = "call_screener"
                session.human_finished_speaking.clear()
                await session.enqueue_tts_and_wait(
                    "Hi, it's Bernard calling on behalf of Joshua Cooper."
                )
                outcome, next_text = await session.wait_for_remote_reply(
                    timeout_seconds=35.0, arm_filler_on_speech=False
                )
                if outcome != "replied":
                    return outcome, current_text
                current_text = next_text
                continue

            if sys_state == "call_screener_hold":
                session.answering_state = "call_screener"
                session.human_finished_speaking.clear()
                # Wait silently through the screening hold until the real human picks up
                outcome, next_text = await session.wait_for_remote_reply(
                    timeout_seconds=35.0, arm_filler_on_speech=False
                )
                if outcome != "replied":
                    return outcome, current_text
                current_text = next_text
                continue

            break
        return "replied", current_text

    async def start_call(
        self,
        phone_number: str,
        call_context: str = "",
        timeout_seconds: float | None = None,
        opening_line: str = "",
        auto_converse: bool = False,
    ) -> dict[str, Any]:
        """Dial an external number, handle call screening, enforce caller-ID trust, and optionally start conversation."""
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

        # Pre-synthesize opening_line while the call is dialing so there is zero TTS delay on pickup
        prewarm_task: asyncio.Task[list[str]] | None = None
        if opening_line.strip():
            prewarm_task = asyncio.create_task(
                self.tts.synthesize_mulaw_frames(opening_line.strip())
            )

        if (
            self.settings.telephony_provider == "telnyx_sip"
            and self.settings.telnyx_sip_username
            and self.settings.telnyx_sip_password
        ):
            normalized_to = normalize_phone_number(phone_number)
            sip_cid = f"{uuid.uuid4().hex}@muse-voice-mcp"
            handle = SipCallHandle(
                call_id=call_id,
                sip_call_id=sip_cid,
                session=session,
                sip_host=self.settings.telnyx_sip_host,
                sip_port=self.settings.telnyx_sip_port,
                sip_username=self.settings.telnyx_sip_username,
                sip_password=self.settings.telnyx_sip_password,
                caller_number=self.settings.outbound_caller_number,
                target_number=normalized_to,
            )
            session.sip_handle = handle
            session.provider = "telnyx_sip"
            session.provider_call_id = sip_cid
            handle.start_background()
        else:
            dial_info = await self.telephony.dial_outbound(call_id, phone_number)
            session.provider = dial_info["provider"]
            session.provider_call_id = dial_info["provider_call_id"]

        wait_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.settings.default_answer_timeout_seconds
        )

        outcome, greeting = await session.wait_for_answer_and_greeting(
            answer_timeout_seconds=wait_timeout,
            post_pickup_greeting_seconds=3.2 if opening_line.strip() else 3.8,
        )
        if prewarm_task is not None and not prewarm_task.done():
            try:
                await prewarm_task
            except Exception:
                pass

        if outcome == "ended":
            self.persist_session(session)
            res = {
                "call_id": call_id,
                "status": session.status,
                "provider": session.provider,
                "message": f"Call ended during dialing with status '{session.status}'.",
            }
            res.update(session.security_envelope(greeting))
            return res

        # If a call screener answered ("record your name" / "please stay on the line"), navigate it automatically
        if outcome == "replied" and greeting:
            outcome, greeting = await self._navigate_call_screener_if_needed(
                session, greeting
            )

        # Check if the call went to voicemail
        if classify_answering_system(greeting) == "voicemail":
            session.status = "connected"
            session.answering_state = "voicemail"
            res = {
                "call_id": call_id,
                "status": "voicemail",
                "provider": session.provider,
                "next_action_hint": (
                    "VOICEMAIL DETECTED: The person did not pick up live—their voicemail answered. "
                    "Either call end_phone_call(call_id, parting_words='') to hang up without leaving a message, "
                    "or call speak_and_listen/end_phone_call with a voicemail message."
                ),
            }
            res.update(session.security_envelope(greeting))
            return res

        # If an opening_line was provided and a live person answered, speak it immediately and listen for their reply
        if opening_line.strip() and outcome in ("replied", "connected_silent"):
            session.status = "connected"
            session.human_finished_speaking.clear()
            await session.enqueue_tts_and_wait(opening_line.strip())
            reply_outcome, first_reply = await session.wait_for_remote_reply(
                timeout_seconds=self.settings.default_listen_timeout_seconds,
                arm_filler_on_speech=False,
            )
            effective_text = first_reply or greeting
            if auto_converse and reply_outcome == "replied":
                return await self.converse_on_phone(call_id=call_id)
            res = {
                "call_id": call_id,
                "status": "connected" if reply_outcome == "replied" else reply_outcome,
                "provider": session.provider,
                "initial_greeting": greeting,
                "opening_line_spoken": opening_line.strip(),
                "next_action_hint": (
                    "Call converse_on_phone(call_id) for fast, natural multi-turn conversation, "
                    "or speak_and_listen(call_id, what_to_say) to reply turn-by-turn. "
                    "IMPORTANT: Do NOT hang up abruptly—keep the conversation natural until a real goodbye."
                ),
            }
            res.update(session.security_envelope(effective_text))
            return res

        if outcome == "replied":
            session.status = "connected"
            if auto_converse:
                return await self.converse_on_phone(call_id=call_id)
            res = {
                "call_id": call_id,
                "status": "connected",
                "provider": session.provider,
                "next_action_hint": (
                    "Call converse_on_phone(call_id) for fast, natural multi-turn conversation, "
                    "or speak_and_listen(call_id, what_to_say) to respond. "
                    "IMPORTANT: Do NOT hang up immediately after getting info—keep the conversation natural."
                ),
            }
            res.update(session.security_envelope(greeting))
            return res

        res = {
            "call_id": call_id,
            "status": "connected_silent",
            "provider": session.provider,
            "next_action_hint": (
                "Line connected with no initial greeting. "
                "Call speak_and_listen(call_id, 'Hey there!') immediately to greet them."
            ),
        }
        res.update(session.security_envelope(""))
        return res

    async def speak_and_listen(
        self,
        call_id: str,
        what_to_say: str,
        listen_timeout_seconds: float | None = None,
        play_filler_after_reply: bool = False,
    ) -> dict[str, Any]:
        """Speak Muse's line over the phone, wait for the person to reply, and return their sanitized words."""
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

        # If a call screener put us on hold ("Thank you. Please stay on the line."), wait for the real human!
        if outcome == "replied" and classify_answering_system(reply_text) == "call_screener_hold":
            session.answering_state = "call_screener"
            session.human_finished_speaking.clear()
            outcome, reply_text = await session.wait_for_remote_reply(
                timeout_seconds=35.0,
                arm_filler_on_speech=False,
            )

        if outcome == "replied":
            res = {
                "call_id": call_id,
                "status": "replied",
                "next_action_hint": (
                    "Continue the conversation naturally with speak_and_listen(call_id, what_to_say) "
                    "or converse_on_phone(call_id). Do NOT hang up unless goodbye_detected is True "
                    "or you have naturally wrapped up the conversation."
                ),
            }
            res.update(session.security_envelope(reply_text))
            return res
        if outcome == "ended":
            self.persist_session(session)
            res = {
                "call_id": call_id,
                "status": "call_ended_by_remote",
            }
            res.update(session.security_envelope(reply_text))
            return res
        res = {
            "call_id": call_id,
            "status": "on_hold_or_silence",
            "hint": (
                f"No speech detected for {wait_timeout:.0f}s (person may be checking or holding). "
                "Call wait_on_hold(call_id) to continue listening, or speak_and_listen to check in."
            ),
        }
        res.update(session.security_envelope(""))
        return res

    async def converse_on_phone(
        self,
        call_id: str,
        max_turns: int = 12,
        listen_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run a low-latency, natural multi-turn voice conversation using the in-pod VoiceBrain.

        - For +12242458826 (Owner): Open-ended casual conversation; yields to Muse only if an external tool is requested.
        - For External Numbers: Air-gapped zero-tool sandbox strictly scoped to call_context with prompt-injection & DLP firewall.
        """
        session = self.get_session(call_id)
        wait_timeout = (
            listen_timeout_seconds
            if listen_timeout_seconds is not None
            else self.settings.default_listen_timeout_seconds
        )

        turns_completed = 0
        while turns_completed < max_turns and session.status == "connected":
            turns_completed += 1
            history_dicts = [
                {"speaker": t.speaker, "text": t.text} for t in session.transcript
            ]
            decision = await self.voice_brain.generate_turn_reply(
                trust_level=session.trust_level,
                call_context=session.call_context,
                transcript_history=history_dicts,
                latest_remote_text=session.latest_remote_utterance,
            )
            if decision.blocked_injection and decision.injection_reason:
                session.security_events.append(decision.injection_reason)

            if decision.needs_owner_tool_handoff:
                await session.enqueue_tts_and_wait(decision.spoken_reply)
                res = {
                    "call_id": call_id,
                    "status": "owner_tool_handoff_requested",
                    "tool_handoff_request": decision.tool_handoff_request,
                    "next_action_hint": (
                        "Verified owner (+12242458826) requested an action requiring your tools. "
                        "Execute the requested tool, then call speak_and_listen(call_id, result) to tell Joshua!"
                    ),
                }
                res.update(session.security_envelope(session.latest_remote_utterance))
                return res

            if decision.should_end_call:
                return await self.end_call(
                    call_id=call_id,
                    parting_words=decision.spoken_reply,
                    summary_for_user=f"Completed natural {session.conversation_mode} conversation.",
                )

            session.human_finished_speaking.clear()
            await session.enqueue_tts_and_wait(decision.spoken_reply)
            outcome, reply_text = await session.wait_for_remote_reply(
                timeout_seconds=wait_timeout, arm_filler_on_speech=False
            )
            if outcome == "ended":
                self.persist_session(session)
                res = {
                    "call_id": call_id,
                    "status": "call_ended_by_remote",
                }
                res.update(session.security_envelope(reply_text))
                return res
            if outcome != "replied":
                res = {
                    "call_id": call_id,
                    "status": "on_hold_or_silence",
                }
                res.update(session.security_envelope(""))
                return res

        res = {
            "call_id": call_id,
            "status": "replied",
            "turns_completed": turns_completed,
            "next_action_hint": (
                "Call converse_on_phone(call_id) to continue the conversation or end_phone_call(call_id) when done."
            ),
        }
        res.update(session.security_envelope(session.latest_remote_utterance))
        return res

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

        if session.sip_handle is not None:
            sent = await session.sip_handle.send_dtmf(digits)
        else:
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

        if session.sip_handle is not None:
            await session.sip_handle.hangup()
        else:
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
