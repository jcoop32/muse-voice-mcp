"""Unit tests for CallSession, UtteranceTracker, Hold Guard, Delayed Filler, and DTMF."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from app.audio_codec import pcm16_to_mulaw_b64_frames
from app.call_bridge import CallBridgeManager
from app.config import Settings
from app.stt import LocalWhisperSTT
from app.tts import LocalTTSEngine


def _make_test_manager(tmp_path: Path) -> CallBridgeManager:
    settings = Settings(
        telephony_provider="telnyx",
        outbound_caller_number="+15550001111",
        data_dir=tmp_path,
        vad_silence_duration_ms=100,
        vad_min_speech_duration_ms=60,
        filler_delay_seconds=0.08,
        default_answer_timeout_seconds=1.5,
        default_listen_timeout_seconds=0.3,
    )
    stt = LocalWhisperSTT(
        custom_transcribe_fn=lambda audio: "Yes, our earliest tee time Saturday is 9:24 AM for $45."
    )
    tts = LocalTTSEngine(
        custom_synth_fn=lambda text: (
            (np.sin(np.linspace(0, 20, 800)) * 8000).astype(np.int16),
            8000,
        )
    )
    return CallBridgeManager(settings=settings, stt=stt, tts=tts)


@pytest.mark.asyncio
async def test_turn_by_turn_call_flow_and_persistence(tmp_path: Path) -> None:
    manager = _make_test_manager(tmp_path)

    # Start call in background and simulate remote receptionist greeting after 50ms
    async def _simulate_greeting() -> None:
        await asyncio.sleep(0.05)
        call_id = next(iter(manager.sessions.keys()))
        session = manager.get_session(call_id)
        session.status = "connected"
        session.record_turn("remote", "Bonnie View Pro Shop, how can I help you?")

    greeting_task = asyncio.create_task(_simulate_greeting())
    start_res = await manager.start_call("+15552345678", call_context="Check Saturday tee times")
    await greeting_task

    assert start_res["status"] == "connected"
    assert "Bonnie View Pro Shop" in start_res["they_said"]
    call_id = start_res["call_id"]

    # Simulate remote answer when Muse speaks
    async def _simulate_clerk_reply() -> None:
        await asyncio.sleep(0.05)
        session = manager.get_session(call_id)
        # Feed active speech frames followed by silence frames to trigger VAD -> STT
        t = np.linspace(0.0, 0.12, 960, endpoint=False)
        loud_pcm8k = (np.sin(2.0 * np.pi * 400.0 * t) * 14000).astype(np.int16)
        silent_pcm8k = np.zeros(1200, dtype=np.int16)
        for frame_b64 in pcm16_to_mulaw_b64_frames(loud_pcm8k, 8000):
            await session.ingest_mulaw_b64(frame_b64)
        for frame_b64 in pcm16_to_mulaw_b64_frames(silent_pcm8k, 8000):
            await session.ingest_mulaw_b64(frame_b64)

    reply_task = asyncio.create_task(_simulate_clerk_reply())
    turn_res = await manager.speak_and_listen(
        call_id, "Hi! Do you have any tee times before 10 AM this Saturday?"
    )
    await reply_task

    assert turn_res["status"] == "replied"
    assert "9:24 AM" in turn_res["they_said"]

    # End call and verify transcript file persisted on disk
    end_res = await manager.end_call(
        call_id,
        parting_words="Awesome, thank you so much!",
        summary_for_user="Earliest Saturday tee time is 9:24 AM for $45.",
    )
    assert end_res["status"] == "completed"
    assert len(end_res["transcript"]) == 4
    saved_file = Path(end_res["saved_transcript_path"])
    assert saved_file.exists()


@pytest.mark.asyncio
async def test_hold_timeout_guard_and_wait_on_hold(tmp_path: Path) -> None:
    manager = _make_test_manager(tmp_path)

    async def _connect_immediately() -> None:
        await asyncio.sleep(0.03)
        call_id = next(iter(manager.sessions.keys()))
        session = manager.get_session(call_id)
        session.status = "connected"
        session.record_turn("remote", "One moment, let me check the back.")

    t1 = asyncio.create_task(_connect_immediately())
    start_res = await manager.start_call("555-888-9999")
    await t1
    call_id = start_res["call_id"]

    # Clerk goes silent on hold -> speak_and_listen should return on_hold_or_silence instead of crashing
    hold_res = await manager.speak_and_listen(
        call_id, "Sure, take your time!", listen_timeout_seconds=0.15
    )
    assert hold_res["status"] == "on_hold_or_silence"

    # Now clerk returns while Muse is in wait_on_hold
    async def _clerk_returns() -> None:
        await asyncio.sleep(0.05)
        session = manager.get_session(call_id)
        session.record_turn("remote", "Thanks for holding, we have 18 sheets in stock.")

    t2 = asyncio.create_task(_clerk_returns())
    resumed = await manager.wait_on_hold(call_id, timeout_seconds=0.5)
    await t2
    assert resumed["status"] == "replied"
    assert "18 sheets" in resumed["they_said"]
