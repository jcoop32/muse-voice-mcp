"""End-to-end FastAPI & WebSocket tests for Muse Voice MCP Bridge."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.audio_codec import pcm16_to_mulaw_b64_frames
from app.call_bridge import CallBridgeManager
from app.config import Settings
from app.main import create_app
from app.sip_client import (
    compute_sip_digest_header,
    extract_uas_route_set,
    parse_sdp_rtp_target,
)
from app.stt import LocalWhisperSTT, is_hallucinated_transcript
from app.tts import LocalTTSEngine


def test_health_and_twiml_webhooks(tmp_path: Path) -> None:
    settings = Settings(
        telephony_provider="telnyx",
        public_domain="voice-mcp.cooperlabs.app",
        data_dir=tmp_path,
    )
    stt = LocalWhisperSTT(custom_transcribe_fn=lambda _: "Hello from test")
    tts = LocalTTSEngine(
        custom_synth_fn=lambda _: (np.zeros(800, dtype=np.int16), 8000)
    )
    manager = CallBridgeManager(settings=settings, stt=stt, tts=tts)
    app = create_app(settings=settings, bridge_manager=manager)
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["telephony_provider"] == "telnyx"

        twiml = client.post("/webhooks/telephony/twiml/call_abc123")
        assert twiml.status_code == 200
        assert "wss://voice-mcp.cooperlabs.app/ws/media/call_abc123" in twiml.text
        assert 'bidirectionalMode="rtp"' in twiml.text


def test_websocket_media_stream_and_barge_in(tmp_path: Path) -> None:
    settings = Settings(
        telephony_provider="telnyx",
        data_dir=tmp_path,
        vad_silence_duration_ms=80,
        vad_min_speech_duration_ms=40,
    )
    stt = LocalWhisperSTT(custom_transcribe_fn=lambda _: "We close at 8 PM tonight.")
    tts = LocalTTSEngine(
        custom_synth_fn=lambda _: (
            (np.sin(np.linspace(0, 10, 480)) * 9000).astype(np.int16),
            8000,
        )
    )
    manager = CallBridgeManager(settings=settings, stt=stt, tts=tts)
    app = create_app(settings=settings, bridge_manager=manager)
    client = TestClient(app)

    # Pre-register a session
    call_id = "call_ws_test"
    from app.call_bridge import CallSession

    session = CallSession(
        call_id=call_id,
        phone_number="+15551234567",
        call_context="Ask closing time",
        settings=settings,
        stt=stt,
        tts=tts,
    )
    manager.sessions[call_id] = session

    with client.websocket_connect(f"/ws/media/{call_id}") as ws:
        ws.send_json({"event": "start", "start": {"streamSid": "MZ_STREAM_1"}})

        # Send active speech frames followed by silence frames over WebSocket
        t = np.linspace(0.0, 0.1, 800, endpoint=False)
        loud_pcm8k = (np.sin(2.0 * np.pi * 440.0 * t) * 15000).astype(np.int16)
        silent_pcm8k = np.zeros(1200, dtype=np.int16)

        for frame_b64 in pcm16_to_mulaw_b64_frames(loud_pcm8k, 8000):
            ws.send_json({"event": "media", "media": {"payload": frame_b64}})
        for frame_b64 in pcm16_to_mulaw_b64_frames(silent_pcm8k, 8000):
            ws.send_json({"event": "media", "media": {"payload": frame_b64}})

        ws.send_json({"event": "stop"})

    assert session.latest_remote_utterance == "We close at 8 PM tonight."
    call_resp = client.get(f"/api/v1/calls/{call_id}")
    assert call_resp.status_code == 200
    assert call_resp.json()["latest_remote_utterance"] == "We close at 8 PM tonight."


async def test_mcp_tools_registered_and_callable(tmp_path: Path) -> None:
    settings = Settings(telephony_provider="telnyx", data_dir=tmp_path)
    stt = LocalWhisperSTT(custom_transcribe_fn=lambda _: "Test greeting")
    tts = LocalTTSEngine(
        custom_synth_fn=lambda _: (np.zeros(800, dtype=np.int16), 8000)
    )
    manager = CallBridgeManager(settings=settings, stt=stt, tts=tts)
    app = create_app(settings=settings, bridge_manager=manager)
    mcp_server = app.state.mcp_server

    tools = await mcp_server.list_tools()
    tool_names = {t.name for t in tools}
    assert {
        "start_phone_call",
        "speak_and_listen",
        "wait_on_hold",
        "send_dtmf",
        "end_phone_call",
    }.issubset(tool_names)


def test_sip_digest_auth_and_sdp_parser() -> None:
    hdr = compute_sip_digest_header(
        header_name="Proxy-Authorization",
        challenge_value='realm="sip.telnyx.com", nonce="12345abcdef", qop="auth"',
        username="tz8cdz6s4w8x",
        password="secret",
        method="INVITE",
        uri="sip:+15551234567@sip.telnyx.com",
    )
    assert hdr.startswith("Proxy-Authorization: Digest ")
    assert 'username="tz8cdz6s4w8x"' in hdr
    assert 'realm="sip.telnyx.com"' in hdr
    assert "qop=auth" in hdr

    sdp = (
        "v=0\r\n"
        "o=Telnyx 1 1 IN IP4 64.16.250.10\r\n"
        "c=IN IP4 64.16.250.10\r\n"
        "m=audio 24680 RTP/AVP 0 101\r\n"
    )
    assert parse_sdp_rtp_target(sdp) == ("64.16.250.10", 24680)

    sip_200_headers = (
        "SIP/2.0 200 OK\r\n"
        "Record-Route: <sip:proxy2.telnyx.com;lr>, <sip:proxy1.telnyx.com;lr>\r\n"
    )
    routes = extract_uas_route_set(sip_200_headers)
    assert routes == [
        "<sip:proxy1.telnyx.com;lr>",
        "<sip:proxy2.telnyx.com;lr>",
    ]


def test_ringback_hallucination_filter() -> None:
    assert is_hallucinated_transcript("ee ee ee ee ee ee") is True
    assert is_hallucinated_transcript("Thank you for watching!") is True
    assert is_hallucinated_transcript("Hello?") is False
    assert is_hallucinated_transcript("Going good, how are you?") is False



