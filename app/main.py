"""FastAPI + MCP Server entrypoint for Muse Outbound Voice Calling Bridge."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.call_bridge import CallBridgeManager
from app.config import Settings, get_settings
from app.mcp_server import create_mcp_server
from app.telephony import build_media_stream_xml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class StartCallRequest(BaseModel):
    phone_number: str = Field(..., description="Target phone number in E.164 or 10-digit format")
    call_context: str = Field(default="", description="Goal/context for the call")
    opening_line: str = Field(
        default="",
        description="Optional opening line to pre-synthesize while ringing and speak immediately upon pickup",
    )
    timeout_seconds: float = Field(default=45.0, ge=1.0, le=120.0)


class SpeakAndListenRequest(BaseModel):
    what_to_say: str = Field(..., min_length=1, description="Sentence to speak on the call")
    listen_timeout_seconds: float = Field(default=25.0, ge=1.0, le=120.0)
    play_filler_after_reply: bool = Field(default=False)


class HoldRequest(BaseModel):
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)


class DtmfRequest(BaseModel):
    digits: str = Field(..., min_length=1, description="DTMF digits (0-9, *, #, w)")
    listen_after: bool = Field(default=True)
    listen_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)


class EndCallRequest(BaseModel):
    parting_words: str = Field(default="Thank you so much, goodbye!")
    summary_for_user: str = Field(default="")


def create_app(
    settings: Settings | None = None,
    bridge_manager: CallBridgeManager | None = None,
) -> FastAPI:
    """Create the FastAPI application with mounted MCP SSE/HTTP endpoints and Telephony WebSockets."""
    cfg = settings or get_settings()
    manager = bridge_manager or CallBridgeManager(cfg)
    mcp_server = create_mcp_server(manager)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if manager.stt.custom_transcribe_fn is None:
            asyncio.create_task(asyncio.to_thread(manager.stt.warmup))
        if manager.tts._custom_synth_fn is None:
            asyncio.create_task(manager.tts.synthesize_mulaw_frames("Hey there!"))
        yield

    app = FastAPI(
        title="Muse Voice MCP Bridge",
        version="0.2.0",
        description="Self-hosted Outbound Phone Call 'Ears & Mouth' MCP Bridge for Muse",
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.bridge_manager = manager
    app.state.mcp_server = mcp_server

    # Strict Homelab CORS Policy:
    # 1. Public Cloudflare Tunnel subdomain (https://voice-mcp.cooperlabs.app)
    # 2. Local development origins (http://localhost:*)
    # 3. Tailscale mesh IP range (100.*.*.*)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"https://{cfg.public_domain}",
            "https://voice-mcp.cooperlabs.app",
            "https://orbit.cooperlabs.app",
        ],
        allow_origin_regex=r"^https?://(localhost(:\d+)?|127\.0\.0\.1(:\d+)?|100\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?)$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    @app.get("/healthz")
    @app.get("/readyz")
    async def healthcheck() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": cfg.app_name,
            "telephony_provider": cfg.telephony_provider,
            "whisper_model": cfg.whisper_model_size,
            "tts_engine": cfg.tts_engine,
            "active_calls": sum(
                1 for s in manager.sessions.values() if s.status in ("dialing", "connected")
            ),
        }

    # --- Telephony Webhooks (Telnyx TeXML & Twilio TwiML + Status Callbacks) ---

    @app.api_route("/webhooks/telephony/twiml/{call_id}", methods=["GET", "POST"])
    async def telephony_twiml_webhook(call_id: str) -> Response:
        """Returns TeXML/TwiML instructing Telnyx or Twilio to open the bidirectional Media Stream."""
        ws_url = f"{cfg.public_wss_base}/ws/media/{call_id}"
        xml_body = build_media_stream_xml(cfg.telephony_provider, ws_url, call_id)
        return Response(content=xml_body, media_type="application/xml")

    @app.post("/webhooks/telephony/status/{call_id}")
    async def telephony_status_webhook(call_id: str, request: Request) -> dict[str, str]:
        """Receives call lifecycle updates (ringing, answered, completed, busy, no-answer, failed)."""
        content_type = request.headers.get("content-type", "")
        call_status = ""
        if "application/json" in content_type:
            body = await request.json()
            # Telnyx webhook format or JSON callback
            event_payload = body.get("data", {}).get("payload", {})
            call_status = (
                body.get("CallStatus")
                or event_payload.get("state")
                or body.get("data", {}).get("event_type", "")
            )
        else:
            form = await request.form()
            call_status = str(form.get("CallStatus", ""))

        normalized_status = call_status.lower().strip()
        session = manager.sessions.get(call_id)
        if session is not None:
            if normalized_status in (
                "completed",
                "busy",
                "no-answer",
                "failed",
                "canceled",
                "call.hangup",
            ):
                final_state = (
                    "completed"
                    if normalized_status == "call.hangup"
                    else normalized_status
                )
                session.mark_terminated(final_state)
                manager.persist_session(session)
        return {"status": "received", "call_id": call_id, "call_status": normalized_status}

    # --- Bidirectional Media Stream WebSocket ---

    @app.websocket("/ws/media/{call_id}")
    async def media_stream_ws(websocket: WebSocket, call_id: str) -> None:
        await manager.handle_media_websocket(websocket, call_id)

    # --- REST Endpoints (for direct HTTP tool calling or UI/debugging) ---

    @app.get("/api/v1/calls")
    async def list_calls() -> dict[str, Any]:
        return {
            "calls": [
                s.to_dict()
                for s in sorted(
                    manager.sessions.values(), key=lambda x: x.created_at, reverse=True
                )
            ]
        }

    @app.get("/api/v1/calls/{call_id}")
    async def get_call(call_id: str) -> dict[str, Any]:
        try:
            return manager.get_session(call_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/calls/start")
    async def api_start_call(req: StartCallRequest) -> dict[str, Any]:
        return await manager.start_call(
            phone_number=req.phone_number,
            call_context=req.call_context,
            timeout_seconds=req.timeout_seconds,
            opening_line=req.opening_line,
        )

    @app.post("/api/v1/calls/{call_id}/speak")
    async def api_speak_and_listen(
        call_id: str, req: SpeakAndListenRequest
    ) -> dict[str, Any]:
        try:
            return await manager.speak_and_listen(
                call_id=call_id,
                what_to_say=req.what_to_say,
                listen_timeout_seconds=req.listen_timeout_seconds,
                play_filler_after_reply=req.play_filler_after_reply,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/calls/{call_id}/hold")
    async def api_wait_on_hold(call_id: str, req: HoldRequest) -> dict[str, Any]:
        try:
            return await manager.wait_on_hold(
                call_id=call_id, timeout_seconds=req.timeout_seconds
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/calls/{call_id}/dtmf")
    async def api_send_dtmf(call_id: str, req: DtmfRequest) -> dict[str, Any]:
        try:
            return await manager.send_dtmf(
                call_id=call_id,
                digits=req.digits,
                listen_after=req.listen_after,
                listen_timeout_seconds=req.listen_timeout_seconds,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/calls/{call_id}/end")
    async def api_end_call(call_id: str, req: EndCallRequest) -> dict[str, Any]:
        try:
            return await manager.end_call(
                call_id=call_id,
                parting_words=req.parting_words,
                summary_for_user=req.summary_for_user,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Mount MCP SSE application at /mcp so Muse can connect over Tailscale or ClusterIP
    try:
        app.mount("/mcp", mcp_server.sse_app())
    except Exception as exc:
        logger.warning("Could not mount FastMCP SSE app (%s)", exc)

    return app


app = create_app()


if __name__ == "__main__":
    import sys

    if "--stdio" in sys.argv:
        # Run in stdio MCP mode if invoked directly by a local MCP client
        settings = get_settings()
        manager = CallBridgeManager(settings)
        mcp = create_mcp_server(manager)
        mcp.run(transport="stdio")
    else:
        import uvicorn

        settings = get_settings()
        uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
