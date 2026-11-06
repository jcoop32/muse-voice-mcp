"""MCP Server exposing Muse's outbound phone call tools ('Ears & Mouth' Turn-by-Turn Bridge)."""

from __future__ import annotations

from typing import Any

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]

from app.call_bridge import CallBridgeManager


def create_mcp_server(bridge_manager: CallBridgeManager) -> FastMCP:
    """Create and configure the FastMCP server bound to a CallBridgeManager instance."""
    mcp = FastMCP(
        "MuseVoiceBridge",
        instructions=(
            "Use these tools to place outbound phone calls on behalf of the user, "
            "speak to businesses/receptionists, navigate keypad menus, gather answers, "
            "and hang up politely. Keep every spoken sentence concise (under 20 words) "
            "and ask one clear question at a time."
        ),
    )

    @mcp.tool()
    async def start_phone_call(
        phone_number: str,
        call_context: str = "",
        timeout_seconds: float = 45.0,
    ) -> dict[str, Any]:
        """Dial an external phone number (E.164 or 10-digit US), wait for them to answer and greet you, and return what they said."""
        return await bridge_manager.start_call(
            phone_number=phone_number,
            call_context=call_context,
            timeout_seconds=timeout_seconds,
        )

    @mcp.tool()
    async def speak_and_listen(
        call_id: str,
        what_to_say: str,
        listen_timeout_seconds: float = 25.0,
        play_filler_after_reply: bool = True,
    ) -> dict[str, Any]:
        """Speak your message on the active phone call, wait for the other person to finish replying, and return their transcribed response."""
        return await bridge_manager.speak_and_listen(
            call_id=call_id,
            what_to_say=what_to_say,
            listen_timeout_seconds=listen_timeout_seconds,
            play_filler_after_reply=play_filler_after_reply,
        )

    @mcp.tool()
    async def wait_on_hold(
        call_id: str,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Continue listening silently when the person on the phone asks you to hold or checks on something."""
        return await bridge_manager.wait_on_hold(
            call_id=call_id,
            timeout_seconds=timeout_seconds,
        )

    @mcp.tool()
    async def send_dtmf(
        call_id: str,
        digits: str,
        listen_after: bool = True,
        listen_timeout_seconds: float = 20.0,
    ) -> dict[str, Any]:
        """Press phone keypad digits (0-9, *, #, w for 0.5s pause) to navigate automated IVR phone trees."""
        return await bridge_manager.send_dtmf(
            call_id=call_id,
            digits=digits,
            listen_after=listen_after,
            listen_timeout_seconds=listen_timeout_seconds,
        )

    @mcp.tool()
    async def end_phone_call(
        call_id: str,
        parting_words: str = "Awesome, thank you so much! Have a great day.",
        summary_for_user: str = "",
    ) -> dict[str, Any]:
        """Say goodbye on the phone call, hang up, save the transcript, and return the full call record."""
        return await bridge_manager.end_call(
            call_id=call_id,
            parting_words=parting_words,
            summary_for_user=summary_for_user,
        )

    return mcp
