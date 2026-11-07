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
            "speak to people or businesses, navigate keypad menus, gather answers, "
            "and wrap up conversations naturally. Keep every spoken sentence concise "
            "and conversational (under 20 words), pass `opening_line` in `start_phone_call` "
            "for instant zero-delay greeting on pickup, and NEVER abruptly hang up immediately "
            "after receiving an answer—acknowledge their reply and wrap up naturally before calling `end_phone_call`."
        ),
    )

    @mcp.tool()
    async def start_phone_call(
        phone_number: str,
        call_context: str = "",
        opening_line: str = "",
        timeout_seconds: float = 45.0,
        auto_converse: bool = False,
    ) -> dict[str, Any]:
        """Dial an external phone number (E.164 or 10-digit US), navigate call screening automatically, enforce caller-ID trust (+12242458826 = trusted owner, all others = untrusted external sandbox), optionally speak `opening_line` immediately, and return what they said."""
        return await bridge_manager.start_call(
            phone_number=phone_number,
            call_context=call_context,
            timeout_seconds=timeout_seconds,
            opening_line=opening_line,
            auto_converse=auto_converse,
        )

    @mcp.tool()
    async def converse_on_phone(
        call_id: str,
        max_turns: int = 12,
        listen_timeout_seconds: float = 25.0,
    ) -> dict[str, Any]:
        """Have a fast, natural, multi-turn conversation on the active phone call using the in-pod VoiceBrain (open-ended companion chat for +12242458826; air-gapped zero-tool sandbox for external numbers)."""
        return await bridge_manager.converse_on_phone(
            call_id=call_id,
            max_turns=max_turns,
            listen_timeout_seconds=listen_timeout_seconds,
        )

    @mcp.tool()
    async def speak_and_listen(
        call_id: str,
        what_to_say: str,
        listen_timeout_seconds: float = 25.0,
        play_filler_after_reply: bool = False,
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
