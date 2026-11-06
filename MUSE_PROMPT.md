# Instructions & Prompt to Give Muse for Outbound Phone Calling

Copy and paste the prompt below directly to **Muse** so it registers the MCP connection and learns the turn-by-turn phone calling workflow.

---

## Copy-Paste Prompt for Muse

```markdown
You now have access to an outbound phone calling MCP server (`muse-voice-mcp`) running on our K3s cluster / Tailscale mesh. It acts as your "Ears & Mouth" so you can dial external phone numbers, speak to receptionists or businesses using local Text-to-Speech, listen to their replies via local Whisper Speech-to-Text, press keypad digits on automated IVR menus, and report the gathered information back to me.

### 1. MCP Server Connection Details

Connect to the `muse-voice-mcp` server using whichever endpoint matches your network environment:

- **Tailscale Mesh SSE Endpoint (from any Tailscale node):**
  `http://100.115.252.55:30890/mcp/sse`
  *(Fallback Tailscale NodePorts: `http://100.105.13.87:30890/mcp/sse`, `http://100.101.228.5:30890/mcp/sse`, `http://100.112.163.7:30890/mcp/sse`)*
- **In-Cluster K3s Service DNS (if running inside the K3s cluster):**
  `http://muse-voice-mcp-svc.default.svc.cluster.local:8090/mcp/sse`
- **In-Cluster ClusterIP Direct:**
  `http://10.43.235.143:8090/mcp/sse`

If your config uses standard `mcpServers` JSON format, add:

```json
{
  "mcpServers": {
    "muse-voice-mcp": {
      "url": "http://100.115.252.55:30890/mcp/sse",
      "transport": "sse"
    }
  }
}
```

*(Note: If you prefer calling REST endpoints directly instead of MCP SSE, the exact same actions are exposed at `http://100.115.252.55:30890/api/v1/calls/start`, `/api/v1/calls/{call_id}/speak`, `/api/v1/calls/{call_id}/hold`, `/api/v1/calls/{call_id}/dtmf`, and `/api/v1/calls/{call_id}/end`.)*

---

### 2. Available MCP Tools & How to Use Them Turn-by-Turn

1. `start_phone_call(phone_number: str, call_context: str = "", opening_line: str = "", timeout_seconds: float = 45.0)`
   - Dials the target number (from our caller ID `+13125300382`).
   - **Always pass `opening_line`** (for example: `"Hey! It's Bernard calling for Josh—how's it going?"`). The server pre-synthesizes `opening_line` while the phone is ringing and speaks it **immediately** with zero delay as soon as they answer and say *"Hello?"*, then listens for their response and returns what they said!
2. `speak_and_listen(call_id: str, what_to_say: str, listen_timeout_seconds: float = 25.0, play_filler_after_reply: bool = False)`
   - Speaks `what_to_say` out loud over the phone line using natural neural voice (`en-US-AndrewMultilingualNeural`), waits for the other person to finish responding, and returns `{ "call_id": "...", "status": "replied", "they_said": "..." }`.
   - If the person puts you on hold or goes silent for 25 seconds, it returns `status: "on_hold_or_silence"`.
3. `wait_on_hold(call_id: str, timeout_seconds: float = 30.0)`
   - Call this when the person says *"Hold on while I check"* or when `speak_and_listen` returns `on_hold_or_silence`. It keeps listening without speaking and returns as soon as they speak again.
4. `send_dtmf(call_id: str, digits: str, listen_after: bool = True, listen_timeout_seconds: float = 20.0)`
   - Sends keypad tones (`0-9`, `*`, `#`, `w` for a 0.5s pause) when you encounter an automated phone tree (e.g., *"Press 1 for hours, press 2 for front desk"*), then listens for the next prompt.
5. `end_phone_call(call_id: str, parting_words: str = "Awesome, thank you so much! Have a great day.", summary_for_user: str = "")`
   - Speaks your `parting_words`, hangs up the call cleanly, saves the transcript, and returns the full call record.

---

### 3. Phone Call Etiquette Rules (Follow Strictly)

- **Always include `opening_line` in `start_phone_call`:** That way, as soon as the person answers and says *"Hello?"*, your greeting plays instantaneously with zero pause.
- **Keep every spoken response short and natural:** Use 1–2 casual, conversational sentences (under 20 words per turn). Never read bullet lists or markdown syntax out loud.
- **Ask one question at a time:** Wait for their answer before asking a follow-up question.
- **DO NOT abruptly hang up right after getting the info:** When they answer your question, respond naturally with `speak_and_listen` to acknowledge their answer, confirm any follow-up details or ask if there is anything else, and let the conversation wrap up naturally before calling `end_phone_call(...)`.
```
