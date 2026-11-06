"""Pure-Python asyncio Outbound SIP + Symmetric RTP client for Telnyx (Zero public webhooks/Cloudflare)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import random
import re
import socket
import struct
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.audio_codec import MULAW_FRAME_BYTES_20MS, MULAW_SILENCE_BYTE

if TYPE_CHECKING:
    from app.call_bridge import CallSession

logger = logging.getLogger(__name__)

DTMF_CHAR_TO_CODE: dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "*": 10,
    "#": 11,
}


def discover_stun_endpoint(
    sock: socket.socket,
    stun_host: str = "stun.l.google.com",
    stun_port: int = 19302,
    timeout: float = 2.0,
) -> tuple[str, int]:
    """Perform an RFC 5389 STUN Binding Request on the RTP UDP socket to find our public NAT IP:port."""
    local_addr = sock.getsockname()
    local_port = int(local_addr[1])
    try:
        tx_id = os.urandom(12)
        magic_cookie = 0x2112A442
        req = struct.pack("!HHI12s", 0x0001, 0, magic_cookie, tx_id)
        sock.settimeout(timeout)
        sock.sendto(req, (stun_host, stun_port))
        data, _ = sock.recvfrom(1024)
        if len(data) >= 20:
            offset = 20
            while offset + 4 <= len(data):
                attr_type, attr_len = struct.unpack("!HH", data[offset : offset + 4])
                val = data[offset + 4 : offset + 4 + attr_len]
                # XOR-MAPPED-ADDRESS (0x0020)
                if attr_type == 0x0020 and len(val) >= 8 and val[1] == 0x01:
                    xport = struct.unpack("!H", val[2:4])[0] ^ (magic_cookie >> 16)
                    xip = struct.unpack("!I", val[4:8])[0] ^ magic_cookie
                    ip_str = socket.inet_ntoa(struct.pack("!I", xip))
                    return ip_str, int(xport)
                # MAPPED-ADDRESS (0x0001)
                if attr_type == 0x0001 and len(val) >= 8 and val[1] == 0x01:
                    port = struct.unpack("!H", val[2:4])[0]
                    ip_str = socket.inet_ntoa(val[4:8])
                    return ip_str, int(port)
                offset += 4 + ((attr_len + 3) & ~3)
    except Exception as exc:
        logger.debug("STUN discovery fallback to local address (%s)", exc)
    finally:
        sock.setBlocking(False) if hasattr(sock, "setBlocking") else sock.setblocking(False)

    return "0.0.0.0", local_port


def compute_sip_digest_header(
    header_name: str,
    challenge_value: str,
    username: str,
    password: str,
    method: str,
    uri: str,
    nc: int = 1,
) -> str:
    """Compute an RFC 2617 / RFC 3261 Digest Authorization / Proxy-Authorization header value."""
    params: dict[str, str] = {}
    for match in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|([^\s,]+))', challenge_value):
        key = match.group(1).lower()
        val = match.group(2) if match.group(2) is not None else match.group(3)
        params[key] = val

    realm = params.get("realm", "sip.telnyx.com")
    nonce = params.get("nonce", "")
    opaque = params.get("opaque")
    qop_raw = params.get("qop", "")
    qop = "auth" if "auth" in [q.strip() for q in qop_raw.split(",")] else ""

    ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode("utf-8")).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode("utf-8")).hexdigest()
    cnonce = uuid.uuid4().hex[:16]
    nc_hex = f"{nc:08x}"

    if qop == "auth":
        resp_hash = hashlib.md5(
            f"{ha1}:{nonce}:{nc_hex}:{cnonce}:{qop}:{ha2}".encode("utf-8")
        ).hexdigest()
    else:
        resp_hash = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode("utf-8")).hexdigest()

    parts = [
        f'Digest username="{username}"',
        f'realm="{realm}"',
        f'nonce="{nonce}"',
        f'uri="{uri}"',
        f'response="{resp_hash}"',
        "algorithm=MD5",
    ]
    if qop:
        parts.extend([f"qop={qop}", f"nc={nc_hex}", f'cnonce="{cnonce}"'])
    if opaque:
        parts.append(f'opaque="{opaque}"')

    return f"{header_name}: " + ", ".join(parts)


def parse_sdp_rtp_target(sdp_body: str) -> tuple[str, int] | None:
    """Extract remote RTP IPv4 address and UDP port from an SDP answer."""
    ip_match = re.search(r"^c=IN IP4\s+([^\r\n\s]+)", sdp_body, re.MULTILINE)
    port_match = re.search(r"^m=audio\s+(\d+)\s+RTP/AVP", sdp_body, re.MULTILINE)
    if not ip_match or not port_match:
        return None
    return ip_match.group(1).strip(), int(port_match.group(1))


def extract_uas_route_set(header_block: str) -> list[str]:
    """Extract Record-Route headers from a SIP 200 OK and reverse them per RFC 3261 Section 12.1.2."""
    routes: list[str] = []
    for line in header_block.splitlines():
        if line.lower().startswith("record-route:"):
            raw_val = line.split(":", 1)[1].strip()
            # Split multiple comma-separated <sip:...> route entries if present on one line
            entries = re.findall(r"(?:[^,<]*<[^>]+>[^,]*)", raw_val)
            if entries:
                routes.extend(e.strip() for e in entries if e.strip())
            elif raw_val:
                routes.append(raw_val)
    routes.reverse()
    return routes


@dataclass
class SipCallHandle:
    """Represents a live Outbound SIP + RTP call session."""

    call_id: str
    sip_call_id: str
    session: CallSession
    sip_host: str
    sip_port: int
    sip_username: str
    sip_password: str
    caller_number: str
    target_number: str

    def __post_init__(self) -> None:
        self._stop_event = asyncio.Event()
        self._dtmf_queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._connected_monotonic: float = 0.0

    def start_background(self) -> None:
        self._task = asyncio.create_task(self._run_sip_and_rtp())

    async def send_dtmf(self, digits: str) -> bool:
        for ch in digits:
            if ch in DTMF_CHAR_TO_CODE or ch.lower() == "w":
                await self._dtmf_queue.put(ch)
        return True

    async def hangup(self) -> None:
        self._stop_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except Exception:
                self._task.cancel()

    async def _run_sip_and_rtp(self) -> None:
        # 1. Bind local UDP socket for RTP audio and discover public NAT IP:port via STUN
        rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rtp_sock.bind(("0.0.0.0", 0))
        pub_ip, pub_port = await asyncio.to_thread(discover_stun_endpoint, rtp_sock)
        if pub_ip == "0.0.0.0":
            pub_ip = "127.0.0.1"

        # 2. Open UDP socket to sip.telnyx.com:5060 for SIP signaling
        sip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sip_sock.bind(("0.0.0.0", 0))
        sip_sock.setblocking(False)
        sip_server_ip = await asyncio.to_thread(socket.gethostbyname, self.sip_host)
        sip_local_port = sip_sock.getsockname()[1]

        from_tag = uuid.uuid4().hex[:10]
        to_tag = ""
        contact_uri = f"<sip:{self.sip_username}@{pub_ip}:{sip_local_port}>"
        request_uri = f"sip:{self.target_number}@{self.sip_host}"
        from_uri = f"<sip:{self.caller_number}@{self.sip_host}>;tag={from_tag}"
        to_uri = f"<sip:{self.target_number}@{self.sip_host}>"

        sdp_offer = (
            "v=0\r\n"
            f"o=muse {random.randint(1000, 9999)} 1 IN IP4 {pub_ip}\r\n"
            "s=MuseVoiceMCP\r\n"
            f"c=IN IP4 {pub_ip}\r\n"
            "t=0 0\r\n"
            f"m=audio {pub_port} RTP/AVP 0 101\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
            "a=rtpmap:101 telephone-event/8000\r\n"
            "a=fmtp:101 0-15\r\n"
            "a=ptime:20\r\n"
            "a=direction:active\r\n"
            "a=sendrecv\r\n"
        )

        def build_invite(cseq: int, branch: str, auth_line: str = "") -> bytes:
            headers = [
                f"INVITE {request_uri} SIP/2.0",
                f"Via: SIP/2.0/UDP {pub_ip}:{sip_local_port};rport;branch={branch}",
                "Max-Forwards: 70",
                f"From: {from_uri}",
                f"To: {to_uri}",
                f"Call-ID: {self.sip_call_id}",
                f"CSeq: {cseq} INVITE",
                f"Contact: {contact_uri}",
                f"P-Asserted-Identity: <sip:{self.caller_number}@{self.sip_host}>",
                "User-Agent: MuseVoiceMCP/0.2",
                "Allow: INVITE, ACK, CANCEL, BYE, OPTIONS",
                "Content-Type: application/sdp",
            ]
            if auth_line:
                headers.append(auth_line)
            headers.append(f"Content-Length: {len(sdp_offer.encode('utf-8'))}")
            return ("\r\n".join(headers) + "\r\n\r\n" + sdp_offer).encode("utf-8")

        def build_ack(
            cseq: int,
            branch: str,
            current_to: str,
            target_contact: str,
            route_set: list[str] | None = None,
        ) -> bytes:
            headers = [
                f"ACK {target_contact} SIP/2.0",
                f"Via: SIP/2.0/UDP {pub_ip}:{sip_local_port};rport;branch={branch}",
                "Max-Forwards: 70",
            ]
            if route_set:
                for r in route_set:
                    headers.append(f"Route: {r}")
            headers.extend(
                [
                    f"From: {from_uri}",
                    f"To: {current_to}",
                    f"Call-ID: {self.sip_call_id}",
                    f"CSeq: {cseq} ACK",
                    f"Contact: {contact_uri}",
                    "Content-Length: 0",
                ]
            )
            return ("\r\n".join(headers) + "\r\n\r\n").encode("utf-8")

        def build_bye(
            cseq: int,
            branch: str,
            current_to: str,
            target_contact: str,
            route_set: list[str] | None = None,
            auth_line: str = "",
        ) -> bytes:
            headers = [
                f"BYE {target_contact} SIP/2.0",
                f"Via: SIP/2.0/UDP {pub_ip}:{sip_local_port};rport;branch={branch}",
                "Max-Forwards: 70",
            ]
            if route_set:
                for r in route_set:
                    headers.append(f"Route: {r}")
            headers.extend(
                [
                    f"From: {from_uri}",
                    f"To: {current_to}",
                    f"Call-ID: {self.sip_call_id}",
                    f"CSeq: {cseq} BYE",
                    f"Contact: {contact_uri}",
                ]
            )
            if auth_line:
                headers.append(auth_line)
            headers.append("Content-Length: 0")
            return ("\r\n".join(headers) + "\r\n\r\n").encode("utf-8")

        loop = asyncio.get_running_loop()
        cseq = 1
        branch1 = f"z9hG4bK{uuid.uuid4().hex[:12]}"
        remote_rtp_addr: tuple[str, int] | None = None
        remote_contact_uri = request_uri
        dialog_route_set: list[str] = []
        last_auth_header = ""
        rtp_task: asyncio.Task[None] | None = None

        try:
            sip_sock.sendto(build_invite(cseq, branch1), (sip_server_ip, self.sip_port))
            logger.info("[call_id=%s] Sent SIP INVITE to %s (%s)", self.call_id, request_uri, sip_server_ip)

            while not self._stop_event.is_set():
                try:
                    raw_sip = await asyncio.wait_for(loop.sock_recv(sip_sock, 4096), timeout=0.25)
                except asyncio.TimeoutError:
                    continue

                text = raw_sip.decode("utf-8", errors="ignore")
                first_line = text.splitlines()[0] if text else ""
                header_block, _, body = text.partition("\r\n\r\n")

                # Track To tag if present
                to_match = re.search(r"^(?:To|t):\s*(.+)$", header_block, re.MULTILINE | re.IGNORECASE)
                current_to_header = to_match.group(1).strip() if to_match else to_uri
                if ";tag=" in current_to_header:
                    to_tag = current_to_header

                # Track Contact URI from 200 OK
                contact_match = re.search(r"^(?:Contact|m):\s*<([^>]+)>", header_block, re.MULTILINE | re.IGNORECASE)
                if contact_match:
                    remote_contact_uri = contact_match.group(1).strip()

                # Extract CSeq number & method
                cseq_m = re.search(r"^CSeq:\s*(\d+)\s+(\w+)", header_block, re.MULTILINE | re.IGNORECASE)
                resp_cseq_num = int(cseq_m.group(1)) if cseq_m else cseq
                resp_cseq_method = cseq_m.group(2).upper() if cseq_m else "INVITE"

                # Handle incoming OPTIONS / keepalive from SIP proxy
                if first_line.startswith("OPTIONS "):
                    via_m = re.search(r"^(?:Via|v):\s*(.+)$", header_block, re.MULTILINE | re.IGNORECASE)
                    cseq_raw = re.search(r"^CSeq:\s*(.+)$", header_block, re.MULTILINE | re.IGNORECASE)
                    ok_opt = (
                        "SIP/2.0 200 OK\r\n"
                        f"Via: {via_m.group(1).strip() if via_m else ''}\r\n"
                        f"From: {current_to_header}\r\n"
                        f"To: {from_uri}\r\n"
                        f"Call-ID: {self.sip_call_id}\r\n"
                        f"CSeq: {cseq_raw.group(1).strip() if cseq_raw else '1 OPTIONS'}\r\n"
                        "Content-Length: 0\r\n\r\n"
                    ).encode("utf-8")
                    sip_sock.sendto(ok_opt, (sip_server_ip, self.sip_port))
                    continue

                # Handle remote BYE
                if first_line.startswith("BYE "):
                    via_m = re.search(r"^(?:Via|v):\s*(.+)$", header_block, re.MULTILINE | re.IGNORECASE)
                    cseq_raw = re.search(r"^CSeq:\s*(.+)$", header_block, re.MULTILINE | re.IGNORECASE)
                    ok_reply = (
                        "SIP/2.0 200 OK\r\n"
                        f"Via: {via_m.group(1).strip() if via_m else ''}\r\n"
                        f"From: {current_to_header}\r\n"
                        f"To: {from_uri}\r\n"
                        f"Call-ID: {self.sip_call_id}\r\n"
                        f"CSeq: {cseq_raw.group(1).strip() if cseq_raw else '1 BYE'}\r\n"
                        "Content-Length: 0\r\n\r\n"
                    ).encode("utf-8")
                    sip_sock.sendto(ok_reply, (sip_server_ip, self.sip_port))
                    self.session.mark_terminated("completed")
                    break

                status_m = re.match(r"^SIP/2\.0\s+(\d{3})\s+(.*)$", first_line)
                if not status_m:
                    continue
                code = int(status_m.group(1))

                if code in (401, 407) and cseq == 1 and resp_cseq_method == "INVITE":
                    sip_sock.sendto(
                        build_ack(cseq, branch1, current_to_header, request_uri),
                        (sip_server_ip, self.sip_port),
                    )
                    auth_hdr_name = "Proxy-Authorization" if code == 407 else "Authorization"
                    chal_hdr_name = "Proxy-Authenticate" if code == 407 else "WWW-Authenticate"
                    chal_m = re.search(
                        rf"^{chal_hdr_name}:\s*Digest\s+(.+)$",
                        header_block,
                        re.MULTILINE | re.IGNORECASE,
                    )
                    if chal_m:
                        cseq = 2
                        branch2 = f"z9hG4bK{uuid.uuid4().hex[:12]}"
                        last_auth_header = compute_sip_digest_header(
                            header_name=auth_hdr_name,
                            challenge_value=chal_m.group(1).strip(),
                            username=self.sip_username,
                            password=self.sip_password,
                            method="INVITE",
                            uri=request_uri,
                        )
                        sip_sock.sendto(
                            build_invite(cseq, branch2, last_auth_header),
                            (sip_server_ip, self.sip_port),
                        )
                elif code in (180, 183) and resp_cseq_method == "INVITE":
                    # Start RTP NAT hole-punching during ringing, but DO NOT transcribe ringback audio
                    target = parse_sdp_rtp_target(body)
                    if target and rtp_task is None:
                        remote_rtp_addr = target
                        rtp_task = asyncio.create_task(
                            self._run_rtp_loop(rtp_sock, remote_rtp_addr)
                        )
                elif code == 200 and resp_cseq_method == "INVITE":
                    routes = extract_uas_route_set(header_block)
                    if routes:
                        dialog_route_set = routes
                    # Always ACK 200 OK (including any UDP retransmissions) with full Route set
                    ack_pkt = build_ack(
                        resp_cseq_num,
                        f"z9hG4bK{uuid.uuid4().hex[:12]}",
                        to_tag or current_to_header,
                        remote_contact_uri,
                        route_set=dialog_route_set,
                    )
                    sip_sock.sendto(ack_pkt, (sip_server_ip, self.sip_port))

                    if self.session.status != "connected":
                        self._connected_monotonic = time.monotonic()
                        self.session.utterance_tracker.reset()
                        self.session.status = "connected"
                        self.session.ws_connected.set()
                        logger.info("[call_id=%s] SIP 200 OK answered; call connected!", self.call_id)

                    target = parse_sdp_rtp_target(body)
                    if target:
                        remote_rtp_addr = target
                    if remote_rtp_addr and rtp_task is None:
                        rtp_task = asyncio.create_task(
                            self._run_rtp_loop(rtp_sock, remote_rtp_addr)
                        )
                elif code >= 400 and resp_cseq_method == "INVITE":
                    sip_sock.sendto(
                        build_ack(resp_cseq_num, branch1, current_to_header, request_uri),
                        (sip_server_ip, self.sip_port),
                    )
                    final_state = "busy" if code == 486 else "failed"
                    logger.warning(
                        "[call_id=%s] SIP error %d %s",
                        self.call_id,
                        code,
                        status_m.group(2),
                    )
                    self.session.mark_terminated(final_state)
                    break

            # Send BYE if stopped locally
            if self.session.status == "connected" and to_tag:
                bye_pkt = build_bye(
                    cseq + 1,
                    f"z9hG4bK{uuid.uuid4().hex[:12]}",
                    to_tag,
                    remote_contact_uri,
                    route_set=dialog_route_set,
                    auth_line=last_auth_header,
                )
                sip_sock.sendto(bye_pkt, (sip_server_ip, self.sip_port))
        finally:
            if rtp_task and not rtp_task.done():
                rtp_task.cancel()
            rtp_sock.close()
            sip_sock.close()

    async def _run_rtp_loop(
        self, rtp_sock: socket.socket, remote_addr: tuple[str, int]
    ) -> None:
        """Bidirectional 20ms G.711 PCMU RTP loop with Symmetric NAT hole-punching and RFC 2833 DTMF."""
        loop = asyncio.get_running_loop()
        seq = random.randint(0, 30000)
        ts = random.randint(0, 100000)
        ssrc = random.randint(1, 0xFFFFFFFF)

        current_item = None
        current_frames: list[str] = []

        async def _recv_loop() -> None:
            while not self._stop_event.is_set() and not self.session.call_terminated.is_set():
                try:
                    pkt = await asyncio.wait_for(loop.sock_recv(rtp_sock, 2048), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
                if len(pkt) <= 12:
                    continue
                # Ignore early-media ringback tones (180/183) before SIP 200 OK + 300ms pickup settle
                if self.session.status != "connected":
                    continue
                if self._connected_monotonic > 0 and (time.monotonic() - self._connected_monotonic) < 0.30:
                    continue
                pt = pkt[1] & 0x7F
                # Payload Type 0 = PCMU 8000Hz
                if pt == 0:
                    payload = pkt[12:]
                    b64 = base64.b64encode(payload).decode("ascii")
                    await self.session.ingest_mulaw_b64(b64)

        recv_task = asyncio.create_task(_recv_loop())
        silence_payload = bytes([MULAW_SILENCE_BYTE]) * MULAW_FRAME_BYTES_20MS

        try:
            next_tick = time.monotonic()
            while not self._stop_event.is_set() and not self.session.call_terminated.is_set():
                # 1. Check if DTMF digit is queued
                if not self._dtmf_queue.empty():
                    digit = self._dtmf_queue.get_nowait()
                    if digit.lower() == "w":
                        await asyncio.sleep(0.5)
                        next_tick = time.monotonic()
                        continue
                    code = DTMF_CHAR_TO_CODE.get(digit, 0)
                    event_ts = ts
                    for step_idx in range(6):
                        is_end = step_idx >= 3
                        end_bit = 0x80 if is_end else 0x00
                        duration = min(480, (step_idx + 1) * 160)
                        marker_pt = (0x80 | 101) if step_idx == 0 else 101
                        hdr = struct.pack("!BBHII", 0x80, marker_pt, seq & 0xFFFF, event_ts & 0xFFFFFFFF, ssrc)
                        dtmf_body = struct.pack("!BBH", code, end_bit | 10, duration)
                        rtp_sock.sendto(hdr + dtmf_body, remote_addr)
                        seq = (seq + 1) & 0xFFFF
                        await asyncio.sleep(0.02)
                    ts = (ts + 640) & 0xFFFFFFFF
                    next_tick = time.monotonic()
                    continue

                # 2. Pull next TTS/filler frame or send silence to keep Symmetric RTP NAT open
                if not current_frames:
                    if current_item is not None:
                        self.session.is_playing_tts = False
                        current_item.done_event.set()
                        current_item = None
                    if not self.session.outbound_queue.empty():
                        nxt = self.session.outbound_queue.get_nowait()
                        if nxt is not None:
                            current_item = nxt
                            current_frames = list(nxt.frames_b64)
                            self.session.is_playing_tts = True
                            self.session.abort_current_playback = False

                if self.session.abort_current_playback:
                    current_frames.clear()
                    if current_item is not None:
                        self.session.is_playing_tts = False
                        current_item.done_event.set()
                        current_item = None

                if current_frames:
                    frame_b64 = current_frames.pop(0)
                    payload = base64.b64decode(frame_b64)
                else:
                    payload = silence_payload

                hdr = struct.pack("!BBHII", 0x80, 0, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)
                try:
                    rtp_sock.sendto(hdr + payload, remote_addr)
                except Exception:
                    pass

                seq = (seq + 1) & 0xFFFF
                ts = (ts + MULAW_FRAME_BYTES_20MS) & 0xFFFFFFFF
                next_tick += 0.02
                sleep_dur = next_tick - time.monotonic()
                if sleep_dur > 0:
                    await asyncio.sleep(sleep_dur)
                else:
                    next_tick = time.monotonic()
        finally:
            if current_item is not None:
                self.session.is_playing_tts = False
                current_item.done_event.set()
            recv_task.cancel()
