"""Unified Telnyx & Twilio outbound telephony driver (REST + TeXML/TwiML Media Streams)."""

from __future__ import annotations

import logging
import re
from html import escape
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

E164_REGEX = re.compile(r"^\+[1-9]\d{6,14}$")


def normalize_phone_number(raw: str) -> str:
    """Normalize a US/international phone number into E.164 format (+1XXXXXXXXXX)."""
    digits = re.sub(r"[^\d+]", "", raw.strip())
    if not digits.startswith("+"):
        if len(digits) == 10:
            digits = f"+1{digits}"
        elif len(digits) == 11 and digits.startswith("1"):
            digits = f"+{digits}"
        else:
            digits = f"+{digits}"
    if not E164_REGEX.match(digits):
        raise ValueError(
            f"Invalid phone number '{raw}'. Expected E.164 format (e.g. +15551234567)."
        )
    return digits


def build_media_stream_xml(provider: str, ws_url: str, call_id: str) -> str:
    """Build TeXML (Telnyx) or TwiML (Twilio) XML instructing bidirectional 8kHz PCMU streaming."""
    safe_url = escape(ws_url, quote=True)
    safe_call_id = escape(call_id, quote=True)
    if provider == "telnyx":
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            "<Connect>"
            f'<Stream url="{safe_url}" bidirectionalMode="rtp" codec="PCMU">'
            f'<Parameter name="call_id" value="{safe_call_id}" />'
            "</Stream>"
            "</Connect>"
            "</Response>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{safe_url}">'
        f'<Parameter name="call_id" value="{safe_call_id}" />'
        "</Stream>"
        "</Connect>"
        "</Response>"
    )


class TelephonyDriver:
    """Dispatches outbound calls, sends DTMF keypad tones, and terminates calls via Telnyx or Twilio."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = http_client

    @property
    def provider(self) -> str:
        return self.settings.telephony_provider

    async def dial_outbound(self, call_id: str, to_number: str) -> dict[str, Any]:
        """Initiate an outbound PSTN call and connect its bidirectional media stream."""
        normalized_to = normalize_phone_number(to_number)
        ws_url = f"{self.settings.public_wss_base}/ws/media/{call_id}"
        twiml_url = f"{self.settings.public_https_base}/webhooks/telephony/twiml/{call_id}"
        status_url = f"{self.settings.public_https_base}/webhooks/telephony/status/{call_id}"

        if self.provider == "telnyx":
            return await self._dial_telnyx(
                call_id=call_id,
                to_number=normalized_to,
                ws_url=ws_url,
                twiml_url=twiml_url,
                status_url=status_url,
            )
        if self.provider == "twilio":
            return await self._dial_twilio(
                call_id=call_id,
                to_number=normalized_to,
                ws_url=ws_url,
                twiml_url=twiml_url,
                status_url=status_url,
            )
        raise ValueError(f"Unsupported TELEPHONY_PROVIDER: {self.provider}")

    async def _dial_telnyx(
        self,
        call_id: str,
        to_number: str,
        ws_url: str,
        twiml_url: str,
        status_url: str,
    ) -> dict[str, Any]:
        if not self.settings.telnyx_api_key or not self.settings.telnyx_connection_id:
            logger.warning(
                "Telnyx credentials not set; running call_id=%s in simulated dry-run mode.",
                call_id,
            )
            return {"provider": "telnyx", "provider_call_id": f"sim_telnyx_{call_id}", "simulated": True}

        url = f"https://api.telnyx.com/v2/texml/calls/{self.settings.telnyx_connection_id}"
        headers = {"Authorization": f"Bearer {self.settings.telnyx_api_key}"}
        payload = {
            "To": to_number,
            "From": self.settings.outbound_caller_number,
            "Url": twiml_url,
            "StatusCallback": status_url,
            "StatusCallbackEvent": "initiated ringing answered completed",
        }
        async with self._get_client() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            sid = data.get("data", {}).get("call_sid") or data.get("call_sid") or f"telnyx_{call_id}"
            return {"provider": "telnyx", "provider_call_id": sid, "simulated": False}

    async def _dial_twilio(
        self,
        call_id: str,
        to_number: str,
        ws_url: str,
        twiml_url: str,
        status_url: str,
    ) -> dict[str, Any]:
        if not self.settings.twilio_account_sid or not self.settings.twilio_auth_token:
            logger.warning(
                "Twilio credentials not set; running call_id=%s in simulated dry-run mode.",
                call_id,
            )
            return {"provider": "twilio", "provider_call_id": f"sim_twilio_{call_id}", "simulated": True}

        sid = self.settings.twilio_account_sid
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json"
        xml = build_media_stream_xml("twilio", ws_url, call_id)
        form_data = {
            "To": to_number,
            "From": self.settings.outbound_caller_number,
            "Twiml": xml,
            "StatusCallback": status_url,
            "StatusCallbackMethod": "POST",
        }
        async with self._get_client() as client:
            resp = await client.post(
                url,
                data=form_data,
                auth=(sid, self.settings.twilio_auth_token),
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "provider": "twilio",
                "provider_call_id": data.get("sid", f"twilio_{call_id}"),
                "simulated": False,
            }

    async def send_dtmf(self, provider_call_id: str, digits: str, ws_url: str, call_id: str) -> bool:
        """Send DTMF keypad tones on an active call."""
        clean_digits = re.sub(r"[^0-9*#wW]", "", digits)
        if not clean_digits:
            raise ValueError(f"Invalid DTMF digits: '{digits}'")

        if provider_call_id.startswith("sim_"):
            return True

        if self.provider == "telnyx":
            url = f"https://api.telnyx.com/v2/calls/{provider_call_id}/actions/send_dtmf"
            headers = {"Authorization": f"Bearer {self.settings.telnyx_api_key}"}
            async with self._get_client() as client:
                resp = await client.post(
                    url, json={"digits": clean_digits}, headers=headers, timeout=10.0
                )
                return resp.status_code < 400

        if self.provider == "twilio":
            sid = self.settings.twilio_account_sid
            url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls/{provider_call_id}.json"
            safe_url = escape(ws_url, quote=True)
            safe_digits = escape(clean_digits, quote=True)
            twiml = (
                f'<Response><Play digits="{safe_digits}"/>'
                f'<Connect><Stream url="{safe_url}"/></Connect></Response>'
            )
            async with self._get_client() as client:
                resp = await client.post(
                    url,
                    data={"Twiml": twiml},
                    auth=(sid, self.settings.twilio_auth_token),
                    timeout=10.0,
                )
                return resp.status_code < 400
        return False

    async def hangup(self, provider_call_id: str) -> bool:
        """Terminate an active call via the provider's REST API."""
        if not provider_call_id or provider_call_id.startswith("sim_"):
            return True

        if self.provider == "telnyx":
            url = f"https://api.telnyx.com/v2/texml/Accounts/{self.settings.telnyx_connection_id}/Calls/{provider_call_id}"
            headers = {"Authorization": f"Bearer {self.settings.telnyx_api_key}"}
            async with self._get_client() as client:
                resp = await client.post(
                    url, json={"Status": "completed"}, headers=headers, timeout=10.0
                )
                return resp.status_code < 400

        if self.provider == "twilio":
            sid = self.settings.twilio_account_sid
            url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls/{provider_call_id}.json"
            async with self._get_client() as client:
                resp = await client.post(
                    url,
                    data={"Status": "completed"},
                    auth=(sid, self.settings.twilio_auth_token),
                    timeout=10.0,
                )
                return resp.status_code < 400
        return False

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        return httpx.AsyncClient()
