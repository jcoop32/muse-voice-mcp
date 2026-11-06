#!/usr/bin/env python3
"""Automated Telnyx provisioning & verification script for muse-voice-mcp.

Given a Telnyx v2 API Key (KEY...), this script:
1. Verifies account balance and credentials.
2. Finds or creates a Telnyx Outbound Voice Profile ('muse-outbound-profile').
3. Finds or creates a Telnyx TeXML Application ('muse-voice-mcp') wired to your
   Cloudflare Tunnel webhook (https://voice-mcp.cooperlabs.app) and Outbound Voice Profile.
4. Lists your owned Telnyx phone numbers (or searches available local numbers by area code)
   and binds the chosen number to the TeXML Application.
5. Writes `/Users/joshuacooper/muse-voice-mcp/.env` and outputs the ready-to-run
   `kubectl create secret generic muse-voice-secrets` command for K3s.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path
from typing import Any

import httpx

TELNYX_BASE = "https://api.telnyx.com/v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class TelnyxProvisioner:
    def __init__(self, api_key: str, public_domain: str = "voice-mcp.cooperlabs.app") -> None:
        self.api_key = api_key.strip()
        self.public_domain = public_domain.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
        self.client = httpx.Client(
            base_url=TELNYX_BASE,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=20.0,
        )

    def verify_account(self) -> dict[str, Any]:
        resp = self.client.get("/balance")
        if resp.status_code == 401:
            raise RuntimeError("Invalid Telnyx API Key (401 Unauthorized). Check your KEY019... token.")
        resp.raise_for_status()
        return resp.json().get("data", {})

    def ensure_outbound_voice_profile(self, name: str = "muse-outbound-profile") -> str:
        """Find or create an Outbound Voice Profile (required by Telnyx to place outbound calls)."""
        resp = self.client.get("/outbound_voice_profiles")
        resp.raise_for_status()
        for prof in resp.json().get("data", []):
            if prof.get("name") == name:
                return str(prof["id"])

        create_resp = self.client.post(
            "/outbound_voice_profiles",
            json={
                "name": name,
                "enabled": True,
                "traffic_type": "conversational",
                "whitelisted_destinations": ["US", "CA"],
            },
        )
        create_resp.raise_for_status()
        return str(create_resp.json()["data"]["id"])

    def ensure_texml_application(
        self, outbound_profile_id: str, app_name: str = "muse-voice-mcp"
    ) -> str:
        """Find or create the TeXML Application linked to the Outbound Voice Profile."""
        resp = self.client.get("/texml_applications")
        resp.raise_for_status()
        voice_url = f"https://{self.public_domain}/webhooks/telephony/twiml/inbound"
        status_url = f"https://{self.public_domain}/webhooks/telephony/status/inbound"

        for app in resp.json().get("data", []):
            if app.get("friendly_name") == app_name:
                app_id = str(app["id"])
                # Ensure outbound voice profile is attached
                self.client.patch(
                    f"/texml_applications/{app_id}",
                    json={
                        "friendly_name": app_name,
                        "voice_url": voice_url,
                        "status_callback": status_url,
                        "outbound": {"outbound_voice_profile_id": outbound_profile_id},
                    },
                )
                return app_id

        create_resp = self.client.post(
            "/texml_applications",
            json={
                "friendly_name": app_name,
                "voice_url": voice_url,
                "voice_method": "POST",
                "status_callback": status_url,
                "status_callback_method": "POST",
                "outbound": {
                    "outbound_voice_profile_id": outbound_profile_id,
                    "channel_limit": 5,
                },
            },
        )
        create_resp.raise_for_status()
        return str(create_resp.json()["data"]["id"])

    def list_owned_numbers(self) -> list[dict[str, Any]]:
        resp = self.client.get("/phone_numbers?page[size]=50")
        resp.raise_for_status()
        return resp.json().get("data", [])

    def search_available_numbers(
        self, area_code: str, limit: int = 5, max_monthly_cost: float = 1.00
    ) -> list[str]:
        params = {
            "filter[country_code]": "US",
            "filter[national_destination_code]": area_code,
            "filter[features][]": "voice",
            "filter[limit]": str(limit),
        }
        resp = self.client.get("/available_phone_numbers", params=params)
        resp.raise_for_status()
        results: list[str] = []
        for item in resp.json().get("data", []):
            cost = item.get("cost_information", {})
            monthly = float(cost.get("monthly_cost", "1.0"))
            upfront = float(cost.get("upfront_cost", "1.0"))
            if monthly <= max_monthly_cost and upfront <= max_monthly_cost:
                results.append(item["phone_number"])
        return results

    def purchase_number(self, phone_number: str, connection_id: str) -> dict[str, Any]:
        resp = self.client.post(
            "/number_orders",
            json={
                "phone_numbers": [{"phone_number": phone_number}],
                "connection_id": connection_id,
            },
        )
        resp.raise_for_status()
        return resp.json().get("data", {})

    def release_number(self, number_id: str) -> None:
        resp = self.client.delete(f"/phone_numbers/{number_id}")
        resp.raise_for_status()

    def bind_number_to_connection(self, number_id: str, connection_id: str) -> None:
        resp = self.client.patch(
            f"/phone_numbers/{number_id}/voice",
            json={"connection_id": connection_id},
        )
        resp.raise_for_status()


def write_env_and_print_k8s(
    api_key: str,
    connection_id: str,
    caller_number: str,
    public_domain: str,
) -> None:
    mcp_api_key = secrets.token_hex(32)
    webhook_secret = secrets.token_hex(32)
    env_content = f"""ENVIRONMENT=production
HOST=0.0.0.0
PORT=8090
PUBLIC_DOMAIN={public_domain}
MCP_API_KEY={mcp_api_key}
WEBHOOK_SIGNING_SECRET={webhook_secret}

TELEPHONY_PROVIDER=telnyx
OUTBOUND_CALLER_NUMBER={caller_number}
TELNYX_API_KEY={api_key}
TELNYX_CONNECTION_ID={connection_id}

WHISPER_MODEL_SIZE=base.en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
TTS_ENGINE=piper
PIPER_VOICE_MODEL=en_US-lessac-medium
FILLER_DELAY_SECONDS=1.25
DATA_DIR=/tmp/muse-voice-data
"""
    ENV_FILE.write_text(env_content, encoding="utf-8")
    os.chmod(ENV_FILE, 0o600)
    print(f"\n[OK] Wrote local configuration to {ENV_FILE} (mode 0600)")
    print("\n==> Run this command to create/update the K3s secret (muse-voice-secrets):")
    print(
        f"""
kubectl create secret generic muse-voice-secrets \\
  --namespace default \\
  --from-literal=TELEPHONY_PROVIDER="telnyx" \\
  --from-literal=OUTBOUND_CALLER_NUMBER="{caller_number}" \\
  --from-literal=TELNYX_API_KEY="{api_key}" \\
  --from-literal=TELNYX_CONNECTION_ID="{connection_id}" \\
  --from-literal=TWILIO_ACCOUNT_SID="" \\
  --from-literal=TWILIO_AUTH_TOKEN="" \\
  --from-literal=MCP_API_KEY="{mcp_api_key}" \\
  --from-literal=WEBHOOK_SIGNING_SECRET="{webhook_secret}" \\
  --dry-run=client -o yaml | kubectl apply -f -
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Provision Telnyx for muse-voice-mcp")
    parser.add_argument("--api-key", default=os.getenv("TELNYX_API_KEY", ""), help="Telnyx API v2 Key (KEY019...)")
    parser.add_argument("--number", default=os.getenv("OUTBOUND_CALLER_NUMBER", ""), help="Existing Telnyx E.164 phone number (+1...)")
    parser.add_argument("--buy-area-code", default="", help="Optional 3-digit US area code to search and purchase a $1/mo number")
    parser.add_argument("--swap-to-area-code", default="", help="Release existing expensive number(s) and swap to a verified $1.00/mo number in this area code (e.g. 773 or 872)")
    parser.add_argument("--release-number", default="", help="Release a specific owned phone number (+1...) immediately so it never renews")
    parser.add_argument("--domain", default="voice-mcp.cooperlabs.app", help="Public Cloudflare Tunnel domain")
    args = parser.parse_args()

    api_key = args.api_key.strip()
    if not api_key:
        print("Error: Provide --api-key KEY019... or set TELNYX_API_KEY in your environment.", file=sys.stderr)
        return 1

    prov = TelnyxProvisioner(api_key=api_key, public_domain=args.domain)

    print("==> [1/4] Verifying Telnyx account...")
    bal = prov.verify_account()
    print(f"    Account verified! Balance: {bal.get('balance', '0.00')} {bal.get('currency', 'USD')}")

    print("==> [2/4] Ensuring Outbound Voice Profile ('muse-outbound-profile')...")
    profile_id = prov.ensure_outbound_voice_profile()
    print(f"    Outbound Voice Profile ID: {profile_id}")

    print("==> [3/4] Ensuring TeXML Application ('muse-voice-mcp')...")
    connection_id = prov.ensure_texml_application(outbound_profile_id=profile_id)
    print(f"    TeXML Connection/App ID:   {connection_id}")

    print("==> [4/4] Checking phone numbers...")
    owned = prov.list_owned_numbers()
    chosen_number = args.number.strip()

    if args.release_number:
        rel = args.release_number.strip()
        for num_obj in list(owned):
            if num_obj.get("phone_number") == rel:
                prov.release_number(str(num_obj["id"]))
                print(f"    Released {rel} (will not renew next month)!")
                owned.remove(num_obj)

    if args.swap_to_area_code:
        candidates = prov.search_available_numbers(args.swap_to_area_code, limit=3, max_monthly_cost=1.00)
        if not candidates:
            print(
                f"Error: No <=$1.00/mo numbers found in area code {args.swap_to_area_code}. Existing number kept safe.",
                file=sys.stderr,
            )
            return 1
        for num_obj in list(owned):
            old_num = num_obj.get("phone_number")
            prov.release_number(str(num_obj["id"]))
            print(f"    Released old number {old_num}!")
        chosen_number = candidates[0]
        print(f"    Purchasing verified $1.00/mo number {chosen_number} and binding to {connection_id}...")
        prov.purchase_number(chosen_number, connection_id)
        print(f"    Purchased and bound {chosen_number}!")
    elif owned:
        target = next((n for n in owned if n.get("phone_number") == chosen_number), owned[0])
        chosen_number = target["phone_number"]
        prov.bind_number_to_connection(str(target["id"]), connection_id)
        print(f"    Bound owned number {chosen_number} to TeXML App {connection_id}!")
    elif args.buy_area_code:
        candidates = prov.search_available_numbers(args.buy_area_code, limit=3, max_monthly_cost=1.00)
        if not candidates:
            print(
                f"Error: No <=$1.00/mo numbers found in area code {args.buy_area_code} (area code may charge a $5 premium). Try 773 or 872.",
                file=sys.stderr,
            )
            return 1
        chosen_number = candidates[0]
        print(f"    Purchasing verified $1.00/mo number {chosen_number} and binding to {connection_id}...")
        prov.purchase_number(chosen_number, connection_id)
        print(f"    Purchased and bound {chosen_number}!")
    else:
        print("    No phone numbers currently on account. Pass --buy-area-code <3-digit-code> or buy one in portal.telnyx.com.")
        chosen_number = chosen_number or "+15550001234"

    write_env_and_print_k8s(
        api_key=api_key,
        connection_id=connection_id,
        caller_number=chosen_number,
        public_domain=args.domain,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
