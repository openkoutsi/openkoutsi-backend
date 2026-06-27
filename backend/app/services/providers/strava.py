"""
Strava provider implementation.

Wraps the low-level StravaClient HTTP wrapper and adapts it to the
BaseProviderClient interface so the generic sync pipeline can drive it.
"""

import asyncio
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

import httpx

from backend.app.core.config import settings
from backend.app.services.providers.base import BaseProviderClient, NormalizedActivity, ZoneData

_AUTH_BASE = "https://www.strava.com"
_API_BASE = f"{_AUTH_BASE}/api/v3"
_STREAM_KEYS = "time,heartrate,watts,cadence,velocity_smooth,altitude,distance"

_STRAVA_AUTH_URL = f"{_AUTH_BASE}/oauth/authorize"
_STRAVA_SCOPE = "read,activity:read_all,profile:read_all"


# Sport type passthrough — Strava already returns human-readable strings.

_PAGE_SIZE = 200
_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)


class StravaProviderClient(BaseProviderClient):
    PROVIDER_NAME = "strava"

    # ── OAuth ──────────────────────────────────────────────────────────────

    def get_oauth_url(self, state: str, redirect_uri: str) -> str:
        params = {
            "client_id": settings.strava_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": _STRAVA_SCOPE,
            "approval_prompt": "auto",
            "state": state,
        }
        return f"{_STRAVA_AUTH_URL}?{urllib.parse.urlencode(params)}"

    @staticmethod
    async def exchange_code(code: str, redirect_uri: str) -> dict:  # type: ignore[override]
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                f"{_AUTH_BASE}/oauth/token",
                json={
                    "client_id": settings.strava_client_id,
                    "client_secret": settings.strava_client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                },
            )
            r.raise_for_status()
            data = r.json()

        athlete = data.get("athlete", {})
        return {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "expires_at": data["expires_at"],
            "provider_athlete_id": str(athlete.get("id", "")),
        }

    @staticmethod
    async def refresh_access_token(refresh_token: str) -> dict:  # type: ignore[override]
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                f"{_AUTH_BASE}/oauth/token",
                json={
                    "client_id": settings.strava_client_id,
                    "client_secret": settings.strava_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            r.raise_for_status()
            data = r.json()

        return {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "expires_at": data["expires_at"],
            "provider_athlete_id": "",  # not returned on refresh
        }

    @staticmethod
    async def deauthorize(access_token: str) -> None:  # type: ignore[override]
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            await client.post(
                f"{_AUTH_BASE}/oauth/deauthorize",
                data={"access_token": access_token},
            )

    # ── Data ───────────────────────────────────────────────────────────────

    async def list_activities(
        self, access_token: str, page: int
    ) -> list[NormalizedActivity]:
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_API_BASE}/athlete/activities",
                headers=headers,
                params={"page": page, "per_page": _PAGE_SIZE},
            )
            r.raise_for_status()
            raw_list: list[dict] = r.json()

        return [_normalize_activity(raw) for raw in raw_list]

    async def fetch_zones(self, access_token: str) -> ZoneData:
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r_athlete, r_zones = await asyncio.gather(
                client.get(f"{_API_BASE}/athlete", headers=headers),
                client.get(f"{_API_BASE}/athlete/zones", headers=headers),
            )
        r_athlete.raise_for_status()
        r_zones.raise_for_status()

        athlete_data = r_athlete.json()
        zones_data = r_zones.json()

        ftp_raw = athlete_data.get("ftp")
        ftp = int(ftp_raw) if ftp_raw else None

        hr_zones = _normalize_strava_zones(zones_data.get("heart_rate", {}).get("zones", []))
        power_zones = _normalize_strava_zones(zones_data.get("power", {}).get("zones", []))

        return ZoneData(
            ftp=ftp,
            hr_zones=hr_zones or None,
            power_zones=power_zones or None,
        )

    async def get_activity_streams(
        self, access_token: str, external_id: str
    ) -> dict[str, list[float]]:
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_API_BASE}/activities/{external_id}/streams",
                headers=headers,
                params={"keys": _STREAM_KEYS, "key_by_type": "true"},
            )
            r.raise_for_status()
            raw: dict = r.json()

        result: dict[str, list[float]] = {}
        _map(raw, result, "watts", "power")
        _map(raw, result, "heartrate", "heartrate")
        _map(raw, result, "cadence", "cadence")
        _map(raw, result, "velocity_smooth", "speed")
        _map(raw, result, "altitude", "altitude")
        return result


# ── Helpers ────────────────────────────────────────────────────────────────

def _normalize_strava_zones(raw_zones: list[dict]) -> list[dict]:
    """Convert Strava {min, max} zone list to internal {low, high, name} format."""
    result = []
    for i, z in enumerate(raw_zones):
        low = z.get("min", 0)
        high_raw = z.get("max", -1)
        high = 9999 if high_raw == -1 else high_raw
        result.append({"name": f"Z{i + 1}", "low": low, "high": high})
    return result


def _map(raw: dict, out: dict, src_key: str, dst_key: str) -> None:
    data = [float(v) for v in raw.get(src_key, {}).get("data", [])]
    if data:
        out[dst_key] = data


def _normalize_activity(raw: dict) -> NormalizedActivity:
    start_time = datetime.fromisoformat(raw["start_date"].replace("Z", "+00:00"))
    return NormalizedActivity(
        external_id=str(raw["id"]),
        source="strava",
        name=raw.get("name"),
        sport_type=raw.get("sport_type") or raw.get("type"),
        start_time=start_time,
        duration_s=raw.get("moving_time") or raw.get("elapsed_time"),
        distance_m=raw.get("distance"),
        elevation_m=raw.get("total_elevation_gain"),
        avg_power=raw.get("average_watts"),
        avg_hr=raw.get("average_heartrate"),
        max_hr=raw.get("max_heartrate"),
        avg_speed_ms=raw.get("average_speed"),
        avg_cadence=raw.get("average_cadence"),
    )
