"""
Wahoo Cloud API provider implementation.

Uses Wahoo's OAuth 2.0 authorization code flow. Activity summaries come from
the /v1/workouts endpoint; time-series streams are obtained by downloading the
FIT file for each workout and parsing it with fitdecode.
"""

import base64
import io
import json
import logging
import urllib.parse
from datetime import datetime, timezone

import fitdecode
import httpx

log = logging.getLogger(__name__)

_dbg = logging.getLogger("wahoo.raw_debug")

from backend.app.core.config import settings
from backend.app.services.providers.base import BaseProviderClient, NormalizedActivity, ZoneData

_BASE = "https://api.wahooligan.com"
_AUTH_URL = f"{_BASE}/oauth/authorize"
_TOKEN_URL = f"{_BASE}/oauth/token"
_API_BASE = f"{_BASE}/v1"

_SCOPES = (
    "user_read workouts_read workouts_write "
    "plans_read plans_write offline_data power_zones_read"
)
_PAGE_SIZE = 30
_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)

# Wahoo workout_type_id → sport_type string
_SPORT_TYPES: dict[int, str] = {
    0:   "Ride",
    1:   "Run",
    2:   "Workout",
    3:   "TrackRun",
    4:   "TrailRun",
    5:   "Treadmill",
    6:   "Walk",
    7:   "Walk",
    8:   "NordicWalk",
    9:   "Hike",
    10:  "Mountaineering",
    11:  "CycloCross",
    12:  "VirtualRide",
    13:  "MountainBikeRide",
    14:  "RecumbentRide",
    15:  "Ride",
    16:  "TrackCycling",
    17:  "MotoSport",
    18:  "Workout",
    19:  "Treadmill",
    20:  "Elliptical",
    21:  "VirtualRide",
    22:  "Rowing",
    23:  "StairStepper",
    25:  "Swim",
    26:  "OpenWaterSwim",
    27:  "Snowboard",
    28:  "Ski",
    29:  "AlpineSki",
    30:  "NordicSki",
    31:  "Skating",
    32:  "IceSkate",
    33:  "InlineSkate",
    34:  "Skateboard",
    35:  "Sailing",
    36:  "Windsurf",
    37:  "Canoeing",
    38:  "Kayaking",
    39:  "Rowing",
    40:  "Kitesurf",
    41:  "StandUpPaddling",
    42:  "Workout",
    43:  "CardioClass",
    44:  "StairStepper",
    45:  "Wheelchair",
    46:  "Golf",
    47:  "Other",
    49:  "VirtualRide",
    56:  "Walk",
    61:  "VirtualRide",
    62:  "Multisport",
    63:  "Transition",
    64:  "EBikeRide",
    65:  "Other",
    66:  "Yoga",
    67:  "Run",
    68:  "VirtualRide",
    69:  "MentalStrength",
    70:  "Handcycle",
    71:  "VirtualRun",
    255: "Other",
}

# sport_type → Wahoo workout_type_id for the workout record created when pushing a
# structured workout. Indoor/virtual rides use the indoor-trainer type so the
# plan plays against a smart trainer; everything else defaults to outdoor biking.
_PUSH_WORKOUT_TYPE_IDS: dict[str, int] = {
    "Ride": 0,            # BIKING
    "VirtualRide": 61,    # BIKING_INDOOR_TRAINER
    "MountainBikeRide": 13,
    "Run": 1,             # RUNNING
    "VirtualRun": 71,
    "TrackRun": 3,
    "TrailRun": 4,
    "Treadmill": 5,
}
_DEFAULT_PUSH_WORKOUT_TYPE_ID = 0


def workout_type_id_for(sport_type: str | None) -> int:
    """Return the Wahoo workout_type_id to use when scheduling a pushed workout."""
    return _PUSH_WORKOUT_TYPE_IDS.get(sport_type or "", _DEFAULT_PUSH_WORKOUT_TYPE_ID)


class WahooClient(BaseProviderClient):
    PROVIDER_NAME = "wahoo"

    def __init__(self) -> None:
        # Populated during list_activities; used as fallback in download_fit_file.
        self._fit_urls: dict[str, str] = {}

    # ── OAuth ──────────────────────────────────────────────────────────────

    def get_oauth_url(self, state: str, redirect_uri: str) -> str:
        params = {
            "client_id": settings.wahoo_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": _SCOPES,
            "state": state,
        }
        return f"{_AUTH_URL}?{urllib.parse.urlencode(params)}"

    @staticmethod
    async def exchange_code(code: str, redirect_uri: str) -> dict:  # type: ignore[override]
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                _TOKEN_URL,
                data={
                    "client_id": settings.wahoo_client_id,
                    "client_secret": settings.wahoo_client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            r.raise_for_status()
            data = r.json()

        # Fetch user profile to get the Wahoo user ID
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            u = await client.get(
                f"{_API_BASE}/user",
                headers={"Authorization": f"Bearer {data['access_token']}"},
            )
            u.raise_for_status()
            user = u.json()

        expires_at = int(datetime.now(timezone.utc).timestamp()) + int(
            data.get("expires_in", 3600)
        )
        return {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "expires_at": expires_at,
            "provider_athlete_id": str(user.get("id", "")),
        }

    @staticmethod
    async def refresh_access_token(refresh_token: str) -> dict:  # type: ignore[override]
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                _TOKEN_URL,
                data={
                    "client_id": settings.wahoo_client_id,
                    "client_secret": settings.wahoo_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            r.raise_for_status()
            data = r.json()

        expires_at = int(datetime.now(timezone.utc).timestamp()) + int(
            data.get("expires_in", 3600)
        )
        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", refresh_token),
            "expires_at": expires_at,
            "provider_athlete_id": "",
        }

    @staticmethod
    async def deauthorize(access_token: str) -> None:  # type: ignore[override]
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            await client.delete(
                f"{_API_BASE}/permissions",
                headers={"Authorization": f"Bearer {access_token}"},
            )

    # ── Data ───────────────────────────────────────────────────────────────

    async def list_activities(
        self, access_token: str, page: int
    ) -> list[NormalizedActivity]:
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_API_BASE}/workouts",
                headers=headers,
                params={
                    "page": page,
                    "per_page": _PAGE_SIZE,
                    "order_by": "starts",
                    "order_dir": "desc",
                },
            )
            r.raise_for_status()
            data = r.json()

        _dbg.debug(
            "list_activities page=%s raw response:\n%s",
            page,
            json.dumps(data, indent=2, default=str),
        )
        workouts: list[dict] = data.get("workouts", [])
        # Cache CDN FIT URLs so download_fit_file can fall back when the API endpoint returns 404.
        for w in workouts:
            summary = w.get("workout_summary") or {}
            file_info = summary.get("file") or {}
            url = file_info.get("url")
            if url:
                self._fit_urls[str(w["id"])] = url
        return [_normalize_workout(w) for w in workouts]

    async def download_fit_file(
        self, access_token: str, external_id: str
    ) -> bytes | None:
        """Download the raw FIT file for a Wahoo workout."""
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(
                f"{_API_BASE}/workouts/{external_id}/fit_file",
                headers=headers,
            )
            if r.status_code == 404:
                _dbg.debug("download_fit_file workout_id=%s → 404 from API endpoint", external_id)
            elif not r.is_success:
                _dbg.debug(
                    "download_fit_file workout_id=%s → HTTP %d from API endpoint: %s",
                    external_id, r.status_code, r.text[:300],
                )
            else:
                _dbg.debug(
                    "download_fit_file workout_id=%s → %d bytes via API endpoint",
                    external_id, len(r.content),
                )
                return r.content

        # API endpoint failed; try the direct CDN URL cached from the workout summary.
        cdn_url = self._fit_urls.get(external_id)
        if cdn_url:
            _dbg.debug("download_fit_file workout_id=%s → trying CDN URL %s", external_id, cdn_url)
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as cdn_client:
                cdn_r = await cdn_client.get(cdn_url)
                if cdn_r.is_success:
                    _dbg.debug(
                        "download_fit_file workout_id=%s → %d bytes via CDN URL",
                        external_id, len(cdn_r.content),
                    )
                    return cdn_r.content
                _dbg.debug(
                    "download_fit_file workout_id=%s → CDN URL returned HTTP %d",
                    external_id, cdn_r.status_code,
                )
        else:
            _dbg.debug("download_fit_file workout_id=%s → no CDN URL cached", external_id)

        return None

    async def get_activity_streams(
        self, access_token: str, external_id: str
    ) -> dict[str, list[float]]:
        """Download the FIT file for this workout and extract streams."""
        fit_bytes = await self.download_fit_file(access_token, external_id)
        if fit_bytes is None:
            return {}
        streams = _parse_fit_streams(fit_bytes)
        _dbg.debug(
            "get_activity_streams workout_id=%s parsed keys=%s lengths=%s",
            external_id,
            list(streams.keys()),
            {k: len(v) for k, v in streams.items()},
        )
        return streams

    async def fetch_zones(self, access_token: str) -> ZoneData:
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(f"{_API_BASE}/power_zones", headers=headers)
        r.raise_for_status()
        raw = r.json()
        entries: list[dict] = raw if isinstance(raw, list) else raw.get("power_zones", [])
        if not entries:
            return ZoneData()

        # Use the first entry (Wahoo returns one record per workout type family;
        # the first is typically the cycling / generic power zone set).
        entry = entries[0]
        ftp_raw = entry.get("ftp")
        ftp = int(ftp_raw) if ftp_raw else None

        zone_count = int(entry.get("zone_count", 7))
        thresholds = [entry.get(f"zone_{i}") for i in range(1, zone_count + 1)]
        thresholds = [int(t) for t in thresholds if t is not None]

        power_zones = _normalize_wahoo_zones(thresholds)

        return ZoneData(ftp=ftp, power_zones=power_zones or None)

    # ── Structured workout upload ───────────────────────────────────────────

    async def find_plan_by_external_id(
        self, access_token: str, external_id: str
    ) -> dict | None:
        """Return the existing plan record for an external_id, or None."""
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_API_BASE}/plans",
                headers=headers,
                params={"external_id": external_id},
            )
            r.raise_for_status()
            data = r.json()

        plans = data.get("plans", data) if isinstance(data, dict) else data
        if isinstance(plans, list):
            for plan in plans:
                if str(plan.get("external_id")) == str(external_id):
                    return plan
            return plans[0] if plans else None
        return None

    async def create_or_update_plan(
        self,
        access_token: str,
        *,
        plan_json: dict,
        external_id: str,
        provider_updated_at: datetime,
        filename: str | None = None,
    ) -> str:
        """Create or update a plan in the user's library; return the plan id.

        The plan file is uploaded as a base64-encoded data URI. When a plan with
        the same external_id already exists it is updated (PUT) rather than
        duplicated.
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        encoded = base64.b64encode(
            json.dumps(plan_json).encode("utf-8")
        ).decode("ascii")
        data = {
            "plan[file]": f"data:application/json;base64,{encoded}",
            "plan[external_id]": external_id,
            "plan[provider_updated_at]": provider_updated_at.isoformat(),
        }
        if filename:
            data["plan[filename]"] = filename

        existing = await self.find_plan_by_external_id(access_token, external_id)

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            if existing and existing.get("id"):
                r = await client.put(
                    f"{_API_BASE}/plans/{existing['id']}", headers=headers, data=data
                )
            else:
                r = await client.post(
                    f"{_API_BASE}/plans", headers=headers, data=data
                )
            r.raise_for_status()
            result = r.json()
        return str(result["id"])

    async def create_or_update_workout(
        self,
        access_token: str,
        *,
        name: str,
        workout_token: str,
        workout_type_id: int,
        starts: datetime,
        minutes: int,
        plan_id: str,
        existing_id: str | None = None,
    ) -> str:
        """Create or update a workout record referencing a plan; return its id.

        Scheduling ``starts`` within today→+6 days is what makes the plan visible
        on ELEMNT / RIVAL devices.
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        data = {
            "workout[name]": name,
            "workout[workout_token]": workout_token,
            "workout[workout_type_id]": str(workout_type_id),
            "workout[starts]": starts.isoformat(),
            "workout[minutes]": str(minutes),
            "workout[plan_id]": str(plan_id),
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            if existing_id:
                r = await client.put(
                    f"{_API_BASE}/workouts/{existing_id}", headers=headers, data=data
                )
            else:
                r = await client.post(
                    f"{_API_BASE}/workouts", headers=headers, data=data
                )
            r.raise_for_status()
            result = r.json()
        return str(result["id"])


# ── Helpers ────────────────────────────────────────────────────────────────

def _normalize_wahoo_zones(thresholds: list[int]) -> list[dict]:
    """Convert Wahoo threshold list (upper bounds) to [{low, high, name}] format."""
    zones = []
    for i, upper in enumerate(thresholds):
        low = thresholds[i - 1] if i > 0 else 0
        high = upper if i < len(thresholds) - 1 else 9999
        zones.append({"name": f"Z{i + 1}", "low": low, "high": high})
    return zones


def _normalize_workout(raw: dict) -> NormalizedActivity:
    _dbg.debug(
        "normalize_workout raw id=%s:\n%s",
        raw.get("id"),
        json.dumps(raw, indent=2, default=str),
    )
    summary: dict = raw.get("workout_summary") or {}

    sport_id: int = raw.get("workout_type_id", 0)
    sport_type = _SPORT_TYPES.get(sport_id, f"Workout_{sport_id}")

    starts: str | None = raw.get("starts")
    if starts:
        start_time = datetime.fromisoformat(starts.replace("Z", "+00:00"))
    else:
        start_time = datetime.now(timezone.utc)

    # Prefer active (moving) time; fall back to total duration.
    duration_s = _int_or_none(summary.get("duration_active_accum")) or \
                 _int_or_none(summary.get("duration_total_accum"))

    return NormalizedActivity(
        external_id=str(raw["id"]),
        source="wahoo",
        name=raw.get("name") or f"{sport_type} {start_time.strftime('%Y-%m-%d')}",
        sport_type=sport_type,
        start_time=start_time,
        duration_s=duration_s,
        distance_m=_float_or_none(summary.get("distance_accum")),
        elevation_m=_float_or_none(summary.get("ascent_accum")),
        avg_power=_float_or_none(summary.get("power_avg")),
        avg_hr=_float_or_none(summary.get("heart_rate_avg")),
        max_hr=_float_or_none(summary.get("heart_rate_max")),
        avg_speed_ms=_float_or_none(summary.get("speed_avg")),
        avg_cadence=_float_or_none(summary.get("cadence_avg")),
    )


def _parse_fit_streams(fit_bytes: bytes) -> dict[str, list[float]]:
    """Extract time-series streams from a FIT file in memory."""
    from openkoutsi.fit import summarizeWorkout

    profile = summarizeWorkout(io.BytesIO(fit_bytes))

    result: dict[str, list[float]] = {}
    if profile.power:
        result["power"] = [float(v) for v in profile.power]
    if profile.heartRate:
        result["heartrate"] = [float(v) for v in profile.heartRate]
    if profile.cadence:
        result["cadence"] = [float(v) for v in profile.cadence]
    if profile.speed:
        # summarizeWorkout stores speed in km/h; callers expect m/s
        result["speed"] = [v / 3.6 for v in profile.speed]
    if profile.altitude:
        result["altitude"] = [float(v) for v in profile.altitude]
    return result


def _int_or_none(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
