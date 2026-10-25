"""
Hammerhead (Karoo) API provider implementation (issue #69).

One-way, like Strava and Wahoo: activities are pulled from the Karoo and
nothing is sent back. The API could also take routes, workouts and training
metrics; that is deliberately left out to keep the integration simple.

OAuth 2.0 authorization code, the shape :class:`BaseProviderClient` models. The
token response carries ``user_id`` directly, so unlike Wahoo there is no
second profile call to learn ``provider_athlete_id``.

**Why the activity list is sparse.** ``GET /activities`` returns only
``{id, name, createdAt, duration, distance}`` — no start time, no sport, and no
power, HR or cadence at all. ``createdAt`` is when the *record* was made, not
when the ride started, so dating an activity by it would break the ±5-minute
duplicate match against the same ride synced from Strava or Wahoo. The FIT is
therefore mandatory per activity, not an optimisation.

Downloading it for every listed activity would make every sync re-download the
whole history just to skip it, so :meth:`HammerheadClient.list_activities`
returns *sparse* records and :meth:`HammerheadClient.complete_activity` fills
one in — detail call plus FIT — only when the sync pipeline is actually going
to work on it. The FIT is cached so the import that follows does not fetch it a
second time.

**A withheld grant.** The documented token response carries no ``scope``
field, so a connection whose athlete declined ``activity:read`` is only found
out on its first 403 (:class:`ScopeNotGranted`).
"""

from __future__ import annotations

import asyncio
import io
import logging
import urllib.parse
from collections import OrderedDict
from datetime import date, datetime, timezone
from typing import Any

import httpx

from backend.app.core.config import settings
from backend.app.services.providers.base import BaseProviderClient, NormalizedActivity
from backend.app.services.providers.http import provider_client
from backend.app.services.providers.throttling import RETRYABLE_STATUSES, from_response
from backend.app.services.providers.wahoo import _parse_fit_streams

log = logging.getLogger(__name__)

_HOST = "https://api.hammerhead.io"
_AUTH_BASE = f"{_HOST}/v1/auth"
_AUTH_URL = f"{_AUTH_BASE}/oauth/authorize"
_TOKEN_URL = f"{_AUTH_BASE}/oauth/token"
_DEAUTHORIZE_URL = f"{_AUTH_BASE}/oauth/deauthorize"
_API_BASE = f"{_HOST}/v1/api"

#: Read-only. ``activity:read`` also covers receiving webhooks.
SCOPES: tuple[str, ...] = ("activity:read",)

#: The API's own maximum, and real pagination metadata comes back with it — so
#: the walk ends on ``currentPage >= totalPages`` without probing an empty page.
_PAGE_SIZE = 100
_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0)

#: Service name recorded against every request below (issue #66).
PROVIDER = "hammerhead"

#: How many downloaded FITs a client keeps for the import that follows
#: :meth:`HammerheadClient.complete_activity`. The sync works on one activity
#: at a time, so a handful is plenty; bounded because a FIT is up to a few MB.
_FIT_CACHE_SIZE = 4

#: Hammerhead ``activityType`` → canonical sport string.
_SPORT_TYPES: dict[str, str] = {
    "RIDE": "Ride",
    "EBIKE": "EBikeRide",
    "MOUNTAIN_BIKE": "MountainBikeRide",
    "GRAVEL": "GravelRide",
    "EMOUNTAIN_BIKE": "EMountainBikeRide",
    "VELOMOBILE": "Velomobile",
}


class ScopeNotGranted(Exception):
    """The athlete did not grant the scope this call needs.

    Raised on a 403 rather than letting ``raise_for_status`` produce an opaque
    HTTP error: with no ``scope`` field on the token response, a refusal is the
    only way we ever learn what was granted, and the caller should be able to
    say so in words.
    """

    def __init__(self, scope: str) -> None:
        self.scope = scope
        super().__init__(f"the Hammerhead '{scope}' permission was not granted")


def sport_type_for(activity_type: str | None) -> str | None:
    """Map a Hammerhead ``activityType`` to our canonical sport strings."""
    if not activity_type:
        return None
    return _SPORT_TYPES.get(str(activity_type).upper(), "Ride")


def _check_scope(response: httpx.Response, scope: str) -> None:
    if response.status_code == 403:
        raise ScopeNotGranted(scope)


def _parse_dt(value: Any) -> datetime | None:
    """An ISO-8601 timestamp or epoch milliseconds, as an aware datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # Epoch seconds or milliseconds — anything past 1e11 cannot be seconds.
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _items(data: Any) -> list[dict]:
    """The records of a paginated response, whatever the envelope is called."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "activities", "items", "routes", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    return []


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _token_result(data: dict, *, fallback_refresh: str | None = None) -> dict:
    expires_at = int(datetime.now(timezone.utc).timestamp()) + int(
        data.get("expires_in", 3600)
    )
    user_id = data.get("user_id") or data.get("userId") or ""
    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token") or fallback_refresh or "",
        "expires_at": expires_at,
        "provider_athlete_id": str(user_id),
    }


class HammerheadClient(BaseProviderClient):
    PROVIDER_NAME = "hammerhead"

    def __init__(self) -> None:
        # From the last list response; `list_activities` answers [] past it
        # without spending a request on an empty page.
        self._total_pages: int | None = None
        self._fit_cache: OrderedDict[str, bytes] = OrderedDict()
        #: Set by an incremental caller to list only activities since a date.
        self.start_date: date | None = None

    # ── OAuth ──────────────────────────────────────────────────────────────

    def get_oauth_url(self, state: str, redirect_uri: str) -> str:
        params = {
            "client_id": settings.hammerhead_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "state": state,
        }
        return f"{_AUTH_URL}?{urllib.parse.urlencode(params)}"

    @staticmethod
    async def exchange_code(code: str, redirect_uri: str) -> dict:  # type: ignore[override]
        async with provider_client(PROVIDER, timeout=_TIMEOUT) as client:
            r = await client.post(
                _TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": settings.hammerhead_client_id,
                    "client_secret": settings.hammerhead_client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
            r.raise_for_status()
            return _token_result(r.json())

    @staticmethod
    async def refresh_access_token(refresh_token: str) -> dict:  # type: ignore[override]
        async with provider_client(PROVIDER, timeout=_TIMEOUT) as client:
            r = await client.post(
                _TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": settings.hammerhead_client_id,
                    "client_secret": settings.hammerhead_client_secret,
                    "refresh_token": refresh_token,
                },
            )
            r.raise_for_status()
            return _token_result(r.json(), fallback_refresh=refresh_token)

    @staticmethod
    async def deauthorize(access_token: str) -> None:  # type: ignore[override]
        """Unlink the account (best-effort; the caller swallows errors).

        Hammerhead also deletes any routes and workouts this client created,
        which, since nothing is ever pushed, is none.
        """
        async with provider_client(PROVIDER, timeout=_TIMEOUT) as client:
            await client.post(
                _DEAUTHORIZE_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                data={"client_id": settings.hammerhead_client_id},
            )

    # ── Activities ─────────────────────────────────────────────────────────

    async def list_activities(
        self, access_token: str, page: int
    ) -> list[NormalizedActivity]:
        """One page of *sparse* activities — see the module docstring.

        ``start_time`` is the record's ``createdAt``: close enough to say how
        far back a walk reached, and never used for duplicate matching, because
        :meth:`complete_activity` replaces it with the FIT's own start before
        the pipeline looks for one.
        """
        if self._total_pages is not None and page > self._total_pages:
            return []

        params: dict[str, Any] = {"page": page, "perPage": _PAGE_SIZE}
        if self.start_date is not None:
            params["startDate"] = self.start_date.isoformat()

        async with provider_client(PROVIDER, timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_API_BASE}/activities",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
            )
            _check_scope(r, "activity:read")
            r.raise_for_status()
            data = r.json()

        if isinstance(data, dict):
            total = _int(data.get("totalPages"))
            current = _int(data.get("currentPage")) or page
            if total is not None:
                self._total_pages = total
                if current > total:
                    return []

        result = []
        for raw in _items(data):
            if raw.get("id") is None:
                continue
            created = _parse_dt(raw.get("createdAt")) or datetime.now(timezone.utc)
            result.append(
                NormalizedActivity(
                    external_id=str(raw["id"]),
                    source=self.PROVIDER_NAME,
                    name=raw.get("name") or None,
                    sport_type=None,
                    start_time=created,
                    # The summary's `duration` and `distance` have no documented
                    # units, and the FIT is authoritative for both — so they are
                    # not guessed at here.
                    duration_s=None,
                    distance_m=None,
                    elevation_m=None,
                    avg_power=None,
                    avg_hr=None,
                    max_hr=None,
                    avg_speed_ms=None,
                    avg_cadence=None,
                )
            )
        return result

    async def get_activity_detail(self, access_token: str, external_id: str) -> dict:
        """``GET /activities/{id}`` — adds ``activityType`` and ``description``."""
        async with provider_client(PROVIDER, timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_API_BASE}/activities/{external_id}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            _check_scope(r, "activity:read")
            r.raise_for_status()
            return r.json()

    async def download_fit_file(
        self, access_token: str, external_id: str
    ) -> bytes | None:
        """``GET /activities/{id}/file``, served from cache after a completion.

        ``None`` is the provider's real answer that there is no file; a throttle
        or a 5xx raises instead, so it is never mistaken for one (issue #67).
        """
        cached = self._fit_cache.get(external_id)
        if cached is not None:
            return cached
        async with provider_client(
            PROVIDER, timeout=_TIMEOUT, follow_redirects=True
        ) as client:
            r = await client.get(
                f"{_API_BASE}/activities/{external_id}/file",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if r.status_code == 404:
            return None
        throttled = from_response(r, self.PROVIDER_NAME)
        if throttled is not None:
            raise throttled
        _check_scope(r, "activity:read")
        r.raise_for_status()
        self._remember_fit(external_id, r.content)
        return r.content

    def _remember_fit(self, external_id: str, content: bytes) -> None:
        self._fit_cache[external_id] = content
        self._fit_cache.move_to_end(external_id)
        while len(self._fit_cache) > _FIT_CACHE_SIZE:
            self._fit_cache.popitem(last=False)

    async def get_activity_streams(
        self, access_token: str, external_id: str
    ) -> dict[str, list[float]]:
        """There is no streams endpoint; the FIT is the only source."""
        fit_bytes = await self.download_fit_file(access_token, external_id)
        if fit_bytes is None:
            return {}
        return await asyncio.to_thread(_parse_fit_streams, fit_bytes)

    async def complete_activity(
        self, access_token: str, norm: NormalizedActivity
    ) -> NormalizedActivity:
        """Turn a sparse listing into a real one: FIT first, then the detail.

        The FIT supplies the true start time — the one field the duplicate
        match cannot do without — and the summary numbers. The detail call adds
        the sport and the athlete's own name for the ride; failing to get it is
        not fatal, because the FIT names a sport too.

        Raises ``LookupError`` when the activity has no file to date it by.
        """
        from openkoutsi.fit import summarizeWorkout
        from openkoutsi.fit_processing import resolve_sport_type

        fit_bytes = await self.download_fit_file(access_token, norm.external_id)
        if fit_bytes is None:
            raise LookupError(f"Hammerhead activity {norm.external_id} has no FIT file")
        profile = await asyncio.to_thread(summarizeWorkout, io.BytesIO(fit_bytes))

        detail: dict = {}
        try:
            detail = await self.get_activity_detail(access_token, norm.external_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in RETRYABLE_STATUSES:
                raise
            log.warning(
                "No detail for Hammerhead activity %s (HTTP %s) — using the FIT alone",
                norm.external_id,
                exc.response.status_code,
            )
        except ScopeNotGranted:
            log.warning(
                "No detail for Hammerhead activity %s (scope refused) — using the FIT alone",
                norm.external_id,
            )

        start_time = profile.start_time
        # The parser stands in the epoch for a file with no record timestamps.
        if start_time is None or start_time == datetime.fromtimestamp(0):
            raise LookupError(
                f"Hammerhead activity {norm.external_id} has an undated FIT file"
            )
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        sport_type = sport_type_for(detail.get("activityType")) or resolve_sport_type(
            profile.sport_type
        )
        speed = [v for v in profile.speed if v is not None]
        return NormalizedActivity(
            external_id=norm.external_id,
            source=self.PROVIDER_NAME,
            name=(
                detail.get("name")
                or norm.name
                or f"{sport_type} {start_time.strftime('%Y-%m-%d')}"
            ),
            sport_type=sport_type,
            start_time=start_time,
            duration_s=profile.duration or None,
            distance_m=float(profile.distance) if profile.distance else None,
            elevation_m=float(profile.elevationGain) if profile.elevationGain else None,
            avg_power=profile.avgPower or None,
            avg_hr=profile.avgHeartRate or None,
            max_hr=profile.peakHR or None,
            avg_speed_ms=(sum(speed) / len(speed) / 3.6) if speed else None,
            avg_cadence=float(profile.avgCadence) if profile.avgCadence else None,
        )
