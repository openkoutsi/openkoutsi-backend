"""
Abstract base class for training data provider integrations.

Each provider implements OAuth 2.0 (authorization code flow) and exposes
a uniform interface for listing activities and fetching time-series streams.
Adding a new provider is a matter of subclassing BaseProviderClient,
implementing the abstract methods, and registering the class in registry.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Optional


@dataclass
class ZoneData:
    """Provider-agnostic training zone data returned by fetch_zones."""
    ftp: Optional[int] = None
    hr_zones: Optional[list[dict]] = None    # [{low, high, name}, ...]
    power_zones: Optional[list[dict]] = None  # [{low, high, name}, ...]


@dataclass
class NormalizedActivity:
    """Provider-agnostic representation of a single workout."""

    external_id: str
    source: str  # provider name, e.g. "strava" or "wahoo"
    name: Optional[str]
    sport_type: Optional[str]
    start_time: datetime
    duration_s: Optional[int]
    distance_m: Optional[float]
    elevation_m: Optional[float]
    avg_power: Optional[float]
    avg_hr: Optional[float]
    max_hr: Optional[float]
    avg_speed_ms: Optional[float]
    avg_cadence: Optional[float]


class BaseProviderClient(ABC):
    """Abstract OAuth 2.0 training-data provider."""

    PROVIDER_NAME: ClassVar[str]

    # ── OAuth ──────────────────────────────────────────────────────────────

    @abstractmethod
    def get_oauth_url(self, state: str, redirect_uri: str) -> str:
        """Return the authorization URL to redirect the user to."""

    @abstractmethod
    async def exchange_code(self, code: str, redirect_uri: str) -> dict:
        """Exchange an authorization code for tokens.

        Returns a dict with keys:
            access_token     str
            refresh_token    str
            expires_at       int  (Unix timestamp)
            provider_athlete_id  str
        """

    @abstractmethod
    async def refresh_access_token(self, refresh_token: str) -> dict:
        """Obtain a new access token from a refresh token.

        Returns the same shape as exchange_code.
        """

    @abstractmethod
    async def deauthorize(self, access_token: str) -> None:
        """Deauthorize the app (best-effort; callers swallow errors)."""

    # ── Data ───────────────────────────────────────────────────────────────

    @abstractmethod
    async def list_activities(
        self, access_token: str, page: int
    ) -> list[NormalizedActivity]:
        """Return a page of activities (1-indexed). Return [] when exhausted."""

    @abstractmethod
    async def get_activity_streams(
        self, access_token: str, external_id: str
    ) -> dict[str, list[float]]:
        """Return time-series streams for a single activity.

        Keys: "power", "heartrate", "cadence", "speed", "altitude"
        Values: parallel float arrays (one sample per second or per record).
        Missing streams are simply absent from the dict.
        """

    async def download_fit_file(
        self, access_token: str, external_id: str
    ) -> bytes | None:
        """Download the raw FIT file for a single activity.

        Return None if the provider does not support FIT downloads.
        Providers that do support it should override this method.
        """
        return None

    async def fetch_zones(self, access_token: str) -> "ZoneData | None":
        """Fetch training zones and FTP from the provider.

        Return None if the provider does not support zone fetching.
        Providers that support it should override this method.
        """
        return None
