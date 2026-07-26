"""Schemas for the achievements API (issue #33).

The catalogue carries ids and tiers only — never display text. Names and
descriptions are i18n strings keyed by ``id`` in the web app, so the same
response serves every locale.
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class AchievementDefinition(BaseModel):
    """One catalogue entry, as the frontend needs it to render a badge."""

    id: str
    category: str
    tiers: list[float]
    # What the tier numbers mean: count, hours, km, metres, load, percent,
    # weeks, months.
    unit: str
    # Data dependency ("distance", "elevation", "load", "plan"); null when the
    # achievement is always reachable.
    requires: Optional[str] = None


class AchievementUnlockResponse(BaseModel):
    """An earned tier."""

    achievement_id: str
    tier: float
    # The day the criterion was actually met, derived from the athlete's
    # history — back-filling old rides moves this earlier, never to today.
    achieved_on: date
    # When we first noticed. Drives the "new" marker, not the badge's date.
    created_at: Optional[datetime] = None
    # Whether the athlete has already looked at this badge (POST /seen).
    seen: bool = False
    context: Optional[dict] = None

    model_config = {"from_attributes": True}


class StreakResponse(BaseModel):
    """Current and best run for one streak type."""

    id: str
    current: int
    longest: int
    # The current period hasn't qualified yet but the streak is still alive —
    # the week simply isn't over. Never render this as "broken".
    in_progress: bool = False


class AchievementsResponse(BaseModel):
    catalogue: list[AchievementDefinition]
    unlocked: list[AchievementUnlockResponse]
    # achievement_id → current value in the definition's unit, for "42 / 50"
    # progress rendering on locked badges.
    progress: dict[str, float]
    streaks: list[StreakResponse]
    # True when the athlete has opted out; the UI hides the feature entirely.
    disabled: bool = False
