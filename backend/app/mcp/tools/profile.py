"""``get_athlete_profile`` — who the athlete is, not what they did (issue #42).

Every other tool here answers a question about training that happened: rides,
load, plans, goals, time in zone. This one answers the question those answers
are read *against*.

Two gaps make it worth its own call rather than more fields bolted onto
``get_training_status``.

**The zone boundaries.** ``get_zone_totals`` and ``get_intensity_distribution``
report time spent in Z1…Z7, and until now nothing said what Z4 *is* for this
athlete. A model handed "4 h 12 in Z2" and no boundaries either says nothing
useful about it or invents the wattage — and an invented threshold is worse than
a refusal, because everything downstream inherits it.

**The constraints.** How many hours a week the athlete actually has, and how
they have asked to be spoken to, are facts a coach uses in every answer and that
no amount of activity data reveals.

What this tool is deliberately **not** is a profile dump. ``athlete:export`` is
excluded from the callable scopes (:mod:`backend.app.mcp.registry`) precisely
because returning the whole record in one call is the opposite of task-shaped,
and this tool must not become that by another door. So: no name, no date of
birth (an age in years is what a coach uses, and it is far less identifying),
no avatar, no FTP-test or weight history, no feature toggles, and none of the
BYOK model configuration. Nothing here is data the athlete would be surprised to
find had travelled to the assistant they connected.

The five physiology fields overlap ``get_training_status``'s ``athlete`` block,
which stays as it is: a Form number returned without its frame invites the model
to invent the frame, and one round trip is cheaper than making it ask twice for
the numbers it always needs.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field

from backend.app.core.timezones import resolve_zone
from backend.app.mcp.dispatch import ToolRun
from backend.app.mcp.registry import ToolArgs, tool
from backend.app.mcp.shaping import int_or_none, round_or_none
from backend.app.services.athlete_experience import coaching_style, experience_level

#: The top power zone is stored with an upper bound of 9999 standing in for
#: "no ceiling" (see ``_MAX_ZONE_BOUND`` in ``backend/app/schemas/athlete.py``).
#: Passed through as a number it becomes a 9999 W zone ceiling a model will
#: quote back, so it is reported as an absent bound instead.
OPEN_ENDED_ABOVE = 9999

#: Zone lists are validated to exactly seven and five entries on write, but rows
#: predating that model can carry other counts and readers are expected to
#: degrade rather than assume (``openkoutsi/zones.py``). The cap bounds the
#: response against a legacy or hand-edited list.
MAX_ZONES = 12


class AthleteProfileArgs(ToolArgs):
    """No arguments. There is one profile, and it is small enough to return whole."""


class PowerZone(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Zone label, e.g. 'Z4 Threshold'. Positional: Z1 is recovery and "
            "the last zone is maximal."
        ),
    )
    low_w: int = Field(..., description="Lower bound of the zone, in watts (W).")
    high_w: Optional[int] = Field(
        None,
        description=(
            "Upper bound of the zone, in watts (W). Null on the top zone, which "
            "is open-ended — it means no ceiling, not an unknown one."
        ),
    )


class HrZone(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Zone label, e.g. 'Z4 Threshold'. Positional: Z1 is recovery and "
            "the last zone is maximal."
        ),
    )
    low_bpm: int = Field(
        ..., description="Lower bound of the zone, in beats per minute (bpm)."
    )
    high_bpm: Optional[int] = Field(
        None,
        description=(
            "Upper bound of the zone, in beats per minute (bpm). Null when the "
            "zone is open-ended at the top."
        ),
    )


class AthleteProfile(BaseModel):
    ftp_w: Optional[int] = Field(
        None,
        description=(
            "Functional threshold power on the profile, in watts (W). Null when "
            "never set — that is an unconfigured profile, not a weak athlete."
        ),
    )
    max_hr_bpm: Optional[int] = Field(
        None, description="Maximum heart rate on the profile, in beats per minute (bpm)."
    )
    resting_hr_bpm: Optional[int] = Field(
        None, description="Resting heart rate on the profile, in beats per minute (bpm)."
    )
    weight_kg: Optional[float] = Field(
        None,
        description=(
            "Current bodyweight on the profile, in kilograms (kg). A snapshot: "
            "no weight history is published here."
        ),
    )
    age_years: Optional[int] = Field(
        None,
        description=(
            "Age in whole years today. Derived from a stored date of birth, "
            "which is never returned. Null when the athlete has not given one."
        ),
    )
    experience_level: Optional[str] = Field(
        None,
        description=(
            "Self-reported experience: novice, intermediate, experienced, "
            "semi-pro or elite. Null when the athlete has not said. Tailor load "
            "and detail to it."
        ),
    )
    coaching_style: Optional[str] = Field(
        None,
        description=(
            "How the athlete asked to be spoken to: stern, friendly or "
            "encouraging. Null when they have no preference. It shapes tone, "
            "never the training judgement."
        ),
    )
    timezone: Optional[str] = Field(
        None,
        description=(
            "The athlete's IANA timezone, e.g. 'Europe/Helsinki'. Null when "
            "unset or unrecognised, in which case their local date is unknown "
            "and dates in these tools are the server's."
        ),
    )
    weekly_hours_low: Optional[float] = Field(
        None,
        description=(
            "Bottom of the training time the athlete says they have, in hours "
            "per week. Null when they have not said — that is unknown, not zero."
        ),
    )
    weekly_hours_high: Optional[float] = Field(
        None,
        description=(
            "Top of the training time the athlete says they have, in hours per "
            "week. Null when they have not said."
        ),
    )
    power_zones: list[PowerZone] = Field(
        default_factory=list,
        description=(
            "Power zone boundaries, ascending from Z1. Empty when the athlete "
            "has never configured them. These are what the zone figures from "
            "get_zone_totals and get_intensity_distribution are measured "
            "against — read them together rather than assuming standard bands."
        ),
    )
    hr_zones: list[HrZone] = Field(
        default_factory=list,
        description=(
            "Heart-rate zone boundaries, ascending from Z1. Empty when the "
            "athlete has never configured them."
        ),
    )


def _age_on(born: Optional[date], today: date) -> Optional[int]:
    """Whole years between ``born`` and ``today``.

    A date of birth in the future is a typo rather than a negative age, and
    reporting one would be worse than reporting nothing.
    """
    if born is None or born > today:
        return None
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _bounds(entry: Any) -> Optional[tuple[str, int, Optional[int]]]:
    """``(name, low, high)`` from one stored zone, or ``None`` if unusable.

    The lists are JSON columns, so a malformed entry is a real possibility.
    Skipping one is better than failing the whole call: the remaining zones
    still tell a model most of what it needed.
    """
    if not isinstance(entry, dict):
        return None
    try:
        low = int(entry["low"])
        high = int(entry["high"])
    except (KeyError, TypeError, ValueError):
        return None
    name = str(entry.get("name") or "")
    return name, low, (None if high >= OPEN_ENDED_ABOVE else high)


def _hours(app_settings: Optional[dict], key: str) -> Optional[float]:
    """One weekly-hours endpoint, or ``None`` when unset or not a number."""
    if not isinstance(app_settings, dict):
        return None
    try:
        return round_or_none(float(app_settings[key]), 1)
    except (KeyError, TypeError, ValueError):
        return None


@tool(
    name="get_athlete_profile",
    title="Athlete profile",
    scopes={"athlete:read"},
    arguments=AthleteProfileArgs,
    returns=AthleteProfile,
)
async def get_athlete_profile(run: ToolRun, args: AthleteProfileArgs) -> AthleteProfile:
    """The athlete's own settings: FTP, maximum and resting heart rate, weight,
    age and experience level, their power and heart-rate zone boundaries, how
    many hours a week they have, and the coaching tone they asked for.

    Call this when you need to *interpret* another tool's numbers rather than
    fetch more of them. Time in Z2 means nothing without knowing what Z2 is in
    watts for this athlete, and a training week only reads as heavy or light
    against the hours they said they had.

    It takes no arguments and describes the profile as it stands today: there is
    no history here, so a null is a setting the athlete never filled in rather
    than a value that has lapsed. Their name and date of birth are deliberately
    not returned — an age in years is what a coach uses.

    get_training_status already returns FTP, heart rates, weight and experience
    level alongside the fitness figures, so if you have called that you need
    this one only for the zones, the available hours or the coaching style.
    """
    athlete = run.athlete
    settings = athlete.app_settings

    power_zones = []
    for entry in (athlete.power_zones or [])[:MAX_ZONES]:
        parsed = _bounds(entry)
        if parsed is not None:
            name, low, high = parsed
            power_zones.append(PowerZone(name=name, low_w=low, high_w=high))

    hr_zones = []
    for entry in (athlete.hr_zones or [])[:MAX_ZONES]:
        parsed = _bounds(entry)
        if parsed is not None:
            name, low, high = parsed
            hr_zones.append(HrZone(name=name, low_bpm=low, high_bpm=high))

    # The stored timezone is unvalidated free text; report it only when it names
    # a zone that actually resolves, since a client reading it will use it to
    # work out the athlete's local date.
    timezone = (settings or {}).get("timezone") if isinstance(settings, dict) else None
    timezone = str(timezone) if resolve_zone(timezone) else None

    return AthleteProfile(
        ftp_w=athlete.ftp,
        max_hr_bpm=int_or_none(athlete.max_hr),
        resting_hr_bpm=int_or_none(athlete.resting_hr),
        weight_kg=round_or_none(athlete.weight_kg, 1),
        age_years=_age_on(athlete.date_of_birth, run.today),
        experience_level=experience_level(settings),
        coaching_style=coaching_style(settings),
        timezone=timezone,
        weekly_hours_low=_hours(settings, "weekly_hours_min"),
        weekly_hours_high=_hours(settings, "weekly_hours_max"),
        power_zones=power_zones,
        hr_zones=hr_zones,
    )
