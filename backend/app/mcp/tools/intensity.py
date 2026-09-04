"""Distribution tools: ``get_intensity_distribution`` and ``get_zone_totals``.

Two views of the same thing at different altitudes. ``get_zone_totals`` answers
"what did the last few weeks hold", week by week, in the athlete's own zone
names; ``get_intensity_distribution`` collapses a block into three bands and
*names the shape*, which is the level a coaching decision is made at.

**Neither writes.** Both underlying paths can freeze missing ``zone_times``
snapshots, and both are asked *not* to here: freezing is permanent, so whichever
caller triggers it decides forever which zone definitions an old ride is judged
by, and it would make ``readOnlyHint`` a lie in the tool descriptor. Rides with
no snapshot are *reported* rather than fixed.

Both carry their caveats in the payload: ``coverage`` travels with the numbers,
a window in which the athlete moved their FTP is flagged, and the method used is
always reported since the time and session methods disagree by design.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.app.mcp.dispatch import ToolRun
from backend.app.mcp.registry import ToolArgs, tool
from backend.app.mcp.shaping import pct, week_start
from backend.app.services.intensity_distribution import (
    compute_intensity_distribution,
    resolve_window,
)
from backend.app.services.zone_times import weekly_zone_buckets

#: Weeks returned by ``get_zone_totals``. Twenty-six weeks of five power zones
#: and five HR zones is already a large table for a model to read; beyond that
#: the block view (``get_intensity_distribution``) is the right tool.
MAX_WEEKS = 26

_SHAPE_MEANING = {
    "polarized": "mostly easy riding with the hard work genuinely hard, and little in between",
    "pyramidal": "an easy base, less tempo and threshold, least hard work",
    "threshold": "a large share of moderate, grey-zone work",
    "predominantly_low": "almost entirely low intensity, with very little above LT1",
}


class IntensityArgs(ToolArgs):
    days: int = Field(
        84,
        ge=14,
        le=365,
        description=(
            "Length of the block to summarise, in days. 84 (twelve weeks) is the "
            "default training block. Shorter than about four weeks and the shape "
            "is noise rather than a pattern."
        ),
    )
    method: Literal["time", "session"] = Field(
        "time",
        description=(
            "'time' sums seconds spent in each band; 'session' counts each ride "
            "whole by its category. They disagree by design — warm-ups pull the "
            "time method toward pyramidal — so pick one deliberately and quote "
            "which you used."
        ),
    )
    basis: Optional[Literal["power", "hr"]] = Field(
        None,
        description=(
            "Whether to band by power or heart-rate zones. Omit to prefer power "
            "and fall back to HR when the window has no power data. Ignored for "
            "method='session'."
        ),
    )


class Band(BaseModel):
    band: int = Field(..., description="1 = below LT1 (easy), 2 = LT1–LT2 (tempo/threshold), 3 = above LT2 (hard).")
    seconds: int = Field(0, description="Time spent in this band over the window, in seconds (s).")
    pct: float = Field(
        0.0,
        description=(
            "This band's share as a percentage (%) — of time for method='time', "
            "of sessions for method='session'."
        ),
    )
    sessions: Optional[int] = Field(
        None, description="Rides counted into this band (count). Only set for method='session'."
    )


class Coverage(BaseModel):
    activities_total: int = Field(0, description="Rides in the window (count).")
    activities_used: int = Field(
        0, description="Rides that had usable zone data and reached the distribution (count)."
    )
    seconds_total: int = Field(0, description="Total banded time over the window, in seconds (s).")
    coverage_pct: float = Field(
        0.0,
        description=(
            "Share of rides that reached the distribution, as a percentage (%). "
            "Below about 60% the shape is drawn from too little to argue from."
        ),
    )


class IntensityDistribution(BaseModel):
    start: date = Field(..., description="First calendar date of the window.")
    end: date = Field(..., description="Last calendar date of the window.")
    method: str = Field(..., description="'time' or 'session' — quote this whenever you quote the numbers.")
    basis: Optional[str] = Field(
        None, description="'power' or 'hr', whichever the bands were drawn from. Null for method='session'."
    )
    bands: list[Band] = Field(default_factory=list, description="The three intensity bands, in order.")
    shape: Optional[str] = Field(
        None,
        description=(
            "polarized, pyramidal, threshold or predominantly_low. Null when the "
            "window holds nothing usable — which is a finding, not an error."
        ),
    )
    shape_meaning: Optional[str] = Field(
        None, description="What that shape name means, in words, so it need not be inferred."
    )
    coverage: Coverage = Field(..., description="How much of the window actually reached the result.")
    zone_definitions_changed: bool = Field(
        False,
        description=(
            "True when the athlete's zones or FTP moved inside this window, so "
            "the band boundaries are not consistent across it. Say so if you "
            "quote the numbers."
        ),
    )


class ZoneTotalsArgs(ToolArgs):
    weeks: int = Field(
        8,
        ge=1,
        le=MAX_WEEKS,
        description="How many recent Monday-based weeks to return (count), most recent last.",
    )
    basis: Literal["power", "hr", "both"] = Field(
        "both", description="Return power zones, heart-rate zones, or both."
    )


class WeekZones(BaseModel):
    week_start: date = Field(..., description="Monday the week began on.")
    total_s: int = Field(0, description="Total zoned time that week, in seconds (s).")
    power: dict[str, int] = Field(
        default_factory=dict,
        description="Seconds (s) in each power zone, keyed by the athlete's own zone names.",
    )
    hr: dict[str, int] = Field(
        default_factory=dict, description="Seconds (s) in each heart-rate zone, keyed the same way."
    )


class ZoneTotals(BaseModel):
    weeks: list[WeekZones] = Field(default_factory=list, description="One entry per week, oldest first.")
    returned: int = Field(0, description="How many weeks are in this response (count).")
    weeks_requested: int = Field(0, description="How many weeks were asked for (count).")
    activities_without_zone_data: int = Field(
        0,
        description=(
            "Rides in the window with no stored time-in-zone, so not counted "
            "into any week (count). They predate zone snapshots rather than "
            "being easy — read a non-zero figure here as under-counted totals."
        ),
    )
    note: Optional[str] = Field(
        None,
        description=(
            "Set when weeks are absent because nothing was ridden in them, or "
            "when rides were left out for want of zone data — so a gap is not "
            "mistaken for an easy week."
        ),
    )


@tool(
    name="get_intensity_distribution",
    title="Intensity distribution",
    scopes={"metrics:read"},
    arguments=IntensityArgs,
    returns=IntensityDistribution,
)
async def get_intensity_distribution(run: ToolRun, args: IntensityArgs) -> IntensityDistribution:
    """How a training block was balanced across three intensity bands — below
    LT1, between LT1 and LT2, above LT2 — and what shape that makes: polarized,
    pyramidal, threshold-heavy, or predominantly low.

    This is the tool for "is this athlete training the right way", as opposed to
    "how much are they training". A rider stuck in the grey zone can look
    perfectly consistent in Load terms while making no progress, and only the
    distribution shows it.

    Read 'coverage' before quoting the shape: computed from six rides out of
    forty it is not evidence. Read 'zone_definitions_changed' too — if the
    athlete moved their FTP mid-window the boundaries shifted underfoot. And
    always say which 'method' you used, since time and session counting disagree
    by design.
    """
    end = run.today
    start, _ = resolve_window(None, end, args.days)

    # No `backfill_window_snapshots` here, deliberately — see the module
    # docstring. `compute_intensity_distribution` never writes, and rides with
    # no frozen snapshot already show up honestly in `coverage`.
    result = await compute_intensity_distribution(
        run.athlete, run.session, start=start, end=end, basis=args.basis, method=args.method
    )

    coverage = result.coverage
    return IntensityDistribution(
        start=result.start or start,
        end=result.end or end,
        method=result.method,
        basis=result.basis,
        bands=[
            Band(band=b.band, seconds=b.seconds, pct=b.pct, sessions=b.sessions)
            for b in result.bands
        ],
        shape=result.classification,
        shape_meaning=_SHAPE_MEANING.get(result.classification or ""),
        coverage=Coverage(
            activities_total=coverage.activities_total,
            activities_used=coverage.activities_used,
            seconds_total=coverage.seconds_total,
            coverage_pct=pct(coverage.activities_used, coverage.activities_total),
        ),
        zone_definitions_changed=result.zone_definitions_changed,
    )


@tool(
    name="get_zone_totals",
    title="Weekly time in zones",
    scopes={"metrics:read"},
    arguments=ZoneTotalsArgs,
    returns=ZoneTotals,
)
async def get_zone_totals(run: ToolRun, args: ZoneTotalsArgs) -> ZoneTotals:
    """Accumulated time in each power and heart-rate zone, week by week
    (Monday-based), for the requested number of recent weeks.

    Use this when the block-level shape from get_intensity_distribution needs
    unpacking — which weeks carried the hard work, whether a recovery week was
    actually easy, whether volume is climbing or just moving around.

    Each ride's time-in-zone is a snapshot frozen when it was processed, using
    the zones in effect then. Editing zones therefore changes future rides only,
    and past weeks stay as they were ridden — so a boundary you see here may not
    be the athlete's current one.

    A week with no entry is a week with nothing recorded, not missing data.
    Rides that predate zone snapshots are counted in
    'activities_without_zone_data' rather than silently omitted — a non-zero
    figure there means the weekly totals are under-counted, not that the athlete
    was taking it easy.
    """
    start = week_start(run.today) - timedelta(weeks=args.weeks - 1)

    buckets, unsnapshotted = await weekly_zone_buckets(
        run.athlete, run.session, start=start, end=run.today, backfill=False
    )

    weeks: list[WeekZones] = []
    for bucket in buckets:
        power = bucket["power"] if args.basis in ("power", "both") else {}
        hr = bucket["hr"] if args.basis in ("hr", "both") else {}
        weeks.append(
            WeekZones(
                week_start=bucket["week_start"],
                total_s=sum(power.values()) + sum(hr.values()),
                power=power,
                hr=hr,
            )
        )

    notes = []
    if len(weeks) < args.weeks:
        notes.append(
            f"{args.weeks - len(weeks)} of the {args.weeks} weeks asked for hold "
            "no recorded riding and are omitted."
        )
    if unsnapshotted:
        notes.append(
            f"{unsnapshotted} ride(s) in this window have no stored time-in-zone "
            "and are not counted — they predate zone snapshots. Opening the "
            "weekly zones view in the web app fills them in."
        )

    return ZoneTotals(
        weeks=weeks,
        returned=len(weeks),
        weeks_requested=args.weeks,
        activities_without_zone_data=unsnapshotted,
        note=" ".join(notes) or None,
    )
