"""Intensity distribution over a training block (issue #38).

Aggregates a window of cycling activities into the three intensity bands and
names the shape. Lives here rather than in the endpoint because the training
status analyzer and the plan generator feed the same numbers to the LLM — one
implementation, so the coach and the chart can never disagree.

Two counting methods, and the difference is methodological rather than
cosmetic. Accumulated time-in-zone makes nearly everyone look pyramidal:
warm-ups, recoveries and coast-downs dump time into band 1 no matter what the
session was *for*. Session counting takes each ride whole by its workout
category, which is how the polarization literature counts. Both ship; the
answer is only meaningful with the method stated next to it.

``compute_intensity_distribution`` **never writes**. One of its LLM callers
(``regenerate_plan`` → ``generate_plan_weeks_llm``) runs on a session carrying
flushed-but-uncommitted deletions of a plan's workouts, left uncommitted so an
LLM failure rolls them back; a commit here would make them permanent before the
LLM was even called. Backfilling snapshots is a separate, explicit call that only
the endpoint makes.
"""
from datetime import date, datetime, time, timedelta
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user_orm import Activity, Athlete
from backend.app.schemas.metrics import (
    IntensityBand,
    IntensityCoverage,
    IntensityDistributionResponse,
)
from backend.app.services.zone_times import ensure_zone_times
from openkoutsi.intensity_distribution import (
    BANDS,
    band_for_category,
    band_percentages,
    bands_from_zone_times,
    classify,
    sort_zone_names,
    zone_number,
)
from openkoutsi.sport_matching import CYCLING_SPORT_TYPES

# Twelve weeks — a training block, and the span over which "did this base phase
# come out polarized or pyramidal" is a question worth asking.
DEFAULT_WINDOW_DAYS = 84

# Ten years, matching the ``days`` bound on the endpoint. An explicit ``start``
# would otherwise be unbounded, and the window decides how many activity rows a
# single request pulls into memory.
MAX_WINDOW_DAYS = 3650


def _window_query(athlete: Athlete, start: Optional[date], end: Optional[date]):
    """Processed cycling activities in the window, for one athlete."""
    query = select(Activity).where(
        Activity.athlete_id == athlete.id,
        Activity.sport_type.in_(CYCLING_SPORT_TYPES),
        Activity.status == "processed",
        Activity.start_time.is_not(None),
    )
    if start:
        query = query.where(Activity.start_time >= datetime.combine(start, time.min))
    if end:
        query = query.where(Activity.start_time <= datetime.combine(end, time.max))
    return query


async def backfill_window_snapshots(
    athlete: Athlete,
    session: AsyncSession,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> int:
    """Freeze missing ``zone_times`` snapshots for a window, and commit.

    Split out of the read path deliberately. This writes and commits, so it may
    only be called where the transaction is owned — the endpoint — and never
    from a service that a caller with pending work might reach. See the module
    docstring.
    """
    activities = (await session.execute(_window_query(athlete, start, end))).scalars().all()
    updated = await ensure_zone_times(athlete, session, activities)
    if updated:
        await session.commit()
    return updated


def _zone_definitions_changed(
    athlete: Athlete,
    activities: Sequence[Activity],
    basis: Optional[str],
    start: Optional[date],
    end: Optional[date],
) -> bool:
    """Whether the zone definitions moved inside the window.

    ``zone_times`` is frozen at processing time using the zones in effect then
    (issue #27) — right for a weekly chart, but it means a 12-week aggregate can
    mix snapshots taken under different FTPs. Nothing records which FTP a given
    snapshot used, so this detects the two things that *are* observable:

    * the recorded FTP *changing value* inside the window — power zones are
      pinned to it, so the boundaries moved with it. A change in value, not the
      mere presence of a test entry: every profile save including an FTP appends
      one;
    * the same zone number carrying different names in different snapshots,
      meaning the zone list itself was replaced.

    Names are compared per zone number rather than as whole key sets, because a
    snapshot only holds the zones its ride touched — an easy ride and a hard one
    under identical zones legitimately have different keys.

    A pure boundary change that kept the names is not detectable from the
    snapshots alone; that is what the FTP-test signal is for. The flag is a
    "treat this as approximate" hint, not a proof of consistency.

    Returns ``False`` outright when there is no basis: session counting reads
    ``workout_category`` and never touches a zone boundary, so warning there
    would tell the athlete (and the LLM) to distrust a figure the change cannot
    have moved.
    """
    if basis is None:
        return False

    history: list[tuple[date, int]] = []
    for entry in athlete.ftp_tests or []:
        if not isinstance(entry, dict) or not entry.get("date"):
            continue
        # ``ftp_tests`` is a raw JSON column. A malformed value compared against
        # a real one would read as a change, so skip it exactly as an
        # unparseable date is skipped.
        ftp = entry.get("ftp")
        if not isinstance(ftp, int) or isinstance(ftp, bool):
            continue
        try:
            tested = date.fromisoformat(str(entry["date"])[:10])
        except ValueError:
            continue
        history.append((tested, ftp))

    previous: Optional[int] = None
    for tested, ftp in sorted(history, key=lambda item: item[0]):
        in_window = (start is None or tested >= start) and (end is None or tested <= end)
        if in_window and previous is not None and ftp != previous:
            return True
        previous = ftp

    names_by_number: dict[int, set[str]] = {}
    for activity in activities:
        times = (activity.zone_times or {}).get(basis) or {}
        for name in sort_zone_names(times):
            number = zone_number(name)
            if number is None:
                continue
            seen = names_by_number.setdefault(number, set())
            seen.add(name)
            if len(seen) > 1:
                return True
    return False


def _aggregate_time(
    activities: Sequence[Activity], basis: str
) -> tuple[dict[int, int], int]:
    """Seconds per band and the number of activities that contributed."""
    seconds = {band: 0 for band in BANDS}
    used = 0
    for activity in activities:
        totals = bands_from_zone_times(activity.zone_times, basis)
        if not any(totals.values()):
            continue
        for band in BANDS:
            seconds[band] += totals[band]
        used += 1
    return seconds, used


def _aggregate_sessions(
    activities: Sequence[Activity],
) -> tuple[dict[int, int], dict[int, int], int]:
    """Seconds and session counts per band, counting each ride whole."""
    seconds = {band: 0 for band in BANDS}
    sessions = {band: 0 for band in BANDS}
    used = 0
    for activity in activities:
        band = band_for_category(activity.workout_category)
        if band is None:
            continue
        sessions[band] += 1
        seconds[band] += int(activity.duration_s or 0)
        used += 1
    return seconds, sessions, used


async def compute_intensity_distribution(
    athlete: Athlete,
    session: AsyncSession,
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
    basis: Optional[str] = None,
    method: str = "time",
) -> IntensityDistributionResponse:
    """Build the distribution for one athlete over one window. Never writes.

    ``basis`` picks power or HR zones for ``method="time"``. When it is omitted
    the choice is resolved against the *data*, not the configuration: power
    zones being present says only that they were configured, and provider sync
    writes them for anyone with an FTP set. An athlete who rides on HR alone
    would otherwise land in the power branch and get an empty result — the
    exact athlete the fallback exists for. So the preferred basis is tried
    first and, if nothing in the window had that data, the other one is tried
    before giving up. The chosen basis is echoed back either way.

    ``basis`` is ignored (and echoed as ``None``) for ``method="session"``,
    which counts workout categories.
    """
    activities = (await session.execute(_window_query(athlete, start, end))).scalars().all()

    sessions = {band: 0 for band in BANDS}
    if method == "session":
        resolved_basis: Optional[str] = None
        seconds, sessions, used = _aggregate_sessions(activities)
    else:
        resolved_basis = basis or ("power" if athlete.power_zones else "hr")
        seconds, used = _aggregate_time(activities, resolved_basis)
        if basis is None and used == 0:
            fallback = "hr" if resolved_basis == "power" else "power"
            alt_seconds, alt_used = _aggregate_time(activities, fallback)
            if alt_used:
                seconds, used, resolved_basis = alt_seconds, alt_used, fallback

    # The session method counts sessions, not time — that is the whole point of
    # offering it, so the percentages have to follow the same unit.
    counted = sessions if method == "session" else seconds
    # Round once, then classify on the rounded values. This endpoint is meant to
    # be the single source of the shape, so a client recomputing it from the
    # percentages it was handed must not be able to disagree near a threshold.
    percentages = {
        band: round(pct, 1) for band, pct in band_percentages(counted).items()
    }

    return IntensityDistributionResponse(
        start=start,
        end=end,
        basis=resolved_basis,
        method=method,
        bands=[
            IntensityBand(
                band=band,
                seconds=seconds[band],
                pct=percentages[band],
                sessions=sessions[band] if method == "session" else None,
            )
            for band in BANDS
        ],
        classification=classify(*(percentages[band] for band in BANDS)),
        coverage=IntensityCoverage(
            activities_total=len(activities),
            activities_used=used,
            seconds_total=sum(seconds.values()),
        ),
        zone_definitions_changed=_zone_definitions_changed(
            athlete, activities, resolved_basis, start, end
        ),
    )


_SHAPE_SUMMARY = {
    "polarized": "polarized",
    "pyramidal": "pyramidal",
    "threshold": "threshold-heavy (a lot of grey-zone work)",
    "predominantly_low": "almost all low intensity",
}


def summarize_for_prompt(distribution: IntensityDistributionResponse) -> Optional[str]:
    """One-line summary of the distribution for an LLM prompt.

    Returns ``None`` when there is nothing to say, so callers can drop the line
    entirely rather than feeding the model an empty distribution to reason from.
    """
    if distribution.classification is None:
        return None

    bands = {band.band: band for band in distribution.bands}
    shape = _SHAPE_SUMMARY.get(distribution.classification, distribution.classification)
    summary = (
        f"{shape} — {bands[1].pct:.0f}% easy (below LT1), "
        f"{bands[2].pct:.0f}% tempo/threshold, {bands[3].pct:.0f}% hard (above LT2), "
        "by time in zones"
    )
    if distribution.zone_definitions_changed:
        summary += "; zones or FTP changed during this window, so treat the split as approximate"
    return summary


def resolve_window(
    start: Optional[date],
    end: Optional[date],
    days: Optional[int],
) -> tuple[date, Optional[date]]:
    """Apply the default block length, matching the sibling metrics endpoints.

    ``days`` only takes effect when no explicit ``start`` was given, as on
    ``/metrics/fitness`` and ``/metrics/zones/weekly``. Unlike those, an
    entirely unspecified window is not "all of history" — a distribution over
    every ride ever recorded answers nothing, so it defaults to one block, and
    an explicit ``start`` is clamped to ``MAX_WINDOW_DAYS`` so one request can't
    be made to load a decade of activities.

    Raises ``ValueError`` when the window is inverted; the caller turns that
    into a 422.
    """
    if start is not None and end is not None and start > end:
        raise ValueError("start must not be after end")

    if start is None:
        start = (end or date.today()) - timedelta(days=days or DEFAULT_WINDOW_DAYS)

    earliest = (end or date.today()) - timedelta(days=MAX_WINDOW_DAYS)
    if start < earliest:
        start = earliest
    return start, end
