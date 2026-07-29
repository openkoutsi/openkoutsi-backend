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


def _zone_definitions_changed(
    athlete: Athlete,
    activities: Sequence[Activity],
    basis: Optional[str],
    start: Optional[date],
    end: Optional[date],
) -> bool:
    """Whether the zone definitions moved inside the window.

    ``zone_times`` is frozen at processing time using the zones in effect then
    (issue #27), which is exactly right for a weekly chart but means a 12-week
    aggregate can mix snapshots taken under different FTPs. Nothing records
    which FTP a given snapshot used, so this detects the two things that *are*
    observable:

    * the recorded FTP *changing value* inside the window — power zones are
      pinned to it, so the boundaries moved with it. It has to be a change in
      value: every profile save that includes an FTP appends a test entry, so
      the mere presence of one means nothing;
    * the same zone number carrying different names in different snapshots,
      which means the zone list itself was replaced.

    Names are compared per zone number rather than as whole key sets, because a
    snapshot only holds the zones its ride touched: an easy ride and a hard one
    under identical zones legitimately have different keys, and treating that
    as a change would raise the warning on almost every window.

    A pure boundary change that kept the names is not detectable from the
    snapshots alone; that is what the FTP-test signal is for. The flag is
    therefore a "treat this as approximate" hint, not a proof of consistency.
    """
    history: list[tuple[date, Optional[int]]] = []
    for entry in athlete.ftp_tests or []:
        if not isinstance(entry, dict) or not entry.get("date"):
            continue
        try:
            tested = date.fromisoformat(str(entry["date"])[:10])
        except ValueError:
            continue
        history.append((tested, entry.get("ftp")))

    previous: Optional[int] = None
    for tested, ftp in sorted(history, key=lambda item: item[0]):
        in_window = (start is None or tested >= start) and (end is None or tested <= end)
        if in_window and previous is not None and ftp != previous:
            return True
        previous = ftp

    if basis is None:
        return False

    names_by_number: dict[int, set[str]] = {}
    for activity in activities:
        times = (activity.zone_times or {}).get(basis) or {}
        for position, name in enumerate(sort_zone_names(times)):
            seen = names_by_number.setdefault(zone_number(name, position + 1), set())
            seen.add(name)
            if len(seen) > 1:
                return True
    return False


async def compute_intensity_distribution(
    athlete: Athlete,
    session: AsyncSession,
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
    basis: Optional[str] = None,
    method: str = "time",
) -> IntensityDistributionResponse:
    """Build the distribution for one athlete over one window.

    ``basis`` picks power or HR zones for ``method="time"``; when omitted it
    prefers power and falls back to HR, so an athlete without a power meter
    doesn't lose the feature. It is ignored (and echoed as ``None``) for
    ``method="session"``, which counts workout categories.
    """
    if method == "session":
        resolved_basis: Optional[str] = None
    elif basis is not None:
        resolved_basis = basis
    else:
        resolved_basis = "power" if athlete.power_zones else "hr"

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

    activities = (await session.execute(query)).scalars().all()

    if method == "time" and await ensure_zone_times(athlete, session, activities):
        await session.commit()

    seconds = {band: 0 for band in BANDS}
    sessions = {band: 0 for band in BANDS}
    used = 0

    for activity in activities:
        if method == "session":
            band = band_for_category(activity.workout_category)
            if band is None:
                continue
            sessions[band] += 1
            seconds[band] += int(activity.duration_s or 0)
            used += 1
            continue

        totals = bands_from_zone_times(activity.zone_times, resolved_basis)
        if not any(totals.values()):
            continue
        for band in BANDS:
            seconds[band] += totals[band]
        used += 1

    # The session method counts sessions, not time — that is the whole point of
    # offering it, so the percentages have to follow the same unit.
    counted = sessions if method == "session" else seconds
    percentages = band_percentages(counted)

    return IntensityDistributionResponse(
        start=start,
        end=end,
        basis=resolved_basis,
        method=method,
        bands=[
            IntensityBand(
                band=band,
                seconds=seconds[band],
                pct=round(percentages[band], 1),
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
) -> tuple[Optional[date], Optional[date]]:
    """Apply the default block length, matching the sibling metrics endpoints.

    ``days`` only takes effect when no explicit ``start`` was given, as on
    ``/metrics/fitness`` and ``/metrics/zones/weekly``. Unlike those, an
    entirely unspecified window is not "all of history" — a distribution over
    every ride ever recorded answers nothing, so it defaults to one block.
    """
    if start is None:
        start = date.today() - timedelta(days=days or DEFAULT_WINDOW_DAYS)
    return start, end
