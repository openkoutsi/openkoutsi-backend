"""Achievement catalogue and streak math (issue #33).

The catalogue lives here as *data*: an achievement is an id, a category and an
ascending list of tiers. There are deliberately **no human-readable strings** —
names and descriptions are i18n keys resolved by the frontend — so adding one
never means putting English in the database.

Only *unlocks* are persisted. Everything here is pure (dates and primitive
numbers in, tier→date maps out); the DB orchestration lives in
``backend.app.services.achievements``.

Two rules shape the design:

1. **Weekly granularity, never daily.** A streak is a run of consecutive
   Monday-based weeks. Daily streaks are deliberately not offered — forcing a
   ride every single day over long periods is not a healthy thing to reward.
2. **Achievements are earned by history, not by when they were computed.** Every
   helper returns the date the criterion was *actually* first met, so
   back-filling an old ride moves the unlock date earlier instead of stamping it
   with today.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Optional

# ── Streak thresholds ────────────────────────────────────────────────────────
# What makes a week "qualify" for each streak type. Finalised design decisions,
# intentionally hard-coded — there is no configuration surface.
STREAK_VOLUME_HOURS = 5.0        # weeks with ≥5 h of activity
STREAK_DISTANCE_KM = 100.0       # weeks with ≥100 km covered
STREAK_CLIMBING_M = 1000.0       # weeks with ≥1000 m climbed
STREAK_MULTISPORT_SPORTS = 2     # weeks touching ≥2 distinct sports

# A "comeback" is the first activity after a gap of at least this long.
COMEBACK_GAP_DAYS = 30

# Height of Everest, in metres — the tier that gives `everesting` its name.
EVEREST_M = 8848


@dataclass(frozen=True)
class AchievementDef:
    """One achievement: a stable id, a category, and ascending tiers.

    ``unit`` says what the tier numbers mean (``count``, ``hours``, ``km``,
    ``metres``, ``load``, ``percent``, ``weeks``, ``months``) so the frontend can
    format them without a lookup table of its own.

    ``requires`` names a data dependency the athlete may never have — elevation is
    FIT-only, Load needs power or HR. Badges whose requirement is unmet are hidden
    rather than shown permanently locked.

    ``threshold``/``threshold_unit`` say what makes a single *period* qualify for
    a streak (5 hours a week, 100 km a week), as opposed to ``tiers``, which count
    the qualifying periods. Set from the ``STREAK_*`` constants rather than
    retyped, and served over the API so the UI has no copy of the number that a
    constant change would silently invalidate.
    """

    id: str
    category: str
    tiers: tuple[float, ...]
    unit: str
    requires: Optional[str] = None
    threshold: Optional[float] = None
    threshold_unit: Optional[str] = None


CATALOGUE: tuple[AchievementDef, ...] = (
    # ── Volume ───────────────────────────────────────────────────────────────
    AchievementDef("activity_count", "volume", (1, 10, 50, 100, 250, 500, 1000), "count"),
    AchievementDef("long_activity", "volume", (2, 3, 4, 5, 6), "hours"),
    AchievementDef("total_hours", "volume", (10, 50, 100, 500, 1000), "hours"),
    AchievementDef(
        "single_ride_distance", "volume", (50, 100, 160, 200, 300), "km",
        requires="distance",
    ),
    # Top tier is the equatorial circumference of the Earth — ride around the world.
    AchievementDef(
        "total_distance", "volume", (100, 1000, 5000, 10000, 25000, 40075), "km",
        requires="distance",
    ),
    AchievementDef(
        "total_load", "volume", (1000, 10000, 50000, 100000), "load", requires="load",
    ),
    # ── Climbing ─────────────────────────────────────────────────────────────
    AchievementDef(
        "single_ride_elevation", "climbing", (1000, 2000, 3000, 5000), "metres",
        requires="elevation",
    ),
    AchievementDef(
        "total_elevation", "climbing", (EVEREST_M, 50000, 100000, 500000), "metres",
        requires="elevation",
    ),
    # Everest's height climbed in one single activity.
    AchievementDef(
        "everesting", "climbing", (EVEREST_M,), "metres", requires="elevation",
    ),
    # ── Variety & character ──────────────────────────────────────────────────
    AchievementDef("multisport", "variety", (2, 3, 5), "count"),
    AchievementDef("race_day", "variety", (1, 5, 10, 25), "count"),
    AchievementDef("commuter", "variety", (10, 50, 100, 250), "count"),
    # Coming back after a long break is worth celebrating, not penalising.
    AchievementDef("comeback", "variety", (1,), "count"),
    # ── Engagement ───────────────────────────────────────────────────────────
    AchievementDef("rpe_recorded", "engagement", (10, 50, 100), "count"),
    AchievementDef("notes_written", "engagement", (10, 50, 100), "count"),
    # ── Plans & goals ────────────────────────────────────────────────────────
    AchievementDef("plans_completed", "plan", (1, 3, 5, 10), "count", requires="plan"),
    AchievementDef("plan_flawless", "plan", (1,), "count", requires="plan"),
    AchievementDef("plan_adherence", "plan", (80, 90, 95), "percent", requires="plan"),
    AchievementDef("goals_reached", "goal", (1, 5, 10), "count"),
    # ── Streaks (weekly and coarser — never daily) ────────────────────────────
    # `threshold` comes straight from the constants above, so the rule the API
    # advertises and the rule `qualifies_*` enforces cannot drift apart.
    # The two "active" streaks need only a single activity, so they have none.
    AchievementDef("streak_active_weeks", "streak", (4, 8, 12, 26, 52), "weeks"),
    AchievementDef(
        "streak_volume_weeks", "streak", (4, 8, 12), "weeks",
        threshold=STREAK_VOLUME_HOURS, threshold_unit="hours",
    ),
    AchievementDef(
        "streak_multisport_weeks", "streak", (4, 8, 12), "weeks",
        threshold=STREAK_MULTISPORT_SPORTS, threshold_unit="sports",
    ),
    AchievementDef(
        "streak_distance_weeks", "streak", (4, 8, 12), "weeks", requires="distance",
        threshold=STREAK_DISTANCE_KM, threshold_unit="km",
    ),
    AchievementDef(
        "streak_climbing_weeks", "streak", (4, 8, 12), "weeks", requires="elevation",
        threshold=STREAK_CLIMBING_M, threshold_unit="metres",
    ),
    AchievementDef("streak_active_months", "streak", (3, 6, 12, 24), "months"),
)

CATALOGUE_BY_ID: dict[str, AchievementDef] = {d.id: d for d in CATALOGUE}


# ── Facts ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActivityFact:
    """One activity, reduced to just what the achievement rules need.

    ``day`` is the activity's date *in the athlete's local timezone* — the caller
    converts, so week boundaries line up with the athlete's own calendar rather
    than UTC's.
    """

    day: date
    duration_s: int = 0
    distance_m: float = 0.0
    elevation_m: float = 0.0
    load: float = 0.0
    sport: Optional[str] = None  # canonical category, e.g. "cycling"
    has_rpe: bool = False
    has_notes: bool = False
    labels: tuple[str, ...] = ()
    # Carried so a single-activity badge (longest ride, biggest climb) can link
    # back to the ride that earned it.
    activity_id: Optional[str] = None


@dataclass(frozen=True)
class PeriodBucket:
    """Activity totals for one week (or month), used by the streak rules."""

    start: date
    seconds: int = 0
    metres: float = 0.0
    elevation_m: float = 0.0
    sports: frozenset[str] = frozenset()
    count: int = 0


def week_start(day: date) -> date:
    """Monday of the week containing *day*.

    Monday-based to match ``GET /api/metrics/zones/weekly``, so "this week" means
    the same thing everywhere in the product.
    """
    return day - timedelta(days=day.weekday())


def month_start(day: date) -> date:
    return day.replace(day=1)


def _month_end(start: date) -> date:
    if start.month == 12:
        return start.replace(day=31)
    return start.replace(month=start.month + 1, day=1) - timedelta(days=1)


def _next_month(start: date) -> date:
    if start.month == 12:
        return start.replace(year=start.year + 1, month=1)
    return start.replace(month=start.month + 1)


def bucket_by(facts: Iterable[ActivityFact], *, monthly: bool = False) -> list[PeriodBucket]:
    """Aggregate facts into ordered week (or month) buckets.

    Only periods that actually contain activity get a bucket; the gaps are what
    break streaks, and :func:`streak_tier_dates` walks the calendar to find them.
    """
    key = month_start if monthly else week_start
    acc: dict[date, dict] = {}
    for f in facts:
        k = key(f.day)
        b = acc.setdefault(
            k, {"seconds": 0, "metres": 0.0, "elevation_m": 0.0, "sports": set(), "count": 0}
        )
        b["seconds"] += f.duration_s or 0
        b["metres"] += f.distance_m or 0.0
        b["elevation_m"] += f.elevation_m or 0.0
        b["count"] += 1
        if f.sport:
            b["sports"].add(f.sport)
    return [
        PeriodBucket(
            start=k,
            seconds=v["seconds"],
            metres=v["metres"],
            elevation_m=v["elevation_m"],
            sports=frozenset(v["sports"]),
            count=v["count"],
        )
        for k, v in sorted(acc.items())
    ]


# ── Generic tier helpers ─────────────────────────────────────────────────────


def cumulative_tier_dates(
    events: Iterable[tuple[date, float]], tiers: Iterable[float]
) -> dict[float, date]:
    """First date a running total reaches each tier.

    *events* are ``(day, increment)`` pairs; they are sorted here, so callers
    don't have to. Used for every "n of something, ever" achievement — activity
    count, lifetime hours, distance, Load, RPEs recorded, notes written.
    """
    reached: dict[float, date] = {}
    tier_list = sorted(tiers)
    total = 0.0
    for day, increment in sorted(events, key=lambda e: e[0]):
        total += increment
        for tier in tier_list:
            if tier not in reached and total >= tier:
                reached[tier] = day
    return reached


def threshold_tier_dates(
    events: Iterable[tuple[date, float]], tiers: Iterable[float]
) -> dict[float, date]:
    """First date any single value reaches each tier.

    For "best ever" achievements — the longest ride, the biggest climb — where
    one activity has to clear the bar on its own rather than accumulate to it.
    """
    reached: dict[float, date] = {}
    tier_list = sorted(tiers)
    for day, value in sorted(events, key=lambda e: e[0]):
        for tier in tier_list:
            if tier not in reached and value >= tier:
                reached[tier] = day
    return reached


def distinct_tier_dates(
    events: Iterable[tuple[date, Optional[str]]], tiers: Iterable[float]
) -> dict[float, date]:
    """First date the number of *distinct* values reaches each tier (multisport)."""
    reached: dict[float, date] = {}
    tier_list = sorted(tiers)
    seen: set[str] = set()
    for day, value in sorted(events, key=lambda e: e[0]):
        if not value:
            continue
        seen.add(value)
        for tier in tier_list:
            if tier not in reached and len(seen) >= tier:
                reached[tier] = day
    return reached


# ── Streaks ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StreakState:
    """A streak as the athlete should see it."""

    current: int
    longest: int
    # True when the *current* period has not qualified yet but the streak is
    # still alive: the week is simply not over. Visiting on a Tuesday must never
    # show a streak as broken.
    in_progress: bool = False


def _period_sequence(first: date, last: date, *, monthly: bool) -> list[date]:
    """Every period start from *first* to *last* inclusive, including empty ones."""
    step = _next_month if monthly else (lambda d: d + timedelta(days=7))
    out: list[date] = []
    cur = first
    while cur <= last:
        out.append(cur)
        cur = step(cur)
    return out


def streak_state(
    buckets: list[PeriodBucket],
    qualifies,
    today: date,
    *,
    monthly: bool = False,
) -> StreakState:
    """Current and longest run of consecutive qualifying periods.

    A period that does not qualify ends the run — there are no grace periods, so
    the number shown always means exactly what it says. The one exception is the
    *current* period, which is still in progress: if this week hasn't qualified
    yet, the streak stands at the run ending last week and is flagged
    ``in_progress`` rather than reported as broken.
    """
    if not buckets:
        return StreakState(current=0, longest=0)

    key = month_start if monthly else week_start
    now = key(today)
    qualifying = {b.start for b in buckets if qualifies(b)}

    periods = _period_sequence(min(b.start for b in buckets), now, monthly=monthly)

    longest = 0
    run = 0
    runs_by_end: dict[date, int] = {}
    for p in periods:
        run = run + 1 if p in qualifying else 0
        runs_by_end[p] = run
        longest = max(longest, run)

    current = runs_by_end.get(now, 0)
    in_progress = False
    if current == 0:
        # This period isn't done yet — fall back to the run ending last period.
        previous = periods[-2] if len(periods) >= 2 else None
        if previous is not None and runs_by_end.get(previous, 0) > 0:
            current = runs_by_end[previous]
            in_progress = True

    return StreakState(current=current, longest=longest, in_progress=in_progress)


def streak_tier_dates(
    buckets: list[PeriodBucket],
    qualifies,
    tiers: Iterable[float],
    today: date,
    *,
    monthly: bool = False,
) -> dict[float, date]:
    """First date each streak length was reached.

    A streak of n is earned on the last day of its n-th period, capped at today
    so an achievement is never dated in the future.
    """
    if not buckets:
        return {}

    key = month_start if monthly else week_start
    qualifying = {b.start for b in buckets if qualifies(b)}
    periods = _period_sequence(min(b.start for b in buckets), key(today), monthly=monthly)

    reached: dict[float, date] = {}
    tier_list = sorted(tiers)
    run = 0
    for p in periods:
        run = run + 1 if p in qualifying else 0
        if not run:
            continue
        earned = _month_end(p) if monthly else p + timedelta(days=6)
        earned = min(earned, today)
        for tier in tier_list:
            if tier not in reached and run >= tier:
                reached[tier] = earned
    return reached


# ── Streak predicates ────────────────────────────────────────────────────────

def qualifies_active(bucket: PeriodBucket) -> bool:
    return bucket.count > 0


def qualifies_volume(bucket: PeriodBucket) -> bool:
    return bucket.seconds >= STREAK_VOLUME_HOURS * 3600


def qualifies_distance(bucket: PeriodBucket) -> bool:
    return bucket.metres >= STREAK_DISTANCE_KM * 1000


def qualifies_climbing(bucket: PeriodBucket) -> bool:
    return bucket.elevation_m >= STREAK_CLIMBING_M


def qualifies_multisport(bucket: PeriodBucket) -> bool:
    return len(bucket.sports) >= STREAK_MULTISPORT_SPORTS


# ── Comeback ─────────────────────────────────────────────────────────────────


def comeback_date(days: Iterable[date]) -> Optional[date]:
    """Date of the first activity following a gap of ≥ ``COMEBACK_GAP_DAYS``.

    Returns the *earliest* such return, so the unlock stays put once earned
    rather than hopping forward with every later break.
    """
    ordered = sorted(set(days))
    for previous, current in zip(ordered, ordered[1:]):
        if (current - previous).days >= COMEBACK_GAP_DAYS:
            return current
    return None
