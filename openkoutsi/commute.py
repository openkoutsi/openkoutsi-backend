"""Commute detection from athlete-defined rules (issue #63).

A commute is not defined by *where* it goes: openkoutsi holds no location data
from activities (see :class:`openkoutsi.gpx.Route`), so the obvious detector is
unavailable. It is defined instead as **the same short ride, at the same times,
on working days** — sport type, distance, duration and the local clock.

Pure, in the spirit of :mod:`openkoutsi.categorization`: rules and primitives in,
a match out. Where the rules are stored and whose timezone this is belong to
``backend.app.services.commute``.

Three deliberate rules shape the design:

1. **An empty rule matches nothing.** A rule with no criteria is a configuration
   mistake, and the generous reading — "match every ride" — would label an
   athlete's whole history in one pass. Every criterion is optional, but a rule
   needs at least one to fire.
2. **Malformed input is dropped, never raised on.** Rules live in
   ``app_settings``, a free-form JSON column with no validation behind it (see
   ``backend.app.core.timezones`` for the same problem with ``timezone``). A typo
   in one rule must not take down the ingest path, so :func:`parse_rules` skips
   what it cannot read.
3. **Windows are local, and may span midnight.** 22:00–02:00 is a night-shift
   commute, not an inverted range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from typing import Any, Iterable, Optional, Sequence

# ── Clustering constants ─────────────────────────────────────────────────────
# How many labelled rides an athlete needs before we will propose a rule from
# their history. Below this a "cluster" is a coincidence: three rides home from
# work in the same half hour describe nothing but three rides.
MIN_SAMPLES_FOR_PROPOSAL = 10

# Start times this far apart (minutes) belong to different windows. Sized to
# separate a morning leg from an evening one while keeping a single leg that
# drifts across the term — nobody's commute home is two hours from their commute
# in, and nobody leaves for work at 07:00 one day and 09:30 the next *as the
# same trip*.
WINDOW_SPLIT_GAP_MIN = 120

# Slack added to each edge of a proposed window, in minutes. A proposal derived
# from history is a starting point, and one exactly as wide as the observed
# samples rejects the very next ride that runs five minutes late.
WINDOW_PAD_MIN = 15

# Fraction of a proposed range's own span added to each edge, for the distance
# and duration bands. Proportional rather than absolute because a 4 km commute
# and a 40 km one need very different slack.
RANGE_PAD_FRACTION = 0.10

# Percentile trimmed from each end before a distance/duration band is proposed,
# so one ride that detoured to the shop doesn't stretch the band around itself.
RANGE_TRIM_FRACTION = 0.10

# A window must hold at least this share of the samples to be proposed at all.
MIN_WINDOW_SHARE = 0.15

#: Ceiling on time windows in one rule. Unlike the rule count, this is an
#: *independent* multiplier on the matcher's cost: `matches` runs
#: `any(w.contains(minute) for w in self.windows)` per rule per activity, so the
#: real shape is activities × rules × windows and capping the rules alone bounds
#: only one of the three. Measured at 50 rules (the cap) × 100 windows, a
#: 20 000-activity re-evaluation runs to minutes of CPU inside one request.
#: Twenty-four is an hourly departure all day, already far past any commute.
MAX_WINDOWS_PER_RULE = 24

_TIME_RE = re.compile(r"^\s*(\d{1,2})\s*:\s*(\d{2})\s*$")

MINUTES_PER_DAY = 24 * 60


def parse_time_of_day(raw: Any) -> Optional[int]:
    """``"07:30"`` → minutes since local midnight, or None if unreadable.

    Accepts an integer minute count too, which is what a client that has already
    done the conversion will send. ``"24:00"`` is rejected rather than folded to
    midnight: it is far more likely to be a typo than an intent.
    """
    if isinstance(raw, bool):  # bool is an int subclass; never a time
        return None
    if isinstance(raw, int):
        return raw if 0 <= raw < MINUTES_PER_DAY else None
    if not isinstance(raw, str):
        return None
    m = _TIME_RE.match(raw)
    if m is None:
        return None
    hours, minutes = int(m.group(1)), int(m.group(2))
    if hours > 23 or minutes > 59:
        return None
    return hours * 60 + minutes


def format_time_of_day(minute: int) -> str:
    """Inverse of :func:`parse_time_of_day`, for rules we propose ourselves."""
    minute %= MINUTES_PER_DAY
    return f"{minute // 60:02d}:{minute % 60:02d}"


@dataclass(frozen=True)
class TimeWindow:
    """A span of the local clock, inclusive of both edges.

    ``start`` > ``end`` means the window wraps midnight (22:00–02:00), which is a
    real commute pattern and not an error. Both are minutes since midnight.
    """

    start: int
    end: int

    def contains(self, minute: int) -> bool:
        if self.start <= self.end:
            return self.start <= minute <= self.end
        return minute >= self.start or minute <= self.end

    def as_dict(self) -> dict[str, str]:
        return {"start": format_time_of_day(self.start), "end": format_time_of_day(self.end)}


@dataclass(frozen=True)
class CommuteRule:
    """One athlete-defined description of what their commute looks like.

    Every criterion is optional and they combine with AND. ``sport_types`` holds
    *exact* sport types rather than a category, because the distinction matters:
    an e-bike is very often the commuting bike specifically, so ``EBikeRide`` may
    be the whole rule for one athlete while another needs a distance band to
    separate the ride to work from the ride for fun on the same bike.

    ``auto_apply`` lets an athlete who has watched a rule get it right stop
    confirming it. It defaults off — a guess does not write to athlete-owned
    data unasked.
    """

    id: str
    name: Optional[str] = None
    sport_types: frozenset[str] = field(default_factory=frozenset)
    min_distance_m: Optional[float] = None
    max_distance_m: Optional[float] = None
    min_duration_s: Optional[int] = None
    max_duration_s: Optional[int] = None
    windows: tuple[TimeWindow, ...] = ()
    weekdays: frozenset[int] = field(default_factory=frozenset)  # 0=Monday
    auto_apply: bool = False
    enabled: bool = True

    @property
    def has_criteria(self) -> bool:
        """Does this rule constrain anything at all?

        A rule that constrains nothing matches nothing — see the module
        docstring. Kept as a property so both the matcher and the API's
        validation ask the same question.
        """
        return bool(
            self.sport_types
            or self.windows
            or self.weekdays
            or self.min_distance_m is not None
            or self.max_distance_m is not None
            or self.min_duration_s is not None
            or self.max_duration_s is not None
        )

    def matches(
        self,
        *,
        sport_type: Optional[str],
        local_start: Optional[datetime],
        duration_s: Optional[int],
        distance_m: Optional[float],
    ) -> bool:
        """Does one activity satisfy every criterion this rule sets?

        A criterion the activity has no data for fails rather than passes: a ride
        with no distance cannot be shown to fall inside a 4–8 km band, and
        labelling it anyway would be asserting something we do not know.
        """
        if not self.enabled or not self.has_criteria:
            return False

        if self.sport_types:
            if sport_type is None or sport_type.lower() not in self.sport_types:
                return False

        if self.min_distance_m is not None or self.max_distance_m is not None:
            if distance_m is None:
                return False
            if self.min_distance_m is not None and distance_m < self.min_distance_m:
                return False
            if self.max_distance_m is not None and distance_m > self.max_distance_m:
                return False

        if self.min_duration_s is not None or self.max_duration_s is not None:
            if duration_s is None:
                return False
            if self.min_duration_s is not None and duration_s < self.min_duration_s:
                return False
            if self.max_duration_s is not None and duration_s > self.max_duration_s:
                return False

        if self.windows or self.weekdays:
            if local_start is None:
                return False
            if self.weekdays and local_start.weekday() not in self.weekdays:
                return False
            if self.windows:
                minute = local_start.hour * 60 + local_start.minute
                if not any(w.contains(minute) for w in self.windows):
                    return False

        return True

    def as_dict(self) -> dict[str, Any]:
        """Round-trips through :func:`parse_rule`. What the API hands back."""
        out: dict[str, Any] = {"id": self.id, "enabled": self.enabled, "auto_apply": self.auto_apply}
        if self.name:
            out["name"] = self.name
        if self.sport_types:
            out["sport_types"] = sorted(self.sport_types)
        for key, value in (
            ("min_distance_m", self.min_distance_m),
            ("max_distance_m", self.max_distance_m),
            ("min_duration_s", self.min_duration_s),
            ("max_duration_s", self.max_duration_s),
        ):
            if value is not None:
                out[key] = value
        if self.windows:
            out["windows"] = [w.as_dict() for w in self.windows]
        if self.weekdays:
            out["weekdays"] = sorted(self.weekdays)
        return out


def _number(raw: Any) -> Optional[float]:
    """A non-negative finite number, or None. Bools are not numbers here."""
    if isinstance(raw, bool) or raw is None:
        return None
    if not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        return None
    return value


def parse_rule(raw: Any) -> Optional[CommuteRule]:
    """One rule out of free-form JSON, or None if it is unusable.

    Unusable means: not a mapping, no id, or no criterion that constrains
    anything. Individual unreadable *fields* are dropped and the rest of the rule
    kept — a bad ``max_distance_m`` should cost the athlete that bound, not the
    whole rule.
    """
    if not isinstance(raw, dict):
        return None

    rule_id = raw.get("id")
    if not isinstance(rule_id, str) or not rule_id.strip():
        return None

    sport_types = frozenset(
        s.lower() for s in raw.get("sport_types") or [] if isinstance(s, str) and s.strip()
    )

    windows: list[TimeWindow] = []
    for entry in raw.get("windows") or []:
        if not isinstance(entry, dict):
            continue
        start = parse_time_of_day(entry.get("start"))
        end = parse_time_of_day(entry.get("end"))
        if start is None or end is None:
            continue
        windows.append(TimeWindow(start, end))
        if len(windows) >= MAX_WINDOWS_PER_RULE:
            # Truncated rather than rejected, and here rather than in the API,
            # so the bound protects rules already stored as well as arriving
            # ones — this is the path every ingest reads through. The API
            # stores what `as_dict` gives back, so an athlete who sends more
            # sees the truncation reflected in their saved rule.
            break

    weekdays = frozenset(
        d for d in raw.get("weekdays") or [] if isinstance(d, int) and not isinstance(d, bool) and 0 <= d <= 6
    )

    min_distance = _number(raw.get("min_distance_m"))
    max_distance = _number(raw.get("max_distance_m"))
    min_duration = _number(raw.get("min_duration_s"))
    max_duration = _number(raw.get("max_duration_s"))

    # An inverted range can never match anything, so it is a rule that silently
    # does nothing — drop the bounds and let the rest of the rule work.
    if min_distance is not None and max_distance is not None and min_distance > max_distance:
        min_distance = max_distance = None
    if min_duration is not None and max_duration is not None and min_duration > max_duration:
        min_duration = max_duration = None

    name = raw.get("name")
    rule = CommuteRule(
        id=rule_id.strip(),
        name=name.strip() if isinstance(name, str) and name.strip() else None,
        sport_types=sport_types,
        min_distance_m=min_distance,
        max_distance_m=max_distance,
        min_duration_s=int(min_duration) if min_duration is not None else None,
        max_duration_s=int(max_duration) if max_duration is not None else None,
        windows=tuple(windows),
        weekdays=weekdays,
        auto_apply=raw.get("auto_apply") is True,
        enabled=raw.get("enabled") is not False,
    )
    return rule if rule.has_criteria else None


def parse_rules(raw: Any) -> list[CommuteRule]:
    """Every readable rule in an ``app_settings.commute_rules`` value.

    Never raises, whatever is in the column: an athlete with a corrupted setting
    gets no commute detection, not a failed upload.
    """
    if not isinstance(raw, list):
        return []
    rules = [parse_rule(entry) for entry in raw]
    return [r for r in rules if r is not None]


def match_commute(
    rules: Sequence[CommuteRule],
    *,
    sport_type: Optional[str],
    start_time: Optional[datetime],
    duration_s: Optional[int],
    distance_m: Optional[float],
    tz: Optional[tzinfo] = None,
) -> Optional[CommuteRule]:
    """The first enabled rule this activity satisfies, or None.

    ``start_time`` is converted into ``tz`` before any clock comparison, because
    a rule's windows are the athlete's local time and ``Activity.start_time`` is
    UTC. Pass the zone from ``backend.app.core.timezones.resolve_zone``; None
    leaves the timestamp as it arrived, which is right for a naive datetime and
    means UTC for an aware one.

    Rules are tried in order, so the first match wins and the athlete's own
    ordering decides precedence.
    """
    local_start = _to_local(start_time, tz)
    for rule in rules:
        if rule.matches(
            sport_type=sport_type,
            local_start=local_start,
            duration_s=duration_s,
            distance_m=distance_m,
        ):
            return rule
    return None


def _to_local(start_time: Optional[datetime], tz: Optional[tzinfo]) -> Optional[datetime]:
    """``start_time`` on the athlete's clock.

    A naive datetime is assumed to be in the target zone already and returned
    untouched — converting it would be inventing an offset. An aware one is
    converted, which is the path every stored ``Activity.start_time`` takes.
    """
    if start_time is None or tz is None:
        return start_time
    if start_time.tzinfo is None:
        return start_time
    return start_time.astimezone(tz)


# ── Proposing rules from history ─────────────────────────────────────────────


@dataclass(frozen=True)
class RideSample:
    """The little an activity has to offer the clustering pass.

    ``local_start`` is already on the athlete's clock — converting is the
    caller's job, the same way it is for :func:`match_commute`.
    """

    sport_type: Optional[str]
    local_start: Optional[datetime]
    duration_s: Optional[int]
    distance_m: Optional[float]


def _trimmed_range(values: Sequence[float]) -> Optional[tuple[float, float]]:
    """A padded low/high band covering the middle of ``values``.

    Trims a tail from each end before padding, so a single ride that went the
    long way home widens the band by nothing rather than by its full excess.
    """
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    if len(clean) >= 5:
        cut = max(1, int(len(clean) * RANGE_TRIM_FRACTION))
        clean = clean[cut : len(clean) - cut] or clean
    low, high = clean[0], clean[-1]
    pad = max((high - low) * RANGE_PAD_FRACTION, high * 0.02)
    return max(0.0, low - pad), high + pad


def _cluster_minutes(minutes: Sequence[int]) -> list[tuple[int, int]]:
    """Split start times into windows, treating the clock as a circle.

    Gap-based rather than k-means: the number of legs is exactly what we are
    trying to discover (one for a one-way commuter who trains home, two for the
    usual there-and-back), and a fixed k would impose an answer. Wraparound is
    handled by rotating the sorted list to start after its largest gap, so a
    22:40/23:10/00:20 cluster stays one window instead of splitting at midnight.
    """
    if not minutes:
        return []
    ordered = sorted(minutes)
    if len(ordered) == 1:
        return [(ordered[0], ordered[0])]

    # Rotate so the sequence begins after the largest circular gap.
    gaps = [
        ((ordered[(i + 1) % len(ordered)] - ordered[i]) % MINUTES_PER_DAY, i)
        for i in range(len(ordered))
    ]
    _, split_at = max(gaps)
    rotated = ordered[split_at + 1 :] + ordered[: split_at + 1]

    clusters: list[list[int]] = [[rotated[0]]]
    for prev, current in zip(rotated, rotated[1:]):
        if (current - prev) % MINUTES_PER_DAY > WINDOW_SPLIT_GAP_MIN:
            clusters.append([current])
        else:
            clusters[-1].append(current)

    threshold = max(1, int(len(ordered) * MIN_WINDOW_SHARE))
    return [
        (cluster[0], cluster[-1]) for cluster in clusters if len(cluster) >= threshold
    ]


def propose_rule(
    samples: Iterable[RideSample],
    *,
    rule_id: str = "proposed",
    name: Optional[str] = None,
    min_samples: int = MIN_SAMPLES_FOR_PROPOSAL,
) -> Optional[CommuteRule]:
    """A rule describing ``samples``, or None if there are too few to bother.

    This is what makes rule configuration something other than a chore: nobody
    is going to hand-type "between 4.2 and 6.8 km, 06:41–08:12", but they will
    happily nudge those numbers once something else has proposed them.

    Weekdays are only constrained when the samples actually avoid part of the
    week — a commuter who also rides to work at weekends should not have their
    weekend legs excluded by a rule derived from their own history. Sport types
    likewise are only pinned when the history is narrow enough for it to mean
    something.
    """
    rides = [s for s in samples if s is not None]
    if len(rides) < min_samples:
        return None

    minutes = [
        s.local_start.hour * 60 + s.local_start.minute
        for s in rides
        if s.local_start is not None
    ]
    windows = tuple(
        TimeWindow(
            (low - WINDOW_PAD_MIN) % MINUTES_PER_DAY,
            (high + WINDOW_PAD_MIN) % MINUTES_PER_DAY,
        )
        for low, high in _cluster_minutes(minutes)
    )

    distance = _trimmed_range([s.distance_m for s in rides if s.distance_m is not None])
    duration = _trimmed_range(
        [float(s.duration_s) for s in rides if s.duration_s is not None]
    )

    observed_days = {s.local_start.weekday() for s in rides if s.local_start is not None}
    weekdays = frozenset(observed_days) if 0 < len(observed_days) < 7 else frozenset()

    observed_sports = {
        s.sport_type.lower() for s in rides if isinstance(s.sport_type, str) and s.sport_type
    }
    # Pinning sport type only helps when the history is consistent about it;
    # past a handful of types the criterion stops excluding anything.
    sport_types = frozenset(observed_sports) if 0 < len(observed_sports) <= 3 else frozenset()

    rule = CommuteRule(
        id=rule_id,
        name=name,
        sport_types=sport_types,
        min_distance_m=round(distance[0], 1) if distance else None,
        max_distance_m=round(distance[1], 1) if distance else None,
        min_duration_s=int(duration[0]) if duration else None,
        max_duration_s=int(duration[1]) if duration else None,
        windows=windows,
        weekdays=weekdays,
    )
    return rule if rule.has_criteria else None


def near_miss_criteria(
    rule: CommuteRule,
    *,
    sport_type: Optional[str],
    local_start: Optional[datetime],
    duration_s: Optional[int],
    distance_m: Optional[float],
) -> list[str]:
    """Which of ``rule``'s criteria this activity fails.

    The signal behind "your rule is too narrow": an athlete who labels a ride a
    commute by hand, when a rule missed it by one criterion, is telling us which
    bound to widen. A ride failing *several* criteria is simply a different ride,
    which is why the caller looks for a list of length one.

    Returns criterion names, not booleans, so the caller can say *what* to widen.
    """
    failed: list[str] = []
    if rule.sport_types and (
        sport_type is None or sport_type.lower() not in rule.sport_types
    ):
        failed.append("sport_types")

    if rule.min_distance_m is not None or rule.max_distance_m is not None:
        if (
            distance_m is None
            or (rule.min_distance_m is not None and distance_m < rule.min_distance_m)
            or (rule.max_distance_m is not None and distance_m > rule.max_distance_m)
        ):
            failed.append("distance")

    if rule.min_duration_s is not None or rule.max_duration_s is not None:
        if (
            duration_s is None
            or (rule.min_duration_s is not None and duration_s < rule.min_duration_s)
            or (rule.max_duration_s is not None and duration_s > rule.max_duration_s)
        ):
            failed.append("duration")

    if rule.weekdays and (
        local_start is None or local_start.weekday() not in rule.weekdays
    ):
        failed.append("weekdays")

    if rule.windows:
        if local_start is None:
            failed.append("windows")
        else:
            minute = local_start.hour * 60 + local_start.minute
            if not any(w.contains(minute) for w in rule.windows):
                failed.append("windows")

    return failed
