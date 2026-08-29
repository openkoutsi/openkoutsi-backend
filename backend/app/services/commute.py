"""Commute detection orchestration (issue #63).

The rules and the matching are pure and live in :mod:`openkoutsi.commute`. This
module is the part that knows about athletes: where the rules are stored, whose
timezone to convert into, and what a suggestion means once it is written down.

**Suggestions are not labels.** ``Activity.labels`` is athlete-owned data; this
writes to ``Activity.label_suggestions`` instead and waits to be confirmed. Two
concrete consequences drive that split rather than mere caution:

- the ``commuter`` badge counts labelled activities
  (:mod:`backend.app.services.achievements`), so auto-applying would mint tiers
  off a heuristic;
- the RPE queue *excludes* commute-labelled rides
  (``api/activities.get_rpe_queue``), so writing the label early would remove
  the ride from the very prompt where the athlete would have seen and confirmed
  it. Leaving it as a suggestion is what lets the prompt show up with the box
  already ticked.

The one thing applied without asking is a provider's own flag: Strava's
``commute`` boolean is the athlete's assertion, not our guess.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from openkoutsi.commute import (
    CommuteRule,
    RideSample,
    match_commute,
    near_miss_criteria,
    parse_rules,
)
from backend.app.core.timezones import resolve_zone
from backend.app.models.user_orm import Activity

log = logging.getLogger(__name__)

#: The label this module suggests. A second label would slot in beside it —
#: the column is keyed by label name for exactly that reason — but `race` is a
#: genuinely different problem and nothing else uses this yet.
COMMUTE = "commute"

#: Where an athlete's rules live inside the free-form ``app_settings`` dict.
RULES_KEY = "commute_rules"

STATE_PENDING = "pending"
STATE_ACCEPTED = "accepted"
STATE_DISMISSED = "dismissed"

#: States that mean "the athlete has already answered". A re-evaluation, a
#: reprocess or a history scan must never overwrite one of these — that is the
#: whole point of persisting the suggestion rather than deriving it.
TERMINAL_STATES = frozenset({STATE_ACCEPTED, STATE_DISMISSED})

#: What ``source`` says when the label came from the provider rather than a rule.
SOURCE_STRAVA = "strava"

#: How many dismissals of one rule's suggestions before we say the rule looks
#: wrong. Three is "not a slip of the finger" without being a season's worth.
DISMISSALS_BEFORE_REVIEW = 3

#: Ceiling on how many rules one athlete may store. The list is walked on every
#: ingest and once per activity in `reevaluate_pending`, which runs inline in
#: `PATCH /api/athlete` — so an unbounded list makes the athlete's own next
#: upload slow and holds a request worker for the length of the re-evaluation.
#: Fifty is far past any real commuter; the cap is against accident and abuse,
#: not against anybody's actual riding.
MAX_RULES = 50

#: Rows per batch for the two passes that need whole entities because they
#: write. The read-only passes select columns instead (`_SAMPLE_COLUMNS`); these
#: two cannot, so they stream. Batching keeps peak memory roughly flat across a
#: history of any size — measured at ~8× lower than buffering for a 20 000-ride
#: athlete — while each batch still flushes into the *same* transaction, so the
#: scan stays one atomic unit rather than becoming a series of partial writes.
_YIELD_PER = 1000


def rules_for(athlete) -> list[CommuteRule]:
    """The athlete's parsed commute rules — never raises, whatever is stored."""
    settings = athlete.app_settings if isinstance(athlete.app_settings, dict) else {}
    return parse_rules(settings.get(RULES_KEY))


def athlete_zone(athlete):
    """The zone a rule's windows are expressed in. Falls back to UTC."""
    settings = athlete.app_settings if isinstance(athlete.app_settings, dict) else {}
    return resolve_zone(settings.get("timezone"))


def _suggestions(activity: Activity) -> dict:
    raw = activity.label_suggestions
    return dict(raw) if isinstance(raw, dict) else {}


def suggestion_state(activity: Activity, label: str = COMMUTE) -> Optional[str]:
    """The recorded state for ``label`` on this activity, or None."""
    entry = _suggestions(activity).get(label)
    if isinstance(entry, dict):
        state = entry.get("state")
        return state if isinstance(state, str) else None
    return None


def has_label(activity: Activity, label: str = COMMUTE) -> bool:
    return label in (activity.labels or [])


def _write_suggestion(
    activity: Activity, state: str, *, source: Optional[str], label: str = COMMUTE
) -> None:
    """Set one label's suggestion entry, leaving any other label's alone."""
    entries = _suggestions(activity)
    entry = {"state": state, "at": datetime.now(timezone.utc).isoformat()}
    if source is not None:
        entry["source"] = source
    entries[label] = entry
    # Reassigned rather than mutated in place: SQLAlchemy does not track
    # mutations inside a plain JSON column, so an in-place edit would not be
    # persisted.
    activity.label_suggestions = entries


def clear_suggestion(activity: Activity, label: str = COMMUTE) -> None:
    """Drop a suggestion entirely — for one that no longer applies at all."""
    entries = _suggestions(activity)
    if entries.pop(label, None) is not None:
        activity.label_suggestions = entries or None


def apply_label(activity: Activity, label: str = COMMUTE) -> bool:
    """Add ``label`` to the athlete-owned list. True if it was not already there."""
    current = list(activity.labels or [])
    if label in current:
        return False
    current.append(label)
    activity.labels = current
    return True


def remove_label(activity: Activity, label: str = COMMUTE) -> bool:
    """Take ``label`` off the athlete-owned list. True if it was there."""
    current = list(activity.labels or [])
    if label not in current:
        return False
    activity.labels = [lbl for lbl in current if lbl != label]
    return True


def answer_suggestion(activity: Activity, label: str, state: str) -> None:
    """Record the athlete's answer, keeping whatever proposed it on record.

    The ``source`` is carried across rather than overwritten: knowing that a
    dismissal was of *this* rule's suggestion is the whole basis of the
    "your rule looks wrong" signal in :func:`rule_feedback`.
    """
    entry = _suggestions(activity).get(label)
    source = entry.get("source") if isinstance(entry, dict) else None
    _write_suggestion(activity, state, source=source, label=label)


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """An aware UTC timestamp, whatever the row handed back.

    ``Activity.start_time`` is declared ``DateTime(timezone=True)``, but SQLite
    has no timezone type and returns a **naive** datetime holding the UTC
    instant. Left as-is, a naive stamp reaches the matcher's conversion and is
    treated as already-local, so 05:30 UTC would be tested against a 06:30–08:30
    *local* window as though it were half past five in the morning — and every
    commute outside UTC would go undetected.

    Same rule, for the same reason, as ``services.achievements._local_day``.
    """
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def sample_for(activity: Activity, tz) -> RideSample:
    """One activity as the pure layer wants to see it, on the athlete's clock."""
    start = _as_utc(activity.start_time)
    if start is not None and tz is not None:
        start = start.astimezone(tz)
    return RideSample(
        sport_type=activity.sport_type,
        local_start=start,
        duration_s=activity.duration_s,
        distance_m=activity.distance_m,
    )


def evaluate(
    activity: Activity,
    rules: list[CommuteRule],
    tz,
    *,
    force: bool = False,
) -> Optional[str]:
    """Run the rules over one activity and record what they say.

    Returns the state written (``pending``/``accepted``), or None when nothing
    changed. Does **not** commit — the caller owns the transaction, because every
    ingest path already has one open.

    Three things stop a suggestion being written, and all three are deliberate:

    - the athlete has already answered (``accepted``/``dismissed``). This is what
      makes a dismissal survive a reprocess.
    - the label is already applied, by hand or by a provider. Nothing to suggest.
    - no rule matches. An existing *pending* suggestion is then withdrawn, which
      is what makes re-evaluation after a rule edit do the right thing when the
      rule was narrowed.

    ``force`` overrides only the first of those, and exists for the history scan,
    where the athlete has explicitly asked for a fresh look.
    """
    state = suggestion_state(activity)
    if state in TERMINAL_STATES and not force:
        return None
    if has_label(activity):
        # Already a commute as far as the athlete is concerned. Record that
        # rather than leaving a stale pending suggestion against it.
        if state == STATE_PENDING:
            clear_suggestion(activity)
        return None

    matched = match_commute(
        rules,
        sport_type=activity.sport_type,
        start_time=_as_utc(activity.start_time),
        duration_s=activity.duration_s,
        distance_m=activity.distance_m,
        tz=tz,
    )
    if matched is None:
        if state == STATE_PENDING:
            clear_suggestion(activity)
        return None

    if matched.auto_apply:
        apply_label(activity)
        _write_suggestion(activity, STATE_ACCEPTED, source=f"rule:{matched.id}")
        return STATE_ACCEPTED

    _write_suggestion(activity, STATE_PENDING, source=f"rule:{matched.id}")
    return STATE_PENDING


async def evaluate_activity(session: AsyncSession, athlete, activity: Activity) -> Optional[str]:
    """:func:`evaluate` for a single freshly-ingested activity.

    The hook every ingest path calls. Cheap when the athlete has no rules, which
    is the common case, so it is safe to call unconditionally.
    """
    rules = rules_for(athlete)
    if not rules:
        return None
    return evaluate(activity, rules, athlete_zone(athlete))


def adopt_provider_flag(activity: Activity, is_commute: Optional[bool]) -> bool:
    """Take a provider's own commute flag at face value.

    Strava's activity payload carries a ``commute`` boolean the athlete set
    themselves. That is not a guess we are making, so unlike a rule match it is
    **applied**, not suggested — with the suggestion recorded as ``accepted`` so
    the surfaces that show provenance can say where the label came from.

    A false flag is not treated as a denial: an athlete who never uses Strava's
    checkbox would otherwise have every rule-based suggestion overruled by a
    default. Only a true flag carries information.
    """
    if is_commute is not True:
        return False
    if suggestion_state(activity) == STATE_DISMISSED:
        # The athlete said no here. A provider default does not overrule them.
        return False
    changed = apply_label(activity)
    _write_suggestion(activity, STATE_ACCEPTED, source=SOURCE_STRAVA)
    return changed


async def reevaluate_pending(session: AsyncSession, athlete) -> int:
    """Re-run the rules over everything not yet answered, after a rule change.

    Issue #63 settled that editing a rule re-evaluates rather than freezing what
    already fired: a narrowed rule should withdraw the suggestions it no longer
    stands behind, and a widened one should pick up what it now covers.

    Only ``pending`` and unsuggested activities are touched — an answered
    suggestion stays answered, so this can never resurrect a dismissal or undo
    an acceptance. Returns the number of activities whose state changed.
    """
    rules = rules_for(athlete)
    tz = athlete_zone(athlete)

    # Streamed rather than buffered: these two need entities (they write), but
    # nothing needs the athlete's whole history resident at once. Each batch is
    # flushed into the same transaction as it goes, so this bounds memory
    # without weakening the all-or-nothing commit — see `_YIELD_PER`.
    result = await session.stream_scalars(
        select(Activity)
        .where(Activity.athlete_id == athlete.id)
        .execution_options(yield_per=_YIELD_PER)
    )
    changed = 0
    async for activity in result:
        state = suggestion_state(activity)
        if state in TERMINAL_STATES:
            continue
        before = (state, tuple(activity.labels or []))
        evaluate(activity, rules, tz)
        if before != (suggestion_state(activity), tuple(activity.labels or [])):
            changed += 1
    if changed:
        await session.commit()
    return changed


async def scan_history(session: AsyncSession, athlete, *, force: bool = False) -> dict:
    """Look at the whole back catalogue and suggest what the rules cover.

    The answer to an imported history: a decade of rides that arrived with no
    provider flag and no labels. Deliberately an explicit request rather than
    something a migration or a rule edit does on its own — it can touch tens of
    thousands of rows and the athlete should be the one asking for it.
    """
    rules = rules_for(athlete)
    tz = athlete_zone(athlete)
    if not rules:
        return {"scanned": 0, "suggested": 0, "applied": 0, "withdrawn": 0}

    result = await session.stream_scalars(
        select(Activity)
        .where(Activity.athlete_id == athlete.id)
        .execution_options(yield_per=_YIELD_PER)
    )
    scanned = suggested = applied = withdrawn = 0
    async for activity in result:
        scanned += 1
        # `evaluate` returns None both for "nothing changed" and for "there was
        # a pending suggestion and I just withdrew it", so the return value
        # alone cannot decide whether there is anything to commit. Comparing
        # before and after can — the same thing `reevaluate_pending` does.
        # Without this a scan whose only effect is retracting suggestions rolls
        # back, and the review screen keeps offering rides no rule stands
        # behind. The realistic trigger is an athlete correcting their timezone,
        # which is not a rule edit and so never reaches `reevaluate_pending`.
        before = (suggestion_state(activity), tuple(activity.labels or []))
        state = evaluate(activity, rules, tz, force=force)
        if state == STATE_PENDING:
            suggested += 1
        elif state == STATE_ACCEPTED:
            applied += 1
        elif before != (suggestion_state(activity), tuple(activity.labels or [])):
            withdrawn += 1
    if suggested or applied or withdrawn:
        await session.commit()
    return {
        "scanned": scanned,
        "suggested": suggested,
        "applied": applied,
        "withdrawn": withdrawn,
    }


#: The columns any of this actually reads. Selected explicitly rather than
#: hydrating whole ``Activity`` entities: a row carries ~45 columns including
#: the LLM analysis prose, the athlete's notes and the `zone_times` blob, and
#: an athlete's whole history can run to tens of thousands of rows. Same
#: reasoning, and the same shape, as ``services.achievements._load_facts``.
_SAMPLE_COLUMNS = (
    Activity.sport_type,
    Activity.start_time,
    Activity.duration_s,
    Activity.distance_m,
    Activity.labels,
    Activity.label_suggestions,
)


def _row_sample(row, tz) -> RideSample:
    """:func:`sample_for` for a column row rather than an ORM entity."""
    start = _as_utc(row.start_time)
    if start is not None and tz is not None:
        start = start.astimezone(tz)
    return RideSample(
        sport_type=row.sport_type,
        local_start=start,
        duration_s=row.duration_s,
        distance_m=row.distance_m,
    )


async def labelled_samples(session: AsyncSession, athlete) -> list[RideSample]:
    """Every ride the athlete has actually labelled a commute, as samples.

    What the rule proposal is derived from. Reads ``labels`` rather than the
    suggestion column on purpose: a proposal should be built from what the
    athlete has confirmed, never from what we guessed.

    Read-only, so it never hydrates entities — see :data:`_SAMPLE_COLUMNS`.
    """
    tz = athlete_zone(athlete)
    result = await session.execute(
        select(*_SAMPLE_COLUMNS).where(Activity.athlete_id == athlete.id)
    )
    return [
        _row_sample(row, tz)
        for row in result
        if COMMUTE in (row.labels or [])
    ]


async def rule_feedback(session: AsyncSession, athlete) -> dict:
    """What the athlete's answers say about their rules being wrong.

    Two signals, both read straight off the suggestion column rather than kept
    as counters — ``source`` already records which rule fired, so there is
    nothing to keep in sync and nothing to drift:

    - **too narrow**: rides the athlete labelled by hand that no rule matched.
      The ones that missed a single criterion name which bound to widen; a ride
      failing several is simply a different ride.
    - **too wide**: rules whose suggestions keep being dismissed.

    Reported, never applied. Silently widening a rule each time its output is
    accepted is a feedback loop with no brake, and the athlete is the one who
    knows whether the 9 km ride was the commute or the long way round.
    """
    rules = rules_for(athlete)
    tz = athlete_zone(athlete)

    # Read-only, so columns rather than entities — see `_SAMPLE_COLUMNS`.
    result = await session.execute(
        select(*_SAMPLE_COLUMNS).where(Activity.athlete_id == athlete.id)
    )

    dismissed_by_rule: dict[str, int] = {}
    widen: dict[str, dict] = {}
    unmatched_manual = 0

    for row in result:
        entry = (row.label_suggestions or {}).get(COMMUTE)
        entry = entry if isinstance(entry, dict) else {}
        state = entry.get("state")
        source = entry.get("source")

        if state == STATE_DISMISSED and isinstance(source, str) and source.startswith("rule:"):
            rule_id = source[len("rule:") :]
            dismissed_by_rule[rule_id] = dismissed_by_rule.get(rule_id, 0) + 1
            continue

        # A hand-labelled ride the rules did not catch is the "too narrow"
        # signal — but only when the athlete labelled it themselves, so a ride
        # whose label came from a rule or from Strava is not evidence.
        if COMMUTE not in (row.labels or []) or source is not None:
            continue
        sample = _row_sample(row, tz)
        if match_commute(
            rules,
            sport_type=sample.sport_type,
            start_time=sample.local_start,
            duration_s=sample.duration_s,
            distance_m=sample.distance_m,
        ) is not None:
            continue

        unmatched_manual += 1
        for rule in rules:
            failed = near_miss_criteria(
                rule,
                sport_type=sample.sport_type,
                local_start=sample.local_start,
                duration_s=sample.duration_s,
                distance_m=sample.distance_m,
            )
            if len(failed) != 1:
                continue
            bucket = widen.setdefault(rule.id, {"rule_id": rule.id, "criteria": {}})
            criterion = failed[0]
            bucket["criteria"][criterion] = bucket["criteria"].get(criterion, 0) + 1

    return {
        "unmatched_manual_labels": unmatched_manual,
        "widen": [
            {"rule_id": v["rule_id"], "criteria": v["criteria"]}
            for v in widen.values()
        ],
        "review": [
            {"rule_id": rule_id, "dismissed": count}
            for rule_id, count in dismissed_by_rule.items()
            if count >= DISMISSALS_BEFORE_REVIEW
        ],
    }
