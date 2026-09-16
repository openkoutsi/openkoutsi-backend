"""Drafting a plan change, and applying it only when the athlete says yes (#72).

The whole design turns on one sentence: **the model never performs the write.**

That is not a stylistic preference. Koutsi in chat calls tools as
``ToolCaller.internal(user_id)``, which sets ``scopes=None``, and
``Tool.missing_scopes`` returns ``[]`` for ``None`` by design — that is how a
session credential means full access. So Koutsi holds every scope implicitly,
and a ``plans:write`` scope would gate an external MCP client while gating Koutsi
by exactly one instruction. The prompt cannot be the control either:
:mod:`.llm_chat` already says so in its own words, and an athlete pointing
openkoutsi at their own local model can make it say anything.

So the control is structural, and it is one sentence long:

    **No tool, called by anyone, through any door, can reach ``training_plans``
    or ``planned_workouts``.** The propose tools write one row to
    ``plan_proposals`` and nothing else. The only code that writes a plan is
    :func:`apply_proposal`, reached from an HTTP route that requires the
    athlete's own session and a proposal id.

Two phases
----------
**Propose.** :func:`draft_proposal` persists a fully-resolved draft — the built
weeks, not the model's arguments — and returns a compact preview. Nothing in
``training_plans`` moves.

**Apply.** The athlete answers a structured yes/no in the thread.
:func:`apply_proposal` re-runs every invariant against the database *now* and
either writes through the same helpers ``api/plans.py`` uses, or refuses with a
sentence. :func:`decline_proposal` marks the other answer and writes nothing.

Staleness is handled at apply, not by trusting ``expires_at``. A proposal drafted
yesterday may now start in the past, target a plan that has been deleted, or
archive a different set of plans than the athlete was shown — and that last one
matters most, because the set they were shown is what they consented to.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..core import audit
from ..models.user_orm import Athlete, PlanProposal, PlannedWorkout, TrainingPlan
from ..schemas.plan_proposals import (
    KIND_CREATE_PLAN,
    KIND_UPDATE_PLAN,
    KIND_UPDATE_WORKOUT,
    STATUS_APPLIED,
    STATUS_DECLINED,
    STATUS_EXPIRED,
    STATUS_PENDING,
    STATUS_SUPERSEDED,
    PlanProposalSummary,
)
from .achievements import mark_achievements_dirty
from .plan_adherence import catch_up_adherence, workout_date
from .plan_lifecycle import (
    archive_overlapping_active_plans,
    overlapping_active_plans,
    plan_end_date,
)

log = logging.getLogger(__name__)

#: How long an offer stands. Long enough to sleep on, short enough that the
#: apply-time re-validation is rarely the thing that catches a stale proposal.
PROPOSAL_TTL = timedelta(hours=24)

#: Refusal codes the web app turns into a sentence. A decided proposal, a lapsed
#: one and one the world has moved under are three different things to be told.
CODE_EXPIRED = "proposal_expired"
CODE_DECIDED = "proposal_decided"
CODE_STALE = "proposal_stale"


class ProposalError(Exception):
    """Why a proposal cannot be applied, in a sentence the athlete can act on."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _record(
    outcome: str,
    proposal: PlanProposal,
    user_id: str,
    *,
    plan_id: Optional[str] = None,
    refusal_code: Optional[str] = None,
) -> None:
    """One audit line for whatever just happened to this proposal.

    Written here rather than at the routes, for the same reason the draft record
    is: this module is the one place a proposal is minted, applied, refused or
    declined, so a caller cannot produce any of those without a record of it.
    """
    audit.plan_proposal(
        outcome=outcome,
        proposal_id=proposal.id,
        kind=proposal.kind,
        user_id=user_id,
        plan_id=plan_id,
        refusal_code=refusal_code,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite hands timestamps back naive; compare them as UTC."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


# ── Drafting ────────────────────────────────────────────────────────────────


async def draft_proposal(
    session: AsyncSession,
    *,
    kind: str,
    payload: dict,
    summary: PlanProposalSummary,
    built_by: str,
    user_id: str = "",
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
    target_plan_id: Optional[str] = None,
    target_workout_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> PlanProposal:
    """Persist one drafted change and supersede any earlier pending offer.

    **Superseded only by another proposal**, never by an ordinary question: an
    athlete asking "what would that do to my Saturdays?" must still be able to
    come back and click yes on the offer they were asking about.

    The audit record is written here rather than by the caller, because this is
    the one place a proposal is minted: a tool that forgot to log would otherwise
    produce an offer with no record of having been made.
    """
    at = now or _now()
    if conversation_id:
        await _supersede_pending(session, conversation_id, at)

    proposal = PlanProposal(
        conversation_id=conversation_id,
        message_id=message_id,
        kind=kind,
        target_plan_id=target_plan_id,
        target_workout_id=target_workout_id,
        payload=payload,
        summary=summary.model_dump(mode="json"),
        status=STATUS_PENDING,
        built_by=built_by,
        created_at=at,
        expires_at=at + PROPOSAL_TTL,
    )
    session.add(proposal)
    await session.commit()
    await session.refresh(proposal)
    _record(audit.DRAFTED, proposal, user_id, plan_id=target_plan_id)
    return proposal


async def _supersede_pending(
    session: AsyncSession, conversation_id: str, now: datetime
) -> int:
    result = await session.execute(
        select(PlanProposal).where(
            PlanProposal.conversation_id == conversation_id,
            PlanProposal.status == STATUS_PENDING,
        )
    )
    rows = list(result.scalars().all())
    for row in rows:
        row.status = STATUS_SUPERSEDED
        row.decided_at = now
    return len(rows)


# ── Reading ─────────────────────────────────────────────────────────────────


async def expire_lapsed(
    session: AsyncSession, *, now: Optional[datetime] = None
) -> int:
    """Settle every pending proposal whose 24 hours are up.

    Called on the thread read, the same moment ``settle_stuck_turns`` runs and
    the only moment anybody cares. Nothing depends on it having run: the apply
    path checks the deadline itself, because an offer that lapsed while the page
    was open must not be applicable merely because nobody read the thread.
    """
    at = now or _now()
    result = await session.execute(
        select(PlanProposal).where(PlanProposal.status == STATUS_PENDING)
    )
    lapsed = [
        row
        for row in result.scalars().all()
        if _aware(row.expires_at) is not None and _aware(row.expires_at) <= at
    ]
    for row in lapsed:
        row.status = STATUS_EXPIRED
        row.decided_at = at
    return len(lapsed)


async def proposals_by_message(
    session: AsyncSession, message_ids: Iterable[str]
) -> dict[str, PlanProposal]:
    """The proposal attached to each of these turns, keyed by message id."""
    ids = [mid for mid in message_ids if mid]
    if not ids:
        return {}
    result = await session.execute(
        select(PlanProposal).where(PlanProposal.message_id.in_(ids))
    )
    found: dict[str, PlanProposal] = {}
    for row in result.scalars().all():
        # One proposal per turn by construction (one per turn is a prompt rule
        # and the tool budget bounds it), but newest wins if that ever slips.
        current = found.get(row.message_id)
        if current is None or _aware(row.created_at) > _aware(current.created_at):
            found[row.message_id] = row
    return found


async def proposal_for_message(
    session: AsyncSession, conversation_id: str, message_id: str
) -> Optional[PlanProposal]:
    result = await session.execute(
        select(PlanProposal)
        .where(
            PlanProposal.conversation_id == conversation_id,
            PlanProposal.message_id == message_id,
        )
        .order_by(PlanProposal.created_at.desc())
    )
    return result.scalars().first()


async def discard_for_message(session: AsyncSession, message_id: str) -> int:
    """Delete the pending proposal a retried turn left behind.

    A retry re-runs a failed turn *in place*. Its old offer would otherwise sit
    under an answer that no longer exists, approvable out of context.
    """
    result = await session.execute(
        select(PlanProposal).where(
            PlanProposal.message_id == message_id,
            PlanProposal.status == STATUS_PENDING,
        )
    )
    rows = list(result.scalars().all())
    for row in rows:
        await session.delete(row)
    return len(rows)


async def delete_for_conversation(session: AsyncSession, conversation_id: str) -> int:
    """Delete a deleted conversation's proposals.

    Explicitly, like the messages: ``PRAGMA foreign_keys`` is off on these
    connections, so a cascade would be documentation rather than behaviour. An
    *applied* proposal's plan obviously survives — the plan is not part of the
    conversation.
    """
    result = await session.execute(
        select(PlanProposal).where(PlanProposal.conversation_id == conversation_id)
    )
    rows = list(result.scalars().all())
    for row in rows:
        await session.delete(row)
    return len(rows)


# ── The decision note replayed into the next turn's history ─────────────────


_NOTES = {
    STATUS_APPLIED: "[The athlete approved this; it has been applied and is now live.]",
    STATUS_DECLINED: "[The athlete declined this; nothing was changed.]",
    STATUS_EXPIRED: "[This offer lapsed without an answer; nothing was changed.]",
    STATUS_SUPERSEDED: "[This offer was replaced by a later one; nothing was changed.]",
    STATUS_PENDING: (
        "[This offer is still in front of the athlete, unanswered; nothing has "
        "been changed. Do not offer it again.]"
    ),
}


def decision_note(proposal: PlanProposal) -> Optional[str]:
    """What to append to the assistant turn that carried this proposal.

    Derived at render time from the row rather than written into
    ``chat_messages``, so it can never be stale, the dialogue stays pure
    dialogue, and the strict alternation :func:`~.llm_chat.build_wire_history`
    goes out of its way to preserve is untouched.
    """
    return _NOTES.get(proposal.status)


# ── Applying ────────────────────────────────────────────────────────────────


async def _load_plan(
    session: AsyncSession, athlete: Athlete, plan_id: Optional[str]
) -> Optional[TrainingPlan]:
    if not plan_id:
        return None
    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
        .options(selectinload(TrainingPlan.workouts))
    )
    return result.scalar_one_or_none()


def _archive_ids(summary: dict) -> set[str]:
    return {
        entry.get("plan_id")
        for entry in (summary or {}).get("archives") or []
        if entry.get("plan_id")
    }


async def _check_archive_set_unchanged(
    session: AsyncSession,
    athlete: Athlete,
    proposal: PlanProposal,
    start_date,
    end_date,
    *,
    ignore_plan_id: Optional[str] = None,
) -> list[TrainingPlan]:
    """Recompute what an approval would archive, and refuse if it has moved.

    The athlete agreed to file away a *named* set of plans. If a plan has been
    created, re-dated or archived since the preview, the set they consented to
    is not the set that would be filed — so this refuses rather than quietly
    archiving something nobody mentioned.
    """
    live = [
        plan
        for plan in await overlapping_active_plans(
            session, athlete.id, start_date, end_date
        )
        if plan.id != ignore_plan_id
    ]
    if {plan.id for plan in live} != _archive_ids(proposal.summary):
        raise ProposalError(
            CODE_STALE,
            "Your plans have changed since Koutsi drafted this, so approving it "
            "would archive a different set of plans than you were shown. Ask "
            "Koutsi again and it will draft it against what is there now.",
        )
    return live


async def apply_proposal(
    session: AsyncSession,
    athlete: Athlete,
    proposal: PlanProposal,
    *,
    today: date,
    now: Optional[datetime] = None,
    user_id: str = "",
) -> Optional[TrainingPlan]:
    """Write the change the athlete has just approved. **The only write path.**

    Idempotent by proposal id: a double-click produces one plan, and the second
    call answers with the plan the first created rather than a second plan or a
    500.

    Every invariant is re-run here rather than trusted from drafting time — see
    the module docstring on staleness. A refusal is audited on its way out, so
    the log carries the offers that were *not* taken up as well as the ones that
    were.
    """
    at = now or _now()

    def refuse(code: str, message: str) -> ProposalError:
        _record(audit.TOOL_ERROR, proposal, user_id, refusal_code=code)
        return ProposalError(code, message)

    if proposal.status == STATUS_APPLIED:
        applied = await _load_plan(session, athlete, proposal.applied_plan_id)
        _record(
            audit.APPROVED,
            proposal,
            user_id,
            plan_id=applied.id if applied is not None else None,
        )
        return applied
    if proposal.status == STATUS_EXPIRED:
        raise refuse(
            CODE_EXPIRED,
            "This offer has expired. Ask Koutsi again and it will draft a fresh "
            "one against your training as it stands now.",
        )
    if proposal.status != STATUS_PENDING:
        raise refuse(
            CODE_DECIDED,
            "This offer has already been dealt with, so there is nothing left to "
            "apply.",
        )

    deadline = _aware(proposal.expires_at)
    if deadline is not None and deadline <= at:
        proposal.status = STATUS_EXPIRED
        proposal.decided_at = at
        await session.commit()
        raise refuse(
            CODE_EXPIRED,
            "This offer has expired. Ask Koutsi again and it will draft a fresh "
            "one against your training as it stands now.",
        )

    try:
        if proposal.kind == KIND_CREATE_PLAN:
            plan = await _apply_create_plan(session, athlete, proposal, today=today)
        elif proposal.kind == KIND_UPDATE_PLAN:
            plan = await _apply_update_plan(session, athlete, proposal, now=at)
        elif proposal.kind == KIND_UPDATE_WORKOUT:
            plan = await _apply_update_workout(session, athlete, proposal)
        else:  # pragma: no cover - a kind no version of this code ever wrote
            raise ProposalError(
                CODE_STALE, "Koutsi cannot apply this offer; ask it again."
            )
    except ProposalError as exc:
        # Every invariant above refuses by raising, so this is the one place
        # that can record all of them without each check remembering to.
        _record(audit.TOOL_ERROR, proposal, user_id, refusal_code=exc.code)
        raise

    proposal.status = STATUS_APPLIED
    proposal.decided_at = at
    proposal.applied_plan_id = plan.id if plan is not None else None
    await session.commit()
    _record(audit.APPROVED, proposal, user_id, plan_id=proposal.applied_plan_id)

    # The same two follow-ups the REST write paths run: a plan's dates decide
    # whether the plan achievements have moved, and adherence is redrawn from
    # whatever is planned now.
    await mark_achievements_dirty(athlete.id, session)
    await catch_up_adherence(athlete.id, session)

    if plan is None:
        return None
    # Re-read rather than `refresh`: both calls above commit, which expires the
    # instance, and `TrainingPlan.workouts` is a lazy relationship — touching it
    # afterwards would need IO from a plain attribute read, which asyncio cannot
    # do (`MissingGreenlet`). The caller renders the plan with its sessions, so
    # it gets one that is actually loaded.
    return await _load_plan(session, athlete, plan.id)


async def _apply_create_plan(
    session: AsyncSession, athlete: Athlete, proposal: PlanProposal, *, today: date
) -> TrainingPlan:
    payload = proposal.payload or {}
    start = date.fromisoformat(payload["start_date"])
    weeks = int(payload["weeks"])

    if start < today:
        raise ProposalError(
            CODE_STALE,
            f"This plan was drafted to start on {start.isoformat()}, which is "
            "now in the past. Ask Koutsi for a new one starting from today.",
        )

    end = plan_end_date(start, weeks)
    await _check_archive_set_unchanged(session, athlete, proposal, start, end)
    await archive_overlapping_active_plans(session, athlete.id, start, end)

    plan = TrainingPlan(
        athlete_id=athlete.id,
        name=payload["name"],
        start_date=start,
        end_date=end,
        goal=payload.get("goal"),
        weeks=weeks,
        status="active",
        config=payload.get("config"),
        generation_method=payload.get("generation_method"),
        week_meta=payload.get("week_meta"),
    )
    session.add(plan)
    await session.flush()

    for week_number, days in enumerate(payload.get("weeks_data") or [], start=1):
        for day in days:
            session.add(
                PlannedWorkout(
                    plan_id=plan.id,
                    week_number=week_number,
                    day_of_week=int(day["day_of_week"]),
                    workout_type=day.get("workout_type"),
                    description=day.get("description"),
                    duration_min=day.get("duration_min"),
                    target_load=day.get("target_load"),
                )
            )
    await session.flush()
    return plan


async def _apply_update_plan(
    session: AsyncSession, athlete: Athlete, proposal: PlanProposal, *, now: datetime
) -> TrainingPlan:
    plan = await _load_plan(session, athlete, proposal.target_plan_id)
    if plan is None:
        raise ProposalError(
            CODE_STALE,
            "The plan this change applies to is no longer there, so there is "
            "nothing to change.",
        )

    # Captured before anything moves: only a plan that *becomes* active files
    # another away, which is what `POST /plans/{id}/unarchive` does. A plan that
    # was already active and is merely renamed archives nothing.
    was_active = plan.status == "active"
    changes = (proposal.payload or {}).get("changes") or {}
    if "name" in changes:
        plan.name = changes["name"]
    if "goal" in changes:
        plan.goal = changes["goal"]
    if "start_date" in changes:
        plan.start_date = date.fromisoformat(changes["start_date"])
    if "weeks" in changes:
        plan.weeks = int(changes["weeks"])

    # Same rule as `PUT /plans/{id}`: the plan's last day moved, so whether it
    # has finished is an open question again.
    if "start_date" in changes or "weeks" in changes:
        if plan.start_date and plan.weeks:
            plan.end_date = plan_end_date(plan.start_date, plan.weeks)
        plan.completed_at = None
        if plan.status == "completed":
            plan.status = "active"

    # Applied last, so an explicit status always wins over the reopen above.
    if "status" in changes:
        plan.status = changes["status"]
        if changes["status"] == "completed" and plan.completed_at is None:
            plan.completed_at = now

    if plan.status == "active" and not was_active:
        # Reopening a plan must not leave two overlapping plans active, exactly
        # as `POST /plans/{id}/unarchive` would not. The set was previewed and
        # is re-checked here against what is actually there — and the plan
        # itself is excluded, since a plan always overlaps its own dates.
        for other in await _check_archive_set_unchanged(
            session,
            athlete,
            proposal,
            plan.start_date,
            plan.end_date,
            ignore_plan_id=plan.id,
        ):
            other.status = "archived"

    await session.flush()
    return plan


async def _apply_update_workout(
    session: AsyncSession, athlete: Athlete, proposal: PlanProposal
) -> TrainingPlan:
    plan = await _load_plan(session, athlete, proposal.target_plan_id)
    if plan is None:
        raise ProposalError(
            CODE_STALE,
            "The plan this session belongs to is no longer there, so there is "
            "nothing to change.",
        )
    workout = next(
        (w for w in plan.workouts if w.id == proposal.target_workout_id), None
    )
    if workout is None:
        raise ProposalError(
            CODE_STALE,
            "That session is no longer in the plan, so there is nothing to "
            "change.",
        )
    if workout.is_completed:
        raise ProposalError(
            CODE_STALE,
            "That session has been completed since Koutsi drafted this, and a "
            "completed session is not edited or skipped.",
        )

    changes = (proposal.payload or {}).get("changes") or {}
    for field in (
        "workout_type",
        "description",
        "duration_min",
        "target_load",
        "day_of_week",
        "week_number",
    ):
        if field in changes:
            setattr(workout, field, changes[field])
    if "skip_reason" in changes:
        # Present-with-null is the unskip; absent means "leave it alone".
        workout.skip_reason = changes["skip_reason"]

    await session.flush()
    return plan


async def decline_proposal(
    session: AsyncSession,
    proposal: PlanProposal,
    *,
    now: Optional[datetime] = None,
    user_id: str = "",
) -> PlanProposal:
    """The other answer. Writes nothing but the decision."""
    at = now or _now()
    if proposal.status == STATUS_DECLINED:
        _record(audit.DECLINED, proposal, user_id)
        return proposal
    if proposal.status != STATUS_PENDING:
        _record(audit.TOOL_ERROR, proposal, user_id, refusal_code=CODE_DECIDED)
        raise ProposalError(
            CODE_DECIDED,
            "This offer has already been dealt with.",
        )
    proposal.status = STATUS_DECLINED
    proposal.decided_at = at
    await session.commit()
    _record(audit.DECLINED, proposal, user_id)
    return proposal


# ── Shared rendering helpers ────────────────────────────────────────────────


def session_date(plan: TrainingPlan, workout: PlannedWorkout) -> Optional[date]:
    """The calendar date a planned session falls on, when the plan has a start."""
    if plan.start_date is None:
        return None
    return workout_date(plan.start_date, workout.week_number, workout.day_of_week)
