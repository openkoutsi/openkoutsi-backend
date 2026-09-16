"""Koutsi drafts, the athlete approves, and only an approval writes (issue #72).

The two load-bearing tests are first, and they are the whole feature:

* **No tool reaches the plan tables.** Over every registered tool, published or
  not, the ``training_plans`` and ``planned_workouts`` row counts are identical
  either side of every call. This is the control — not a scope, and not the
  prompt. Koutsi calls tools as a session credential, which carries every scope
  implicitly, so ``plans:write`` would gate an external client and gate Koutsi by
  one sentence; and ``llm_chat`` says in its own words why a system prompt is a
  first line rather than a boundary.
* **Only the approval writes.** A pending proposal left alone — through a
  restart, through a later turn — never becomes a plan.

The rest is what has to be true for those two to mean anything: the apply path
re-runs its invariants, a decided offer cannot be applied twice, and a proposal
minted in one athlete's database does not exist in another's.
"""
import asyncio
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import func, select

from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.mcp.dispatch import ToolCaller, call_tool
from backend.app.mcp.limits import tool_limiter
from backend.app.mcp.registry import all_tools
from backend.app.models.chat_orm import (
    ROLE_ASSISTANT,
    ROLE_USER,
    STATUS_COMPLETE,
    STATUS_ERROR,
    ChatConversation,
    ChatMessage,
)
from backend.app.models.user_orm import (
    Athlete,
    PlanProposal,
    PlannedWorkout,
    TrainingPlan,
)
from backend.app.services import plan_proposals
from backend.app.services.plan_proposals import (
    PROPOSAL_TTL,
    ProposalError,
    apply_proposal,
)

_PREFIX = "/api/chat"
_TEST_USER_ID = "test-user-00000000"
_OTHER_USER_ID = "other-user-11111111"


@pytest.fixture(autouse=True)
def _clean_rate_limiter():
    tool_limiter.reset()
    yield
    tool_limiter.reset()


@pytest.fixture
def caller() -> ToolCaller:
    return ToolCaller(user_id=_TEST_USER_ID, scopes=None, kind="session")


@pytest.fixture
def scripted_turn(monkeypatch):
    """Drive a real turn against the scripted provider from the agent tests.

    The same fixture ``test_chat.py`` defines; duplicated rather than moved to
    ``conftest`` because it is two test modules' worth of use and moving it would
    put a stack of ``llm_agent`` internals in front of every test in the suite.
    """
    from backend.app.services import llm_agent
    from tests.unit.test_llm_agent import FakeDispatch, FakeProvider, FakeTool, _setup

    def _install(provider, dispatch=None, setup=None):
        resolved = setup or _setup(house_style=None)

        async def _resolve(athlete, user_id, *, usage_out=None):
            if usage_out is not None:
                usage_out["cfg"] = resolved.cfg
            return resolved

        monkeypatch.setattr(llm_agent, "stream_completion_events", provider)
        monkeypatch.setattr(llm_agent, "resolve_stream_setup", _resolve)
        monkeypatch.setattr(llm_agent, "call_tool", dispatch or FakeDispatch())

    _install.provider = FakeProvider
    _install.dispatch = FakeDispatch
    _install.tool = FakeTool
    _install.setup = _setup
    return _install


@pytest.fixture
def no_turns():
    """Accept turns without running them — these are not model tests."""
    with patch("backend.app.api.chat.run_chat_turn_bg") as spawn:
        async def _noop(*args, **kwargs):
            return None

        spawn.side_effect = _noop
        yield spawn


async def _run(name, args, *, caller, session, athlete, registry_session, **kwargs):
    return await call_tool(
        caller, name, args, session=session, athlete=athlete,
        registry_session=registry_session, **kwargs,
    )


@pytest.fixture
async def planned_athlete(session, seeded_athlete):
    """An athlete with one active plan, so a change has something to change."""
    today = date.today()
    plan = TrainingPlan(
        id="plan-1",
        athlete_id=seeded_athlete.id,
        name="Spring base",
        goal="Build aerobic base",
        start_date=today - timedelta(days=7),
        end_date=today + timedelta(days=20),
        weeks=4,
        status="active",
    )
    session.add(plan)
    session.add_all(
        [
            PlannedWorkout(
                id="pw-thursday", plan_id=plan.id, week_number=2, day_of_week=4,
                workout_type="threshold", description="4x8", duration_min=75,
                target_load=95,
            ),
            PlannedWorkout(
                id="pw-saturday", plan_id=plan.id, week_number=2, day_of_week=6,
                workout_type="long", duration_min=180, target_load=150,
            ),
        ]
    )
    await session.commit()
    return seeded_athlete


async def _plan_rows(session) -> tuple[int, int]:
    plans = (
        await session.execute(select(func.count()).select_from(TrainingPlan))
    ).scalar_one()
    workouts = (
        await session.execute(select(func.count()).select_from(PlannedWorkout))
    ).scalar_one()
    return plans, workouts


#: The arguments each tool needs to do its job over ``planned_athlete``. Kept in
#: one place so the sweep below cannot quietly skip a tool by failing to call it.
TOOL_ARGS: dict[str, dict] = {
    "get_activity_detail": {"activity_id": "nope"},
    "propose_training_plan": {
        "name": "October fondo",
        "start_date": (date.today() + timedelta(days=7)).isoformat(),
        "weeks": 6,
        "goal": "Hilly gran fondo",
    },
    "propose_plan_change": {
        "plan_id": "plan-1",
        "change": "update_workout",
        "workout_id": "pw-thursday",
        "duration_min": 55,
    },
}


# ── The control ─────────────────────────────────────────────────────────────


async def test_no_tool_writes_to_the_plan_tables(
    caller, session, planned_athlete, registry_session
):
    """**The load-bearing test.**

    Every registered tool, called in turn. Whatever any of them does, the plan
    tables are byte-identical afterwards — which is what makes "the model never
    performs the write" a fact about the code rather than an instruction.

    Guarded against passing vacuously three ways: the propose tools must be in
    the swept set, their calls must have succeeded, and they must have inserted
    the ``plan_proposals`` rows they exist to insert.
    """
    swept = {t.name for t in all_tools()}
    assert {"propose_training_plan", "propose_plan_change"} <= swept

    before = await _plan_rows(session)
    outcomes = {}
    for name in sorted(swept):
        outcomes[name] = await _run(
            name, TOOL_ARGS.get(name, {}), caller=caller, session=session,
            athlete=planned_athlete, registry_session=registry_session,
        )
        assert await _plan_rows(session) == before, f"{name} touched the plan tables"

    assert outcomes["propose_training_plan"].ok, outcomes["propose_training_plan"].error
    assert outcomes["propose_plan_change"].ok, outcomes["propose_plan_change"].error

    proposals = (
        (await session.execute(select(PlanProposal))).scalars().all()
    )
    assert {p.kind for p in proposals} == {"create_plan", "update_workout"}


async def test_a_proposal_is_not_a_plan_anywhere_it_could_be_mistaken_for_one(
    caller, session, seeded_athlete, registry_session
):
    """Inert by construction rather than by a flag someone must remember.

    A proposal is not a ``TrainingPlan`` row, so the plan page, the adherence
    snapshots, the activity matcher and the achievements cannot see it — none of
    them has to be taught to exclude it.
    """
    result = await _run(
        "propose_training_plan", TOOL_ARGS["propose_training_plan"], caller=caller,
        session=session, athlete=seeded_athlete, registry_session=registry_session,
    )
    assert result.ok, result.error
    assert (await _plan_rows(session)) == (0, 0)
    assert (
        await session.execute(select(func.count()).select_from(PlanProposal))
    ).scalar_one() == 1


# ── The draft ───────────────────────────────────────────────────────────────


async def test_the_preview_names_the_plans_an_approval_would_archive(
    caller, session, planned_athlete, registry_session
):
    """Uninformed consent is the failure this exists to prevent.

    ``create_plan`` archives every active plan whose dates overlap the new one,
    so a yes/no that did not say so would be a yes to something the athlete was
    never told about.
    """
    args = dict(TOOL_ARGS["propose_training_plan"])
    args["start_date"] = date.today().isoformat()  # overlaps the existing plan
    result = await _run(
        "propose_training_plan", args, caller=caller, session=session,
        athlete=planned_athlete, registry_session=registry_session,
    )
    archives = result.data["summary"]["archives"]
    assert [a["plan_id"] for a in archives] == ["plan-1"]
    assert archives[0]["name"] == "Spring base"
    # And the model is told, in words, to pass it on.
    assert "Spring base" in result.data["note"]


async def test_a_plan_that_overlaps_nothing_archives_nothing(
    caller, session, planned_athlete, registry_session
):
    args = dict(TOOL_ARGS["propose_training_plan"])
    args["start_date"] = (date.today() + timedelta(days=60)).isoformat()
    result = await _run(
        "propose_training_plan", args, caller=caller, session=session,
        athlete=planned_athlete, registry_session=registry_session,
    )
    assert result.data["summary"]["archives"] == []


async def test_a_plan_cannot_be_drafted_to_start_in_the_past(
    caller, session, seeded_athlete, registry_session
):
    args = dict(TOOL_ARGS["propose_training_plan"])
    args["start_date"] = (date.today() - timedelta(days=1)).isoformat()
    result = await _run(
        "propose_training_plan", args, caller=caller, session=session,
        athlete=seeded_athlete, registry_session=registry_session,
    )
    assert not result.ok
    assert "in the past" in result.error


async def test_a_change_to_a_completed_session_is_refused_with_a_sentence(
    caller, session, planned_athlete, registry_session
):
    """A completed session is not edited or skipped — the REST routes say so
    with a 409, and the model has to be told the same thing in prose."""
    from backend.app.models.user_orm import Activity, PlannedWorkoutActivity

    session.add(
        Activity(
            id="act-1", athlete_id=planned_athlete.id,
            start_time=datetime.now(timezone.utc), sport_type="Ride",
        )
    )
    await session.flush()
    session.add(
        PlannedWorkoutActivity(planned_workout_id="pw-thursday", activity_id="act-1")
    )
    await session.commit()

    result = await _run(
        "propose_plan_change", TOOL_ARGS["propose_plan_change"], caller=caller,
        session=session, athlete=planned_athlete, registry_session=registry_session,
    )
    assert not result.ok
    assert "completed" in result.error


async def test_a_change_that_changes_nothing_is_refused(
    caller, session, planned_athlete, registry_session
):
    result = await _run(
        "propose_plan_change",
        {"plan_id": "plan-1", "change": "update_plan", "name": "Spring base"},
        caller=caller, session=session, athlete=planned_athlete,
        registry_session=registry_session,
    )
    assert not result.ok
    assert "exactly as it is" in result.error


async def test_a_skip_needs_a_reason(
    caller, session, planned_athlete, registry_session
):
    result = await _run(
        "propose_plan_change",
        {"plan_id": "plan-1", "change": "skip_workout", "workout_id": "pw-thursday"},
        caller=caller, session=session, athlete=planned_athlete,
        registry_session=registry_session,
    )
    assert not result.ok
    assert "reason" in result.error


async def test_moving_a_plan_shows_what_it_moves_from_and_to(
    caller, session, planned_athlete, registry_session
):
    new_start = date.today() + timedelta(days=7)
    result = await _run(
        "propose_plan_change",
        {
            "plan_id": "plan-1",
            "change": "update_plan",
            "start_date": new_start.isoformat(),
        },
        caller=caller, session=session, athlete=planned_athlete,
        registry_session=registry_session,
    )
    assert result.ok, result.error
    changes = {c["field"]: c for c in result.data["summary"]["changes"]}
    assert changes["start_date"]["after"] == new_start.isoformat()
    assert changes["start_date"]["before"] == (date.today() - timedelta(days=7)).isoformat()


# ── Superseding ─────────────────────────────────────────────────────────────


async def test_a_second_proposal_supersedes_the_first(
    caller, session, planned_athlete, registry_session
):
    for _ in range(2):
        result = await _run(
            "propose_training_plan", TOOL_ARGS["propose_training_plan"],
            caller=caller, session=session, athlete=planned_athlete,
            registry_session=registry_session,
            conversation_id="conv-1", message_id="msg-1",
        )
        assert result.ok, result.error

    rows = (
        (
            await session.execute(
                select(PlanProposal).order_by(PlanProposal.created_at)
            )
        )
        .scalars()
        .all()
    )
    assert [r.status for r in rows] == ["superseded", "pending"]


async def test_a_proposal_in_another_conversation_is_left_alone(
    caller, session, planned_athlete, registry_session
):
    """Superseding is per thread. Two conversations are two separate offers."""
    for conversation in ("conv-1", "conv-2"):
        await _run(
            "propose_training_plan", TOOL_ARGS["propose_training_plan"],
            caller=caller, session=session, athlete=planned_athlete,
            registry_session=registry_session,
            conversation_id=conversation, message_id=f"msg-{conversation}",
        )
    rows = (await session.execute(select(PlanProposal))).scalars().all()
    assert {r.status for r in rows} == {"pending"}


# ── Applying ────────────────────────────────────────────────────────────────


async def _draft(session, athlete, caller, registry_session, **overrides):
    args = dict(TOOL_ARGS["propose_training_plan"])
    args.update(overrides)
    result = await _run(
        "propose_training_plan", args, caller=caller, session=session,
        athlete=athlete, registry_session=registry_session,
        conversation_id="conv-1", message_id="msg-1",
    )
    assert result.ok, result.error
    return (
        await session.execute(
            select(PlanProposal).where(PlanProposal.id == result.data["proposal_id"])
        )
    ).scalar_one()


async def test_only_the_approval_writes(
    caller, session, planned_athlete, registry_session
):
    """**The second load-bearing test.**

    A pending proposal, left alone, is not a plan — not after the turn that
    drafted it, not after a later turn, not after the process that wrote it has
    gone. Nothing in the system turns it into one except an athlete's yes.
    """
    proposal = await _draft(session, planned_athlete, caller, registry_session)
    before = await _plan_rows(session)

    # A later turn, and every read path anyone might expect to settle it.
    for name in ("get_plan_status", "get_training_status"):
        await _run(
            name, {}, caller=caller, session=session, athlete=planned_athlete,
            registry_session=registry_session,
        )
    await plan_proposals.expire_lapsed(session)
    await session.commit()
    assert await _plan_rows(session) == before

    # And then the approval, which is the one thing that does.
    plan = await apply_proposal(
        session, planned_athlete, proposal, today=date.today()
    )
    assert plan is not None
    assert (await _plan_rows(session))[0] == before[0] + 1


async def test_applying_twice_produces_one_plan(
    caller, session, planned_athlete, registry_session
):
    """A double-click is one plan, and the second call answers with it."""
    proposal = await _draft(session, planned_athlete, caller, registry_session)
    first = await apply_proposal(session, planned_athlete, proposal, today=date.today())
    after_first = await _plan_rows(session)
    second = await apply_proposal(session, planned_athlete, proposal, today=date.today())
    assert second is not None
    assert second.id == first.id
    assert await _plan_rows(session) == after_first


async def test_an_approval_archives_exactly_what_the_preview_named(
    caller, session, planned_athlete, registry_session
):
    proposal = await _draft(
        session, planned_athlete, caller, registry_session,
        start_date=date.today().isoformat(),
    )
    assert [a["plan_id"] for a in proposal.summary["archives"]] == ["plan-1"]

    await apply_proposal(session, planned_athlete, proposal, today=date.today())
    old = (
        await session.execute(select(TrainingPlan).where(TrainingPlan.id == "plan-1"))
    ).scalar_one()
    assert old.status == "archived"


@pytest.mark.parametrize("status", ["declined", "expired", "superseded"])
async def test_a_decided_proposal_cannot_be_applied(
    status, caller, session, planned_athlete, registry_session
):
    proposal = await _draft(session, planned_athlete, caller, registry_session)
    proposal.status = status
    await session.commit()

    with pytest.raises(ProposalError) as exc:
        await apply_proposal(session, planned_athlete, proposal, today=date.today())
    assert exc.value.code in {"proposal_decided", "proposal_expired"}
    assert (await _plan_rows(session))[0] == 1  # only the fixture's plan


async def test_an_offer_past_its_deadline_is_refused_even_unread(
    caller, session, planned_athlete, registry_session
):
    """``expires_at`` is checked at apply, not merely swept on a thread read.

    An offer that lapsed while the page sat open must not be applicable because
    nobody happened to reload it.
    """
    proposal = await _draft(session, planned_athlete, caller, registry_session)
    proposal.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    await session.commit()

    with pytest.raises(ProposalError) as exc:
        await apply_proposal(session, planned_athlete, proposal, today=date.today())
    assert exc.value.code == "proposal_expired"
    assert proposal.status == "expired"


async def test_the_offer_stands_for_a_day(
    caller, session, planned_athlete, registry_session
):
    proposal = await _draft(session, planned_athlete, caller, registry_session)
    assert PROPOSAL_TTL == timedelta(hours=24)
    delta = proposal.expires_at - proposal.created_at
    assert delta == PROPOSAL_TTL


# ── Staleness, re-checked at apply ──────────────────────────────────────────


async def test_a_plan_whose_start_has_passed_is_refused(
    caller, session, planned_athlete, registry_session
):
    proposal = await _draft(session, planned_athlete, caller, registry_session)
    later = date.fromisoformat(proposal.payload["start_date"]) + timedelta(days=1)

    with pytest.raises(ProposalError) as exc:
        await apply_proposal(session, planned_athlete, proposal, today=later)
    assert exc.value.code == "proposal_stale"
    assert "in the past" in str(exc.value)


async def test_an_archive_set_that_moved_since_the_preview_is_refused(
    caller, session, planned_athlete, registry_session
):
    """The athlete consented to archiving a *named* set of plans.

    A plan created since would be filed away without ever having been mentioned,
    which is exactly the consent the preview exists to obtain.
    """
    proposal = await _draft(
        session, planned_athlete, caller, registry_session,
        start_date=date.today().isoformat(),
    )
    session.add(
        TrainingPlan(
            id="plan-2", athlete_id=planned_athlete.id, name="Sneaky block",
            start_date=date.today(), end_date=date.today() + timedelta(days=30),
            weeks=4, status="active",
        )
    )
    await session.commit()

    with pytest.raises(ProposalError) as exc:
        await apply_proposal(session, planned_athlete, proposal, today=date.today())
    assert exc.value.code == "proposal_stale"
    assert (await _plan_rows(session))[0] == 2  # nothing created, nothing archived


async def test_a_change_whose_target_plan_is_gone_is_refused(
    caller, session, planned_athlete, registry_session
):
    result = await _run(
        "propose_plan_change", TOOL_ARGS["propose_plan_change"], caller=caller,
        session=session, athlete=planned_athlete, registry_session=registry_session,
    )
    proposal = (
        await session.execute(
            select(PlanProposal).where(PlanProposal.id == result.data["proposal_id"])
        )
    ).scalar_one()

    plan = (
        await session.execute(select(TrainingPlan).where(TrainingPlan.id == "plan-1"))
    ).scalar_one()
    await session.delete(plan)
    await session.commit()

    with pytest.raises(ProposalError) as exc:
        await apply_proposal(session, planned_athlete, proposal, today=date.today())
    assert exc.value.code == "proposal_stale"


async def test_a_session_completed_since_drafting_is_not_edited(
    caller, session, planned_athlete, registry_session
):
    from backend.app.models.user_orm import Activity, PlannedWorkoutActivity

    result = await _run(
        "propose_plan_change", TOOL_ARGS["propose_plan_change"], caller=caller,
        session=session, athlete=planned_athlete, registry_session=registry_session,
    )
    proposal = (
        await session.execute(
            select(PlanProposal).where(PlanProposal.id == result.data["proposal_id"])
        )
    ).scalar_one()

    session.add(
        Activity(
            id="act-1", athlete_id=planned_athlete.id,
            start_time=datetime.now(timezone.utc), sport_type="Ride",
        )
    )
    await session.flush()
    session.add(
        PlannedWorkoutActivity(planned_workout_id="pw-thursday", activity_id="act-1")
    )
    await session.commit()

    with pytest.raises(ProposalError) as exc:
        await apply_proposal(session, planned_athlete, proposal, today=date.today())
    assert exc.value.code == "proposal_stale"


async def test_a_workout_change_is_applied_to_the_right_session(
    caller, session, planned_athlete, registry_session
):
    result = await _run(
        "propose_plan_change", TOOL_ARGS["propose_plan_change"], caller=caller,
        session=session, athlete=planned_athlete, registry_session=registry_session,
    )
    proposal = (
        await session.execute(
            select(PlanProposal).where(PlanProposal.id == result.data["proposal_id"])
        )
    ).scalar_one()
    await apply_proposal(session, planned_athlete, proposal, today=date.today())

    workout = (
        await session.execute(
            select(PlannedWorkout).where(PlannedWorkout.id == "pw-thursday")
        )
    ).scalar_one()
    assert workout.duration_min == 55
    untouched = (
        await session.execute(
            select(PlannedWorkout).where(PlannedWorkout.id == "pw-saturday")
        )
    ).scalar_one()
    assert untouched.duration_min == 180


async def test_a_skip_and_then_an_unskip_round_trip(
    caller, session, planned_athlete, registry_session
):
    async def decide(args):
        result = await _run(
            "propose_plan_change", args, caller=caller, session=session,
            athlete=planned_athlete, registry_session=registry_session,
        )
        assert result.ok, result.error
        proposal = (
            await session.execute(
                select(PlanProposal).where(
                    PlanProposal.id == result.data["proposal_id"]
                )
            )
        ).scalar_one()
        await apply_proposal(session, planned_athlete, proposal, today=date.today())

    await decide(
        {
            "plan_id": "plan-1", "change": "skip_workout",
            "workout_id": "pw-thursday", "skip_reason": "Woke up ill",
        }
    )
    workout = (
        await session.execute(
            select(PlannedWorkout).where(PlannedWorkout.id == "pw-thursday")
        )
    ).scalar_one()
    assert workout.skip_reason == "Woke up ill"

    await decide(
        {"plan_id": "plan-1", "change": "unskip_workout", "workout_id": "pw-thursday"}
    )
    await session.refresh(workout)
    assert workout.skip_reason is None


async def test_renaming_an_active_plan_archives_nothing(
    caller, session, planned_athlete, registry_session
):
    """Only a **reopen** files another plan away, as `unarchive_plan` does.

    Computing the overlap set for every change would quietly turn "fix this
    typo" into "archive the other plan you are following" — the plan is already
    active, so nothing about its place in the athlete's training has moved.
    """
    session.add(
        TrainingPlan(
            id="plan-2", athlete_id=planned_athlete.id, name="Parallel block",
            start_date=date.today(), end_date=date.today() + timedelta(days=30),
            weeks=4, status="active",
        )
    )
    await session.commit()

    result = await _run(
        "propose_plan_change",
        {"plan_id": "plan-1", "change": "update_plan", "name": "Spring base II"},
        caller=caller, session=session, athlete=planned_athlete,
        registry_session=registry_session,
    )
    assert result.ok, result.error
    assert result.data["summary"]["archives"] == []
    assert result.data["summary"]["reopens_plan"] is False

    proposal = (
        await session.execute(
            select(PlanProposal).where(PlanProposal.id == result.data["proposal_id"])
        )
    ).scalar_one()
    await apply_proposal(session, planned_athlete, proposal, today=date.today())

    other = (
        await session.execute(select(TrainingPlan).where(TrainingPlan.id == "plan-2"))
    ).scalar_one()
    assert other.status == "active"


async def test_reopening_a_plan_says_so_and_names_what_it_would_archive(
    caller, session, planned_athlete, registry_session
):
    """Reactivating a filed-away plan *is* something the athlete should be told,
    and it archives whatever it overlaps — exactly as unarchiving does."""
    plan = (
        await session.execute(select(TrainingPlan).where(TrainingPlan.id == "plan-1"))
    ).scalar_one()
    plan.status = "archived"
    session.add(
        TrainingPlan(
            id="plan-2", athlete_id=planned_athlete.id, name="Parallel block",
            start_date=date.today(), end_date=date.today() + timedelta(days=30),
            weeks=4, status="active",
        )
    )
    await session.commit()

    result = await _run(
        "propose_plan_change",
        {"plan_id": "plan-1", "change": "update_plan", "status": "active"},
        caller=caller, session=session, athlete=planned_athlete,
        registry_session=registry_session,
    )
    assert result.ok, result.error
    assert result.data["summary"]["reopens_plan"] is True
    assert [a["plan_id"] for a in result.data["summary"]["archives"]] == ["plan-2"]

    proposal = (
        await session.execute(
            select(PlanProposal).where(PlanProposal.id == result.data["proposal_id"])
        )
    ).scalar_one()
    await apply_proposal(session, planned_athlete, proposal, today=date.today())

    statuses = {
        row.id: row.status
        for row in (await session.execute(select(TrainingPlan))).scalars().all()
    }
    assert statuses == {"plan-1": "active", "plan-2": "archived"}


async def test_declining_writes_nothing_but_the_decision(
    caller, session, planned_athlete, registry_session
):
    proposal = await _draft(session, planned_athlete, caller, registry_session)
    before = await _plan_rows(session)
    await plan_proposals.decline_proposal(session, proposal)
    assert proposal.status == "declined"
    assert proposal.decided_at is not None
    assert await _plan_rows(session) == before


# ── The decision, over HTTP ─────────────────────────────────────────────────
#
# The chat routes resolve their own per-user session rather than the in-memory
# override, so these drive the real file-backed database the app uses — which is
# also where cross-user isolation stops being a predicate and becomes a file.


async def _seed_file_backed(user_id: str = _TEST_USER_ID, *, with_plan: bool = True):
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as s:
        athlete = Athlete(
            global_user_id=user_id, ftp_tests=[],
            app_settings={"agentic_koutsi": True},
        )
        s.add(athlete)
        await s.flush()
        if with_plan:
            plan = TrainingPlan(
                id="plan-1", athlete_id=athlete.id, name="Spring base",
                start_date=date.today() - timedelta(days=7),
                end_date=date.today() + timedelta(days=20),
                weeks=4, status="active",
            )
            s.add(plan)
            s.add(
                PlannedWorkout(
                    id="pw-thursday", plan_id=plan.id, week_number=2, day_of_week=4,
                    workout_type="threshold", duration_min=75, target_load=95,
                )
            )
        await s.commit()
        return athlete.id


async def _seed_turn(user_id: str = _TEST_USER_ID, *, content: str = "Here you go."):
    """A finished assistant turn, ready for a proposal to hang off."""
    now = datetime.now(timezone.utc)
    async with get_user_session_factory(user_id)() as s:
        conversation = ChatConversation(created_at=now, updated_at=now)
        s.add(conversation)
        await s.flush()
        question = ChatMessage(
            conversation_id=conversation.id, role=ROLE_USER,
            content="Build me something for October", created_at=now, updated_at=now,
        )
        answer = ChatMessage(
            conversation_id=conversation.id, role=ROLE_ASSISTANT, content=content,
            status=STATUS_COMPLETE, created_at=now, updated_at=now,
        )
        s.add_all([question, answer])
        await s.commit()
        return conversation.id, answer.id


def _create_payload(start: date, weeks: int = 4, archives=()) -> dict:
    days = [
        {
            "day_of_week": day,
            "workout_type": "endurance" if day in (2, 6) else "rest",
            "description": "Steady Zone 2" if day in (2, 6) else None,
            "duration_min": 90 if day in (2, 6) else None,
            "target_load": 80 if day in (2, 6) else None,
        }
        for day in range(1, 8)
    ]
    return {
        "name": "October fondo",
        "goal": "Hilly gran fondo",
        "start_date": start.isoformat(),
        "weeks": weeks,
        "config": None,
        "week_meta": None,
        "weeks_data": [days for _ in range(weeks)],
        "generation_method": "rule_based",
    }


def _create_summary(start: date, weeks: int = 4, archives=()) -> dict:
    return {
        "kind": "create_plan",
        "built_by": "rule_based",
        "fallback_reason": None,
        "plan_name": "October fondo",
        "goal": "Hilly gran fondo",
        "start_date": start.isoformat(),
        "end_date": (start + timedelta(weeks=weeks) - timedelta(days=1)).isoformat(),
        "weeks": weeks,
        "weekly": [],
        "first_week": [],
        "remaining_weeks": weeks - 1,
        "changes": [],
        "target_plan_id": None,
        "target_workout_id": None,
        "target_date": None,
        "target_label": None,
        "reopens_plan": False,
        "archives": list(archives),
    }


async def _attach_proposal(
    conversation_id: str,
    message_id: str,
    *,
    user_id: str = _TEST_USER_ID,
    start: date | None = None,
    status: str = "pending",
    archives=(),
):
    start = start or (date.today() + timedelta(days=7))
    now = datetime.now(timezone.utc)
    async with get_user_session_factory(user_id)() as s:
        proposal = PlanProposal(
            conversation_id=conversation_id,
            message_id=message_id,
            kind="create_plan",
            payload=_create_payload(start),
            summary=_create_summary(start, archives=archives),
            status=status,
            built_by="rule_based",
            created_at=now,
            expires_at=now + PROPOSAL_TTL,
        )
        s.add(proposal)
        await s.commit()
        return proposal.id


async def _proposal_rows(user_id: str = _TEST_USER_ID):
    async with get_user_session_factory(user_id)() as s:
        result = await s.execute(select(PlanProposal))
        return list(result.scalars().all())


async def _plans(user_id: str = _TEST_USER_ID):
    async with get_user_session_factory(user_id)() as s:
        result = await s.execute(select(TrainingPlan))
        return list(result.scalars().all())


class TestDecidingOverHttp:
    async def test_the_thread_carries_the_offer_under_the_turn_that_made_it(
        self, client, auth_headers
    ):
        await _seed_file_backed()
        conversation_id, message_id = await _seed_turn()
        proposal_id = await _attach_proposal(conversation_id, message_id)

        body = (
            await client.get(
                f"{_PREFIX}/conversations/{conversation_id}", headers=auth_headers
            )
        ).json()
        question, answer = body["messages"]
        assert question["proposal"] is None
        assert answer["proposal"]["id"] == proposal_id
        assert answer["proposal"]["status"] == "pending"
        assert answer["proposal"]["summary"]["plan_name"] == "October fondo"

    async def test_an_approval_creates_the_plan_and_hands_it_back(
        self, client, auth_headers
    ):
        await _seed_file_backed(with_plan=False)
        conversation_id, message_id = await _seed_turn()
        await _attach_proposal(conversation_id, message_id)

        resp = await client.post(
            f"{_PREFIX}/conversations/{conversation_id}/messages/{message_id}"
            "/proposal/approve",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["proposal"]["status"] == "applied"
        assert body["plan"]["name"] == "October fondo"
        # The athlete's next step is a link to the thing they just accepted.
        assert body["proposal"]["applied_plan_id"] == body["plan"]["id"]
        assert len(await _plans()) == 1

    async def test_a_decline_writes_nothing(self, client, auth_headers):
        await _seed_file_backed(with_plan=False)
        conversation_id, message_id = await _seed_turn()
        await _attach_proposal(conversation_id, message_id)

        resp = await client.post(
            f"{_PREFIX}/conversations/{conversation_id}/messages/{message_id}"
            "/proposal/decline",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["proposal"]["status"] == "declined"
        assert resp.json()["plan"] is None
        assert await _plans() == []

    async def test_a_decision_spends_no_chat_turn(self, client, auth_headers):
        """Approving is not a question. It costs nothing from the daily budget,
        cannot fail on a slow model, and asks nothing of one."""
        await _seed_file_backed(with_plan=False)
        conversation_id, message_id = await _seed_turn()
        await _attach_proposal(conversation_id, message_id)

        before = (
            await client.get(f"{_PREFIX}/availability", headers=auth_headers)
        ).json()["turns_remaining_today"]
        await client.post(
            f"{_PREFIX}/conversations/{conversation_id}/messages/{message_id}"
            "/proposal/approve",
            headers=auth_headers,
        )
        after = (
            await client.get(f"{_PREFIX}/availability", headers=auth_headers)
        ).json()["turns_remaining_today"]
        assert after == before

    async def test_a_stale_offer_is_refused_with_a_code_the_ui_can_render(
        self, client, auth_headers
    ):
        await _seed_file_backed(with_plan=False)
        conversation_id, message_id = await _seed_turn()
        await _attach_proposal(
            conversation_id, message_id, start=date.today() - timedelta(days=1)
        )

        resp = await client.post(
            f"{_PREFIX}/conversations/{conversation_id}/messages/{message_id}"
            "/proposal/approve",
            headers=auth_headers,
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "proposal_stale"
        assert await _plans() == []

    async def test_an_offer_on_a_turn_that_never_made_one_is_a_404(
        self, client, auth_headers
    ):
        await _seed_file_backed(with_plan=False)
        conversation_id, message_id = await _seed_turn()

        resp = await client.post(
            f"{_PREFIX}/conversations/{conversation_id}/messages/{message_id}"
            "/proposal/approve",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_a_proposal_from_another_athletes_database_does_not_exist_here(
        self, client, auth_headers
    ):
        """Isolation is the database file, not a predicate.

        A conversation and a proposal minted in another user's DB are simply not
        in this one, so an approval carrying them 404s without anything here
        having to remember to filter.
        """
        await _seed_file_backed(with_plan=False)
        await _seed_file_backed(_OTHER_USER_ID, with_plan=False)
        other_conversation, other_message = await _seed_turn(_OTHER_USER_ID)
        await _attach_proposal(
            other_conversation, other_message, user_id=_OTHER_USER_ID
        )

        resp = await client.post(
            f"{_PREFIX}/conversations/{other_conversation}/messages/{other_message}"
            "/proposal/approve",
            headers=auth_headers,
        )
        assert resp.status_code == 404
        assert await _plans() == []
        # And the other athlete's offer is untouched.
        assert [p.status for p in await _proposal_rows(_OTHER_USER_ID)] == ["pending"]

    async def test_deleting_a_conversation_deletes_its_offers(
        self, client, auth_headers, no_turns
    ):
        await _seed_file_backed(with_plan=False)
        conversation_id, message_id = await _seed_turn()
        await _attach_proposal(conversation_id, message_id)

        resp = await client.delete(
            f"{_PREFIX}/conversations/{conversation_id}", headers=auth_headers
        )
        assert resp.status_code == 204
        assert await _proposal_rows() == []

    async def test_an_applied_offers_plan_survives_the_conversation(
        self, client, auth_headers, no_turns
    ):
        """The plan is the athlete's training, not part of the thread."""
        await _seed_file_backed(with_plan=False)
        conversation_id, message_id = await _seed_turn()
        await _attach_proposal(conversation_id, message_id)
        await client.post(
            f"{_PREFIX}/conversations/{conversation_id}/messages/{message_id}"
            "/proposal/approve",
            headers=auth_headers,
        )
        await client.delete(
            f"{_PREFIX}/conversations/{conversation_id}", headers=auth_headers
        )
        assert await _proposal_rows() == []
        assert [p.name for p in await _plans()] == ["October fondo"]

    async def test_retrying_a_failed_turn_discards_its_pending_offer(
        self, client, auth_headers, no_turns
    ):
        """Otherwise a retried turn leaves an orphan the athlete can approve out
        of context — a plan created from a reply that no longer exists."""
        await _seed_file_backed(with_plan=False)
        conversation_id, message_id = await _seed_turn(content="")
        await _attach_proposal(conversation_id, message_id)
        async with get_user_session_factory(_TEST_USER_ID)() as s:
            row = (
                await s.execute(
                    select(ChatMessage).where(ChatMessage.id == message_id)
                )
            ).scalar_one()
            row.status = STATUS_ERROR
            row.error_code = "upstream"
            await s.commit()

        resp = await client.post(
            f"{_PREFIX}/conversations/{conversation_id}/messages/{message_id}/retry",
            json={}, headers=auth_headers,
        )
        assert resp.status_code == 202, resp.text
        assert await _proposal_rows() == []

    async def test_an_offer_nobody_answered_lapses_on_the_next_read(
        self, client, auth_headers
    ):
        await _seed_file_backed(with_plan=False)
        conversation_id, message_id = await _seed_turn()
        await _attach_proposal(conversation_id, message_id)
        async with get_user_session_factory(_TEST_USER_ID)() as s:
            row = (await s.execute(select(PlanProposal))).scalar_one()
            row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
            await s.commit()

        body = (
            await client.get(
                f"{_PREFIX}/conversations/{conversation_id}", headers=auth_headers
            )
        ).json()
        assert body["messages"][1]["proposal"]["status"] == "expired"

    async def test_an_approval_is_audited_without_the_plans_contents(
        self, client, auth_headers, caplog
    ):
        import logging

        await _seed_file_backed(with_plan=False)
        conversation_id, message_id = await _seed_turn()
        proposal_id = await _attach_proposal(conversation_id, message_id)

        with caplog.at_level(logging.INFO, logger="openkoutsi.audit"):
            await client.post(
                f"{_PREFIX}/conversations/{conversation_id}/messages/{message_id}"
                "/proposal/approve",
                headers=auth_headers,
            )

        record = next(
            r for r in caplog.records
            if getattr(r, "event", None) == "plan_proposal"
        )
        assert record.proposal_outcome == "approved"
        assert record.proposal_id == proposal_id
        assert record.proposal_plan_id
        # The decision and what it produced — never what the plan says.
        assert "October fondo" not in record.getMessage()
        assert not hasattr(record, "proposal_payload")


async def test_drafting_and_deciding_share_a_key_in_the_audit_log(
    caller, session, planned_athlete, registry_session, caplog
):
    """`mcp_tool_call` records the invocation but never the result, and the
    proposal id lives only in the result — so drafting writes its own record,
    keyed on that id, and the decision joins to it.

    Without the join, "what did this account agree to?" is two half-answers: a
    tool call with no outcome, and an outcome with no provenance.
    """
    import logging

    with caplog.at_level(logging.INFO, logger="openkoutsi.audit"):
        proposal = await _draft(session, planned_athlete, caller, registry_session)
        await apply_proposal(session, planned_athlete, proposal, today=date.today())

    records = [
        r for r in caplog.records if getattr(r, "event", None) == "plan_proposal"
    ]
    assert [r.proposal_outcome for r in records] == ["drafted", "approved"]
    assert {r.proposal_id for r in records} == {proposal.id}
    assert all(r.proposal_kind == "create_plan" for r in records)
    # The tool invocation is still recorded separately, by the dispatcher.
    tool_calls = [
        r for r in caplog.records if getattr(r, "event", None) == "mcp_tool_call"
    ]
    assert any(r.mcp_tool == "propose_training_plan" for r in tool_calls)


async def test_a_refused_approval_says_why_in_the_audit_log(
    caller, session, planned_athlete, registry_session, caplog
):
    """An offer that was put up and *not* taken up is as much a part of the
    record as one that was — and the refusal names which invariant stopped it."""
    import logging

    proposal = await _draft(session, planned_athlete, caller, registry_session)
    later = date.fromisoformat(proposal.payload["start_date"]) + timedelta(days=1)

    with caplog.at_level(logging.INFO, logger="openkoutsi.audit"):
        with pytest.raises(ProposalError):
            await apply_proposal(session, planned_athlete, proposal, today=later)

    record = next(
        r
        for r in caplog.records
        if getattr(r, "event", None) == "plan_proposal"
        and r.proposal_outcome != "drafted"
    )
    assert record.proposal_outcome == "tool_error"
    assert record.proposal_refusal_code == "proposal_stale"
    assert record.proposal_id == proposal.id


async def test_consent_is_checked_before_a_proposal_is_drafted(
    caller, session, planned_athlete, registry_session
):
    """Writing an athlete's health data deserves at least the check reading it
    gets. Consent fires per invocation in `call_tool`, for these tools as for
    every other."""
    from sqlalchemy import update

    from backend.app.models.registry_orm import User

    await registry_session.execute(
        update(User).where(User.id == _TEST_USER_ID).values(consented_at=None)
    )
    await registry_session.commit()

    result = await _run(
        "propose_training_plan", TOOL_ARGS["propose_training_plan"], caller=caller,
        session=session, athlete=planned_athlete, registry_session=registry_session,
    )
    assert not result.ok
    assert "data-processing policy" in result.error
    assert (await session.execute(select(func.count()).select_from(PlanProposal))).scalar_one() == 0


# ── What the next turn is told ──────────────────────────────────────────────


class TestTheNextTurnKnowsWhatWasDecided:
    """The decision is a button, so it is not in the dialogue at all.

    Without replaying it, Koutsi's next turn sees an offer it made and no answer
    to it, and offers the same plan again — to an athlete who already has it.
    """

    async def _run_second_turn(self, client, auth_headers, scripted_turn, usage_db):
        from tests.unit.test_llm_agent import text
        from backend.app.services.llm_chat import run_chat_turn_bg

        provider = scripted_turn.provider(
            text("MOOD:knowing\n\nGood. Start on Monday.")
        )
        scripted_turn(provider)
        resp = await client.post(
            f"{_PREFIX}/conversations", json={"message": "Build me something"},
            headers=auth_headers,
        )
        body = resp.json()
        conversation_id, first_answer = body["id"], body["messages"][1]["id"]

        # The first turn answered and offered a plan; the athlete approved it.
        async with get_user_session_factory(_TEST_USER_ID)() as s:
            row = (
                await s.execute(
                    select(ChatMessage).where(ChatMessage.id == first_answer)
                )
            ).scalar_one()
            row.status = STATUS_COMPLETE
            row.content = "MOOD:knowing\n\nHere is eight weeks for October."
            await s.commit()
        await _attach_proposal(conversation_id, first_answer, status="applied")

        second = await client.post(
            f"{_PREFIX}/conversations/{conversation_id}/messages",
            json={"message": "When do I start?"}, headers=auth_headers,
        )
        second_answer = second.json()["id"]
        await run_chat_turn_bg(_TEST_USER_ID, conversation_id, second_answer)
        return provider

    async def test_the_decision_is_replayed_into_the_next_turns_history(
        self, client, auth_headers, no_turns, scripted_turn, usage_db
    ):
        await _seed_file_backed(with_plan=False)
        provider = await self._run_second_turn(
            client, auth_headers, scripted_turn, usage_db
        )
        sent = provider.sent[0]["messages"]
        offered = next(m for m in sent if m["role"] == "assistant")
        assert "The athlete approved this" in offered["content"]
        # And the turn is still a turn: the note rides inside it rather than
        # becoming a message of its own.
        assert [m["role"] for m in sent if m["role"] != "system"] == [
            "user", "assistant", "user",
        ]

    async def test_the_dialogue_itself_is_left_exactly_as_it_was(
        self, client, auth_headers, no_turns, scripted_turn, usage_db
    ):
        """Derived at render time, so it can never be stale — and
        ``chat_messages`` stays the conversation rather than a log of state."""
        await _seed_file_backed(with_plan=False)
        await self._run_second_turn(client, auth_headers, scripted_turn, usage_db)
        async with get_user_session_factory(_TEST_USER_ID)() as s:
            rows = (
                (await s.execute(select(ChatMessage).order_by(ChatMessage.created_at)))
                .scalars()
                .all()
            )
        assert all("approved this" not in row.content for row in rows)
