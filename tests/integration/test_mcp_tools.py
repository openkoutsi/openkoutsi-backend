"""Every tool, executed against real fixtures (issue #42).

Schemas and declarations are checked in ``tests/unit/test_mcp_registry.py``; this
module runs the things. It covers the four properties the issue asks for that
only show up at execution time — data isolation between users, the encryption
context being established per invocation, scope enforcement end to end, and
admin data staying out of a session-authenticated administrator's results — plus
one execution of every listed tool, so a handler that raises on an empty database
cannot ship.
"""
from datetime import date, datetime, time, timedelta, timezone

import pytest
from sqlalchemy import select

from backend.app.mcp.dispatch import ToolCaller, call_tool
from backend.app.mcp.limits import tool_limiter
from backend.app.mcp.registry import all_tools
from backend.app.models.user_orm import (
    Activity,
    ActivityInterval,
    ActivityPowerBest,
    Athlete,
    DailyMetric,
    Goal,
    PlannedWorkout,
    PlannedWorkoutActivity,
    TrainingPlan,
)

_TEST_USER_ID = "test-user-00000000"

ALL_SCOPES = ["activities:read", "athlete:read", "goals:read", "metrics:read", "plans:read"]


@pytest.fixture(autouse=True)
def _clean_rate_limiter():
    """The limiter is process-wide; a busy test file must not throttle the next."""
    tool_limiter.reset()
    yield
    tool_limiter.reset()


@pytest.fixture
def caller() -> ToolCaller:
    """A session credential — full access, as everywhere else in the codebase."""
    return ToolCaller(user_id=_TEST_USER_ID, scopes=None, kind="session")


async def run(name, args=None, *, caller, session, athlete, registry_session):
    return await call_tool(
        caller,
        name,
        args or {},
        session=session,
        athlete=athlete,
        registry_session=registry_session,
    )


@pytest.fixture
async def training_data(session, seeded_athlete):
    """A small but complete athlete: rides, metrics, a plan, a goal.

    Deliberately sparse rather than realistic — every tool has to answer over a
    thin history without raising, which is the state a new user is in.
    """
    today = date.today()
    athlete = seeded_athlete
    athlete.ftp = 250
    athlete.max_hr = 185
    athlete.resting_hr = 48
    athlete.weight_kg = 74.0
    athlete.app_settings = {"experience_level": "intermediate"}
    athlete.power_zones = [
        {"name": f"Z{i}", "low": low, "high": high}
        for i, (low, high) in enumerate(
            [(0, 137), (137, 187), (187, 217), (217, 237), (237, 265), (265, 300), (300, 9999)],
            start=1,
        )
    ]
    athlete.hr_zones = [
        {"name": f"Z{i}", "low": low, "high": high}
        for i, (low, high) in enumerate(
            [(0, 120), (120, 140), (140, 160), (160, 172), (172, 200)], start=1
        )
    ]

    endurance = Activity(
        id="act-endurance",
        athlete_id=athlete.id,
        name="Long Sunday",
        sport_type="Ride",
        workout_category="endurance",
        start_time=datetime.combine(today - timedelta(days=2), time(9, 0), tzinfo=timezone.utc),
        duration_s=7412,
        distance_m=61000.0,
        elevation_m=640.0,
        avg_power=180.0,
        weighted_power=195.0,
        avg_hr=139.0,
        max_hr=171.0,
        avg_cadence=86.0,
        load=145.0,
        intensity=0.78,
        rpe=6,
        labels=["race"],
        notes="Felt strong on the climbs.",
        status="processed",
        decoupling_pct=3.4,
        cp_w=248.0,
        w_prime_j=19500.0,
        zone_times={"power": {"Z1": 900, "Z2": 5400, "Z3": 1112}, "hr": {"Z2": 6000, "Z3": 1412}},
    )
    commute = Activity(
        id="act-commute",
        athlete_id=athlete.id,
        name="To work",
        sport_type="Ride",
        workout_category="recovery",
        start_time=datetime.combine(today - timedelta(days=1), time(7, 30), tzinfo=timezone.utc),
        duration_s=1500,
        distance_m=9000.0,
        avg_power=120.0,
        weighted_power=130.0,
        avg_hr=118.0,
        load=18.0,
        labels=["commute"],
        status="processed",
        # No decoupling figure, and the reason for that is a fact the tools must
        # carry rather than flatten into a null.
        decoupling_reason="too_short",
        zone_times={"power": {"Z1": 1500}},
    )
    session.add_all([endurance, commute])

    session.add_all(
        [
            ActivityInterval(
                activity_id=endurance.id,
                interval_number=n,
                start_offset_s=n * 600,
                duration_s=480,
                avg_power=240.0 + n,
                avg_hr=162.0,
                avg_cadence=90.0,
                is_auto_split=False,
            )
            for n in range(1, 4)
        ]
    )
    session.add_all(
        [
            ActivityPowerBest(
                activity_id=endurance.id,
                athlete_id=athlete.id,
                duration_s=duration,
                power_w=power,
                activity_start_time=endurance.start_time,
                weight_kg=74.0,
                w_per_kg=power / 74.0,
            )
            for duration, power in [(5, 780.0), (60, 460.0), (300, 320.0), (1200, 268.0)]
        ]
    )

    for offset in range(0, 30):
        day = today - timedelta(days=offset)
        session.add(
            DailyMetric(
                athlete_id=athlete.id,
                date=day,
                fitness=60.0 - offset * 0.4,
                fatigue=55.0 - offset * 0.2,
                form=5.0 - offset * 0.2,
                load_day=40.0 if offset % 3 else 0.0,
            )
        )

    plan = TrainingPlan(
        id="plan-1",
        athlete_id=athlete.id,
        name="Spring base",
        goal="Build aerobic base",
        start_date=today - timedelta(days=7),
        end_date=today + timedelta(days=20),
        weeks=4,
        status="active",
        week_meta=[{"kind": "build", "focus": "endurance volume"}] * 4,
    )
    session.add(plan)
    workouts = [
        # Week 1 day 1 == plan start: completed by the long ride.
        PlannedWorkout(
            id="pw-done", plan_id=plan.id, week_number=1, day_of_week=1,
            workout_type="endurance", description="3 h steady", duration_min=180, target_load=150,
        ),
        PlannedWorkout(
            id="pw-missed", plan_id=plan.id, week_number=1, day_of_week=3,
            workout_type="threshold", description="4x8", duration_min=75, target_load=95,
        ),
        PlannedWorkout(
            id="pw-skipped", plan_id=plan.id, week_number=1, day_of_week=4,
            workout_type="tempo", duration_min=60, target_load=70, skip_reason="illness",
        ),
        PlannedWorkout(
            id="pw-rest", plan_id=plan.id, week_number=1, day_of_week=5, workout_type="rest",
        ),
        # Week 2 day 1 is exactly seven days after the plan start, i.e. today.
        PlannedWorkout(
            id="pw-today", plan_id=plan.id, week_number=2, day_of_week=1,
            workout_type="endurance", duration_min=90, target_load=80,
        ),
        PlannedWorkout(
            id="pw-upcoming", plan_id=plan.id, week_number=2, day_of_week=3,
            workout_type="vo2max", duration_min=60, target_load=90,
        ),
    ]
    session.add_all(workouts)
    await session.flush()
    session.add(
        PlannedWorkoutActivity(planned_workout_id="pw-done", activity_id=endurance.id)
    )

    session.add_all(
        [
            Goal(
                id="goal-active",
                athlete_id=athlete.id,
                title="Raise FTP to 280 W",
                description="Before the spring series",
                metric="FTP (W)",
                target_value=280.0,
                current_value=250.0,
                target_date=today + timedelta(days=60),
                status="active",
                guidance_verdict="ambitious",
            ),
            Goal(
                id="goal-overdue",
                athlete_id=athlete.id,
                title="Ride a 200 km day",
                target_date=today - timedelta(days=10),
                status="active",
            ),
            Goal(
                id="goal-done",
                athlete_id=athlete.id,
                title="Complete a century",
                status="achieved",
                outcome_note="Done in March.",
            ),
        ]
    )
    await session.commit()
    return athlete


# ── Every tool runs ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool_name", [t.name for t in all_tools()])
async def test_every_tool_answers_over_a_seeded_athlete(
    tool_name, caller, session, training_data, registry_session
):
    """One execution of each listed tool. The arguments are the defaults except
    where a tool needs an identifier, which is the shape an agent's first call
    takes."""
    args = {"activity_id": "act-endurance"} if tool_name == "get_activity_detail" else {}
    result = await run(
        tool_name, args, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    assert result.ok, result.error
    assert isinstance(result.data, dict)


@pytest.mark.parametrize("tool_name", [t.name for t in all_tools()])
async def test_every_tool_answers_over_an_empty_athlete(
    tool_name, caller, session, seeded_athlete, registry_session
):
    """A brand-new user has no rides, no plan and no metrics. Every tool must
    still answer — an exception there would be the model's first impression."""
    args = {"activity_id": "nope"} if tool_name == "get_activity_detail" else {}
    result = await run(
        tool_name, args, caller=caller, session=session,
        athlete=seeded_athlete, registry_session=registry_session,
    )
    if tool_name == "get_activity_detail":
        assert not result.ok  # an unknown id, reported as prose
    else:
        assert result.ok, result.error


async def test_every_tool_stays_inside_the_response_bound(
    caller, session, training_data, registry_session
):
    from backend.app.mcp.dispatch import MAX_RESULT_BYTES

    for t in all_tools():
        args = {"activity_id": "act-endurance"} if t.name == "get_activity_detail" else {}
        result = await run(
            t.name, args, caller=caller, session=session,
            athlete=training_data, registry_session=registry_session,
        )
        assert len(result.text().encode()) < MAX_RESULT_BYTES, t.name


# ── The content each tool owes ───────────────────────────────────────────────


async def test_training_status_reports_form_in_words_and_the_profile_context(
    caller, session, training_data, registry_session
):
    result = await run(
        "get_training_status", caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    data = result.data
    assert data["form_label"] in {"peak", "fresh", "neutral", "tired", "overreached"}
    assert data["athlete"]["ftp_w"] == 250
    assert data["athlete"]["experience_level"] == "intermediate"
    assert data["volume"]["activities"] == 2
    # 30 days of stored metrics means a week-ago figure exists to compare against.
    assert data["fitness_change_7d"] is not None
    # With no 'as_of' the answer is about today, and says so (issue #48).
    assert data["as_of"] == date.today().isoformat()
    assert data["requested_as_of"] == date.today().isoformat()
    assert data["stale"] is False


# ── Training status on a past date (issue #48) ───────────────────────────────


async def test_training_status_answers_for_a_past_date(
    caller, session, training_data, registry_session
):
    """'Where was I then' — the question the tool could not answer before.

    The fixture's only rides are one and two days old, so ten days ago the
    athlete was genuinely unloaded and today they are not. A correct past answer
    is therefore a *different* set of numbers, not today's row with a different
    date on it.
    """
    then = date.today() - timedelta(days=10)
    past = (await run(
        "get_training_status", {"as_of": then.isoformat()}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )).data
    now = (await run(
        "get_training_status", caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )).data

    assert past["as_of"] == then.isoformat()
    assert past["requested_as_of"] == then.isoformat()
    assert past["stale"] is False
    assert past["fatigue"] < now["fatigue"]
    assert past["form"] > now["form"]  # unloaded then, carrying the rides now
    assert past["form_label"] in {"peak", "fresh", "neutral", "tired", "overreached"}
    assert past["form_label"] != now["form_label"]
    # The profile context still comes along — the numbers are no more readable
    # without it for a past date than for today.
    assert past["athlete"]["ftp_w"] == 250


async def test_asking_for_today_explicitly_is_the_same_answer_as_asking_for_nothing(
    caller, session, training_data, registry_session
):
    """The default is a value of the argument, not a separate code path."""
    implicit = (await run(
        "get_training_status", caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )).data
    explicit = (await run(
        "get_training_status", {"as_of": date.today().isoformat()}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )).data
    assert implicit == explicit


async def test_the_volume_window_moves_with_the_date_it_was_asked_about(
    caller, session, training_data, registry_session
):
    """The part that makes a past answer trustworthy rather than misleading.

    Both fixture rides are 1 and 2 days old. A window ending five days ago must
    not count them: Fitness from March beside every ride since would read as a
    training block that never happened, and nothing in the response would show
    the model that the two halves disagree.
    """
    then = (await run(
        "get_training_status",
        {"as_of": (date.today() - timedelta(days=5)).isoformat(), "window_days": 7},
        caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )).data["volume"]
    now = (await run(
        "get_training_status", {"window_days": 7}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )).data["volume"]

    assert then["days"] == 7 and now["days"] == 7
    assert then["activities"] == 0
    assert then["duration_s"] == 0
    assert then["distance_m"] == 0
    assert then["load_total"] == 0
    # The same window ending today does find them, so this is a window that
    # moved rather than one that stopped counting.
    assert now["activities"] == 2
    assert now["load_total"] > 0


async def test_the_fitness_trend_is_measured_from_the_date_asked_about(
    caller, session, training_data, registry_session
):
    """At the very start of the history there is nothing a week earlier to
    compare against, so the trend is honestly null rather than measured from
    today's figures."""
    oldest = date.today() - timedelta(days=29)
    result = await run(
        "get_training_status", {"as_of": oldest.isoformat()}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    assert result.data["as_of"] == oldest.isoformat()
    assert result.data["fitness_change_7d"] is None
    assert result.data["fitness_change_28d"] is None


async def test_a_future_date_is_refused_with_a_sentence_naming_today(
    caller, session, training_data, registry_session
):
    ahead = date.today() + timedelta(days=3)
    result = await run(
        "get_training_status", {"as_of": ahead.isoformat()}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    assert not result.ok
    assert ahead.isoformat() in result.error
    assert date.today().isoformat() in result.error
    # The model is told what to do next, not just what went wrong.
    assert "get_plan_status" in result.error


async def test_a_date_before_the_history_names_where_the_history_starts(
    caller, session, training_data, registry_session
):
    """A refusal that only said 'nothing there' would cost a call to find out
    where 'there' begins."""
    result = await run(
        "get_training_status",
        {"as_of": (date.today() - timedelta(days=400)).isoformat()},
        caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert not result.ok
    assert (date.today() - timedelta(days=29)).isoformat() in result.error


async def test_a_brand_new_athlete_is_told_their_history_starts_today(
    caller, session, seeded_athlete, registry_session
):
    """Zeros for an explicit past date are indistinguishable from a genuine
    week off, and the model would report a collapse that never happened.

    With no date asked for, zeros stay the right answer — that is the
    empty-athlete test above, which every tool has to pass.
    """
    empty = (await run(
        "get_training_status", caller=caller, session=session,
        athlete=seeded_athlete, registry_session=registry_session,
    ))
    assert empty.ok and empty.data["fitness"] == 0.0

    refused = await run(
        "get_training_status",
        {"as_of": (date.today() - timedelta(days=3)).isoformat()},
        caller=caller, session=session, athlete=seeded_athlete,
        registry_session=registry_session,
    )
    assert not refused.ok
    assert "No training metrics stored on or before" in refused.error
    # The catch-up writes today's row, so today is exactly where this athlete's
    # history begins — and saying so is what lets the model retry correctly.
    assert date.today().isoformat() in refused.error


async def test_a_past_date_still_needs_both_scopes(
    session, training_data, registry_session
):
    """The new argument is not a way around the profile grant."""
    metrics_only = ToolCaller(user_id=_TEST_USER_ID, scopes=["metrics:read"], kind="pat")
    result = await run(
        "get_training_status",
        {"as_of": (date.today() - timedelta(days=5)).isoformat()},
        caller=metrics_only, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert not result.ok
    assert "athlete:read" in result.error


async def test_list_recent_activities_can_drop_commutes(
    caller, session, training_data, registry_session
):
    everything = await run(
        "list_recent_activities", caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    assert everything.data["total"] == 2

    training_only = await run(
        "list_recent_activities", {"exclude_labels": ["commute"]},
        caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert training_only.data["total"] == 1
    assert training_only.data["items"][0]["activity_id"] == "act-endurance"


async def test_every_filter_argument_actually_filters(
    caller, session, training_data, registry_session
):
    """A filter the model asked for and did not get is a confidently wrong
    answer, so each one is exercised rather than assumed."""
    cases = [
        ("list_recent_activities", {"sport_type": "Run"}, 0),
        ("list_recent_activities", {"sport_type": "Ride"}, 2),
        ("list_recent_activities", {"days": 1}, 1),
        ("find_activity", {"name_contains": "sunday"}, 1),
        ("find_activity", {"name_contains": "nothing like this"}, 0),
        ("find_activity", {"min_duration_s": 7000}, 1),
        ("find_activity", {"start": (date.today() - timedelta(days=1)).isoformat()}, 1),
        ("find_activity", {"end": (date.today() - timedelta(days=2)).isoformat()}, 1),
        ("find_activity", {"sport_type": "Ride"}, 2),
    ]
    for tool_name, args, expected in cases:
        result = await run(
            tool_name, args, caller=caller, session=session, athlete=training_data,
            registry_session=registry_session,
        )
        assert result.ok, (tool_name, args, result.error)
        assert result.data["total"] == expected, (tool_name, args)


async def test_the_plan_window_argument_widens_what_is_listed(
    caller, session, training_data, registry_session
):
    narrow = await run(
        "get_plan_status", {"week_window_days": 1}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    wide = await run(
        "get_plan_status", {"week_window_days": 21}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    assert len(wide.data["plans"][0]["upcoming"]) > len(narrow.data["plans"][0]["upcoming"])


async def test_a_truncated_list_says_so(caller, session, training_data, registry_session):
    result = await run(
        "list_recent_activities", {"limit": 1}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    assert result.data["returned"] == 1
    assert result.data["total"] == 2
    assert result.data["truncated"] is True


async def test_find_activity_on_an_empty_date_names_the_nearest_rides(
    caller, session, training_data, registry_session
):
    """The error the issue specifies, more or less verbatim."""
    empty_day = (date.today() - timedelta(days=30)).isoformat()
    result = await run(
        "find_activity", {"on_date": empty_day}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    assert not result.ok
    assert f"No activity on {empty_day}" in result.error
    assert "Nearest rides:" in result.error
    assert "endurance" in result.error


async def test_find_activity_says_so_when_there_is_nothing_either_side(
    caller, session, seeded_athlete, registry_session
):
    result = await run(
        "find_activity", {"on_date": date.today().isoformat()}, caller=caller,
        session=session, athlete=seeded_athlete, registry_session=registry_session,
    )
    assert not result.ok
    assert "no activities recorded on either side" in result.error.lower()


async def test_find_activity_matches_by_category_and_label(
    caller, session, training_data, registry_session
):
    by_category = await run(
        "find_activity", {"workout_category": "endurance"}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    assert [i["activity_id"] for i in by_category.data["items"]] == ["act-endurance"]

    by_label = await run(
        "find_activity", {"label": "commute"}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    assert [i["activity_id"] for i in by_label.data["items"]] == ["act-commute"]


async def test_an_inverted_window_is_explained_rather_than_returned_empty(
    caller, session, training_data, registry_session
):
    result = await run(
        "find_activity",
        {"start": date.today().isoformat(), "end": (date.today() - timedelta(days=5)).isoformat()},
        caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert not result.ok
    assert "inverted" in result.error


async def test_activity_detail_carries_zones_intervals_and_the_linked_workout(
    caller, session, training_data, registry_session
):
    result = await run(
        "get_activity_detail", {"activity_id": "act-endurance"}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    data = result.data
    assert data["notes"] == "Felt strong on the climbs."
    assert data["rpe"] == 6
    assert {z["zone"] for z in data["power_zones"]} == {"Z1", "Z2", "Z3"}
    assert sum(z["pct"] for z in data["power_zones"]) == pytest.approx(100.0, abs=0.2)
    assert data["intervals_total"] == 3
    assert len(data["intervals"]) == 3
    assert data["linked_workout"]["plan_name"] == "Spring base"
    assert data["power_bests"][0]["duration_s"] == 5
    assert data["aerobic"]["decoupling_pct"] == 3.4
    assert data["aerobic"]["efficiency_factor"] is not None


async def test_a_reason_code_is_preserved_rather_than_reported_as_a_null(
    caller, session, training_data, registry_session
):
    """"Too short to measure" and "no measurement" are different answers."""
    result = await run(
        "get_activity_detail", {"activity_id": "act-commute"}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    aerobic = result.data["aerobic"]
    assert aerobic["decoupling_pct"] is None
    assert aerobic["decoupling_reason"] == "too_short"


async def test_activity_detail_returns_no_streams_and_no_coordinates(
    caller, session, training_data, registry_session
):
    result = await run(
        "get_activity_detail", {"activity_id": "act-endurance"}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    text = result.text().lower()
    for forbidden in ("latitude", "longitude", "lat", "lng", "position", "streams", "coordinates"):
        assert forbidden not in text, forbidden


async def test_activity_detail_refuses_another_athletes_activity(
    caller, session, training_data, registry_session
):
    """Ownership is a predicate on the query, not an afterthought."""
    other = Athlete(id="other-athlete", global_user_id="other-user", ftp_tests=[])
    session.add(other)
    session.add(
        Activity(
            id="act-someone-else",
            athlete_id=other.id,
            name="Not yours",
            sport_type="Ride",
            start_time=datetime.now(timezone.utc),
            duration_s=3600,
            status="processed",
        )
    )
    await session.commit()

    result = await run(
        "get_activity_detail", {"activity_id": "act-someone-else"}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    assert not result.ok
    assert "belongs to this athlete" in result.error


async def test_plan_status_distinguishes_missed_from_not_yet_due(
    caller, session, training_data, registry_session
):
    """The distinction the fixed-blob prompts kept getting wrong."""
    result = await run(
        "get_plan_status", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    plan = result.data["plans"][0]
    assert plan["name"] == "Spring base"
    assert plan["current_week"] == 2
    assert plan["phase"] == "build: endurance volume"

    states = {
        s["date"]: s["state"] for s in plan["recent"] + plan["upcoming"]
    }
    today = date.today().isoformat()
    assert states[today] == "due_today"
    assert states[(date.today() + timedelta(days=2)).isoformat()] == "upcoming"
    assert states[(date.today() - timedelta(days=7)).isoformat()] == "completed"
    assert states[(date.today() - timedelta(days=5)).isoformat()] == "missed"
    assert states[(date.today() - timedelta(days=4)).isoformat()] == "skipped"
    assert states[(date.today() - timedelta(days=3)).isoformat()] == "rest"

    skipped = next(s for s in plan["recent"] if s["state"] == "skipped")
    assert skipped["skip_reason"] == "illness"
    completed = next(s for s in plan["recent"] if s["state"] == "completed")
    assert completed["linked_activities"] == 1
    assert completed["actual_load"] == 145.0
    assert completed["match_score"] is not None


async def test_plan_status_hides_archived_plans_unless_asked(
    caller, session, training_data, registry_session
):
    session.add(
        TrainingPlan(
            id="plan-old", athlete_id=training_data.id, name="Winter block",
            start_date=date.today() - timedelta(days=200),
            end_date=date.today() - timedelta(days=100), weeks=14, status="archived",
        )
    )
    await session.commit()

    active = await run(
        "get_plan_status", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert {p["name"] for p in active.data["plans"]} == {"Spring base"}

    everything = await run(
        "get_plan_status", {"include_archived": True}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    assert {p["name"] for p in everything.data["plans"]} == {"Spring base", "Winter block"}


async def test_goal_progress_computes_progress_and_flags_the_overdue_one(
    caller, session, training_data, registry_session
):
    result = await run(
        "get_goal_progress", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    goals = {g["goal_id"]: g for g in result.data["items"]}
    assert set(goals) == {"goal-active", "goal-overdue"}  # 'active' by default

    ftp_goal = goals["goal-active"]
    assert ftp_goal["progress_pct"] == pytest.approx(89.3, abs=0.1)
    assert ftp_goal["days_remaining"] == 60
    assert ftp_goal["overdue"] is False
    assert ftp_goal["guidance_verdict"] == "ambitious"

    assert goals["goal-overdue"]["overdue"] is True
    assert goals["goal-overdue"]["days_remaining"] == -10
    # No target number means no percentage — which is different from zero.
    assert goals["goal-overdue"]["progress_pct"] is None


async def test_goal_progress_can_include_finished_goals(
    caller, session, training_data, registry_session
):
    result = await run(
        "get_goal_progress", {"status": "all"}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    assert result.data["total"] == 3
    done = next(g for g in result.data["items"] if g["goal_id"] == "goal-done")
    assert done["outcome_note"] == "Done in March."


async def test_power_profile_reports_the_curve_and_the_ftp_disagreement(
    caller, session, training_data, registry_session
):
    result = await run(
        "get_power_profile", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    data = result.data
    assert [b["duration_s"] for b in data["bests"]] == [5, 60, 300, 1200]
    assert data["bests"][0]["w_per_kg"] == pytest.approx(780 / 74.0, abs=0.01)
    assert data["ftp"]["profile_ftp_w"] == 250
    assert data["ftp"]["twenty_min_power_w"] == 268.0
    assert data["ftp"]["ftp_from_20min_w"] == round(268.0 * 0.95)
    # The estimate sits above the profile figure, which is what the athlete
    # should be told rather than left to work out.
    assert data["ftp"]["disagreement_w"] is not None


async def test_zone_totals_bucket_by_monday_and_say_when_weeks_are_absent(
    caller, session, training_data, registry_session
):
    result = await run(
        "get_zone_totals", {"weeks": 4}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    data = result.data
    assert data["weeks_requested"] == 4
    weeks = data["weeks"]
    assert weeks, "the seeded rides should land in at least one week"
    for week in weeks:
        assert date.fromisoformat(week["week_start"]).weekday() == 0
    assert data["note"] is not None  # only one week has riding in it


async def test_zone_totals_can_ask_for_one_basis(
    caller, session, training_data, registry_session
):
    result = await run(
        "get_zone_totals", {"weeks": 2, "basis": "power"}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    for week in result.data["weeks"]:
        assert week["hr"] == {}
        assert week["power"]


async def test_intensity_distribution_reports_its_method_and_coverage(
    caller, session, training_data, registry_session
):
    result = await run(
        "get_intensity_distribution", {"days": 30}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    data = result.data
    assert data["method"] == "time"
    assert [b["band"] for b in data["bands"]] == [1, 2, 3]
    assert data["coverage"]["activities_total"] >= 2
    assert 0 <= data["coverage"]["coverage_pct"] <= 100
    if data["shape"]:
        assert data["shape_meaning"]


async def test_the_session_method_is_reported_as_such(
    caller, session, training_data, registry_session
):
    """The two methods disagree by design, so the answer must say which it is."""
    result = await run(
        "get_intensity_distribution", {"days": 30, "method": "session"},
        caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert result.data["method"] == "session"
    assert result.data["basis"] is None


# ── Scope enforcement ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tool_name,scope",
    [
        ("list_recent_activities", "activities:read"),
        ("find_activity", "activities:read"),
        ("get_activity_detail", "activities:read"),
        ("get_plan_status", "plans:read"),
        ("get_goal_progress", "goals:read"),
        ("get_zone_totals", "metrics:read"),
        ("get_intensity_distribution", "metrics:read"),
    ],
)
async def test_a_token_needs_the_declared_scope(
    tool_name, scope, session, training_data, registry_session
):
    args = {"activity_id": "act-endurance"} if tool_name == "get_activity_detail" else {}

    without = ToolCaller(user_id=_TEST_USER_ID, scopes=[], kind="pat", token_id="tok-1")
    refused = await run(
        tool_name, args, caller=without, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert not refused.ok
    assert scope in refused.error
    assert "missing" in refused.error

    with_scope = ToolCaller(user_id=_TEST_USER_ID, scopes=[scope], kind="pat", token_id="tok-1")
    allowed = await run(
        tool_name, args, caller=with_scope, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert allowed.ok, allowed.error


@pytest.mark.parametrize("tool_name", ["get_training_status", "get_power_profile"])
async def test_a_tool_asking_for_two_scopes_needs_both(
    tool_name, session, training_data, registry_session
):
    """Scopes are an AND. Reporting FTP under a metrics-only grant would be
    serving profile data on a metrics ticket."""
    half = ToolCaller(user_id=_TEST_USER_ID, scopes=["metrics:read"], kind="pat", token_id="t")
    refused = await run(
        tool_name, caller=half, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert not refused.ok
    assert "athlete:read" in refused.error

    both = ToolCaller(
        user_id=_TEST_USER_ID, scopes=["metrics:read", "athlete:read"], kind="pat", token_id="t"
    )
    allowed = await run(
        tool_name, caller=both, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert allowed.ok, allowed.error


async def test_a_scope_refusal_explains_that_the_token_cannot_be_widened(
    session, training_data, registry_session
):
    """A model that thinks a refusal is transient will retry it forever."""
    caller = ToolCaller(user_id=_TEST_USER_ID, scopes=[], kind="pat", token_id="t")
    result = await run(
        "get_plan_status", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert "cannot be widened" in result.error


# ── Admin exclusion ──────────────────────────────────────────────────────────


async def test_an_administrators_session_gets_exactly_what_an_athletes_does(
    session, training_data, registry_session
):
    """The seeded test user holds the ``administrator`` role. A caller carries
    no roles at all, so there is nothing for a tool to widen on."""
    from backend.app.core.auth import UserContext

    admin_ctx = UserContext(user_id=_TEST_USER_ID, roles=["administrator", "user"])
    admin = ToolCaller.from_context(admin_ctx)
    assert not hasattr(admin, "roles")

    plain_ctx = UserContext(user_id=_TEST_USER_ID, roles=["user"])
    plain = ToolCaller.from_context(plain_ctx)

    as_admin = await run(
        "get_training_status", caller=admin, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    as_athlete = await run(
        "get_training_status", caller=plain, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert as_admin.data == as_athlete.data


async def test_no_tool_result_mentions_another_user(
    caller, session, training_data, registry_session
):
    """Registry data — users, invites, instance settings — never appears."""
    for t in all_tools():
        args = {"activity_id": "act-endurance"} if t.name == "get_activity_detail" else {}
        result = await run(
            t.name, args, caller=caller, session=session, athlete=training_data,
            registry_session=registry_session,
        )
        text = result.text().lower()
        for forbidden in ("password", "invitation", "instance_settings", "llm_api_key", "test-user"):
            assert forbidden not in text, f"{t.name} leaked {forbidden}"


# ── Isolation and the encryption context ─────────────────────────────────────


async def test_each_invocation_opens_the_callers_own_database(isolate_user_dbs, registry_session):
    """Isolation is physical: a separate SQLite file per user.

    Run without an explicit session, so the dispatcher does the resolving. Two
    users get two databases, and a caller for one never sees the other's rides
    even though both tools are the same code.
    """
    from backend.app.db.user_session import get_user_session_factory, init_user_db

    alice, bob = "user-alice", "user-bob"
    for user_id, ride_name in ((alice, "Alice's ride"), (bob, "Bob's ride")):
        await init_user_db(user_id)
        async with get_user_session_factory(user_id)() as s:
            athlete = Athlete(global_user_id=user_id, ftp_tests=[])
            s.add(athlete)
            await s.flush()
            s.add(
                Activity(
                    athlete_id=athlete.id,
                    name=ride_name,
                    sport_type="Ride",
                    start_time=datetime.now(timezone.utc),
                    duration_s=3600,
                    status="processed",
                )
            )
            await s.commit()

    # Both users must be consented, since the gate runs before any read.
    from backend.app.api.consent import CURRENT_CONSENT_VERSION
    from backend.app.models.registry_orm import User

    for user_id in (alice, bob):
        registry_session.add(
            User(
                id=user_id,
                username=user_id,
                password_hash="x",
                roles="[]",
                consented_at=datetime.now(timezone.utc),
                consent_version=CURRENT_CONSENT_VERSION,
            )
        )
    await registry_session.commit()

    for user_id, expected in ((alice, "Alice's ride"), (bob, "Bob's ride")):
        result = await call_tool(
            ToolCaller(user_id=user_id),
            "list_recent_activities",
            {},
            registry_session=registry_session,
        )
        assert result.ok, result.error
        names = [item["name"] for item in result.data["items"]]
        assert names == [expected]


async def test_the_encryption_context_is_established_before_any_read(
    isolate_user_dbs, registry_session, monkeypatch
):
    """The context and the session must be set up together, or a tool can read
    rows it cannot decrypt — or, worse, decrypt them with the wrong key."""
    from backend.app.api.consent import CURRENT_CONSENT_VERSION
    from backend.app.db.user_session import get_user_session_factory, init_user_db
    from backend.app.models.registry_orm import User

    user_id = "user-crypto"
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as s:
        s.add(Athlete(global_user_id=user_id, ftp_tests=[]))
        await s.commit()
    registry_session.add(
        User(
            id=user_id, username=user_id, password_hash="x", roles="[]",
            consented_at=datetime.now(timezone.utc),
            consent_version=CURRENT_CONSENT_VERSION,
        )
    )
    await registry_session.commit()

    seen: list[str] = []
    import backend.app.core.deps as deps

    original = deps.set_user_encryption_context
    monkeypatch.setattr(
        deps, "set_user_encryption_context", lambda uid: (seen.append(uid), original(uid))[1]
    )

    result = await call_tool(
        ToolCaller(user_id=user_id), "get_training_status", {},
        registry_session=registry_session,
    )
    assert result.ok, result.error
    assert seen == [user_id]


async def test_a_user_with_no_athlete_profile_is_told_so(isolate_user_dbs, registry_session):
    """An un-onboarded account is a fact about the data, not a 404 to leak."""
    from backend.app.api.consent import CURRENT_CONSENT_VERSION
    from backend.app.db.user_session import init_user_db
    from backend.app.models.registry_orm import User

    user_id = "user-empty"
    await init_user_db(user_id)
    registry_session.add(
        User(
            id=user_id, username=user_id, password_hash="x", roles="[]",
            consented_at=datetime.now(timezone.utc),
            consent_version=CURRENT_CONSENT_VERSION,
        )
    )
    await registry_session.commit()

    result = await call_tool(
        ToolCaller(user_id=user_id), "get_training_status", {},
        registry_session=registry_session,
    )
    assert not result.ok
    assert "no athlete profile" in result.error
    assert "setup wizard" in result.error


# ── Consent, arguments, unknown tools, limits ────────────────────────────────


async def test_consent_is_required_per_invocation(
    caller, session, training_data, registry_session
):
    """Reading health data back out is the same processing the ingestion paths
    already gate on."""
    from backend.app.models.registry_orm import User

    user = (
        await registry_session.execute(select(User).where(User.id == _TEST_USER_ID))
    ).scalar_one()
    user.consented_at = None
    await registry_session.commit()

    result = await run(
        "get_training_status", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert not result.ok
    assert "data-processing policy" in result.error


async def test_a_stale_consent_version_is_not_consent(
    caller, session, training_data, registry_session
):
    from backend.app.models.registry_orm import User

    user = (
        await registry_session.execute(select(User).where(User.id == _TEST_USER_ID))
    ).scalar_one()
    user.consent_version = "0.1"
    await registry_session.commit()

    result = await run(
        "get_training_status", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert not result.ok


async def test_an_unknown_tool_names_the_real_ones(
    caller, session, training_data, registry_session
):
    result = await run(
        "get_everything", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert not result.ok
    assert "No tool named 'get_everything'" in result.error
    assert "get_training_status" in result.error


async def test_bad_arguments_are_explained_in_prose(
    caller, session, training_data, registry_session
):
    result = await run(
        "list_recent_activities", {"limit": 5000}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    assert not result.ok
    assert "limit" in result.error
    assert "Accepted arguments:" in result.error


async def test_a_misspelled_argument_is_refused_rather_than_ignored(
    caller, session, training_data, registry_session
):
    """Silently dropping an argument makes a model believe a filter applied.

    ``sport`` is not ``sport_type``. Ignoring it would return every sport and
    let the model report it as a filtered answer — a confident wrong answer,
    which nothing downstream can tell from a right one.
    """
    result = await run(
        "list_recent_activities", {"limit": 5, "sport": "Ride"}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    assert not result.ok
    assert "sport" in result.error
    assert "sport_type" in result.error  # the accepted-arguments list names the real one


async def test_an_oversized_result_is_refused_rather_than_returned(
    caller, session, training_data, registry_session, monkeypatch
):
    """The size bound is defence in depth behind the shaping rules."""
    import backend.app.mcp.dispatch as dispatch

    monkeypatch.setattr(dispatch, "MAX_RESULT_BYTES", 10)
    result = await run(
        "get_training_status", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert not result.ok
    assert "too large" in result.error


async def test_external_callers_are_rate_limited_and_the_agent_is_not(
    session, training_data, registry_session, monkeypatch
):
    monkeypatch.setattr(tool_limiter, "limit", 2)

    token_caller = ToolCaller(user_id=_TEST_USER_ID, scopes=None, kind="pat", token_id="t")
    outcomes = [
        (
            await run(
                "get_goal_progress", caller=token_caller, session=session,
                athlete=training_data, registry_session=registry_session,
            )
        ).ok
        for _ in range(3)
    ]
    assert outcomes == [True, True, False]

    # The in-process agent is exempt: its calls are already bounded by the LLM
    # loop that issues them, and a limit firing mid-turn would leave the model
    # reasoning from half a picture.
    tool_limiter.reset()
    agent = ToolCaller.internal(_TEST_USER_ID)
    assert agent.kind == "agent"
    for _ in range(5):
        result = await run(
            "get_goal_progress", caller=agent, session=session,
            athlete=training_data, registry_session=registry_session,
        )
        assert result.ok, result.error


async def test_every_invocation_is_audited(
    caller, session, training_data, registry_session, caplog
):
    """Caller, tool, arguments, duration and outcome — the issue's list."""
    import logging

    with caplog.at_level(logging.INFO, logger="openkoutsi.audit"):
        await run(
            "find_activity", {"sport_type": "Ride"}, caller=caller, session=session,
            athlete=training_data, registry_session=registry_session,
        )

    record = next(r for r in caplog.records if getattr(r, "event", None) == "mcp_tool_call")
    assert record.mcp_tool == "find_activity"
    assert record.mcp_outcome == "ok"
    assert record.mcp_arguments == {"sport_type": "Ride"}
    assert record.mcp_caller_kind == "session"
    assert record.pat_user_id == _TEST_USER_ID
    assert record.mcp_duration_ms >= 0


async def test_a_refusal_is_audited_too(
    session, training_data, registry_session, caplog
):
    import logging

    denied = ToolCaller(user_id=_TEST_USER_ID, scopes=[], kind="pat", token_id="tok-9")
    with caplog.at_level(logging.INFO, logger="openkoutsi.audit"):
        await run(
            "get_plan_status", caller=denied, session=session, athlete=training_data,
            registry_session=registry_session,
        )

    record = next(r for r in caplog.records if getattr(r, "event", None) == "mcp_tool_call")
    assert record.mcp_outcome == "denied_scope"
    assert record.pat_token_id == "tok-9"


async def test_the_audit_log_never_carries_the_result(
    caller, session, training_data, registry_session, caplog
):
    """Arguments are dates and ids; results are the health data itself."""
    import logging

    with caplog.at_level(logging.INFO, logger="openkoutsi.audit"):
        await run(
            "get_activity_detail", {"activity_id": "act-endurance"}, caller=caller,
            session=session, athlete=training_data, registry_session=registry_session,
        )

    record = next(r for r in caplog.records if getattr(r, "event", None) == "mcp_tool_call")
    assert "Felt strong on the climbs." not in str(vars(record))


# ── The tools do not write (review of #86) ───────────────────────────────────


async def test_the_zone_tools_never_freeze_a_snapshot(
    caller, session, training_data, registry_session
):
    """``readOnlyHint: True`` has to be true.

    Both zone paths can backfill missing ``zone_times`` and commit. Freezing is
    permanent by design, so letting a `metrics:read` tool trigger it would let
    the moment a coaching agent asked a question decide, forever, which zone
    definitions an old ride is judged against — and would make the hint an MCP
    client uses to decide whether to ask the user first a lie.
    """
    from backend.app.models.user_orm import ActivityStream

    bare = Activity(
        id="act-no-snapshot",
        athlete_id=training_data.id,
        name="Before snapshots existed",
        sport_type="Ride",
        start_time=datetime.combine(date.today() - timedelta(days=3), time(9, 0), tzinfo=timezone.utc),
        duration_s=3600,
        status="processed",
        zone_times=None,
    )
    session.add(bare)
    await session.flush()
    # A stream the backfill would have consumed, so the test fails for the right
    # reason rather than because there was nothing to compute from.
    session.add(ActivityStream(activity_id=bare.id, stream_type="power", data=[200] * 3600))
    session.add(ActivityStream(activity_id=bare.id, stream_type="heartrate", data=[140] * 3600))
    await session.commit()

    for tool_name, args in (
        ("get_zone_totals", {"weeks": 4}),
        ("get_intensity_distribution", {"days": 30}),
    ):
        result = await run(
            tool_name, args, caller=caller, session=session, athlete=training_data,
            registry_session=registry_session,
        )
        assert result.ok, result.error

    await session.refresh(bare)
    assert bare.zone_times is None, "a read-only tool froze a zone snapshot"


async def test_zone_totals_reports_rides_it_could_not_count(
    caller, session, training_data, registry_session
):
    """Not counted is said out loud, so under-counted totals aren't read as an
    easy week."""
    session.add(
        Activity(
            id="act-unsnapshotted",
            athlete_id=training_data.id,
            name="No zone data",
            sport_type="Ride",
            start_time=datetime.combine(date.today() - timedelta(days=1), time(18, 0), tzinfo=timezone.utc),
            duration_s=3600,
            status="processed",
            zone_times=None,
        )
    )
    await session.commit()

    result = await run(
        "get_zone_totals", {"weeks": 2}, caller=caller, session=session,
        athlete=training_data, registry_session=registry_session,
    )
    assert result.data["activities_without_zone_data"] == 1
    assert "no stored time-in-zone" in result.data["note"]


async def test_the_api_route_still_backfills(client, auth_headers, session, seeded_athlete):
    """The tool layer stopped freezing snapshots; the endpoint that always did
    must not have. It is a browser action by the data's owner, not a scoped
    credential's side effect."""
    from backend.app.models.user_orm import ActivityStream

    seeded_athlete.power_zones = [
        {"name": f"Z{i}", "low": low, "high": high}
        for i, (low, high) in enumerate(
            [(0, 137), (137, 187), (187, 217), (217, 237), (237, 265), (265, 300), (300, 9999)],
            start=1,
        )
    ]
    activity = Activity(
        id="act-for-route",
        athlete_id=seeded_athlete.id,
        name="Needs a snapshot",
        sport_type="Ride",
        start_time=datetime.now(timezone.utc) - timedelta(days=1),
        duration_s=3600,
        status="processed",
        zone_times=None,
    )
    session.add(activity)
    await session.flush()
    session.add(ActivityStream(activity_id=activity.id, stream_type="power", data=[200] * 3600))
    await session.commit()

    resp = await client.get("/api/metrics/zones/weekly", headers=auth_headers)
    assert resp.status_code == 200

    await session.refresh(activity)
    assert activity.zone_times is not None


# ── Ordering of the checks (review of #86) ───────────────────────────────────


async def test_the_rate_limiter_counts_calls_that_cannot_succeed(
    session, training_data, registry_session, monkeypatch
):
    """The loop most likely to actually happen is a client retrying against a
    credential that can never succeed. Counting only the successes inverts the
    limiter's whole intent."""
    monkeypatch.setattr(tool_limiter, "limit", 2)

    denied = ToolCaller(user_id=_TEST_USER_ID, scopes=[], kind="pat", token_id="t")
    outcomes = []
    for _ in range(3):
        result = await run(
            "get_plan_status", caller=denied, session=session, athlete=training_data,
            registry_session=registry_session,
        )
        outcomes.append(result.error)

    assert "missing" in outcomes[0]
    assert "missing" in outcomes[1]
    assert "Too many tool calls" in outcomes[2]


async def test_a_consent_refusal_is_counted_too(
    caller, session, training_data, registry_session, monkeypatch
):
    """Each of these costs a registry round-trip, so an uncounted loop is not
    free."""
    from backend.app.models.registry_orm import User

    user = (
        await registry_session.execute(select(User).where(User.id == _TEST_USER_ID))
    ).scalar_one()
    user.consented_at = None
    await registry_session.commit()

    monkeypatch.setattr(tool_limiter, "limit", 1)
    first = await run(
        "get_goal_progress", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    second = await run(
        "get_goal_progress", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert "data-processing policy" in first.error
    assert "Too many tool calls" in second.error


async def test_the_encryption_context_is_set_for_a_caller_supplied_session(
    caller, session, training_data, registry_session, monkeypatch
):
    """`caller.user_id` decides the scope check, the consent check and every
    audit record; nothing ties it to a session handed in. Deriving the key from
    it makes a mismatch a decryption failure rather than one athlete's data
    returned under another's name."""
    seen: list[str] = []
    import backend.app.mcp.dispatch as dispatch

    monkeypatch.setattr(dispatch, "set_user_encryption_context", lambda uid: seen.append(uid))

    result = await run(
        "get_goal_progress", caller=caller, session=session, athlete=training_data,
        registry_session=registry_session,
    )
    assert result.ok, result.error
    assert seen == [_TEST_USER_ID]


# ── Filters mean what their descriptions say (review of #86) ─────────────────


async def test_like_wildcards_in_a_name_search_are_literal(
    caller, session, training_data, registry_session
):
    """The field promises a substring match, which is what the model believes
    it got. Unescaped, `_` matches any character and `%` matches anything."""
    session.add_all([
        Activity(
            id="act-underscore", athlete_id=training_data.id, name="_intervals",
            sport_type="Ride", status="processed",
            start_time=datetime.now(timezone.utc) - timedelta(days=4),
            duration_s=3600,
        ),
        Activity(
            id="act-plain", athlete_id=training_data.id, name="4x8 intervals",
            sport_type="Ride", status="processed",
            start_time=datetime.now(timezone.utc) - timedelta(days=5),
            duration_s=3600,
        ),
    ])
    await session.commit()

    literal = await run(
        "find_activity", {"name_contains": "_intervals"}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    assert [i["activity_id"] for i in literal.data["items"]] == ["act-underscore"]

    # And `%` is not "anything".
    none_match = await run(
        "find_activity", {"name_contains": "4x8%interv"}, caller=caller,
        session=session, athlete=training_data, registry_session=registry_session,
    )
    assert none_match.data["total"] == 0
