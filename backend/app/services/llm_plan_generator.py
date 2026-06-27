"""
LLM-based training plan generator.

Uses any OpenAI-compatible chat completions API (Ollama, OpenAI, Mistral, etc.)
via httpx. No additional dependencies required.

LLM settings are resolved with the same priority as the chat proxy:
  athlete app_settings → team settings → global env vars (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL)
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.registry_orm import Team
from ..models.team_orm import TrainingPlan, PlannedWorkout, Athlete, DailyMetric
from ..schemas.plans import PlanConfig
from .llm_client import call_llm, extract_json, resolve_llm_config

log = logging.getLogger(__name__)

# Backwards-compatible aliases — these names are imported/patched elsewhere.
_extract_json = extract_json
_resolve_llm_config = resolve_llm_config


_SYSTEM_PROMPT = """\
You are an expert endurance sports coach that creates personalised training plans.
You MUST respond with ONLY valid JSON — no markdown, no prose, no code fences.
The JSON must conform exactly to the schema provided by the user.
Do not include any explanation or commentary outside the JSON object.
"""

_SCHEMA_EXAMPLE = """\
{
  "weeks": [
    {
      "week_number": 1,
      "workouts": [
        {
          "day_of_week": 1,
          "workout_type": "rest",
          "description": null,
          "duration_min": null,
          "target_tss": null
        },
        {
          "day_of_week": 2,
          "workout_type": "threshold",
          "description": "2x20 min at threshold power",
          "duration_min": 60,
          "target_tss": 80
        }
      ]
    }
  ]
}

Rules:
- day_of_week: integer 1 (Monday) to 7 (Sunday)
- workout_type: one of "recovery", "tempo", "threshold", "vo2max", "endurance", "long", "strength", "yoga", "cross-training", "rest"
- Every week must have exactly 7 workouts, one per day_of_week (1-7)
- Days not scheduled as training should be "rest" with null duration and tss
- TSS and duration_min must be null for rest days, integers otherwise
- Scale TSS and duration progressively across weeks (base building, recovery every 4th week, taper at end)
"""


def _build_user_prompt(
    config: PlanConfig,
    goal: Optional[str],
    num_weeks: int,
    ftp: Optional[int],
    ctl: Optional[float],
) -> str:
    day_names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
                 5: "Friday", 6: "Saturday", 7: "Sunday"}

    scheduled = []
    for dc in sorted(config.day_configs, key=lambda d: d.day_of_week):
        note = f" ({dc.notes})" if dc.notes else ""
        scheduled.append(f"  - {day_names[dc.day_of_week]}: {dc.workout_type}{note}")

    lines = [
        f"Create a {num_weeks}-week training plan with the following requirements:",
        "",
        f"Periodization style: {config.periodization}",
        f"Intensity preference: {config.intensity_preference}",
        f"Training days per week: {config.days_per_week}",
        "",
        "Scheduled training days:",
    ] + scheduled

    if goal:
        lines += ["", f"Goal/event: {goal}"]
    if config.long_description:
        lines += ["", f"Additional context: {config.long_description}"]
    if ftp:
        lines += ["", f"Athlete FTP: {ftp}W"]
    if ctl is not None:
        lines += [f"Current fitness (CTL): {ctl:.1f} TSS/day"]

    lines += [
        "",
        f"Output exactly {num_weeks} weeks in the JSON schema below.",
        "",
        _SCHEMA_EXAMPLE,
    ]

    return "\n".join(lines)


def _parse_response(raw: str, num_weeks: int) -> list[list[dict]]:
    """Parse LLM JSON response into a list of weeks, each a list of day dicts."""
    data = json.loads(_extract_json(raw))
    weeks_data = data["weeks"]
    if len(weeks_data) != num_weeks:
        raise ValueError(
            f"Expected {num_weeks} weeks, got {len(weeks_data)}"
        )
    result = []
    for week in weeks_data:
        workouts = week["workouts"]
        if len(workouts) != 7:
            raise ValueError(
                f"Week {week['week_number']} has {len(workouts)} days, expected 7"
            )
        # Normalise each workout dict
        normalised = []
        for w in sorted(workouts, key=lambda x: x["day_of_week"]):
            normalised.append({
                "day_of_week": int(w["day_of_week"]),
                "workout_type": str(w.get("workout_type", "rest")),
                "description": w.get("description") or None,
                "duration_min": int(w["duration_min"]) if w.get("duration_min") is not None else None,
                "target_tss": int(w["target_tss"]) if w.get("target_tss") is not None else None,
            })
        result.append(normalised)
    return result


async def _call_llm(user_prompt: str, base_url: str, model: str, api_key: str | None) -> str:
    """Call the chat completions endpoint with the plan-generation system prompt."""
    return await call_llm(
        user_prompt, base_url, model, api_key, system_prompt=_SYSTEM_PROMPT
    )


async def generate_plan_weeks_llm(
    athlete: Athlete,
    config: PlanConfig,
    num_weeks: int,
    goal: Optional[str],
    session: AsyncSession,
    team: Team | None = None,
    team_id: str = "",
    user_id: str = "",
) -> list[list[dict]]:
    """Call the LLM and return parsed weeks (list of weeks, each a list of day dicts).

    Persistence-free so callers can build PlannedWorkout rows for either a new or
    an existing plan.
    """
    base_url, model, api_key = _resolve_llm_config(athlete, team, team_id, user_id)

    # Fetch athlete's latest CTL for context
    ctl: Optional[float] = None
    result = await session.execute(
        select(DailyMetric)
        .where(DailyMetric.athlete_id == athlete.id)
        .order_by(DailyMetric.date.desc())
        .limit(1)
    )
    latest_metric = result.scalar_one_or_none()
    if latest_metric:
        ctl = latest_metric.ctl

    user_prompt = _build_user_prompt(config, goal, num_weeks, athlete.ftp, ctl)

    # Call LLM with one retry on parse failure
    raw = await _call_llm(user_prompt, base_url, model, api_key)
    try:
        return _parse_response(raw, num_weeks)
    except (json.JSONDecodeError, KeyError, ValueError):
        # Retry with a correction nudge
        correction = (
            user_prompt
            + "\n\nYour previous response could not be parsed as valid JSON matching "
            "the required schema. Respond with ONLY the JSON object, nothing else."
        )
        raw = await _call_llm(correction, base_url, model, api_key)
        return _parse_response(raw, num_weeks)  # raises HTTP 503 if still invalid


async def generate_plan_llm(
    athlete: Athlete,
    config: PlanConfig,
    name: str,
    start_date: date,
    num_weeks: int,
    goal: Optional[str],
    session: AsyncSession,
    team: Team | None = None,
    team_id: str = "",
    user_id: str = "",
) -> TrainingPlan:
    """Generate a TrainingPlan using an LLM via OpenAI-compatible API."""

    weeks_data = await generate_plan_weeks_llm(
        athlete=athlete,
        config=config,
        num_weeks=num_weeks,
        goal=goal,
        session=session,
        team=team,
        team_id=team_id,
        user_id=user_id,
    )

    end_date = start_date + timedelta(weeks=num_weeks) - timedelta(days=1)

    plan = TrainingPlan(
        athlete_id=athlete.id,
        name=name,
        start_date=start_date,
        end_date=end_date,
        goal=goal,
        weeks=num_weeks,
        status="active",
        config=config.model_dump(),
        generation_method="llm",
    )
    session.add(plan)
    await session.flush()

    workouts: list[PlannedWorkout] = []
    for week_num, week_days in enumerate(weeks_data, start=1):
        for day in week_days:
            workouts.append(
                PlannedWorkout(
                    plan_id=plan.id,
                    week_number=week_num,
                    **day,
                )
            )

    session.add_all(workouts)
    await session.commit()
    await session.refresh(plan)
    return plan
