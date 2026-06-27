"""
LLM-based structured workout synthesizer.

Turns a loose ``PlannedWorkout`` descriptor (workout_type / description /
duration_min / target_tss) into a fully structured ``WorkoutDefinition`` with an
interval tree, suitable for pushing to Wahoo (or any other device target).

Reuses the shared ``llm_client`` helpers for config resolution, the HTTP call
and JSON extraction. The parsed steps are validated against the canonical
``WorkoutStepOrRepeat`` pydantic schema and the max-repeat-nesting-depth-1 rule;
on persistent invalid output the caller is expected to skip that workout rather
than aborting a whole batch.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from pydantic import TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.registry_orm import Team
from ..models.team_orm import Athlete, PlannedWorkout, WorkoutDefinition
from .llm_client import call_llm, extract_json, resolve_llm_config
from openkoutsi.workout_estimator import estimate_duration_s, estimate_tss
from openkoutsi.workout_schema import RepeatBlock, WorkoutStepOrRepeat

log = logging.getLogger(__name__)

_steps_adapter: TypeAdapter[list[WorkoutStepOrRepeat]] = TypeAdapter(
    list[WorkoutStepOrRepeat]
)


class WorkoutGenerationError(Exception):
    """Raised when the LLM cannot produce a valid structured workout."""


_SYSTEM_PROMPT = """\
You are an expert endurance sports coach that designs structured interval workouts.
You MUST respond with ONLY valid JSON — no markdown, no prose, no code fences.
The JSON must conform exactly to the schema provided by the user.
Do not include any explanation or commentary outside the JSON object.
"""

_SCHEMA_EXAMPLE = """\
{
  "steps": [
    {
      "kind": "step",
      "step_type": "warmup",
      "duration": {"type": "time", "seconds": 600},
      "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 50}}
    },
    {
      "kind": "repeat",
      "repeat_count": 4,
      "steps": [
        {
          "kind": "step",
          "step_type": "active",
          "duration": {"type": "time", "seconds": 300},
          "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 105}}
        },
        {
          "kind": "step",
          "step_type": "recovery",
          "duration": {"type": "time", "seconds": 180},
          "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 50}}
        }
      ]
    },
    {
      "kind": "step",
      "step_type": "cooldown",
      "duration": {"type": "time", "seconds": 600},
      "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 45}}
    }
  ]
}

Rules:
- "steps" is an ordered list of step objects and/or repeat blocks.
- Each step has "kind": "step", a "step_type" of one of
  "warmup", "active", "recovery", "cooldown", "rest", "other",
  a "duration" ({"type": "time", "seconds": <int>}), and an optional power "target".
- A repeat block has "kind": "repeat", an integer "repeat_count" (>= 2) and a
  nested "steps" list. Repeat blocks MUST NOT contain other repeat blocks.
- Power targets use "spec": {"type": "pct_ftp", "pct": <number>} (percent of FTP),
  or {"type": "absolute", "value": <watts>}, or {"type": "zone", "zone_number": <int>}.
- Prefer percent-of-FTP targets so the workout scales to the athlete.
- Start with a warmup and end with a cooldown.
- Make the total time and intensity roughly match the requested duration and TSS.
"""


def _build_user_prompt(planned: PlannedWorkout, ftp: Optional[int], sport: str) -> str:
    lines = [
        "Design a single structured workout matching this description:",
        "",
        f"Workout type: {planned.workout_type or 'endurance'}",
        f"Sport: {sport}",
    ]
    if planned.description:
        lines.append(f"Description: {planned.description}")
    if planned.duration_min is not None:
        lines.append(f"Target duration: {planned.duration_min} minutes")
    if planned.target_tss is not None:
        lines.append(f"Target training stress (TSS): {planned.target_tss}")
    if ftp:
        lines.append(f"Athlete FTP: {ftp}W")

    lines += [
        "",
        "Output ONLY the JSON object in the schema below.",
        "",
        _SCHEMA_EXAMPLE,
    ]
    return "\n".join(lines)


def _parse_steps(raw: str) -> list[dict]:
    """Parse, validate and serialise the LLM response into workout steps.

    Raises ``WorkoutGenerationError`` (wrapping the underlying cause) when the
    response is not valid JSON, does not match the schema, or violates the
    max-repeat-nesting-depth-1 rule.
    """
    try:
        data = json.loads(extract_json(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        raise WorkoutGenerationError(f"response was not valid JSON: {exc}") from exc

    steps_raw = data.get("steps") if isinstance(data, dict) else data
    if not isinstance(steps_raw, list) or not steps_raw:
        raise WorkoutGenerationError("response did not contain a non-empty 'steps' list")

    try:
        validated = _steps_adapter.validate_python(steps_raw)
    except ValidationError as exc:
        raise WorkoutGenerationError(f"steps did not match schema: {exc}") from exc

    for step in validated:
        if isinstance(step, RepeatBlock) and step.max_depth() > 1:
            raise WorkoutGenerationError("repeat blocks may not be nested")

    return [s.model_dump() for s in validated]


async def generate_workout_definition_llm(
    athlete: Athlete,
    planned_workout: PlannedWorkout,
    session: AsyncSession,
    team: Team | None = None,
    team_id: str = "",
    user_id: str = "",
    sport_type: str = "Ride",
) -> WorkoutDefinition:
    """Synthesize and persist a structured ``WorkoutDefinition`` for a planned workout.

    The returned definition is linked back onto ``planned_workout.workout_definition_id``
    (the row is flushed, not committed — the caller owns the transaction).

    Raises ``ValueError`` when the LLM is not configured and
    ``WorkoutGenerationError`` when the model cannot produce a valid workout.
    """
    base_url, model, api_key = resolve_llm_config(athlete, team, team_id, user_id)

    user_prompt = _build_user_prompt(planned_workout, athlete.ftp, sport_type)

    raw = await call_llm(
        user_prompt, base_url, model, api_key, system_prompt=_SYSTEM_PROMPT
    )
    try:
        steps = _parse_steps(raw)
    except WorkoutGenerationError:
        # Retry once with a correction nudge.
        correction = (
            user_prompt
            + "\n\nYour previous response could not be parsed as valid JSON matching "
            "the required schema. Respond with ONLY the JSON object, nothing else."
        )
        raw = await call_llm(
            correction, base_url, model, api_key, system_prompt=_SYSTEM_PROMPT
        )
        steps = _parse_steps(raw)  # raises WorkoutGenerationError if still invalid

    name = (planned_workout.workout_type or "Workout").replace("-", " ").title()

    workout = WorkoutDefinition(
        id=str(uuid.uuid4()),
        athlete_id=athlete.id,
        name=name,
        description=planned_workout.description,
        sport_type=sport_type,
        steps=steps,
        estimated_duration_s=estimate_duration_s(steps),
        estimated_tss=estimate_tss(steps, athlete.ftp),
    )
    session.add(workout)
    await session.flush()

    planned_workout.workout_definition_id = workout.id

    return workout
