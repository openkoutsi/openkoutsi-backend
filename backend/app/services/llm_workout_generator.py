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

import httpx
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.registry_orm import InstanceSettings
from ..models.user_orm import Athlete, PlannedWorkout, WorkoutDefinition
from .llm_access import record_llm_usage
from .llm_client import (
    ResolvedLlm,
    call_llm,
    extract_json,
    is_response_format_unsupported_error,
    resolve_llm_config,
)
from .llm_schemas import WORKOUT_RESPONSE_FORMAT
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


async def _call_llm_with_schema(user_prompt: str, cfg: ResolvedLlm) -> tuple[str, dict | None]:
    """Call the chat endpoint with the workout system prompt and strict schema.

    Sends the workout ``response_format`` by default (unless the preset opts out
    via ``structured_outputs: false``). If the provider rejects it, transparently
    drops the field and re-issues the prompt-instructed call, so unsupported
    providers degrade to the prompt+parse+retry path.
    """
    response_format = WORKOUT_RESPONSE_FORMAT if cfg.structured_outputs else None
    try:
        return await call_llm(
            user_prompt, cfg.base_url, cfg.model, cfg.api_key, system_prompt=_SYSTEM_PROMPT,
            extra_headers=cfg.extra_headers, extra_body=cfg.extra_body,
            response_format=response_format,
        )
    except httpx.HTTPStatusError as exc:
        if response_format is None or not is_response_format_unsupported_error(exc):
            raise
        log.info("provider rejected response_format for workout generation; retrying without it")
        return await call_llm(
            user_prompt, cfg.base_url, cfg.model, cfg.api_key, system_prompt=_SYSTEM_PROMPT,
            extra_headers=cfg.extra_headers, extra_body=cfg.extra_body,
        )


async def generate_workout_definition_llm(
    athlete: Athlete,
    planned_workout: PlannedWorkout,
    session: AsyncSession,
    instance: InstanceSettings | None = None,
    user_id: str = "",
    sport_type: str = "Ride",
    allow_instance_fallback: bool = True,
) -> WorkoutDefinition:
    """Synthesize and persist a structured ``WorkoutDefinition`` for a planned workout.

    The returned definition is linked back onto ``planned_workout.workout_definition_id``
    (the row is flushed, not committed — the caller owns the transaction).

    ``allow_instance_fallback=False`` (issue #9, BYOK-mode on a gated instance)
    forbids falling back to the instance credentials. Raises ``ValueError`` when
    the LLM is not configured and ``WorkoutGenerationError`` when the model
    cannot produce a valid workout.
    """
    cfg = resolve_llm_config(
        athlete, instance, user_id, allow_instance_fallback=allow_instance_fallback
    )

    user_prompt = _build_user_prompt(planned_workout, athlete.ftp, sport_type)

    raw, usage = await _call_llm_with_schema(user_prompt, cfg)
    await record_llm_usage(user_id=user_id, feature="workout_generate", cfg=cfg, usage=usage)
    try:
        steps = _parse_steps(raw)
    except WorkoutGenerationError:
        # Retry once with a correction nudge.
        correction = (
            user_prompt
            + "\n\nYour previous response could not be parsed as valid JSON matching "
            "the required schema. Respond with ONLY the JSON object, nothing else."
        )
        raw, usage = await _call_llm_with_schema(correction, cfg)
        await record_llm_usage(user_id=user_id, feature="workout_generate", cfg=cfg, usage=usage)
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
