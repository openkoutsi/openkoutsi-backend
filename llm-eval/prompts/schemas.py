"""Pydantic output schemas for the JSON families, as provider ``response_format``.

The ``plan`` and ``workout`` families must return a JSON object that the backend
parsers accept. Instead of only asking for "a JSON object", we hand the provider
a JSON *schema* derived from a pydantic class, so a model that supports
structured outputs is constrained to the exact shape the app parses.

Matching the app
----------------
* ``WorkoutOutput`` reuses the backend's canonical :class:`WorkoutStep` and mirrors
  the workout prompt's own rule that *repeat blocks must not contain other repeat
  blocks* — so ``steps`` is a flat ``WorkoutStep | RepeatBlock`` list rather than the
  recursive ``WorkoutStepOrRepeat``. Anything valid against this schema passes
  ``llm_workout_generator._parse_steps`` (schema + max-nesting-depth-1 rule).
* ``PlanOutput`` mirrors ``llm_plan_generator._parse_response``: N weeks, each with
  a 7-entry ``workouts`` list, the app's ``workout_type`` enum, and null-on-rest
  duration/TSS.

Strict structured outputs accept only a subset of JSON Schema (no recursion, no
numeric/length bounds, ``additionalProperties: false`` with every property
required, ``anyOf`` rather than ``oneOf``). :func:`response_format` post-processes
pydantic's schema into that subset — the validation bounds still live on the
classes (and in the backend parsers) even though they're stripped from the wire
schema.
"""
from __future__ import annotations

import copy
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from openkoutsi.workout_schema import WorkoutStep

# ── workout ──────────────────────────────────────────────────────────────────
# Flat repeat block: a repeat contains only plain steps, never another repeat —
# exactly what the workout prompt demands and what the parser's depth rule allows.


class RepeatBlock(BaseModel):
    kind: Literal["repeat"]
    repeat_count: int = Field(ge=2)
    steps: list[WorkoutStep] = Field(min_length=1)


class WorkoutOutput(BaseModel):
    steps: list[
        Annotated[Union[WorkoutStep, RepeatBlock], Field(discriminator="kind")]
    ] = Field(min_length=1)


# ── plan ─────────────────────────────────────────────────────────────────────
# Mirrors llm_plan_generator._parse_response: the app's workout_type enum, days
# 1–7, and null duration/TSS on rest days.

_WORKOUT_TYPE = Literal[
    "recovery", "tempo", "threshold", "vo2max", "endurance",
    "long", "strength", "yoga", "cross-training", "rest",
]


class PlanWorkoutDay(BaseModel):
    day_of_week: int = Field(ge=1, le=7)
    workout_type: _WORKOUT_TYPE
    description: Optional[str] = None
    duration_min: Optional[int] = None
    target_tss: Optional[int] = None


class PlanWeek(BaseModel):
    week_number: int = Field(ge=1)
    workouts: list[PlanWorkoutDay]


class PlanOutput(BaseModel):
    weeks: list[PlanWeek]


# ── schema → strict response_format ──────────────────────────────────────────
# Keywords strict structured outputs don't accept — dropped from the wire schema
# (the pydantic classes and the backend parsers still enforce them).
_UNSUPPORTED_KEYS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern",
    "minItems", "maxItems", "uniqueItems",
    "discriminator", "default",
})


def _tighten(node):
    """Recursively coerce a pydantic JSON schema into the strict-output subset."""
    if isinstance(node, list):
        return [_tighten(n) for n in node]
    if not isinstance(node, dict):
        return node

    node = {k: v for k, v in node.items() if k not in _UNSUPPORTED_KEYS}
    if "oneOf" in node:  # discriminated unions render as oneOf; strict wants anyOf
        node["anyOf"] = node.pop("oneOf")

    node = {k: _tighten(v) for k, v in node.items()}

    if node.get("type") == "object" and "properties" in node:
        node["additionalProperties"] = False
        node["required"] = list(node["properties"].keys())
    return node


def response_format(model: type[BaseModel], name: str) -> dict:
    """Build an OpenAI-style ``response_format`` for ``model``.

    promptfoo merges this into the provider config, so the provider constrains the
    model to emit JSON conforming to ``model``'s (strict-subset) JSON schema.
    """
    schema = _tighten(copy.deepcopy(model.model_json_schema()))
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }
