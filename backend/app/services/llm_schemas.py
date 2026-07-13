"""Pydantic output schemas for the JSON generators, as provider ``response_format``.

The training-plan generator (``llm_plan_generator``) and the structured-workout
synthesizer (``llm_workout_generator``) both return a JSON object that a backend
parser accepts. Instead of only *asking* the model for "a JSON object", we hand
the provider a JSON *schema* derived from a pydantic class, so a model that
supports structured outputs is constrained to the exact shape the app parses.

This module is the single source of truth for those schemas: the runtime
generators build their ``response_format`` from it, and the offline eval harness
(``llm-eval/prompts/schemas.py``) re-exports it, so the two never drift.

Reusing the backend's schemas
-----------------------------
Where the backend already models a shape, we import it rather than restate it, so
these stay in lock-step with production:

* ``WorkoutOutput.steps`` reuses the canonical :class:`WorkoutStep`. The canonical
  ``RepeatBlock`` / ``WorkoutStepOrRepeat`` (and ``WorkoutDefinitionCreate.steps``)
  are **recursive**, which strict structured outputs don't allow — so the repeat
  block is restated here as a flat ``list[WorkoutStep]``. That also matches the
  workout prompt's own rule that *repeat blocks must not contain other repeat
  blocks*; anything valid here passes ``_parse_steps`` (schema + depth-1 rule).
* ``PlanWeek.workouts`` reuses :class:`WorkoutCreate` — the backend's own "single
  workout day as returned by the LLM". Only the ``{weeks: [{week_number,
  workouts}]}`` wrapper that ``llm_plan_generator._parse_response`` expects is not
  modelled anywhere in the backend, so ``PlanWeek`` / ``PlanOutput`` define it.

Strict structured outputs accept only a subset of JSON Schema (no recursion, no
numeric/length bounds, ``additionalProperties: false`` with every property
required, ``anyOf`` rather than ``oneOf``). :func:`response_format` post-processes
pydantic's schema into that subset — the validation bounds still live on the
classes (and in the backend parsers) even though they're stripped from the wire
schema.
"""
from __future__ import annotations

import copy
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from ..schemas.plans import WorkoutCreate
from openkoutsi.workout_schema import WorkoutStep

# ── workout ──────────────────────────────────────────────────────────────────
# Flat repeat block: a repeat contains only plain steps, never another repeat —
# the recursive canonical RepeatBlock can't be expressed as a strict json_schema.


class RepeatBlock(BaseModel):
    kind: Literal["repeat"]
    repeat_count: int = Field(ge=2)
    steps: list[WorkoutStep] = Field(min_length=1)


class WorkoutOutput(BaseModel):
    steps: list[
        Annotated[Union[WorkoutStep, RepeatBlock], Field(discriminator="kind")]
    ] = Field(min_length=1)


# ── plan ─────────────────────────────────────────────────────────────────────
# WorkoutCreate is the backend's own LLM-day model; only the weeks wrapper that
# _parse_response walks is unmodelled, so it lives here.


class PlanWeek(BaseModel):
    week_number: int = Field(ge=1)
    workouts: list[WorkoutCreate]


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

    The provider merges this into the chat-completion payload so it constrains the
    model to emit JSON conforming to ``model``'s (strict-subset) JSON schema.
    """
    schema = _tighten(copy.deepcopy(model.model_json_schema()))
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


# Pre-built once at import — the two payloads are static, so build them here and
# reuse them for every request rather than re-deriving the schema each call.
PLAN_RESPONSE_FORMAT: dict = response_format(PlanOutput, "training_plan")
WORKOUT_RESPONSE_FORMAT: dict = response_format(WorkoutOutput, "structured_workout")
