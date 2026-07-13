"""Unit tests for the shared LLM output schemas / ``response_format`` builder.

Mirrors ``llm-eval/selftest.py``'s assertions but as a first-class pytest: the
strict ``response_format`` for both generators must be structured-output
conformant (closed objects, every property required, no unsupported keyword) and
its shape must agree with the backend parsers on a valid sample.
"""
import json

import pytest

from backend.app.services.llm_plan_generator import _parse_response
from backend.app.services.llm_workout_generator import _parse_steps
from backend.app.services.llm_schemas import (
    PLAN_RESPONSE_FORMAT,
    WORKOUT_RESPONSE_FORMAT,
    PlanOutput,
    WorkoutOutput,
    response_format,
)

# Keywords a strict json_schema must not contain (they live on the pydantic
# classes / backend parsers, not on the wire schema).
_UNSUPPORTED = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems",
    "discriminator", "default", "oneOf",
}


def _strict_problems(node, path="$") -> list[str]:
    problems: list[str] = []
    if isinstance(node, list):
        for i, n in enumerate(node):
            problems += _strict_problems(n, f"{path}[{i}]")
        return problems
    if not isinstance(node, dict):
        return problems
    for bad in _UNSUPPORTED:
        if bad in node:
            problems.append(f"{path}: unsupported keyword {bad!r}")
    if node.get("type") == "object" and "properties" in node:
        if node.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties must be false")
        if set(node.get("required", [])) != set(node["properties"]):
            problems.append(f"{path}: every property must be required")
    for key, val in node.items():
        problems += _strict_problems(val, f"{path}.{key}")
    return problems


@pytest.mark.parametrize(
    "model, name",
    [(PlanOutput, "training_plan"), (WorkoutOutput, "structured_workout")],
)
def test_response_format_is_strict_conformant(model, name):
    rf = response_format(model, name)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == name
    assert rf["json_schema"]["strict"] is True
    problems = _strict_problems(rf["json_schema"]["schema"])
    assert not problems, "; ".join(problems)


def test_prebuilt_payloads_match_the_builder():
    assert PLAN_RESPONSE_FORMAT == response_format(PlanOutput, "training_plan")
    assert WORKOUT_RESPONSE_FORMAT == response_format(WorkoutOutput, "structured_workout")


def _valid_plan_json(num_weeks: int) -> str:
    weeks = []
    for w in range(1, num_weeks + 1):
        workouts = [
            {"day_of_week": d, "workout_type": "rest", "description": None,
             "duration_min": None, "target_tss": None}
            for d in range(1, 8)
        ]
        workouts[1] = {"day_of_week": 2, "workout_type": "threshold",
                       "description": "2x20 at threshold", "duration_min": 60, "target_tss": 80}
        weeks.append({"week_number": w, "workouts": workouts})
    return json.dumps({"weeks": weeks})


_VALID_WORKOUT = json.dumps({"steps": [
    {"kind": "step", "step_type": "warmup", "duration": {"type": "time", "seconds": 600},
     "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 50}}},
    {"kind": "repeat", "repeat_count": 4, "steps": [
        {"kind": "step", "step_type": "active", "duration": {"type": "time", "seconds": 300},
         "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 105}}},
        {"kind": "step", "step_type": "recovery", "duration": {"type": "time", "seconds": 180}},
    ]},
    {"kind": "step", "step_type": "cooldown", "duration": {"type": "time", "seconds": 600}},
]})


def test_plan_sample_agrees_with_pydantic_and_parser():
    sample = _valid_plan_json(3)
    PlanOutput.model_validate_json(sample)  # satisfies the wire schema's model
    assert len(_parse_response(sample, 3)) == 3  # and the backend parser


def test_workout_sample_agrees_with_pydantic_and_parser():
    WorkoutOutput.model_validate_json(_VALID_WORKOUT)
    steps = _parse_steps(_VALID_WORKOUT)
    assert steps[1]["kind"] == "repeat"
