"""promptfoo prompt provider — one function that renders any family's prompt.

promptfoo forms the cross-product of prompts x tests, so a single dispatching
prompt keyed on ``vars.family`` yields exactly the right matrix: each test row
(``{family, scenario}``) renders through the matching backend builder and no
other. The returned value is an OpenAI-style ``[system, user]`` chat array.

The ``plan`` and ``workout`` families expect a JSON object back, so for those we
return promptfoo's ``{"prompt": ..., "config": ...}`` shape and pin
``response_format`` to a JSON *schema* derived from a pydantic class that matches
what the backend parser accepts (see ``prompts/schemas.py``). The provider then
constrains the model to emit JSON of exactly that shape, which is what those
families' objective asserts parse. The prose families (``activity``, ``status``)
return the plain chat array unchanged.

Referenced from ``promptfooconfig.yaml`` as ``file://prompts/build.py:build``.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

from backend.app.services import (  # noqa: E402
    llm_activity_analyzer as activity_svc,
    llm_goal_guidance as goal_svc,
    llm_plan_generator as plan_svc,
    llm_training_status_analyzer as status_svc,
    llm_workout_generator as workout_svc,
)
from fixtures.scenarios import (  # noqa: E402
    ACTIVITY_SCENARIOS,
    GOAL_SCENARIOS,
    PLAN_SCENARIOS,
    STATUS_SCENARIOS,
    WORKOUT_SCENARIOS,
)
from prompts.schemas import PlanOutput, WorkoutOutput, response_format  # noqa: E402


def _plan(s: dict) -> tuple[str, str]:
    return (
        plan_svc._SYSTEM_PROMPT,
        plan_svc._build_user_prompt(s["config"], s["goal"], s["num_weeks"], s["ftp"], s["fitness"]),
    )


def _workout(s: dict) -> tuple[str, str]:
    return (
        workout_svc._SYSTEM_PROMPT,
        workout_svc._build_user_prompt(s["planned"], s["ftp"], s["sport"]),
    )


def _activity(s: dict) -> tuple[str, str]:
    return (
        activity_svc._build_system_prompt(s.get("locale"), s["activity"].sport_type),
        activity_svc._build_prompt(
            s["activity"], s["athlete"], s.get("fatigue"),
            s.get("power_pr_badges"), s.get("distance_pr_badges"),
        ),
    )


def _status(s: dict) -> tuple[str, str]:
    return (
        status_svc._build_system_prompt(s.get("locale"), s.get("coaching_style")),
        status_svc._build_status_prompt(
            s["athlete"], s["recent_activities"], s["current_metric"],
            s["active_plan"], s["this_week_workouts"], s["active_goals"], s["now"],
        ),
    )


def _goal(s: dict) -> tuple[str, str]:
    return (
        goal_svc._build_system_prompt(s.get("locale"), s.get("coaching_style")),
        goal_svc._build_goal_prompt(
            s["athlete"], s["goal"], s["recent_activities"],
            s["current_metric"], s["active_plan"], s["now"],
        ),
    )


_FAMILIES = {
    "plan": (PLAN_SCENARIOS, _plan),
    "workout": (WORKOUT_SCENARIOS, _workout),
    "activity": (ACTIVITY_SCENARIOS, _activity),
    "status": (STATUS_SCENARIOS, _status),
    "goal": (GOAL_SCENARIOS, _goal),
}

# JSON families → the pydantic output schema whose shape the backend parser accepts.
_JSON_SCHEMAS = {
    "plan": (PlanOutput, "training_plan"),
    "workout": (WorkoutOutput, "structured_workout"),
}


def build(context: dict):
    variables = context.get("vars", {})
    family = variables["family"]
    scenario = variables["scenario"]
    if family not in _FAMILIES:
        raise ValueError(f"unknown family {family!r} (expected one of {sorted(_FAMILIES)})")
    scenarios, builder = _FAMILIES[family]
    if scenario not in scenarios:
        raise ValueError(f"unknown {family} scenario {scenario!r} (have {sorted(scenarios)})")
    system, user = builder(scenarios[scenario])
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if family in _JSON_SCHEMAS:
        # promptfoo merges this `config` into the provider's config for this row, so
        # the JSON families get a schema-constrained response_format for free.
        model, name = _JSON_SCHEMAS[family]
        return {
            "prompt": messages,
            "config": {"response_format": response_format(model, name)},
        }
    return messages
