"""Offline sanity check — no network, no API keys.

Verifies the two things promptfoo would otherwise only exercise against a live
model: (1) every scenario renders a [system, user] prompt through the real
backend builders, and (2) each objective assert passes a hand-crafted valid
output and fails a hand-crafted bad one — i.e. the checks actually bite.

Run:  cd llm-eval && ../.venv/bin/python selftest.py   (or: uv run --project .. python selftest.py)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "asserts"))

from prompts.build import build  # noqa: E402
from asserts import checks  # noqa: E402
from fixtures.scenarios import (  # noqa: E402
    ACTIVITY_SCENARIOS,
    PLAN_SCENARIOS,
    STATUS_SCENARIOS,
    WORKOUT_SCENARIOS,
)

failures: list[str] = []


def expect(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


# ── 1. Every scenario renders through the real builders ──────────────────────
print("[render] all scenarios build a [system, user] prompt")
_families = {
    "plan": PLAN_SCENARIOS,
    "workout": WORKOUT_SCENARIOS,
    "activity": ACTIVITY_SCENARIOS,
    "status": STATUS_SCENARIOS,
}
_JSON_FAMILIES = {"plan", "workout"}
for family, scenarios in _families.items():
    for name in scenarios:
        result = build({"vars": {"family": family, "scenario": name}})
        if family in _JSON_FAMILIES:
            # JSON families return promptfoo's {prompt, config} shape and pin a
            # json_schema response_format so the model must emit the parseable shape.
            rf = result.get("config", {}).get("response_format") if isinstance(result, dict) else None
            forced = (
                isinstance(rf, dict)
                and rf.get("type") == "json_schema"
                and rf.get("json_schema", {}).get("strict") is True
                and isinstance(rf.get("json_schema", {}).get("schema"), dict)
            )
            expect(forced, f"{family}/{name} pins a strict json_schema response")
            msgs = result["prompt"] if isinstance(result, dict) else result
        else:
            msgs = result
        ok = (
            isinstance(msgs, list) and len(msgs) == 2
            and msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
            and msgs[0]["content"].strip() and msgs[1]["content"].strip()
        )
        expect(ok, f"{family}/{name} renders")


# ── 2. Objective asserts pass valid output and fail bad output ───────────────
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
# repeat_count=1 violates the >=2 schema rule → production would reject it.
_BAD_WORKOUT = json.dumps({"steps": [
    {"kind": "repeat", "repeat_count": 1, "steps": [
        {"kind": "step", "step_type": "active", "duration": {"type": "time", "seconds": 60}},
    ]},
]})

_VALID_PROSE = "MOOD:cheer\n\nGreat ride today, the numbers back it up.\n\nRecover well tomorrow."
_BAD_PROSE = "Here is your analysis:\n\n## Summary\n- great ride"

print("\n[plan] check passes valid, fails wrong week count")
for name, s in PLAN_SCENARIOS.items():
    ctx = {"vars": {"scenario": name}}
    good = checks.plan(_valid_plan_json(s["num_weeks"]), ctx)
    bad = checks.plan(_valid_plan_json(s["num_weeks"] + 1), ctx)  # too many weeks
    expect(good["pass"] and not bad["pass"], f"plan/{name}: good pass={good['pass']} bad pass={bad['pass']}")

print("\n[workout] check passes valid, fails repeat_count<2")
gw = checks.workout(_VALID_WORKOUT, {"vars": {}})
bw = checks.workout(_BAD_WORKOUT, {"vars": {}})
expect(gw["pass"] and not bw["pass"], f"workout: good pass={gw['pass']} ({gw['reason']}); bad pass={bw['pass']} ({bw['reason']})")

print("\n[mood_prose] check passes valid MOOD prose, fails missing MOOD / markdown")
gm = checks.mood_prose(_VALID_PROSE, {"vars": {}})
bm = checks.mood_prose(_BAD_PROSE, {"vars": {}})
expect(gm["pass"] and not bm["pass"], f"mood: good pass={gm['pass']}; bad pass={bm['pass']} ({bm['reason']})")


# ── 3. The JSON-schema response_format is strict-conformant and app-aligned ───
from prompts.schemas import PlanOutput, WorkoutOutput, response_format  # noqa: E402

_ALLOWED_UNSUPPORTED = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems",
    "discriminator", "default", "oneOf",
}


def _strict_problems(node, path="$") -> list[str]:
    """Every object must be closed + fully-required, with no unsupported keyword."""
    problems: list[str] = []
    if isinstance(node, list):
        for i, n in enumerate(node):
            problems += _strict_problems(n, f"{path}[{i}]")
        return problems
    if not isinstance(node, dict):
        return problems
    for bad in _ALLOWED_UNSUPPORTED:
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


print("\n[schema] response_format is strict-output-conformant")
for model, name in ((PlanOutput, "training_plan"), (WorkoutOutput, "structured_workout")):
    rf = response_format(model, name)
    probs = _strict_problems(rf["json_schema"]["schema"])
    expect(not probs, f"{name}: {'; '.join(probs) if probs else 'strict-conformant'}")

print("\n[schema] pydantic model and backend parser agree on a valid sample")
# A sample that satisfies the pydantic schema must also satisfy the app parser.
pw = PLAN_SCENARIOS[next(iter(PLAN_SCENARIOS))]["num_weeks"]
plan_sample = _valid_plan_json(pw)
PlanOutput.model_validate_json(plan_sample)
expect(checks.plan(plan_sample, {"vars": {"scenario": next(iter(PLAN_SCENARIOS))}})["pass"],
       "plan sample validates against PlanOutput and the app parser")
WorkoutOutput.model_validate_json(_VALID_WORKOUT)
expect(checks.workout(_VALID_WORKOUT, {"vars": {}})["pass"],
       "workout sample validates against WorkoutOutput and the app parser")

print("\n" + ("PASSED" if not failures else f"FAILED ({len(failures)} problem(s))"))
sys.exit(1 if failures else 0)
