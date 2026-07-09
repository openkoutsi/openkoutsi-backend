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
for family, scenarios in _families.items():
    for name in scenarios:
        msgs = build({"vars": {"family": family, "scenario": name}})
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

print("\n" + ("PASSED" if not failures else f"FAILED ({len(failures)} problem(s))"))
sys.exit(1 if failures else 0)
