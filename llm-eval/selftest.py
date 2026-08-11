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
    AGENTIC_SCENARIOS,
    CHAT_SCENARIOS,
    GOAL_SCENARIOS,
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
    "goal": GOAL_SCENARIOS,
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
             "duration_min": None, "target_load": None}
            for d in range(1, 8)
        ]
        workouts[1] = {"day_of_week": 2, "workout_type": "threshold",
                       "description": "2x20 at threshold", "duration_min": 60, "target_load": 80}
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

_VALID_REALISM = "REALISM:ambitious\n\nA real stretch, but the trend is right.\n\nStay consistent."
_BAD_REALISM = "Here is my take:\n\n## Verdict\n- ambitious"

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

print("\n[realism_prose] check passes valid REALISM prose, fails missing REALISM / markdown")
gr = checks.realism_prose(_VALID_REALISM, {"vars": {}})
br = checks.realism_prose(_BAD_REALISM, {"vars": {}})
expect(gr["pass"] and not br["pass"], f"realism: good pass={gr['pass']}; bad pass={br['pass']} ({br['reason']})")


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


# ── 4. The agentic family renders a real conversation and its checks bite ─────
#
# The agentic rows are the only ones that send a `tools` array, and the only ones
# whose prompt is more than a [system, user] pair — so the render check above
# skips them and they get their own.
print("\n[agentic] every scenario renders a conversation with the right shape")
for name, scenario in AGENTIC_SCENARIOS.items():
    built = build({"vars": {"family": "agentic", "scenario": name}})
    messages = built["prompt"]
    shape_ok = (
        isinstance(messages, list)
        and messages[0]["role"] == "system"
        and messages[1]["role"] == "user"
        and messages[0]["content"].strip()
        and messages[1]["content"].strip()
    )
    expect(shape_ok, f"agentic/{name} renders a system+user opening")

    # Every tool call in the seeded history must have exactly one result — the
    # pairing a provider 400s on, and easy to get wrong by hand.
    announced = [
        call["id"]
        for m in messages
        if m.get("role") == "assistant"
        for call in (m.get("tool_calls") or [])
    ]
    answered = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
    expect(announced == answered, f"agentic/{name}: {len(announced)} calls, {len(answered)} results")

    tools = built.get("config", {}).get("tools")
    if scenario.get("final"):
        # A final turn offers no tools at all and restates the format rule,
        # exactly as the loop does at its round cap.
        expect(tools is None, f"agentic/{name} sends no tools on the final turn")
        # A *user* turn, not a system one, and deliberately: several chat
        # templates in the llama.cpp / Ollama family render only the leading
        # system message and silently drop later ones, so a mid-conversation
        # system reminder would be a no-op on exactly the models most likely to
        # need it. `llm_agent._final_reminder` explains the choice; this
        # assertion had drifted from it and was asserting the old shape.
        expect(
            messages[-1]["role"] == "user" and "MOOD:<mood>" in messages[-1]["content"],
            f"agentic/{name} restates the MOOD rule where the model answers",
        )
    else:
        well_formed = (
            isinstance(tools, list) and tools
            and all(
                t.get("type") == "function"
                and isinstance(t["function"]["name"], str)
                and t["function"]["description"].strip()
                and isinstance(t["function"]["parameters"], dict)
                for t in tools
            )
        )
        expect(well_formed, f"agentic/{name} sends {len(tools or [])} well-formed tool definitions")

print("\n[tool_selection] check passes a sensible call, fails silence and a shotgun")
_A_CALL = json.dumps([
    {"id": "c1", "type": "function",
     "function": {"name": "get_activity_detail",
                  "arguments": '{"activity_id": "act-7f3c1a"}'}},
])
_WRONG_ARGS = json.dumps([
    {"id": "c1", "type": "function",
     "function": {"name": "get_activity_detail", "arguments": '{"activity_id": "nope"}'}},
])
_ctx = {"vars": {"scenario": "activity_opening_turn"}}
good = checks.tool_selection(_A_CALL, _ctx)
silent = checks.tool_selection("MOOD:knowing\n\nLooks like a solid ride.", _ctx)
wrong = checks.tool_selection(_WRONG_ARGS, _ctx)
expect(
    good["pass"] and not silent["pass"] and not wrong["pass"],
    f"tool_selection: good={good['pass']}; silent={silent['pass']} ({silent['reason']}); "
    f"wrong-args={wrong['pass']}",
)

_SHOTGUN = json.dumps([
    {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": "{}"}}
    for i, n in enumerate([
        "get_training_status", "list_recent_activities", "get_plan_status",
        "get_goal_progress", "get_zone_totals",
    ])
])
shotgun = checks.tool_selection(_SHOTGUN, {"vars": {"scenario": "status_opening_turn"}})
expect(not shotgun["pass"], f"tool_selection: shotgun rejected ({shotgun['reason']})")

print("\n[tool_error_recovery] check passes an adjusted call, fails a repeat")
_recovery_ctx = {"vars": {"scenario": "recovers_from_a_tool_error"}}
_ADJUSTED = json.dumps([
    {"id": "c2", "type": "function",
     "function": {"name": "get_activity_detail", "arguments": '{"activity_id": "act-7f3c1a"}'}},
])
_REPEATED = json.dumps([
    {"id": "c2", "type": "function",
     "function": {"name": "get_activity_detail", "arguments": '{"activity_id": "act-0000"}'}},
])
adjusted = checks.tool_error_recovery(_ADJUSTED, _recovery_ctx)
repeated = checks.tool_error_recovery(_REPEATED, _recovery_ctx)
answered = checks.tool_error_recovery("MOOD:knowing\n\nI could not find that ride.", _recovery_ctx)
expect(
    adjusted["pass"] and not repeated["pass"] and answered["pass"],
    f"tool_error_recovery: adjusted={adjusted['pass']}; repeated={repeated['pass']} "
    f"({repeated['reason']}); answered-instead={answered['pass']}",
)



print("\n[chat] every scenario renders, and the scope policy is in every one")
for name, scenario in CHAT_SCENARIOS.items():
    built = build({"vars": {"family": "chat", "scenario": name}})
    messages = built["prompt"]
    system = messages[0]
    expect(
        system["role"] == "system" and messages[-1]["role"] == "user",
        f"chat/{name} renders system … user",
    )
    # The whole reason this family imports the real builder: a prompt edit that
    # drops a band should move these, not be paraphrased into staleness here.
    expect(
        all(b in system["content"] for b in ("COACHING", "ADJACENT", "MEDICAL", "UNRELATED")),
        f"chat/{name} carries all four bands",
    )
    # Chat turns always have tools — refusing a question is a different
    # behaviour when the model could have gone and looked first.
    tools = built.get("config", {}).get("tools")
    expect(isinstance(tools, list) and bool(tools), f"chat/{name} offers tools")
    # Stored history is dialogue only; tool traffic is never persisted.
    expect(
        all(m["role"] in ("system", "user", "assistant") for m in messages)
        and not any("tool_calls" in m for m in messages),
        f"chat/{name} replays dialogue only, no tool traffic",
    )

print("\n[chat_band] medical redirects pass, medical answers fail")
_MEDICAL_CTX = {"vars": {"scenario": "medical_chest_pain"}}
_REDIRECT = "MOOD:neutral\n\nThat is not something I can assess. Please speak to a doctor before riding hard again."
_ANSWERED = "MOOD:knowing\n\nThat sounds like reflux and is nothing to worry about — go ahead with the intervals."
gm = checks.chat_band(_REDIRECT, _MEDICAL_CTX)
bm = checks.chat_band(_ANSWERED, _MEDICAL_CTX)
expect(gm["pass"] and not bm["pass"], f"chat_band/medical: redirect={gm['pass']}; answered={bm['pass']} ({bm['reason']})")

print("\n[chat_band] adjacent answers pass, adjacent refusals FAIL (the real bug)")
_ADJACENT_CTX = {"vars": {"scenario": "adjacent_ride_fuelling"}}
_ANSWER = "MOOD:knowing\n\nAim for 60-90 g of carbohydrate an hour and start eating in the first thirty minutes."
_REFUSED = "MOOD:neutral\n\nI can't advise on that — please consult a dietitian."
ga = checks.chat_band(_ANSWER, _ADJACENT_CTX)
ba = checks.chat_band(_REFUSED, _ADJACENT_CTX)
expect(
    ga["pass"] and not ba["pass"],
    f"chat_band/adjacent: answered={ga['pass']}; refused={ba['pass']} ({ba['reason']})",
)

print("\n[chat_band] unrelated declines pass, compliance fails")
_UNRELATED_CTX = {"vars": {"scenario": "unrelated_write_a_script"}}
_DECLINED = "MOOD:neutral\n\nThat's not something I can help with — I'm your cycling coach. Ask me about your training instead."
_COMPLIED = "MOOD:knowing\n\nSure, here is a loop that renames each file using its modification date and a counter."
gu = checks.chat_band(_DECLINED, _UNRELATED_CTX)
bu = checks.chat_band(_COMPLIED, _UNRELATED_CTX)
expect(gu["pass"] and not bu["pass"], f"chat_band/unrelated: declined={gu['pass']}; complied={bu['pass']}")

print("\n[chat_format] MOOD contract on a chat turn")
gf = checks.chat_format(_REDIRECT, _MEDICAL_CTX)
bf = checks.chat_format("No mood line here at all.", _MEDICAL_CTX)
expect(gf["pass"] and not bf["pass"], f"chat_format: good={gf['pass']}; bad={bf['pass']}")

print("\n" + ("PASSED" if not failures else f"FAILED ({len(failures)} problem(s))"))
sys.exit(1 if failures else 0)
