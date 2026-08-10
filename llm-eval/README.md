# llm-eval — comparing LLM providers/models for openkoutsi

openkoutsi calls an LLM in five places, all through one OpenAI-compatible
`/chat/completions` path (`backend/app/services/llm_client.py:call_llm`) — and,
since issue #43, two of those five can also run as an agent loop over the MCP
tools. This subproject sends prompts that **mirror what the platform actually
sends** to a matrix of models and grades the results, so hosters and BYOK users
can pick a model with evidence.

It's a thin [promptfoo](https://www.promptfoo.dev/) project. The substance is the
evaluation prompt set: instead of copying the prompts, we **import the real
backend builders**, so the text each model sees is byte-identical to production
and can never drift.

## The six families

| Family | Backend source (`backend/app/services/…`) | Output | How it's graded |
|---|---|---|---|
| `plan` | `llm_plan_generator.py` | JSON | **objective** — reuses `_parse_response` (N weeks, 7 days/week, valid type, null-on-rest) |
| `workout` | `llm_workout_generator.py` | JSON | **objective** — reuses `_parse_steps` (`WorkoutStepOrRepeat` schema + nesting rule) |

The two JSON families (`plan`, `workout`) pin `response_format` to a JSON
**schema** (`{"type": "json_schema", ...}`) via the prompt function, so a model
that supports structured outputs is constrained to emit exactly the shape the
backend parser accepts — not just "some JSON object". The schemas are derived
from pydantic classes that mirror the parsers (`WorkoutOutput` ↔ `_parse_steps`,
`PlanOutput` ↔ `_parse_response`) and are post-processed into the strict
structured-output subset (closed objects, all properties required, `anyOf`
unions, no unsupported keywords). These now live in the backend
(`backend/app/services/llm_schemas.py`) as the single source of truth — the
runtime generators send the same `response_format`; `prompts/schemas.py`
re-exports them so the eval and production never drift. The prose families
(`activity`, `status`) are left unconstrained.
| `activity` | `llm_activity_analyzer.py` | prose | **format objective** (`MOOD:` line, no markdown) + **subjective** (web UI / optional rubric) |
| `status` | `llm_training_status_analyzer.py` | prose | same as `activity`, plus plan-adherence reasoning |
| `goal` | `llm_goal_guidance.py` | prose | **format objective** (`REALISM:` line, no markdown) + **subjective** (realism judgement + concrete steps) |
| `agentic` | `llm_agent.py` + the two analyzers | tool calls / prose | **objective** — did it call tools, the right ones, recover from a tool error, and still start with `MOOD:`? |

### The `agentic` family and the tool-calling verdict

The other five families are one prompt in, one answer out, which is what those
call sites do. The agentic path (issue #43) is a *conversation*, and promptfoo
evaluates one turn per row — so rather than pretend to drive a loop, each row
freezes the conversation at the turn whose behaviour is in question and asks one
thing of the model:

| Scenario | Question | Assert |
|---|---|---|
| `status_opening_turn` | handed `tools` and a broad question, does it go and look — and not shotgun every tool at once? | `tool_selection` |
| `activity_opening_turn` | the activity id is already in the brief; is `get_activity_detail` the first call, with that id? | `tool_selection` |
| `recovers_from_a_tool_error` | a tool replied with prose naming the nearby rides; does it adjust, or retry the call that just failed? | `tool_error_recovery` |
| `final_turn_after_tool_results` | does `MOOD:` survive a turn that follows tool results? | `mood_prose` |
| `final_turn_finnish` | the same, in Finnish, with the `MOOD:` token still English | `mood_prose` |

Read together, these are the roster's **"can this model run agentic Koutsi"**
column — and the two halves of it fail differently, so grade them differently:

* **Fails an opening turn** → the model gains nothing from the agentic path. It
  is still perfectly usable: in production the run detects this and falls back
  to the single-shot blob prompt, which is well-tuned. Leave `agentic_koutsi`
  off for it.
* **Fails a final turn** → the model is *actively unsuited* to the agentic path.
  A missing `MOOD:` line costs the Koutsi avatar on every card, and unlike a
  refused `tools` param nothing detects it at runtime. This is what
  `"tools_supported": false` on the preset is for.

The seeded tool results are hand-written stand-ins shaped like the real tools'
output; the *prompts*, tool definitions and final-turn reminder all come from the
running code, so what the model reads is what production sends.

## Layout

```
promptfooconfig.yaml   # providers × tests; per-family asserts. Model roster lives here.
prompts/build.py       # one prompt fn; dispatches on vars.family to the real backend builder
prompts/schemas.py     # re-exports the backend's llm_schemas (json_schema response_format for plan/workout)
fixtures/scenarios.py  # in-memory ORM objects / PlanConfig per scenario (the eval inputs);
                       # the agentic scenarios also carry frozen conversations + expectations
asserts/checks.py      # objective asserts that reuse the backend's own parsers
selftest.py            # offline check: renders every scenario, proves asserts bite (no keys)
_bootstrap.py          # puts repo root on sys.path + sets a dummy SECRET_KEY for imports
```

## Running

Prereqs: the repo's `uv` env (`uv sync` in the repo root creates `.venv`) and Node.

```sh
cd llm-eval

# 1. Offline sanity check — no API keys, no network:
../.venv/bin/python selftest.py

# 2. Full evaluation against real models:
export PROMPTFOO_PYTHON=../.venv/bin/python   # so promptfoo's Python can import backend/*
export ANTHROPIC_API_KEY=...                  # + OPENAI_API_KEY / GEMINI_API_KEY as needed
npx --yes promptfoo@latest eval

# 3. Review side-by-side (objective pass/fail + eyeball the prose):
npx --yes promptfoo@latest view
```

`PROMPTFOO_PYTHON` **must** point at the project venv — otherwise promptfoo's
Python subprocess can't import `backend.*` / `openkoutsi.*`.

## Adding a model

Uncomment or add a row under `providers:` in `promptfooconfig.yaml`. Every model
is called as `openai:chat:<model>` with a `config.apiBaseUrl` pointing at that
provider's OpenAI-compatible endpoint — the same way `call_llm` (and BYOK) reach
it. Current Claude ids/pricing are noted inline; fill in ids for OpenAI, Gemini,
and local (Ollama/vLLM) as needed. Keys come from env vars, never the file.

## Adding a scenario

Add an entry to the relevant `*_SCENARIOS` dict in `fixtures/scenarios.py` and a
matching test row (`{family, scenario}` + the family's assert) in
`promptfooconfig.yaml`. `selftest.py` picks it up automatically.

An `AGENTIC_SCENARIOS` entry carries a bit more: `surface` (`status` or
`activity`) picks which builders render the prompt, `history` is the frozen
conversation, and the remaining keys are the expectations the assert reads —
`allowed_tools` / `max_calls` / `expected_arguments` for `tool_selection`,
`must_not_repeat` for `tool_error_recovery`, or `final: true` for a turn that
drops the tools array and restates the format rule. The selftest checks that
every seeded tool call in a `history` has exactly one matching result, which is
the pairing a provider 400s on and the easiest thing to get wrong by hand.

## Subjective grading

Objective checks only cover structure/format. For coaching quality, use
`promptfoo view` to read outputs side-by-side per scenario. For an automated
first pass, uncomment the `llm-rubric` asserts in `promptfooconfig.yaml` (costs
extra grader tokens).

## Keeping prompts in sync

The prompts are imported, not copied, so they track the backend automatically.
If the backend refactors these builders, update the imports in `prompts/build.py`
/ `asserts/checks.py` accordingly. Source files to watch:
`llm_plan_generator.py`, `llm_workout_generator.py`, `llm_activity_analyzer.py`,
`llm_training_status_analyzer.py`, `llm_goal_guidance.py`, `llm_agent.py`.

> This is an offline decision-support tool — it is not wired into the app or CI.
