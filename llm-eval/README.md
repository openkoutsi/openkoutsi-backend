# llm-eval — comparing LLM providers/models for openkoutsi

openkoutsi calls an LLM in four places, all through one OpenAI-compatible
`/chat/completions` path (`backend/app/services/llm_client.py:call_llm`). This
subproject sends prompts that **mirror what the platform actually sends** to a
matrix of models and grades the results, so hosters and BYOK users can pick a
model with evidence.

It's a thin [promptfoo](https://www.promptfoo.dev/) project. The substance is the
evaluation prompt set: instead of copying the prompts, we **import the real
backend builders**, so the text each model sees is byte-identical to production
and can never drift.

## The four families

| Family | Backend source (`backend/app/services/…`) | Output | How it's graded |
|---|---|---|---|
| `plan` | `llm_plan_generator.py` | JSON | **objective** — reuses `_parse_response` (N weeks, 7 days/week, valid type, null-on-rest) |
| `workout` | `llm_workout_generator.py` | JSON | **objective** — reuses `_parse_steps` (`WorkoutStepOrRepeat` schema + nesting rule) |

The two JSON families (`plan`, `workout`) pin `response_format` to
`{"type": "json_object"}` via the prompt function, so the provider is forced to
return a JSON object rather than prose or fenced text. The prose families
(`activity`, `status`) are left unconstrained.
| `activity` | `llm_activity_analyzer.py` | prose | **format objective** (`MOOD:` line, no markdown) + **subjective** (web UI / optional rubric) |
| `status` | `llm_training_status_analyzer.py` | prose | same as `activity`, plus plan-adherence reasoning |

## Layout

```
promptfooconfig.yaml   # providers × tests; per-family asserts. Model roster lives here.
prompts/build.py       # one prompt fn; dispatches on vars.family to the real backend builder
fixtures/scenarios.py  # in-memory ORM objects / PlanConfig per scenario (the eval inputs)
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
`llm_training_status_analyzer.py`.

> This is an offline decision-support tool — it is not wired into the app or CI.
