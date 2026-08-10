"""Objective promptfoo assertions that reuse the backend's own validators.

The two JSON families are graded by running the model output through the exact
parsers the app uses to accept a response (``_parse_response`` /
``_parse_steps``): if production would reject it, the eval fails it. The two
prose families are graded on the format contract their prompts demand — a first
``MOOD:<enum>`` line followed by plain prose (no markdown). Subjective quality
is left to the web UI and the optional ``llm-rubric`` assert in the config.

Each function returns a promptfoo GradingResult dict ``{pass, score, reason}``.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

from fixtures.scenarios import PLAN_SCENARIOS  # noqa: E402


def _result(passed: bool, reason: str, score: float | None = None) -> dict:
    return {"pass": passed, "score": (1.0 if passed else 0.0) if score is None else score, "reason": reason}


def plan(output: str | dict, context: dict) -> dict:
    """Pass iff the output parses as a valid N-week plan (same contract as the app)."""
    import json as _json
    from backend.app.services.llm_plan_generator import _parse_response

    # Anthropic's json_schema response_format causes promptfoo to deserialize the
    # content into a dict before calling asserts; re-serialize so the backend parser
    # receives the string it expects.
    if isinstance(output, dict):
        output = _json.dumps(output)
    num_weeks = PLAN_SCENARIOS[context["vars"]["scenario"]]["num_weeks"]
    try:
        weeks = _parse_response(output, num_weeks)
    except Exception as exc:  # JSONDecodeError, KeyError, ValueError — as the app catches
        return _result(False, f"{type(exc).__name__}: {exc}")
    return _result(True, f"valid plan: {len(weeks)} weeks x 7 days")


def workout(output: str | dict, context: dict) -> dict:
    """Pass iff the output parses into valid workout steps (schema + nesting rule)."""
    import json as _json
    from backend.app.services.llm_workout_generator import WorkoutGenerationError, _parse_steps

    if isinstance(output, dict):
        output = _json.dumps(output)
    try:
        steps = _parse_steps(output)
    except WorkoutGenerationError as exc:
        return _result(False, str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return _result(False, f"{type(exc).__name__}: {exc}")
    return _result(True, f"valid workout: {len(steps)} top-level steps")


_MOOD_RE = re.compile(r"^MOOD:\s?(cheer|knowing|neutral|stern)\s*$")
_MARKDOWN_RE = re.compile(r"(?m)^(\s*#{1,6}\s|\s*[-*+]\s|\s*\d+\.\s)|```")


def mood_prose(output: str, context: dict) -> dict:
    """Pass iff first line is a valid MOOD tag and the body is plain prose.

    Encodes the format both prose prompts demand: ``MOOD:<mood>`` first line, a
    blank line, then paragraphs with no markdown headers/bullets/code fences.
    Language adherence for non-English locales is a subjective check — left to
    the optional ``llm-rubric`` rather than a brittle keyword heuristic.
    """
    lines = output.splitlines()
    if not lines or not _MOOD_RE.match(lines[0].strip()):
        head = (lines[0] if lines else "")[:60]
        return _result(False, f"first line is not a valid MOOD tag: {head!r}")

    problems: list[str] = []
    if len(lines) < 2 or lines[1].strip() != "":
        problems.append("MOOD line should be followed by a blank line")
    body = "\n".join(lines[2:])
    if _MARKDOWN_RE.search(body):
        problems.append("body contains markdown (headers, bullets, or code fences)")
    if not body.strip():
        problems.append("no feedback paragraphs after the MOOD line")

    if problems:
        # Format is close but imperfect — partial credit so the web UI still surfaces it.
        return _result(False, "; ".join(problems), score=0.5)
    return _result(True, f"valid MOOD ({lines[0].strip()}) + plain prose")


_REALISM_RE = re.compile(r"^REALISM:\s?(realistic|ambitious|unrealistic)\s*$")


def realism_prose(output: str, context: dict) -> dict:
    """Pass iff first line is a valid REALISM tag and the body is plain prose.

    The goal-guidance prompt demands ``REALISM:<verdict>`` on the first line, a
    blank line, then plain-prose paragraphs (no markdown). Modelled on
    ``mood_prose``; the verdict token stays English even for localized prose, so
    the check is language-agnostic. Coaching quality is left to the web UI or the
    optional ``llm-rubric``.
    """
    lines = output.splitlines()
    if not lines or not _REALISM_RE.match(lines[0].strip()):
        head = (lines[0] if lines else "")[:60]
        return _result(False, f"first line is not a valid REALISM tag: {head!r}")

    problems: list[str] = []
    if len(lines) < 2 or lines[1].strip() != "":
        problems.append("REALISM line should be followed by a blank line")
    body = "\n".join(lines[2:])
    if _MARKDOWN_RE.search(body):
        problems.append("body contains markdown (headers, bullets, or code fences)")
    if not body.strip():
        problems.append("no guidance paragraphs after the REALISM line")

    if problems:
        return _result(False, "; ".join(problems), score=0.5)
    return _result(True, f"valid REALISM ({lines[0].strip()}) + plain prose")


# ── Family 6: the agentic loop (issue #43) ───────────────────────────────────


def _tool_calls(output) -> list[dict]:
    """Every tool call in a turn's output, as ``{"name", "arguments"}``.

    promptfoo hands a tool-calling turn back in whichever shape the provider
    used — a list of call objects, a message wrapping them, or a JSON string of
    either — and the point of this family is to compare providers, so the
    extractor has to tolerate all of them rather than pin one. Anything with no
    recognisable call yields ``[]``, which the checks read as "called nothing".
    """
    import json as _json

    if isinstance(output, str):
        text = output.strip()
        if not text.startswith(("{", "[")):
            return []  # ordinary prose
        try:
            output = _json.loads(text)
        except ValueError:
            return []

    if isinstance(output, dict):
        output = output.get("tool_calls") or output.get("toolCalls") or []
    if not isinstance(output, list):
        return []

    calls: list[dict] = []
    for entry in output:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        raw = function.get("arguments")
        if isinstance(raw, str):
            try:
                arguments = _json.loads(raw or "{}")
            except ValueError:
                arguments = {"__unparseable__": raw}
        elif isinstance(raw, dict):
            arguments = raw
        else:
            arguments = {}
        calls.append({"name": name, "arguments": arguments})
    return calls


def tool_selection(output, context: dict) -> dict:
    """Pass iff the turn called tools, and called sensible ones.

    Three failures this separates, because they mean different things for a
    hoster choosing a model:

    * **called nothing** — the model accepted ``tools`` and ignored them. In
      production this run falls back to the blob prompt, so the model is usable
      but gains nothing from the agentic path.
    * **called something irrelevant** — worse: the run spends the tokens and
      arrives at the answer through the wrong door.
    * **called too many at once** — a shotgun rather than research; it burns the
      context window the later turns need.
    """
    from fixtures.scenarios import AGENTIC_SCENARIOS

    scenario = AGENTIC_SCENARIOS[context["vars"]["scenario"]]
    calls = _tool_calls(output)

    if not calls:
        return _result(False, "called no tool — this run would fall back to the blob prompt")

    names = [c["name"] for c in calls]
    allowed = scenario.get("allowed_tools")
    if allowed:
        unexpected = sorted({n for n in names if n not in allowed})
        if unexpected:
            return _result(
                False,
                f"called {unexpected} — not a sensible opening for this question "
                f"(expected one of {sorted(allowed)})",
                score=0.25,
            )

    max_calls = scenario.get("max_calls")
    if max_calls is not None and len(calls) > max_calls:
        return _result(
            False,
            f"called {len(calls)} tools at once ({names}); {max_calls} is the "
            "point past which this is a shotgun rather than research",
            score=0.5,
        )

    expected_arguments = scenario.get("expected_arguments")
    if expected_arguments:
        matched = any(
            all(c["arguments"].get(k) == v for k, v in expected_arguments.items())
            for c in calls
        )
        if not matched:
            return _result(
                False,
                f"no call carried the expected arguments {expected_arguments}; got "
                f"{[c['arguments'] for c in calls]}",
                score=0.5,
            )

    return _result(True, f"called {names}")


def tool_error_recovery(output, context: dict) -> dict:
    """Pass iff the turn adjusted after a tool error instead of repeating it.

    Issue #42 shapes a tool failure as a sentence naming what is nearby rather
    than a 404, precisely so the model can act on it. This asks whether it does.
    Answering outright counts as recovery — giving up on a lookup that cannot
    succeed is a legitimate move; retrying the identical call is not, and is the
    shape of the loop that spins until the round cap.
    """
    from fixtures.scenarios import AGENTIC_SCENARIOS

    scenario = AGENTIC_SCENARIOS[context["vars"]["scenario"]]
    banned_name, banned_arguments = scenario["must_not_repeat"]
    calls = _tool_calls(output)

    if not calls:
        body = output if isinstance(output, str) else ""
        if body.strip():
            return _result(True, "answered rather than retrying — an acceptable recovery")
        return _result(False, "returned neither a tool call nor an answer")

    for call in calls:
        if call["name"] == banned_name and all(
            call["arguments"].get(k) == v for k, v in banned_arguments.items()
        ):
            return _result(
                False,
                f"repeated the call that just failed ({banned_name} "
                f"{banned_arguments}) — the error message named the alternatives",
            )
    return _result(True, f"adjusted to {[c['name'] for c in calls]}")
