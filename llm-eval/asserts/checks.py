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
