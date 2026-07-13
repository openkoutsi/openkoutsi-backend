"""Pydantic output schemas for the JSON families, as provider ``response_format``.

These schemas now live in the backend as the single source of truth
(:mod:`backend.app.services.llm_schemas`) so the runtime generators and this eval
harness use byte-identical schemas — no drift. This module re-exports them; see
the backend module for the full rationale (strict-subset handling, why the
recursive ``RepeatBlock`` is flattened, and the reuse of ``WorkoutStep`` /
``WorkoutCreate``).
"""
from __future__ import annotations

# The repo root is on sys.path by the time this is imported (entry points import
# ``_bootstrap`` first — see build.py / selftest.py), so ``backend.*`` resolves.
from backend.app.services.llm_schemas import (  # noqa: F401
    PlanOutput,
    PlanWeek,
    RepeatBlock,
    WorkoutOutput,
    _tighten,
    response_format,
)

__all__ = [
    "PlanOutput",
    "PlanWeek",
    "RepeatBlock",
    "WorkoutOutput",
    "_tighten",
    "response_format",
]
