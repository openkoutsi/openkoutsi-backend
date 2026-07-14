"""Shared helpers for surfacing the athlete's self-reported experience level
in LLM prompts (issue #32).

The level is stored on ``athlete.app_settings["experience_level"]`` (added in
#18). This module is the single source of truth for the valid levels and for
turning a stored value into prompt text, so the athlete API (validation) and the
LLM prompt builders stay in sync.
"""
from __future__ import annotations

from typing import Optional

# Canonical self-reported experience levels (see #18). The athlete API imports
# this tuple for write-validation; the prompt builders use it to guard against
# stale/invalid stored values.
VALID_EXPERIENCE_LEVELS = (
    "novice",
    "intermediate",
    "experienced",
    "semi-pro",
    "elite",
)

# System-prompt hint describing how to use the experience level. Worded to fit
# both coaching prose (status/activity/goal) and structured JSON generation
# (plan/workout), since both are ultimately about the training content.
EXPERIENCE_GUIDANCE = (
    "When the athlete's self-reported experience level is given, tailor your "
    "response to it: novices need conservative loads, gentle progression, and a "
    "focus on fundamentals with more explanation of the reasoning; intermediate "
    "athletes can handle moderate progression and some technical detail; "
    "experienced, semi-pro and elite athletes can absorb higher intensity, finer "
    "nuance, sport-specific terminology, and less hand-holding. Never prescribe "
    "load or complexity beyond what the stated level can safely handle."
)


def experience_level(app_settings: Optional[dict]) -> Optional[str]:
    """Return the athlete's stored experience level, or ``None`` if unset/unknown.

    Defensive against missing settings and stale/invalid values: anything not in
    :data:`VALID_EXPERIENCE_LEVELS` is treated as absent.
    """
    if not isinstance(app_settings, dict):
        return None
    level = app_settings.get("experience_level")
    if level and str(level).strip() in VALID_EXPERIENCE_LEVELS:
        return str(level).strip()
    return None
