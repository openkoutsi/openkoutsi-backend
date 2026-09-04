"""Shared helpers for the athlete's self-reported profile settings (issue #32).

``app_settings`` is a free-form JSON dict — the athlete API validates
``experience_level`` on write but not ``coaching_style``, and nothing revalidates
either when the vocabulary changes. So a stale or bogus stored value must read as
*absent* rather than reaching a prompt or a tool result.

The single source of truth for those vocabularies:

- ``experience_level`` — stored on ``app_settings["experience_level"]`` (#18),
  imported by the athlete API for write-validation and by the prompt builders.
- ``coaching_style`` — stored on ``app_settings["coaching_style"]``, never
  validated on write. The prompt *text* for each style lives with the prompts
  (``llm_training_status_analyzer._COACHING_STYLE_PROMPTS``); the set of names
  lives here, and a test pins the two together.
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


# The tones the athlete can ask to be coached in. Unlike the experience level
# this is *not* validated when written, so anything reading it has to treat the
# vocabulary as the authority rather than the stored string.
VALID_COACHING_STYLES = (
    "stern",
    "friendly",
    "encouraging",
)


def coaching_style(app_settings: Optional[dict]) -> Optional[str]:
    """Return the athlete's stored coaching style, or ``None`` if unset/unknown.

    Same defensive shape as :func:`experience_level`: anything outside
    :data:`VALID_COACHING_STYLES` is treated as absent, because a style nothing
    recognises cannot be honoured and reporting it would invite a reader to try.
    """
    if not isinstance(app_settings, dict):
        return None
    style = app_settings.get("coaching_style")
    if style and str(style).strip() in VALID_COACHING_STYLES:
        return str(style).strip()
    return None
