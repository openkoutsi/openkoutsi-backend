"""Resolving the athlete's local timezone — one place, one set of rules.

``app_settings`` is a free-form JSON dict (``schemas/athlete.py``), so its
``timezone`` key holds whatever a client sent: a valid IANA name, a typo, or a
value that isn't a string at all. Every consumer needs the same answer to "what
is today for this athlete", and needs it to never raise — a bad setting must
degrade to UTC, not 500 a page.

Two consumers with different needs, which is why there are two functions:

- the LLM prompts want "now" as a datetime;
- the achievement rules (issue #33) want the ``ZoneInfo`` itself, to convert each
  activity's ``start_time`` into a local calendar date. Getting that wrong is
  visible in the streak numbers, not just in prompt text.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger(__name__)


def resolve_zone(tz_value: Any) -> Optional[ZoneInfo]:
    """The athlete's ``ZoneInfo``, or None when it can't be resolved.

    Accepts anything, because the stored value is unvalidated: a non-string
    (``{"timezone": 3}``) raises ``TypeError`` from ``ZoneInfo``, an unknown or
    malformed name raises ``ZoneInfoNotFoundError`` / ``ValueError``. All of them
    fall back to None (meaning UTC) and log at debug — silently shifting week
    boundaries is exactly the kind of thing that later arrives as a "my streak
    broke for no reason" report, so leave a trace.
    """
    if not tz_value:
        return None
    try:
        return ZoneInfo(tz_value)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        log.debug("Unusable app_settings timezone %r — falling back to UTC", tz_value)
        return None


def local_now(tz_value: Any) -> datetime:
    """Current time in the athlete's timezone, falling back to UTC."""
    return datetime.now(resolve_zone(tz_value) or timezone.utc)
