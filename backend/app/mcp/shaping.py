"""Response shaping for the tool layer (issue #42).

A tool result is read by a model with a finite context window, and the platform's
underlying data is not shaped for that at all: a three-hour ride carries eleven
thousand samples per stream, and handing over even one of those streams would
spend the entire budget on numbers no model can reason about anyway. So the rule
here is that **tools return computed aggregates, never raw series**, and the
helpers in this module are what makes that convenient enough to be the path of
least resistance.

Three conventions the tools follow throughout:

*Units live in the field name and in its description.* ``duration_s``,
``distance_m``, ``power_w`` — a model reading ``"duration": 7412`` has to guess,
and it guesses minutes about as often as seconds.

*Missing is said out loud, with a reason.* Where the platform stores a reason
code for an absent figure — aerobic decoupling is the elaborate case — the code
travels with the null instead of being flattened away. "No decoupling figure,
because the ride was under an hour" is a fact a coach can use; a bare ``null``
is one it has to speculate about.

*Collections are bounded and say so.* :func:`page` returns what fits alongside
the true ``total``, so a model can tell "that's all of them" from "there are
more" without being handed the rest to find out.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Sequence, TypeVar

T = TypeVar("T")


def page(items: Sequence[T], total: int) -> dict:
    """Build the shared pagination envelope around an already-sliced list."""
    returned = len(items)
    return {
        "items": list(items),
        "returned": returned,
        "total": total,
        "truncated": total > returned,
    }


def round_or_none(value: Optional[float], digits: int = 1) -> Optional[float]:
    """Round, preserving ``None`` — an absent measurement is not a zero."""
    return None if value is None else round(float(value), digits)


def int_or_none(value: Optional[float]) -> Optional[int]:
    return None if value is None else int(round(float(value)))


def hhmm(seconds: Optional[int]) -> Optional[str]:
    """``7412`` → ``"2 h 04"``. For prose a model is going to quote back."""
    if seconds is None:
        return None
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    if hours:
        return f"{hours} h {minutes:02d}"
    return f"{minutes} min"


def pct(part: float, whole: float, digits: int = 1) -> float:
    """A percentage that is 0.0 rather than an exception when nothing happened."""
    if not whole:
        return 0.0
    return round(100.0 * part / whole, digits)


def week_start(day: date) -> date:
    """The Monday of ``day``'s week — the platform's week boundary everywhere."""
    return day - timedelta(days=day.weekday())


def progress_pct(current: Optional[float], target: Optional[float]) -> Optional[float]:
    """How far along a goal is, or ``None`` when the question doesn't apply.

    A goal with no target value, or a target of zero, has no meaningful
    percentage; reporting ``0`` there would read as "no progress" rather than
    "not a measurable goal", which is a different thing entirely.
    """
    if current is None or not target:
        return None
    return round(100.0 * current / target, 1)
