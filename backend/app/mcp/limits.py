"""Per-user rate limiting for tool invocations (issue #42).

Separate from the HTTP limiter in :mod:`backend.app.core.limiter` because the
thing being counted is different: that one counts *requests*, and one MCP request
can carry one tool call or, over a long-lived agent turn, dozens. Counting
requests would let a client take the whole database a page at a time inside a
single POST.

Deliberately **not applied to the in-process agent**. Its calls are already
bounded by the LLM loop that issues them and by the entitlement gate in front of
that loop (issue #43); throttling them would only make the server slower at work
the user is already paying for, and a limit that fires mid-turn would leave the
model reasoning from a partial picture — worse than either finishing or not
starting.

A fixed window rather than a token bucket: the failure this defends against is a
script in a loop, which a window catches just as well, and the window's whole
state is one integer per active user. It inherits the pollers' single-process
assumption — a multi-worker deployment gets the limit per worker, which is the
same caveat the expiry sweeper already carries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

#: Calls per user per window. Generous on purpose: an agent answering one
#: question may legitimately make eight or ten calls, and the point is to stop a
#: runaway loop, not to ration ordinary use.
DEFAULT_LIMIT = 240
WINDOW_SECONDS = 60


@dataclass
class _Window:
    started_at: float
    count: int


class ToolRateLimiter:
    """Fixed-window per-user counter."""

    def __init__(self, limit: int = DEFAULT_LIMIT, window_s: int = WINDOW_SECONDS):
        self.limit = limit
        self.window_s = window_s
        self._windows: dict[str, _Window] = {}

    def check(self, user_id: str, *, now: float | None = None) -> tuple[bool, int]:
        """Record one call. Returns ``(allowed, retry_after_seconds)``.

        ``retry_after`` is 0 when allowed; when refused it is how long until the
        current window rolls, so the caller can say something more useful than
        "too many requests".
        """
        now = time.monotonic() if now is None else now
        window = self._windows.get(user_id)
        if window is None or now - window.started_at >= self.window_s:
            self._windows[user_id] = _Window(started_at=now, count=1)
            return True, 0
        if window.count >= self.limit:
            return False, max(1, int(self.window_s - (now - window.started_at)))
        window.count += 1
        return True, 0

    def reset(self, user_id: str | None = None) -> None:
        """Clear one user's window, or every window. Used by tests."""
        if user_id is None:
            self._windows.clear()
        else:
            self._windows.pop(user_id, None)


#: The process-wide limiter the dispatcher consults.
tool_limiter = ToolRateLimiter()
