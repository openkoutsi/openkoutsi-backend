"""Model-facing failures for the tool layer (issue #42).

A tool's caller is a language model, and a bare ``404`` teaches it nothing: it
will either give up or invent an answer. :class:`ToolError` carries prose meant
to be *read* — what was asked for, why it isn't there, and where to look instead::

    No activity on 2026-07-14. Nearest rides: 2026-07-13 (endurance, 2 h 04)
    and 2026-07-16 (threshold, 1 h 12).

These are returned as tool *results* with ``isError`` set, never raised out of
the transport. An exception would abort the model's turn; a result lets it read
the sentence and try the next thing — which is the whole point of a tool loop.

The one deliberate exception is :class:`ToolAccessError`, which reports a
missing scope. That is also returned as a result rather than raised, so the
model can say "your token can't see plans" instead of the client dying, but it
is a distinct type so the audit log can tell a refusal from a miss.
"""

from __future__ import annotations


class ToolError(Exception):
    """A tool could not answer, with a sentence explaining why.

    ``suggestions`` are optional follow-ups the model can act on immediately;
    they are appended to the message when it is rendered.
    """

    def __init__(self, message: str, *, suggestions: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.suggestions = suggestions or []

    def rendered(self) -> str:
        if not self.suggestions:
            return self.message
        return self.message + " " + " ".join(self.suggestions)


class ToolAccessError(ToolError):
    """The caller's credential does not carry the scopes this tool needs."""

    def __init__(self, tool_name: str, missing: list[str]) -> None:
        scopes = ", ".join(f"'{s}'" for s in sorted(missing))
        super().__init__(
            f"The credential in use cannot call '{tool_name}': it is missing the "
            f"{scopes} scope(s). This is a property of the token, not of the "
            f"data — a token's scopes are fixed when it is created and cannot be "
            f"widened, so a different tool or a new token is the only way "
            f"forward."
        )
        self.tool_name = tool_name
        self.missing = sorted(missing)


class ToolNotFound(ToolError):
    """No tool by that name. Names the ones that do exist."""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(
            f"No tool named '{name}'. Available tools: {', '.join(sorted(available))}."
        )
        self.name = name
