"""The tool registry and its default-deny guarantee (issue #42).

A tool is declared with the :func:`tool` decorator and is nothing but data plus a
handler::

    @tool(
        name="get_plan_status",
        title="Training plan status",
        scopes={"plans:read"},
        arguments=PlanStatusArgs,
        returns=PlanStatus,
    )
    async def get_plan_status(run: ToolRun, args: PlanStatusArgs) -> PlanStatus:
        ...

Why this needs its own default-deny check
-----------------------------------------
Personal access tokens (#46) are default-deny because
:func:`backend.app.core.scopes.build_access_map` walks ``app.routes`` and a route
that declared nothing is absent from the map. That walk has nothing to say about
the tools here, since ``POST /mcp`` resolves its own credential rather than
depending on ``get_current_user`` — there is no honest declaration to make when
the scope a call needs belongs to the tool named in the request body, not the
URL. (``test_pat_scopes.py::test_the_mcp_endpoint_is_outside_this_walk_by_design``
pins that shape.)

So the registry enforces the same property at *registration* rather than call
time: :func:`tool` raises unless the declaration names at least one scope, every
scope is in the shared vocabulary, and every one is a **read** scope. A tool that
forgot to declare cannot be registered, so it cannot exist to be called.
``tests/unit/test_mcp_registry.py`` asserts the same over whatever registered.

Scopes are an **AND**: the caller must hold all of them, which lets
``get_training_status`` report Fitness/Fatigue/Form *and* the FTP needed to
interpret them rather than serving profile data under a metrics grant. No
``mcp:*`` scope exists — a scope named after the transport would tell the person
ticking the box nothing about what it hands over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from backend.app.core.scopes import SCOPES, SENSITIVE_SCOPES

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from backend.app.mcp.dispatch import ToolRun

#: MCP tool names are identifiers a model types back verbatim, so they are held
#: to the narrow shape every client and every provider's function-calling schema
#: agrees on: lowercase, snake_case, starting with a letter.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,47}$")

#: The scopes a tool may ask for. Read-only by construction — this iteration
#: publishes no mutating tools, so a write scope here would be a grant nothing
#: could spend. ``athlete:export`` is excluded even though it reads: it returns
#: the entire record in one call, which is the opposite of task-shaped, and it is
#: a box the user ticks deliberately for a backup script rather than something a
#: coaching agent quietly inherits.
TOOL_SCOPES: frozenset[str] = frozenset(
    scope
    for scope in SCOPES
    if scope.endswith(":read") and scope not in SENSITIVE_SCOPES
)

Handler = Callable[["ToolRun", BaseModel], Awaitable[BaseModel]]


class ToolArgs(BaseModel):
    """Base for every tool's arguments. Unknown keys are an **error**.

    Pydantic ignores extra keys by default, which is the wrong default when the
    caller is a language model. A model that asks for ``sport="Ride"`` where the
    field is ``sport_type`` would otherwise get every sport back and report it
    confidently as a filtered answer — a wrong answer is worse than a refusal,
    because nothing downstream can tell it apart from a right one.

    It also makes the published schema strict (``additionalProperties: false``),
    which is what a provider's structured function-calling needs to constrain
    the model in the first place.
    """

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Tool:
    """One published capability.

    ``arguments`` and ``returns`` are pydantic models rather than hand-written
    JSON Schema, so the schema a client sees and the validation a call goes
    through can never drift apart, and so ``returns`` can be walked by tests
    (for units in every description, and for the absence of anything
    stream-shaped) instead of being trusted.
    """

    name: str
    title: str
    description: str
    scopes: frozenset[str]
    arguments: type[BaseModel]
    returns: type[BaseModel]
    handler: Handler
    #: Bumped when the *shape* of ``returns`` changes incompatibly. Reported in
    #: ``tools/list`` so a client that cached a schema can notice.
    version: int = 1
    annotations: dict[str, Any] = field(default_factory=dict)

    def input_schema(self) -> dict:
        return _clean_schema(self.arguments.model_json_schema())

    def output_schema(self) -> dict:
        return _clean_schema(self.returns.model_json_schema())

    def missing_scopes(self, held: list[str] | None) -> list[str]:
        """Which of this tool's scopes ``held`` lacks.

        ``held is None`` means a session credential — full access, the same
        meaning :class:`~backend.app.core.auth.UserContext` gives it — so
        nothing is missing.
        """
        if held is None:
            return []
        return sorted(self.scopes - set(held))

    def describe(self) -> dict:
        """The MCP ``tools/list`` entry for this tool."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema(),
            "outputSchema": self.output_schema(),
            "annotations": {
                # Every tool in this iteration reads; none writes, and none is
                # open-world (all data is the caller's own, already on this box).
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
                **self.annotations,
            },
            "_meta": {
                "openkoutsi/scopes": sorted(self.scopes),
                "openkoutsi/version": self.version,
            },
        }


def _clean_schema(schema: dict) -> dict:
    """Drop the pydantic bookkeeping MCP clients have no use for."""
    schema.pop("title", None)
    return schema


_REGISTRY: dict[str, Tool] = {}


def tool(
    *,
    name: str,
    title: str,
    scopes: set[str] | frozenset[str],
    arguments: type[BaseModel],
    returns: type[BaseModel],
    version: int = 1,
    annotations: dict[str, Any] | None = None,
):
    """Register a tool. Refuses anything that has not declared itself.

    The description is the handler's docstring: a tool's prose is read by a
    model deciding whether to call it, which makes it interface, not commentary,
    and keeping the two in one place stops them diverging.
    """

    def decorate(handler: Handler) -> Handler:
        if not NAME_PATTERN.match(name):
            raise ValueError(
                f"Tool name {name!r} must be lowercase snake_case, 3–48 chars, "
                "starting with a letter."
            )
        if name in _REGISTRY:
            raise ValueError(f"Tool {name!r} is already registered.")
        if not scopes:
            raise ValueError(
                f"Tool {name!r} declares no scopes. Every tool must say what a "
                "credential needs to hold to call it — there is no implicit "
                "grant here, because the route-policy walk that provides one "
                "for ordinary HTTP routes never covers the MCP endpoint: it "
                "resolves its own credential, since the scope a call needs "
                "depends on which tool was named in the request body."
            )
        unknown = sorted(set(scopes) - TOOL_SCOPES)
        if unknown:
            raise ValueError(
                f"Tool {name!r} declares scope(s) {unknown} that are not "
                f"callable read scopes. Allowed: {sorted(TOOL_SCOPES)}."
            )
        if not issubclass(arguments, ToolArgs):
            raise ValueError(
                f"Tool {name!r} must take a ToolArgs subclass, so an argument the "
                "model misspelled is refused rather than silently ignored."
            )
        doc = (handler.__doc__ or "").strip()
        if not doc:
            raise ValueError(
                f"Tool {name!r} has no docstring. The description is what a model "
                "reads to decide whether to call it."
            )
        _REGISTRY[name] = Tool(
            name=name,
            title=title,
            description=doc,
            scopes=frozenset(scopes),
            arguments=arguments,
            returns=returns,
            handler=handler,
            version=version,
            annotations=annotations or {},
        )
        return handler

    return decorate


def all_tools() -> list[Tool]:
    """Every registered tool, in a stable (alphabetical) order."""
    _load_tools()
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def get_tool(name: str) -> Tool | None:
    _load_tools()
    return _REGISTRY.get(name)


def tool_names() -> list[str]:
    _load_tools()
    return sorted(_REGISTRY)


def _load_tools() -> None:
    """Import the tool modules so the decorators have run.

    Done lazily rather than at package import to keep the import graph acyclic:
    the tools import :mod:`backend.app.mcp.dispatch` for ``ToolRun``, which
    imports this module.
    """
    if _REGISTRY:
        return
    import backend.app.mcp.tools  # noqa: F401  (import side effect is the point)
