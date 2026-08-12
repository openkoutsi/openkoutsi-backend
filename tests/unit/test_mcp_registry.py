"""The tool registry's own guarantees (issue #42).

The route-policy walk in ``tests/integration/test_pat_scopes.py`` never covers
``POST /mcp`` — that route resolves its own credential, so ``route_requires_auth``
is false for it and the walk never asks it to declare anything. None of what that
walk proves about HTTP routes therefore applies to the tools behind this one.
This module is the equivalent control for the tool layer: every
tool declares its scopes, those scopes are read-only and in the shared
vocabulary, every field is described with its unit, and nothing stream-shaped can
be returned.

Each of those is enforced by the decorator or the dispatcher too. The point of
asserting them again here is that the enforcement itself can be weakened by
accident, and these tests fail when it is.
"""
import json

import pytest
from pydantic import BaseModel, Field

from backend.app.core.scopes import SCOPES, SENSITIVE_SCOPES
from backend.app.mcp.dispatch import MAX_RESULT_BYTES
from backend.app.mcp.registry import (
    NAME_PATTERN,
    TOOL_SCOPES,
    ToolArgs,
    all_tools,
    get_tool,
    tool,
)

#: Every tool the issue specifies. Listed literally so dropping one is a test
#: failure rather than a silently smaller server.
EXPECTED_TOOLS = {
    "find_activity",
    "get_activity_detail",
    "get_athlete_profile",
    "get_goal_progress",
    "get_intensity_distribution",
    "get_plan_status",
    "get_power_profile",
    "get_training_status",
    "get_zone_totals",
    "list_recent_activities",
}


def test_the_expected_tools_are_registered():
    assert {t.name for t in all_tools()} == EXPECTED_TOOLS


def test_there_are_tools_to_check():
    """Guards the rest of this module against passing vacuously."""
    assert len(all_tools()) == 10


# ── Default deny ─────────────────────────────────────────────────────────────


def test_every_tool_declares_at_least_one_scope():
    """**The default-deny control.**

    No route policy covers these tools, so one that declared nothing would be
    reachable by any live credential. The decorator refuses to register such a
    tool; this asserts that none got in another way.
    """
    undeclared = [t.name for t in all_tools() if not t.scopes]
    assert undeclared == [], f"tools with no declared scope: {undeclared}"


def test_declaring_no_scope_is_impossible():
    """The check above only means something if it can fail."""

    class Args(ToolArgs):
        pass

    with pytest.raises(ValueError, match="declares no scopes"):

        @tool(name="silent_tool", title="x", scopes=set(), arguments=Args, returns=Args)
        async def _handler(run, args):
            """Doc."""


def test_a_tool_cannot_ask_for_a_scope_outside_the_vocabulary():
    class Args(ToolArgs):
        pass

    with pytest.raises(ValueError, match="not callable read scopes"):

        @tool(
            name="greedy_tool",
            title="x",
            scopes={"everything:always"},
            arguments=Args,
            returns=Args,
        )
        async def _handler(run, args):
            """Doc."""


def test_a_tool_cannot_ask_for_a_write_scope():
    """This iteration publishes no mutating tools, so a write grant would be
    one nothing could spend — and one a user might tick believing otherwise."""

    class Args(ToolArgs):
        pass

    with pytest.raises(ValueError, match="not callable read scopes"):

        @tool(
            name="writing_tool",
            title="x",
            scopes={"activities:write"},
            arguments=Args,
            returns=Args,
        )
        async def _handler(run, args):
            """Doc."""


def test_the_export_scope_is_not_callable():
    """``athlete:export`` returns the whole record in one call — the opposite of
    task-shaped, and a box the user ticks deliberately for a backup script."""
    assert SENSITIVE_SCOPES & TOOL_SCOPES == frozenset()
    assert "athlete:export" not in TOOL_SCOPES


def test_every_declared_scope_is_in_the_shared_vocabulary():
    """No tool may demand a scope the token endpoints cannot grant."""
    for t in all_tools():
        for scope in t.scopes:
            assert scope in SCOPES, f"{t.name}: {scope}"
            assert scope.endswith(":read"), f"{t.name}: {scope} is not a read scope"


def test_the_five_read_scopes_are_all_reachable():
    """No dead scopes: each of the five the issue names opens something."""
    declared = {scope for t in all_tools() for scope in t.scopes}
    assert declared == {
        "activities:read",
        "athlete:read",
        "goals:read",
        "metrics:read",
        "plans:read",
    }


def test_there_is_no_scope_named_after_the_transport():
    """An ``mcp:*`` scope would tell the person ticking it nothing about what it
    hands over, which is the whole job of a scope name."""
    assert not any(s.startswith("mcp") for s in SCOPES)


def test_a_session_credential_is_missing_nothing():
    """``scopes is None`` means full access, as it does everywhere else."""
    for t in all_tools():
        assert t.missing_scopes(None) == []


def test_a_token_must_hold_every_declared_scope():
    status = get_tool("get_training_status")
    assert set(status.scopes) == {"metrics:read", "athlete:read"}
    assert status.missing_scopes(["metrics:read"]) == ["athlete:read"]
    assert status.missing_scopes(["metrics:read", "athlete:read"]) == []
    assert status.missing_scopes([]) == ["athlete:read", "metrics:read"]


# ── Naming ───────────────────────────────────────────────────────────────────


def test_every_name_is_a_plain_snake_case_identifier():
    """Models type these back verbatim; anything else is a transcription risk."""
    for t in all_tools():
        assert NAME_PATTERN.match(t.name), t.name


@pytest.mark.parametrize("bad", ["Get_Status", "get status", "x", "get-status", "1tool"])
def test_a_malformed_name_is_refused(bad):
    class Args(ToolArgs):
        pass

    with pytest.raises(ValueError, match="snake_case"):

        @tool(name=bad, title="x", scopes={"metrics:read"}, arguments=Args, returns=Args)
        async def _handler(run, args):
            """Doc."""


def test_registering_the_same_name_twice_is_refused():
    class Args(ToolArgs):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @tool(
            name="get_training_status",
            title="x",
            scopes={"metrics:read"},
            arguments=Args,
            returns=Args,
        )
        async def _handler(run, args):
            """Doc."""


def test_a_tool_without_a_description_is_refused():
    """The description is interface: it is what a model reads to decide."""

    class Args(ToolArgs):
        pass

    with pytest.raises(ValueError, match="no docstring"):

        @tool(
            name="mystery_tool",
            title="x",
            scopes={"metrics:read"},
            arguments=Args,
            returns=Args,
        )
        async def _handler(run, args):
            pass


def test_arguments_must_be_strict_about_unknown_keys():
    """A model that misspells a filter must be told, not quietly served the
    unfiltered answer."""

    class Loose(BaseModel):
        pass

    with pytest.raises(ValueError, match="ToolArgs subclass"):

        @tool(
            name="loose_tool",
            title="x",
            scopes={"metrics:read"},
            arguments=Loose,
            returns=Loose,
        )
        async def _handler(run, args):
            """Doc."""


def test_every_published_input_schema_forbids_unknown_keys():
    """``additionalProperties: false`` is also what strict function-calling
    needs to constrain the model in the first place."""
    for t in all_tools():
        assert t.input_schema().get("additionalProperties") is False, t.name


def test_every_description_says_what_the_tool_is_for():
    """Short enough to read, long enough to choose between."""
    for t in all_tools():
        assert len(t.description) > 120, f"{t.name}'s description is too thin"
        assert t.title


# ── Schemas ──────────────────────────────────────────────────────────────────


def _properties(schema: dict):
    """Every ``(name, property_schema)`` in a JSON schema, definitions included."""
    for name, prop in (schema.get("properties") or {}).items():
        yield name, prop
    for definition in (schema.get("$defs") or {}).values():
        yield from _properties(definition)


def _description_of(prop: dict) -> str:
    """A field's description, looking through ``anyOf`` for optional fields."""
    if prop.get("description"):
        return prop["description"]
    for variant in prop.get("anyOf", []):
        if variant.get("description"):
            return variant["description"]
    return ""


#: Suffix → tokens any one of which counts as naming the unit. Longest suffix
#: wins, so ``coverage_pct`` is checked as a percentage rather than as a ``_t``.
UNIT_TOKENS = {
    "_bpm": ("bpm",),
    "_rpm": ("rpm",),
    "_pct": ("percent", "(%)"),
    "_kg": ("kg", "kilogram"),
    "_min": ("minute", "(min)"),
    "_days": ("day",),
    "_s": ("second", "(s)"),
    "_m": ("metre", "meter", "(m)"),
    "_w": ("watt", "(w)"),
    "_j": ("joule", "(j)"),
}

#: Fields whose whole name is the unit.
EXACT_UNITS = {
    "pct": ("percent", "(%)"),
    "seconds": ("second", "(s)"),
    "days": ("day",),
    "weeks": ("week",),
}


def test_every_field_of_every_schema_is_described():
    """An undescribed field is one a model has to guess the meaning of."""
    undescribed = []
    for t in all_tools():
        for schema, side in ((t.input_schema(), "in"), (t.output_schema(), "out")):
            for name, prop in _properties(schema):
                if not _description_of(prop):
                    undescribed.append(f"{t.name}.{side}.{name}")
    assert undescribed == []


def test_every_field_that_carries_a_unit_names_it():
    """``"duration": 7412`` is read as minutes about as often as seconds."""
    unlabelled = []
    checked = 0
    for t in all_tools():
        for schema, side in ((t.input_schema(), "in"), (t.output_schema(), "out")):
            for name, prop in _properties(schema):
                tokens = EXACT_UNITS.get(name)
                if tokens is None:
                    for suffix in sorted(UNIT_TOKENS, key=len, reverse=True):
                        if name.endswith(suffix):
                            tokens = UNIT_TOKENS[suffix]
                            break
                if tokens is None:
                    continue
                checked += 1
                described = _description_of(prop).lower()
                if not any(token in described for token in tokens):
                    unlabelled.append(f"{t.name}.{side}.{name}: {described!r}")
    assert unlabelled == [], "fields whose description does not name their unit"
    # The check is only worth having while it is actually reaching fields; a
    # rename that stopped matching every suffix would otherwise pass silently.
    assert checked > 50, f"only {checked} fields were unit-checked"


def test_no_tool_can_return_a_raw_data_stream():
    """The size rule, enforced statically.

    A three-hour ride holds ~11 000 samples per stream. An array of bare numbers
    in an output schema is what one looks like, so the shape is forbidden
    outright rather than left to a reviewer to notice.
    """
    offenders = []
    for t in all_tools():
        for name, prop in _properties(t.output_schema()):
            for variant in [prop, *prop.get("anyOf", [])]:
                items = variant.get("items") or {}
                if variant.get("type") == "array" and items.get("type") in (
                    "number",
                    "integer",
                ):
                    offenders.append(f"{t.name}.{name}")
    assert offenders == [], f"stream-shaped output fields: {offenders}"


def test_every_collection_reports_a_total_alongside_what_it_returned():
    """A truncated list that does not say so reads as a complete one."""
    for name in ("list_recent_activities", "find_activity", "get_goal_progress"):
        fields = set(get_tool(name).returns.model_fields)
        assert {"items", "returned", "total", "truncated"} <= fields, name


def test_every_collection_argument_is_bounded():
    """An unbounded ``limit`` is an unbounded response."""
    for t in all_tools():
        for field_name, field in t.arguments.model_fields.items():
            if field_name not in ("limit", "days", "weeks", "window_days", "week_window_days"):
                continue
            bounds = [
                m for m in field.metadata if hasattr(m, "le") or hasattr(m, "ge")
            ]
            assert bounds, f"{t.name}.{field_name} has no upper bound"


def test_the_published_descriptor_is_what_an_mcp_client_expects():
    for t in all_tools():
        described = t.describe()
        assert described["name"] == t.name
        assert described["inputSchema"]["type"] == "object"
        assert described["outputSchema"]["type"] == "object"
        assert described["annotations"]["readOnlyHint"] is True
        assert described["annotations"]["destructiveHint"] is False
        assert described["_meta"]["openkoutsi/scopes"] == sorted(t.scopes)
        # Serialisable, because it is about to be sent as JSON.
        json.dumps(described)


def test_the_response_bound_is_small_enough_to_matter():
    """64 KiB is a bound on a *context window*, not on a database row."""
    assert MAX_RESULT_BYTES <= 128 * 1024


def test_every_read_scope_opens_something_on_its_own():
    """No scope is callable only in company.

    ``athlete:read`` used to be one: both tools declaring it also demanded
    ``metrics:read``, so a token granted exactly the profile scope could call
    nothing at all and had no way to find that out except by trying. A scope a
    user can tick and spend on nothing is a worse lie than a missing one.
    """
    reachable = {
        scope
        for t in all_tools()
        if len(t.scopes) == 1
        for scope in t.scopes
    }
    assert reachable == {
        "activities:read",
        "athlete:read",
        "goals:read",
        "metrics:read",
        "plans:read",
    }


def test_the_coaching_styles_match_the_prompts_that_implement_them():
    """A style the profile tool reports has to be one something can honour.

    The vocabulary lives in ``athlete_experience`` and the prompt text lives
    with the prompts, so the two can drift — and the failure would be silent:
    a style reported to a model that no prompt implements, or a style the
    athlete set that the tool refuses to pass on.
    """
    from backend.app.services.athlete_experience import VALID_COACHING_STYLES
    from backend.app.services.llm_training_status_analyzer import (
        _COACHING_STYLE_PROMPTS,
    )

    assert set(_COACHING_STYLE_PROMPTS) == set(VALID_COACHING_STYLES)


def test_the_profile_tool_returns_no_identifying_fields():
    """The line between a profile tool and ``athlete:export``.

    Name, date of birth and avatar are on the same record and are exactly what
    a coaching model has no use for, so their absence is asserted on the schema
    rather than left to whoever next edits the handler.
    """
    schema = get_tool("get_athlete_profile").output_schema()
    everywhere = {name for name, _ in _properties(schema)}
    assert {"date_of_birth", "avatar_url", "avatar_path", "user_id"} & everywhere == set()
    # `name` does appear — but only inside the zone models, where it is the
    # zone's label. The profile itself must not carry one.
    assert "name" not in (schema.get("properties") or {})


def test_the_activity_labels_match_the_rest_of_the_platform():
    """A third label added to the API must not become unfilterable here."""
    from backend.app.api.activities import _VALID_LABELS
    from backend.app.mcp.tools.activities import VALID_LABELS

    assert set(VALID_LABELS) == _VALID_LABELS


def test_no_tool_module_reaches_the_registry_database():
    """Admin-data exclusion, enforced structurally.

    Users, invitations, instance settings and tokens live in the registry DB.
    No tool opens it, so there is no path by which an administrator's session
    could be served more than an ordinary athlete's.
    """
    import pkgutil

    import backend.app.mcp.tools as tools_pkg

    offenders = []
    for module in pkgutil.iter_modules(tools_pkg.__path__):
        source = (
            __import__(
                f"backend.app.mcp.tools.{module.name}", fromlist=["__file__"]
            ).__file__
        )
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        for forbidden in ("registry_orm", "get_registry_session", "is_admin", "roles"):
            if forbidden in text:
                offenders.append(f"{module.name}: {forbidden}")
    assert offenders == []


# ── Audit-line safety (review of #86) ────────────────────────────────────────


def test_caller_controlled_text_cannot_forge_an_audit_line():
    """`tool` is `params["name"]` from the request body and `token_id` is
    whatever sat between the underscores of an `okp_…` bearer. Unsanitised,
    either can carry a newline and forge a second, plausible record under the
    default formatter — which `audit.py` deliberately still supports."""
    from backend.app.core.audit import _safe

    forged = "x\nmcp ok tool=get_activity_detail caller=session user=someone-else 3.2ms"
    cleaned = _safe(forged)
    assert "\n" not in cleaned
    assert "\r" not in cleaned
    # Replaced rather than stripped, so the attempt stays visible in the record
    # instead of being tidied into something that reads as ordinary.
    assert "?" in cleaned


def test_an_overlong_field_is_truncated():
    from backend.app.core.audit import _MAX_FIELD, _safe

    assert len(_safe("a" * 5000)) == _MAX_FIELD + 1  # + the ellipsis


def test_an_absent_field_reads_as_absent():
    from backend.app.core.audit import _safe

    assert _safe(None) == "-"
    assert _safe("") == "-"


def test_the_structured_fields_are_left_intact(caplog):
    """A JSON formatter escapes them correctly, and truncating there would lose
    data an operator may need."""
    import logging

    from backend.app.core import audit

    long_name = "n" * 200
    with caplog.at_level(logging.INFO, logger="openkoutsi.audit"):
        audit.mcp_tool_call(tool=long_name, outcome="unknown_tool", user_id="u")

    record = next(r for r in caplog.records if getattr(r, "event", None) == "mcp_tool_call")
    assert record.mcp_tool == long_name
    assert len(record.getMessage()) < 200
