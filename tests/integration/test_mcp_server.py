"""The MCP endpoint at ``/mcp`` — protocol, transport and credentials (issue #42).

The tools themselves are exercised in ``test_mcp_tools.py``. This module is about
the door: does the handshake work, does a real MCP client get schemas it can use,
and — the part that matters most — does an endpoint the route-policy walk never
covered still refuse everything it should?

That last question is why these tests mint *real* personal access tokens instead
of overriding the identity dependency. This route is the one place in the API
where authorization is not decided by the route, so an override here would hide
exactly the bug worth catching.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from backend.app.mcp.limits import tool_limiter
from backend.app.models.registry_orm import PersonalAccessToken
from backend.app.services import personal_access_tokens as pat

_TEST_USER_ID = "test-user-00000000"

READ_SCOPES = ["activities:read", "athlete:read", "goals:read", "metrics:read", "plans:read"]


@pytest.fixture(autouse=True)
def _clean_rate_limiter():
    tool_limiter.reset()
    yield
    tool_limiter.reset()


@pytest.fixture
async def mcp_athlete(isolate_user_dbs):
    """An athlete in the caller's **real** per-user database.

    The tool-call tests deliberately do not override the per-user session: a
    request arriving over HTTP resolves its own identity and opens its own
    database, and that resolution is the part worth exercising end to end. The
    in-memory ``session`` fixture would replace exactly the step under test.
    """
    from backend.app.db.user_session import get_user_session_factory, init_user_db
    from backend.app.models.user_orm import Activity, Athlete

    await init_user_db(_TEST_USER_ID)
    async with get_user_session_factory(_TEST_USER_ID)() as s:
        athlete = Athlete(global_user_id=_TEST_USER_ID, ftp=250, ftp_tests=[])
        s.add(athlete)
        await s.flush()
        s.add(
            Activity(
                athlete_id=athlete.id,
                name="Evening spin",
                sport_type="Ride",
                start_time=datetime.now(timezone.utc),
                duration_s=3600,
                load=55.0,
                status="processed",
            )
        )
        await s.commit()
    return _TEST_USER_ID


@pytest.fixture
async def issue_token(registry_session):
    """Mint a real token against the in-memory registry."""

    async def _issue(scopes=READ_SCOPES, *, revoked=False, expired=False):
        token_id, raw, token_hash = pat.mint_token()
        now = datetime.now(timezone.utc)
        registry_session.add(
            PersonalAccessToken(
                id=token_id,
                user_id=_TEST_USER_ID,
                name="test token",
                token_hash=token_hash,
                scopes=json.dumps(scopes),
                expires_at=now - timedelta(days=1) if expired else now + timedelta(days=30),
                revoked_at=now if revoked else None,
            )
        )
        await registry_session.commit()
        return raw

    return _issue


async def rpc(client, method, params=None, *, token=None, request_id=1):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    body = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return await client.post("/mcp", json=body, headers=headers)


# ── Handshake ────────────────────────────────────────────────────────────────


async def test_initialize_advertises_tools_and_nothing_else(client):
    """Resources and prompts are absent rather than stubbed, so a client never
    offers a user something that will fail."""
    resp = await rpc(client, "initialize", {"protocolVersion": "2025-06-18"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["serverInfo"]["name"] == "openkoutsi"
    assert "get_training_status" in result["instructions"]


async def test_the_handshake_needs_no_credential(client):
    """A client has nowhere to put a token until it has connected."""
    assert (await rpc(client, "initialize")).status_code == 200
    assert (await rpc(client, "ping")).status_code == 200


async def test_the_initialized_notification_is_accepted_with_no_body(client):
    resp = await client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
    )
    assert resp.status_code == 202


async def test_an_unknown_method_is_a_json_rpc_error(client, issue_token):
    resp = await rpc(client, "resources/list", token=await issue_token())
    assert resp.status_code == 200
    error = resp.json()["error"]
    assert error["code"] == -32601
    assert "tools only" in error["message"]


async def test_batches_are_refused(client):
    """Removed from the protocol in the 2025-06-18 revision."""
    resp = await client.post(
        "/mcp", json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"}]
    )
    assert resp.json()["error"]["code"] == -32600
    assert "batching" in resp.json()["error"]["message"]


async def test_a_non_json_rpc_body_is_refused(client):
    resp = await client.post("/mcp", json={"hello": "world"})
    assert resp.json()["error"]["code"] == -32600


async def test_non_object_params_are_refused(client):
    resp = await client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping", "params": ["x"]}
    )
    assert resp.json()["error"]["code"] == -32602


async def test_malformed_json_is_refused(client):
    resp = await client.post(
        "/mcp", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert resp.json()["error"]["code"] == -32700


# ── Credentials ──────────────────────────────────────────────────────────────


async def test_listing_tools_needs_a_credential(client):
    resp = await rpc(client, "tools/list")
    assert resp.status_code == 401
    assert "Personal access tokens" in resp.json()["error"]["message"]


async def test_calling_a_tool_needs_a_credential(client):
    resp = await rpc(client, "tools/call", {"name": "get_training_status"})
    assert resp.status_code == 401


async def test_a_garbage_token_is_refused(client):
    resp = await rpc(client, "tools/list", token="okp_nope_nope")
    assert resp.status_code == 401


async def test_a_revoked_token_is_refused(client, issue_token):
    resp = await rpc(client, "tools/list", token=await issue_token(revoked=True))
    assert resp.status_code == 401


async def test_an_expired_token_is_refused(client, issue_token):
    resp = await rpc(client, "tools/list", token=await issue_token(expired=True))
    assert resp.status_code == 401


async def test_the_instance_kill_switch_closes_the_mcp_endpoint_too(
    client, issue_token, registry_session
):
    """``allow_personal_access_tokens`` refuses *authentication*. A surface that
    honoured it only at the HTTP routes would leave this door open."""
    from backend.app.models.registry_orm import InstanceSettings

    token = await issue_token()
    registry_session.add(InstanceSettings(allow_personal_access_tokens=False))
    await registry_session.commit()

    resp = await rpc(client, "tools/list", token=token)
    assert resp.status_code == 401


async def test_a_session_jwt_also_works(client, auth_headers, seeded_athlete):
    """Both consumers reach the same tools; a browser-issued token is one of
    them, and the internal agent will hold one of these."""
    token = auth_headers["Authorization"].removeprefix("Bearer ")
    resp = await rpc(client, "tools/list", token=token)
    assert resp.status_code == 200
    assert len(resp.json()["result"]["tools"]) == 9


# ── tools/list ───────────────────────────────────────────────────────────────


async def test_tools_list_publishes_usable_schemas(client, issue_token):
    resp = await rpc(client, "tools/list", token=await issue_token())
    tools = {t["name"]: t for t in resp.json()["result"]["tools"]}
    assert len(tools) == 9

    status = tools["get_training_status"]
    assert status["inputSchema"]["type"] == "object"
    assert status["inputSchema"]["additionalProperties"] is False
    assert status["outputSchema"]["properties"]["form_label"]["description"]
    assert status["annotations"]["readOnlyHint"] is True
    assert status["_meta"]["openkoutsi/scopes"] == ["athlete:read", "metrics:read"]


async def test_tools_the_credential_cannot_call_are_still_listed(client, issue_token):
    """Hiding them would make a scope refusal look like a missing feature. The
    scopes each needs travel in ``_meta``, so a client can explain the gap."""
    resp = await rpc(client, "tools/list", token=await issue_token(["goals:read"]))
    tools = resp.json()["result"]["tools"]
    assert len(tools) == 9
    assert any(t["name"] == "get_plan_status" for t in tools)


# ── tools/call ───────────────────────────────────────────────────────────────


async def test_calling_a_tool_returns_content_and_structured_content(
    client, issue_token, mcp_athlete
):
    resp = await rpc(
        client, "tools/call", {"name": "get_training_status", "arguments": {}},
        token=await issue_token(),
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    assert result["structuredContent"]["form_label"]
    # The text block is the structured content, so a client that reads only one
    # of the two still gets the whole answer.
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]


async def test_the_default_deny_the_route_walk_cannot_provide(
    client, issue_token, mcp_athlete
):
    """**The control this mount needs.**

    ``build_access_map`` never saw these tools, so nothing about the route policy
    protects them. A token holding one scope reaches exactly the tools that
    declared it, and is refused the rest — decided per tool, in the registry.
    """
    token = await issue_token(["goals:read"])

    allowed = await rpc(
        client, "tools/call", {"name": "get_goal_progress", "arguments": {}}, token=token
    )
    assert allowed.json()["result"]["isError"] is False

    for refused_tool in (
        "get_training_status",
        "get_plan_status",
        "list_recent_activities",
        "get_power_profile",
        "get_zone_totals",
    ):
        resp = await rpc(
            client, "tools/call", {"name": refused_tool, "arguments": {}}, token=token
        )
        result = resp.json()["result"]
        assert result["isError"] is True, refused_tool
        assert "missing" in result["content"][0]["text"]


async def test_a_scopeless_token_reaches_nothing(client, issue_token, mcp_athlete):
    token = await issue_token([])
    for name in ("get_goal_progress", "get_training_status", "find_activity"):
        resp = await rpc(client, "tools/call", {"name": name, "arguments": {}}, token=token)
        assert resp.json()["result"]["isError"] is True


async def test_a_tool_failure_is_a_result_not_a_transport_error(
    client, issue_token, mcp_athlete
):
    """An exception would abort the model's turn; a result lets it read the
    sentence and try the next thing."""
    resp = await rpc(
        client,
        "tools/call",
        {"name": "get_activity_detail", "arguments": {"activity_id": "nope"}},
        token=await issue_token(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "error" not in body
    assert body["result"]["isError"] is True
    assert "belongs to this athlete" in body["result"]["content"][0]["text"]
    assert "structuredContent" not in body["result"]


async def test_an_unknown_tool_is_a_result_naming_the_real_ones(
    client, issue_token, mcp_athlete
):
    resp = await rpc(
        client, "tools/call", {"name": "drop_everything", "arguments": {}},
        token=await issue_token(),
    )
    text = resp.json()["result"]["content"][0]["text"]
    assert "No tool named 'drop_everything'" in text
    assert "get_training_status" in text


async def test_a_call_without_a_name_is_an_invalid_params_error(client, issue_token):
    resp = await rpc(client, "tools/call", {"arguments": {}}, token=await issue_token())
    assert resp.json()["error"]["code"] == -32602


async def test_non_object_arguments_are_refused(client, issue_token):
    resp = await rpc(
        client, "tools/call", {"name": "get_goal_progress", "arguments": ["x"]},
        token=await issue_token(),
    )
    assert resp.json()["error"]["code"] == -32602


async def test_the_json_rpc_id_is_echoed_back(client, issue_token, mcp_athlete):
    resp = await rpc(
        client, "tools/call", {"name": "get_goal_progress", "arguments": {}},
        token=await issue_token(), request_id="abc-123",
    )
    assert resp.json()["id"] == "abc-123"


async def test_using_the_endpoint_marks_the_token_as_used(
    client, issue_token, registry_session, mcp_athlete
):
    """A token used only through MCP must not look dormant in the UI."""
    raw = await issue_token()
    token_id = raw.split("_")[1]

    before = await pat.load_by_id(registry_session, token_id)
    assert before.last_used_at is None

    await rpc(client, "tools/list", token=raw)

    await registry_session.refresh(before)
    assert before.last_used_at is not None
