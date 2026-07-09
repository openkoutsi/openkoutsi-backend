"""Integration tests for the LLM subscription gate + usage stats (issue #9)."""

import json

import pytest

from backend.app.core.auth import create_access_token, hash_password
from backend.app.models.registry_orm import User
from backend.app.services.llm_access import record_llm_usage
from backend.app.services.llm_client import ResolvedLlm

# The seeded admin/user id from conftest.
_TEST_USER_ID = "test-user-00000000"


async def _set_gate(client, auth_headers, on: bool):
    resp = await client.patch(
        "/api/admin/settings",
        json={"llm_requires_subscription": on},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["llm_requires_subscription"] is on


class TestAccessEndpoint:
    async def test_ungated_by_default(self, client, auth_headers):
        resp = await client.get("/api/llm/access", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["gated"] is False
        assert data["mode"] == "ungated"

    async def test_gated_no_entitlement_is_none(self, client, auth_headers):
        await _set_gate(client, auth_headers, True)
        resp = await client.get("/api/llm/access", headers=auth_headers)
        data = resp.json()
        assert data["gated"] is True
        assert data["mode"] == "none"
        assert data["entitlement"] is None

    async def test_gated_with_entitlement_is_entitled(self, client, auth_headers):
        await _set_gate(client, auth_headers, True)
        grant = await client.put(
            f"/api/admin/users/{_TEST_USER_ID}/llm-entitlement",
            json={"status": "active"},
            headers=auth_headers,
        )
        assert grant.status_code == 200
        resp = await client.get("/api/llm/access", headers=auth_headers)
        data = resp.json()
        assert data["mode"] == "entitled"
        assert data["entitlement"]["status"] == "active"

    async def test_gated_with_byok_is_byok(self, client, auth_headers, session, seeded_athlete):
        await _set_gate(client, auth_headers, True)
        seeded_athlete.app_settings = {"llm_base_url": "http://my-ollama:11434/v1"}
        await session.commit()
        resp = await client.get("/api/llm/access", headers=auth_headers)
        assert resp.json()["mode"] == "byok"


class TestChatGate:
    async def test_denied_chat_returns_structured_403(self, client, auth_headers):
        await _set_gate(client, auth_headers, True)
        resp = await client.post(
            "/api/llm/chat",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=auth_headers,
        )
        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert detail["code"] == "llm_subscription_required"
        assert "message" in detail

    async def test_gating_off_regression(self, client, auth_headers):
        # With the gate off, a non-entitled user is not blocked by the gate; the
        # request proceeds to config resolution (400: no LLM configured here).
        resp = await client.post(
            "/api/llm/chat",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=auth_headers,
        )
        assert resp.status_code != 403


class TestPlansGate:
    async def test_denied_plan_generate_returns_403(self, client, auth_headers):
        await _set_gate(client, auth_headers, True)
        resp = await client.post(
            "/api/plans",
            json={
                "name": "P",
                "start_date": "2026-01-05",
                "weeks": 4,
                "use_llm": True,
                "config": {
                    "periodization": "linear",
                    "intensity_preference": "balanced",
                    "days_per_week": 3,
                    "day_configs": [
                        {"day_of_week": 2, "workout_type": "threshold"},
                    ],
                },
            },
            headers=auth_headers,
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "llm_subscription_required"


class TestAnalyzeGate:
    async def test_denied_training_status_trigger_403(self, client, auth_headers):
        await _set_gate(client, auth_headers, True)
        resp = await client.post("/api/athlete/training-status", headers=auth_headers)
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "llm_subscription_required"


class TestAdminEntitlementCrud:
    async def test_grant_then_revoke(self, client, auth_headers):
        grant = await client.put(
            f"/api/admin/users/{_TEST_USER_ID}/llm-entitlement",
            json={"status": "active", "notes": "vip"},
            headers=auth_headers,
        )
        assert grant.status_code == 200
        ent = grant.json()["llm_entitlement"]
        assert ent["active"] is True
        assert ent["source"] == "manual"
        assert ent["notes"] == "vip"

        revoke = await client.put(
            f"/api/admin/users/{_TEST_USER_ID}/llm-entitlement",
            json={"status": "revoked"},
            headers=auth_headers,
        )
        assert revoke.status_code == 200
        assert revoke.json()["llm_entitlement"]["active"] is False

    async def test_user_list_includes_entitlement(self, client, auth_headers):
        await client.put(
            f"/api/admin/users/{_TEST_USER_ID}/llm-entitlement",
            json={"status": "active"},
            headers=auth_headers,
        )
        resp = await client.get("/api/admin/users", headers=auth_headers)
        assert resp.status_code == 200
        me = next(u for u in resp.json()["items"] if u["id"] == _TEST_USER_ID)
        assert me["llm_entitlement"]["active"] is True

    async def test_grant_unknown_user_404(self, client, auth_headers):
        resp = await client.put(
            "/api/admin/users/does-not-exist/llm-entitlement",
            json={"status": "active"},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_non_admin_forbidden(self, client, auth_headers, registry_session):
        plain = User(
            id="plain-user-1",
            username="plain",
            password_hash=hash_password("Testpass1234"),
            roles=json.dumps(["user"]),
        )
        registry_session.add(plain)
        await registry_session.commit()
        token = create_access_token("plain-user-1", ["user"])
        resp = await client.put(
            f"/api/admin/users/{_TEST_USER_ID}/llm-entitlement",
            json={"status": "active"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403


class TestUsageRecording:
    async def test_instance_call_recorded(self, client, auth_headers, usage_db):
        cfg = ResolvedLlm(
            base_url="https://api.openai.com/v1", model="gpt-4o", api_key="k",
            source="instance", key_source="instance",
        )
        await record_llm_usage(
            user_id=_TEST_USER_ID, feature="chat", cfg=cfg,
            usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )
        resp = await client.get(
            "/api/admin/llm-usage/summary?group_by=provider", headers=auth_headers
        )
        assert resp.status_code == 200
        buckets = resp.json()["buckets"]
        assert len(buckets) == 1
        b = buckets[0]
        assert b["key"] == "api.openai.com"
        assert b["calls"] == 1
        assert b["prompt_tokens"] == 10
        assert b["completion_tokens"] == 4
        assert b["unknown_usage_calls"] == 0

    async def test_byok_call_not_recorded(self, client, auth_headers, usage_db):
        cfg = ResolvedLlm(
            base_url="http://my-ollama:11434/v1", model="llama3", api_key=None,
            source="user", key_source="user",
        )
        await record_llm_usage(
            user_id=_TEST_USER_ID, feature="chat", cfg=cfg,
            usage={"prompt_tokens": 99, "completion_tokens": 99, "total_tokens": 198},
        )
        resp = await client.get(
            "/api/admin/llm-usage/summary?group_by=feature", headers=auth_headers
        )
        assert resp.json()["buckets"] == []

    async def test_missing_usage_recorded_as_unknown(self, client, auth_headers, usage_db):
        cfg = ResolvedLlm(
            base_url="http://localhost:11434/v1", model="llama3", api_key=None,
            source="instance", key_source="none",
        )
        await record_llm_usage(user_id=_TEST_USER_ID, feature="activity_analysis", cfg=cfg, usage=None)
        resp = await client.get(
            "/api/admin/llm-usage/summary?group_by=feature", headers=auth_headers
        )
        b = resp.json()["buckets"][0]
        assert b["key"] == "activity_analysis"
        assert b["calls"] == 1
        assert b["unknown_usage_calls"] == 1
        assert b["prompt_tokens"] == 0

    async def test_summary_month_bucket_per_user(self, client, auth_headers, usage_db):
        cfg = ResolvedLlm(
            base_url="https://api.openai.com/v1", model="gpt-4o", api_key="k",
            source="instance", key_source="instance",
        )
        for _ in range(3):
            await record_llm_usage(
                user_id=_TEST_USER_ID, feature="chat", cfg=cfg,
                usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            )
        resp = await client.get(
            f"/api/admin/llm-usage/summary?group_by=month&user_id={_TEST_USER_ID}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        buckets = resp.json()["buckets"]
        assert len(buckets) == 1
        assert buckets[0]["calls"] == 3
        assert buckets[0]["prompt_tokens"] == 15
        assert buckets[0]["completion_tokens"] == 15

    async def test_summary_bad_group_by(self, client, auth_headers, usage_db):
        resp = await client.get(
            "/api/admin/llm-usage/summary?group_by=bogus", headers=auth_headers
        )
        assert resp.status_code == 400
