"""Daily active users: a dashboard load (``POST /api/metrics/catch-up``)
records the caller as active today, and admins read the counts back."""
from datetime import date, timedelta

import pytest
from fastapi import Request
from sqlalchemy import func, select

from backend.app.core.auth import PERSONAL_ACCESS_TOKEN, UserContext
from backend.app.core.deps import get_ctx_and_session
from backend.app.models.registry_orm import User, UserActiveDay
from backend.app.services.active_users import record_active_day, utc_today

_TEST_USER_ID = "test-user-00000000"


async def _rows(registry_session):
    return (
        await registry_session.execute(
            select(UserActiveDay.user_id, UserActiveDay.day)
        )
    ).all()


async def _add_user(registry_session, user_id: str) -> None:
    registry_session.add(User(id=user_id, username=user_id, password_hash="x"))
    await registry_session.commit()


class TestRecording:
    async def test_dashboard_load_records_today(self, client, auth_headers, registry_session):
        resp = await client.post("/api/metrics/catch-up", headers=auth_headers)
        assert resp.status_code == 200
        assert await _rows(registry_session) == [(_TEST_USER_ID, utc_today())]

    async def test_repeat_loads_count_once(self, client, auth_headers, registry_session):
        for _ in range(3):
            resp = await client.post("/api/metrics/catch-up", headers=auth_headers)
            assert resp.status_code == 200
        assert len(await _rows(registry_session)) == 1

    async def test_personal_access_token_is_not_counted(
        self, app, client, auth_headers, session, registry_session
    ):
        pat_ctx = UserContext(
            user_id=_TEST_USER_ID,
            roles=["user"],
            scopes=["metrics:read", "metrics:write"],
            token_kind=PERSONAL_ACCESS_TOKEN,
            token_id="pat-1",
        )

        async def _pat_ctx(request: Request):
            yield pat_ctx, session

        app.dependency_overrides[get_ctx_and_session] = _pat_ctx
        resp = await client.post("/api/metrics/catch-up", headers=auth_headers)
        assert resp.status_code == 200
        assert await _rows(registry_session) == []

    async def test_rows_go_with_the_account(self, registry_session):
        await _add_user(registry_session, "u-gone")
        await record_active_day("u-gone", registry_session)
        user = await registry_session.get(User, "u-gone")
        await registry_session.delete(user)
        await registry_session.commit()
        assert await _rows(registry_session) == []


class TestSummary:
    @pytest.fixture
    async def history(self, registry_session):
        # u1 active on 2 days of the same week, u2 on one of them, u3 next month.
        for uid in ("u1", "u2", "u3"):
            await _add_user(registry_session, uid)
        await record_active_day("u1", registry_session, date(2026, 10, 5))
        await record_active_day("u1", registry_session, date(2026, 10, 6))
        await record_active_day("u2", registry_session, date(2026, 10, 6))
        await record_active_day("u3", registry_session, date(2026, 11, 2))

    async def test_daily(self, client, auth_headers, history):
        resp = await client.get("/api/admin/active-users/summary", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["group_by"] == "day"
        assert data["total_active_users"] == 3
        assert data["buckets"] == [
            {"key": "2026-10-05", "active_users": 1},
            {"key": "2026-10-06", "active_users": 2},
            {"key": "2026-11-02", "active_users": 1},
        ]

    async def test_weekly_counts_distinct_users(self, client, auth_headers, history):
        resp = await client.get(
            "/api/admin/active-users/summary?group_by=week", headers=auth_headers
        )
        counts = [b["active_users"] for b in resp.json()["buckets"]]
        assert counts == [2, 1]

    async def test_monthly(self, client, auth_headers, history):
        resp = await client.get(
            "/api/admin/active-users/summary?group_by=month", headers=auth_headers
        )
        assert resp.json()["buckets"] == [
            {"key": "2026-10", "active_users": 2},
            {"key": "2026-11", "active_users": 1},
        ]

    async def test_range_is_inclusive(self, client, auth_headers, history):
        resp = await client.get(
            "/api/admin/active-users/summary?from=2026-10-06&to=2026-10-06",
            headers=auth_headers,
        )
        data = resp.json()
        assert data["from"] == "2026-10-06"
        assert data["total_active_users"] == 2
        assert data["buckets"] == [{"key": "2026-10-06", "active_users": 2}]

    async def test_bad_group_by(self, client, auth_headers):
        resp = await client.get(
            "/api/admin/active-users/summary?group_by=user", headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_bad_date(self, client, auth_headers):
        resp = await client.get(
            "/api/admin/active-users/summary?from=yesterday", headers=auth_headers
        )
        assert resp.status_code == 400
