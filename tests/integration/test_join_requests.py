"""Integration tests for the self-serve join-request flow.

Covers public submission, validation, admin listing, approve (creates user +
membership), reject, role enforcement, and the admin notification.
"""
import json

from sqlalchemy import select

from backend.app.core.auth import create_access_token
from backend.app.models.registry_orm import TeamMembership, User

_SLUG = "test-team"
_TEST_TEAM_ID = "test-team-00000000"
_TEST_USER_ID = "test-user-00000000"
_BASE = f"/api/teams/{_SLUG}/join-requests"
_VALID = {
    "username": "newbie",
    "password": "Password1234",
    "display_name": "New Bie",
    "message": "let me in",
}


def _user_headers() -> dict:
    token = create_access_token(_TEST_USER_ID, _TEST_TEAM_ID, ["user"])
    return {"Authorization": f"Bearer {token}"}


async def test_submit_public(client):
    resp = await client.post(_BASE, json=_VALID)
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "pending"
    assert body["username"] == "newbie"
    assert body["team_slug"] == _SLUG


async def test_submit_notifies_admins(client, auth_headers):
    await client.post(_BASE, json=_VALID)
    msgs = (await client.get("/api/messages", headers=auth_headers)).json()
    assert any(
        m["type"] == "join_request" and m["data"]["username"] == "newbie"
        for m in msgs
    )


async def test_weak_password_rejected(client):
    resp = await client.post(_BASE, json={**_VALID, "password": "weak"})
    assert resp.status_code == 422


async def test_duplicate_pending_rejected(client):
    assert (await client.post(_BASE, json=_VALID)).status_code == 201
    resp = await client.post(_BASE, json=_VALID)
    assert resp.status_code == 400


async def test_existing_member_rejected(client):
    resp = await client.post(_BASE, json={**_VALID, "username": "test-user"})
    assert resp.status_code == 400


async def test_list_requires_admin(client):
    resp = await client.get(_BASE, headers=_user_headers())
    assert resp.status_code == 403


async def test_admin_list_and_approve(client, auth_headers, registry_session):
    req_id = (await client.post(_BASE, json=_VALID)).json()["id"]

    listing = await client.get(_BASE, headers=auth_headers)
    assert listing.status_code == 200
    assert len(listing.json()) == 1

    approved = await client.post(f"{_BASE}/{req_id}/approve", headers=auth_headers)
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    user = (
        await registry_session.execute(select(User).where(User.username == "newbie"))
    ).scalar_one()
    membership = (
        await registry_session.execute(
            select(TeamMembership).where(TeamMembership.user_id == user.id)
        )
    ).scalar_one()
    assert json.loads(membership.roles) == ["user"]

    # No longer pending.
    assert (await client.get(_BASE, headers=auth_headers)).json() == []


async def test_reject_then_cannot_approve(client, auth_headers):
    req_id = (await client.post(_BASE, json=_VALID)).json()["id"]

    rejected = await client.post(f"{_BASE}/{req_id}/reject", headers=auth_headers)
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"

    again = await client.post(f"{_BASE}/{req_id}/approve", headers=auth_headers)
    assert again.status_code == 400
