"""Integration tests for the user-scoped message inbox (/api/messages).

Covers listing, unread count, mark-read, mark-all-read, hard delete, auth, and
per-user isolation. Messages are seeded via the notifications service (the only
writer) so the tests exercise the real per-user DB machinery.
"""
from backend.app.services import notifications

_TEST_USER_ID = "test-user-00000000"
_PREFIX = "/api/messages"


async def _seed(user_id: str, n: int = 1) -> None:
    for i in range(n):
        await notifications.notify_user(
            user_id, notifications.INVITE_USED, {"username": f"u{i}"}
        )


async def test_requires_auth(client):
    resp = await client.get(_PREFIX)
    assert resp.status_code == 401


async def test_empty_inbox(client, auth_headers):
    resp = await client.get(_PREFIX, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_and_unread_count(client, auth_headers):
    await _seed(_TEST_USER_ID, 2)
    msgs = (await client.get(_PREFIX, headers=auth_headers)).json()
    assert len(msgs) == 2
    assert all(m["read_at"] is None for m in msgs)

    cnt = await client.get(f"{_PREFIX}/unread-count", headers=auth_headers)
    assert cnt.json()["count"] == 2


async def test_mark_read(client, auth_headers):
    await _seed(_TEST_USER_ID, 1)
    msg_id = (await client.get(_PREFIX, headers=auth_headers)).json()[0]["id"]

    resp = await client.post(f"{_PREFIX}/{msg_id}/read", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["read_at"] is not None

    cnt = await client.get(f"{_PREFIX}/unread-count", headers=auth_headers)
    assert cnt.json()["count"] == 0


async def test_mark_all_read(client, auth_headers):
    await _seed(_TEST_USER_ID, 3)
    resp = await client.post(f"{_PREFIX}/read-all", headers=auth_headers)
    assert resp.status_code == 204

    cnt = await client.get(f"{_PREFIX}/unread-count", headers=auth_headers)
    assert cnt.json()["count"] == 0


async def test_delete_is_hard_delete(client, auth_headers):
    await _seed(_TEST_USER_ID, 1)
    msg_id = (await client.get(_PREFIX, headers=auth_headers)).json()[0]["id"]

    resp = await client.delete(f"{_PREFIX}/{msg_id}", headers=auth_headers)
    assert resp.status_code == 204
    assert (await client.get(_PREFIX, headers=auth_headers)).json() == []

    # Really gone — a second delete 404s.
    again = await client.delete(f"{_PREFIX}/{msg_id}", headers=auth_headers)
    assert again.status_code == 404


async def test_per_user_isolation(client, auth_headers):
    await _seed("some-other-user", 2)
    await _seed(_TEST_USER_ID, 1)
    msgs = (await client.get(_PREFIX, headers=auth_headers)).json()
    assert len(msgs) == 1
