"""Tests for the notifications service fan-out and mailbox routing."""
import json

from sqlalchemy import select

from backend.app.core.auth import hash_password
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.message_orm import Message
from backend.app.models.registry_orm import User
from backend.app.services import notifications

_TEST_USER_ID = "test-user-00000000"


async def _mailbox(user_id: str):
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as session:
        return (await session.execute(select(Message))).scalars().all()


async def test_notify_user_routes_to_mailbox():
    await notifications.notify_user(
        _TEST_USER_ID, notifications.INVITE_USED, {"username": "x"}
    )
    msgs = await _mailbox(_TEST_USER_ID)
    assert len(msgs) == 1
    assert msgs[0].type == "invite_used"
    assert msgs[0].data["username"] == "x"


async def test_notify_admins_only_targets_admins(registry_session):
    # Seed a non-admin user alongside the seeded admin (test-user).
    registry_session.add(
        User(
            id="plain-user",
            username="plain",
            password_hash=hash_password("Password1234"),
            roles=json.dumps(["user"]),
        )
    )
    await registry_session.commit()

    await notifications.notify_admins(
        registry_session, notifications.INVITE_USED, {"username": "x"}
    )

    assert len(await _mailbox(_TEST_USER_ID)) == 1
    assert await _mailbox("plain-user") == []
