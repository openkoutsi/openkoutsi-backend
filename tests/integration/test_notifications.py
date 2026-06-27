"""Tests for the notifications service fan-out and mailbox routing."""
import json

from sqlalchemy import select

from backend.app.core.auth import hash_password
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.message_orm import Message
from backend.app.models.registry_orm import TeamMembership, User
from backend.app.services import notifications

_TEST_TEAM_ID = "test-team-00000000"
_TEST_USER_ID = "test-user-00000000"


async def _mailbox(user_id: str):
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as session:
        return (await session.execute(select(Message))).scalars().all()


async def test_notify_superadmin_routes_to_reserved_mailbox():
    await notifications.notify_superadmin(
        notifications.TEAM_REQUEST, {"team_name": "Acme"}
    )
    msgs = await _mailbox(notifications.SUPERADMIN_MAILBOX)
    assert len(msgs) == 1
    assert msgs[0].type == "team_request"
    assert msgs[0].data["team_name"] == "Acme"


async def test_notify_team_admins_only_targets_admins(registry_session):
    # Seed a non-admin member alongside the seeded admin (test-user).
    registry_session.add(
        User(id="plain-user", username="plain", password_hash=hash_password("Password1234"))
    )
    await registry_session.flush()
    registry_session.add(
        TeamMembership(team_id=_TEST_TEAM_ID, user_id="plain-user", roles=json.dumps(["user"]))
    )
    await registry_session.commit()

    await notifications.notify_team_admins(
        registry_session, _TEST_TEAM_ID, notifications.INVITE_USED, {"username": "x"}
    )

    assert len(await _mailbox(_TEST_USER_ID)) == 1
    assert await _mailbox("plain-user") == []
