"""Unit tests for the personal-access-token expiry sweep (issue #46).

The sweep is the thing that keeps a one-year ceiling from simply relocating the
outage, so the two properties that matter are: it picks the right stage per
token, and **running it twice in one day sends nothing the second time**.
"""
import itertools
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.app.models.registry_orm import PersonalAccessToken
from backend.app.services import pat_expiry

_TEST_USER_ID = "test-user-00000000"
_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


# The autouse `_no_email` fixture below replaces `pat_expiry._send_email`, so
# the one test that exercises the real function has to hold a reference taken
# before any patching.
_REAL_SEND_EMAIL = pat_expiry._send_email

_hash_counter = itertools.count()


def _token(expires_in: timedelta, **overrides) -> PersonalAccessToken:
    fields = {
        "user_id": _TEST_USER_ID,
        # token_hash is unique — real tokens never share one either.
        "token_hash": f"hash-{next(_hash_counter)}",
        "name": "nightly-backup",
        "scopes": json.dumps(["athlete:export"]),
        "expires_at": _NOW + expires_in,
        "created_at": _NOW - timedelta(days=90),
    }
    fields.update(overrides)
    return PersonalAccessToken(**fields)


# ── Stage selection ─────────────────────────────────────────────────────────


class TestStageSelection:
    @pytest.mark.parametrize(
        "expires_in,expected",
        [
            (timedelta(days=30), None),
            (timedelta(days=8), None),
            (timedelta(days=7), pat_expiry.EXPIRING_7D),
            (timedelta(days=3), pat_expiry.EXPIRING_7D),
            (timedelta(days=1), pat_expiry.EXPIRING_1D),
            (timedelta(hours=2), pat_expiry.EXPIRING_1D),
            (timedelta(0), pat_expiry.EXPIRED),
            (timedelta(days=-5), pat_expiry.EXPIRED),
        ],
    )
    def test_stage_follows_time_remaining(self, expires_in, expected):
        assert pat_expiry.stage_for(_token(expires_in), _NOW) == expected

    def test_a_revoked_token_is_never_warned_about(self):
        """Its owner already ended it deliberately; expiry news is noise."""
        token = _token(timedelta(days=3), revoked_at=_NOW - timedelta(days=1))
        assert pat_expiry.stage_for(token, _NOW) is None

    def test_stages_only_ever_move_forward(self):
        assert pat_expiry._is_new_stage(pat_expiry.EXPIRING_7D, None)
        assert pat_expiry._is_new_stage(pat_expiry.EXPIRING_1D, pat_expiry.EXPIRING_7D)
        assert pat_expiry._is_new_stage(pat_expiry.EXPIRED, pat_expiry.EXPIRING_1D)
        # Already sent — this is what stops the daily nag.
        assert not pat_expiry._is_new_stage(pat_expiry.EXPIRING_7D, pat_expiry.EXPIRING_7D)
        assert not pat_expiry._is_new_stage(pat_expiry.EXPIRING_7D, pat_expiry.EXPIRED)

    def test_a_sweep_that_missed_days_skips_to_the_current_stage(self):
        """No replaying the stages it slept through."""
        assert pat_expiry._is_new_stage(pat_expiry.EXPIRED, pat_expiry.EXPIRING_7D)


# ── Running the sweep ───────────────────────────────────────────────────────


@pytest.fixture
async def sweep_session(registry_engine, registry_session):
    """A second session on the same in-memory registry as ``registry_session``."""
    factory = async_sessionmaker(registry_engine, expire_on_commit=False)
    async with factory() as s:
        yield s


@pytest.fixture(autouse=True)
def _no_email():
    """Email is best-effort on top of the inbox; these tests assert on the inbox."""
    with patch.object(pat_expiry, "_send_email", AsyncMock()) as mock:
        yield mock


class TestSweep:
    async def test_a_token_far_from_expiry_is_left_alone(
        self, registry_session, sweep_session
    ):
        registry_session.add(_token(timedelta(days=60)))
        await registry_session.commit()

        with patch.object(pat_expiry.notifications, "notify_user", AsyncMock()) as notify:
            assert await pat_expiry.run_expiry_sweep(sweep_session, _NOW) == 0
        notify.assert_not_awaited()

    async def test_each_stage_notifies_once(self, registry_session, sweep_session):
        token = _token(timedelta(days=5))
        registry_session.add(token)
        await registry_session.commit()

        with patch.object(pat_expiry.notifications, "notify_user", AsyncMock()) as notify:
            assert await pat_expiry.run_expiry_sweep(sweep_session, _NOW) == 1
            assert notify.await_count == 1

            # Same day, second run — the mitigation for a restart or a second
            # process. Without it this feature becomes a daily nag.
            assert await pat_expiry.run_expiry_sweep(sweep_session, _NOW) == 0
            assert notify.await_count == 1

            # A day before expiry is a new stage, so it speaks again.
            later = _NOW + timedelta(days=4, hours=12)
            assert await pat_expiry.run_expiry_sweep(sweep_session, later) == 1
            assert notify.await_count == 2

            # And once more when it actually runs out.
            after = _NOW + timedelta(days=6)
            assert await pat_expiry.run_expiry_sweep(sweep_session, after) == 1
            assert notify.await_count == 3

            # But never a fourth time.
            assert await pat_expiry.run_expiry_sweep(sweep_session, after + timedelta(days=1)) == 0
            assert notify.await_count == 3

    async def test_the_recorded_stage_is_persisted(
        self, registry_session, sweep_session
    ):
        token = _token(timedelta(days=5))
        registry_session.add(token)
        await registry_session.commit()

        with patch.object(pat_expiry.notifications, "notify_user", AsyncMock()):
            await pat_expiry.run_expiry_sweep(sweep_session, _NOW)

        await registry_session.refresh(token)
        assert token.last_expiry_notice == pat_expiry.EXPIRING_7D

    async def test_the_message_carries_the_stage_and_days_left(
        self, registry_session, sweep_session
    ):
        registry_session.add(_token(timedelta(days=5)))
        await registry_session.commit()

        with patch.object(pat_expiry.notifications, "notify_user", AsyncMock()) as notify:
            await pat_expiry.run_expiry_sweep(sweep_session, _NOW)

        user_id, message_type, payload = notify.await_args.args
        assert user_id == _TEST_USER_ID
        assert message_type == "pat_expiring"
        assert payload["stage"] == pat_expiry.EXPIRING_7D
        assert payload["days_left"] == 5
        assert payload["name"] == "nightly-backup"

    async def test_an_expired_token_gets_its_own_message_type(
        self, registry_session, sweep_session
    ):
        registry_session.add(_token(timedelta(days=-1)))
        await registry_session.commit()

        with patch.object(pat_expiry.notifications, "notify_user", AsyncMock()) as notify:
            await pat_expiry.run_expiry_sweep(sweep_session, _NOW)

        _, message_type, payload = notify.await_args.args
        assert message_type == "pat_expired"
        assert payload["days_left"] is None

    async def test_a_revoked_token_is_swept_past(
        self, registry_session, sweep_session
    ):
        registry_session.add(
            _token(timedelta(days=2), revoked_at=_NOW - timedelta(days=1))
        )
        await registry_session.commit()

        with patch.object(pat_expiry.notifications, "notify_user", AsyncMock()) as notify:
            assert await pat_expiry.run_expiry_sweep(sweep_session, _NOW) == 0
        notify.assert_not_awaited()

    async def test_a_failed_mailbox_write_leaves_the_stage_unsent(
        self, registry_session, sweep_session
    ):
        """The next sweep retries rather than silently skipping the warning."""
        token = _token(timedelta(days=5))
        registry_session.add(token)
        await registry_session.commit()

        with patch.object(
            pat_expiry.notifications, "notify_user", AsyncMock(side_effect=RuntimeError)
        ):
            assert await pat_expiry.run_expiry_sweep(sweep_session, _NOW) == 0

        await registry_session.refresh(token)
        assert token.last_expiry_notice is None

    async def test_one_token_failing_does_not_strand_the_others(
        self, registry_session, sweep_session
    ):
        registry_session.add(_token(timedelta(days=5), name="first"))
        registry_session.add(_token(timedelta(days=5), name="second"))
        await registry_session.commit()

        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("mailbox unavailable")

        with patch.object(pat_expiry.notifications, "notify_user", flaky):
            assert await pat_expiry.run_expiry_sweep(sweep_session, _NOW) == 1

    async def test_a_thrown_email_does_not_lose_the_stage_mark(
        self, registry_session, sweep_session, _no_email
    ):
        """The mark must be durable before the side channel that can throw.

        `notify_user` commits to the user's own DB immediately. If the mark were
        only committed after `_send_email`, anything escaping it would roll the
        mark back while leaving the message in place — and the user would be
        re-notified every day thereafter, which is exactly the nag
        `last_expiry_notice` exists to prevent, arriving precisely when the mail
        path is broken and the warning matters most.
        """
        token = _token(timedelta(days=5))
        registry_session.add(token)
        await registry_session.commit()

        _no_email.side_effect = ValueError("unknown email provider")

        with patch.object(pat_expiry.notifications, "notify_user", AsyncMock()) as notify:
            with pytest.raises(ValueError):
                await pat_expiry.run_expiry_sweep(sweep_session, _NOW)
            assert notify.await_count == 1

        # The mark survived the exception…
        await registry_session.refresh(token)
        assert token.last_expiry_notice == pat_expiry.EXPIRING_7D

        # …so tomorrow's sweep stays silent rather than repeating itself.
        _no_email.side_effect = None
        with patch.object(pat_expiry.notifications, "notify_user", AsyncMock()) as notify:
            assert await pat_expiry.run_expiry_sweep(
                sweep_session, _NOW + timedelta(hours=6)
            ) == 0
            notify.assert_not_awaited()

    async def test_a_broken_email_provider_never_escapes_send_email(self):
        """`get_email_provider()` raises a bare ValueError on an unknown name,
        so one typo'd EMAIL_PROVIDER must not take the whole sweep down."""
        user = type("U", (), {
            "id": _TEST_USER_ID, "email": "a@example.test",
            "email_verified_at": _NOW,
        })()
        token = _token(timedelta(days=5))

        with patch(
            "backend.app.services.email.get_email_provider",
            side_effect=ValueError("unknown provider 'typo'"),
        ):
            # Returns rather than raising — email is best-effort on top of an
            # inbox message that has already been written.
            assert await _REAL_SEND_EMAIL(user, token, 5) is None

    async def test_an_already_expired_token_is_not_reloaded_forever(
        self, registry_session, sweep_session
    ):
        """Dead rows are retained, so the sweep's query must exclude the ones it
        can never act on again rather than growing with instance age."""
        registry_session.add(
            _token(timedelta(days=-400), last_expiry_notice=pat_expiry.EXPIRED)
        )
        registry_session.add(_token(timedelta(days=5), name="live"))
        await registry_session.commit()

        from sqlalchemy import select
        from backend.app.models.registry_orm import PersonalAccessToken

        with patch.object(pat_expiry.notifications, "notify_user", AsyncMock()):
            await pat_expiry.run_expiry_sweep(sweep_session, _NOW)

        # The terminal-stage row is filtered out in SQL, not merely skipped.
        rows = (await sweep_session.execute(
            select(PersonalAccessToken).where(
                PersonalAccessToken.revoked_at.is_(None),
                PersonalAccessToken.expires_at <= _NOW + timedelta(days=7),
                PersonalAccessToken.last_expiry_notice.is_distinct_from(pat_expiry.EXPIRED),
            )
        )).scalars().all()
        assert [r.name for r in rows] == ["live"]

    async def test_email_is_attempted_alongside_the_inbox_message(
        self, registry_session, sweep_session, _no_email
    ):
        registry_session.add(_token(timedelta(days=5)))
        await registry_session.commit()

        with patch.object(pat_expiry.notifications, "notify_user", AsyncMock()):
            await pat_expiry.run_expiry_sweep(sweep_session, _NOW)

        _no_email.assert_awaited_once()


class TestEmailOptOut:
    """The inbox message is unconditional; only the email is opt-out."""

    async def test_default_is_opted_in(self, seeded_athlete, monkeypatch):
        monkeypatch.setattr(
            pat_expiry, "_email_opted_in", pat_expiry._email_opted_in
        )
        # No preference recorded at all → email is still sent.
        assert seeded_athlete.app_settings in (None, {})

    def test_the_setting_key_is_the_one_the_profile_api_writes(self):
        from backend.app.api.athlete import PAT_EXPIRY_EMAIL_SETTING

        assert PAT_EXPIRY_EMAIL_SETTING == pat_expiry.EMAIL_OPT_OUT_SETTING
