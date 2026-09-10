"""Unit tests for outbound-email accounting (issue #66).

Email is counted per **message**, which is how it is billed, and at the
``EmailProvider.send()`` seam — the vendor SDKs are not ``httpx`` clients we
build, so the counting transport cannot see them.
"""

from collections.abc import Mapping

import pytest
from sqlalchemy import select

from backend.app.models.api_usage_orm import ApiUsage
from backend.app.services.email import messages
from backend.app.services.email.base import (
    EmailConfigurationError,
    EmailError,
    EmailProvider,
    InboundEmail,
    OutboundMessage,
)
from backend.app.services.email.counting import CountingEmailProvider


class _FakeProvider(EmailProvider):
    PROVIDER_NAME = "lettermint"

    def __init__(self, *, fail: bool = False, unconfigured: bool = False) -> None:
        self.fail = fail
        self.unconfigured = unconfigured
        self.sent: list[OutboundMessage] = []

    @classmethod
    def from_settings(cls, settings):  # pragma: no cover - not used
        return cls()

    @property
    def is_configured(self) -> bool:
        return not self.unconfigured

    async def send(self, message: OutboundMessage) -> str:
        if self.unconfigured:
            raise EmailConfigurationError("outbound sending is not configured")
        if self.fail:
            raise EmailError("vendor refused the message")
        self.sent.append(message)
        return "msg-1"

    def verify_inbound_signature(self, raw_body: bytes, headers: Mapping[str, str]) -> bool:
        return True

    def parse_inbound(self, raw_body: bytes, headers: Mapping[str, str]) -> InboundEmail:
        raise NotImplementedError


async def _rows(factory):
    async with factory() as session:
        return (
            (await session.execute(select(ApiUsage).order_by(ApiUsage.created_at)))
            .scalars()
            .all()
        )


class TestEveryKindIsTagged:
    async def test_the_five_message_kinds_record_distinctly(self, api_usage_db):
        provider = CountingEmailProvider(_FakeProvider())
        await messages.send_verification_email(
            provider, to="a@b.co", action_url="https://x/verify"
        )
        await messages.send_email_change_email(
            provider, to="a@b.co", action_url="https://x/change"
        )
        await messages.send_email_change_authorisation(
            provider, to="a@b.co", action_url="https://x/auth", new_email="n@b.co"
        )
        await messages.send_password_reset_email(
            provider, to="a@b.co", action_url="https://x/reset"
        )
        await messages.send_token_expiry_email(
            provider, to="a@b.co", token_name="ci", days_left=3,
            manage_url="https://x/tokens",
        )

        rows = await _rows(api_usage_db)
        assert [r.endpoint for r in rows] == [
            "send/verification",
            "send/email_change",
            "send/email_change_authorisation",
            "send/password_reset",
            "send/token_expiry",
        ]
        assert {r.service for r in rows} == {"lettermint"}
        assert {r.method for r in rows} == {"POST"}
        assert {r.outcome for r in rows} == {"ok"}

    async def test_an_unlabelled_send_still_counts(self, api_usage_db):
        provider = CountingEmailProvider(_FakeProvider())
        await provider.send(
            OutboundMessage(to="a@b.co", subject="s", html="<p>h</p>", text="h")
        )
        (row,) = await _rows(api_usage_db)
        assert row.endpoint == "send"

    async def test_no_recipient_address_is_persisted(self, api_usage_db):
        provider = CountingEmailProvider(_FakeProvider())
        await messages.send_verification_email(
            provider, to="athlete@example.com", action_url="https://x/verify"
        )
        (row,) = await _rows(api_usage_db)
        assert "athlete@example.com" not in row.endpoint


class TestFailureModes:
    async def test_a_delivery_failure_is_recorded(self, api_usage_db):
        provider = CountingEmailProvider(_FakeProvider(fail=True))
        with pytest.raises(EmailError):
            await messages.send_verification_email(
                provider, to="a@b.co", action_url="https://x/verify"
            )
        (row,) = await _rows(api_usage_db)
        assert row.status_code is None
        assert row.outcome == "transport_error"

    async def test_an_unconfigured_provider_records_nothing(self, api_usage_db):
        # Nothing left the building: no vendor was contacted, nothing was billed.
        provider = CountingEmailProvider(_FakeProvider(unconfigured=True))
        with pytest.raises(EmailConfigurationError):
            await messages.send_verification_email(
                provider, to="a@b.co", action_url="https://x/verify"
            )
        assert await _rows(api_usage_db) == []


class TestDelegation:
    async def test_the_wrapper_is_transparent(self, api_usage_db):
        inner = _FakeProvider()
        provider = CountingEmailProvider(inner)
        assert provider.PROVIDER_NAME == "lettermint"
        assert provider.wrapped is inner
        assert provider.is_configured is True
        assert provider.verify_inbound_signature(b"body", {}) is True

        await provider.send(
            OutboundMessage(to="a@b.co", subject="s", html="<p>h</p>", text="h")
        )
        assert len(inner.sent) == 1

    async def test_a_recording_failure_does_not_fail_the_send(
        self, api_usage_db, monkeypatch
    ):
        from backend.app.services import api_usage as api_usage_service

        def boom():
            raise RuntimeError("database is locked")

        monkeypatch.setattr(api_usage_service, "api_usage_session_factory", boom)
        provider = CountingEmailProvider(_FakeProvider())
        message_id = await messages.send_verification_email(
            provider, to="a@b.co", action_url="https://x/verify"
        )
        assert message_id == "msg-1"
