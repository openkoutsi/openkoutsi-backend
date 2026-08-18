"""Bridge secret handling: constant-time comparison, and no `changeme` start.

**F-09** — both bridges compared their bearer secret with `!=`, which returns as
soon as two bytes differ. Remote timing attacks over HTTP are impractical in most
conditions, so this is hygiene rather than a hole; what made it a finding is that
the Wahoo bridge's *webhook token* check already used `compare_digest`, so one of
the two comparisons was careful and the other was not, which reads as a decision
rather than an oversight.

**F-10** — both bridges started happily on the literal default `changeme`. Each
sits on a public HTTPS URL, so an unconfigured one hands its event queue to
anyone who sends `Bearer changeme`. On the Strava bridge that same value is also
the `hub.verify_token` authorising a subscription against it.

The constant-time property itself is not measurable in a test, so what is
asserted here is the observable consequence of doing it correctly: the encoding
step that stops an attacker-supplied value from turning the comparison into a
500. `compare_digest` raises `TypeError` on a `str` holding non-ASCII, so a naive
swap from `!=` to `compare_digest` would have introduced exactly that.
"""
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import strava_bridge.main as strava_bridge
import wahoo_bridge.main as wahoo_bridge

_STRONG = "b" * 32
# Sent as raw bytes: httpx refuses to encode a non-ASCII str header, and
# Starlette latin-1 decodes it back into the non-ASCII str the app sees.
_NON_ASCII_HEADER = b"Bearer \xfc\xfc\xfc\xfc"


# ── F-10: the placeholder is refused ──────────────────────────────────────


class TestStravaBridgeSecretRequired:
    @pytest.mark.parametrize("value", ["changeme", "", "short", "b" * 31])
    def test_weak_secret_refuses_to_start(self, value):
        with pytest.raises(ValueError) as exc_info:
            strava_bridge.Settings(bridge_secret=value)
        assert "BRIDGE_SECRET" in str(exc_info.value)

    def test_a_strong_secret_is_accepted(self):
        assert strava_bridge.Settings(bridge_secret=_STRONG).bridge_secret == _STRONG

    def test_the_message_says_it_must_match_the_main_app(self):
        """Changing it on one side only is the way this gets misconfigured."""
        with pytest.raises(ValueError) as exc_info:
            strava_bridge.Settings(bridge_secret="changeme")
        assert "match" in str(exc_info.value)


class TestWahooBridgeSecretRequired:
    @pytest.mark.parametrize("value", ["changeme", "", "short", "b" * 31])
    def test_weak_secret_refuses_to_start(self, value):
        with pytest.raises(ValueError) as exc_info:
            wahoo_bridge.Settings(wahoo_bridge_secret=value)
        assert "WAHOO_BRIDGE_SECRET" in str(exc_info.value)

    def test_a_strong_secret_is_accepted(self):
        settings = wahoo_bridge.Settings(wahoo_bridge_secret=_STRONG)
        assert settings.wahoo_bridge_secret == _STRONG


# ── F-09: the comparison is constant-time, and total ──────────────────────


@pytest.mark.parametrize("module", [strava_bridge, wahoo_bridge], ids=["strava", "wahoo"])
class TestSecretComparison:
    def test_equal_secrets_match(self, module):
        assert module._secret_equals("s3cret", "s3cret")

    @pytest.mark.parametrize("supplied", ["", "s3cre", "s3cret ", "S3CRET", "x" * 200])
    def test_different_secrets_do_not(self, module, supplied):
        assert not module._secret_equals(supplied, "s3cret")

    def test_a_non_ascii_value_is_refused_rather_than_raising(self, module):
        """The trap in this fix.

        `compare_digest` raises TypeError on a str holding non-ASCII. Swapping
        `!=` for it without encoding would have turned an attacker-supplied
        header into an unauthenticated 500.
        """
        assert not module._secret_equals("Bearer üüüü", "Bearer s3cret")

    def test_a_prefix_match_is_still_a_mismatch(self, module):
        assert not module._secret_equals("s3", "s3cret")


# ── The same, through the actual endpoints ────────────────────────────────


@pytest.fixture
async def strava_db(monkeypatch):
    """In-memory queue for the one test that gets past auth and reads it."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(strava_bridge.Base.metadata.create_all)
    monkeypatch.setattr(
        strava_bridge, "AsyncSessionLocal", async_sessionmaker(engine, expire_on_commit=False)
    )
    yield
    await engine.dispose()


@pytest.fixture
async def strava_client():
    async with AsyncClient(
        transport=ASGITransport(app=strava_bridge.app), base_url="http://bridge"
    ) as c:
        yield c


@pytest.fixture
async def wahoo_client():
    async with AsyncClient(
        transport=ASGITransport(app=wahoo_bridge.app), base_url="http://bridge"
    ) as c:
        yield c


class TestBearerEndToEnd:
    async def test_strava_correct_secret_is_accepted(
        self, strava_client, strava_db, monkeypatch
    ):
        monkeypatch.setattr(strava_bridge.settings, "bridge_secret", _STRONG)
        resp = await strava_client.get(
            "/events/pending", headers={"Authorization": f"Bearer {_STRONG}"}
        )
        assert resp.status_code == 200

    async def test_strava_wrong_secret_is_rejected(self, strava_client, monkeypatch):
        monkeypatch.setattr(strava_bridge.settings, "bridge_secret", _STRONG)
        resp = await strava_client.get(
            "/events/pending", headers={"Authorization": "Bearer wrong"}
        )
        assert resp.status_code == 401

    async def test_strava_non_ascii_header_is_401_not_500(
        self, strava_client, monkeypatch
    ):
        monkeypatch.setattr(strava_bridge.settings, "bridge_secret", _STRONG)
        resp = await strava_client.get(
            "/events/pending", headers={"Authorization": _NON_ASCII_HEADER}
        )
        assert resp.status_code == 401

    async def test_wahoo_non_ascii_header_is_401_not_500(
        self, wahoo_client, monkeypatch
    ):
        monkeypatch.setattr(wahoo_bridge.settings, "wahoo_bridge_secret", _STRONG)
        resp = await wahoo_client.get(
            "/events/pending", headers={"Authorization": _NON_ASCII_HEADER}
        )
        assert resp.status_code == 401

    async def test_wahoo_non_ascii_webhook_token_is_403_not_500(
        self, wahoo_client, monkeypatch
    ):
        """The token comes out of attacker-supplied JSON, so it has the same trap.

        This comparison already used `compare_digest` — on strs — so it was a
        latent 500 in the code the finding held up as the good example.
        """
        monkeypatch.setattr(wahoo_bridge.settings, "wahoo_webhook_token", _STRONG)
        resp = await wahoo_client.post(
            "/webhook", json={"webhook_token": "üüüü", "event_type": "workout_summary"}
        )
        assert resp.status_code == 403


class TestTheComparisonIsConstantTimeByConstruction:
    """Asserted on the mechanism, because the property is not observable.

    `!=` and `compare_digest` agree on every input; what differs is only *how
    long* they take, and a functional test cannot see that — a timing test on a
    string comparison would be measuring scheduler noise. So these assert that
    the bearer check goes through `compare_digest` at all, which is the thing
    that would silently regress if someone simplified the helper back to `!=`.
    """

    async def test_strava_bearer_check_uses_compare_digest(
        self, strava_client, monkeypatch
    ):
        monkeypatch.setattr(strava_bridge.settings, "bridge_secret", _STRONG)
        calls = []
        real = strava_bridge.hmac.compare_digest

        def _spy(a, b):
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr(strava_bridge.hmac, "compare_digest", _spy)
        await strava_client.get(
            "/events/pending", headers={"Authorization": "Bearer wrong"}
        )

        assert calls, "bearer secret was compared without compare_digest"
        assert all(
            isinstance(a, bytes) and isinstance(b, bytes) for a, b in calls
        ), "compare_digest was handed strs, which raises on non-ASCII"

    async def test_wahoo_bearer_check_uses_compare_digest(
        self, wahoo_client, monkeypatch
    ):
        monkeypatch.setattr(wahoo_bridge.settings, "wahoo_bridge_secret", _STRONG)
        calls = []
        real = wahoo_bridge.secrets.compare_digest

        def _spy(a, b):
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr(wahoo_bridge.secrets, "compare_digest", _spy)
        await wahoo_client.get(
            "/events/pending", headers={"Authorization": "Bearer wrong"}
        )

        assert calls, "bearer secret was compared without compare_digest"
        assert all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in calls)


class TestHubChallenge:
    """The Strava bridge's third comparison of the same secret."""

    async def test_correct_verify_token_echoes_the_challenge(
        self, strava_client, monkeypatch
    ):
        monkeypatch.setattr(strava_bridge.settings, "bridge_secret", _STRONG)
        resp = await strava_client.get(
            "/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": _STRONG,
                "hub.challenge": "abc123",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"hub.challenge": "abc123"}

    async def test_wrong_verify_token_is_refused(self, strava_client, monkeypatch):
        monkeypatch.setattr(strava_bridge.settings, "bridge_secret", _STRONG)
        resp = await strava_client.get(
            "/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong",
                "hub.challenge": "abc123",
            },
        )
        assert resp.status_code == 403

    async def test_missing_verify_token_is_refused(self, strava_client, monkeypatch):
        """`None` reached the comparison; it must not be treated as a match."""
        monkeypatch.setattr(strava_bridge.settings, "bridge_secret", _STRONG)
        resp = await strava_client.get(
            "/webhook", params={"hub.mode": "subscribe", "hub.challenge": "abc123"}
        )
        assert resp.status_code == 403
