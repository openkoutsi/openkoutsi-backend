"""Unit tests for the personal-access-token primitives (issue #46).

Format, hashing, verification, status and the coarse ``last_used_at`` write —
the pieces the identity resolver leans on, tested without a request in flight.
"""
from datetime import datetime, timedelta, timezone

import pytest

from backend.app.core.scopes import SCOPES, PatAccess, validate_scopes
from backend.app.models.registry_orm import PersonalAccessToken
from backend.app.services import personal_access_tokens as pat


def _token(**overrides) -> PersonalAccessToken:
    now = datetime.now(timezone.utc)
    fields = {
        "id": "token-id",
        "user_id": "user-id",
        "token_hash": "x" * 64,
        "name": "test",
        "scopes": '["activities:read"]',
        "expires_at": now + timedelta(days=30),
        "created_at": now,
    }
    fields.update(overrides)
    return PersonalAccessToken(**fields)


# ── Format ──────────────────────────────────────────────────────────────────


class TestTokenFormat:
    def test_minted_token_round_trips_through_parse(self):
        token_id, raw, token_hash = pat.mint_token()

        assert raw.startswith("okp_")
        parsed = pat.parse_token(raw)
        assert parsed is not None
        parsed_id, secret = parsed
        assert parsed_id == token_id
        assert pat.hash_secret(secret) == token_hash

    def test_secret_is_never_recoverable_from_what_is_stored(self):
        """Only the hash is persisted — the raw secret exists once, at creation."""
        _, raw, token_hash = pat.mint_token()
        _, secret = pat.parse_token(raw)

        assert secret not in token_hash
        assert raw not in token_hash
        assert len(token_hash) == 64  # sha256 hex

    def test_secret_carries_256_bits(self):
        _, raw, _ = pat.mint_token()
        _, secret = pat.parse_token(raw)
        # token_urlsafe(32) — 32 bytes, base64url-encoded without padding.
        assert len(secret) >= 43

    def test_two_tokens_never_collide(self):
        raws = {pat.mint_token()[1] for _ in range(50)}
        assert len(raws) == 50

    def test_underscores_in_the_secret_do_not_break_the_parse(self):
        """The urlsafe alphabet includes `_`; the split is capped at two."""
        raw = "okp_11111111-2222-3333-4444-555555555555_abc_def_ghi"
        assert pat.parse_token(raw) == (
            "11111111-2222-3333-4444-555555555555",
            "abc_def_ghi",
        )

    @pytest.mark.parametrize(
        "raw",
        ["", "okp_", "okp_onlyid", "not-a-token", "eyJhbGciOiJIUzI1NiJ9.x.y", "okp__secret"],
    )
    def test_malformed_values_parse_to_none(self, raw):
        assert pat.parse_token(raw) is None

    def test_looks_like_pat_only_matches_the_prefix(self):
        assert pat.looks_like_pat("okp_a_b")
        assert not pat.looks_like_pat("eyJhbGciOiJIUzI1NiJ9.abc.def")


class TestVerification:
    def test_correct_secret_verifies(self):
        _, raw, token_hash = pat.mint_token()
        _, secret = pat.parse_token(raw)
        assert pat.verify_secret(secret, token_hash)

    def test_wrong_secret_does_not_verify(self):
        _, _, token_hash = pat.mint_token()
        _, other_raw, _ = pat.mint_token()
        _, other_secret = pat.parse_token(other_raw)
        assert not pat.verify_secret(other_secret, token_hash)

    def test_comparison_is_timing_safe(self, monkeypatch):
        """Verification must go through `hmac.compare_digest`, not `==`."""
        calls = []
        real = pat.hmac.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr(pat.hmac, "compare_digest", spy)
        _, raw, token_hash = pat.mint_token()
        _, secret = pat.parse_token(raw)

        assert pat.verify_secret(secret, token_hash)
        assert len(calls) == 1


# ── Lifetime ────────────────────────────────────────────────────────────────


class TestLifetime:
    def test_there_is_no_never_expires(self):
        assert 0 not in pat.ALLOWED_LIFETIME_DAYS
        assert all(days > 0 for days in pat.ALLOWED_LIFETIME_DAYS)

    def test_ceiling_is_one_year(self):
        assert pat.MAX_LIFETIME_DAYS == 365
        assert max(pat.ALLOWED_LIFETIME_DAYS) == pat.MAX_LIFETIME_DAYS

    def test_expiry_is_computed_from_now(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert pat.expires_at_for(90, now) == now + timedelta(days=90)

    @pytest.mark.parametrize("days", [366, 3650, 0, -1])
    def test_lifetimes_outside_the_band_are_rejected(self, days):
        with pytest.raises(ValueError):
            pat.expires_at_for(days)


class TestStatus:
    def test_a_fresh_token_is_active(self):
        assert pat.status_of(_token()) == "active"
        assert pat.is_active(_token())

    def test_an_expired_token_is_not_active(self):
        token = _token(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        assert pat.status_of(token) == "expired"
        assert not pat.is_active(token)

    def test_a_revoked_token_is_not_active(self):
        token = _token(revoked_at=datetime.now(timezone.utc))
        assert pat.status_of(token) == "revoked"
        assert not pat.is_active(token)

    def test_revocation_wins_over_expiry(self):
        """A token withdrawn before its time still reads as a deliberate act."""
        now = datetime.now(timezone.utc)
        token = _token(
            expires_at=now - timedelta(days=2), revoked_at=now - timedelta(days=5)
        )
        assert pat.status_of(token) == "revoked"

    def test_naive_timestamps_from_sqlite_are_read_as_utc(self):
        token = _token(expires_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1))
        assert pat.status_of(token) == "expired"

    def test_scopes_survive_a_round_trip(self):
        assert pat.scopes_of(_token(scopes='["a","b"]')) == ["a", "b"]

    @pytest.mark.parametrize("raw", ["", "not json", "null", '[1, 2]'])
    def test_unreadable_scopes_deny_rather_than_explode(self, raw):
        assert pat.scopes_of(_token(scopes=raw)) == []


# ── Scope vocabulary ────────────────────────────────────────────────────────


class TestScopeVocabulary:
    def test_validate_returns_canonical_order(self):
        assert validate_scopes(["metrics:read", "activities:read"]) == [
            "activities:read",
            "metrics:read",
        ]

    def test_duplicates_collapse(self):
        assert validate_scopes(["activities:read"] * 3) == ["activities:read"]

    def test_unknown_scopes_are_rejected(self):
        with pytest.raises(ValueError, match="admin:everything"):
            validate_scopes(["activities:read", "admin:everything"])

    def test_every_scope_is_resource_qualified(self):
        for scope in SCOPES:
            resource, _, action = scope.partition(":")
            assert resource and action, scope

    def test_there_is_no_wildcard_or_admin_scope(self):
        """A PAT can never be granted the admin surface, by vocabulary alone."""
        assert not any(s.startswith("admin") or "*" in s for s in SCOPES)


class TestPatAccessDeclaration:
    def test_reads_and_writes_map_to_their_own_scopes(self):
        access = PatAccess(read="activities:read", write="activities:write")
        assert access.scope_for("GET") == "activities:read"
        assert access.scope_for("HEAD") == "activities:read"
        assert access.scope_for("POST") == "activities:write"
        assert access.scope_for("PATCH") == "activities:write"
        assert access.scope_for("DELETE") == "activities:write"

    def test_a_read_only_router_exposes_no_write_scope(self):
        access = PatAccess(read="metrics:read")
        assert access.scope_for("GET") == "metrics:read"
        assert access.scope_for("POST") is None

    def test_a_forbidden_declaration_yields_no_scope_at_all(self):
        access = PatAccess(allowed=False)
        assert access.scope_for("GET") is None
        assert access.scope_for("POST") is None

    def test_declaring_an_unknown_scope_fails_loudly(self):
        with pytest.raises(ValueError):
            PatAccess(read="nonsense:read")


# ── last_used_at ────────────────────────────────────────────────────────────


class _FakeSession:
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1


class TestCoarseLastUsed:
    async def test_first_use_records_a_timestamp(self):
        token, session = _token(), _FakeSession()
        assert await pat.touch_last_used(session, token) is True
        assert token.last_used_at is not None
        assert session.commits == 1

    async def test_a_recent_timestamp_is_left_alone(self):
        """A write on every request would be the hottest writer in the system."""
        now = datetime.now(timezone.utc)
        token = _token(last_used_at=now - timedelta(minutes=5))
        session = _FakeSession()

        assert await pat.touch_last_used(session, token, now) is False
        assert token.last_used_at == now - timedelta(minutes=5)
        assert session.commits == 0

    async def test_a_stale_timestamp_is_refreshed(self):
        now = datetime.now(timezone.utc)
        token = _token(last_used_at=now - timedelta(hours=2))
        session = _FakeSession()

        assert await pat.touch_last_used(session, token, now) is True
        assert token.last_used_at == now
