"""Integration tests for personal access tokens (issue #46).

``tests/conftest.py`` overrides ``get_ctx_and_session`` and lets **any**
``Bearer …`` through, which would make every assertion here vacuous. So this
module builds its own client that overrides only the registry session, leaving
the real identity path — ``oauth2_scheme`` → ``get_current_user`` → the PAT
resolver → ``get_ctx_and_session`` → the per-user DB — intact, the way
``test_auth.py`` does for the login flow.

Per-user DBs are real files under the autouse ``isolate_user_dbs`` temp dir, so
the encryption-context and cross-user-isolation tests exercise the same code
production does.
"""
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient

from backend.app.core.auth import create_access_token
from backend.app.core.encryption import set_user_encryption_context
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.registry_orm import InstanceSettings, PersonalAccessToken, User
from backend.app.models.user_orm import Activity, Athlete
from backend.app.services import personal_access_tokens as pat

_TEST_USER_ID = "test-user-00000000"
_TEST_ROLES = ["administrator", "user"]


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
async def pat_client(app, registry_session):
    """A client with the **real** identity path — only the registry is overridden.

    Never request this *and* the shared ``client`` fixture in the same test: both
    write ``dependency_overrides`` on the module-scoped app, and ``client``'s
    ``get_ctx_and_session`` override waves every ``Bearer …`` through, which
    would quietly make every assertion here pass for the wrong reason.
    ``_TEST_USER_ID`` is an administrator, so ``session_headers`` reaches the
    admin API through this client too.
    """
    from backend.app.core.limiter import limiter

    async def _override_registry():
        yield registry_session

    app.dependency_overrides[get_registry_session] = _override_registry
    limiter.enabled = False
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
    finally:
        limiter.enabled = True
        app.dependency_overrides.clear()


async def _provision(user_id: str, athlete_id: str | None = None) -> str:
    """Create a user's own DB and athlete profile, as account activation does."""
    set_user_encryption_context(user_id)
    await init_user_db(user_id)
    athlete_id = athlete_id or str(uuid.uuid4())
    async with get_user_session_factory(user_id)() as session:
        session.add(
            Athlete(id=athlete_id, global_user_id=user_id, name="Test", ftp_tests=[])
        )
        await session.commit()
    return athlete_id


async def _add_user(registry_session, username: str, roles=("user",)) -> User:
    from backend.app.api.consent import CURRENT_CONSENT_VERSION

    user = User(
        id=str(uuid.uuid4()),
        username=username,
        password_hash="x",
        roles=json.dumps(list(roles)),
        consented_at=datetime.now(timezone.utc),
        consent_version=CURRENT_CONSENT_VERSION,
    )
    registry_session.add(user)
    await registry_session.commit()
    return user


async def _issue(
    registry_session,
    user_id: str = _TEST_USER_ID,
    scopes: list[str] | None = None,
    expires_in_days: int = 90,
    revoked: bool = False,
    name: str = "test-token",
) -> str:
    """Mint a real credential straight into the registry and return the raw token."""
    token_id, raw, token_hash = pat.mint_token()
    registry_session.add(
        PersonalAccessToken(
            id=token_id,
            user_id=user_id,
            token_hash=token_hash,
            name=name,
            scopes=json.dumps(scopes if scopes is not None else ["activities:read"]),
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_in_days),
            revoked_at=datetime.now(timezone.utc) if revoked else None,
        )
    )
    await registry_session.commit()
    return raw


def _pat_headers(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


@pytest.fixture
def session_headers() -> dict:
    return {"Authorization": f"Bearer {create_access_token(_TEST_USER_ID, _TEST_ROLES, token_version=0)}"}


@pytest.fixture
async def own_athlete():
    """The seeded test user's own per-user DB + athlete profile."""
    return await _provision(_TEST_USER_ID)


# ── Issuing, listing, revoking (session-authenticated only) ─────────────────


class TestIssuing:
    async def test_create_returns_the_secret_exactly_once(
        self, pat_client, registry_session, session_headers
    ):
        resp = await pat_client.post(
            "/api/tokens",
            json={"name": "backup", "scopes": ["athlete:export"], "expires_in_days": 30},
            headers=session_headers,
        )
        assert resp.status_code == 201
        created = resp.json()
        assert created["token"].startswith("okp_")
        assert created["name"] == "backup"
        assert created["scopes"] == ["athlete:export"]
        assert created["status"] == "active"

        listed = await pat_client.get("/api/tokens", headers=session_headers)
        assert listed.status_code == 200
        assert len(listed.json()) == 1
        # The secret is never returned again, by any route.
        assert "token" not in listed.json()[0]

    async def test_only_the_hash_is_persisted(
        self, pat_client, registry_session, session_headers
    ):
        resp = await pat_client.post(
            "/api/tokens",
            json={"name": "backup", "scopes": ["activities:read"]},
            headers=session_headers,
        )
        raw = resp.json()["token"]
        _, secret = pat.parse_token(raw)

        token = await pat.load_by_id(registry_session, resp.json()["id"])
        assert token.token_hash == pat.hash_secret(secret)
        assert secret not in token.token_hash
        assert raw not in token.token_hash

    async def test_lifetime_beyond_the_ceiling_is_rejected_server_side(
        self, pat_client, session_headers
    ):
        """A hand-rolled POST asking for ten years does not get one."""
        resp = await pat_client.post(
            "/api/tokens",
            json={"name": "forever", "scopes": ["activities:read"], "expires_in_days": 3650},
            headers=session_headers,
        )
        assert resp.status_code == 422

    async def test_there_is_no_never_expires(self, pat_client, session_headers):
        resp = await pat_client.post(
            "/api/tokens",
            json={"name": "forever", "scopes": ["activities:read"], "expires_in_days": 0},
            headers=session_headers,
        )
        assert resp.status_code == 422

    async def test_unknown_scopes_are_rejected(self, pat_client, session_headers):
        resp = await pat_client.post(
            "/api/tokens",
            json={"name": "x", "scopes": ["activities:read", "admin:everything"]},
            headers=session_headers,
        )
        assert resp.status_code == 400
        assert "admin:everything" in resp.json()["detail"]

    async def test_a_token_with_no_scopes_is_rejected(self, pat_client, session_headers):
        resp = await pat_client.post(
            "/api/tokens", json={"name": "x", "scopes": []}, headers=session_headers
        )
        assert resp.status_code == 400

    async def test_the_scope_vocabulary_is_published_for_the_picker(
        self, pat_client, session_headers
    ):
        resp = await pat_client.get("/api/tokens/scopes", headers=session_headers)
        assert resp.status_code == 200
        body = resp.json()
        names = {s["name"] for s in body["scopes"]}
        assert "activities:read" in names
        assert body["allowed_lifetime_days"] == [7, 30, 90, 180, 365]
        assert body["default_lifetime_days"] == 90
        # `athlete:export` is presented apart from the ordinary read scopes.
        export = next(s for s in body["scopes"] if s["name"] == "athlete:export")
        assert export["sensitive"] is True
        assert all(
            s["sensitive"] is False for s in body["scopes"] if s["name"] != "athlete:export"
        )

    async def test_a_token_is_immutable(self, app):
        """No route mutates an issued token's scopes or expiry — by absence."""
        mutating = [
            (sorted(r.methods), r.path)
            for r in app.routes
            if isinstance(r, APIRoute)
            and r.path.startswith("/api/tokens")
            and r.methods & {"PUT", "PATCH"}
        ]
        assert mutating == []

    async def test_listing_shows_dead_tokens_too(
        self, pat_client, registry_session, session_headers
    ):
        await _issue(registry_session, name="live")
        await _issue(registry_session, name="withdrawn", revoked=True)
        await _issue(registry_session, name="past", expires_in_days=-1)

        resp = await pat_client.get("/api/tokens", headers=session_headers)
        by_name = {t["name"]: t["status"] for t in resp.json()}
        assert by_name == {"live": "active", "withdrawn": "revoked", "past": "expired"}


# ── Authenticating with a token ────────────────────────────────────────────


class TestAuthentication:
    async def test_a_scoped_token_reaches_the_route_it_was_granted(
        self, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, scopes=["activities:read"])
        resp = await pat_client.get("/api/activities", headers=_pat_headers(raw))
        assert resp.status_code == 200

    async def test_a_read_token_cannot_write(
        self, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, scopes=["activities:read"])
        resp = await pat_client.delete(
            "/api/activities/whatever", headers=_pat_headers(raw)
        )
        assert resp.status_code == 403
        assert "activities:write" in resp.json()["detail"]

    async def test_a_token_cannot_reach_a_resource_it_holds_no_scope_for(
        self, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, scopes=["activities:read"])
        resp = await pat_client.get("/api/goals", headers=_pat_headers(raw))
        assert resp.status_code == 403
        assert "goals:read" in resp.json()["detail"]

    async def test_an_unknown_token_is_401(self, pat_client, own_athlete):
        _, raw, _ = pat.mint_token()
        resp = await pat_client.get("/api/activities", headers=_pat_headers(raw))
        assert resp.status_code == 401

    async def test_the_right_id_with_the_wrong_secret_is_401(
        self, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session)
        token_id, _ = pat.parse_token(raw)
        forged = f"okp_{token_id}_not-the-secret"
        resp = await pat_client.get("/api/activities", headers=_pat_headers(forged))
        assert resp.status_code == 401

    async def test_an_expired_token_is_401(
        self, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, expires_in_days=-1)
        resp = await pat_client.get("/api/activities", headers=_pat_headers(raw))
        assert resp.status_code == 401

    async def test_a_deleted_user_takes_their_tokens_with_them(
        self, pat_client, registry_session, own_athlete
    ):
        user = await _add_user(registry_session, "goner")
        await _provision(user.id)
        raw = await _issue(registry_session, user_id=user.id)

        await registry_session.delete(user)
        await registry_session.commit()

        resp = await pat_client.get("/api/activities", headers=_pat_headers(raw))
        assert resp.status_code == 401
        # ondelete=CASCADE: the row is gone with the account.
        assert await pat.load_by_id(registry_session, pat.parse_token(raw)[0]) is None

    async def test_last_used_is_recorded(
        self, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, scopes=["activities:read"])
        token_id, _ = pat.parse_token(raw)
        assert (await pat.load_by_id(registry_session, token_id)).last_used_at is None

        await pat_client.get("/api/activities", headers=_pat_headers(raw))

        registry_session.expire_all()
        assert (await pat.load_by_id(registry_session, token_id)).last_used_at is not None

    async def test_last_used_is_written_coarsely(
        self, pat_client, registry_session, own_athlete
    ):
        """A write on every request would be the hottest writer in the system."""
        raw = await _issue(registry_session, scopes=["activities:read"])
        token_id, _ = pat.parse_token(raw)

        await pat_client.get("/api/activities", headers=_pat_headers(raw))
        registry_session.expire_all()
        first = (await pat.load_by_id(registry_session, token_id)).last_used_at

        for _ in range(3):
            await pat_client.get("/api/activities", headers=_pat_headers(raw))
        registry_session.expire_all()
        assert (await pat.load_by_id(registry_session, token_id)).last_used_at == first


# ── Revocation ─────────────────────────────────────────────────────────────


class TestRevocation:
    async def test_revocation_is_immediate(
        self, pat_client, registry_session, session_headers, own_athlete
    ):
        """No cache, no grace window."""
        raw = await _issue(registry_session, scopes=["activities:read"])
        token_id, _ = pat.parse_token(raw)

        assert (
            await pat_client.get("/api/activities", headers=_pat_headers(raw))
        ).status_code == 200

        revoke = await pat_client.delete(
            f"/api/tokens/{token_id}", headers=session_headers
        )
        assert revoke.status_code == 204

        assert (
            await pat_client.get("/api/activities", headers=_pat_headers(raw))
        ).status_code == 401

    async def test_the_revoked_row_survives_with_its_hash(
        self, pat_client, registry_session, session_headers
    ):
        """Keeping the hash keeps a presented-but-revoked token *recognisable*."""
        raw = await _issue(registry_session)
        token_id, secret = pat.parse_token(raw)

        await pat_client.delete(f"/api/tokens/{token_id}", headers=session_headers)

        registry_session.expire_all()
        row = await pat.load_by_id(registry_session, token_id)
        assert row is not None
        assert row.revoked_at is not None
        assert pat.verify_secret(secret, row.token_hash)

    async def test_a_revoked_attempt_is_audited_apart_from_an_unknown_one(
        self, pat_client, registry_session, own_athlete
    ):
        revoked_raw = await _issue(registry_session, revoked=True)
        _, unknown_raw, _ = pat.mint_token()

        with patch("backend.app.core.audit.log") as log:
            await pat_client.get("/api/activities", headers=_pat_headers(revoked_raw))
            await pat_client.get("/api/activities", headers=_pat_headers(unknown_raw))

        outcomes = [
            call.kwargs["extra"]["pat_outcome"] for call in log.info.call_args_list
        ]
        assert outcomes == ["revoked", "unknown_token"]

    async def test_a_user_cannot_revoke_someone_elses_token(
        self, pat_client, registry_session, session_headers
    ):
        other = await _add_user(registry_session, "someone-else")
        raw = await _issue(registry_session, user_id=other.id)
        token_id, _ = pat.parse_token(raw)

        resp = await pat_client.delete(
            f"/api/tokens/{token_id}", headers=session_headers
        )
        assert resp.status_code == 404

    async def test_a_password_reset_revokes_every_live_token(
        self, pat_client, registry_session
    ):
        from backend.app.core.auth import hash_password
        from backend.app.models.registry_orm import PasswordResetToken

        user = await _add_user(registry_session, "resetter")
        user.password_hash = hash_password("Oldpass12345")
        raw_a = await _issue(registry_session, user_id=user.id, name="a")
        raw_b = await _issue(registry_session, user_id=user.id, name="b")

        reset_token = "reset-me"
        registry_session.add(PasswordResetToken(
            user_id=user.id,
            token_hash=__import__("hashlib").sha256(reset_token.encode()).hexdigest(),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        ))
        await registry_session.commit()

        resp = await pat_client.post(
            "/api/auth/reset-password",
            json={"token": reset_token, "new_password": "Newpass123456"},
        )
        assert resp.status_code == 204

        registry_session.expire_all()
        for raw in (raw_a, raw_b):
            row = await pat.load_by_id(registry_session, pat.parse_token(raw)[0])
            assert row.revoked_at is not None


# ── What a token can never reach ───────────────────────────────────────────


def _authenticated_routes(app, prefix: str) -> list[tuple[str, str]]:
    from backend.app.core.scopes import route_requires_auth

    found = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith(prefix):
            continue
        if not route_requires_auth(route):
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            found.append((method, route.path))
    return found


def _fill_path(path: str) -> str:
    """Substitute a placeholder for every path parameter."""
    out = []
    for part in path.split("/"):
        out.append("placeholder" if part.startswith("{") else part)
    return "/".join(out)


#: Every scope there is — so "unreachable" means unreachable, not "unscoped".
_ALL_SCOPES: list[str] = []


@pytest.fixture(scope="session", autouse=True)
def _load_all_scopes():
    from backend.app.core.scopes import SCOPES

    _ALL_SCOPES.clear()
    _ALL_SCOPES.extend(SCOPES)


class TestUnreachableSurfaces:
    async def test_admin_is_unreachable_even_for_an_admin_owner(
        self, app, pat_client, registry_session, own_athlete
    ):
        """Admin status must not widen the athlete-data surface."""
        assert "administrator" in _TEST_ROLES  # the token's owner really is an admin
        raw = await _issue(registry_session, scopes=_ALL_SCOPES)

        routes = _authenticated_routes(app, "/api/admin")
        assert routes, "expected admin routes to exist"
        for method, path in routes:
            resp = await pat_client.request(
                method, _fill_path(path), headers=_pat_headers(raw), json={}
            )
            assert resp.status_code == 403, f"{method} {path} → {resp.status_code}"

    async def test_a_token_can_never_mint_or_refresh_a_credential(
        self, app, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, scopes=_ALL_SCOPES)

        for prefix in ("/api/auth", "/api/setup", "/api/tokens"):
            routes = _authenticated_routes(app, prefix)
            for method, path in routes:
                resp = await pat_client.request(
                    method, _fill_path(path), headers=_pat_headers(raw), json={}
                )
                assert resp.status_code == 403, f"{method} {path} → {resp.status_code}"

    async def test_a_token_cannot_create_another_token(
        self, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, scopes=_ALL_SCOPES)
        resp = await pat_client.post(
            "/api/tokens",
            json={"name": "child", "scopes": ["activities:read"]},
            headers=_pat_headers(raw),
        )
        assert resp.status_code == 403

    async def test_the_inbox_is_unreachable_under_every_scope(
        self, app, pat_client, registry_session, own_athlete
    ):
        """A credential should not be able to read the message saying it is
        about to be cut off."""
        raw = await _issue(registry_session, scopes=_ALL_SCOPES)

        routes = _authenticated_routes(app, "/api/messages")
        assert routes, "expected inbox routes to exist"
        for method, path in routes:
            resp = await pat_client.request(
                method, _fill_path(path), headers=_pat_headers(raw), json={}
            )
            assert resp.status_code == 403, f"{method} {path} → {resp.status_code}"

    async def test_the_llm_endpoints_are_unreachable(
        self, app, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, scopes=_ALL_SCOPES)
        for method, path in _authenticated_routes(app, "/api/llm"):
            resp = await pat_client.request(
                method, _fill_path(path), headers=_pat_headers(raw), json={}
            )
            assert resp.status_code == 403, f"{method} {path} → {resp.status_code}"

    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/api/athlete/training-status"),
            ("POST", "/api/activities/placeholder/analyze"),
            ("POST", "/api/goals/placeholder/guidance"),
            ("POST", "/api/plans/placeholder/regenerate"),
            ("POST", "/api/plans/placeholder/generate-upcoming/workouts"),
        ],
    )
    async def test_the_llm_triggers_are_unreachable(
        self, pat_client, registry_session, own_athlete, method, path
    ):
        """They spend money; a token never does."""
        raw = await _issue(registry_session, scopes=_ALL_SCOPES)
        resp = await pat_client.request(method, path, headers=_pat_headers(raw), json={})
        assert resp.status_code == 403

    async def test_consent_cannot_be_granted_by_a_credential(
        self, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, scopes=_ALL_SCOPES)
        resp = await pat_client.post(
            "/api/consent", json={"consent_version": "1.0"}, headers=_pat_headers(raw)
        )
        assert resp.status_code == 403

    async def test_the_oauth_connect_flow_cannot_be_started(
        self, pat_client, registry_session, own_athlete
    ):
        """`GET /{provider}/connect` is a GET but not a read.

        It mints a signed `state` that the unauthenticated callback trusts alone
        to decide whose row the provider tokens are written to — so a read scope
        must not be able to produce an account-linking capability.
        """
        raw = await _issue(registry_session, scopes=_ALL_SCOPES)
        resp = await pat_client.get(
            "/api/integrations/strava/connect", headers=_pat_headers(raw)
        )
        assert resp.status_code == 403

    async def test_llm_configuration_cannot_be_changed(
        self, pat_client, registry_session, own_athlete
    ):
        """Closing the *spending* surfaces is not enough on its own.

        Repointing `llm_base_url` would make the user's own browser session ship
        their training data to a host of the token holder's choosing, with every
        other control here still green — the token never calls an LLM route.
        """
        raw = await _issue(registry_session, scopes=_ALL_SCOPES)
        resp = await pat_client.patch(
            "/api/athlete",
            json={"app_settings": {"llm_base_url": "https://evil.example.com/v1"}},
            headers=_pat_headers(raw),
        )
        assert resp.status_code == 403
        assert "personal access token" in resp.json()["detail"]

    @pytest.mark.parametrize(
        "key,value",
        [
            ("llm_base_url", "https://evil.example.com/v1"),
            ("llm_api_key", "sk-attacker"),
            ("llm_api_key_enc", "pre-encrypted"),
            ("llm_model", "attacker-model"),
            ("llm_models", [{"name": "x", "base_url": "https://evil.example.com/v1"}]),
        ],
    )
    async def test_every_llm_setting_key_is_refused(
        self, pat_client, registry_session, own_athlete, key, value
    ):
        raw = await _issue(registry_session, scopes=_ALL_SCOPES)
        resp = await pat_client.patch(
            "/api/athlete",
            json={"app_settings": {key: value}},
            headers=_pat_headers(raw),
        )
        assert resp.status_code == 403, key

    async def test_ordinary_settings_are_still_writable(
        self, pat_client, registry_session, own_athlete
    ):
        """The LLM guard must not have closed `athlete:write` generally."""
        raw = await _issue(registry_session, scopes=["athlete:write", "athlete:read"])
        resp = await pat_client.patch(
            "/api/athlete",
            json={"app_settings": {"ask_for_rpe": False}},
            headers=_pat_headers(raw),
        )
        assert resp.status_code == 200
        assert resp.json()["app_settings"]["ask_for_rpe"] is False

    async def test_commute_rules_are_ordinary_settings(
        self, pat_client, registry_session, own_athlete
    ):
        """Issue #63: only the LLM keys are guarded, and the guard is specific.

        Commute rules describe the athlete's own riding — they spend nothing and
        redirect nothing — so a token with `athlete:write` may set them.
        """
        raw = await _issue(registry_session, scopes=["athlete:write", "athlete:read"])
        resp = await pat_client.patch(
            "/api/athlete",
            json={"app_settings": {"commute_rules": [{"id": "x", "sport_types": ["Ride"]}]}},
            headers=_pat_headers(raw),
        )
        assert resp.status_code == 200
        assert resp.json()["app_settings"]["commute_rules"][0]["id"] == "x"


class TestExportScope:
    async def test_export_needs_its_own_grant(
        self, pat_client, registry_session, own_athlete
    ):
        """One call returning the entire record is never folded into a general read."""
        raw = await _issue(registry_session, scopes=["athlete:read", "activities:read"])
        resp = await pat_client.get("/api/athlete/export", headers=_pat_headers(raw))
        assert resp.status_code == 403
        assert "athlete:export" in resp.json()["detail"]

    async def test_export_works_with_the_explicit_grant(
        self, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, scopes=["athlete:export"])
        resp = await pat_client.get("/api/athlete/export", headers=_pat_headers(raw))
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    async def test_export_is_audited_by_token_id(
        self, pat_client, registry_session, own_athlete
    ):
        raw = await _issue(registry_session, scopes=["athlete:export"])
        token_id, _ = pat.parse_token(raw)

        with patch("backend.app.core.audit.log") as log:
            await pat_client.get("/api/athlete/export", headers=_pat_headers(raw))

        record = log.info.call_args.kwargs["extra"]
        assert record["pat_outcome"] == "ok"
        assert record["pat_token_id"] == token_id
        assert record["http_path"] == "/api/athlete/export"
        assert record["required_scope"] == "athlete:export"


# ── Isolation and the encryption context ───────────────────────────────────


class TestIsolation:
    async def test_one_users_token_never_returns_anothers_data(
        self, pat_client, registry_session
    ):
        """Two users, overlapping activity dates — the fixture that would catch
        a filter that leaked."""
        when = datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc)

        user_a = await _add_user(registry_session, "athlete-a")
        user_b = await _add_user(registry_session, "athlete-b")
        athlete_a = await _provision(user_a.id)
        athlete_b = await _provision(user_b.id)

        for user_id, athlete_id, name in (
            (user_a.id, athlete_a, "A's ride"),
            (user_b.id, athlete_b, "B's ride"),
        ):
            set_user_encryption_context(user_id)
            async with get_user_session_factory(user_id)() as session:
                session.add(Activity(
                    athlete_id=athlete_id,
                    name=name,
                    sport_type="cycling",
                    start_time=when,
                    duration_s=3600,
                ))
                await session.commit()

        raw_a = await _issue(
            registry_session, user_id=user_a.id, scopes=["activities:read"]
        )
        resp = await pat_client.get("/api/activities", headers=_pat_headers(raw_a))
        assert resp.status_code == 200

        names = [item["name"] for item in resp.json()["items"]]
        assert names == ["A's ride"]
        assert "B's ride" not in names

    async def test_a_pat_request_decrypts_under_the_users_own_key(
        self, pat_client, registry_session
    ):
        """The per-user key is derived from the user id, so a PAT-authenticated
        request resolves it exactly as a session-authenticated one does."""
        from backend.app.core import encryption

        user = await _add_user(registry_session, "encrypted")
        await _provision(user.id)
        raw = await _issue(registry_session, user_id=user.id, scopes=["athlete:read"])

        seen: list[str] = []
        real = encryption.set_user_encryption_context

        def spy(user_id):
            seen.append(user_id)
            return real(user_id)

        with patch("backend.app.core.deps.set_user_encryption_context", spy):
            resp = await pat_client.get("/api/athlete", headers=_pat_headers(raw))

        assert resp.status_code == 200
        assert resp.json()["user_id"] == user.id
        assert seen == [user.id]


# ── The instance kill switch ───────────────────────────────────────────────


class TestInstanceSwitch:
    async def test_it_defaults_on(self, pat_client, registry_session):
        resp = await pat_client.get("/api/public/instance-info")
        assert resp.json()["allow_personal_access_tokens"] is True

    async def test_off_refuses_an_already_issued_token(
        self, pat_client, registry_session, own_athlete
    ):
        """Otherwise the admin has been told a comforting untruth."""
        raw = await _issue(registry_session, scopes=["activities:read"])
        assert (
            await pat_client.get("/api/activities", headers=_pat_headers(raw))
        ).status_code == 200

        registry_session.add(
            InstanceSettings(id=1, allow_personal_access_tokens=False)
        )
        await registry_session.commit()

        assert (
            await pat_client.get("/api/activities", headers=_pat_headers(raw))
        ).status_code == 401

    async def test_off_also_hides_issuance(
        self, pat_client, registry_session, session_headers
    ):
        registry_session.add(
            InstanceSettings(id=1, allow_personal_access_tokens=False)
        )
        await registry_session.commit()

        resp = await pat_client.post(
            "/api/tokens",
            json={"name": "x", "scopes": ["activities:read"]},
            headers=session_headers,
        )
        assert resp.status_code == 404

    async def test_off_is_reported_publicly_so_the_ui_can_hide_the_card(
        self, pat_client, registry_session
    ):
        registry_session.add(
            InstanceSettings(id=1, allow_personal_access_tokens=False)
        )
        await registry_session.commit()

        resp = await pat_client.get("/api/public/instance-info")
        assert resp.json()["allow_personal_access_tokens"] is False

    async def test_a_session_is_unaffected_by_the_switch(
        self, pat_client, registry_session, session_headers, own_athlete
    ):
        """The switch forbids long-lived credentials, not the app."""
        registry_session.add(
            InstanceSettings(id=1, allow_personal_access_tokens=False)
        )
        await registry_session.commit()

        resp = await pat_client.get("/api/activities", headers=session_headers)
        assert resp.status_code == 200


# ── Admin visibility ───────────────────────────────────────────────────────


class TestAdminVisibility:
    async def test_the_admin_view_never_returns_a_token_name(
        self, pat_client, registry_session, session_headers
    ):
        user = await _add_user(registry_session, "watched")
        await _issue(
            registry_session,
            user_id=user.id,
            name="garmin-sync-for-my-cardiologist",
            scopes=["activities:read"],
        )

        resp = await pat_client.get(
            f"/api/admin/users/{user.id}/tokens", headers=session_headers
        )
        assert resp.status_code == 200
        entry = resp.json()[0]
        assert "name" not in entry
        assert "garmin-sync-for-my-cardiologist" not in resp.text
        # Metadata only: enough to act on, nothing more.
        assert set(entry) == {
            "id", "scopes", "status", "expires_at",
            "last_used_at", "revoked_at", "created_at",
        }

    async def test_there_is_no_issue_on_behalf_route(self, app):
        """An admin-minted token would be indistinguishable from a user's own."""
        creating = [
            (sorted(r.methods), r.path)
            for r in app.routes
            if isinstance(r, APIRoute)
            and "tokens" in r.path
            and r.path.startswith("/api/admin")
            and r.methods & {"POST", "PUT", "PATCH"}
        ]
        assert creating == []

    async def test_admin_revocation_stops_the_token_immediately(
        self, pat_client, registry_session, session_headers
    ):
        user = await _add_user(registry_session, "runaway")
        await _provision(user.id)
        raw = await _issue(
            registry_session, user_id=user.id, scopes=["activities:read"]
        )
        token_id, _ = pat.parse_token(raw)

        assert (
            await pat_client.get("/api/activities", headers=_pat_headers(raw))
        ).status_code == 200

        with patch(
            "backend.app.services.notifications.notify_user", AsyncMock()
        ):
            resp = await pat_client.delete(
                f"/api/admin/users/{user.id}/tokens/{token_id}", headers=session_headers
            )
        assert resp.status_code == 204

        assert (
            await pat_client.get("/api/activities", headers=_pat_headers(raw))
        ).status_code == 401

    async def test_admin_revocation_reaches_the_users_inbox(
        self, pat_client, registry_session, session_headers
    ):
        """A power used on someone should be visible to them."""
        user = await _add_user(registry_session, "notified")
        raw = await _issue(registry_session, user_id=user.id, name="nightly")
        token_id, _ = pat.parse_token(raw)

        with patch(
            "backend.app.services.notifications.notify_user", AsyncMock()
        ) as notify:
            await pat_client.delete(
                f"/api/admin/users/{user.id}/tokens/{token_id}", headers=session_headers
            )

        notify.assert_awaited_once()
        notified_user, message_type, payload = notify.await_args.args
        assert notified_user == user.id
        assert message_type == "pat_revoked_by_admin"
        assert payload["token_id"] == token_id

    async def test_admin_revocation_is_audited(
        self, pat_client, registry_session, session_headers
    ):
        user = await _add_user(registry_session, "audited")
        raw = await _issue(registry_session, user_id=user.id)
        token_id, _ = pat.parse_token(raw)

        with patch("backend.app.core.audit.log") as log, patch(
            "backend.app.services.notifications.notify_user", AsyncMock()
        ):
            await pat_client.delete(
                f"/api/admin/users/{user.id}/tokens/{token_id}", headers=session_headers
            )

        record = log.warning.call_args.kwargs["extra"]
        assert record["event"] == "pat_admin_revoke"
        assert record["pat_token_id"] == token_id
        assert record["admin_user_id"] == _TEST_USER_ID


# ── Export completeness ────────────────────────────────────────────────────


class TestExportCompleteness:
    async def test_tokens_are_in_the_export_without_their_hash(
        self, pat_client, registry_session, session_headers, own_athlete
    ):
        """Issue #21 exists because an export gap was missed once already."""
        import io
        import zipfile

        await _issue(registry_session, name="live", scopes=["activities:read"])
        await _issue(registry_session, name="withdrawn", revoked=True)
        await _issue(registry_session, name="past", expires_in_days=-1)

        resp = await pat_client.get("/api/athlete/export", headers=session_headers)
        assert resp.status_code == 200

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            assert "personal_access_tokens.json" in zf.namelist()
            entries = json.loads(zf.read("personal_access_tokens.json"))

        # Active *and* dead tokens — the export is the user's complete record.
        assert {e["name"] for e in entries} == {"live", "withdrawn", "past"}
        assert {e["status"] for e in entries} == {"active", "revoked", "expired"}
        # Never the hash.
        for entry in entries:
            assert "token_hash" not in entry
            assert "token" not in entry


# ── Consent ────────────────────────────────────────────────────────────────


class TestConsentStillApplies:
    async def test_upload_is_refused_without_consent(
        self, pat_client, registry_session
    ):
        user = await _add_user(registry_session, "unconsented")
        user.consented_at = None
        user.consent_version = None
        await registry_session.commit()
        await _provision(user.id)

        raw = await _issue(
            registry_session, user_id=user.id, scopes=["activities:write"]
        )
        resp = await pat_client.post(
            "/api/activities/upload",
            files={"file": ("ride.fit", b"not-a-real-fit", "application/octet-stream")},
            headers=_pat_headers(raw),
        )
        assert resp.status_code == 403
        assert "Consent" in resp.json()["detail"]


# ── Rate limiting ──────────────────────────────────────────────────────────


class TestRateLimitKey:
    def test_an_authenticated_request_keys_on_the_user_not_the_address(self):
        from backend.app.core.limiter import principal_key

        class _Request:
            def __init__(self, user_id=None):
                self.state = type("S", (), {})()
                if user_id:
                    self.state.pat_user_id = user_id
                self.client = type("C", (), {"host": "10.0.0.1"})()
                self.headers = {}

        assert principal_key(_Request("user-abc")) == "user:user-abc"

    def test_two_tokens_of_one_user_share_a_bucket(self):
        """Keying on the token would make the limit multiplicative in a number
        nothing caps — a user may mint tokens freely."""
        from backend.app.core.limiter import principal_key

        class _Request:
            def __init__(self, user_id, token_id):
                self.state = type("S", (), {})()
                self.state.pat_user_id = user_id
                self.state.pat_token_id = token_id
                self.client = type("C", (), {"host": "10.0.0.1"})()
                self.headers = {}

        assert principal_key(_Request("u1", "token-a")) == principal_key(
            _Request("u1", "token-b")
        )

    def test_unauthenticated_traffic_still_keys_on_the_address(self):
        """The limits protecting login and signup behave exactly as before."""
        from backend.app.core.limiter import principal_key

        class _Request:
            state = type("S", (), {})()
            client = type("C", (), {"host": "10.0.0.1"})()
            headers = {}

        assert principal_key(_Request()) == "10.0.0.1"
