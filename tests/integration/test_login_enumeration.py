"""Login must cost the same whether or not the account exists (issue #102, F-06).

``if user is None or not await verify_password_async(...)`` short-circuits: a
known identifier paid for bcrypt, an unknown one returned immediately. The
finding measured the difference over six requests each:

    existing user + wrong password :  280.3 ms
    non-existent user              :    4.3 ms
    ratio                          :  66×

That is a reliable "does this account exist?" answer, and it gives away what
the rest of this surface works to hide — ``signup`` returns a fixed
acknowledgement so a taken address cannot be detected, ``request-password-reset``
does the same, and the ``IntegrityError`` path carries a comment explaining that
a concurrent signup must collapse to the generic response "so the response stays
uniform (no enumeration)".

The absolute numbers are machine- and cost-dependent, so these tests do not
reproduce them. What is actually being asserted is the structural property the
numbers were evidence of: **both branches run the verification**. The timing
test below makes that observable without depending on real bcrypt cost, by
replacing the verification with something slow by a known amount and asserting
the unknown-identifier path waits for it too.
"""
import json
import time
import uuid
from unittest.mock import patch

import pytest

from backend.app.core.auth import dummy_password_hash, hash_password, verify_password
from backend.app.models.registry_orm import User

_PREFIX = "/api/auth"
_GOOD_PW = "Testpass1234"
_WRONG_PW = "Wrongpass1234"

# How long the stand-in verification takes. Only lower bounds are asserted
# against it, which is what keeps the test off the flaky end of timing tests.
_VERIFY_COST = 0.2


async def _add_user(registry_session, username: str = "known-user") -> User:
    user = User(
        id=str(uuid.uuid4()),
        username=username,
        password_hash=hash_password(_GOOD_PW),
        roles=json.dumps(["user"]),
    )
    registry_session.add(user)
    await registry_session.commit()
    return user


async def _login(client, username: str, password: str = _WRONG_PW):
    return await client.post(
        f"{_PREFIX}/login", json={"username": username, "password": password}
    )


class TestBothBranchesVerify:
    async def test_unknown_identifier_still_runs_the_verification(
        self, client, registry_session
    ):
        """The structural fix: no short-circuit on a missing account."""
        calls = []

        async def _spy(plain, hashed):
            calls.append(hashed)
            return False

        with patch("backend.app.api.auth.verify_password_async", _spy):
            resp = await _login(client, "no-such-account")

        assert resp.status_code == 401
        assert len(calls) == 1, "unknown identifier skipped the password check"

    async def test_known_identifier_verifies_against_its_own_hash(
        self, client, registry_session
    ):
        user = await _add_user(registry_session)
        calls = []

        async def _spy(plain, hashed):
            calls.append(hashed)
            return False

        with patch("backend.app.api.auth.verify_password_async", _spy):
            await _login(client, "known-user")

        assert calls == [user.password_hash]

    async def test_unknown_identifier_verifies_against_the_dummy(
        self, client, registry_session
    ):
        calls = []

        async def _spy(plain, hashed):
            calls.append(hashed)
            return False

        with patch("backend.app.api.auth.verify_password_async", _spy):
            await _login(client, "no-such-account")

        assert calls == [dummy_password_hash()]


class TestTimingIsUniform:
    """The measurement, made deterministic.

    Real bcrypt at the test suite's reduced cost is too fast for a ratio to
    mean anything, so the verification is replaced with a known delay. A
    branch that skips it returns immediately and fails the lower bound; a
    branch that runs it cannot.
    """

    @pytest.fixture
    def slow_verify(self):
        async def _slow(_plain, _hashed):
            time.sleep(_VERIFY_COST)
            return False

        with patch("backend.app.api.auth.verify_password_async", _slow):
            yield

    async def _elapsed(self, client, username: str) -> float:
        start = time.perf_counter()
        resp = await _login(client, username)
        assert resp.status_code == 401
        return time.perf_counter() - start

    async def test_unknown_identifier_pays_the_same_cost(
        self, client, registry_session, slow_verify
    ):
        await _add_user(registry_session)

        known = await self._elapsed(client, "known-user")
        unknown = await self._elapsed(client, "no-such-account")

        assert known >= _VERIFY_COST
        assert unknown >= _VERIFY_COST, (
            f"unknown identifier answered in {unknown:.3f}s against "
            f"{known:.3f}s for a known one — the account oracle is open"
        )

    async def test_unverified_email_pays_the_same_cost(
        self, client, registry_session, slow_verify
    ):
        """An account that exists but cannot log in yet must not stand out.

        Login accepts a *verified* email as an identifier. An unverified one
        finds no row and takes the same path as a nonexistent address, so it
        has to cost the same too — otherwise "this address is registered but
        unconfirmed" is readable from the clock.
        """
        user = await _add_user(registry_session, username="pending")
        user.email = "pending@example.com"
        user.email_verified_at = None
        await registry_session.commit()

        elapsed = await self._elapsed(client, "pending@example.com")
        assert elapsed >= _VERIFY_COST


class TestResponsesAreIdentical:
    async def test_same_status_and_detail_for_both(self, client, registry_session):
        """Timing is not the only channel — the body must not differ either."""
        await _add_user(registry_session)

        wrong_password = await _login(client, "known-user")
        no_account = await _login(client, "no-such-account")

        assert wrong_password.status_code == no_account.status_code == 401
        assert wrong_password.json() == no_account.json()


class TestTheDummyHash:
    def test_is_a_real_bcrypt_hash(self):
        """Not a placeholder string — it has to cost what a real hash costs."""
        assert dummy_password_hash().startswith("$2b$")

    def test_is_stable_within_the_process(self):
        """One hash per process, not one per login."""
        assert dummy_password_hash() is dummy_password_hash()

    def test_never_matches_a_supplied_password(self):
        for candidate in ("", _GOOD_PW, _WRONG_PW, "password", dummy_password_hash()):
            assert not verify_password(candidate, dummy_password_hash())

    async def test_a_nonexistent_account_cannot_be_logged_into(
        self, client, registry_session
    ):
        """Belt and braces: even a verification that returned True must not
        authenticate an account that does not exist."""

        async def _always_true(_plain, _hashed):
            return True

        with patch("backend.app.api.auth.verify_password_async", _always_true):
            resp = await _login(client, "no-such-account", _GOOD_PW)

        assert resp.status_code == 401
