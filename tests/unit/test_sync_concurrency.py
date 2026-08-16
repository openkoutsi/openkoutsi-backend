"""Concurrency properties of the provider sync path (issue #50).

These are the two races that are reachable **inside a single process** today, so
they are bugs on the current one-box deployment rather than replica-safety
hypotheticals:

* two syncs for one connection rotating the OAuth tokens at the same time,
* two syncs importing the same real-world workout at the same time, and
* a sync and a manual FIT upload racing for the same ride.

The last of those is the same find-or-create as the second, reached through a
different door: ``upload_activity`` does a ±5-minute window query and then
inserts, so it needs the same lock and the same lease as the sync path it
competes with (issue #36's review). The barrier harness below is what turns
"these two might interleave" into "they certainly do", and it is what makes a
test of that guard mean anything.

Both are tested against **file-based** SQLite. An in-memory URL gives SQLAlchemy
a ``StaticPool``, so every session would share one connection and serialise for
free — the race would not be reproduced, only hidden. ``test_db_concurrency.py``
takes the same approach for the same reason.
"""
import asyncio
import contextlib
import io
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import backend.app.models.registry_orm  # noqa: F401 — populate RegistryBase.metadata
import backend.app.models.user_orm  # noqa: F401 — populate UserBase.metadata
from backend.app.db import leases
from backend.app.db.base import RegistryBase, UserBase, _set_wal_mode
from backend.app.models.registry_orm import ProviderConnection, User
from backend.app.models.user_orm import (
    Activity,
    ActivitySource,
    ActivityStream,
    Athlete,
    SyncLease,
)
from backend.app.services.provider_sync import ensure_fresh_token, sync_provider_activities
from openkoutsi.fit import getStartTime
from backend.app.services.providers.base import NormalizedActivity

_USER_ID = "concurrency-user"
_TTL = timedelta(seconds=30)


def _engine(db_path):
    """An engine matching production: WAL, a real pool, a busy timeout."""
    eng = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=False,
        pool_size=3,
        max_overflow=2,
        connect_args={"timeout": 30},
    )
    event.listen(eng.sync_engine, "connect", _set_wal_mode)
    return eng


# ── Token rotation ──────────────────────────────────────────────────────────


class _RotatingProvider:
    """A provider that revokes the old refresh token when it issues a new one.

    This is Wahoo's actual behaviour (see ``_REFRESH_LOOKAHEAD`` in
    ``provider_sync``), and it is what makes a lost race permanent rather than
    merely wasteful: presenting a superseded refresh token does not just fail,
    it fails *forever*.
    """

    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.presented: list[str] = []
        self.live_refresh_token = "refresh-1"
        self._issued = 1

    async def refresh_access_token(self, refresh_token: str) -> dict:
        self.presented.append(refresh_token)
        if refresh_token != self.live_refresh_token:
            raise RuntimeError("refresh token has been revoked")
        # The window in which a second caller could interleave.
        await asyncio.sleep(self.delay)
        self._issued += 1
        self.live_refresh_token = f"refresh-{self._issued}"
        return {
            "access_token": f"access-{self._issued}",
            "refresh_token": self.live_refresh_token,
            "expires_at": int(
                (datetime.now(timezone.utc) + timedelta(hours=6)).timestamp()
            ),
        }


@pytest.fixture
async def registry_db(tmp_path):
    """A file-based registry DB holding one expiring wahoo connection."""
    engine = _engine(tmp_path / "registry.db")
    async with engine.begin() as conn:
        await conn.run_sync(RegistryBase.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(User(id=_USER_ID, username="conc", password_hash="x", roles='["user"]'))
        session.add(
            ProviderConnection(
                id="conn-1",
                user_id=_USER_ID,
                provider="wahoo",
                access_token="access-1",
                refresh_token="refresh-1",
                token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            )
        )
        await session.commit()

    yield factory
    await engine.dispose()


async def _fresh_token_from_own_session(factory) -> str:
    """One caller, with its own session — a second worker, or a second process."""
    async with factory() as session:
        conn = (
            await session.execute(
                select(ProviderConnection).where(ProviderConnection.id == "conn-1")
            )
        ).scalar_one()
        return await ensure_fresh_token(conn, session)


class TestConcurrentTokenRefresh:
    async def test_two_concurrent_syncs_rotate_exactly_once(self, registry_db):
        provider = _RotatingProvider()

        with patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": provider}):
            first, second = await asyncio.gather(
                _fresh_token_from_own_session(registry_db),
                _fresh_token_from_own_session(registry_db),
            )

        # One rotation, so the provider was asked exactly once and was never
        # handed a token it had already revoked.
        assert provider.presented == ["refresh-1"]

        # Both callers came away with the rotation's access token — the loser
        # observed the winner's result rather than the token it started with.
        assert first == "access-2"
        assert second == "access-2"

    async def test_the_loser_does_not_store_a_revoked_refresh_token(self, registry_db):
        provider = _RotatingProvider()

        with patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": provider}):
            await asyncio.gather(
                _fresh_token_from_own_session(registry_db),
                _fresh_token_from_own_session(registry_db),
            )

        async with registry_db() as session:
            stored = (
                await session.execute(
                    select(ProviderConnection).where(ProviderConnection.id == "conn-1")
                )
            ).scalar_one()

        # The stored refresh token is the live one. Before the lock, the loser's
        # write could land second and leave `refresh-1` here — already revoked,
        # and unrecoverable without the user reconnecting by hand.
        assert stored.refresh_token == provider.live_refresh_token == "refresh-2"
        assert stored.access_token == "access-2"
        assert stored.refresh_lock_until is None

    async def test_five_concurrent_syncs_still_rotate_once(self, registry_db):
        provider = _RotatingProvider()

        with patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": provider}):
            tokens = await asyncio.gather(
                *[_fresh_token_from_own_session(registry_db) for _ in range(5)]
            )

        assert provider.presented == ["refresh-1"]
        assert set(tokens) == {"access-2"}


# ── Activity de-duplication ─────────────────────────────────────────────────


def _norm(ext_id: str, source: str, start_time: datetime) -> NormalizedActivity:
    return NormalizedActivity(
        external_id=ext_id,
        source=source,
        name="Morning Ride",
        sport_type="Ride",
        start_time=start_time,
        duration_s=3600,
        distance_m=50_000.0,
        elevation_m=500.0,
        avg_power=None,
        avg_hr=None,
        max_hr=None,
        avg_speed_ms=14.0,
        avg_cadence=None,
    )


def _provider_returning(norm: NormalizedActivity, *, fit: bytes | None = None) -> MagicMock:
    """A provider client that offers exactly one activity.

    ``fit`` decides the source priority the import will compute: bytes make this
    a FIT-capable provider (priority 2, ahead of Strava's 3), which is what sends
    the import down the repopulate path rather than the record-the-source one.
    """
    client = MagicMock()
    client.list_activities = AsyncMock(side_effect=[[norm], []])
    if fit is None:
        client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
    else:
        client.download_fit_file = AsyncMock(return_value=fit)
    client.get_activity_streams = AsyncMock(return_value={})
    return MagicMock(return_value=client)


@pytest.fixture
async def user_db(tmp_path):
    """A file-based per-user DB holding one athlete."""
    engine = _engine(tmp_path / "user.db")
    async with engine.begin() as conn:
        await conn.run_sync(UserBase.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(Athlete(id="athlete-1", global_user_id=_USER_ID, ftp_tests=[]))
        await session.commit()

    yield factory
    await engine.dispose()


async def _sync_from(
    factory, provider_name: str, norm: NormalizedActivity, *, fit: bytes | None = None
) -> None:
    """Import one activity, on this caller's own session."""
    async with factory() as session:
        athlete = (
            await session.execute(select(Athlete).where(Athlete.id == "athlete-1"))
        ).scalar_one()
        conn = MagicMock(spec=ProviderConnection)
        conn.user_id = _USER_ID
        conn.provider = provider_name
        with patch(
            "backend.app.services.provider_sync.PROVIDERS",
            {provider_name: _provider_returning(norm, fit=fit)},
        ):
            await sync_provider_activities(
                athlete, conn, session, user_id=_USER_ID, access_token="tok"
            )


def _both_callers_inside_at_once(parties: int):
    """Replace the in-process lock with a barrier of ``parties``.

    The stand-in does the opposite of the lock it replaces: instead of keeping
    callers out, it holds each one at the door until every other has arrived, and
    then lets them all through together. That turns "these two might interleave
    at the worst possible point" into "they certainly do", which is the only way
    a guard below it can be shown to work.
    """
    barrier = asyncio.Barrier(parties)

    class _Barrier:
        async def __aenter__(self):
            # Bounded, so a guard that deadlocks fails the test instead of
            # hanging the suite.
            await asyncio.wait_for(barrier.wait(), timeout=10)
            return None

        async def __aexit__(self, *exc):
            return False

    return patch(
        "backend.app.services.provider_sync._get_activity_lock",
        lambda *args: _Barrier(),
    )


async def _activity_and_source_counts(factory) -> tuple[int, int]:
    async with factory() as session:
        activities = (await session.execute(select(Activity))).scalars().all()
        sources = (await session.execute(select(ActivitySource))).scalars().all()
    return len(activities), len(sources)


class TestConcurrentDuplicateImport:
    async def test_same_workout_from_two_providers_yields_one_activity(self, user_db):
        start = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)

        await asyncio.gather(
            _sync_from(user_db, "strava", _norm("s-1", "strava", start)),
            # Three minutes apart: inside the ±5-minute dedup window, so this is
            # the same ride arriving from a second provider.
            _sync_from(
                user_db, "wahoo", _norm("w-1", "wahoo", start + timedelta(minutes=3))
            ),
        )

        activities, sources = await _activity_and_source_counts(user_db)
        assert activities == 1
        assert sources == 2

    async def test_still_one_activity_without_the_in_process_lock(self, user_db):
        """The test that proves the *database* is doing the work.

        Two things happen here. The ``asyncio.Lock`` is replaced, so the guard
        that used to be the only one is gone — which is also exactly what a
        second process sees, since that lock never spanned one. And what replaces
        it is a *barrier*, so both callers are guaranteed to be inside the
        find-or-create at the same moment rather than merely likely to be.

        Without the barrier this test passes whether or not the database guard
        exists, because the two coroutines happen not to interleave at the one
        point where it matters. With it, only a real cross-caller guard can keep
        the count at one.
        """
        start = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)

        with _both_callers_inside_at_once(2):
            await asyncio.gather(
                _sync_from(user_db, "strava", _norm("s-1", "strava", start)),
                _sync_from(
                    user_db, "wahoo", _norm("w-1", "wahoo", start + timedelta(minutes=3))
                ),
            )

        activities, sources = await _activity_and_source_counts(user_db)
        assert activities == 1
        assert sources == 2

    async def test_distinct_workouts_are_not_merged(self, user_db):
        """The lease must not turn de-duplication into over-merging."""
        start = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)

        await asyncio.gather(
            _sync_from(user_db, "strava", _norm("s-1", "strava", start)),
            _sync_from(
                user_db, "wahoo", _norm("w-1", "wahoo", start + timedelta(hours=4))
            ),
        )

        activities, sources = await _activity_and_source_counts(user_db)
        assert activities == 2
        assert sources == 2


# ── Failure inside the guarded section ──────────────────────────────────────


class TestAFailedImportCommitsNothing:
    """Releasing the lease must not publish the work that failed.

    ``leases.release`` commits, and ``commit`` flushes first, so the tidy-up at
    the end of a failed section is in a position to make that section's partial
    writes durable. The attach path is where that bites: by the time
    ``_fill_from_source`` does its unguarded disk and parse work,
    ``_repopulate_activity`` has already deleted every stream and best belonging
    to the activity it is rebuilding.
    """

    async def test_a_failure_after_the_deletes_leaves_the_streams_intact(self, user_db):
        start = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)

        # An activity already imported from Strava, with a stream to lose.
        async with user_db() as session:
            activity = Activity(
                id="act-1", athlete_id="athlete-1", name="Morning Ride",
                start_time=start, duration_s=3600, status="processed",
            )
            session.add(activity)
            await session.flush()
            session.add(
                ActivitySource(activity_id="act-1", provider="strava", external_id="s-1")
            )
            session.add(
                ActivityStream(
                    id="stream-1", activity_id="act-1", stream_type="power",
                    data=[100, 200, 300],
                )
            )
            await session.commit()

        # Wahoo arrives with a FIT file, so it outranks Strava and repopulates —
        # and then the rebuild blows up partway through, after the deletes.
        with patch(
            "backend.app.services.provider_sync._fill_from_source",
            AsyncMock(side_effect=OSError("No space left on device")),
        ):
            with pytest.raises(OSError):
                await _sync_from(
                    user_db,
                    "wahoo",
                    _norm("w-1", "wahoo", start + timedelta(minutes=2)),
                    fit=b"a FIT file, which outranks Strava's streams",
                )

        async with user_db() as reader:
            streams = (await reader.execute(select(ActivityStream))).scalars().all()
            activities = (await reader.execute(select(Activity))).scalars().all()

        # The deletion was rolled back with the rest of the failed section.
        assert [s.stream_type for s in streams] == ["power"]
        assert len(activities) == 1

    async def test_the_lease_is_free_again_after_a_failed_import(self, user_db):
        """So the next activity in the same sync does not wait out the TTL."""
        start = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)

        with patch(
            "backend.app.services.provider_sync._populate_activity",
            AsyncMock(side_effect=OSError("No space left on device")),
        ):
            with pytest.raises(OSError):
                await _sync_from(user_db, "strava", _norm("s-1", "strava", start))

        async with user_db() as session:
            lease = (
                await session.execute(
                    select(SyncLease).where(
                        SyncLease.name == "activity-create:athlete-1"
                    )
                )
            ).scalar_one()
        assert lease.holder is None


class _FailsThenWorksProvider:
    """Fails the first rotation, succeeds on the next.

    Stands in for a transient provider fault — a 5xx, a connection timeout, a
    malformed token payload. The refresh token is *not* revoked by a failed
    attempt, which is what makes a second try worth making.
    """

    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.presented: list[str] = []

    async def refresh_access_token(self, refresh_token: str) -> dict:
        self.presented.append(refresh_token)
        await asyncio.sleep(self.delay)
        if len(self.presented) == 1:
            raise RuntimeError("provider returned 503")
        return {
            "access_token": "access-2",
            "refresh_token": "refresh-2",
            "expires_at": int(
                (datetime.now(timezone.utc) + timedelta(hours=6)).timestamp()
            ),
        }


class TestTheWaiterIsNotTakenDownWithTheWinner:
    async def test_the_waiter_rotates_itself_when_the_winners_attempt_fails(
        self, registry_db
    ):
        """Before the claim, both callers refreshed independently, so one failing
        did not cost the other its token. Waiting must not undo that."""
        provider = _FailsThenWorksProvider()

        with patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": provider}):
            results = await asyncio.gather(
                _fresh_token_from_own_session(registry_db),
                _fresh_token_from_own_session(registry_db),
                return_exceptions=True,
            )

        # The winner's caller sees the provider error — that is its own call
        # failing, and swallowing it would be worse.
        errors = [r for r in results if isinstance(r, BaseException)]
        tokens = [r for r in results if not isinstance(r, BaseException)]
        assert len(errors) == 1
        assert len(tokens) == 1

        # The waiter did not simply hand back the expiring token it started with.
        assert tokens == ["access-2"]
        assert provider.presented == ["refresh-1", "refresh-1"]

        async with registry_db() as session:
            stored = (
                await session.execute(
                    select(ProviderConnection).where(ProviderConnection.id == "conn-1")
                )
            ).scalar_one()
        assert stored.access_token == "access-2"
        assert stored.refresh_lock_until is None


# ── Upload racing a sync ─────────────────────────────────────────────────────


def _fit_bytes() -> bytes:
    """A real FIT fixture, since `upload_activity` validates the magic bytes."""
    from pathlib import Path as _Path

    fixtures = _Path(__file__).parent.parent.parent / "testdata" / "fixtures"
    for candidate in sorted(fixtures.glob("*.fit")):
        return candidate.read_bytes()
    pytest.skip("no FIT fixture available")


@contextlib.contextmanager
def _upload_environment(tmp_path):
    """Globals the upload handler reads, set once for the whole test.

    Both of these are process-wide, so a concurrent test must not have each
    caller set and restore them: the first to finish would put the limiter back
    while the second is still inside slowapi's wrapper, which reads
    ``self.enabled`` again on the way out and raises on a `request` it never
    bound. That is a bug in the test, but it is the kind that reads as a bug in
    the code under test, so it is worth naming.
    """
    from backend.app.core.config import settings
    from backend.app.core.limiter import limiter

    enabled, data_dir = limiter.enabled, settings.data_dir
    limiter.enabled = False
    settings.data_dir = str(tmp_path)
    try:
        yield
    finally:
        limiter.enabled = enabled
        settings.data_dir = data_dir


async def _upload(factory, payload: bytes) -> None:
    """Call the upload endpoint's handler on this caller's own session.

    Must run inside :func:`_upload_environment`.
    """
    from fastapi import BackgroundTasks
    from starlette.datastructures import Headers, UploadFile as StarletteUploadFile

    from backend.app.api.activities import upload_activity

    from backend.app.core.auth import UserContext

    async with factory() as session:
        athlete = (
            await session.execute(select(Athlete).where(Athlete.id == "athlete-1"))
        ).scalar_one()
        ctx = UserContext(user_id=_USER_ID, roles=["user"])

        upload = StarletteUploadFile(
            file=io.BytesIO(payload),
            filename="ride.fit",
            headers=Headers({"content-type": "application/octet-stream"}),
        )
        await upload_activity(
            request=MagicMock(),
            background_tasks=BackgroundTasks(),
            file=upload,
            ctx_athlete=(ctx, session, athlete),
        )


class TestUploadRacingASync:
    """The manual upload holds the same guards as the paths it competes with.

    Before issue #36's review this path did the ±5-minute check and the insert
    holding neither, so a Wahoo webhook or a Strava backfill landing while an
    athlete uploaded the same ride could have both writers see an empty window
    and each create an activity.

    These do not take the barrier approach the sync tests above use, and the
    reason is a property of this path rather than a shortcoming of the test: an
    activity created by an upload has ``start_time = NULL`` until the background
    task has parsed the FIT, so a *second* upload's window query cannot see it
    however the two interleave — that overlap is settled afterwards, by the
    merge in ``_bg_process_and_recalculate``. What the lease buys is exclusion
    against the **sync** path, which inserts rows that do carry a start time.
    So the property worth pinning is that the section is held under the lease at
    all, which is what these assert directly.
    """

    _LEASE = "activity-create:athlete-1"

    async def test_the_upload_waits_for_the_activity_create_lease(
        self, user_db, tmp_path
    ):
        payload = _fit_bytes()

        async with user_db() as holder:
            token = await leases.acquire(
                holder, SyncLease, self._LEASE, ttl=_TTL, wait=1.0
            )
            assert token is not None, "could not set the test up"

            with _upload_environment(tmp_path):
                upload = asyncio.create_task(_upload(user_db, payload))
                # Long enough for an unguarded upload to have finished — it does
                # no IO beyond writing one small file and two inserts.
                await asyncio.sleep(0.25)
                assert not upload.done(), (
                    "the upload created an activity while another writer held "
                    "the activity-create lease"
                )

                await leases.release(holder, SyncLease, self._LEASE, token)
                await asyncio.wait_for(upload, timeout=10)

        activities, sources = await _activity_and_source_counts(user_db)
        assert (activities, sources) == (1, 1)

    async def test_an_upload_attaches_to_a_synced_ride_rather_than_duplicating_it(
        self, user_db, tmp_path
    ):
        """The outcome the lease protects, with the ordering it can act on."""
        payload = _fit_bytes()
        start = getStartTime(io.BytesIO(payload))
        assert start is not None

        # The sync lands first, so its row — which carries a start time — is
        # what the upload's window query sees.
        await _sync_from(user_db, "strava", _norm("s-1", "strava", start))
        with _upload_environment(tmp_path):
            await _upload(user_db, payload)

        activities, sources = await _activity_and_source_counts(user_db)
        assert activities == 1, "the upload and the sync described the same ride"
        assert sources == 2, "and both are recorded against it"

    async def test_the_lease_is_free_again_after_an_upload(self, user_db, tmp_path):
        """So the next writer does not wait out the TTL."""
        with _upload_environment(tmp_path):
            await _upload(user_db, _fit_bytes())

        async with user_db() as session:
            token = await leases.acquire(
                session, SyncLease, self._LEASE, ttl=_TTL, wait=0.5
            )
        assert token is not None
