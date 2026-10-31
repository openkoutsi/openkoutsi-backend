"""
Hammerhead through the sync pipeline and the webhook path (issue #69).

* ``provider_sync``: Hammerhead-with-FIT ranks beside Wahoo-with-FIT, above
  Strava; a sparse listing is completed before the duplicate match, and only
  for activities the walk is actually going to work on.
* ``process_hammerhead_webhook``: unknown user, malformed payload, a real
  import, and the documented duplicate retry.
"""
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from backend.app.models.registry_orm import ProviderConnection
from backend.app.models.user_orm import Activity, ActivitySource, Athlete
from backend.app.services.hammerhead_sync import process_hammerhead_webhook
from backend.app.services.provider_sync import _source_priority, sync_provider_activities
from backend.app.services.providers.base import NormalizedActivity
from backend.app.services.providers.hammerhead import HammerheadClient
from openkoutsi.fit import summarizeWorkout

FIT_BYTES = (
    Path(__file__).parent.parent.parent
    / "testdata" / "fixtures" / "synthetic_bike_power_gps.fit"
).read_bytes()
FIT_START = summarizeWorkout(io.BytesIO(FIT_BYTES)).start_time

_USER_ID = "hh-sync-user"
_HH_USER = "hh-777"


def _sparse(ext_id: str) -> NormalizedActivity:
    # createdAt: an hour after the ride — far outside the duplicate window.
    return NormalizedActivity(
        external_id=ext_id, source="hammerhead", name="Karoo ride", sport_type=None,
        start_time=FIT_START + timedelta(hours=1), duration_s=None, distance_m=None,
        elevation_m=None, avg_power=None, avg_hr=None, max_hr=None,
        avg_speed_ms=None, avg_cadence=None,
    )


class _FakeHammerhead(HammerheadClient):
    """The real client, with the network calls stubbed at the method level."""

    def __init__(self, pages=None, fit=FIT_BYTES):
        super().__init__()
        self.pages = pages or []
        self.fit = fit
        self.completed: list[str] = []
        self.fit_downloads = 0

    async def list_activities(self, access_token, page):
        return self.pages[page - 1] if page <= len(self.pages) else []

    async def download_fit_file(self, access_token, external_id):
        cached = self._fit_cache.get(external_id)
        if cached is not None:
            return cached
        self.fit_downloads += 1
        if self.fit is None:
            return None
        self._remember_fit(external_id, self.fit)
        return self.fit

    async def get_activity_detail(self, access_token, external_id):
        return {"id": external_id, "name": "Karoo ride", "activityType": "GRAVEL"}

    async def complete_activity(self, access_token, norm):
        self.completed.append(norm.external_id)
        return await super().complete_activity(access_token, norm)


async def _athlete(session) -> Athlete:
    athlete = Athlete(global_user_id=_USER_ID, ftp_tests=[])
    session.add(athlete)
    await session.commit()
    await session.refresh(athlete)
    return athlete


def _conn() -> ProviderConnection:
    conn = MagicMock(spec=ProviderConnection)
    conn.user_id = _USER_ID
    conn.provider = "hammerhead"
    return conn


async def _sync(session, athlete, client):
    with patch(
        "backend.app.services.provider_sync.PROVIDERS",
        {"hammerhead": MagicMock(return_value=client)},
    ):
        return await sync_provider_activities(
            athlete, _conn(), session, user_id=_USER_ID, access_token="tok"
        )


# ── Priority ──────────────────────────────────────────────────────────────


def test_hammerhead_with_fit_ranks_beside_wahoo_and_above_strava():
    assert _source_priority("hammerhead", True) == _source_priority("wahoo", True) == 2
    assert _source_priority("hammerhead", True) < _source_priority("strava", False)


# ── provider_sync ─────────────────────────────────────────────────────────


class TestProviderSync:
    async def test_new_activity_is_dated_by_its_fit(self, session):
        athlete = await _athlete(session)
        client = _FakeHammerhead(pages=[[_sparse("h1")]])

        count, _ = await _sync(session, athlete, client)

        assert count == 1
        act = (await session.execute(select(Activity))).scalar_one()
        assert act.start_time.replace(tzinfo=None) == FIT_START.replace(tzinfo=None)
        assert act.sport_type == "GravelRide"
        src = (await session.execute(select(ActivitySource))).scalar_one()
        assert src.provider == "hammerhead"
        assert src.fit_file_path
        # The completion's download is the only one: the import used the cache.
        assert client.fit_downloads == 1

    async def test_completion_matches_a_strava_copy_and_takes_it_over(self, session):
        athlete = await _athlete(session)
        strava_act = Activity(
            athlete_id=athlete.id, name="Strava copy", sport_type="Ride",
            start_time=FIT_START + timedelta(minutes=1), status="processed",
        )
        session.add(strava_act)
        await session.flush()
        session.add(ActivitySource(
            activity_id=strava_act.id, provider="strava", external_id="s1",
            streams_fetched_at=datetime.now(timezone.utc),
        ))
        await session.commit()

        count, _ = await _sync(session, athlete, _FakeHammerhead(pages=[[_sparse("h1")]]))

        assert count == 1
        activities = (await session.execute(select(Activity))).scalars().all()
        assert len(activities) == 1, "the Karoo ride must attach, not duplicate"
        providers = {
            s.provider for s in (await session.execute(select(ActivitySource))).scalars()
        }
        assert providers == {"strava", "hammerhead"}

    async def test_already_imported_activities_are_never_completed(self, session):
        athlete = await _athlete(session)
        await _sync(session, athlete, _FakeHammerhead(pages=[[_sparse("h1")]]))

        again = _FakeHammerhead(pages=[[_sparse("h1")]])
        count, _ = await _sync(session, athlete, again)

        assert count == 0
        assert again.completed == []
        assert again.fit_downloads == 0

    async def test_activity_without_a_fit_is_skipped_not_imported(self, session):
        athlete = await _athlete(session)
        count, _ = await _sync(
            session, athlete, _FakeHammerhead(pages=[[_sparse("h1")]], fit=None)
        )
        assert count == 0
        assert (await session.execute(select(Activity))).scalars().all() == []


# ── process_hammerhead_webhook ────────────────────────────────────────────


def _session_cm(session):
    class _CM:
        def __call__(self):
            return self

        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            pass

    return _CM()


async def _seed(session, registry_session):
    athlete = await _athlete(session)
    registry_session.add(ProviderConnection(
        user_id=_USER_ID, provider="hammerhead", provider_athlete_id=_HH_USER,
        access_token="tok", refresh_token="rt",
    ))
    await registry_session.flush()
    return athlete


def _webhook_patches(session, registry_session, client):
    return (
        patch("backend.app.db.registry._RegistrySessionLocal",
              new=_session_cm(registry_session)),
        patch("backend.app.db.user_session.get_user_session_factory",
              return_value=_session_cm(session)),
        patch("backend.app.db.user_session.init_user_db", new=AsyncMock()),
        patch("backend.app.services.hammerhead_sync.ensure_fresh_token",
              new=AsyncMock(return_value="tok")),
        patch("backend.app.services.hammerhead_sync.HammerheadClient",
              return_value=client),
        patch("backend.app.services.metrics_engine.recalculate_from", new=AsyncMock()),
    )


class TestWebhook:
    @pytest.mark.parametrize("payload", [
        {}, {"activityId": "h1"}, {"userId": _HH_USER}, ["not", "a", "dict"],
    ])
    async def test_malformed_payload_is_ignored(self, session, payload):
        await process_hammerhead_webhook(payload)
        assert (await session.execute(select(Activity))).scalars().all() == []

    async def test_unknown_user_is_ignored(self, session, registry_session):
        await _seed(session, registry_session)
        client = _FakeHammerhead()
        p = _webhook_patches(session, registry_session, client)
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            await process_hammerhead_webhook({"activityId": "h1", "userId": "someone-else"})
        assert client.completed == []
        assert (await session.execute(select(Activity))).scalars().all() == []

    async def test_event_imports_the_activity(self, session, registry_session):
        await _seed(session, registry_session)
        client = _FakeHammerhead()
        p = _webhook_patches(session, registry_session, client)
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            await process_hammerhead_webhook({"activityId": "h1", "userId": _HH_USER})

        act = (await session.execute(select(Activity))).scalar_one()
        assert act.start_time.replace(tzinfo=None) == FIT_START.replace(tzinfo=None)
        assert act.status == "processed"
        src = (await session.execute(select(ActivitySource))).scalar_one()
        assert (src.provider, src.external_id) == ("hammerhead", "h1")

    async def test_duplicate_retry_is_a_no_op(self, session, registry_session):
        await _seed(session, registry_session)
        first = _FakeHammerhead()
        p = _webhook_patches(session, registry_session, first)
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            await process_hammerhead_webhook({"activityId": "h1", "userId": _HH_USER})

        retry = _FakeHammerhead()
        p = _webhook_patches(session, registry_session, retry)
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            await process_hammerhead_webhook({"activityId": "h1", "userId": _HH_USER})

        assert retry.completed == [], "no provider call for an activity already here"
        assert len((await session.execute(select(Activity))).scalars().all()) == 1
        assert len((await session.execute(select(ActivitySource))).scalars().all()) == 1

    async def test_event_without_a_fit_is_dropped_quietly(self, session, registry_session):
        await _seed(session, registry_session)
        client = _FakeHammerhead(fit=None)
        p = _webhook_patches(session, registry_session, client)
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            await process_hammerhead_webhook({"activityId": "h1", "userId": _HH_USER})
        assert (await session.execute(select(Activity))).scalars().all() == []


# ── Review fixes: nothing that cannot recover may cost a ride ─────────────


class _Fileless(_FakeHammerhead):
    """Activities whose id starts with ``nofile`` have no FIT; ``junk`` ones an unreadable one."""

    async def download_fit_file(self, access_token, external_id):
        if external_id.startswith("nofile"):
            self.fit_downloads += 1
            return None
        if external_id.startswith("junk"):
            self.fit_downloads += 1
            return b"not a fit file"
        return await super().download_fit_file(access_token, external_id)


class TestFilelessActivities:
    async def test_a_run_of_fileless_activities_is_not_an_outage(self, session):
        """More than the outage limit of no-file activities in a row must not stop the walk."""
        from backend.app.models.user_orm import ProviderSyncState
        from backend.app.services.provider_sync import _MAX_CONSECUTIVE_UNRESOLVED

        athlete = await _athlete(session)
        listing = [_sparse(f"nofile-{i}") for i in range(_MAX_CONSECUTIVE_UNRESOLVED + 5)]
        listing += [_sparse(f"junk-{i}") for i in range(3)]
        listing.append(_sparse("good"))

        count, _ = await _sync(session, athlete, _Fileless(pages=[listing]))

        assert count == 1, "the ride after the fileless run must still be imported"
        src = (await session.execute(select(ActivitySource))).scalar_one()
        assert src.external_id == "good"
        state = (await session.execute(select(ProviderSyncState))).scalar_one()
        assert state.stop_reason != ProviderSyncState.STOP_PROVIDER_OUTAGE

    async def test_an_unreadable_fit_is_a_lookup_error(self):
        with pytest.raises(LookupError):
            await _Fileless().complete_activity("tok", _sparse("junk-1"))


class TestWebhookRobustness:
    async def test_one_karoo_account_on_two_users_imports_for_both(
        self, session, registry_session
    ):
        await _seed(session, registry_session)
        second = Athlete(global_user_id="hh-second-user", ftp_tests=[])
        session.add(second)
        await session.commit()
        registry_session.add(ProviderConnection(
            user_id="hh-second-user", provider="hammerhead",
            provider_athlete_id=_HH_USER, access_token="tok2", refresh_token="rt2",
        ))
        await registry_session.flush()

        p = _webhook_patches(session, registry_session, _FakeHammerhead())
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            await process_hammerhead_webhook({"activityId": "h1", "userId": _HH_USER})

        owners = {
            a.athlete_id for a in (await session.execute(select(Activity))).scalars()
        }
        assert len(owners) == 2, "both linked users get the ride"

    async def test_two_activities_in_the_window_attach_to_the_closest(
        self, session, registry_session
    ):
        """A ride Strava split in two must not make the webhook raise and retire."""
        athlete = await _seed(session, registry_session)
        far = Activity(athlete_id=athlete.id, name="far", sport_type="Ride",
                       start_time=FIT_START + timedelta(minutes=4), status="processed")
        near = Activity(athlete_id=athlete.id, name="near", sport_type="Ride",
                        start_time=FIT_START + timedelta(seconds=30), status="processed")
        session.add_all([far, near])
        await session.flush()
        session.add_all([
            ActivitySource(activity_id=far.id, provider="strava", external_id="s-far",
                           streams_fetched_at=datetime.now(timezone.utc)),
            ActivitySource(activity_id=near.id, provider="strava", external_id="s-near",
                           streams_fetched_at=datetime.now(timezone.utc)),
        ])
        await session.commit()

        p = _webhook_patches(session, registry_session, _FakeHammerhead())
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            await process_hammerhead_webhook({"activityId": "h1", "userId": _HH_USER})

        hh = (await session.execute(
            select(ActivitySource).where(ActivitySource.provider == "hammerhead")
        )).scalar_one()
        assert hh.activity_id == near.id
        assert len((await session.execute(select(Activity))).scalars().all()) == 2
