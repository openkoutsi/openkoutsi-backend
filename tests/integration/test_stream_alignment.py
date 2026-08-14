"""Streams reach the database on one clock, with gaps intact (issue #76).

The unit tests pin the resampler; these pin the whole pipeline around it —
upload, parse, persist, serve — because the failure mode this replaces was
invisible at every one of those boundaries. The old parser produced dense lists
that serialised fine, stored fine and rendered fine; the only symptom was that a
heart-rate sample and a power sample at the same index came from different
moments of the ride.

``synthetic_bike_hr_dropout.fit`` carries both kinds of hole a real file has:
a heart-rate dropout while everything else keeps recording, and a device pause
where no record is written at all. See ``testdata/fixtures/README.md``.
"""
import json
from pathlib import Path

from sqlalchemy import select

from backend.app.models.user_orm import Activity, ActivitySource, ActivityStream, Athlete

from scripts.generate_synthetic_fit_fixtures import (
    HR_DROPOUT_END_S,
    HR_DROPOUT_START_S,
    RECORD_GAP_END_S,
    RECORD_GAP_START_S,
)

_FIXTURE = (
    Path(__file__).parent.parent.parent
    / "testdata" / "fixtures" / "synthetic_bike_hr_dropout.fit"
)

# The fixture's full elapsed span, and the seconds a device actually recorded.
_ELAPSED_S = 600
_RECORDED_S = _ELAPSED_S - (RECORD_GAP_END_S - RECORD_GAP_START_S)

# Channels the device records, as opposed to the ones openkoutsi derives.
_RECORDED_CHANNELS = ("power", "heartrate", "cadence", "speed", "altitude")


async def _upload_and_process(client, auth_headers, session) -> str:
    """Push the dropout fixture through the real upload → process path."""
    await client.patch("/api/athlete", json={"ftp": 280}, headers=auth_headers)

    with open(_FIXTURE, "rb") as f:
        resp = await client.post(
            "/api/activities/upload",
            files={"file": ("dropout.fit", f, "application/octet-stream")},
            headers=auth_headers,
        )
    assert resp.status_code == 201
    activity_id = resp.json()["id"]

    activity = (
        await session.execute(select(Activity).where(Activity.id == activity_id))
    ).scalar_one()
    src = (
        await session.execute(
            select(ActivitySource).where(
                ActivitySource.activity_id == activity_id,
                ActivitySource.provider == "upload",
            )
        )
    ).scalar_one()
    athlete = (
        await session.execute(select(Athlete).where(Athlete.id == activity.athlete_id))
    ).scalar_one()

    from backend.app.services.fit_processor import process_fit_file

    await process_fit_file(src.fit_file_path, athlete, activity, session)
    return activity_id


async def _stored_streams(session, activity_id: str) -> dict[str, list]:
    result = await session.execute(
        select(ActivityStream).where(ActivityStream.activity_id == activity_id)
    )
    return {s.stream_type: s.data for s in result.scalars()}


class TestStoredStreamsShareAClock:
    async def test_every_stream_has_the_same_length(self, client, auth_headers, session):
        activity_id = await _upload_and_process(client, auth_headers, session)
        streams = await _stored_streams(session, activity_id)

        lengths = {k: len(v) for k, v in streams.items()}
        assert len(set(lengths.values())) == 1, f"streams disagree on length: {lengths}"

    async def test_the_grid_spans_elapsed_time_not_recorded_time(
        self, client, auth_headers, session
    ):
        # The device paused for a minute, so the ride is 600 s wide while the
        # session's timer time is 540 s. Sizing the streams by the timer would
        # drop the last minute of the ride off the end.
        activity_id = await _upload_and_process(client, auth_headers, session)
        streams = await _stored_streams(session, activity_id)
        assert len(streams["power"]) == _ELAPSED_S

        activity = (
            await session.execute(select(Activity).where(Activity.id == activity_id))
        ).scalar_one()
        assert activity.duration_s == _RECORDED_S

    async def test_the_hr_dropout_is_a_hole_at_the_right_seconds(
        self, client, auth_headers, session
    ):
        activity_id = await _upload_and_process(client, auth_headers, session)
        streams = await _stored_streams(session, activity_id)
        hr = streams["heartrate"]

        assert all(v is None for v in hr[HR_DROPOUT_START_S:HR_DROPOUT_END_S])
        assert hr[HR_DROPOUT_START_S - 1] is not None
        assert hr[HR_DROPOUT_END_S] is not None

    async def test_the_dropout_does_not_shift_hr_against_power(
        self, client, auth_headers, session
    ):
        # The regression itself. Under the old parser heart rate was 120 samples
        # shorter, so every HR sample after second 120 sat where an earlier
        # second's power was — and nothing in the pipeline noticed.
        activity_id = await _upload_and_process(client, auth_headers, session)
        streams = await _stored_streams(session, activity_id)

        # Power kept recording through the strap dropout, so those seconds have
        # power and no heart rate — the shape a dropout is supposed to leave.
        for i in range(HR_DROPOUT_START_S, HR_DROPOUT_END_S):
            assert streams["power"][i] is not None
            assert streams["heartrate"][i] is None

    async def test_the_device_pause_is_a_hole_in_every_channel(
        self, client, auth_headers, session
    ):
        activity_id = await _upload_and_process(client, auth_headers, session)
        streams = await _stored_streams(session, activity_id)

        for channel in _RECORDED_CHANNELS:
            window = streams[channel][RECORD_GAP_START_S:RECORD_GAP_END_S]
            assert all(v is None for v in window), f"{channel} filled the pause"

    async def test_derived_streams_are_on_the_same_grid(
        self, client, auth_headers, session
    ):
        activity_id = await _upload_and_process(client, auth_headers, session)
        streams = await _stored_streams(session, activity_id)

        # Torque comes from power and cadence, so it must be a gap wherever
        # either is — not a confident 0.0 Nm from the coasting branch.
        assert len(streams["torque"]) == _ELAPSED_S
        assert all(
            v is None for v in streams["torque"][RECORD_GAP_START_S:RECORD_GAP_END_S]
        )


class TestGapsSurviveStorageAndTheApi:
    async def test_stored_gaps_are_json_null(self, client, auth_headers, session):
        # NaN would serialise to a bare `NaN` token: valid to SQLite, rejected by
        # Postgres, so the bug would only appear in production.
        activity_id = await _upload_and_process(client, auth_headers, session)
        streams = await _stored_streams(session, activity_id)

        encoded = json.dumps(streams["heartrate"])
        assert "NaN" not in encoded
        assert json.loads(encoded) == streams["heartrate"]

    async def test_the_streams_endpoint_serves_the_gaps(
        self, client, auth_headers, session
    ):
        activity_id = await _upload_and_process(client, auth_headers, session)
        await session.commit()

        resp = await client.get(
            f"/api/activities/{activity_id}/streams", headers=auth_headers
        )
        assert resp.status_code == 200
        served = resp.json()["streams"]

        assert len(set(len(v) for v in served.values())) == 1
        assert served["heartrate"][HR_DROPOUT_START_S] is None
        assert served["power"][HR_DROPOUT_START_S] is not None


class TestDerivedMetricsOverGaps:
    async def test_averages_ignore_the_gaps(self, client, auth_headers, session):
        # A strap that dropped for two minutes must not pull average HR toward
        # zero — the average is over what was measured.
        activity_id = await _upload_and_process(client, auth_headers, session)
        activity = (
            await session.execute(select(Activity).where(Activity.id == activity_id))
        ).scalar_one()

        assert activity.avg_hr is not None
        assert 100 < activity.avg_hr < 160
        assert activity.avg_power is not None
        assert 150 < activity.avg_power < 250

    async def test_zone_times_count_recorded_seconds_only(
        self, client, auth_headers, session
    ):
        await client.patch(
            "/api/athlete",
            json={
                "hr_zones": [
                    {"name": "Z1 Recovery", "low": 0, "high": 110},
                    {"name": "Z2 Endurance", "low": 110, "high": 130},
                    {"name": "Z3 Tempo", "low": 130, "high": 150},
                    {"name": "Z4 Threshold", "low": 150, "high": 170},
                    {"name": "Z5 VO2max", "low": 170, "high": 220},
                ]
            },
            headers=auth_headers,
        )
        activity_id = await _upload_and_process(client, auth_headers, session)
        activity = (
            await session.execute(select(Activity).where(Activity.id == activity_id))
        ).scalar_one()

        # 600 elapsed seconds, 180 of them with no heart rate (a 120 s strap
        # dropout plus the 60 s pause). Gaps are time in no zone, so a NaN
        # clamping into Z1 would show up here as a total of 600.
        assert sum(activity.zone_times["hr"].values()) == _ELAPSED_S - 180

    async def test_intervals_cover_the_full_elapsed_ride(
        self, client, auth_headers, session
    ):
        from backend.app.models.user_orm import ActivityInterval

        activity_id = await _upload_and_process(client, auth_headers, session)
        result = await session.execute(
            select(ActivityInterval)
            .where(ActivityInterval.activity_id == activity_id)
            .order_by(ActivityInterval.interval_number)
        )
        intervals = list(result.scalars())
        assert intervals

        last = intervals[-1]
        assert last.start_offset_s + last.duration_s >= _ELAPSED_S
