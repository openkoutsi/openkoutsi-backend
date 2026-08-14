"""Strava's stream arrays must land on the same clock as a FIT file (issue #76).

Strava returns one array per channel plus a ``time`` array of second offsets.
Those arrays are internally consistent — index i is the i-th *sample* in every
channel — but that is not the contract the rest of openkoutsi is written
against, which is that index i is *second* i. On a ride recorded at anything
other than 1 Hz the two differ, and everything downstream that reads an index as
a clock (``w_bal``, the interval slicing, time-in-zone) would silently read a
different clock depending on where the activity came from.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.services.providers.strava import StravaProviderClient


def _mock_streams_response(raw: dict):
    resp = MagicMock()
    resp.json.return_value = raw
    resp.raise_for_status = MagicMock()

    http = AsyncMock()
    http.get = AsyncMock(return_value=resp)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)
    return http


async def _fetch(raw: dict) -> dict:
    with patch("httpx.AsyncClient", return_value=_mock_streams_response(raw)):
        return await StravaProviderClient().get_activity_streams("token", "123")


class TestStravaStreamResampling:
    async def test_one_hz_activity_is_unchanged(self):
        result = await _fetch({
            "time": {"data": [0, 1, 2]},
            "watts": {"data": [200, 210, 220]},
            "heartrate": {"data": [140, 141, 142]},
        })
        assert result["power"] == [200.0, 210.0, 220.0]
        assert result["heartrate"] == [140.0, 141.0, 142.0]

    async def test_irregular_sampling_is_spread_onto_seconds(self):
        # Four samples over eleven seconds is an eleven-second stream, not a
        # four-element list whose index pretends to be a clock.
        result = await _fetch({
            "time": {"data": [0, 1, 5, 10]},
            "watts": {"data": [200, 210, 250, 300]},
        })
        assert len(result["power"]) == 11
        assert result["power"][0] == 200.0
        assert result["power"][5] == 250.0
        assert result["power"][10] == 300.0
        assert result["power"][2] is None

    async def test_channels_stay_aligned_with_each_other(self):
        result = await _fetch({
            "time": {"data": [0, 30, 60]},
            "watts": {"data": [200, 250, 300]},
            "heartrate": {"data": [140, 150, 160]},
        })
        assert len(result["power"]) == len(result["heartrate"]) == 61
        for i, (p, h) in enumerate(zip(result["power"], result["heartrate"])):
            assert (p is None) == (h is None), f"channels disagree at second {i}"

    async def test_a_time_stream_not_starting_at_zero_is_zero_based(self):
        # Strava offsets are relative to the activity start, but a stream that
        # begins late must not allocate a leading run of gaps.
        result = await _fetch({
            "time": {"data": [7, 8]},
            "watts": {"data": [200, 210]},
        })
        assert result["power"] == [200.0, 210.0]

    async def test_channels_strava_did_not_return_stay_absent(self):
        result = await _fetch({
            "time": {"data": [0, 1]},
            "watts": {"data": [200, 210]},
        })
        assert "heartrate" not in result
        assert "cadence" not in result

    async def test_without_a_time_stream_the_arrays_pass_through(self):
        # Older activities, or a Strava response that omits `time`. Nothing to
        # resample against, so this behaves as it did before #76.
        result = await _fetch({"watts": {"data": [200, 210, 220]}})
        assert result["power"] == [200.0, 210.0, 220.0]

    async def test_empty_response(self):
        assert await _fetch({}) == {}

    @pytest.mark.parametrize("key,dest", [
        ("watts", "power"),
        ("heartrate", "heartrate"),
        ("cadence", "cadence"),
        ("velocity_smooth", "speed"),
        ("altitude", "altitude"),
    ])
    async def test_every_mapped_channel_is_resampled(self, key, dest):
        result = await _fetch({
            "time": {"data": [0, 2]},
            key: {"data": [1, 2]},
        })
        assert result[dest] == [1.0, None, 2.0]
