"""Unit tests for quota headroom (issue #66).

The behaviour under test is the one the issue calls out as the difference
between a useful panel and a misleading one: a usage reading is only valid
inside the window it was observed in, so an observation from an earlier window
must display as *reset*, never as current usage. Getting that backwards would
show alarming usage against a window that is actually empty — and would stop an
import that had full headroom.
"""

from datetime import datetime, timedelta, timezone

import pytest

from backend.app.models.api_usage_orm import ApiUsage
from backend.app.services.quota import (
    all_headroom,
    current_headroom,
    daily_window_start,
    short_window_start,
)

# 10:14 UTC — inside the 10:00–10:15 quarter-hour window, one minute from its end.
OBSERVED_AT = datetime(2026, 9, 10, 10, 14, 0, tzinfo=timezone.utc)


async def _record(factory, **overrides):
    fields = dict(
        created_at=OBSERVED_AT,
        service="strava",
        endpoint="/athlete",
        method="GET",
        status_code=200,
        outcome="ok",
        ratelimit_usage_short=580,
        ratelimit_limit_short=600,
        ratelimit_usage_daily=27536,
        ratelimit_limit_daily=30000,
        ratelimit_read_usage_short=290,
        ratelimit_read_limit_short=300,
        ratelimit_read_usage_daily=14000,
        ratelimit_read_limit_daily=15000,
    )
    fields.update(overrides)
    async with factory() as session:
        session.add(ApiUsage(**fields))
        await session.commit()


def _window(headroom, scope, window):
    return next(
        w for w in headroom.windows if w.scope == scope and w.window == window
    )


class TestWindowBoundaries:
    @pytest.mark.parametrize(
        "when,expected_minute",
        [(0, 0), (14, 0), (15, 15), (29, 15), (30, 30), (44, 30), (45, 45), (59, 45)],
    )
    def test_short_window_aligns_to_the_quarter_hour(self, when, expected_minute):
        start = short_window_start(datetime(2026, 9, 10, 10, when, 30, tzinfo=timezone.utc))
        assert start.minute == expected_minute
        assert (start.second, start.microsecond) == (0, 0)

    def test_daily_window_starts_at_utc_midnight(self):
        start = daily_window_start(datetime(2026, 9, 10, 23, 59, 59, tzinfo=timezone.utc))
        assert start == datetime(2026, 9, 10, 0, 0, 0, tzinfo=timezone.utc)


class TestRemaining:
    def test_is_none_when_either_side_is_unknown(self):
        from backend.app.services.quota import WindowHeadroom

        def w(usage, limit):
            return WindowHeadroom(
                window="short", scope="overall", usage=usage, limit=limit,
                window_start=OBSERVED_AT, resets_at=OBSERVED_AT,
                observed_in_window=True,
            )

        # "We do not know" must not collapse into a number. A limit we never saw
        # would otherwise read as a headroom figure.
        assert w(100, None).remaining is None
        assert w(None, 600).remaining is None
        assert w(None, None).remaining is None
        assert w(100, 600).remaining == 500

    def test_never_goes_negative(self):
        from backend.app.services.quota import WindowHeadroom

        over = WindowHeadroom(
            window="short", scope="overall", usage=700, limit=600,
            window_start=OBSERVED_AT, resets_at=OBSERVED_AT,
            observed_in_window=True,
        )
        assert over.remaining == 0


class TestHeadroomWithinTheWindow:
    async def test_observation_inside_the_window_is_reported_as_seen(self, api_usage_db):
        await _record(api_usage_db)
        headroom = await current_headroom(
            "strava", now=OBSERVED_AT + timedelta(seconds=30)
        )

        short = _window(headroom, "overall", "short")
        assert short.observed_in_window is True
        assert (short.usage, short.limit, short.remaining) == (580, 600, 20)

        read = _window(headroom, "read", "short")
        assert (read.usage, read.limit, read.remaining) == (290, 300, 10)

    async def test_age_is_reported_so_staleness_is_visible(self, api_usage_db):
        await _record(api_usage_db)
        headroom = await current_headroom(
            "strava", now=OBSERVED_AT + timedelta(minutes=3)
        )
        assert headroom.observed_at == OBSERVED_AT
        assert headroom.age_seconds == pytest.approx(180.0)


class TestWindowRollover:
    """The clock is pinned across a window boundary — the issue's own test ask."""

    async def test_short_window_reset_reports_zero_not_the_stale_number(
        self, api_usage_db
    ):
        await _record(api_usage_db)
        # 10:15:01 — 61 seconds after the observation, but a *different* window.
        headroom = await current_headroom(
            "strava", now=datetime(2026, 9, 10, 10, 15, 1, tzinfo=timezone.utc)
        )

        short = _window(headroom, "overall", "short")
        assert short.observed_in_window is False
        assert short.usage == 0, "a reset window is empty, not 580/600"
        assert short.remaining == 600

        # The daily counter has *not* rolled over, so it keeps its real value.
        daily = _window(headroom, "overall", "daily")
        assert daily.observed_in_window is True
        assert daily.usage == 27536

    async def test_read_quota_resets_on_the_same_boundary(self, api_usage_db):
        await _record(api_usage_db)
        headroom = await current_headroom(
            "strava", now=datetime(2026, 9, 10, 10, 15, 1, tzinfo=timezone.utc)
        )
        read = _window(headroom, "read", "short")
        assert read.observed_in_window is False
        assert read.usage == 0

    async def test_next_day_resets_the_daily_counter_too(self, api_usage_db):
        await _record(api_usage_db)
        headroom = await current_headroom(
            "strava", now=datetime(2026, 9, 11, 0, 0, 1, tzinfo=timezone.utc)
        )
        for scope in ("overall", "read"):
            for window in ("short", "daily"):
                w = _window(headroom, scope, window)
                assert w.observed_in_window is False
                assert w.usage == 0

    async def test_one_second_before_the_boundary_is_still_the_same_window(
        self, api_usage_db
    ):
        await _record(api_usage_db)
        headroom = await current_headroom(
            "strava", now=datetime(2026, 9, 10, 10, 14, 59, tzinfo=timezone.utc)
        )
        assert _window(headroom, "overall", "short").observed_in_window is True
        assert _window(headroom, "overall", "short").usage == 580


class TestNoObservation:
    async def test_a_service_we_have_never_called_says_so(self, api_usage_db):
        headroom = await current_headroom("strava")
        assert headroom.has_observation is False
        assert headroom.observed_at is None
        assert headroom.windows == []

    async def test_rows_without_a_reading_do_not_shadow_the_newest_one(
        self, api_usage_db
    ):
        # A transport failure carries no headers. It is newer than the real
        # reading, and must not be mistaken for "we have no reading".
        await _record(api_usage_db)
        await _record(
            api_usage_db,
            created_at=OBSERVED_AT + timedelta(seconds=10),
            status_code=None,
            outcome="transport_error",
            ratelimit_usage_short=None,
            ratelimit_limit_short=None,
            ratelimit_usage_daily=None,
            ratelimit_limit_daily=None,
            ratelimit_read_usage_short=None,
            ratelimit_read_limit_short=None,
            ratelimit_read_usage_daily=None,
            ratelimit_read_limit_daily=None,
        )
        headroom = await current_headroom(
            "strava", now=OBSERVED_AT + timedelta(seconds=20)
        )
        assert headroom.observed_at == OBSERVED_AT
        assert _window(headroom, "overall", "short").usage == 580

    async def test_wahoo_publishes_no_quota(self, api_usage_db):
        await _record(api_usage_db, service="wahoo", **{
            f"ratelimit_{k}": None
            for k in (
                "usage_short", "limit_short", "usage_daily", "limit_daily",
                "read_usage_short", "read_limit_short",
                "read_usage_daily", "read_limit_daily",
            )
        })
        headroom = await current_headroom("wahoo")
        assert headroom.has_observation is False


class TestRateLimitGroundTruth:
    async def test_the_last_429_is_reported(self, api_usage_db):
        await _record(api_usage_db)
        throttled_at = OBSERVED_AT + timedelta(seconds=5)
        await _record(
            api_usage_db,
            created_at=throttled_at,
            status_code=429,
            outcome="rate_limited",
        )
        headroom = await current_headroom(
            "strava", now=throttled_at + timedelta(seconds=1)
        )
        assert headroom.last_rate_limited_at == throttled_at

    async def test_no_429_reports_none(self, api_usage_db):
        await _record(api_usage_db)
        headroom = await current_headroom("strava", now=OBSERVED_AT)
        assert headroom.last_rate_limited_at is None


class TestAllHeadroom:
    async def test_covers_every_watched_service(self, api_usage_db):
        await _record(api_usage_db)
        services = await all_headroom(now=OBSERVED_AT)
        assert [s.service for s in services] == ["strava", "wahoo"]
        assert services[0].has_observation is True
        assert services[1].has_observation is False
