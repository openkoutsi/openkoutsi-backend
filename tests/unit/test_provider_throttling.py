"""
Unit tests for backend.app.services.providers.throttling.

The module exists to keep "the provider has nothing for this activity" and
"the provider is refusing to talk to us" from arriving at the sync pipeline as
the same answer (issue #67), so these tests are mostly about which failures land
on which side of that line.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.services.providers.throttling import (
    MAX_RETRY_AFTER_WAIT,
    ProviderThrottled,
    as_throttle,
    call_with_retry_after,
    from_response,
    is_definitive,
    parse_retry_after,
)


def _status_error(status: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://provider.test/activities/1/streams")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


# ── parse_retry_after ─────────────────────────────────────────────────────────


class TestParseRetryAfter:
    def test_delta_seconds(self):
        assert parse_retry_after("120") == 120.0

    def test_http_date(self):
        when = datetime.now(timezone.utc) + timedelta(seconds=90)
        stamp = when.strftime("%a, %d %b %Y %H:%M:%S GMT")
        parsed = parse_retry_after(stamp)
        assert parsed is not None
        assert 60 <= parsed <= 120

    def test_http_date_in_the_past_is_zero_not_negative(self):
        when = datetime.now(timezone.utc) - timedelta(hours=1)
        stamp = when.strftime("%a, %d %b %Y %H:%M:%S GMT")
        assert parse_retry_after(stamp) == 0.0

    @pytest.mark.parametrize("value", [None, "", "   ", "soon", "next tuesday"])
    def test_unusable_values_read_as_no_answer(self, value):
        """None means "it did not say", which callers must not read as "wait 0"."""
        assert parse_retry_after(value) is None


# ── as_throttle ───────────────────────────────────────────────────────────────


class TestAsThrottle:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retryable_statuses_are_throttles(self, status):
        throttle = as_throttle(_status_error(status), "strava")
        assert throttle is not None
        assert throttle.status_code == status
        assert throttle.provider == "strava"

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_errors_are_answers_not_throttles(self, status):
        assert as_throttle(_status_error(status), "strava") is None

    def test_retry_after_is_carried_through(self):
        throttle = as_throttle(
            _status_error(429, {"Retry-After": "45"}), "strava"
        )
        assert throttle is not None
        assert throttle.retry_after == 45.0

    def test_non_http_failures_are_left_alone(self):
        """A transport error or a client's own exception is the caller's to judge."""
        assert as_throttle(Exception("no FIT for this one"), "wahoo") is None
        assert as_throttle(httpx.ConnectError("boom"), "wahoo") is None

    def test_an_existing_throttle_passes_through(self):
        original = ProviderThrottled("strava", 429, 30.0)
        assert as_throttle(original, "strava") is original


# ── from_response ─────────────────────────────────────────────────────────────


class TestFromResponse:
    """For clients that read the status themselves instead of raising."""

    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_a_retryable_response_is_a_throttle(self, status):
        response = httpx.Response(status, headers={"Retry-After": "7"})
        throttle = from_response(response, "wahoo")
        assert throttle is not None
        assert throttle.status_code == status
        assert throttle.retry_after == 7.0

    @pytest.mark.parametrize("status", [200, 404, 403])
    def test_everything_else_is_the_response_it_looks_like(self, status):
        assert from_response(httpx.Response(status), "wahoo") is None

    def test_no_response_at_all_is_not_a_throttle(self):
        assert from_response(None, "wahoo") is None


# ── is_definitive ─────────────────────────────────────────────────────────────


class TestIsDefinitive:
    def test_a_404_settles_the_question(self):
        """Strava answers a manual entry's stream request with a 404, forever."""
        assert is_definitive(_status_error(404)) is True

    @pytest.mark.parametrize("status", [429, 503])
    def test_a_throttle_settles_nothing(self, status):
        assert is_definitive(_status_error(status)) is False

    def test_a_transport_failure_settles_nothing(self):
        assert is_definitive(httpx.ReadTimeout("timed out")) is False


# ── call_with_retry_after ─────────────────────────────────────────────────────


class TestCallWithRetryAfter:
    async def test_returns_the_value_when_nothing_goes_wrong(self):
        call = AsyncMock(return_value={"power": [1, 2, 3]})
        got = await call_with_retry_after(
            call, "tok", "act-1", provider="strava", what="streams"
        )
        assert got == {"power": [1, 2, 3]}
        call.assert_awaited_once_with("tok", "act-1")

    async def test_waits_the_named_delay_and_retries_once(self):
        call = AsyncMock(
            side_effect=[_status_error(429, {"Retry-After": "12"}), {"power": [9]}]
        )
        with patch(
            "backend.app.services.providers.throttling.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            got = await call_with_retry_after(
                call, "tok", provider="strava", what="streams"
            )
        assert got == {"power": [9]}
        sleep.assert_awaited_once_with(12.0)
        assert call.await_count == 2

    async def test_a_second_refusal_is_final(self):
        call = AsyncMock(side_effect=_status_error(429, {"Retry-After": "1"}))
        with patch(
            "backend.app.services.providers.throttling.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            with pytest.raises(ProviderThrottled):
                await call_with_retry_after(
                    call, "tok", provider="strava", what="streams"
                )
        assert call.await_count == 2

    async def test_no_retry_after_means_no_waiting(self):
        """Without a delay to honour there is nothing to wait for — stop instead."""
        call = AsyncMock(side_effect=_status_error(429))
        with patch(
            "backend.app.services.providers.throttling.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            with pytest.raises(ProviderThrottled) as exc:
                await call_with_retry_after(
                    call, "tok", provider="strava", what="streams"
                )
        assert exc.value.retry_after is None
        sleep.assert_not_awaited()
        assert call.await_count == 1

    async def test_a_long_window_is_not_waited_out(self):
        """A quarter-hour quota window is for the next sync, not this one."""
        delay = MAX_RETRY_AFTER_WAIT + 1
        call = AsyncMock(side_effect=_status_error(429, {"Retry-After": str(delay)}))
        with patch(
            "backend.app.services.providers.throttling.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            with pytest.raises(ProviderThrottled) as exc:
                await call_with_retry_after(
                    call, "tok", provider="strava", what="streams"
                )
        assert exc.value.retry_after == delay
        sleep.assert_not_awaited()
        assert call.await_count == 1

    async def test_other_failures_are_not_swallowed_or_retried(self):
        call = AsyncMock(side_effect=_status_error(404))
        with pytest.raises(httpx.HTTPStatusError):
            await call_with_retry_after(call, "tok", provider="strava", what="streams")
        assert call.await_count == 1


class TestCoverageOfTheRemainingBranches:
    """The paths the main cases above happen not to walk through."""

    def test_a_naive_http_date_is_read_as_utc(self):
        """`parsedate_to_datetime` returns a naive datetime for a zoneless stamp.

        Subtracting one from an aware `now` raises, so the tz has to be filled in
        — and UTC is the only reading of an HTTP-date that is ever correct.
        """
        when = datetime.now(timezone.utc) + timedelta(seconds=45)
        parsed = parse_retry_after(when.strftime("%a, %d %b %Y %H:%M:%S"))
        assert parsed is not None
        assert 15 <= parsed <= 75

    def test_a_throttle_is_never_definitive(self):
        """The one exception type that carries a status but settles nothing."""
        assert is_definitive(ProviderThrottled("strava", 429)) is False
