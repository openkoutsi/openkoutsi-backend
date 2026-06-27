"""Unit tests for the PR badge detection service."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.services.pr_detection import detect_pr_badges

UTC = timezone.utc


def _dt(days_ago: int) -> datetime:
    return datetime(2025, 6, 1, tzinfo=UTC) - timedelta(days=days_ago)


def _make_power_row(activity_id: str, duration_s: int, power_w: float, days_ago: int) -> MagicMock:
    r = MagicMock()
    r.activity_id = activity_id
    r.duration_s = duration_s
    r.power_w = power_w
    r.activity_start_time = _dt(days_ago)
    return r


def _DistRow(activity_id, distance_m, time_s, days_ago, sport_type="Ride"):
    """Return a (best_mock, sport_type_str) tuple as stored in dist_result.all()."""
    best = MagicMock()
    best.activity_id = activity_id
    best.distance_m = distance_m
    best.time_s = time_s
    best.activity_start_time = _dt(days_ago)
    return (best, sport_type)


def _make_session(power_rows: list, dist_rows: list) -> AsyncMock:
    """Build a mock AsyncSession that returns power_rows then dist_rows on successive execute() calls."""
    session = AsyncMock()

    call_tracker = {"n": 0}

    async def _execute(query):
        call_tracker["n"] += 1
        if call_tracker["n"] == 1:
            # Power bests: needs .scalars() iteration
            mock_result = MagicMock()
            mock_result.scalars.return_value = power_rows
            return mock_result
        # Distance bests: needs .all() returning list of (best, sport_type) tuples
        mock_result = MagicMock()
        mock_result.all.return_value = dist_rows
        return mock_result

    session.execute = _execute
    return session


# ── None start_time returns empty ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_none_start_time_returns_empty():
    session = _make_session([], [])
    p, d = await detect_pr_badges("ath1", "act1", None, "Ride", session)
    assert p == {}
    assert d == {}


# ── Power bests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_single_activity_is_all_time_gold():
    rows = [_make_power_row("act1", 300, 350.0, 10)]
    session = _make_session(rows, [])
    p, _ = await detect_pr_badges("ath1", "act1", _dt(10), "Ride", session)
    assert p[300] == {"all_time": "gold", "12mo": "gold", "6mo": "gold", "3mo": "gold"}


@pytest.mark.asyncio
async def test_top_3_power_tiers():
    rows = [
        _make_power_row("gold", 300, 400.0, 10),
        _make_power_row("silver", 300, 380.0, 11),
        _make_power_row("bronze", 300, 360.0, 12),
        _make_power_row("nothing", 300, 340.0, 13),
    ]
    session = _make_session(rows, [])

    p, _ = await detect_pr_badges("ath1", "gold", _dt(10), "Ride", session)
    assert p[300]["all_time"] == "gold"

    session = _make_session(rows, [])
    p, _ = await detect_pr_badges("ath1", "silver", _dt(11), "Ride", session)
    assert p[300]["all_time"] == "silver"

    session = _make_session(rows, [])
    p, _ = await detect_pr_badges("ath1", "bronze", _dt(12), "Ride", session)
    assert p[300]["all_time"] == "bronze"

    session = _make_session(rows, [])
    p, _ = await detect_pr_badges("ath1", "nothing", _dt(13), "Ride", session)
    assert 300 not in p


@pytest.mark.asyncio
async def test_power_tie_broken_by_earlier_start_time():
    rows = [
        _make_power_row("earlier", 300, 350.0, 20),  # earlier — should win
        _make_power_row("later", 300, 350.0, 10),     # same power but later
    ]
    session = _make_session(rows, [])
    p, _ = await detect_pr_badges("ath1", "earlier", _dt(20), "Ride", session)
    assert p[300]["all_time"] == "gold"

    session = _make_session(rows, [])
    p, _ = await detect_pr_badges("ath1", "later", _dt(10), "Ride", session)
    assert p[300]["all_time"] == "silver"


@pytest.mark.asyncio
async def test_window_excludes_old_activity():
    """Competitor older than 12 months before the activity is excluded from windowed comparison."""
    # old_best is 500 days ago; ref is 100 days ago → 12mo cutoff = 465 days ago → old_best excluded
    old_best = _make_power_row("old", 300, 400.0, 500)
    target = _make_power_row("target", 300, 380.0, 100)
    rows = [old_best, target]
    session = _make_session(rows, [])
    p, _ = await detect_pr_badges("ath1", "target", _dt(100), "Ride", session)
    assert p[300]["all_time"] == "silver"   # old_best included all-time
    assert p[300]["12mo"] == "gold"         # old_best excluded from 12mo window
    assert p[300]["6mo"] == "gold"
    assert p[300]["3mo"] == "gold"


@pytest.mark.asyncio
async def test_window_excludes_very_old_competitor_from_3mo():
    """Competitor older than 3 months before activity does not affect 3mo ranking."""
    # competitor is 400 days ago; ref is 10 days ago → 3mo cutoff = 101 days ago → competitor excluded
    competitor = _make_power_row("comp", 300, 400.0, 400)
    target = _make_power_row("target", 300, 380.0, 10)
    rows = [competitor, target]
    session = _make_session(rows, [])
    p, _ = await detect_pr_badges("ath1", "target", _dt(10), "Ride", session)
    assert p[300]["3mo"] == "gold"      # competitor outside 3mo window
    assert p[300]["all_time"] == "silver"  # competitor included all-time


# ── Distance bests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_faster_time_earns_higher_rank():
    dist_rows = [
        _DistRow("fast", 5000, 900, 10),
        _DistRow("slow", 5000, 1000, 11),
    ]
    session = _make_session([], dist_rows)
    _, d = await detect_pr_badges("ath1", "fast", _dt(10), "Ride", session)
    assert d[5000]["all_time"] == "gold"

    session = _make_session([], dist_rows)
    _, d = await detect_pr_badges("ath1", "slow", _dt(11), "Ride", session)
    assert d[5000]["all_time"] == "silver"


@pytest.mark.asyncio
async def test_virtual_ride_isolated_from_real_ride_prs():
    """VirtualRide distance bests must not compete against real ride bests."""
    dist_rows = [
        _DistRow("real", 5000, 900, 5, sport_type="Ride"),
        _DistRow("virtual", 5000, 800, 3, sport_type="VirtualRide"),
    ]

    # Real ride target: should see only the real row (time_s=900) → gold
    session = _make_session([], dist_rows)
    _, d = await detect_pr_badges("ath1", "real", _dt(5), "Ride", session)
    assert d[5000]["all_time"] == "gold"

    # Virtual ride target: should see only the virtual row (time_s=800) → gold
    session = _make_session([], dist_rows)
    _, d = await detect_pr_badges("ath1", "virtual", _dt(3), "VirtualRide", session)
    assert d[5000]["all_time"] == "gold"


@pytest.mark.asyncio
async def test_distance_tie_broken_by_earlier_start_time():
    dist_rows = [
        _DistRow("earlier", 10000, 2000, 20),
        _DistRow("later", 10000, 2000, 10),
    ]
    session = _make_session([], dist_rows)
    _, d = await detect_pr_badges("ath1", "earlier", _dt(20), "Ride", session)
    assert d[10000]["all_time"] == "gold"

    session = _make_session([], dist_rows)
    _, d = await detect_pr_badges("ath1", "later", _dt(10), "Ride", session)
    assert d[10000]["all_time"] == "silver"
