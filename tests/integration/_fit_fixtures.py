"""Discovery and capability helpers for FIT test fixtures.

Every ``*.fit`` file under ``testdata/fixtures/`` is picked up automatically and
used to parametrize the FIT-processing integration tests, so dropping in a new
file from any producer (Garmin, Wahoo, Zwift, Coros, …) exercises the upload and
analysis pipeline against it — no test edits required. Strip location data first
with ``scripts/strip_fit_location.py``.

Because producers differ in what they record (an outdoor run has no power, an
indoor trainer has no speed/GPS, …), tests gate their producer-specific
assertions on :func:`capabilities` rather than assuming every file has every
stream. Files that lack a capability a test needs are skipped for that test.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent.parent / "testdata" / "fixtures"

# Smallest standard distance-best bucket (metres); a file must cover at least
# this to produce any distance bests.
_MIN_DISTANCE_BEST_M = 1_000


def fit_fixture_files() -> list[Path]:
    """All committed/available FIT fixtures, sorted for stable test ids."""
    if not FIXTURES_DIR.is_dir():
        return []
    return sorted(FIXTURES_DIR.glob("*.fit"))


def fit_fixture_params() -> list:
    """``pytest.param`` list for parametrizing over every FIT fixture.

    Yields one parameter per discovered file (id = filename stem). If none are
    present, yields a single skipped placeholder so the test is reported as
    skipped rather than silently collected away.
    """
    files = fit_fixture_files()
    if not files:
        return [
            pytest.param(
                None,
                marks=pytest.mark.skip(reason="no FIT fixtures in testdata/fixtures/"),
                id="no-fit-fixtures",
            )
        ]
    return [pytest.param(p, id=p.stem) for p in files]


@dataclass(frozen=True)
class FitCapabilities:
    has_power: bool
    has_speed: bool
    has_hr: bool
    has_cadence: bool
    covers_min_distance: bool  # long enough for at least one distance best
    duration_s: int


@lru_cache(maxsize=None)
def capabilities(path: Path) -> FitCapabilities:
    """Inspect a FIT fixture once and report which streams it carries."""
    from openkoutsi.fit import summarizeWorkout

    prof = summarizeWorkout(str(path))
    # speed samples are km/h at 1 Hz; integrate to metres.
    integrated_distance_m = sum(v / 3.6 for v in prof.speed)
    return FitCapabilities(
        has_power=bool(prof.power),
        has_speed=bool(prof.speed),
        has_hr=bool(prof.heartRate),
        has_cadence=bool(prof.cadence),
        covers_min_distance=integrated_distance_m >= _MIN_DISTANCE_BEST_M,
        duration_s=prof.duration,
    )
