"""In-batch deduplication for bulk import (issue #36).

A Strava export can hold one ride three times — ``.fit``, ``.tcx`` and ``.gpx``
of the same session — so the batch has to be collapsed *before* anything is
written, and the copy that survives has to be the one with the most in it.
These are the grouping rules on their own, without a database in the way.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.app.services.activity_archive import ExpandedFile
from backend.app.services.activity_import import (
    DUPLICATE_WINDOW,
    _Candidate,
    deduplicate_batch,
)

_T0 = datetime(2024, 3, 2, 9, 0, tzinfo=timezone.utc)


def candidate(name: str, fmt: str, offset_s: float | None = 0.0) -> _Candidate:
    start = None if offset_s is None else _T0 + timedelta(seconds=offset_s)
    return _Candidate(ExpandedFile(name=name, path=None, format=fmt), start)


def survivors(candidates: list[_Candidate]) -> list[str]:
    return [c.file.name for c in candidates if c.duplicate_of is None]


class TestFormatPreference:
    def test_fit_beats_tcx_beats_gpx(self):
        batch = [
            candidate("ride.gpx", "gpx"),
            candidate("ride.fit", "fit"),
            candidate("ride.tcx", "tcx"),
        ]
        deduplicate_batch(batch)

        assert survivors(batch) == ["ride.fit"]
        assert all(
            c.duplicate_of == "ride.fit" for c in batch if c.file.name != "ride.fit"
        )

    def test_tcx_wins_when_there_is_no_fit(self):
        batch = [candidate("ride.gpx", "gpx"), candidate("ride.tcx", "tcx")]
        deduplicate_batch(batch)
        assert survivors(batch) == ["ride.tcx"]

    def test_a_lone_file_is_never_a_duplicate(self):
        batch = [candidate("ride.gpx", "gpx")]
        deduplicate_batch(batch)
        assert batch[0].duplicate_of is None


class TestTheWindow:
    def test_files_within_the_window_are_one_ride(self):
        batch = [
            candidate("a.gpx", "gpx", 0),
            candidate("b.gpx", "gpx", DUPLICATE_WINDOW.total_seconds() - 1),
        ]
        deduplicate_batch(batch)
        assert len(survivors(batch)) == 1

    def test_files_outside_the_window_are_separate_rides(self):
        batch = [
            candidate("morning.gpx", "gpx", 0),
            candidate("evening.gpx", "gpx", DUPLICATE_WINDOW.total_seconds() + 1),
        ]
        deduplicate_batch(batch)
        assert sorted(survivors(batch)) == ["evening.gpx", "morning.gpx"]

    def test_a_chain_of_close_files_does_not_swallow_a_whole_day(self):
        # Six files four minutes apart span twenty minutes. Each is within the
        # window of the one before it, but they are not all one ride — the group
        # is anchored to the file that opened it, not to the last one seen.
        step = DUPLICATE_WINDOW.total_seconds() - 60
        batch = [candidate(f"{i}.gpx", "gpx", i * step) for i in range(6)]
        deduplicate_batch(batch)

        assert len(survivors(batch)) > 1

    def test_order_in_the_archive_does_not_matter(self):
        forwards = [candidate("late.tcx", "tcx", 60), candidate("early.gpx", "gpx", 0)]
        backwards = [candidate("early.gpx", "gpx", 0), candidate("late.tcx", "tcx", 60)]
        deduplicate_batch(forwards)
        deduplicate_batch(backwards)

        assert survivors(forwards) == survivors(backwards) == ["late.tcx"]


class TestUndatedFiles:
    def test_a_file_with_no_start_time_is_left_alone(self):
        # Nothing is known about when it happened, so it cannot be shown to be
        # a duplicate of anything. It is imported, and the parse failure (or the
        # database's own duplicate check) decides its fate.
        batch = [candidate("ride.gpx", "gpx", 0), candidate("mystery.gpx", "gpx", None)]
        deduplicate_batch(batch)

        assert batch[1].duplicate_of is None
        assert sorted(survivors(batch)) == ["mystery.gpx", "ride.gpx"]


class TestNaiveTimestamps:
    def test_naive_and_aware_timestamps_compare_as_utc(self):
        aware = candidate("aware.gpx", "gpx", 0)
        naive = _Candidate(
            ExpandedFile(name="naive.tcx", path=None, format="tcx"),
            datetime(2024, 3, 2, 9, 0, 30),
        )
        batch = [aware, naive]
        deduplicate_batch(batch)

        # A parser that returns a naive datetime must not create a phantom
        # second activity thirty seconds after the first.
        assert survivors(batch) == ["naive.tcx"]


@pytest.mark.parametrize("window", [DUPLICATE_WINDOW])
def test_the_window_matches_the_rest_of_the_ingest_paths(window):
    from backend.app.api.activities import _DUPLICATE_WINDOW

    assert window == _DUPLICATE_WINDOW
