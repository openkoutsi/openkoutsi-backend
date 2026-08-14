#!/usr/bin/env python3
"""Generate small, synthetic FIT fixtures for the test suite.

Usage:
    uv run python scripts/generate_synthetic_fit_fixtures.py

Writes a handful of tiny, entirely made-up FIT activity files into
``testdata/fixtures/``. They contain no real personal data — the numbers are
generated — so they are safe to commit and give CI concrete coverage of the
FIT upload/processing pipeline without depending on anyone's private ride.

The files deliberately span the capability matrix the tests gate on:

    synthetic_bike_power_gps    power + speed + HR + cadence + GPS  (outdoor bike)
    synthetic_run_no_power      speed + HR + cadence, no power      (outdoor run)
    synthetic_indoor_no_gps     power + HR + cadence, no speed/GPS  (indoor trainer)
    synthetic_bike_hr_dropout   as the first, with an HR dropout and a device pause

Real, GPS-stripped ride files can be dropped alongside these; the tests pick
up every ``*.fit`` in the directory automatically.
"""
import math
from datetime import datetime, timezone
from pathlib import Path

from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.lap_message import LapMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.session_message import SessionMessage
from fit_tool.profile.messages.sport_message import SportMessage
from fit_tool.profile.profile_type import FileType, Manufacturer, Sport

FIXTURES_DIR = Path(__file__).parent.parent / "testdata" / "fixtures"
_EPOCH = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

# The holes injected into `synthetic_bike_hr_dropout.fit`, as second offsets
# from the first record. Tests import these rather than restating the numbers,
# so moving a window here can't silently stop a test from checking anything.
HR_DROPOUT_START_S, HR_DROPOUT_END_S = 120, 240
RECORD_GAP_START_S, RECORD_GAP_END_S = 400, 460


def _build(
    name,
    *,
    sport,
    seconds,
    speed_ms,
    with_power,
    with_gps,
    hr_dropout=None,
    record_gap=None,
):
    """Assemble one synthetic FIT activity and write it to the fixtures dir.

    ``hr_dropout`` is a ``(start, end)`` second range where the heart-rate field
    is left off the records — a strap losing contact. ``record_gap`` is a range
    where no ``record`` frame is written at all — a device pause. Both are
    half-open and exist for the stream-alignment fixture; see issue #76.
    """
    builder = FitFileBuilder(auto_define=True)
    start_ms = int(_EPOCH.timestamp() * 1000)

    file_id = FileIdMessage()
    file_id.type = FileType.ACTIVITY
    file_id.manufacturer = Manufacturer.DEVELOPMENT.value
    file_id.product = 0
    file_id.serial_number = 1
    file_id.time_created = start_ms
    builder.add(file_id)

    sport_msg = SportMessage()
    sport_msg.sport = sport
    builder.add(sport_msg)

    total_distance = 0.0
    recorded_s = 0
    for i in range(seconds):
        if record_gap and record_gap[0] <= i < record_gap[1]:
            # Device paused: no frame at all, so *every* channel has a hole here.
            continue
        r = RecordMessage()
        r.timestamp = start_ms + i * 1000
        if not (hr_dropout and hr_dropout[0] <= i < hr_dropout[1]):
            r.heart_rate = 130 + int(15 * math.sin(i / 30))
        r.cadence = 85 + (i % 5)
        r.altitude = 50.0 + 10 * math.sin(i / 60)
        if speed_ms is not None:
            r.speed = speed_ms
            total_distance += speed_ms
            r.distance = total_distance
        if with_power:
            r.power = 200 + int(40 * math.sin(i / 20))
        if with_gps:
            # Made-up coordinates; degrees are converted to semicircles by fit_tool.
            r.position_lat = 51.5 + i * 0.0001
            r.position_long = -0.12 + i * 0.0001
        builder.add(r)
        recorded_s += 1

    # Timestamps are date_times (milliseconds); elapsed/timer times are scaled
    # duration fields expressed in seconds. Timer time counts only the seconds
    # actually recorded, which is what a device does across a pause — so a file
    # with a `record_gap` genuinely has timer < elapsed.
    lap = LapMessage()
    lap.timestamp = start_ms + seconds * 1000
    lap.start_time = start_ms
    lap.total_elapsed_time = seconds
    lap.total_timer_time = recorded_s
    lap.total_distance = total_distance
    builder.add(lap)

    session = SessionMessage()
    session.timestamp = start_ms + seconds * 1000
    session.start_time = start_ms
    session.sport = sport
    session.total_elapsed_time = seconds
    session.total_timer_time = recorded_s
    session.total_distance = total_distance
    builder.add(session)

    out = FIXTURES_DIR / name
    builder.build().to_file(str(out))
    print(f"wrote {out}  ({seconds}s, power={with_power}, speed={speed_ms}, gps={with_gps})")


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    # Outdoor bike: power + speed + GPS, long enough for 1s..300s power bests and 1-2 km.
    _build("synthetic_bike_power_gps.fit", sport=Sport.CYCLING,
           seconds=300, speed_ms=9.0, with_power=True, with_gps=True)
    # Outdoor run: speed + HR but no power; covers >1 km so distance bests appear.
    _build("synthetic_run_no_power.fit", sport=Sport.RUNNING,
           seconds=600, speed_ms=3.3, with_power=False, with_gps=True)
    # Indoor trainer: power + HR, no speed stream and no GPS.
    _build("synthetic_indoor_no_gps.fit", sport=Sport.CYCLING,
           seconds=300, speed_ms=None, with_power=True, with_gps=False)
    # Outdoor bike with two kinds of hole, for the stream-alignment contract
    # (issue #76): heart rate drops out for 120 s while every other channel keeps
    # recording, and separately the device stops writing records for 60 s. The
    # first is what used to shift HR against power; the second is what makes the
    # grid wider than the recorded timer time.
    _build("synthetic_bike_hr_dropout.fit", sport=Sport.CYCLING,
           seconds=600, speed_ms=9.0, with_power=True, with_gps=False,
           hr_dropout=(HR_DROPOUT_START_S, HR_DROPOUT_END_S),
           record_gap=(RECORD_GAP_START_S, RECORD_GAP_END_S))


if __name__ == "__main__":
    main()
