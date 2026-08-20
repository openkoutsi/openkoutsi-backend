#!/usr/bin/env python3
"""Generate small, synthetic GPX and TCX fixtures for the test suite.

Usage:
    uv run python scripts/generate_synthetic_activity_fixtures.py

The FIT counterpart is ``generate_synthetic_fit_fixtures.py``; this is the same
idea for the two XML formats a Strava bulk export contains (issue #36). Every
file written here is entirely made up — the coordinates trace a closed loop in
the middle of the Gulf of Bothnia — so they are safe to commit and give CI real
coverage of the GPX/TCX import path.

One file is generated per format from the *same synthetic ride*:

    synthetic_ride.gpx      HR + cadence + power, GPS, 1 Hz
    synthetic_ride.tcx      the same ride, with device distance and two laps
    synthetic_hr_only.gpx   HR only — no power at all, which is most GPX
    synthetic_ride.gpx.gz   the gzip form Strava exports

Because the GPX and the TCX describe one ride, the tests can assert that two
different parsers agree on its distance and elevation to within the error each
format's own arithmetic implies — which is the property that makes a mixed
archive import trustworthy.
"""
import gzip
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape

FIXTURES_DIR = Path(__file__).parent.parent / "testdata" / "fixtures"
_EPOCH = datetime(2024, 3, 2, 9, 0, 0, tzinfo=timezone.utc)

# Open water, far from anyone's home. A ride "recorded" here cannot identify a
# person, which is the whole reason these fixtures can be committed.
_ORIGIN_LAT, _ORIGIN_LON = 61.5, 20.5
_DURATION_S = 600
_LAP_SPLIT_S = 300

# One degree of latitude is ~111.32 km; longitude shrinks by cos(latitude).
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = _M_PER_DEG_LAT * math.cos(math.radians(_ORIGIN_LAT))


def _sample(second: int) -> dict:
    """One second of the synthetic ride, every channel."""
    # A straight line east at a steady 8 m/s, so the derived distance is a
    # number the tests can state in the assertion rather than read back from
    # the parser they are testing.
    metres = 8.0 * second
    return {
        "time": _EPOCH + timedelta(seconds=second),
        "lat": _ORIGIN_LAT,
        "lon": _ORIGIN_LON + metres / _M_PER_DEG_LON,
        # A single 60 m climb and nothing else, so the expected elevation gain
        # is exactly 60 m under any sane noise threshold.
        "ele": 10.0 + 60.0 * min(1.0, second / 300.0),
        "hr": 130.0 + 20.0 * math.sin(second / 60.0),
        "cadence": 85.0,
        "power": 200.0 + 40.0 * math.sin(second / 45.0),
        "distance": metres,
        "speed": 8.0,
    }


def _gpx(*, with_power: bool, with_cadence: bool = True) -> str:
    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx creator="openkoutsi synthetic fixtures" version="1.1"\n'
        '     xmlns="http://www.topografix.com/GPX/1/1"\n'
        '     xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">\n'
        "  <metadata><time>%s</time></metadata>\n"
        "  <trk>\n"
        "    <name>%s</name>\n"
        "    <type>cycling</type>\n"
        "    <trkseg>\n"
    ) % (
        _EPOCH.isoformat().replace("+00:00", "Z"),
        escape("Synthetic Ride" if with_power else "Synthetic HR-only Ride"),
    )

    body = []
    for second in range(_DURATION_S + 1):
        s = _sample(second)
        extensions = ["        <extensions>", "          <gpxtpx:TrackPointExtension>"]
        extensions.append(f"            <gpxtpx:hr>{s['hr']:.0f}</gpxtpx:hr>")
        if with_cadence:
            extensions.append(f"            <gpxtpx:cad>{s['cadence']:.0f}</gpxtpx:cad>")
        extensions.append("          </gpxtpx:TrackPointExtension>")
        if with_power:
            # Strava writes power as a bare element beside the Garmin extension.
            extensions.append(f"          <power>{s['power']:.0f}</power>")
        extensions.append("        </extensions>")

        body.append(
            f'      <trkpt lat="{s["lat"]:.7f}" lon="{s["lon"]:.7f}">\n'
            f"        <ele>{s['ele']:.1f}</ele>\n"
            f"        <time>{s['time'].isoformat().replace('+00:00', 'Z')}</time>\n"
            + "\n".join(extensions)
            + "\n      </trkpt>\n"
        )

    return head + "".join(body) + "    </trkseg>\n  </trk>\n</gpx>\n"


def _tcx() -> str:
    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<TrainingCenterDatabase\n'
        '    xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"\n'
        '    xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2">\n'
        "  <Activities>\n"
        '    <Activity Sport="Biking">\n'
        f"      <Id>{_EPOCH.isoformat().replace('+00:00', 'Z')}</Id>\n"
    )

    laps = []
    for lap_start in range(0, _DURATION_S, _LAP_SPLIT_S):
        lap_end = min(lap_start + _LAP_SPLIT_S, _DURATION_S)
        start = _EPOCH + timedelta(seconds=lap_start)
        points = []
        for second in range(lap_start, lap_end + 1):
            s = _sample(second)
            points.append(
                "          <Trackpoint>\n"
                f"            <Time>{s['time'].isoformat().replace('+00:00', 'Z')}</Time>\n"
                "            <Position>\n"
                f"              <LatitudeDegrees>{s['lat']:.7f}</LatitudeDegrees>\n"
                f"              <LongitudeDegrees>{s['lon']:.7f}</LongitudeDegrees>\n"
                "            </Position>\n"
                f"            <AltitudeMeters>{s['ele']:.1f}</AltitudeMeters>\n"
                f"            <DistanceMeters>{s['distance']:.1f}</DistanceMeters>\n"
                f"            <HeartRateBpm><Value>{s['hr']:.0f}</Value></HeartRateBpm>\n"
                f"            <Cadence>{s['cadence']:.0f}</Cadence>\n"
                "            <Extensions>\n"
                "              <ns3:TPX>\n"
                f"                <ns3:Speed>{s['speed']:.2f}</ns3:Speed>\n"
                f"                <ns3:Watts>{s['power']:.0f}</ns3:Watts>\n"
                "              </ns3:TPX>\n"
                "            </Extensions>\n"
                "          </Trackpoint>\n"
            )
        laps.append(
            f'      <Lap StartTime="{start.isoformat().replace("+00:00", "Z")}">\n'
            f"        <TotalTimeSeconds>{float(lap_end - lap_start):.1f}</TotalTimeSeconds>\n"
            f"        <DistanceMeters>{8.0 * (lap_end - lap_start):.1f}</DistanceMeters>\n"
            "        <Intensity>Active</Intensity>\n"
            "        <TriggerMethod>Manual</TriggerMethod>\n"
            "        <Track>\n" + "".join(points) + "        </Track>\n"
            "      </Lap>\n"
        )

    return (
        head
        + "".join(laps)
        + "      <Notes>Synthetic Ride</Notes>\n"
        "    </Activity>\n  </Activities>\n</TrainingCenterDatabase>\n"
    )


# ── the course fixture (issue #55) ───────────────────────────────────────────
#
# A *course* is a route to be ridden, not a ride that happened: no <time>
# elements at all, ~10 m point spacing, and an elevation profile with real
# structure — flat, a distinct climb, a descent, flat again — so segmentation
# tests have boundaries to find. Distances and elevations are stated here so
# tests can assert against the generator's numbers rather than the parser's.
_COURSE_LENGTH_M = 15_000.0
_COURSE_SPACING_M = 10.0


def _course_elevation(metres: float) -> float:
    if metres < 4000:
        return 20.0
    if metres < 6000:
        return 20.0 + 0.07 * (metres - 4000)   # 2 km climb at 7% → +140 m
    if metres < 7500:
        return 160.0 - 0.06 * (metres - 6000)  # 1.5 km descent at 6% → −90 m
    return 70.0 + 0.005 * (metres - 7500)      # 7.5 km false flat at 0.5%


def _course_gpx(*, with_elevation: bool = True) -> str:
    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx creator="openkoutsi synthetic fixtures" version="1.1"\n'
        '     xmlns="http://www.topografix.com/GPX/1/1">\n'
        "  <trk>\n"
        f"    <name>{escape('Synthetic Course')}</name>\n"
        "    <type>cycling</type>\n"
        "    <trkseg>\n"
    )
    body = []
    metres = 0.0
    while metres <= _COURSE_LENGTH_M:
        lat = _ORIGIN_LAT + metres / _M_PER_DEG_LAT  # due north
        ele = f"        <ele>{_course_elevation(metres):.1f}</ele>\n" if with_elevation else ""
        body.append(
            f'      <trkpt lat="{lat:.7f}" lon="{_ORIGIN_LON:.7f}">\n'
            + ele
            + "      </trkpt>\n"
        )
        metres += _COURSE_SPACING_M
    return head + "".join(body) + "    </trkseg>\n  </trk>\n</gpx>\n"


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for name, text in (
        ("synthetic_ride.gpx", _gpx(with_power=True)),
        ("synthetic_hr_only.gpx", _gpx(with_power=False, with_cadence=False)),
        ("synthetic_ride.tcx", _tcx()),
        ("synthetic_course.gpx", _course_gpx()),
        ("synthetic_course_no_ele.gpx", _course_gpx(with_elevation=False)),
    ):
        path = FIXTURES_DIR / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    # The gzip form a Strava bulk export actually contains.
    gz_path = FIXTURES_DIR / "synthetic_ride.gpx.gz"
    gz_path.write_bytes(gzip.compress((FIXTURES_DIR / "synthetic_ride.gpx").read_bytes()))
    written.append(gz_path)

    for path in written:
        print(f"wrote {path.relative_to(Path(__file__).parent.parent)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
