"""GPX and TCX parsing (issue #36).

Three properties are load-bearing here and each has tests that fail loudly if it
stops holding:

* **A parsed activity carries no location.** GPX is made of coordinates and
  openkoutsi stores none. The parsers read them, derive distance and elevation,
  and drop them — so the test is not "is there a lat field" (there is no such
  field to check) but "does any number the file's coordinates would produce
  appear anywhere in the ``Profile``".
* **The formats agree.** The GPX and TCX fixtures describe the *same* synthetic
  ride, so two independent parsers must land on the same distance, ascent,
  duration and averages. A mixed archive is only trustworthy if they do.
* **A broken file fails with a sentence, not a stack trace.** Every failure mode
  a bulk import will actually hit — empty, truncated, wrong format, a route with
  no timestamps, a DTD — comes back as ``ActivityParseError`` carrying a reason
  the import job can put in front of the athlete.
"""
from __future__ import annotations

import gzip
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openkoutsi import geo, gpx, streams, tcx
from openkoutsi.activity_formats import (
    ActivityParseError,
    detect_format,
    format_capabilities,
    format_priority,
    parser_for,
    strip_compression_suffix,
)
from openkoutsi.fit_processing import resolve_sport_type
from openkoutsi.training_math import calculate_load
from openkoutsi.xmlsafe import (
    XmlSafetyError,
    parse_float,
    reject_doctype,
    root_tag,
)

FIXTURES = Path(__file__).parent.parent.parent / "testdata" / "fixtures"
RIDE_GPX = FIXTURES / "synthetic_ride.gpx"
HR_ONLY_GPX = FIXTURES / "synthetic_hr_only.gpx"
RIDE_TCX = FIXTURES / "synthetic_ride.tcx"

# What `scripts/generate_synthetic_activity_fixtures.py` builds: ten minutes at a
# steady 8 m/s with a single 60 m climb. Stated here rather than read back from
# the parser under test.
RIDE_SECONDS = 600
RIDE_DISTANCE_M = 8.0 * RIDE_SECONDS
RIDE_ASCENT_M = 60.0
RIDE_ORIGIN = (61.5, 20.5)

pytestmark = pytest.mark.skipif(
    not RIDE_GPX.exists(),
    reason="synthetic GPX/TCX fixtures missing — run scripts/generate_synthetic_activity_fixtures.py",
)


def _gpx_document(points: str, *, track_extra: str = "") -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1"'
        ' xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">'
        f"<trk>{track_extra}{points}</trk></gpx>"
    ).encode()


def _trkpt(second: int, *, lat: float = 61.5, lon: float = 20.5, hr: int | None = None,
           ele: float | None = None) -> str:
    time = datetime(2024, 3, 2, 9, 0, second, tzinfo=timezone.utc)
    inner = f"<time>{time.isoformat().replace('+00:00', 'Z')}</time>"
    if ele is not None:
        inner = f"<ele>{ele}</ele>" + inner
    if hr is not None:
        inner += (
            "<extensions><gpxtpx:TrackPointExtension>"
            f"<gpxtpx:hr>{hr}</gpxtpx:hr>"
            "</gpxtpx:TrackPointExtension></extensions>"
        )
    return f'<trkpt lat="{lat}" lon="{lon}">{inner}</trkpt>'


class TestGpxProfile:
    def test_summarises_the_ride(self):
        profile = gpx.summarizeWorkout(str(RIDE_GPX))

        assert profile.start_time == datetime(2024, 3, 2, 9, 0, tzinfo=timezone.utc)
        assert profile.duration == RIDE_SECONDS
        # Derived from coordinates, so agreement is to within the difference
        # between the fixture's flat-earth arithmetic and the parser's haversine.
        assert profile.distance == pytest.approx(RIDE_DISTANCE_M, rel=0.01)
        assert profile.elevationGain == pytest.approx(RIDE_ASCENT_M, abs=2)
        assert profile.name == "Synthetic Ride"
        assert resolve_sport_type(profile.sport_type) == "Ride"

    def test_every_channel_is_on_the_shared_second_grid(self):
        profile = gpx.summarizeWorkout(str(RIDE_GPX))

        lengths = {
            len(profile.heartRate),
            len(profile.speed),
            len(profile.power),
            len(profile.cadence),
            len(profile.altitude),
        }
        # One entry per second of the ride, in every channel — the contract in
        # `openkoutsi.streams`, which every downstream metric indexes as a clock.
        assert lengths == {RIDE_SECONDS + 1}

    def test_speed_is_derived_from_the_track(self):
        profile = gpx.summarizeWorkout(str(RIDE_GPX))
        # 8 m/s = 28.8 km/h. GPX rarely states speed, so this is the parser's
        # own arithmetic over consecutive fixes.
        assert profile.avgSpeed == pytest.approx(28.8, rel=0.01)

    def test_gzip_fixture_matches_the_plain_one(self):
        plain = gpx.summarizeWorkout(str(RIDE_GPX))
        unzipped = gzip.decompress((FIXTURES / "synthetic_ride.gpx.gz").read_bytes())
        from_gz = gpx.summarizeWorkout(unzipped)

        assert from_gz.distance == plain.distance
        assert from_gz.start_time == plain.start_time
        assert from_gz.power == plain.power


class TestLocationIsDropped:
    """The privacy promise, tested as a property of the returned object."""

    def test_no_coordinate_survives_into_the_profile(self):
        route = gpx.extract_route(str(RIDE_GPX))
        profile = gpx.summarizeWorkout(str(RIDE_GPX))

        assert route.points, "fixture should contain coordinates in the first place"
        longitudes = {round(p.longitude, 4) for p in route.points}
        latitudes = {round(p.latitude, 4) for p in route.points}

        for channel in ("heartRate", "speed", "power", "cadence", "altitude"):
            values = {
                round(v, 4) for v in getattr(profile, channel) if v is not None
            }
            assert not values & longitudes
            assert not values & latitudes

    def test_the_serialised_profile_has_no_location_fields(self):
        profile = gpx.summarizeWorkout(str(RIDE_GPX))
        payload = profile.to_json()

        for token in ("lat", "lon", "coord", "position", "gps"):
            assert token not in payload.lower()

    def test_the_profile_type_has_nowhere_to_put_one(self):
        profile = gpx.summarizeWorkout(str(RIDE_GPX))
        for attribute in vars(profile):
            assert "lat" not in attribute.lower()
            assert "lon" not in attribute.lower()


class TestRouteExtraction:
    """The deliberate exception: coordinates, when a caller asks by name."""

    def test_route_keeps_the_geometry(self):
        route = gpx.extract_route(str(RIDE_GPX))

        assert len(route.points) == RIDE_SECONDS + 1
        assert route.points[0].latitude == pytest.approx(RIDE_ORIGIN[0])
        assert route.points[0].longitude == pytest.approx(RIDE_ORIGIN[1])
        assert route.points[0].offset_s == 0
        assert route.points[-1].offset_s == RIDE_SECONDS
        assert route.distance_m == pytest.approx(RIDE_DISTANCE_M, rel=0.01)
        assert route.elevation_gain_m == pytest.approx(RIDE_ASCENT_M, abs=2)
        assert route.name == "Synthetic Ride"

    def test_bounds_cover_every_point(self):
        route = gpx.extract_route(str(RIDE_GPX))
        min_lat, min_lon, max_lat, max_lon = route.bounds()

        assert all(min_lat <= p.latitude <= max_lat for p in route.points)
        assert all(min_lon <= p.longitude <= max_lon for p in route.points)

    def test_empty_route_has_no_bounds(self):
        route = gpx.Route()
        assert route.bounds() is None

    def test_points_without_a_fix_are_left_out(self):
        # (0, 0) is what a receiver writes before it locks on — a point in the
        # Gulf of Guinea would otherwise bound every route in the world.
        document = _gpx_document(
            "<trkseg>"
            + _trkpt(0, lat=0.0, lon=0.0)
            + _trkpt(1, lon=20.5001)
            + _trkpt(2, lon=20.5002)
            + "</trkseg>"
        )
        route = gpx.extract_route(document)
        assert len(route.points) == 2


class TestGpxWithoutPower:
    def test_hr_only_file_has_no_power_channel(self):
        profile = gpx.summarizeWorkout(str(HR_ONLY_GPX))

        assert profile.power == []
        assert profile.cadence == []
        assert profile.heartRate, "the file does carry heart rate"

    def test_hr_only_activity_still_gets_a_load(self):
        profile = gpx.summarizeWorkout(str(HR_ONLY_GPX))
        load, intensity = calculate_load(
            profile.duration, None, profile.avgHeartRate, 250, 190
        )
        # The HR fallback in `calculate_load` is what makes a GPS-and-strap
        # ride a first-class import rather than a failed one. It yields a Load
        # and deliberately no intensity factor — that ratio is defined against
        # FTP, and this ride has no power to compare with it.
        assert load is not None and load > 0
        assert intensity is None


class TestFormatsAgree:
    """The GPX and TCX fixtures are the same ride recorded twice."""

    def test_distance_and_ascent_match(self):
        from_gpx = gpx.summarizeWorkout(str(RIDE_GPX))
        from_tcx = tcx.summarizeWorkout(str(RIDE_TCX))

        assert from_tcx.distance == pytest.approx(from_gpx.distance, rel=0.01)
        assert from_tcx.elevationGain == pytest.approx(from_gpx.elevationGain, abs=2)
        assert from_tcx.duration == from_gpx.duration
        assert from_tcx.start_time == from_gpx.start_time

    def test_averages_match(self):
        from_gpx = gpx.summarizeWorkout(str(RIDE_GPX))
        from_tcx = tcx.summarizeWorkout(str(RIDE_TCX))

        assert from_tcx.avgHeartRate == pytest.approx(from_gpx.avgHeartRate, rel=0.01)
        assert from_tcx.avgPower == pytest.approx(from_gpx.avgPower, rel=0.01)
        assert from_tcx.avgSpeed == pytest.approx(from_gpx.avgSpeed, rel=0.01)


class TestTcx:
    def test_summarises_the_ride(self):
        profile = tcx.summarizeWorkout(str(RIDE_TCX))

        assert profile.duration == RIDE_SECONDS
        # TCX states its own cumulative distance, so this is exact.
        assert profile.distance == pytest.approx(RIDE_DISTANCE_M, abs=1)
        assert resolve_sport_type(profile.sport_type) == "Ride"
        assert profile.name == "Synthetic Ride"

    def test_laps_come_back_as_intervals(self):
        intervals = tcx.extractIntervals(str(RIDE_TCX))

        assert len(intervals) == 2
        assert [iv["duration_s"] for iv in intervals] == [300.0, 300.0]
        assert intervals[0]["start_time"] < intervals[1]["start_time"]
        assert intervals[0]["distance_m"] == pytest.approx(2400, abs=1)

    def test_gpx_has_no_laps_to_extract(self):
        # Not a failure: `[]` is the same answer `fit.extractIntervals` gives
        # for a file with no lap frames, and the caller auto-splits.
        assert gpx.extractIntervals(str(RIDE_GPX)) == []

    def test_start_time_without_a_full_parse(self):
        assert tcx.getStartTime(str(RIDE_TCX)) == datetime(
            2024, 3, 2, 9, 0, tzinfo=timezone.utc
        )
        assert gpx.getStartTime(str(RIDE_GPX)) == datetime(
            2024, 3, 2, 9, 0, tzinfo=timezone.utc
        )

    def test_start_time_of_an_unreadable_file_is_none(self):
        assert gpx.getStartTime(b"not xml at all") is None
        assert tcx.getStartTime(b"") is None


class TestStreamAlignment:
    """A dropout must leave a hole, not shift every later sample (issue #76)."""

    def test_a_missing_sensor_reading_becomes_a_gap(self):
        document = _gpx_document(
            "<trkseg>"
            + _trkpt(0, hr=120)
            + _trkpt(1)  # strap dropped
            + _trkpt(2, hr=124)
            + "</trkseg>"
        )
        profile = gpx.summarizeWorkout(document)

        assert profile.heartRate == [120.0, None, 124.0]
        assert profile.avgHeartRate == pytest.approx(122.0)

    def test_a_recording_gap_widens_the_grid(self):
        document = _gpx_document(
            "<trkseg>" + _trkpt(0, hr=120) + _trkpt(10, hr=130) + "</trkseg>"
        )
        profile = gpx.summarizeWorkout(document)

        assert len(profile.heartRate) == 11
        assert profile.heartRate[0] == 120.0
        assert profile.heartRate[10] == 130.0
        assert all(v is None for v in profile.heartRate[1:10])

    def test_distance_is_not_derived_across_a_segment_break(self):
        # Two segments a long way apart: the device was paused and moved. Only
        # the metres inside each segment were ridden.
        near = _gpx_document(
            "<trkseg>" + _trkpt(0, lon=20.5) + _trkpt(1, lon=20.5001) + "</trkseg>"
            "<trkseg>" + _trkpt(2, lon=20.6) + _trkpt(3, lon=20.6001) + "</trkseg>"
        )
        profile = gpx.summarizeWorkout(near)

        one_step = geo.haversine_m(61.5, 20.5, 61.5, 20.5001)
        assert profile.distance == pytest.approx(2 * one_step, abs=2)


class TestBrokenFiles:
    """Every one of these is something a real bulk import hits."""

    @pytest.mark.parametrize(
        "payload, expected",
        [
            (b"", "empty"),
            (b"   \n  ", "empty"),
            (b"this is not xml", "malformed"),
            (b'<?xml version="1.0"?><gpx><trk><trkseg>', "malformed"),
            (b'<?xml version="1.0"?><gpx version="1.1"></gpx>', "track points"),
        ],
    )
    def test_gpx_failures_carry_a_reason(self, payload, expected):
        with pytest.raises(ActivityParseError) as exc:
            gpx.summarizeWorkout(payload)
        assert expected in str(exc.value).lower()

    def test_a_route_file_is_not_an_activity(self):
        # A GPX with coordinates but no timestamps is a planned route. It has no
        # clock, so it cannot become an activity — and saying so is much more
        # use than an activity dated 1970.
        document = _gpx_document(
            '<trkseg><trkpt lat="61.5" lon="20.5"><ele>10</ele></trkpt>'
            '<trkpt lat="61.5" lon="20.51"><ele>12</ele></trkpt></trkseg>'
        )
        with pytest.raises(ActivityParseError, match="no timestamps"):
            gpx.summarizeWorkout(document)

        # ...but its geometry is still extractable, which is exactly what a
        # route-analysis feature would want from one.
        route = gpx.extract_route(document)
        assert len(route.points) == 2
        assert route.start_time is None

    @pytest.mark.parametrize(
        "payload, expected",
        [
            (b"", "empty"),
            (b'<?xml version="1.0"?><TrainingCenterDatabase></TrainingCenterDatabase>', "track points"),
            (b'<?xml version="1.0"?><gpx><trk></trk></gpx>', "tcx"),
        ],
    )
    def test_tcx_failures_carry_a_reason(self, payload, expected):
        with pytest.raises(ActivityParseError) as exc:
            tcx.summarizeWorkout(payload)
        assert expected in str(exc.value).lower()

    def test_a_tcx_course_is_not_an_activity(self):
        """Garmin Connect exports a *planned route* as `.tcx`.

        `Courses/Course/Track/Trackpoint` matches `Trackpoint` on local name
        exactly as an activity's does, so without a guard a course parses as a
        ride nobody rode — no Load (a course has no HR or power), but a phantom
        row with a distance and a duration, and a derived speed stream that can
        reach the distance bests and produce a PR from a ride that never
        happened. It is also exactly the sort of file that ends up in a folder
        someone drags into a bulk import.
        """
        course = (
            '<?xml version="1.0"?><TrainingCenterDatabase><Courses><Course>'
            "<Name>Sunday Loop</Name>"
            "<Lap><TotalTimeSeconds>1800</TotalTimeSeconds>"
            "<DistanceMeters>5000</DistanceMeters></Lap><Track>"
            "<Trackpoint><Time>2024-03-02T09:00:00Z</Time>"
            "<Position><LatitudeDegrees>61.5</LatitudeDegrees>"
            "<LongitudeDegrees>20.5</LongitudeDegrees></Position>"
            "<DistanceMeters>0</DistanceMeters></Trackpoint>"
            "<Trackpoint><Time>2024-03-02T09:30:00Z</Time>"
            "<Position><LatitudeDegrees>61.55</LatitudeDegrees>"
            "<LongitudeDegrees>20.5</LongitudeDegrees></Position>"
            "<DistanceMeters>5000</DistanceMeters></Trackpoint>"
            "</Track></Course></Courses></TrainingCenterDatabase>"
        ).encode()

        with pytest.raises(ActivityParseError, match="course"):
            tcx.summarizeWorkout(course)

    def test_every_format_refuses_a_planned_route(self):
        """The same policy in all three shapes a route arrives in."""
        route_gpx = (
            '<?xml version="1.0"?><gpx><rte>'
            '<rtept lat="61.5" lon="20.5"><ele>10</ele></rtept></rte></gpx>'
        )
        untimed_gpx = _gpx_document(
            '<trkseg><trkpt lat="61.5" lon="20.5"><ele>10</ele></trkpt></trkseg>'
        )
        for document, parser in (
            (route_gpx.encode(), gpx),
            (untimed_gpx, gpx),
        ):
            with pytest.raises(ActivityParseError):
                parser.summarizeWorkout(document)

    def test_a_truncated_file_fails_cleanly(self):
        whole = RIDE_GPX.read_bytes()
        with pytest.raises(ActivityParseError, match="[Mm]alformed"):
            gpx.summarizeWorkout(whole[: len(whole) // 2])


class TestXmlSafety:
    def test_a_document_type_declaration_is_refused(self):
        # The billion laughs: ten nested entities, each ten copies of the last.
        bomb = (
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE gpx [<!ENTITY a "aaaaaaaaaa">'
            b'<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>'
            b"<gpx><trk><name>&b;</name></trk></gpx>"
        )
        with pytest.raises(XmlSafetyError, match="document type"):
            reject_doctype(bomb)
        with pytest.raises(ActivityParseError, match="document type"):
            gpx.summarizeWorkout(bomb)

    def test_an_external_entity_is_refused_with_the_declaration(self):
        xxe = (
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE gpx [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            b"<gpx><trk><name>&xxe;</name></trk></gpx>"
        )
        with pytest.raises(ActivityParseError, match="document type"):
            gpx.summarizeWorkout(xxe)

    def test_comments_and_processing_instructions_are_fine(self):
        document = (
            b'<?xml version="1.0"?><!-- exported by something --><?display mode="x"?>'
            + _gpx_document("<trkseg>" + _trkpt(0, hr=120) + "</trkseg>")[21:]
        )
        reject_doctype(document)  # does not raise

    def test_the_word_doctype_in_a_track_name_is_not_a_declaration(self):
        document = _gpx_document(
            "<trkseg>" + _trkpt(0, hr=120) + _trkpt(1, hr=121) + "</trkseg>",
            track_extra="<name>&lt;!DOCTYPE ride</name>",
        )
        profile = gpx.summarizeWorkout(document)
        assert profile.name == "<!DOCTYPE ride"

    @pytest.mark.parametrize(
        "encoding",
        ["utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be"],
    )
    @pytest.mark.parametrize("declaration", ['<?xml version="1.0"?>', ""])
    def test_a_wide_encoding_is_refused_rather_than_scanned_through(
        self, encoding, declaration
    ):
        """The prolog scan compares bytes, so it can only speak for ASCII.

        In UTF-16 the markup is ``3C 00 21 00 44 00 …``, so every literal in
        `reject_doctype` silently fails to match and a document carrying a DTD
        is reported as having a clean prolog. `root_tag` misses it too, so
        `detect_format` falls back to the filename and the file is parsed
        anyway. The guarantee has to be one check with no holes in it, so these
        are refused instead of scanned through a decoder.

        Parametrised **without** the XML declaration as well as with it: the
        declaration is optional, a document may open straight into
        ``<!DOCTYPE``, and a first cut of this guard recognised the BOM-less
        forms by their ``<?`` — which left exactly that shape unscanned.
        """
        bomb = (
            declaration
            + '<!DOCTYPE gpx [<!ENTITY a "' + "A" * 64 + '">'
            '<!ENTITY b "' + "&a;" * 10 + '">]>'
            "<gpx><trk><name>&b;</name></trk></gpx>"
        ).encode(encoding)

        with pytest.raises(XmlSafetyError, match="ASCII-compatible"):
            reject_doctype(bomb)
        with pytest.raises(ActivityParseError, match="ASCII-compatible"):
            gpx.summarizeWorkout(bomb)

    def test_a_utf8_bom_is_still_fine(self):
        document = b"\xef\xbb\xbf" + _gpx_document(
            "<trkseg>" + _trkpt(0, hr=120) + _trkpt(1, hr=121) + "</trkseg>"
        )
        assert gpx.summarizeWorkout(document).duration == 1

    def test_root_tag_reads_only_the_head(self):
        assert root_tag(RIDE_GPX.read_bytes()) == "gpx"
        assert root_tag(RIDE_TCX.read_bytes()) == "TrainingCenterDatabase"
        assert root_tag(b"\x00\x01\x02") is None


class TestFormatDetection:
    def test_content_wins_over_the_name(self):
        gpx_bytes = RIDE_GPX.read_bytes()
        # A GPX that someone renamed. The bytes are not an opinion.
        assert detect_format(gpx_bytes, "ride.tcx") == "gpx"
        assert detect_format(RIDE_TCX.read_bytes(), "ride.gpx") == "tcx"

    def test_fit_is_detected_by_its_magic(self):
        fit = next(FIXTURES.glob("*.fit"), None)
        if fit is None:
            pytest.skip("no FIT fixtures available")
        assert detect_format(fit.read_bytes(), fit.name) == "fit"

    def test_the_name_decides_when_the_content_does_not(self):
        assert detect_format(b"garbage", "ride.gpx") == "gpx"
        assert detect_format(b"garbage", "ride.fit") == "fit"
        assert detect_format(b"garbage", "notes.txt") is None
        assert detect_format(b"garbage", None) is None

    def test_gzip_suffix_is_stripped_for_naming(self):
        assert strip_compression_suffix("ride.fit.gz") == "ride.fit"
        assert strip_compression_suffix("ride.gpx") == "ride.gpx"

    def test_richest_format_sorts_first(self):
        # The preference order in-batch deduplication uses when a Strava export
        # holds the same ride three times.
        assert format_priority("fit") < format_priority("tcx") < format_priority("gpx")
        assert format_priority("nonsense") > format_priority("gpx")

    def test_capabilities_describe_what_a_format_can_carry(self):
        assert format_capabilities("fit")["power"] is True
        assert format_capabilities("gpx")["power"] is False
        assert format_capabilities("gpx")["laps"] is False
        assert format_capabilities("tcx")["laps"] is True

    def test_every_format_offers_the_same_three_functions(self):
        for fmt in ("fit", "gpx", "tcx"):
            module = parser_for(fmt)
            for function in ("summarizeWorkout", "getStartTime", "extractIntervals"):
                assert callable(getattr(module, function))

    def test_an_unknown_format_is_a_parse_error(self):
        with pytest.raises(ActivityParseError, match="Unsupported"):
            parser_for("csv")


class TestGeo:
    def test_haversine_against_a_known_distance(self):
        # Helsinki to Tampere, ~160 km as the crow flies.
        distance = geo.haversine_m(60.1699, 24.9384, 61.4978, 23.7610)
        assert distance == pytest.approx(160_000, rel=0.02)

    def test_a_degree_of_latitude_is_about_111_km(self):
        assert geo.haversine_m(60.0, 25.0, 61.0, 25.0) == pytest.approx(111_195, rel=0.001)

    def test_zero_distance(self):
        assert geo.haversine_m(61.5, 20.5, 61.5, 20.5) == 0.0

    def test_a_stationary_receiver_banks_no_ascent(self):
        # ±2 m of wander with a ~19 s period, for an hour: a receiver sitting
        # still. Summing positive deltas would call this hundreds of metres of
        # climbing, which is exactly the bug this function exists to not have.
        drift = [10.0 + 2.0 * math.sin(i / 3) for i in range(3600)]
        assert sum(max(0.0, b - a) for a, b in zip(drift, drift[1:])) > 500
        assert geo.elevation_gain_m(drift) == 0.0

    def test_a_steady_climb_is_counted_in_full(self):
        climb = [float(i) / 10 for i in range(4000)]  # 400 m, 0.1 m at a time
        assert geo.elevation_gain_m(climb) == pytest.approx(400, abs=5)

    def test_descent_does_not_subtract(self):
        assert geo.elevation_gain_m([0, 100, 0, 100]) == pytest.approx(200, abs=1)

    def test_gaps_in_the_elevation_series_are_skipped(self):
        assert geo.elevation_gain_m([0, None, 50, None, 100]) == pytest.approx(100, abs=1)

    def test_a_teleporting_fix_is_not_distance(self):
        # One point in the middle of the Atlantic between two in the Baltic: a
        # receiver reacquiring, not a 14 000 km round trip in two seconds. Both
        # steps are rejected, so the few metres actually ridden around the
        # glitch are lost too — a bounded, obviously-safe trade against a total
        # that would otherwise be wrong by four orders of magnitude.
        glitched = [(61.5, 20.5), (0.5, -30.0), (61.5, 20.5001)]
        assert geo.track_distance_m(glitched) < 100

        clean = [(61.5, 20.5), (61.5, 20.5001)]
        assert geo.track_distance_m(clean) == pytest.approx(
            geo.haversine_m(61.5, 20.5, 61.5, 20.5001), abs=0.1
        )

    def test_points_with_no_fix_carry_the_distance_forward(self):
        cumulative = geo.cumulative_distance_m(
            [(61.5, 20.5), None, (61.5, 20.5001)]
        )
        assert len(cumulative) == 3
        assert cumulative[0] == 0.0
        assert cumulative[1] == 0.0
        assert cumulative[2] > 0.0


class TestNumbersThatAreNotNumbers:
    """``float()`` accepts NaN and Infinity; a training file cannot mean them.

    Left alone they reach ``int()`` several layers later as an uncaught
    ``ValueError``/``OverflowError``, which the import job then shows the
    athlete as the wording of a Python exception, and which makes
    ``POST /activities/{id}/reprocess`` a 500.
    """

    @pytest.mark.parametrize("text", ["NaN", "nan", "inf", "-inf", "Infinity", "+inf"])
    def test_parse_float_rejects_them(self, text):
        assert parse_float(text) is None

    def test_parse_float_still_reads_ordinary_numbers(self):
        assert parse_float("12.5") == 12.5
        assert parse_float(" -3 ") == -3.0
        assert parse_float("nonsense") is None
        assert parse_float(None) is None

    @pytest.mark.parametrize(
        "total, distance",
        [("NaN", "100"), ("inf", "100"), ("600", "inf"), ("600", "NaN")],
    )
    def test_tcx_survives_them(self, total, distance):
        document = (
            '<?xml version="1.0"?><TrainingCenterDatabase><Activities>'
            '<Activity Sport="Biking"><Lap StartTime="2024-03-02T09:00:00Z">'
            f"<TotalTimeSeconds>{total}</TotalTimeSeconds>"
            f"<DistanceMeters>{distance}</DistanceMeters><Track>"
            f'<Trackpoint><Time>2024-03-02T09:00:00Z</Time>'
            f"<DistanceMeters>{distance}</DistanceMeters></Trackpoint>"
            f'<Trackpoint><Time>2024-03-02T09:00:01Z</Time>'
            f"<DistanceMeters>{distance}</DistanceMeters></Trackpoint>"
            "</Track></Lap></Activity></Activities></TrainingCenterDatabase>"
        ).encode()

        profile = tcx.summarizeWorkout(document)
        assert profile.duration >= 0
        assert profile.distance >= 0

    def test_a_gpx_sensor_reading_of_nan_is_dropped(self):
        document = _gpx_document(
            "<trkseg>"
            + _trkpt(0, hr=120).replace(
                "<gpxtpx:hr>120</gpxtpx:hr>", "<gpxtpx:hr>NaN</gpxtpx:hr>"
            )
            + _trkpt(1, hr=121)
            + "</trkseg>"
        )
        profile = gpx.summarizeWorkout(document)
        # A gap, not a NaN masquerading as a reading.
        assert profile.heartRate == [None, 121.0]


class TestDurationIsBounded:
    """One bad timestamp must not become a duration of centuries.

    The streams are already clamped by `MAX_STREAM_SECONDS`; the duration is
    derived from the same grid so the two cannot disagree. Without it, a head
    unit with a flat backup battery writing one 2099 stamp produces a Load in
    the billions, which `recalculate_from` then folds into the athlete's whole
    history.
    """

    def test_a_far_future_timestamp_does_not_stretch_the_duration(self):
        document = _gpx_document(
            "<trkseg>"
            + _trkpt(0, hr=120)
            + _trkpt(1, hr=121)
            + '<trkpt lat="61.5" lon="20.5"><time>9999-12-31T23:59:59Z</time></trkpt>'
            + "</trkseg>"
        )
        profile = gpx.summarizeWorkout(document)

        assert profile.duration == 1
        assert len(profile.heartRate) == 2

        load, _ = calculate_load(profile.duration, None, profile.avgHeartRate, 250, 190)
        assert load < 1

    def test_a_ride_longer_than_the_cap_is_clamped_not_rejected(self):
        document = _gpx_document(
            "<trkseg>"
            + _trkpt(0, hr=120)
            + '<trkpt lat="61.5" lon="20.5"><time>2024-03-05T09:00:00Z</time></trkpt>'
            + "</trkseg>"
        )
        profile = gpx.summarizeWorkout(document)
        assert profile.duration <= streams.MAX_STREAM_SECONDS

    def test_an_absurd_lap_total_is_clamped_too(self):
        # TCX has a second route to the duration: the laps' own timer time.
        document = (
            '<?xml version="1.0"?><TrainingCenterDatabase><Activities>'
            '<Activity Sport="Biking"><Lap StartTime="2024-03-02T09:00:00Z">'
            "<TotalTimeSeconds>99999999999</TotalTimeSeconds><Track>"
            "<Trackpoint><Time>2024-03-02T09:00:00Z</Time></Trackpoint>"
            "<Trackpoint><Time>2024-03-02T09:00:01Z</Time></Trackpoint>"
            "</Track></Lap></Activity></Activities></TrainingCenterDatabase>"
        ).encode()
        assert tcx.summarizeWorkout(document).duration == streams.MAX_STREAM_SECONDS


class TestNameIsBounded:
    """``<name>`` becomes a String column every list endpoint echoes back."""

    def test_a_huge_track_name_is_truncated(self):
        document = _gpx_document(
            "<trkseg>" + _trkpt(0, hr=120) + _trkpt(1, hr=121) + "</trkseg>",
            track_extra=f"<name>{'x' * 100_000}</name>",
        )
        name = gpx.summarizeWorkout(document).name
        assert name is not None and len(name) <= 120

    def test_a_huge_tcx_note_is_not_used_as_a_name(self):
        document = (
            '<?xml version="1.0"?><TrainingCenterDatabase><Activities>'
            '<Activity Sport="Biking"><Lap StartTime="2024-03-02T09:00:00Z">'
            "<TotalTimeSeconds>1</TotalTimeSeconds><Track>"
            "<Trackpoint><Time>2024-03-02T09:00:00Z</Time></Trackpoint>"
            "<Trackpoint><Time>2024-03-02T09:00:01Z</Time></Trackpoint>"
            "</Track></Lap>"
            f"<Notes>{'x' * 100_000}</Notes>"
            "</Activity></Activities></TrainingCenterDatabase>"
        ).encode()
        assert tcx.summarizeWorkout(document).name is None


class TestBoundedParseCost:
    """Two claims the parsers make about their own cost, kept honest."""

    def test_unwanted_elements_do_not_accumulate(self):
        import resource

        body = "<wpt><name>x</name></wpt>" * 200_000
        document = f'<?xml version="1.0"?><gpx>{body}</gpx>'.encode()

        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with pytest.raises(ActivityParseError):
            gpx.summarizeWorkout(document)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Holding every unwanted element cost roughly 8x the file; the bound is
        # generous so this checks the shape rather than a number.
        assert (after - before) / 1024 < len(document) / 1e6

    def test_start_time_is_the_first_track_point_not_the_files_creation_date(self):
        """GPX 1.1 puts the file's creation date in ``<metadata><time>``.

        It comes before ``<trk>``, so a parser that stops at the first ``<time>``
        anywhere returns the moment the file was written rather than the moment
        the ride started. The activity row would still be right — it comes from
        the full parse — but deduplication runs entirely on *this* function, so
        a disagreement wider than the duplicate window silently stops
        re-imports skipping and stops a ride held as both FIT and GPX
        collapsing. No error anywhere; just duplicates.
        """
        document = (
            '<?xml version="1.0"?><gpx>'
            "<metadata><time>2020-01-01T00:00:00Z</time></metadata>"
            "<trk><trkseg>"
            + _trkpt(0)
            + _trkpt(1)
            + "</trkseg></trk></gpx>"
        ).encode()

        assert gpx.getStartTime(document) == gpx.summarizeWorkout(document).start_time
        assert gpx.getStartTime(document) == datetime(
            2024, 3, 2, 9, 0, tzinfo=timezone.utc
        )

    def test_start_time_agrees_with_a_full_parse_on_every_fixture(self):
        for module, path in ((gpx, RIDE_GPX), (tcx, RIDE_TCX), (gpx, HR_ONLY_GPX)):
            assert module.getStartTime(str(path)) == module.summarizeWorkout(
                str(path)
            ).start_time

    def test_start_time_does_not_parse_the_whole_file(self, monkeypatch):
        """``getStartTime`` must stop at the first timestamp, not read on.

        A bulk import calls it for *every* file before it can decide which are
        duplicates, so a full parse here costs the archive twice over.

        Counted in parse events rather than timed. The wall-clock version of
        this test — ``cheap < full / 3`` over ``perf_counter`` — measured a
        millisecond of work on a shared runner and failed reproducibly in CI
        while passing everywhere else, reporting two bare floats and nothing
        that would explain them. Events are the thing the name actually
        claims, and they do not depend on what else the machine is doing.
        """
        from openkoutsi import xmlsafe

        real_iterparse = xmlsafe.ET.iterparse

        def counting_iterparse(*args, **kwargs):
            for event in real_iterparse(*args, **kwargs):
                counted[0] += 1
                yield event

        monkeypatch.setattr(xmlsafe.ET, "iterparse", counting_iterparse)

        def events(fn, arg) -> int:
            counted[0] = 0
            fn(arg)
            return counted[0]

        counted = [0]
        for module, path in ((gpx, RIDE_GPX), (tcx, RIDE_TCX)):
            cheap = events(module.getStartTime, str(path))
            full = events(module.summarizeWorkout, str(path))
            assert cheap < full / 3, (
                f"{module.__name__}.getStartTime parsed {cheap} events where a "
                f"full parse of {path.name} takes {full} — it is reading on "
                f"past the first timestamp"
            )
