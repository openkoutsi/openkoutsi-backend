"""Course recon math (issue #55): thinning, smoothing, segmentation, physics.

The load-bearing properties:

* Thinning preserves the shape of the course — endpoints, total distance —
  while enforcing the target spacing.
* Elevation is smoothed over a *distance* window before any gradient is
  computed, and recovers a known gradient from a noisy synthetic profile.
* Segmentation splits where gradient meaningfully changes and never shatters
  a course into sub-200 m confetti; segments tile the course exactly.
* The physics solves the steady-state power balance to hand-computed values,
  and the target-time inverse *refuses* impossible targets with a reason code
  rather than returning a power nobody can hold.
"""
from __future__ import annotations

import math
import random

import pytest

from openkoutsi import course, geo
from openkoutsi.gpx import Route, RoutePoint

# One degree of latitude is ~111 320 m, so tracks are laid out due north with
# a known metre spacing. The origin is open water (Gulf of Bothnia) and
# identifies nobody — same convention as the activity fixtures.
_ORIGIN_LAT = 61.5
_ORIGIN_LON = 20.5
_M_PER_DEG_LAT = 111_320.0


def _route(spacing_m: float, count: int, elevation) -> Route:
    """A due-north track with ``count`` points ``spacing_m`` apart.

    ``elevation`` is a callable of distance in metres (or None for no data).
    """
    points = []
    for i in range(count):
        d = i * spacing_m
        points.append(
            RoutePoint(
                latitude=_ORIGIN_LAT + d / _M_PER_DEG_LAT,
                longitude=_ORIGIN_LON,
                elevation_m=None if elevation is None else elevation(d),
            )
        )
    return Route(points=points)


def _profile(spacing_m: float, total_m: float, gradient_at) -> course.CourseProfile:
    """A synthetic profile built directly, bypassing smoothing.

    ``gradient_at(d)`` gives the gradient of the interval starting at ``d``.
    """
    points = [course.ProfilePoint(distance_m=0.0, elevation_m=100.0, gradient=gradient_at(0.0))]
    d = 0.0
    while d < total_m - 1e-9:
        g = gradient_at(d)
        d2 = min(d + spacing_m, total_m)
        points.append(
            course.ProfilePoint(
                distance_m=d2,
                elevation_m=points[-1].elevation_m + g * (d2 - d),
                gradient=g,
            )
        )
        d = d2
    return course.CourseProfile(points=points, total_distance_m=total_m)


RIDER = course.RiderParams(ftp_w=250.0, weight_kg=75.0)
BIKE = course.BikeParams(tyre_width_mm=28, riding_position="hoods")
TOTAL_MASS = RIDER.weight_kg + course.BIKE_AND_KIT_MASS_KG


class TestThinning:
    def test_spacing_lands_between_one_and_two_targets(self):
        route = _route(1.0, 2001, lambda d: 100.0)  # 2 km at 1 m spacing
        track = course.thin_track(route)
        gaps = [
            b.distance_m - a.distance_m
            for a, b in zip(track.points, track.points[1:-1])
        ]
        assert all(course.THIN_SPACING_M <= g < 2 * course.THIN_SPACING_M for g in gaps)

    def test_endpoints_are_kept_and_distance_is_preserved(self):
        route = _route(1.0, 1001, lambda d: 100.0)
        track = course.thin_track(route)
        assert track.points[0].distance_m == 0.0
        assert track.total_distance_m == pytest.approx(1000.0, rel=0.01)

    def test_a_sparse_route_keeps_its_whole_distance(self):
        """A planner's export is spaced by direction change, not by the clock.

        ``thin_track`` re-derives distance rather than trusting ``Route``, so
        it has to apply the glitch rule itself — and applying it as a bare
        metre cap silently shortened long straights, which then shortened
        every gradient, segment and pacing split computed from them.
        """
        spacing = 2_000.0  # 2 km between consecutive points
        route = _route(spacing, 11, lambda d: 100.0)
        # Three minutes a leg: 40 km/h, quick but entirely rideable.
        route = Route(
            points=[
                RoutePoint(
                    latitude=p.latitude,
                    longitude=p.longitude,
                    elevation_m=p.elevation_m,
                    offset_s=i * 180,
                )
                for i, p in enumerate(route.points)
            ]
        )
        track = course.thin_track(route)

        assert track.total_distance_m == pytest.approx(10 * spacing, rel=0.01)

    def test_a_sparse_route_with_no_times_still_keeps_its_distance(self):
        # A course file states no clock at all, which is the common case for
        # the upload path; the timeless fallback must still pass 2 km legs.
        track = course.thin_track(_route(2_000.0, 11, lambda d: 100.0))
        assert track.total_distance_m == pytest.approx(20_000.0, rel=0.01)

    def test_an_empty_route_yields_an_empty_track(self):
        assert course.thin_track(Route(points=[])).points == []

    def test_a_single_point_survives(self):
        track = course.thin_track(_route(1.0, 1, lambda d: 100.0))
        assert len(track.points) == 1

    def test_duplicate_points_do_not_produce_zero_length_intervals(self):
        point = RoutePoint(latitude=_ORIGIN_LAT, longitude=_ORIGIN_LON, elevation_m=10.0)
        route = Route(points=[point] * 50)
        track = course.thin_track(route)
        assert len(track.points) == 1

    def test_a_repeated_point_object_does_not_end_the_track_early(self):
        """A closed loop may write its start coordinate again at the end, and
        a parser may hand back the same object twice. The last point is the
        last *position*, never whichever object happens to compare identical."""
        def at(metres):
            return RoutePoint(
                latitude=_ORIGIN_LAT + metres / _M_PER_DEG_LAT,
                longitude=_ORIGIN_LON,
                elevation_m=10.0 + metres * 0.01,
            )

        start = at(0)
        # Out 500 m and back again in 1 m steps, closing on the *same object*
        # the route started with.
        loop = [at(d) for d in range(1, 501)] + [at(d) for d in range(499, 0, -1)]
        track = course.thin_track(Route(points=[start, *loop, start]))

        assert len(track.points) > 50
        # The loop closes: the last point is back at the start coordinate,
        # with the whole 1 km ridden behind it.
        assert track.points[-1].latitude == pytest.approx(_ORIGIN_LAT)
        assert track.points[-1].distance_m == pytest.approx(1000, rel=0.02)

    def test_the_point_count_is_bounded_however_long_the_course(self):
        """An unbounded point count is an unbounded stored track: the thinned
        series is one JSON row that every re-analysis re-materialises."""
        # 1000 km at 10 m spacing would be 100k points at the 8 m target.
        route = _route(10.0, 100_001, lambda d: 100.0)
        track = course.thin_track(route)
        assert len(track.points) <= course.MAX_THINNED_POINTS
        # Still the whole course, just sampled more coarsely.
        assert track.total_distance_m == pytest.approx(1_000_000, rel=0.01)
        gaps = [
            b.distance_m - a.distance_m
            for a, b in zip(track.points, track.points[1:-1])
        ]
        assert all(g >= course.THIN_SPACING_M for g in gaps)

    def test_an_ordinary_course_is_not_coarsened(self):
        route = _route(1.0, 20_001, lambda d: 100.0)  # 20 km
        track = course.thin_track(route)
        gaps = [
            b.distance_m - a.distance_m
            for a, b in zip(track.points, track.points[1:-1])
        ]
        assert all(course.THIN_SPACING_M <= g < 2 * course.THIN_SPACING_M for g in gaps)

    def test_missing_elevations_are_interpolated_between_known_neighbours(self):
        def ele(d):
            return None if 300 <= d <= 500 else 0.10 * d

        track = course.thin_track(_route(1.0, 1001, ele))
        for p in track.points:
            assert p.elevation_m == pytest.approx(0.10 * p.distance_m, abs=1.5)

    def test_a_track_with_no_elevation_keeps_none_throughout(self):
        track = course.thin_track(_route(1.0, 1001, None))
        assert all(p.elevation_m is None for p in track.points)


class TestDistanceSmoothing:
    def test_matches_the_sample_smoother_on_uniform_spacing(self):
        rng = random.Random(7)
        values = [rng.uniform(0, 100) for _ in range(200)]
        distances = [float(i) for i in range(200)]
        # A 6 m window at 1 m spacing spans the same seven points as the
        # seven-sample window at every index, edges included.
        assert geo.smoothed_by_distance(values, distances, 6.0) == pytest.approx(
            geo._smoothed(values, 7)
        )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            geo.smoothed_by_distance([1.0, 2.0], [0.0], 10.0)

    def test_a_non_positive_window_is_a_passthrough(self):
        values = [3.0, 1.0, 4.0, 1.0]
        assert geo.smoothed_by_distance(values, [0.0, 1.0, 2.0, 3.0], 0.0) == values


class TestProfile:
    def test_recovers_a_known_gradient_from_a_noisy_elevation(self):
        # 5% grade with 2 m amplitude noise at a 30 m wavelength — much
        # faster than terrain, exactly what the 60 m window must remove.
        def ele(d):
            return 100.0 + 0.05 * d + 2.0 * math.sin(2 * math.pi * d / 30.0)

        track = course.thin_track(_route(1.0, 2001, ele))
        profile, reason = course.course_profile(track)
        assert reason is None
        interior = [p for p in profile.points if 100 < p.distance_m < 1900]
        for p in interior:
            assert p.gradient == pytest.approx(0.05, abs=0.01)

    def test_random_noise_is_averaged_below_the_split_threshold(self):
        rng = random.Random(42)

        def ele(d):
            return 100.0 + 0.05 * d + rng.uniform(-1.0, 1.0)

        track = course.thin_track(_route(8.0, 251, ele))
        profile, reason = course.course_profile(track)
        assert reason is None
        interior = [p.gradient for p in profile.points if 100 < p.distance_m < 1900]
        mean_error = sum(abs(g - 0.05) for g in interior) / len(interior)
        assert mean_error < 0.01

    def test_no_elevation_is_a_reason_code_not_an_exception(self):
        track = course.thin_track(_route(1.0, 1001, None))
        profile, reason = course.course_profile(track)
        assert profile is None
        assert reason == "no_elevation_data"

    def test_a_short_course_is_a_reason_code(self):
        track = course.thin_track(_route(1.0, 301, lambda d: 100.0))
        profile, reason = course.course_profile(track)
        assert profile is None
        assert reason == "course_too_short"


class TestChartProfile:
    """The chart payload: evenly spaced samples, at most the cap, ends exact.

    Even spacing is the property the chart depends on, not a nicety. A profile
    chart draws one mark per point and sizes every mark from the *smallest* gap
    in the series, so a payload that repeats a distance sizes them all to zero
    and draws nothing: correct axes, correct selection highlight, empty plot.
    Snapping each grid target to the last source point at or below it did
    exactly that on any track sparser than the grid — which a route planner's
    export is, along every straight it draws with two points.
    """

    def _uneven(self) -> list[course.ProfilePoint]:
        """A planner's spacing: 10 m through the turns, 500 m on the straights."""
        points: list[course.ProfilePoint] = []
        d = 0.0
        while d <= 15_000.0:
            points.append(
                course.ProfilePoint(
                    distance_m=d, elevation_m=100.0 + 0.05 * d, gradient=0.05
                )
            )
            d += 10.0 if (d % 1500.0) < 500.0 else 500.0
        return points

    def _gaps(self, points) -> list[float]:
        return [b.distance_m - a.distance_m for a, b in zip(points, points[1:])]

    def test_a_sparse_track_never_repeats_a_distance(self):
        source = self._uneven()
        assert len(source) > course.CHART_PROFILE_MAX_POINTS  # the case that broke
        out = course._resample_profile(source)

        assert len(out) == course.CHART_PROFILE_MAX_POINTS
        assert min(self._gaps(out)) > 0  # a repeat is a chart that draws nothing

    def test_the_grid_is_even(self):
        out = course._resample_profile(self._uneven())
        gaps = self._gaps(out)
        assert max(gaps) == pytest.approx(min(gaps), rel=1e-9)

    def test_the_ends_are_the_sources_own_ends(self):
        source = self._uneven()
        out = course._resample_profile(source)
        for sampled, original in ((out[0], source[0]), (out[-1], source[-1])):
            assert sampled.distance_m == pytest.approx(original.distance_m)
            assert sampled.elevation_m == pytest.approx(original.elevation_m)

    def test_a_grid_point_between_samples_is_the_line_between_them(self):
        # 200 m of nothing between two samples: the chart draws a straight line
        # across it either way, so the resampled points sit on that line.
        source = [
            course.ProfilePoint(distance_m=0.0, elevation_m=100.0, gradient=0.0),
            course.ProfilePoint(distance_m=200.0, elevation_m=120.0, gradient=0.10),
            course.ProfilePoint(distance_m=400.0, elevation_m=120.0, gradient=0.0),
        ]
        out = course._resample_profile(source)
        assert len(out) == 3  # never more samples than the source had
        assert out[1].distance_m == pytest.approx(200.0)
        assert out[1].elevation_m == pytest.approx(120.0)

    def test_a_dense_track_keeps_its_shape(self):
        # 15 km at 8 m spacing: the grid is coarser than the source, so every
        # sample is an interpolation between neighbours a few metres apart.
        source = [
            course.ProfilePoint(distance_m=d * 8.0, elevation_m=100.0 + 0.04 * d * 8.0, gradient=0.04)
            for d in range(1876)
        ]
        out = course._resample_profile(source)
        assert len(out) == course.CHART_PROFILE_MAX_POINTS
        for p in out:
            assert p.elevation_m == pytest.approx(100.0 + 0.04 * p.distance_m, abs=0.01)
            assert p.gradient == pytest.approx(0.04)

    def test_a_short_series_is_not_padded_with_invented_samples(self):
        source = [
            course.ProfilePoint(distance_m=i * 100.0, elevation_m=100.0, gradient=0.0)
            for i in range(20)
        ]
        assert len(course._resample_profile(source)) == 20

    @pytest.mark.parametrize(
        "source",
        [
            [],
            [course.ProfilePoint(distance_m=0.0, elevation_m=10.0, gradient=0.0)],
            # A track that never moves: no distance to lay a grid over.
            [
                course.ProfilePoint(distance_m=5.0, elevation_m=10.0, gradient=0.0),
                course.ProfilePoint(distance_m=5.0, elevation_m=11.0, gradient=0.0),
            ],
        ],
    )
    def test_degenerate_series_are_handed_back_rather_than_divided_by_zero(self, source):
        assert course._resample_profile(source) == source

    def test_the_analysis_payload_is_even_for_a_planner_style_route(self):
        # End to end from a GPX-shaped route: uneven spacing in, even out.
        def ele(d):
            return 100.0 + (0.06 * (d - 4000) if 4000 <= d < 6000 else 0.0)

        points = []
        d = 0.0
        while d <= 15_000.0:
            points.append(
                RoutePoint(
                    latitude=_ORIGIN_LAT + d / _M_PER_DEG_LAT,
                    longitude=_ORIGIN_LON,
                    elevation_m=ele(min(d, 6000.0)),
                )
            )
            d += 10.0 if (d % 1500.0) < 500.0 else 500.0

        profile, reason = course.course_profile(course.thin_track(Route(points=points)))
        assert reason is None
        analysis = course.analyze_course(profile, RIDER, BIKE)
        gaps = self._gaps(analysis.profile)
        assert min(gaps) > 0
        assert max(gaps) == pytest.approx(min(gaps), rel=1e-9)


class TestSegmentation:
    def test_flat_climb_flat_becomes_three_segments(self):
        def grad(d):
            return 0.08 if 2000 <= d < 3500 else 0.0

        segments = course.segment_by_gradient(_profile(10.0, 5500.0, grad))
        assert [s.segment_type for s in segments] == ["flat", "climb", "flat"]
        assert segments[1].start_distance_m == pytest.approx(2000, abs=100)
        assert segments[1].end_distance_m == pytest.approx(3500, abs=100)
        assert segments[1].avg_gradient == pytest.approx(0.08, abs=0.005)

    def test_a_false_flat_does_not_split_a_climb(self):
        def grad(d):
            if d < 1000:
                return 0.06
            if d < 1300:
                return 0.05  # within the 2 pp tolerance of the running mean
            return 0.06

        segments = course.segment_by_gradient(_profile(10.0, 2300.0, grad))
        assert len(segments) == 1
        assert segments[0].segment_type == "climb"

    def test_a_short_blip_is_dissolved(self):
        def grad(d):
            return 0.10 if 1000 <= d < 1100 else 0.0

        segments = course.segment_by_gradient(_profile(10.0, 2200.0, grad))
        assert all(s.length_m >= course.MIN_SEGMENT_LENGTH_M for s in segments)
        assert all(s.segment_type == "flat" for s in segments)

    def test_segments_tile_the_course_exactly(self):
        def grad(d):
            if d < 1500:
                return 0.0
            if d < 3000:
                return 0.07
            if d < 4000:
                return -0.05
            return 0.01

        profile = _profile(10.0, 6000.0, grad)
        segments = course.segment_by_gradient(profile)
        assert segments[0].start_distance_m == profile.points[0].distance_m
        assert segments[-1].end_distance_m == pytest.approx(6000.0)
        for a, b in zip(segments, segments[1:]):
            assert a.end_distance_m == pytest.approx(b.start_distance_m)
        assert all(
            s.length_m >= course.MIN_SEGMENT_LENGTH_M for s in segments
        ) or len(segments) == 1

    def test_dissolving_a_very_long_course_stays_tractable(self):
        """The dissolve used to restart its scan from zero after every merge,
        and to delete from the middle of a list — quadratic in the raw segment
        count, and worst on exactly the rolling terrain the dissolve exists
        for. A 3000 km course spent ~10 s here; it now takes well under one.

        An absolute budget rather than a ratio: a ratio between two timed runs
        is noise-sensitive on a shared CI box, while the gap being defended
        here is more than an order of magnitude.
        """
        import time

        pts, d = [], 0.0
        while d <= 3_000_000:  # 3000 km at ~9 m, the shape that exposed this
            pts.append(
                course.ProfilePoint(
                    distance_m=d,
                    elevation_m=100.0 + 6.0 * math.sin(2 * math.pi * d / 300.0),
                    gradient=0.0,
                )
            )
            d += 9.0
        profile = course.CourseProfile(points=pts, total_distance_m=pts[-1].distance_m)

        started = time.perf_counter()
        segments = course.segment_by_gradient(profile)
        elapsed = time.perf_counter() - started

        assert segments
        # Quadratic measured ~10 s here and grows from there; linear is <1 s.
        assert elapsed < 6.0, f"segmentation took {elapsed:.1f}s — quadratic again?"

    def test_several_climbs_stay_separate(self):
        def grad(d):
            if 500 <= d < 1500 or 2500 <= d < 3500:
                return 0.08
            return 0.0

        segments = course.segment_by_gradient(_profile(10.0, 4500.0, grad))
        assert [s.segment_type for s in segments].count("climb") == 2


class TestPhysics:
    def test_flat_speed_matches_the_hand_computed_value(self):
        # 243.75 W at the pedal = 3.75·v + 0.216·v³ solves to ≈ 9.85 m/s.
        v = course.solve_speed_ms(250.0, 0.0, 85.0, 0.0045, 0.36)
        assert v == pytest.approx(9.85, rel=0.03)

    def test_a_steep_climb_is_gravity_dominated(self):
        v = course.solve_speed_ms(250.0, 0.08, 85.0, 0.0045, 0.36)
        theta = math.atan(0.08)
        linear = 85.0 * course.GRAVITY_MS2 * (0.0045 * math.cos(theta) + math.sin(theta))
        closed_form = 250.0 * course.DRIVETRAIN_EFFICIENCY / linear
        assert v == pytest.approx(closed_form, rel=0.05)
        assert v < closed_form  # drag only ever slows the climb

    def test_zero_power_on_a_descent_is_terminal_velocity(self):
        v = course.solve_speed_ms(0.0, -0.10, 85.0, 0.0045, 0.36)
        theta = math.atan(-0.10)
        linear = 85.0 * course.GRAVITY_MS2 * (0.0045 * math.cos(theta) + math.sin(theta))
        terminal = math.sqrt(-linear / (0.5 * course.AIR_DENSITY_KGM3 * 0.36))
        assert v == pytest.approx(terminal, rel=0.01)

    def test_zero_power_on_the_flat_is_stationary_not_a_crash(self):
        assert course.solve_speed_ms(0.0, 0.0, 85.0, 0.0045, 0.36) == 0.0

    def test_speed_is_monotone_in_power(self):
        speeds = [
            course.solve_speed_ms(p, 0.02, 85.0, 0.0045, 0.36)
            for p in (100.0, 200.0, 300.0, 400.0)
        ]
        assert speeds == sorted(speeds)
        assert speeds[0] < speeds[-1]

    def test_a_capped_descent_is_ridden_coasting(self):
        segment = course.Segment(
            index=0,
            start_distance_m=0.0,
            end_distance_m=1000.0,
            length_m=1000.0,
            avg_gradient=-0.10,
            elevation_change_m=-100.0,
            segment_type="descent",
        )
        [plan] = course.predict_splits([segment], RIDER, BIKE, 0.8)
        assert plan.speed_capped is True
        assert plan.speed_ms == course.DESCENT_SPEED_CAP_MS
        assert plan.power_w == 0.0


def _rolling_segments() -> list[course.Segment]:
    def grad(d):
        if d < 4000:
            return 0.0
        if d < 6000:
            return 0.07
        if d < 7500:
            return -0.06
        return 0.01

    return course.segment_by_gradient(_profile(10.0, 15000.0, grad))


class TestTargetTimeSolver:
    def test_round_trips_a_forward_prediction(self):
        segments = _rolling_segments()
        forward = course.predict_splits(segments, RIDER, BIKE, 0.80)
        target = sum(p.duration_s for p in forward)
        solution = course.solve_target_time(segments, RIDER, BIKE, target)
        assert solution.feasible is True
        assert solution.intensity == pytest.approx(0.80, abs=0.005)
        assert len(solution.splits) == len(segments)
        assert solution.predicted_time_s == pytest.approx(target, rel=0.01)

    def test_refuses_a_target_faster_than_physics(self):
        segments = _rolling_segments()
        floor = sum(
            p.duration_s
            for p in course.predict_splits(segments, RIDER, BIKE, course.INTENSITY_MAX)
        )
        solution = course.solve_target_time(segments, RIDER, BIKE, 0.5 * floor)
        assert solution.feasible is False
        assert solution.refusal_reason == "target_faster_than_physics"
        assert solution.splits == []

    def test_refuses_an_unsustainable_all_day_effort(self):
        # 160 flat km at ~0.95 × FTP is nearly five hours of threshold-adjacent
        # power: reachable on paper, held by nobody.
        def grad(d):
            return 0.0

        segments = course.segment_by_gradient(_profile(10.0, 160_000.0, grad))
        near_max = sum(
            p.duration_s for p in course.predict_splits(segments, RIDER, BIKE, 0.95)
        )
        solution = course.solve_target_time(segments, RIDER, BIKE, near_max)
        assert solution.feasible is False
        assert solution.refusal_reason == "exceeds_sustainable_power"
        assert solution.required_intensity is not None
        assert solution.required_intensity > course.max_sustainable_intensity(near_max)

    def test_a_comfortable_target_clamps_to_minimum_intensity(self):
        segments = _rolling_segments()
        easy = sum(
            p.duration_s
            for p in course.predict_splits(segments, RIDER, BIKE, course.INTENSITY_MIN)
        )
        solution = course.solve_target_time(segments, RIDER, BIKE, easy * 10)
        assert solution.feasible is True
        assert solution.intensity == course.INTENSITY_MIN
        assert solution.predicted_time_s < easy * 10

    def test_descent_heavy_course_still_bisects_cleanly(self):
        def grad(d):
            return -0.08 if d < 6000 else 0.02

        segments = course.segment_by_gradient(_profile(10.0, 10000.0, grad))
        forward = course.predict_splits(segments, RIDER, BIKE, 0.85)
        target = sum(p.duration_s for p in forward)
        solution = course.solve_target_time(segments, RIDER, BIKE, target)
        assert solution.feasible is True
        assert solution.predicted_time_s == pytest.approx(target, rel=0.02)

    @pytest.mark.parametrize(
        ("duration_s", "expected"),
        [(1800.0, 1.0), (3600.0, 1.0), (7200.0, 0.95), (10 * 24 * 3600.0, 0.70)],
    )
    def test_max_sustainable_intensity_spot_values(self, duration_s, expected):
        assert course.max_sustainable_intensity(duration_s) == pytest.approx(expected, abs=0.001)


def _avg_power_w(plans) -> float:
    """The time-weighted average power of a plan — what a head unit shows."""
    total = sum(p.duration_s for p in plans)
    return sum(p.power_w * p.duration_s for p in plans) / total


class TestTargetPowerSolver:
    """Issue #61 — the other half of the question: fix the watts, read the time."""

    def test_holds_the_requested_average_power(self):
        segments = _rolling_segments()
        solution = course.solve_target_power(segments, RIDER, BIKE, 200.0)
        assert solution.feasible is True
        assert _avg_power_w(solution.splits) == pytest.approx(200.0, rel=0.005)
        assert len(solution.splits) == len(segments)

    def test_agrees_with_the_time_solver(self):
        # Solve for a time, read the average power it asks for, then ask for
        # that power back: the two solvers must land on the same ride.
        segments = _rolling_segments()
        target = sum(p.duration_s for p in course.predict_splits(segments, RIDER, BIKE, 0.75))
        by_time = course.solve_target_time(segments, RIDER, BIKE, target)
        by_power = course.solve_target_power(
            segments, RIDER, BIKE, _avg_power_w(by_time.splits)
        )
        assert by_power.intensity == pytest.approx(by_time.intensity, abs=0.002)
        assert by_power.predicted_time_s == pytest.approx(by_time.predicted_time_s, rel=0.005)

    def test_the_average_is_not_a_flat_power_everywhere(self):
        # A target power is an average, not a prescription per segment: the
        # gradient weighting still spends on the climb and eases on the flat.
        segments = _rolling_segments()
        solution = course.solve_target_power(segments, RIDER, BIKE, 200.0)
        climbs = [p.power_w for p in solution.splits if p.segment.segment_type == "climb"]
        flats = [p.power_w for p in solution.splits if p.segment.segment_type == "flat"]
        assert climbs and flats
        assert min(climbs) > max(flats)

    def test_an_unsustainable_average_keeps_its_splits(self):
        # The deliberate difference from the time solver: an impossible time
        # describes no ride, but an unsustainable power describes one in full.
        def grad(d):
            return 0.0

        segments = course.segment_by_gradient(_profile(10.0, 160_000.0, grad))
        solution = course.solve_target_power(segments, RIDER, BIKE, 0.95 * RIDER.ftp_w)
        assert solution.feasible is False
        assert solution.refusal_reason == "exceeds_sustainable_power"
        assert len(solution.splits) == len(segments)
        assert solution.required_intensity > course.max_sustainable_intensity(
            solution.predicted_time_s
        )

    def test_a_soft_pedal_target_clamps_to_the_minimum_intensity(self):
        segments = _rolling_segments()
        solution = course.solve_target_power(segments, RIDER, BIKE, 5.0)
        assert solution.intensity == course.INTENSITY_MIN
        assert solution.feasible is True
        # The clamp is visible rather than silent: what the plan asks for is
        # reported, not the 5 W that were requested.
        assert solution.required_intensity > 0.02

    def test_a_superhuman_target_clamps_to_the_ceiling_and_is_refused(self):
        segments = _rolling_segments()
        solution = course.solve_target_power(segments, RIDER, BIKE, 3.0 * RIDER.ftp_w)
        assert solution.intensity == course.INTENSITY_MAX
        assert solution.feasible is False
        assert solution.refusal_reason == "exceeds_sustainable_power"
        assert solution.required_intensity < 3.0

    def test_a_descent_heavy_course_still_solves(self):
        def grad(d):
            return -0.08 if d < 6000 else 0.02

        segments = course.segment_by_gradient(_profile(10.0, 10000.0, grad))
        solution = course.solve_target_power(segments, RIDER, BIKE, 180.0)
        assert solution.predicted_time_s > 0
        assert _avg_power_w(solution.splits) == pytest.approx(180.0, rel=0.02)

    def test_nothing_to_solve_is_a_reason_code_not_a_crash(self):
        solution = course.solve_target_power([], RIDER, BIKE, 200.0)
        assert solution.feasible is False
        assert solution.splits == []
        assert solution.predicted_time_s is None


class TestAnalyzeCourse:
    def test_a_full_analysis_is_coherent(self):
        def ele(d):
            if d < 4000:
                return 100.0
            if d < 6000:
                return 100.0 + 0.07 * (d - 4000)
            if d < 7500:
                return 240.0 - 0.06 * (d - 6000)
            return 150.0 + 0.01 * (d - 7500)

        track = course.thin_track(_route(4.0, 3751, ele))
        profile, reason = course.course_profile(track)
        assert reason is None
        analysis = course.analyze_course(profile, RIDER, BIKE)
        assert analysis.total_distance_m == pytest.approx(15000.0, rel=0.01)
        assert analysis.elevation_gain_m == pytest.approx(140.0 + 75.0, rel=0.1)
        assert analysis.min_elevation_m == pytest.approx(100.0, abs=2.0)
        assert analysis.max_elevation_m == pytest.approx(240.0, abs=2.0)
        assert len(analysis.profile) <= course.CHART_PROFILE_MAX_POINTS
        assert analysis.profile[-1].distance_m == pytest.approx(15000.0, rel=0.01)
        assert analysis.pacing.feasible is True
        assert analysis.pacing.splits
        offsets = [p.start_offset_s for p in analysis.pacing.splits]
        assert offsets == sorted(offsets)

    def test_a_power_target_is_solved_for_power(self):
        profile, reason = course.course_profile(
            course.thin_track(_route(4.0, 3751, lambda d: 100.0 + 0.01 * d))
        )
        assert reason is None
        analysis = course.analyze_course(profile, RIDER, BIKE, target_power_w=210)
        assert analysis.pacing.splits
        assert _avg_power_w(analysis.pacing.splits) == pytest.approx(210.0, rel=0.01)

    def test_a_power_target_wins_when_both_arrive(self):
        # The API refuses a request carrying both; if one gets here anyway the
        # power target is the one the model can always honour.
        profile, reason = course.course_profile(
            course.thin_track(_route(4.0, 3751, lambda d: 100.0 + 0.01 * d))
        )
        assert reason is None
        analysis = course.analyze_course(
            profile, RIDER, BIKE, target_time_s=60, target_power_w=210
        )
        assert analysis.pacing.feasible is True
        assert _avg_power_w(analysis.pacing.splits) == pytest.approx(210.0, rel=0.01)

    def test_the_default_intensity_backs_off_for_a_long_day(self):
        def grad(d):
            return 0.0

        segments = course.segment_by_gradient(_profile(10.0, 200_000.0, grad))
        pacing = course.default_pacing(segments, RIDER, BIKE)
        assert pacing.feasible is True
        assert pacing.intensity < course.DEFAULT_INTENSITY

    def test_key_climbs_group_adjacent_climb_segments(self):
        segments = [
            course.Segment(0, 0.0, 1000.0, 1000.0, 0.0, 0.0, "flat"),
            course.Segment(1, 1000.0, 1500.0, 500.0, 0.06, 30.0, "climb"),
            course.Segment(2, 1500.0, 2000.0, 500.0, 0.09, 45.0, "climb"),
            course.Segment(3, 2000.0, 3000.0, 1000.0, 0.0, 0.0, "flat"),
            course.Segment(4, 3000.0, 3200.0, 200.0, 0.03, 6.0, "climb"),
        ]
        plans = course.predict_splits(segments, RIDER, BIKE, 0.8)
        climbs = course.key_climbs(plans)
        assert len(climbs) == 1  # the 6 m rise is below the feature threshold
        assert climbs[0].start_distance_m == 1000.0
        assert climbs[0].elevation_gain_m == pytest.approx(75.0)
        assert climbs[0].avg_gradient == pytest.approx(0.075)
