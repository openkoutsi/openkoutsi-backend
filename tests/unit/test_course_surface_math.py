"""Surface classification math (issue #56): classes, confidence, Crr, dissolving.

The load-bearing properties, in the order they matter:

* **Confidence never overstates.** ``paved_smooth`` is what the routing engine
  returns for a way carrying no surface information at all, so it reads
  *inferred*; every other value requires an explicit tag and reads *confirmed*.
* **An unrecognised value classifies as unknown**, never as a default that
  looks confident.
* **A short sector is not automatically noise.** 130 m of mud inside 40 km of
  asphalt survives, keeps its own segment and its own rolling resistance; a
  40 m snap artefact does not. That distinction is severity, not length.
* **Enabling the matcher never silently restates an existing plan** — on
  asphalt, and wherever the surface is unknown, the Crr is exactly what Stage 1
  used.
"""
from __future__ import annotations

import time

import pytest

from openkoutsi import course, surface

# Points are laid out at a fixed spacing so a run's length in metres is
# readable straight off the spec: ("dirt", 130) is 130 m of dirt.
_SPACING_M = 10.0


def _series(spec) -> tuple[list[surface.SurfacePoint], list[float]]:
    """``[(matcher_value, metres), …]`` → per-point surface and distance."""
    points: list[surface.SurfacePoint] = []
    distances: list[float] = []
    distance = 0.0
    for raw, metres in spec:
        for _ in range(int(metres / _SPACING_M)):
            points.append(
                surface.SurfacePoint(
                    surface=surface.normalise(raw),
                    confidence=surface.confidence_for(raw),
                    raw=raw,
                )
            )
            distances.append(distance)
            distance += _SPACING_M
    return points, distances


def _runs(spec) -> list[surface.SurfaceRun]:
    points, distances = _series(spec)
    return surface.build_runs(points, distances)


def _classes(spec) -> list[str]:
    """The distinct classes surviving the dissolve, in order."""
    return [run.surface for run in _runs(spec)]


class TestNormalisation:
    def test_every_engine_value_maps_to_a_class(self):
        assert surface.normalise("paved_smooth") == surface.ASPHALT
        assert surface.normalise("paved") == surface.PAVED
        assert surface.normalise("paved_rough") == surface.COBBLES
        assert surface.normalise("compacted") == surface.COMPACTED
        assert surface.normalise("gravel") == surface.GRAVEL
        assert surface.normalise("dirt") == surface.DIRT
        assert surface.normalise("path") == surface.GRASS

    def test_impassable_is_unknown_not_a_class_of_its_own(self):
        """On a course somebody intends to ride it is far likelier a bad snap.

        Inventing a scary label out of a probable match error is its own kind
        of dishonesty.
        """
        assert surface.normalise("impassable") == surface.UNKNOWN

    @pytest.mark.parametrize("value", [None, "", "   ", "asphalt", "cobblestone", "wat"])
    def test_anything_unrecognised_is_unknown(self, value):
        """Including a plausible-looking OSM value the engine never emits.

        A future engine version adding an enum member must show up as "we could
        not tell", not as a default that reads like an answer.
        """
        assert surface.normalise(value) == surface.UNKNOWN

    def test_values_are_matched_case_and_space_insensitively(self):
        assert surface.normalise("  GRAVEL  ") == surface.GRAVEL

    def test_every_class_is_declared(self):
        for value in surface._FROM_MATCHER.values():
            assert value in surface.SURFACE_CLASSES


class TestConfidence:
    def test_paved_smooth_is_inferred_because_it_is_the_untagged_default(self):
        """The whole confidence layer rests on this one fact.

        ``OSMWay`` zeroes its three-bit surface field on construction and
        ``kPavedSmooth`` is enumerator 0, so an untagged way arrives as
        ``paved_smooth``. It is therefore indistinguishable from a way somebody
        explicitly tagged as asphalt, and we report the weaker claim.
        """
        assert surface.confidence_for("paved_smooth") == surface.INFERRED

    @pytest.mark.parametrize(
        "value", ["paved", "paved_rough", "compacted", "gravel", "dirt", "path"]
    )
    def test_every_other_engine_value_is_confirmed(self, value):
        """Only an explicit surface/tracktype/smoothness tag produces these."""
        assert surface.confidence_for(value) == surface.CONFIRMED

    @pytest.mark.parametrize("value", [None, "impassable", "wat"])
    def test_unusable_values_are_inferred(self, value):
        assert surface.confidence_for(value) == surface.INFERRED

    @pytest.mark.parametrize(
        "value", [None, "", "impassable", "wat", "asphalt", "paved_smooth", "gravel"]
    )
    def test_unknown_never_reports_as_confirmed(self, value):
        """The invariant that keeps the two halves coherent.

        ``impassable`` is a value the engine really does emit and which we
        classify as unknown; reading it as confirmed would put a fact label on
        a surface we just admitted we could not identify.
        """
        if surface.normalise(value) == surface.UNKNOWN:
            assert surface.confidence_for(value) == surface.INFERRED


class TestSeverity:
    def test_the_scale_runs_smooth_to_rough(self):
        ranks = [
            surface.severity_rank(c)
            for c in (
                surface.ASPHALT,
                surface.PAVED,
                surface.COMPACTED,
                surface.COBBLES,
                surface.GRAVEL,
                surface.DIRT,
                surface.GRASS,
            )
        ]
        assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks)

    def test_a_change_involving_unknown_is_not_a_change(self):
        """We have no basis for calling a surface we could not identify severe.

        Without this an unmatched stretch could survive the dissolve by
        pretending to be dramatic.
        """
        assert surface.severity_delta(surface.UNKNOWN, surface.DIRT) == 0
        assert surface.severity_delta(surface.ASPHALT, surface.UNKNOWN) == 0

    def test_delta_is_symmetric_rank_distance(self):
        assert surface.severity_delta(surface.ASPHALT, surface.DIRT) == 5
        assert surface.severity_delta(surface.DIRT, surface.ASPHALT) == 5


class TestRollingResistance:
    @pytest.mark.parametrize("width", [None, 25, 28, 40, 50])
    def test_asphalt_reproduces_stage_1_exactly(self, width):
        """Turning the matcher on must not silently restate an existing plan.

        A course matched as paved throughout has to solve to the numbers it
        solved to before the sidecar existed, or every stored plan quietly
        becomes wrong on the day the toggle is flipped.
        """
        assert surface.crr_for(surface.ASPHALT, width) == surface.crr_for_tyre_width(width)

    @pytest.mark.parametrize("value", [None, surface.UNKNOWN, "not-a-class"])
    def test_absent_or_unknown_surface_falls_back_to_the_paved_curve(self, value):
        assert surface.crr_for(value, 28) == surface.crr_for_tyre_width(28)

    def test_rougher_surfaces_cost_more(self):
        ordered = [
            surface.ASPHALT,
            surface.PAVED,
            surface.COMPACTED,
            surface.COBBLES,
            surface.GRAVEL,
            surface.DIRT,
            surface.GRASS,
        ]
        values = [surface.crr_for(c, 40) for c in ordered]
        assert values == sorted(values)

    def test_the_width_relationship_inverts_off_pavement(self):
        """On tarmac a wide tyre costs casing losses; on gravel it floats."""
        assert surface.crr_for(surface.ASPHALT, 50) > surface.crr_for(surface.ASPHALT, 28)
        assert surface.crr_for(surface.GRAVEL, 50) < surface.crr_for(surface.GRAVEL, 28)

    def test_hand_computed_gravel_values(self):
        # 0.0180 base at the ~40 mm reference, × 1.25 for a 32 mm tyre.
        assert surface.crr_for(surface.GRAVEL, 32) == pytest.approx(0.0225)
        # × 0.88 for something wider than 45 mm.
        assert surface.crr_for(surface.GRAVEL, 50) == pytest.approx(0.01584)


class TestDissolving:
    """Severity, not length, decides whether a short run survives."""

    def test_a_130m_mud_sector_inside_40km_of_asphalt_survives(self):
        """The case the whole severity rule exists for.

        A rider cannot expect to roll through 40 km of smooth asphalt if 130 m
        of it is mud and rocks, and a floor that judged on length alone would
        erase exactly that.
        """
        runs = _runs([("paved_smooth", 20_000), ("dirt", 130), ("paved_smooth", 20_000)])
        assert [r.surface for r in runs] == [
            surface.ASPHALT,
            surface.DIRT,
            surface.ASPHALT,
        ]
        assert runs[1].length_m == pytest.approx(130.0)
        assert runs[1].confidence == surface.CONFIRMED

    def test_a_40m_paving_stone_blip_inside_asphalt_dissolves(self):
        """One rank of difference over 40 m is the matcher snapping, not a road."""
        assert _classes([("paved_smooth", 5000), ("paved", 40), ("paved_smooth", 5000)]) == [
            surface.ASPHALT
        ]

    def test_nothing_survives_below_the_hard_floor_however_severe(self):
        """Two match samples is not evidence of anything, at any severity."""
        assert _classes([("paved_smooth", 5000), ("dirt", 30), ("paved_smooth", 5000)]) == [
            surface.ASPHALT
        ]

    def test_50m_of_grass_in_tarmac_survives_on_severity_alone(self):
        assert _classes([("paved_smooth", 3000), ("path", 50), ("paved_smooth", 3000)]) == [
            surface.ASPHALT,
            surface.GRASS,
            surface.ASPHALT,
        ]

    def test_a_run_between_its_neighbours_is_a_transition_not_an_outlier(self):
        """Paving stones between asphalt and gravel sit *between* them.

        Measuring severity against the further neighbour would score this 3 and
        spare a 40 m snap artefact; against the nearer one it scores 1 and goes.
        """
        assert _classes([("paved_smooth", 300), ("paved", 40), ("gravel", 2000)]) == [
            surface.ASPHALT,
            surface.GRAVEL,
        ]

    def test_a_short_run_dissolves_into_the_longer_neighbour(self):
        runs = _runs([("paved_smooth", 300), ("paved", 40), ("gravel", 2000)])
        # The gravel neighbour is the longer of the two, so it absorbs the blip.
        assert runs[1].surface == surface.GRAVEL
        assert runs[1].start_distance_m == pytest.approx(300.0)

    def test_a_storm_of_short_runs_collapses(self):
        """Severity only buys a run its life when the run is *isolated*.

        Twenty alternating 50 m runs is the matcher flicking between a road and
        the cycleway beside it, and each blip looks dramatic on its own. Nothing
        in a storm is trustworthy enough to pace or warn on.
        """
        runs = _runs([("gravel", 50), ("paved_smooth", 50)] * 20)
        assert len(runs) <= 2

    def test_a_class_that_survives_only_by_absorbing_noise_reads_inferred(self):
        """The dissolve must not launder a guess into a fact.

        Confidence is measured against what the matcher *originally* said, so a
        run assembled out of points matched as something else cannot inherit
        the confidence of the class it ended up with.
        """
        runs = _runs([("gravel", 50), ("paved_smooth", 50)] * 20)
        assert runs[0].confidence == surface.INFERRED

    def test_a_road_degrading_through_grades_keeps_both_stretches(self):
        assert _classes(
            [("paved_smooth", 5000), ("gravel", 100), ("dirt", 130), ("paved_smooth", 5000)]
        ) == [surface.ASPHALT, surface.GRAVEL, surface.DIRT, surface.ASPHALT]

    def test_an_unmatched_stretch_stays_unknown(self):
        assert _classes([("paved_smooth", 2000), (None, 500), ("paved_smooth", 2000)]) == [
            surface.ASPHALT,
            surface.UNKNOWN,
            surface.ASPHALT,
        ]

    def test_runs_tile_the_course_without_gaps(self):
        runs = _runs([("paved_smooth", 3000), ("gravel", 2000), ("dirt", 1000)])
        for before, after in zip(runs, runs[1:]):
            assert before.end_distance_m == pytest.approx(after.start_distance_m)

    @pytest.mark.parametrize("spec", [[], [("gravel", 10)], [("gravel", 20)]])
    def test_degenerate_series_do_not_raise(self, spec):
        _runs(spec)

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError):
            surface.dissolve_runs([surface.SurfacePoint("gravel", "confirmed", "gravel")], [])

    def test_dissolving_a_very_long_noisy_course_stays_tractable(self):
        """3000 km of 60 m alternations — the pathological input, in one pass.

        Restarting the scan after every merge makes this quadratic in the run
        count, worst on exactly the terrain the dissolve exists for.
        """
        points, distances = _series([("paved_smooth", 60), ("gravel", 60)] * 25_000)
        started = time.perf_counter()
        runs = surface.build_runs(points, distances)
        assert time.perf_counter() - started < 6.0
        assert len(runs) <= 2


class TestRoughSectors:
    def test_a_short_sector_is_named_even_though_it_is_short(self):
        """The point of the sector list: a colour on a chart is missable.

        "130 m of mud from km 20" read out in the plan is not, and that is the
        sentence a rider expecting tarmac needs before the day rather than
        during it.
        """
        sectors = surface.rough_sectors(
            _runs([("paved_smooth", 20_000), ("dirt", 130), ("paved_smooth", 20_000)])
        )
        assert len(sectors) == 1
        assert sectors[0].surface == surface.DIRT
        assert sectors[0].length_m == pytest.approx(130.0)
        assert sectors[0].severity_step == 5

    def test_smooth_courses_have_none(self):
        assert surface.rough_sectors(_runs([("paved_smooth", 10_000)])) == []

    def test_paving_stones_are_not_rough_enough_to_warn_about(self):
        assert (
            surface.rough_sectors(
                _runs([("paved_smooth", 3000), ("paved", 2000), ("paved_smooth", 3000)])
            )
            == []
        )

    def test_cobbles_are(self):
        sectors = surface.rough_sectors(
            _runs([("paved_smooth", 4000), ("paved_rough", 2200), ("paved_smooth", 4000)])
        )
        assert [s.surface for s in sectors] == [surface.COBBLES]


class TestStoredSeries:
    def test_the_per_point_series_round_trips(self):
        points, _ = _series([("gravel", 100), ("paved_smooth", 100)])
        stored = [[p.raw, p.confidence] for p in points]
        assert surface.points_from_json(stored) == points

    def test_the_class_is_re_derived_rather_than_stored(self):
        """Storage keeps only what the matcher said, plus confidence.

        Every dissolving decision is re-made on read, so tuning a threshold
        later re-reads correctly from what is already on disk instead of
        needing every stored course re-matched.
        """
        [point] = surface.points_from_json([["gravel", surface.CONFIRMED]])
        assert point.surface == surface.GRAVEL and point.raw == "gravel"

    @pytest.mark.parametrize(
        "raw", ["future_enum", "impassable", 1.5, None]
    )
    def test_a_stored_confirmed_cannot_survive_a_class_we_no_longer_recognise(self, raw):
        """"Unknown implies inferred" has to hold on the read path too.

        The class is re-derived on read precisely so `_FROM_MATCHER` can be
        tuned later without re-matching every stored course. That makes stored
        confidence a value from a *previous* release's vocabulary — so taking
        it at face value would let a row written before a mapping changed
        re-read as unknown-and-confirmed, which is "we confirmed that we do
        not know".
        """
        [point] = surface.points_from_json([[raw, surface.CONFIRMED]])
        assert point.surface == surface.UNKNOWN
        assert point.confidence == surface.INFERRED

    def test_a_stored_confidence_is_still_honoured_for_a_class_we_do_know(self):
        """The tuning the storage format exists for still works."""
        [point] = surface.points_from_json([["gravel", surface.INFERRED]])
        assert point.surface == surface.GRAVEL
        assert point.confidence == surface.INFERRED

    def test_the_invariant_holds_however_the_point_was_built(self):
        """Read path and write path cannot disagree about the same input."""
        for raw in ("paved_smooth", "gravel", "impassable", "wat", None, 1.5):
            written = surface.SurfacePoint(
                surface=surface.normalise(raw),
                confidence=surface.confidence_for(raw),
                raw=raw,
            )
            [read] = surface.points_from_json([[raw, written.confidence]])
            assert read.surface == written.surface
            assert read.confidence == written.confidence

    @pytest.mark.parametrize("stored", [None, []])
    def test_nothing_stored_reads_as_nothing(self, stored):
        assert surface.points_from_json(stored) is None

    def test_the_ribbon_is_compact_and_json_safe(self):
        ribbon = surface.ribbon_json(_runs([("paved_smooth", 2000), ("gravel", 2000)]))
        assert ribbon[0][2] == surface.ASPHALT and ribbon[1][2] == surface.GRAVEL
        for entry in ribbon:
            assert [type(v) for v in entry] == [float, float, str, str, int]


class TestSurfaceAwareSegmentation:
    """What the segment table does with a matched surface."""

    @staticmethod
    def _profile(spec):
        points, _ = _series(spec)
        profile_points = [
            course.ProfilePoint(
                distance_m=i * _SPACING_M, elevation_m=100.0, gradient=0.0
            )
            for i in range(len(points))
        ]
        return (
            course.CourseProfile(
                points=profile_points,
                total_distance_m=profile_points[-1].distance_m,
            ),
            points,
        )

    RIDER = course.RiderParams(ftp_w=250.0, weight_kg=75.0)
    BIKE = course.BikeParams(tyre_width_mm=32, riding_position="hoods")

    def test_a_surface_change_splits_a_constant_gradient_run(self):
        """Nothing about the gradient changes here, so the split is the surface."""
        profile, points = self._profile(
            [("paved_smooth", 5000), ("dirt", 130), ("paved_smooth", 5000)]
        )
        analysis = course.analyze_course(
            profile, self.RIDER, self.BIKE, surface_points=points
        )
        assert [s.surface for s in analysis.segments] == [
            surface.ASPHALT,
            surface.DIRT,
            surface.ASPHALT,
        ]

    def test_the_short_sector_keeps_its_own_row_below_the_segment_floor(self):
        """MIN_SEGMENT_LENGTH_M is 200 m and the sector is 130 m.

        The gradient dissolve must not swallow a boundary the surface pass just
        went to the trouble of preserving, or the sector's rolling resistance —
        and therefore its split — silently reverts to asphalt.
        """
        profile, points = self._profile(
            [("paved_smooth", 5000), ("dirt", 130), ("paved_smooth", 5000)]
        )
        analysis = course.analyze_course(
            profile, self.RIDER, self.BIKE, surface_points=points
        )
        sector = analysis.segments[1]
        assert sector.length_m == pytest.approx(130.0)
        assert sector.length_m < course.MIN_SEGMENT_LENGTH_M

    def test_the_sector_is_solved_with_its_own_rolling_resistance(self):
        profile, points = self._profile(
            [("paved_smooth", 5000), ("dirt", 130), ("paved_smooth", 5000)]
        )
        analysis = course.analyze_course(
            profile, self.RIDER, self.BIKE, surface_points=points
        )
        paved, mud, _ = analysis.pacing.splits
        assert mud.crr_used == pytest.approx(surface.crr_for(surface.DIRT, 32))
        assert paved.crr_used == pytest.approx(surface.crr_for_tyre_width(32))
        assert mud.speed_ms < paved.speed_ms

    def test_confidence_reaches_the_segments(self):
        profile, points = self._profile(
            [("paved_smooth", 5000), ("dirt", 130), ("paved_smooth", 5000)]
        )
        analysis = course.analyze_course(
            profile, self.RIDER, self.BIKE, surface_points=points
        )
        assert [s.surface_confidence for s in analysis.segments] == [
            surface.INFERRED,
            surface.CONFIRMED,
            surface.INFERRED,
        ]

    def test_the_raw_matcher_value_is_preserved_not_discarded(self):
        profile, points = self._profile([("paved_smooth", 2000), ("gravel", 2000)])
        analysis = course.analyze_course(
            profile, self.RIDER, self.BIKE, surface_points=points
        )
        assert [s.surface_raw for s in analysis.segments] == ["paved_smooth", "gravel"]

    def test_without_surface_data_nothing_changes(self):
        """A course on an instance with no matcher is a Stage 1 course.

        Same single segment, same Crr, same predicted time — the feature is
        absent rather than degraded.
        """
        profile, _ = self._profile([("paved_smooth", 10_000)])
        analysis = course.analyze_course(profile, self.RIDER, self.BIKE)
        assert len(analysis.segments) == 1
        assert analysis.segments[0].surface is None
        assert analysis.segments[0].surface_confidence is None
        assert analysis.surface_runs == [] and analysis.rough_sectors == []
        assert analysis.pacing.splits[0].crr_used == pytest.approx(
            surface.crr_for_tyre_width(32)
        )

    def test_an_all_asphalt_match_solves_to_the_stage_1_time(self):
        """Matching a fully paved course must not move a single number."""
        profile, points = self._profile([("paved_smooth", 10_000)])
        without = course.analyze_course(profile, self.RIDER, self.BIKE)
        with_surface = course.analyze_course(
            profile, self.RIDER, self.BIKE, surface_points=points
        )
        assert with_surface.pacing.predicted_time_s == pytest.approx(
            without.pacing.predicted_time_s
        )

    def test_segments_still_tile_the_course(self):
        profile, points = self._profile(
            [("paved_smooth", 3000), ("gravel", 2000), ("dirt", 1000)]
        )
        analysis = course.analyze_course(
            profile, self.RIDER, self.BIKE, surface_points=points
        )
        for before, after in zip(analysis.segments, analysis.segments[1:]):
            assert before.end_distance_m == pytest.approx(after.start_distance_m)

    def test_the_ribbon_and_the_sectors_ride_along(self):
        profile, points = self._profile(
            [("paved_smooth", 20_000), ("dirt", 130), ("paved_smooth", 20_000)]
        )
        analysis = course.analyze_course(
            profile, self.RIDER, self.BIKE, surface_points=points
        )
        assert len(analysis.surface_runs) == 3
        assert [s.surface for s in analysis.rough_sectors] == [surface.DIRT]
