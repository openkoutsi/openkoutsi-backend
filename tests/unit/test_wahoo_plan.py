"""Tests for the Wahoo plan.json workout serializer."""

from openkoutsi.workout_formats.wahoo_plan import build_wahoo_plan

POWER_ZONES = [
    {"low": 0, "high": 150},    # Z1
    {"low": 151, "high": 210},  # Z2
    {"low": 211, "high": 250},  # Z3
]

_WARMUP = {
    "kind": "step",
    "step_type": "warmup",
    "duration": {"type": "time", "seconds": 600},
    "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 50.0}},
}

_ACTIVE = {
    "kind": "step",
    "step_type": "active",
    "duration": {"type": "time", "seconds": 1200},
    "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 90.0}},
}

_REPEAT = {
    "kind": "repeat",
    "repeat_count": 3,
    "steps": [
        {"kind": "step", "step_type": "active", "duration": {"type": "time", "seconds": 300},
         "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 105.0}}},
        {"kind": "step", "step_type": "recovery", "duration": {"type": "time", "seconds": 120}},
    ],
}


def _build(steps, sport_type="Ride", ftp=250, zones=POWER_ZONES):
    return build_wahoo_plan(
        steps=steps,
        workout_name="Test Workout",
        workout_description="A test",
        sport_type=sport_type,
        athlete_ftp=ftp,
        athlete_power_zones=zones,
    )


class TestHeader:
    def test_basic_header(self):
        plan = _build([_ACTIVE])
        header = plan["header"]
        assert header["name"] == "Test Workout"
        assert header["version"] == "1.0.0"
        assert header["description"] == "A test"
        assert header["ftp"] == 250

    def test_biking_outdoor_defaults(self):
        header = _build([_ACTIVE], sport_type="Ride")["header"]
        assert header["workout_type_family"] == 0  # Biking
        assert header["workout_type_location"] == 1  # Outdoor

    def test_indoor_ride(self):
        header = _build([_ACTIVE], sport_type="VirtualRide")["header"]
        assert header["workout_type_family"] == 0
        assert header["workout_type_location"] == 0  # Indoor

    def test_running_family(self):
        header = _build([_ACTIVE], sport_type="Run")["header"]
        assert header["workout_type_family"] == 1

    def test_ftp_omitted_when_absent(self):
        header = _build([_ACTIVE], ftp=None)["header"]
        assert "ftp" not in header


class TestIntervals:
    def test_time_duration(self):
        interval = _build([_ACTIVE])["intervals"][0]
        assert interval["exit_trigger_type"] == "time"
        assert interval["exit_trigger_value"] == 1200

    def test_distance_duration(self):
        step = {
            "kind": "step",
            "step_type": "active",
            "duration": {"type": "distance", "meters": 400},
        }
        interval = _build([step])["intervals"][0]
        assert interval["exit_trigger_type"] == "distance"
        assert interval["exit_trigger_value"] == 400

    def test_open_duration_falls_back_to_time(self):
        step = {"kind": "step", "step_type": "active", "duration": {"type": "open"}}
        interval = _build([step])["intervals"][0]
        assert interval["exit_trigger_type"] == "time"
        assert interval["exit_trigger_value"] > 0

    def test_intensity_mapping(self):
        plan = _build([_WARMUP, _ACTIVE])
        assert plan["intervals"][0]["intensity_type"] == "wu"
        assert plan["intervals"][1]["intensity_type"] == "active"


class TestRepeats:
    def test_native_repeat_structure(self):
        interval = _build([_REPEAT])["intervals"][0]
        assert interval["exit_trigger_type"] == "repeat"
        # repeat_count 3 → 2 iterations after the first
        assert interval["exit_trigger_value"] == 2
        assert "targets" not in interval
        assert len(interval["intervals"]) == 2

    def test_repeat_children_are_intervals(self):
        interval = _build([_REPEAT])["intervals"][0]
        child = interval["intervals"][0]
        assert child["exit_trigger_type"] == "time"
        assert child["exit_trigger_value"] == 300
        assert child["targets"][0]["type"] == "ftp"


class TestTargets:
    def test_pct_ftp(self):
        target = _build([_ACTIVE])["intervals"][0]["targets"][0]
        assert target == {"type": "ftp", "low": 0.90, "high": 0.90}

    def test_absolute_watts(self):
        step = {
            "kind": "step",
            "step_type": "active",
            "duration": {"type": "time", "seconds": 300},
            "target": {"metric": "power", "spec": {"type": "absolute", "value": 200}},
        }
        target = _build([step])["intervals"][0]["targets"][0]
        assert target == {"type": "watts", "low": 200.0, "high": 200.0}

    def test_power_range(self):
        step = {
            "kind": "step",
            "step_type": "active",
            "duration": {"type": "time", "seconds": 300},
            "target": {"metric": "power", "spec": {"type": "range", "low": 180, "high": 220}},
        }
        target = _build([step])["intervals"][0]["targets"][0]
        assert target == {"type": "watts", "low": 180.0, "high": 220.0}
        assert target["high"] >= target["low"]

    def test_power_zone_resolves_to_watts(self):
        step = {
            "kind": "step",
            "step_type": "active",
            "duration": {"type": "time", "seconds": 300},
            "target": {"metric": "power", "spec": {"type": "zone", "zone_number": 2}},
        }
        target = _build([step])["intervals"][0]["targets"][0]
        assert target == {"type": "watts", "low": 151.0, "high": 210.0}

    def test_hr_absolute(self):
        step = {
            "kind": "step",
            "step_type": "active",
            "duration": {"type": "time", "seconds": 300},
            "target": {"metric": "hr", "spec": {"type": "absolute", "value": 150}},
        }
        target = _build([step])["intervals"][0]["targets"][0]
        assert target == {"type": "hr", "low": 150.0, "high": 150.0}

    def test_no_target_omits_targets(self):
        step = {"kind": "step", "step_type": "rest", "duration": {"type": "time", "seconds": 60}}
        interval = _build([step])["intervals"][0]
        assert "targets" not in interval
