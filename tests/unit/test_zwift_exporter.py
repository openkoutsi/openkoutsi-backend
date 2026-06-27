"""Tests for the Zwift .zwo workout exporter."""
import xml.etree.ElementTree as ET
import pytest

from openkoutsi.workout_formats.zwift import (
    _zone_midpoint_pct,
    _spec_to_pct,
    _step_to_element,
    _repeat_to_elements,
    _steps_to_elements,
    ZwiftExporter,
)


POWER_ZONES = [
    {"low": 0, "high": 150},    # Z1
    {"low": 151, "high": 210},  # Z2
    {"low": 211, "high": 250},  # Z3
]


def _make_step(step_type="active", duration_type="time", seconds=300, spec=None):
    step = {
        "kind": "step",
        "step_type": step_type,
        "duration": {"type": duration_type, "seconds": seconds},
    }
    if spec:
        step["target"] = {"metric": "power", "spec": spec}
    return step


class TestZoneMidpointPct:
    def test_uses_provided_zones(self):
        # Z1: 0-150W at FTP 250 → mid=75W → 75/250=0.30
        result = _zone_midpoint_pct(1, POWER_ZONES, 250)
        assert result == pytest.approx(75 / 250, rel=1e-6)

    def test_uses_fallback_when_no_zones(self):
        assert _zone_midpoint_pct(1, None, 250) == pytest.approx(0.55, rel=1e-6)
        assert _zone_midpoint_pct(4, None, 250) == pytest.approx(0.92, rel=1e-6)

    def test_uses_fallback_for_out_of_range_zone(self):
        # Zone 99 doesn't exist → fallback default 0.75
        assert _zone_midpoint_pct(99, POWER_ZONES, 250) == pytest.approx(0.75, rel=1e-6)


class TestSpecToPct:
    def test_pct_ftp(self):
        lo, hi = _spec_to_pct({"type": "pct_ftp", "pct": 90}, 250, None)
        assert lo == pytest.approx(0.9, rel=1e-6)
        assert hi == pytest.approx(0.9, rel=1e-6)

    def test_absolute(self):
        lo, hi = _spec_to_pct({"type": "absolute", "value": 250}, 250, None)
        assert lo == pytest.approx(1.0, rel=1e-6)
        assert hi == pytest.approx(1.0, rel=1e-6)

    def test_range(self):
        lo, hi = _spec_to_pct({"type": "range", "low": 200, "high": 300}, 250, None)
        assert lo == pytest.approx(0.8, rel=1e-6)
        assert hi == pytest.approx(1.2, rel=1e-6)

    def test_zone(self):
        lo, hi = _spec_to_pct({"type": "zone", "zone_number": 1}, 250, POWER_ZONES)
        expected = 75 / 250
        assert lo == pytest.approx(expected, rel=1e-6)
        assert hi == pytest.approx(expected, rel=1e-6)

    def test_unknown_type_returns_half(self):
        lo, hi = _spec_to_pct({"type": "unknown"}, 250, None)
        assert lo == pytest.approx(0.5, rel=1e-6)
        assert hi == pytest.approx(0.5, rel=1e-6)


class TestStepToElement:
    def test_warmup_with_power(self):
        step = _make_step("warmup", spec={"type": "pct_ftp", "pct": 50})
        el = _step_to_element(step, 250, None)
        assert el.tag == "Warmup"
        assert el.get("Duration") == "300"
        assert el.get("PowerLow") == "0.500"

    def test_warmup_without_power_uses_defaults(self):
        step = _make_step("warmup")
        el = _step_to_element(step, 250, None)
        assert el.tag == "Warmup"
        assert el.get("PowerLow") == "0.500"
        assert el.get("PowerHigh") == "0.750"

    def test_cooldown_with_power(self):
        step = _make_step("cooldown", spec={"type": "pct_ftp", "pct": 60})
        el = _step_to_element(step, 250, None)
        assert el.tag == "Cooldown"
        assert el.get("PowerLow") == "0.600"

    def test_cooldown_without_power_uses_defaults(self):
        step = _make_step("cooldown")
        el = _step_to_element(step, 250, None)
        assert el.get("PowerLow") == "0.750"
        assert el.get("PowerHigh") == "0.400"

    def test_rest_without_power_is_free_ride(self):
        step = _make_step("rest")
        el = _step_to_element(step, 250, None)
        assert el.tag == "FreeRide"

    def test_recovery_without_power_is_free_ride(self):
        step = _make_step("recovery")
        el = _step_to_element(step, 250, None)
        assert el.tag == "FreeRide"

    def test_steady_state_when_power_constant(self):
        step = _make_step(spec={"type": "pct_ftp", "pct": 85})
        el = _step_to_element(step, 250, None)
        assert el.tag == "SteadyState"
        assert el.get("Power") == "0.850"

    def test_ramp_when_power_range(self):
        step = _make_step(spec={"type": "range", "low": 150, "high": 250})
        el = _step_to_element(step, 250, None)
        assert el.tag == "Ramp"
        assert el.get("PowerLow") is not None
        assert el.get("PowerHigh") is not None

    def test_free_ride_when_no_power(self):
        step = _make_step()  # no power target
        el = _step_to_element(step, 250, None)
        assert el.tag == "FreeRide"

    def test_non_time_duration_defaults_to_60s(self):
        step = {"kind": "step", "step_type": "active", "duration": {"type": "distance", "meters": 500},
                "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 100}}}
        el = _step_to_element(step, 250, None)
        assert el.get("Duration") == "60"


class TestRepeatToElements:
    def test_two_step_work_rest_becomes_intervalst(self):
        work = _make_step("active", seconds=120, spec={"type": "pct_ftp", "pct": 120})
        rest = _make_step("recovery", seconds=60)
        block = {"kind": "repeat", "repeat_count": 5, "steps": [work, rest]}
        elements = _repeat_to_elements(block, 250, None)
        assert len(elements) == 1
        el = elements[0]
        assert el.tag == "IntervalsT"
        assert el.get("Repeat") == "5"
        assert el.get("OnDuration") == "120"
        assert el.get("OffDuration") == "60"

    def test_two_step_with_rest_power_sets_off_power(self):
        work = _make_step("active", seconds=60, spec={"type": "pct_ftp", "pct": 120})
        rest = _make_step("recovery", seconds=30, spec={"type": "pct_ftp", "pct": 50})
        block = {"kind": "repeat", "repeat_count": 3, "steps": [work, rest]}
        elements = _repeat_to_elements(block, 250, None)
        assert elements[0].get("OffPower") == "0.500"

    def test_three_step_repeat_expands_inline(self):
        s1 = _make_step("active", seconds=60)
        s2 = _make_step("active", seconds=30)
        s3 = _make_step("recovery", seconds=30)
        block = {"kind": "repeat", "repeat_count": 2, "steps": [s1, s2, s3]}
        elements = _repeat_to_elements(block, 250, None)
        # 2 repeats × 3 steps = 6 elements (all FreeRide since no power)
        assert len(elements) == 6


class TestZwiftExporter:
    def test_export_returns_bytes(self):
        exporter = ZwiftExporter()
        steps = [_make_step("warmup", seconds=600), _make_step(seconds=3600)]
        result = exporter.export(steps, "Test Workout", "A test", 250, None)
        assert isinstance(result, bytes)
        assert b"workout_file" in result

    def test_export_includes_name(self):
        exporter = ZwiftExporter()
        result = exporter.export([], "My Workout", None, 250, None)
        assert b"My Workout" in result

    def test_export_includes_description(self):
        exporter = ZwiftExporter()
        result = exporter.export([], "W", "Long description here", 250, None)
        assert b"Long description here" in result

    def test_export_omits_description_when_none(self):
        exporter = ZwiftExporter()
        result = exporter.export([], "W", None, 250, None)
        assert b"description" not in result

    def test_export_valid_xml(self):
        exporter = ZwiftExporter()
        steps = [
            _make_step("warmup", seconds=300),
            {"kind": "repeat", "repeat_count": 4, "steps": [
                _make_step("active", seconds=120, spec={"type": "pct_ftp", "pct": 110}),
                _make_step("recovery", seconds=60),
            ]},
            _make_step("cooldown", seconds=300),
        ]
        xml_bytes = exporter.export(steps, "Intervals", None, 250, None)
        root = ET.fromstring(xml_bytes)
        assert root.tag == "workout_file"
        workout = root.find("workout")
        assert workout is not None
        tags = [el.tag for el in workout]
        assert "Warmup" in tags
        assert "IntervalsT" in tags
        assert "Cooldown" in tags

    def test_export_with_no_ftp_still_works(self):
        exporter = ZwiftExporter()
        result = exporter.export([_make_step()], "W", None, None, None)
        assert isinstance(result, bytes)
