"""Tests for the FIT workout exporter."""
import io
import fitdecode
import pytest
from openkoutsi.workout_formats.fit_workout import (
    _flatten_steps,
    FitWorkoutExporter,
)


def _step(step_type="active", duration_type="time", seconds=300, spec=None, notes=None):
    s = {
        "kind": "step",
        "step_type": step_type,
        "duration": {"type": duration_type, "seconds": seconds},
    }
    if spec:
        s["target"] = {"metric": "power", "spec": spec}
    if notes:
        s["notes"] = notes
    return s


def _repeat(count, steps):
    return {"kind": "repeat", "repeat_count": count, "steps": steps}


def _decode_steps(data: bytes) -> list[dict]:
    """Return decoded workout_step field dicts from a FIT bytes blob."""
    steps = []
    with fitdecode.FitReader(io.BytesIO(data)) as fit:
        for frame in fit:
            if isinstance(frame, fitdecode.FitDataMessage) and frame.name == "workout_step":
                fields = {f.name: f.value for f in frame.fields}
                steps.append(fields)
    return steps


class TestFlattenSteps:
    def test_flat_steps(self):
        flat = _flatten_steps([_step(), _step(step_type="recovery")])
        assert len(flat) == 2
        assert all(f["_type"] == "step" for f in flat)

    def test_empty_list(self):
        assert _flatten_steps([]) == []

    def test_no_repeat_markers_emitted(self):
        # Flattening must never leave a "repeat" marker behind — every entry is a step.
        flat = _flatten_steps([_repeat(3, [_step(seconds=60), _step(step_type="recovery", seconds=30)])])
        assert all(f["_type"] == "step" for f in flat)

    def test_repeat_duplicates_children(self):
        block = _repeat(3, [_step(seconds=60), _step(step_type="recovery", seconds=30)])
        flat = _flatten_steps([block])
        # 2 children x 3 reps = 6 expanded steps
        assert len(flat) == 6
        # order: active, recovery, active, recovery, active, recovery
        assert [f["step_type"] for f in flat] == [
            "active", "recovery", "active", "recovery", "active", "recovery",
        ]

    def test_repeat_single_child(self):
        flat = _flatten_steps([_repeat(5, [_step()])])
        assert len(flat) == 5
        assert all(f["_type"] == "step" for f in flat)

    def test_rep_counter_appended_to_notes(self):
        flat = _flatten_steps([_repeat(3, [_step()])])
        assert flat[0]["notes"] == "(#1/3)"
        assert flat[1]["notes"] == "(#2/3)"
        assert flat[2]["notes"] == "(#3/3)"

    def test_rep_counter_preserves_existing_notes(self):
        flat = _flatten_steps([_repeat(2, [_step(notes="Hard effort")])])
        assert flat[0]["notes"] == "Hard effort (#1/2)"
        assert flat[1]["notes"] == "Hard effort (#2/2)"

    def test_rep_counter_survives_long_notes(self):
        # A note at/over the 50-char FIT limit must still end with the rep marker
        # (the original note is truncated to make room), not have it cut off.
        long_note = "x" * 60
        flat = _flatten_steps([_repeat(3, [_step(notes=long_note)])])
        for rep in range(1, 4):
            notes = flat[rep - 1]["notes"]
            assert len(notes) <= 50
            assert notes.endswith(f"(#{rep}/3)")

    def test_nested_repeats(self):
        # outer 3 x (inner 2 x step) = 6 steps
        inner = _repeat(2, [_step()])
        outer = _repeat(3, [inner])
        flat = _flatten_steps([outer])
        assert len(flat) == 6
        assert all(f["_type"] == "step" for f in flat)

    def test_mixed_steps_and_repeat(self):
        flat = _flatten_steps([_step(), _repeat(2, [_step()]), _step()])
        # warmup + (1 child x 2 reps) + cooldown = 4 steps
        assert len(flat) == 4
        assert all(f["_type"] == "step" for f in flat)

    def test_multiple_children_preceded_by_step(self):
        flat = _flatten_steps([_step(), _repeat(3, [_step(), _step()])])
        # 1 preceding step + (2 children x 3 reps) = 7 steps
        assert len(flat) == 7
        assert all(f["_type"] == "step" for f in flat)


class TestFitWorkoutExporter:
    def test_export_returns_bytes(self):
        exporter = FitWorkoutExporter()
        steps = [_step(seconds=600), _step(step_type="recovery", seconds=300)]
        result = exporter.export(steps, "Test Workout", None, 250, None)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_export_starts_with_fit_magic(self):
        exporter = FitWorkoutExporter()
        result = exporter.export([_step()], "W", None, 250, None)
        assert len(result) >= 14

    def test_zone_target_encodes_as_zone_number(self):
        exporter = FitWorkoutExporter()
        step = _step(spec={"type": "zone", "zone_number": 3})
        data = exporter.export([step], "Zone Test", None, 250, None)
        decoded = _decode_steps(data)
        assert decoded[0]["target_type"] == "power"
        assert decoded[0]["target_power_zone"] == 3

    def test_pct_ftp_target_uses_custom_power_fields_as_percentage(self):
        exporter = FitWorkoutExporter()
        step = _step(spec={"type": "pct_ftp", "pct": 100})
        data = exporter.export([step], "Power Test", None, 250, None)
        decoded = _decode_steps(data)
        assert decoded[0]["target_type"] == "power"
        # Stored as percentage directly (no +1000 offset); device uses its own FTP
        assert decoded[0]["custom_target_power_low"] == 100
        assert decoded[0]["custom_target_power_high"] == 100

    def test_pct_ftp_does_not_require_ftp(self):
        exporter = FitWorkoutExporter()
        step = _step(spec={"type": "pct_ftp", "pct": 90})
        data = exporter.export([step], "No FTP", None, None, None)
        decoded = _decode_steps(data)
        assert decoded[0]["target_type"] == "power"
        assert decoded[0]["custom_target_power_low"] == 90
        assert decoded[0]["custom_target_power_high"] == 90

    def test_absolute_power_target_uses_custom_power_fields(self):
        exporter = FitWorkoutExporter()
        step = _step(spec={"type": "absolute", "value": 200})
        data = exporter.export([step], "Absolute Power", None, 250, None)
        decoded = _decode_steps(data)
        assert decoded[0]["target_type"] == "power"
        assert decoded[0]["custom_target_power_low"] == 1200
        assert decoded[0]["custom_target_power_high"] == 1200

    def test_range_power_target_uses_custom_power_fields(self):
        exporter = FitWorkoutExporter()
        step = _step(spec={"type": "range", "low": 200, "high": 250})
        data = exporter.export([step], "Range Power", None, 250, None)
        decoded = _decode_steps(data)
        assert decoded[0]["target_type"] == "power"
        assert decoded[0]["custom_target_power_low"] == 1200
        assert decoded[0]["custom_target_power_high"] == 1250

    def test_intensity_encoded_correctly(self):
        exporter = FitWorkoutExporter()
        steps = [
            _step(step_type="warmup"),
            _step(step_type="active"),
            _step(step_type="cooldown"),
        ]
        data = exporter.export(steps, "Intensity Test", None, 250, None)
        decoded = _decode_steps(data)
        assert decoded[0]["intensity"] == "warmup"
        assert decoded[1]["intensity"] == "active"
        assert decoded[2]["intensity"] == "cooldown"

    def test_export_with_repeat(self):
        exporter = FitWorkoutExporter()
        block = _repeat(4, [
            _step(seconds=60, spec={"type": "pct_ftp", "pct": 120}),
            _step(step_type="recovery", seconds=30),
        ])
        result = exporter.export([block], "Intervals", None, 250, None)
        assert isinstance(result, bytes)

    def test_repeat_flattened_no_marker(self):
        exporter = FitWorkoutExporter()
        block = _repeat(3, [
            _step(seconds=1200, spec={"type": "pct_ftp", "pct": 90}),
            _step(step_type="recovery", seconds=600),
        ])
        data = exporter.export([block], "3x20", None, 250, None)
        decoded = _decode_steps(data)
        # No native repeat marker — the block is expanded into 2 x 3 = 6 steps.
        assert not any(s.get("duration_type") == "repeat_until_steps_cmplt" for s in decoded)
        assert len(decoded) == 6

    def test_repeat_only_workout_expands_to_step_sequence(self):
        # The block [active, recovery] repeated 3x becomes 6 sequential steps,
        # alternating active/recovery, with no loop-back marker.
        exporter = FitWorkoutExporter()
        block = _repeat(3, [
            _step(seconds=60, spec={"type": "pct_ftp", "pct": 120}),
            _step(step_type="recovery", seconds=30),
        ])
        data = exporter.export([block], "Intervals", None, 250, None)
        decoded = _decode_steps(data)
        assert len(decoded) == 6
        assert [s["intensity"] for s in decoded] == [
            "active", "recovery", "active", "recovery", "active", "recovery",
        ]

    def test_repeat_after_warmup_expands_in_order(self):
        # warmup + 5x(active+recovery) → 1 + 10 = 11 sequential steps.
        exporter = FitWorkoutExporter()
        steps = [
            _step(step_type="warmup", seconds=600),
            _repeat(5, [
                _step(seconds=120, spec={"type": "pct_ftp", "pct": 110}),
                _step(step_type="recovery", seconds=60),
            ]),
        ]
        data = exporter.export(steps, "Warmup + Intervals", None, 250, None)
        decoded = _decode_steps(data)
        assert len(decoded) == 11
        assert decoded[0]["intensity"] == "warmup"
        assert all(s.get("duration_type") != "repeat_until_steps_cmplt" for s in decoded)
        # rep counters present on the repeated steps
        assert decoded[1]["notes"].endswith("(#1/5)")
        assert decoded[-1]["notes"].endswith("(#5/5)")

    def test_full_structured_workout_expands_block(self):
        # warmup + 3x(active+recovery) + cooldown → 1 + 6 + 1 = 8 steps.
        exporter = FitWorkoutExporter()
        steps = [
            _step(step_type="warmup", seconds=600),
            _repeat(3, [
                _step(seconds=300, spec={"type": "pct_ftp", "pct": 100}),
                _step(step_type="recovery", seconds=150),
            ]),
            _step(step_type="cooldown", seconds=300),
        ]
        data = exporter.export(steps, "Full Workout", None, 250, None)
        decoded = _decode_steps(data)
        assert len(decoded) == 8
        assert decoded[0]["intensity"] == "warmup"
        assert decoded[-1]["intensity"] == "cooldown"
        assert not any(s.get("duration_type") == "repeat_until_steps_cmplt" for s in decoded)

    def test_nested_repeat_expands_fully(self):
        # outer 3 x (inner 2 x active) = 6 active steps, no markers.
        exporter = FitWorkoutExporter()
        steps = [
            _repeat(3, [
                _repeat(2, [_step(seconds=60, spec={"type": "pct_ftp", "pct": 110})]),
            ]),
        ]
        data = exporter.export(steps, "Nested", None, 250, None)
        decoded = _decode_steps(data)
        assert len(decoded) == 6
        assert all(s.get("duration_type") != "repeat_until_steps_cmplt" for s in decoded)

    def test_export_distance_duration(self):
        exporter = FitWorkoutExporter()
        step = {"kind": "step", "step_type": "active", "duration": {"type": "distance", "meters": 1000}}
        result = exporter.export([step], "Distance", None, 250, None)
        assert isinstance(result, bytes)

    def test_export_open_duration(self):
        exporter = FitWorkoutExporter()
        step = {"kind": "step", "step_type": "active", "duration": {"type": "open"}}
        result = exporter.export([step], "Open", None, 250, None)
        assert isinstance(result, bytes)

    def test_export_with_notes(self):
        exporter = FitWorkoutExporter()
        step = _step(notes="Steady effort")
        result = exporter.export([step], "Notes", None, 250, None)
        assert isinstance(result, bytes)

    def test_export_hr_target(self):
        exporter = FitWorkoutExporter()
        step = {
            "kind": "step", "step_type": "active",
            "duration": {"type": "time", "seconds": 600},
            "target": {"metric": "hr", "spec": {"type": "absolute", "value": 150}},
        }
        result = exporter.export([step], "HR", None, 250, None)
        assert isinstance(result, bytes)

    def test_hr_absolute_target_uses_custom_heart_rate_fields(self):
        # Mirrors the power encoding: custom values must go into custom_target_heart_rate_low/high
        # with a +100 offset (same convention as custom_target_power_low/high uses +1000).
        # Storing bpm+100 in target_value is wrong — devices read target_value as a zone number.
        exporter = FitWorkoutExporter()
        step = {
            "kind": "step", "step_type": "active",
            "duration": {"type": "time", "seconds": 600},
            "target": {"metric": "hr", "spec": {"type": "absolute", "value": 150}},
        }
        data = exporter.export([step], "HR Absolute", None, None, None)
        decoded = _decode_steps(data)
        assert decoded[0]["target_type"] == "heart_rate"
        # custom_target_heart_rate_low/high must carry the BPM+100 encoded value
        assert decoded[0]["custom_target_heart_rate_low"] == 250   # 150 + 100
        assert decoded[0]["custom_target_heart_rate_high"] == 250  # 150 + 100

    def test_export_no_ftp(self):
        exporter = FitWorkoutExporter()
        result = exporter.export([_step()], "No FTP", None, None, None)
        assert isinstance(result, bytes)
