"""Round-trip and inspection tests for FIT workout export.

Round-trip tests encode a full workout definition, decode the resulting FIT
bytes with fitdecode, and assert every field in the decoded output matches
the input — catching encoding bugs without needing a physical device.

The describe_fit_workout tests verify the human-readable inspection helper
that developers can use instead of uploading to a device.
"""
import io
import fitdecode
import pytest

from openkoutsi.workout_formats.fit_workout import FitWorkoutExporter
from openkoutsi.workout_formats.fit_debug import describe_fit_workout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export(steps, name="Test", ftp=250):
    return FitWorkoutExporter().export(steps, name, None, ftp, None)


def _decode(data: bytes) -> list[dict]:
    steps = []
    with fitdecode.FitReader(io.BytesIO(data)) as fit:
        for frame in fit:
            if isinstance(frame, fitdecode.FitDataMessage) and frame.name == "workout_step":
                steps.append({f.name: f.value for f in frame.fields})
    return steps


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_warmup_intervals_cooldown(self):
        """Full structured workout: warmup + 5x(active+recovery) + cooldown."""
        steps = [
            {"kind": "step", "step_type": "warmup",
             "duration": {"type": "time", "seconds": 600}},
            {"kind": "repeat", "repeat_count": 5, "steps": [
                {"kind": "step", "step_type": "active",
                 "duration": {"type": "time", "seconds": 120},
                 "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 100}}},
                {"kind": "step", "step_type": "recovery",
                 "duration": {"type": "time", "seconds": 60}},
            ]},
            {"kind": "step", "step_type": "cooldown",
             "duration": {"type": "time", "seconds": 600}},
        ]
        decoded = _decode(_export(steps, "5x2min"))

        # Flattened layout: warmup(0) + 5x(active, recovery) + cooldown = 12 steps
        assert len(decoded) == 12
        assert not any(s.get("duration_type") == "repeat_until_steps_cmplt" for s in decoded)

        warmup, cooldown = decoded[0], decoded[-1]

        assert warmup["intensity"] == "warmup"
        assert warmup["duration_type"] == "time"
        assert warmup["duration_time"] == pytest.approx(600.0)

        # The 5 active/recovery pairs occupy indices 1..10.
        for rep in range(5):
            active = decoded[1 + rep * 2]
            recovery = decoded[2 + rep * 2]

            assert active["intensity"] == "active"
            assert active["duration_type"] == "time"
            assert active["duration_time"] == pytest.approx(120.0)
            assert active["target_type"] == "power"
            assert active["custom_target_power_low"] == 100
            assert active["custom_target_power_high"] == 100
            assert active["notes"].endswith(f"(#{rep + 1}/5)")

            assert recovery["intensity"] == "recovery"
            assert recovery["duration_time"] == pytest.approx(60.0)
            assert recovery["notes"].endswith(f"(#{rep + 1}/5)")

        assert cooldown["intensity"] == "cooldown"
        assert cooldown["duration_time"] == pytest.approx(600.0)

    def test_repeat_only(self):
        """Repeat block with no preceding steps — first child is at index 0."""
        steps = [
            {"kind": "repeat", "repeat_count": 3, "steps": [
                {"kind": "step", "step_type": "active",
                 "duration": {"type": "time", "seconds": 300},
                 "target": {"metric": "power", "spec": {"type": "zone", "zone_number": 4}}},
                {"kind": "step", "step_type": "recovery",
                 "duration": {"type": "time", "seconds": 120}},
            ]},
        ]
        decoded = _decode(_export(steps, "3x5min"))

        # [active, recovery] x 3 = 6 flattened steps, no repeat marker.
        assert len(decoded) == 6
        assert not any(s.get("duration_type") == "repeat_until_steps_cmplt" for s in decoded)
        assert [s["intensity"] for s in decoded] == [
            "active", "recovery", "active", "recovery", "active", "recovery",
        ]
        assert decoded[0]["target_type"] == "power"
        assert decoded[0]["target_power_zone"] == 4

    def test_multiple_repeat_blocks(self):
        """Two back-to-back repeat blocks — second block's marker must reference correct index."""
        steps = [
            {"kind": "step", "step_type": "warmup",
             "duration": {"type": "time", "seconds": 300}},
            {"kind": "repeat", "repeat_count": 3, "steps": [
                {"kind": "step", "step_type": "active",
                 "duration": {"type": "time", "seconds": 60},
                 "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 120}}},
                {"kind": "step", "step_type": "recovery",
                 "duration": {"type": "time", "seconds": 30}},
            ]},
            {"kind": "repeat", "repeat_count": 2, "steps": [
                {"kind": "step", "step_type": "active",
                 "duration": {"type": "time", "seconds": 300},
                 "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 90}}},
                {"kind": "step", "step_type": "recovery",
                 "duration": {"type": "time", "seconds": 120}},
            ]},
            {"kind": "step", "step_type": "cooldown",
             "duration": {"type": "time", "seconds": 300}},
        ]
        decoded = _decode(_export(steps, "Mixed"))

        # Flattened: warmup + 3x(active,recovery) + 2x(active,recovery) + cooldown
        # = 1 + 6 + 4 + 1 = 12 steps, no repeat markers.
        assert len(decoded) == 12
        assert not any(s.get("duration_type") == "repeat_until_steps_cmplt" for s in decoded)
        assert decoded[0]["intensity"] == "warmup"
        assert decoded[-1]["intensity"] == "cooldown"
        # First block reps run 60s active; second block reps run 300s active.
        assert decoded[1]["duration_time"] == pytest.approx(60.0)
        assert decoded[7]["duration_time"] == pytest.approx(300.0)

    def test_hr_absolute_target_round_trip(self):
        """HR absolute targets encode into custom_target_heart_rate_low/high, not target_value."""
        steps = [
            {"kind": "step", "step_type": "active",
             "duration": {"type": "time", "seconds": 1800},
             "target": {"metric": "hr", "spec": {"type": "absolute", "value": 155}}},
        ]
        decoded = _decode(_export(steps, "HR Zone"))
        assert decoded[0]["target_type"] == "heart_rate"
        assert decoded[0]["custom_target_heart_rate_low"] == 255   # 155 + 100
        assert decoded[0]["custom_target_heart_rate_high"] == 255

    def test_distance_duration_round_trip(self):
        steps = [
            {"kind": "step", "step_type": "active",
             "duration": {"type": "distance", "meters": 1000}},
        ]
        decoded = _decode(_export(steps, "1km"))
        assert decoded[0]["duration_type"] == "distance"
        assert decoded[0]["duration_distance"] == pytest.approx(1000000.0)  # fitdecode returns mm (1000m = 1_000_000mm)

    def test_absolute_power_round_trip(self):
        steps = [
            {"kind": "step", "step_type": "active",
             "duration": {"type": "time", "seconds": 600},
             "target": {"metric": "power", "spec": {"type": "absolute", "value": 250}}},
        ]
        decoded = _decode(_export(steps, "Absolute"))
        assert decoded[0]["custom_target_power_low"] == 1250   # 250 + 1000
        assert decoded[0]["custom_target_power_high"] == 1250

    def test_range_power_round_trip(self):
        steps = [
            {"kind": "step", "step_type": "active",
             "duration": {"type": "time", "seconds": 600},
             "target": {"metric": "power", "spec": {"type": "range", "low": 200, "high": 250}}},
        ]
        decoded = _decode(_export(steps, "Range"))
        assert decoded[0]["custom_target_power_low"] == 1200
        assert decoded[0]["custom_target_power_high"] == 1250

    def test_nested_repeat_round_trip(self):
        """Nested repeat flattens to outer x inner copies of the inner steps."""
        steps = [
            {"kind": "step", "step_type": "warmup",
             "duration": {"type": "time", "seconds": 300}},
            {"kind": "repeat", "repeat_count": 3, "steps": [
                {"kind": "repeat", "repeat_count": 2, "steps": [
                    {"kind": "step", "step_type": "active",
                     "duration": {"type": "time", "seconds": 60}},
                ]},
            ]},
        ]
        decoded = _decode(_export(steps, "Nested"))

        # Flattened: warmup + (3 x 2 = 6 active steps) = 7 steps, no markers.
        assert len(decoded) == 7
        assert not any(s.get("duration_type") == "repeat_until_steps_cmplt" for s in decoded)
        assert decoded[0]["intensity"] == "warmup"
        assert all(s["intensity"] == "active" for s in decoded[1:])


# ---------------------------------------------------------------------------
# describe_fit_workout tests
# ---------------------------------------------------------------------------

class TestDescribeFitWorkout:
    def test_contains_workout_name(self):
        steps = [{"kind": "step", "step_type": "active",
                  "duration": {"type": "time", "seconds": 300}}]
        output = describe_fit_workout(_export(steps, "My Workout"))
        assert "My Workout" in output

    def test_shows_step_count(self):
        steps = [
            {"kind": "step", "step_type": "warmup", "duration": {"type": "time", "seconds": 300}},
            {"kind": "step", "step_type": "active", "duration": {"type": "time", "seconds": 600}},
        ]
        output = describe_fit_workout(_export(steps, "W"))
        assert "2 steps" in output

    def test_repeat_block_flattened_in_description(self):
        """Repeat blocks are flattened, so the description lists every expanded step
        (warmup + 4x(active,recovery) = 9 steps) and contains no repeat marker."""
        steps = [
            {"kind": "step", "step_type": "warmup", "duration": {"type": "time", "seconds": 300}},
            {"kind": "repeat", "repeat_count": 4, "steps": [
                {"kind": "step", "step_type": "active",
                 "duration": {"type": "time", "seconds": 60}},
                {"kind": "step", "step_type": "recovery",
                 "duration": {"type": "time", "seconds": 30}},
            ]},
        ]
        output = describe_fit_workout(_export(steps, "Intervals"))
        assert "9 steps" in output
        # No native repeat loop-back marker remains.
        assert "→" not in output

    def test_power_pct_ftp_shown_in_description(self):
        steps = [{"kind": "step", "step_type": "active",
                  "duration": {"type": "time", "seconds": 300},
                  "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 105}}}]
        output = describe_fit_workout(_export(steps, "W"))
        assert "105" in output
        assert "FTP" in output

    def test_duration_formatted_as_time(self):
        steps = [{"kind": "step", "step_type": "active",
                  "duration": {"type": "time", "seconds": 3661}}]  # 1h 1m 1s
        output = describe_fit_workout(_export(steps, "W"))
        assert "01:01:01" in output

    def test_returns_string(self):
        steps = [{"kind": "step", "step_type": "active",
                  "duration": {"type": "time", "seconds": 60}}]
        assert isinstance(describe_fit_workout(_export(steps, "W")), str)
