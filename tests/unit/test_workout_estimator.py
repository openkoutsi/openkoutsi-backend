import pytest
from openkoutsi.workout_estimator import estimate_duration_s, estimate_tss, _step_tss


def _time_step(seconds, step_type="active", spec=None):
    step = {
        "kind": "step",
        "step_type": step_type,
        "duration": {"type": "time", "seconds": seconds},
    }
    if spec:
        step["target"] = {"metric": "power", "spec": spec}
    return step


def _dist_step():
    return {"kind": "step", "step_type": "active", "duration": {"type": "distance", "meters": 1000}}


def _repeat(count, steps):
    return {"kind": "repeat", "repeat_count": count, "steps": steps}


class TestEstimateDurationS:
    def test_single_time_step(self):
        assert estimate_duration_s([_time_step(600)]) == 600

    def test_multiple_steps(self):
        assert estimate_duration_s([_time_step(300), _time_step(600)]) == 900

    def test_distance_step_counts_as_zero(self):
        assert estimate_duration_s([_dist_step()]) == 0

    def test_empty_list(self):
        assert estimate_duration_s([]) == 0

    def test_repeat_multiplies_children(self):
        block = _repeat(5, [_time_step(60), _time_step(30)])
        assert estimate_duration_s([block]) == 5 * 90

    def test_repeat_count_one(self):
        block = _repeat(1, [_time_step(120)])
        assert estimate_duration_s([block]) == 120

    def test_nested_repeat(self):
        inner = _repeat(2, [_time_step(60)])
        outer = _repeat(3, [inner])
        # outer repeats inner 3 times; inner contributes 2*60=120 each time
        assert estimate_duration_s([outer]) == 3 * 2 * 60

    def test_mixed_steps_and_repeat(self):
        steps = [_time_step(300), _repeat(4, [_time_step(60)]), _dist_step()]
        assert estimate_duration_s(steps) == 300 + 4 * 60


class TestEstimateTss:
    def test_no_ftp_returns_none(self):
        assert estimate_tss([_time_step(3600)], None) is None

    def test_zero_ftp_returns_none(self):
        assert estimate_tss([_time_step(3600)], 0) is None

    def test_empty_steps_returns_zero(self):
        assert estimate_tss([], 250) == 0.0

    def test_pct_ftp_spec(self):
        # 1 hour at 100% FTP => IF=1.0, TSS = 1 * 1.0^2 * 100 = 100
        step = _time_step(3600, spec={"type": "pct_ftp", "pct": 100})
        result = estimate_tss([step], 250)
        assert result == pytest.approx(100.0, rel=1e-6)

    def test_pct_ftp_below_threshold(self):
        # 1 hour at 75% FTP => TSS = 1 * 0.75^2 * 100 = 56.25
        step = _time_step(3600, spec={"type": "pct_ftp", "pct": 75})
        result = estimate_tss([step], 250)
        assert result == pytest.approx(56.25, rel=1e-6)

    def test_absolute_spec(self):
        # 1 hour at 250W with FTP 250 => IF=1.0, TSS=100
        step = _time_step(3600, spec={"type": "absolute", "value": 250})
        result = estimate_tss([step], 250)
        assert result == pytest.approx(100.0, rel=1e-6)

    def test_range_spec_uses_midpoint(self):
        # 1 hour at range 200-300W with FTP 250 => midpoint 250W, IF=1.0, TSS=100
        step = _time_step(3600, spec={"type": "range", "low": 200, "high": 300})
        result = estimate_tss([step], 250)
        assert result == pytest.approx(100.0, rel=1e-6)

    def test_unknown_spec_returns_zero(self):
        step = _time_step(3600, spec={"type": "heartrate", "value": 150})
        result = estimate_tss([step], 250)
        assert result == 0.0

    def test_no_power_target_returns_zero(self):
        step = {"kind": "step", "step_type": "active", "duration": {"type": "time", "seconds": 3600}}
        result = estimate_tss([step], 250)
        assert result == 0.0

    def test_repeat_block_tss(self):
        # 5x (5min at 100% FTP) => 5 * (5/60) * 1.0^2 * 100 ≈ 41.67
        interval = _time_step(300, spec={"type": "pct_ftp", "pct": 100})
        block = _repeat(5, [interval])
        result = estimate_tss([block], 250)
        assert result == pytest.approx(5 * (300 / 3600) * 100.0, rel=1e-6)


class TestStepTss:
    def test_non_time_duration_returns_zero(self):
        step = {"kind": "step", "duration": {"type": "distance", "meters": 1000},
                "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 100}}}
        assert _step_tss(step, 250) == 0.0

    def test_no_target_returns_zero(self):
        step = {"kind": "step", "duration": {"type": "time", "seconds": 600}}
        assert _step_tss(step, 250) == 0.0

    def test_non_power_metric_returns_zero(self):
        step = {"kind": "step", "duration": {"type": "time", "seconds": 600},
                "target": {"metric": "hr", "spec": {"type": "absolute", "value": 150}}}
        assert _step_tss(step, 250) == 0.0
