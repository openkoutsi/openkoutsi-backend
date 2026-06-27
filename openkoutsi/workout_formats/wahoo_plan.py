"""Serialise workout definitions to the Wahoo public ``plan.json`` format.

The plan.json object (schema v1.0.0) has a ``header`` section with athlete /
workout metadata and an ``intervals`` array. Unlike the FIT exporter — which
flattens repeats to work around a device rendering bug — the Wahoo plan format
supports native repeats via ``exit_trigger_type: "repeat"`` with a nested
``intervals`` array, so our ``RepeatBlock`` maps directly.

See ``docs/plan-json-format`` (Wahoo) for the authoritative schema.
"""

from __future__ import annotations

_PLAN_VERSION = "1.0.0"

# Internal step_type → Wahoo INTENSITY_TYPE enum value.
_INTENSITY = {
    "warmup": "wu",
    "active": "active",
    "recovery": "recover",
    "cooldown": "cd",
    "rest": "rest",
    "other": "active",
}

# sport_type → WORKOUT_TYPE_FAMILY (0 = Biking, 1 = Running).
_RUN_SPORTS = {"Run", "TrackRun", "TrailRun", "Treadmill", "VirtualRun"}

# sport_type values that take place indoors → WORKOUT_TYPE_LOCATION 0 (Indoor).
_INDOOR_SPORTS = {"VirtualRide", "Treadmill", "Elliptical", "StairStepper"}

# When an interval has no fixed duration we cannot represent "open" in a plan;
# fall back to this many seconds so the interval is still playable on-device.
_OPEN_FALLBACK_S = 600


def build_wahoo_plan(
    steps: list[dict],
    workout_name: str,
    workout_description: str | None,
    sport_type: str,
    athlete_ftp: int | None,
    athlete_power_zones: list[dict] | None,
) -> dict:
    """Build a Wahoo plan.json object from internal workout steps.

    Returns a dict ready to be JSON-serialised and uploaded to ``POST /v1/plans``.
    """
    family = 1 if sport_type in _RUN_SPORTS else 0
    location = 0 if sport_type in _INDOOR_SPORTS else 1

    header: dict = {
        "name": workout_name,
        "version": _PLAN_VERSION,
        "workout_type_family": family,
        "workout_type_location": location,
    }
    if workout_description:
        header["description"] = workout_description[:5000]
    if athlete_ftp:
        # Required whenever an interval uses a relative ``ftp`` target.
        header["ftp"] = int(athlete_ftp)

    intervals = [_interval(step, athlete_power_zones) for step in steps]

    return {"header": header, "intervals": intervals}


def _interval(step: dict, power_zones: list[dict] | None) -> dict:
    """Convert a single step or repeat block into a Wahoo interval object."""
    if step.get("kind") == "repeat":
        # Wahoo's repeat value is the number of iterations AFTER the first, so a
        # repeat_count of 3 (run three times) becomes a value of 2.
        count = int(step.get("repeat_count", 1))
        return {
            "name": f"{count}x",
            "exit_trigger_type": "repeat",
            "exit_trigger_value": max(count - 1, 0),
            "intervals": [_interval(s, power_zones) for s in step.get("steps", [])],
        }

    trigger_type, trigger_value = _duration(step.get("duration", {}))
    interval: dict = {
        "exit_trigger_type": trigger_type,
        "exit_trigger_value": trigger_value,
        "intensity_type": _INTENSITY.get(step.get("step_type", "active"), "active"),
    }
    if step.get("notes"):
        interval["name"] = step["notes"]

    target = _target(step.get("target"), power_zones)
    if target is not None:
        interval["targets"] = [target]

    return interval


def _duration(dur: dict) -> tuple[str, float]:
    """Map an internal duration to (exit_trigger_type, exit_trigger_value)."""
    dur_type = dur.get("type")
    if dur_type == "time":
        return "time", float(dur["seconds"])
    if dur_type == "distance":
        return "distance", float(dur["meters"])
    # "open" has no native representation — fall back to a fixed time block.
    return "time", float(_OPEN_FALLBACK_S)


def _target(target: dict | None, power_zones: list[dict] | None) -> dict | None:
    """Map an internal ``WorkoutTarget`` to a single Wahoo target dict.

    Returns ``None`` when there is no target (open interval) or the target type
    cannot be represented. Wahoo intervals carry an array of targets, but devices
    only honour the first, so we emit a single target.
    """
    if not target:
        return None

    metric = target.get("metric")
    spec = target.get("spec") or {}
    spec_type = spec.get("type")

    if metric == "power":
        if spec_type == "pct_ftp":
            frac = float(spec["pct"]) / 100.0
            return {"type": "ftp", "low": frac, "high": frac}
        if spec_type == "absolute":
            watts = float(spec["value"])
            return {"type": "watts", "low": watts, "high": watts}
        if spec_type == "range":
            return {"type": "watts", "low": float(spec["low"]), "high": float(spec["high"])}
        if spec_type == "zone":
            return _zone_target(int(spec["zone_number"]), power_zones)
        return None

    if metric == "hr" and spec_type == "absolute":
        bpm = float(spec["value"])
        return {"type": "hr", "low": bpm, "high": bpm}

    if metric == "cadence" and spec_type == "absolute":
        rpm = float(spec["value"])
        return {"type": "rpm", "low": rpm, "high": rpm}

    if metric == "pace" and spec_type == "absolute":
        # internal pace is stored in m/s, matching Wahoo's absolute speed target
        ms = float(spec["value"])
        return {"type": "speed", "low": ms, "high": ms}

    return None


def _zone_target(zone_number: int, power_zones: list[dict] | None) -> dict | None:
    """Resolve a power-zone number to an absolute watts target via athlete zones."""
    if not power_zones or zone_number < 1 or zone_number > len(power_zones):
        return None
    zone = power_zones[zone_number - 1]
    low = zone.get("low")
    high = zone.get("high")
    if low is None or high is None:
        return None
    return {"type": "watts", "low": float(low), "high": float(high)}
