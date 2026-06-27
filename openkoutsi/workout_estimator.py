"""Estimate duration and TSS for a workout definition from its step tree."""

from __future__ import annotations


def estimate_duration_s(steps: list[dict]) -> int:
    """Recursively sum time-based step durations. Open/distance steps count as 0."""
    total = 0
    for step in steps:
        if step.get("kind") == "repeat":
            child_duration = estimate_duration_s(step.get("steps", []))
            total += child_duration * step.get("repeat_count", 1)
        elif step.get("kind") == "step":
            dur = step.get("duration", {})
            if dur.get("type") == "time":
                total += dur.get("seconds", 0)
    return total


def _step_tss(step: dict, ftp: int) -> float:
    """Estimate TSS contribution for a single step."""
    dur = step.get("duration", {})
    if dur.get("type") != "time":
        return 0.0
    duration_h = dur.get("seconds", 0) / 3600.0
    target = step.get("target")
    if not target or target.get("metric") != "power":
        return 0.0
    spec = target.get("spec", {})
    spec_type = spec.get("type")
    if spec_type == "pct_ftp":
        pct = spec.get("pct", 0) / 100.0
    elif spec_type == "absolute":
        pct = spec.get("value", 0) / ftp if ftp else 0.0
    elif spec_type == "range":
        low = spec.get("low", 0)
        high = spec.get("high", 0)
        pct = ((low + high) / 2.0) / ftp if ftp else 0.0
    else:
        return 0.0
    # TSS = duration_h * IF^2 * 100 (simplified steady-state approximation)
    return duration_h * (pct ** 2) * 100.0


def estimate_tss(steps: list[dict], ftp: int | None) -> float | None:
    """Estimate total TSS for a workout. Returns None if athlete has no FTP."""
    if not ftp:
        return None
    return _tss_recursive(steps, ftp)


def _tss_recursive(steps: list[dict], ftp: int) -> float:
    total = 0.0
    for step in steps:
        if step.get("kind") == "repeat":
            child_tss = _tss_recursive(step.get("steps", []), ftp)
            total += child_tss * step.get("repeat_count", 1)
        elif step.get("kind") == "step":
            total += _step_tss(step, ftp)
    return total
