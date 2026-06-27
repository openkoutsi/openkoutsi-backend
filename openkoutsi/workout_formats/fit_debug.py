"""Human-readable inspection of exported FIT workout files.

Useful for debugging without uploading to a physical device:

    from openkoutsi.workout_formats.fit_debug import describe_fit_workout
    print(describe_fit_workout(fit_bytes))

Or from the command line:

    python scripts/inspect_fit_workout.py my_workout.fit
"""

from __future__ import annotations

import io


def _fmt_duration(duration_type: str, step: dict) -> str:
    if duration_type == "time":
        # fitdecode applies FIT profile scale=1000, so duration_time is in seconds
        s = int(step.get("duration_time") or 0)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    if duration_type == "distance":
        # fitdecode returns duration_distance in mm; divide by 1000 for meters
        mm = step.get("duration_distance") or 0
        return f"{float(mm)/1000:.0f} m"
    return str(duration_type or "open")


def _fmt_target(fields: dict) -> str:
    ttype = fields.get("target_type") or "open"
    if ttype == "power":
        zone = fields.get("target_power_zone")
        if zone:
            return f"power zone {zone}"
        lo = fields.get("custom_target_power_low") or 0
        hi = fields.get("custom_target_power_high") or 0
        if lo >= 1000:
            lo_w, hi_w = lo - 1000, hi - 1000
            return f"power {lo_w}–{hi_w} W" if lo_w != hi_w else f"power {lo_w} W"
        return f"power {lo}–{hi} %FTP" if lo != hi else f"power {lo} %FTP"
    if ttype == "heart_rate":
        lo = fields.get("custom_target_heart_rate_low") or 0
        hi = fields.get("custom_target_heart_rate_high") or 0
        if lo:
            lo_b, hi_b = lo - 100, hi - 100
            return f"hr {lo_b}–{hi_b} bpm" if lo_b != hi_b else f"hr {lo_b} bpm"
        zone = fields.get("target_hr_zone")
        return f"hr zone {zone}" if zone else "heart_rate"
    return str(ttype)


def describe_fit_workout(data: bytes) -> str:
    """Decode FIT workout bytes and return a human-readable step table.

    Example output::

        Workout: "5x2min" — 5 steps
         #  kind    intensity   duration   target
         0  step    warmup      00:10:00   open
         1  step    active      00:02:00   power 100 %FTP
         2  step    recovery    00:01:00   open
         3  repeat  -           →1 ×5      -
         4  step    cooldown    00:10:00   open
    """
    try:
        import fitdecode
    except ImportError:
        return "(fitdecode not installed)"

    workout_name = ""
    raw_steps: list[dict] = []

    with fitdecode.FitReader(io.BytesIO(data)) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "workout":
                fields = {f.name: f.value for f in frame.fields}
                workout_name = fields.get("wkt_name") or ""
            elif frame.name == "workout_step":
                raw_steps.append({f.name: f.value for f in frame.fields})

    header = f'Workout: "{workout_name}" — {len(raw_steps)} steps'
    col = f" {'#':>3}  {'kind':<7}  {'intensity':<10}  {'duration':<10}  target"
    sep = "─" * 62
    rows = [header, col, sep]

    for i, step in enumerate(raw_steps):
        dtype = step.get("duration_type") or ""
        if dtype == "repeat_until_steps_cmplt":
            back = step.get("duration_step", "?")
            count = step.get("repeat_steps", "?")
            rows.append(
                f" {i:>3}  {'repeat':<7}  {'-':<10}  {f'→{back} ×{count}':<10}  -"
            )
        else:
            intensity = str(step.get("intensity") or "-")
            duration = _fmt_duration(dtype, step)
            target = _fmt_target(step)
            rows.append(
                f" {i:>3}  {'step':<7}  {intensity:<10}  {duration:<10}  {target}"
            )

    return "\n".join(rows)
