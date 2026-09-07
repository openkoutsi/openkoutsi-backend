#!/usr/bin/env python3
"""Show why a ride does or doesn't get an aerobic decoupling figure.

The gate reports one word — ``stream_mismatch``, ``fragmented``, ``too_short``
— and a ride that disagrees with it leaves nothing to argue with. This prints
what the two streams actually contain: where each channel stopped recording,
how long for, which stops the two share, and which block of the ride the figure
is measured over.

Usage:
    uv run python scripts/inspect_decoupling.py path/to/ride.fit
    uv run python scripts/inspect_decoupling.py streams.json

``streams.json`` is what ``GET /api/activities/{id}/streams`` returns, so a ride
already ingested can be inspected as it is stored rather than as its file reads:

    curl -H "Authorization: Bearer $TOKEN" \\
        https://your-instance/api/activities/$ID/streams > streams.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from openkoutsi import streams as stream_utils  # noqa: E402
from openkoutsi.training_math import (  # noqa: E402
    DECOUPLING_MIN_PAIRED_COVERAGE,
    analyse_decoupling,
    decoupling_window,
    variability_index,
    weighted_power,
)

# What each hole means, in the terms the gate thinks in.
_MISSING = {
    1: "power only    (meter asleep — the cranks stopped)",
    2: "heart rate    (strap dropped while the meter reported watts)",
    3: "both          (device paused — nothing recorded)",
}


@dataclass
class Hole:
    start_s: int
    length_s: int
    code: int


@dataclass
class Profile:
    grid_s: int
    power_s: int
    hr_s: int
    paired_s: int
    holes: list[Hole]

    @property
    def hr_anchored(self) -> float:
        """Paired seconds over the better-covered channel — the old denominator."""
        covered = max(self.power_s, self.hr_s)
        return self.paired_s / covered if covered else 0.0

    @property
    def power_anchored(self) -> float:
        """Paired seconds over what the power meter recorded — the current one."""
        return self.paired_s / self.power_s if self.power_s else 0.0


def profile_streams(power, heartrate) -> Profile:
    """Where each channel is missing, and how much the two can say together."""
    watts = stream_utils.as_array(power)
    beats = stream_utils.as_array(heartrate)
    n = min(watts.size, beats.size)
    if n == 0:
        return Profile(0, 0, 0, 0, [])

    no_power = np.isnan(watts[:n])
    no_hr = np.isnan(beats[:n])
    # 0 paired, 1 power missing, 2 heart rate missing, 3 both.
    code = no_power.astype(int) + 2 * no_hr.astype(int)

    holes: list[Hole] = []
    edges = np.flatnonzero(np.diff(code)) + 1
    for start, end in zip(
        np.concatenate(([0], edges)), np.concatenate((edges, [n]))
    ):
        if code[start]:
            holes.append(Hole(int(start), int(end - start), int(code[start])))
    holes.sort(key=lambda h: h.length_s, reverse=True)

    return Profile(
        grid_s=n,
        power_s=int(np.count_nonzero(~no_power)),
        hr_s=int(np.count_nonzero(~no_hr)),
        paired_s=int(np.count_nonzero(~no_power & ~no_hr)),
        holes=holes,
    )


def _clock(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _span(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h {rem // 60:02d}m"


def render(profile: Profile, power, heartrate, duration_s: int | None) -> str:
    out: list[str] = []
    grid = profile.grid_s
    out.append(f"Grid                {grid:>7} s   ({_span(grid)} of clock)")
    if duration_s:
        out.append(f"Timer time          {duration_s:>7} s   ({_span(duration_s)})")
    out.append("")
    out.append("Recorded")
    out.append(f"  power             {profile.power_s:>7} s")
    out.append(f"  heart rate        {profile.hr_s:>7} s")
    out.append(f"  paired            {profile.paired_s:>7} s   (both channels)")
    out.append("")

    if profile.holes:
        shown = [h for h in profile.holes if h.length_s >= 30][:15]
        out.append(f"Holes ({len(profile.holes)} in total, longest first)")
        for hole in shown:
            out.append(
                f"  {_clock(hole.start_s)} +{hole.length_s:>5} s   "
                f"{_MISSING[hole.code]}"
            )
        rest = len(profile.holes) - len(shown)
        if rest:
            brief = {code: 0 for code in _MISSING}
            for hole in profile.holes[len(shown):]:
                brief[hole.code] += hole.length_s
            summary = ", ".join(
                f"{seconds} s {_MISSING[code].split('(')[0].strip()}"
                for code, seconds in brief.items()
                if seconds
            )
            out.append(f"  … and {rest} shorter: {summary}")
        out.append("")

    out.append(f"Pairing (refused below {DECOUPLING_MIN_PAIRED_COVERAGE:.0%})")
    out.append(
        f"  over the better-covered channel   {profile.hr_anchored:.3f}"
        "   ← what the gate used to divide by"
    )
    out.append(
        f"  over what the power meter saw     {profile.power_anchored:.3f}"
        "   ← what it divides by now"
    )
    out.append("")

    window = decoupling_window(power, heartrate)
    if window is None:
        out.append("Window              none — no second carries both channels")
    else:
        lo, hi = window
        out.append(
            f"Window              {_clock(lo)}–{_clock(hi)}   "
            f"({_span(hi - lo)}, the longest continuous block)"
        )

    recorded = stream_utils.present(power)
    vi = (
        variability_index(weighted_power(power), float(recorded.mean()))
        if recorded.size
        else None
    )
    result = analyse_decoupling(duration_s or grid, power, heartrate, None, vi)
    if result.pct is not None:
        out.append(
            f"Verdict             {result.pct:+.2f}%   "
            f"measured over {_span(result.window_s or 0)}"
        )
    else:
        out.append(f"Verdict             no figure — {result.reason}")
    out.append(
        "                    (the real gate also reads the ride's stored"
        " workout category and variability index)"
    )
    return "\n".join(out)


def load(path: Path) -> tuple[list, list, int | None]:
    """Streams and timer time from a FIT activity, or from a streams JSON dump."""
    if path.suffix.lower() == ".json":
        blob = json.loads(path.read_text())
        data = blob.get("streams", blob)
        return (
            data.get("power") or [],
            data.get("heartrate") or data.get("heart_rate") or [],
            blob.get("duration_s"),
        )

    from openkoutsi.fit import summarizeWorkout

    profile = summarizeWorkout(str(path))
    return profile.power, profile.heartRate, profile.duration


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <ride.fit | streams.json>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    power, heartrate, duration_s = load(path)
    if not power or not heartrate:
        print("Needs both a power and a heart-rate stream.", file=sys.stderr)
        sys.exit(1)

    print(render(profile_streams(power, heartrate), power, heartrate, duration_s))


if __name__ == "__main__":
    main()
