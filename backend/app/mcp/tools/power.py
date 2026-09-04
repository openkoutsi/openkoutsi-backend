"""``get_power_profile`` — what the athlete can actually produce (issue #42).

The power curve is capability, where Fitness/Fatigue is recent *behaviour*, and a
model prescribing intervals needs the first: "4×8 min at threshold" is a
different session for an 8-minute best of 290 W than for 420 W.

The single best effort per standard duration, not the API's top three — ranks two
and three answer "was this a fluke", a question about the chart — which is what
keeps this inside the size budget alongside the FTP estimates.

Asks for ``athlete:read`` as well as ``metrics:read`` because the profile FTP,
which every zone and intensity figure is computed against, is athlete data rather
than a derived metric.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from backend.app.mcp.dispatch import ToolRun
from backend.app.mcp.registry import ToolArgs, tool
from backend.app.mcp.shaping import hhmm, round_or_none
from backend.app.services.power_profile import rank1_bests
from backend.app.services.weight import w_per_kg
from openkoutsi.training_math import (
    CP_FIT_DURATIONS,
    POWER_BEST_DURATIONS,
    estimate_cp_wprime,
    estimate_ftp_simple,
)


class PowerProfileArgs(ToolArgs):
    days: Optional[int] = Field(
        None,
        ge=7,
        le=3650,
        description=(
            "Only consider efforts from the past N days. Omit for all-time. 90 "
            "or 180 answers 'what can they do now'; all-time answers 'what have "
            "they ever done', which for an athlete returning from a layoff is a "
            "different and possibly misleading number."
        ),
    )


class PowerBest(BaseModel):
    duration_s: int = Field(..., description="Effort length in seconds (s).")
    duration_text: str = Field(..., description="Effort length as readable text, e.g. '20 min'.")
    power_w: float = Field(..., description="Best average power sustained for that length, in watts (W).")
    w_per_kg: Optional[float] = Field(
        None,
        description=(
            "The same effort in watts per kilogram (W/kg), using the athlete's "
            "bodyweight at the time of the effort. Null when no weight was "
            "recorded then."
        ),
    )


class FtpEstimates(BaseModel):
    profile_ftp_w: Optional[int] = Field(
        None,
        description=(
            "FTP currently set on the athlete's profile, in watts (W). This is "
            "what every zone and intensity figure on the platform is computed "
            "against, whether or not the estimates below agree with it."
        ),
    )
    twenty_min_power_w: Optional[float] = Field(
        None, description="Best 20-minute power in the window, in watts (W)."
    )
    ftp_from_20min_w: Optional[int] = Field(
        None, description="FTP estimated as 95% of the 20-minute best, in watts (W). Null without a 20-minute effort."
    )
    critical_power_w: Optional[float] = Field(
        None,
        description=(
            "Critical power in watts (W) from a work–time fit over the 2–20 "
            "minute bests. Null when those efforts are too sparse to fit."
        ),
    )
    w_prime_j: Optional[float] = Field(
        None, description="Anaerobic work capacity W′ in joules (J) from the same fit."
    )
    disagreement_w: Optional[float] = Field(
        None,
        description=(
            "Gap between the profile FTP and the best available estimate, in "
            "watts (W). Positive means the estimates are above the profile "
            "figure, i.e. the athlete's zones may be stale and too easy."
        ),
    )


class PowerProfile(BaseModel):
    window_days: Optional[int] = Field(
        None, description="The window these bests were drawn from, in days. Null means all-time."
    )
    weight_kg: Optional[float] = Field(
        None, description="Athlete's current bodyweight in kilograms (kg), for context on the W/kg figures."
    )
    bests: list[PowerBest] = Field(
        default_factory=list, description="Single best effort per standard duration, shortest first."
    )
    ftp: FtpEstimates = Field(..., description="Profile FTP alongside the estimates derived from the curve.")


@tool(
    name="get_power_profile",
    title="Power profile",
    scopes={"metrics:read", "athlete:read"},
    arguments=PowerProfileArgs,
    returns=PowerProfile,
)
async def get_power_profile(run: ToolRun, args: PowerProfileArgs) -> PowerProfile:
    """The athlete's power curve — their best sustained power at each standard
    duration, in watts and in watts per kilogram — together with the FTP on their
    profile and the FTP estimates derived from the curve (95% of the 20-minute
    best, and a critical-power fit with W′).

    This is capability, not recent training: use it to pitch interval targets in
    real numbers rather than in percentages the athlete has to convert.

    Pass 'days' to ask what they can do *now*. An all-time curve on an athlete
    returning from a layoff describes a rider who does not currently exist, and
    prescribing against it is how a session becomes unrideable.

    When the estimates sit well above the profile FTP, the athlete's zones are
    probably stale — 'disagreement_w' is that gap, and it is worth raising.
    """
    athlete = run.athlete
    since = (
        datetime.combine(run.today - timedelta(days=args.days), time.min)
        if args.days is not None
        else None
    )

    curve = await rank1_bests(athlete.id, run.session, list(POWER_BEST_DURATIONS), since=since)
    weight = athlete.weight_kg

    bests = [
        PowerBest(
            duration_s=duration,
            duration_text=hhmm(duration) or f"{duration} s",
            power_w=round(curve[duration], 1),
            w_per_kg=round_or_none(w_per_kg(curve[duration], weight), 2),
        )
        for duration in POWER_BEST_DURATIONS
        if duration in curve
    ]

    fit_window = {d: w for d, w in curve.items() if d in CP_FIT_DURATIONS}
    twenty = curve.get(1200)
    ftp_simple = estimate_ftp_simple(twenty)
    cp, w_prime = estimate_cp_wprime(fit_window)

    # Prefer the CP fit over the 20-minute rule when both exist: it is the one
    # that used more than a single effort.
    best_estimate = cp if cp is not None else ftp_simple
    disagreement = (
        round(best_estimate - athlete.ftp, 1)
        if best_estimate is not None and athlete.ftp
        else None
    )

    return PowerProfile(
        window_days=args.days,
        weight_kg=round_or_none(weight, 1),
        bests=bests,
        ftp=FtpEstimates(
            profile_ftp_w=athlete.ftp,
            twenty_min_power_w=round_or_none(twenty, 1),
            ftp_from_20min_w=round(ftp_simple) if ftp_simple is not None else None,
            critical_power_w=round_or_none(cp, 1),
            w_prime_j=round_or_none(w_prime, 0),
            disagreement_w=disagreement,
        ),
    )
