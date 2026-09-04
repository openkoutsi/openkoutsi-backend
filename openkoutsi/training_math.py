"""
Shared training load calculations — peak power, weighted power, Load, distance bests.

The per-second stream math is vectorised with numpy.  Public functions take any
sequence (``np.asarray`` at the top) and return plain Python floats and lists, so
results from ``ActivityStream.data`` persist straight back as JSON — an
``np.float64`` reaching the ORM fails at the JSON encoder, not the call site.
"""

import math
from typing import Sequence

import numpy as np

from . import streams

POWER_BEST_DURATIONS: list[int] = [
    1, 3, 5, 10, 15, 30, 45, 60, 120, 180, 300, 480, 600,
    900, 1200, 1800, 2700, 3600, 7200, 10800, 14400,
    18000, 21600, 25200, 28800,
]


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    """Cumulative sums with a leading zero, so ``c[j] - c[i]`` sums ``[i:j]``."""
    return np.concatenate(([0.0], np.cumsum(values)))


def _peak_from_prefix(c: np.ndarray, duration_s: int) -> float | None:
    """Highest mean over any ``duration_s``-wide window, from a prefix sum."""
    if c.size - 1 < duration_s or duration_s <= 0:
        return None
    return float((c[duration_s:] - c[:-duration_s]).max()) / duration_s


def peak_average_power(
    stream: Sequence[float | None], duration_s: int
) -> float | None:
    """
    Return the highest mean wattage over any contiguous `duration_s`-second
    window in `stream`.  Returns None if the stream is shorter than the window.

    One prefix sum serves every window, so each duration costs a single
    vectorised subtraction rather than a pass over the stream.

    Gaps are dropped rather than counted as zero watts: a best is a claim about
    the rider, and a power meter that missed a second is not evidence the rider
    stopped pedalling.
    """
    return _peak_from_prefix(_prefix_sum(streams.present(stream)), duration_s)


def compute_power_bests(stream: Sequence[float | None]) -> dict[int, float]:
    """
    Compute peak_average_power for every standard duration in POWER_BEST_DURATIONS.
    Only returns entries where the stream is long enough to cover the duration.
    """
    c = _prefix_sum(streams.present(stream))
    return {
        d: v
        for d in POWER_BEST_DURATIONS
        if (v := _peak_from_prefix(c, d)) is not None
    }


# Durations (seconds) used for the Critical Power fit: 2–20 minutes.
CP_FIT_DURATIONS: list[int] = [120, 180, 300, 480, 900, 1200]


def estimate_ftp_simple(twenty_min_power: float | None) -> float | None:
    """
    Simple FTP estimate: 95% of the 20-minute (1200s) best mean power.
    Returns None if no 20-minute best is available.
    """
    if twenty_min_power is None:
        return None
    return 0.95 * twenty_min_power


def estimate_cp_wprime(bests: dict[int, float]) -> tuple[float | None, float | None]:
    """
    Estimate Critical Power (CP) and anaerobic work capacity (W') from the
    2–20 minute power bests using the linear work–time model.

    For each duration t (seconds) with mean power P(t) watts, total work is
    W(t) = P(t)·t joules.  The model W(t) = CP·t + W' is fit by ordinary
    least squares; the slope is CP (watts) and the intercept is W' (joules).

    `bests` maps duration_s -> mean power (watts); only durations in
    CP_FIT_DURATIONS are used.  Needs at least 2 data points.  Returns
    (None, None) if there are fewer than 2 points or the fit yields CP <= 0.
    """
    durations = [d for d in CP_FIT_DURATIONS if d in bests]
    if len(durations) < 2:
        return None, None

    # Slope is CP, intercept is W'.
    cp, w_prime = np.polyfit(
        [float(d) for d in durations], [bests[d] * d for d in durations], 1
    )
    if cp <= 0:
        return None, None
    return float(cp), float(w_prime)


# ---------------------------------------------------------------------------
# Advanced power–duration models
#
# Each model predicts mean power P(t) for an effort of ``t`` seconds and is fit
# to the athlete's rank-1 best power per duration (same input as the CP fit
# above).  The two nonlinear models (3-parameter CP and exponential) evaluate
# their single nonlinear parameter over a dense grid and solve the rest in closed
# form by ordinary least squares, the whole grid as one vectorised matrix solve
# (``_ols_grid``).  A continuous optimiser (scipy) was measured and rejected: it
# improves the fit by 0.0001 %, runs 3× slower on ~10 points, and its bounded
# solvers manufacture an answer where the grid correctly returns None.
#
# Fit-duration windows below are deliberately conservative and tunable.
# ---------------------------------------------------------------------------

# 3-parameter CP is usable down to ~15–30 s; fit 30 s – 20 min.
CP3_FIT_DURATIONS: list[int] = [30, 60, 120, 180, 300, 480, 600, 900, 1200]

# Exponential spans sprint → threshold; include short efforts near P_max.
EXP_FIT_DURATIONS: list[int] = [5, 15, 30, 60, 120, 180, 300, 480, 600, 900, 1200]

# Power law (Riegel) describes the endurance portion; skip the very short
# sprints where it over-predicts.  Extended out to the longest bests present.
POWER_LAW_FIT_DURATIONS: list[int] = [
    60, 120, 180, 300, 480, 600, 900, 1200, 1800, 2700, 3600,
    7200, 10800, 14400, 18000, 21600, 25200, 28800,
]

# A flat power curve fits b ≈ 0, and the least-squares solver returns a few ulp
# either side of it — so "is this a decaying power law" cannot be asked as
# ``b < 0``.  Real endurance fits land near −0.05 … −0.15, far below this floor.
_POWER_LAW_MIN_DECAY = 1e-9

# Durations (seconds) reported as the athlete's estimated potential:
# 5 s neuromuscular / P_max, 60 s anaerobic capacity, 300 s maximal aerobic
# power, 1200 s ≈ threshold.
POTENTIAL_DURATIONS: list[int] = [5, 60, 300, 1200]


def _log_spaced(lo: int, hi: int, count: int) -> list[int]:
    """Return ``count`` roughly log-spaced integer durations from ``lo`` to ``hi``."""
    lo = max(lo, 1)
    if hi <= lo or count < 2:
        return [lo]
    ratio = (hi / lo) ** (1.0 / (count - 1))
    return sorted({int(round(lo * ratio ** i)) for i in range(count)})


# Dense log-spaced grid used to sample a smooth model curve for plotting.
MODEL_CURVE_DURATIONS: list[int] = _log_spaced(5, 28800, 56)


# Grid resolution for the nonlinear parameter of the CP3 / exponential fits.
# This replaced a two-pass search (121 coarse points, then a fine pass over ±1
# coarse spacing), whose effective resolution was span/7200 — matching it takes
# 7201 single-pass points, not the 121 a naive port would use.  The whole grid
# solves in ~2 ms against a full bests table, and these fits run once per
# power-profile request rather than per activity, so the resolution is worth
# more here than the milliseconds are.  ``test_grid_is_fine_enough_to_pin_k``
# fails if it is cut back.
_FIT_GRID_POINTS = 7201


def _ols_grid(xs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ordinary least squares of ``y = slope·x + intercept`` for every row of ``xs``.

    ``xs`` is ``(grid_points, n)`` — one candidate predictor vector per grid
    point — and ``y`` is the ``(n,)`` observation vector shared by all of them.
    Returns ``(slope, intercept, sse)``, each ``(grid_points,)``.  Rows whose
    ``x`` has no spread yield NaN, which the callers' plausibility masks reject
    along with any other unusable grid point.
    """
    n = y.size
    sum_x = xs.sum(axis=1)
    denom = n * (xs * xs).sum(axis=1) - sum_x * sum_x
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * (xs @ y) - sum_x * y.sum()) / denom
    intercept = (y.sum() - slope * sum_x) / n
    residuals = y - (slope[:, None] * xs + intercept[:, None])
    return slope, intercept, (residuals * residuals).sum(axis=1)


def _best_grid_point(sse: np.ndarray, valid: np.ndarray) -> int | None:
    """Index of the lowest-SSE grid point among ``valid``, or None if none are."""
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return None
    return int(candidates[np.argmin(sse[candidates])])


def estimate_cp3(
    bests: dict[int, float],
) -> tuple[float, float, float, float] | None:
    """
    Fit the 3-parameter Critical Power model (Morton) to the power bests.

    Model: ``P(t) = CP + W'/(t − k)`` with ``k < 0``, giving a finite maximal
    instantaneous power ``P_max = CP − W'/k`` at ``t → 0``.  For a fixed ``k``
    the model is linear in ``CP`` and ``W'`` (regress ``P`` on ``1/(t − k)``), so
    ``k`` is found by grid search.

    Uses durations in ``CP3_FIT_DURATIONS``; needs at least 3 points.  Returns
    ``(cp, w_prime, k, pmax)`` or ``None`` if it cannot be fit sensibly.
    """
    durations = [d for d in CP3_FIT_DURATIONS if d in bests]
    if len(durations) < 3:
        return None
    ts = np.array(durations, dtype=float)
    ps = np.array([bests[d] for d in durations], dtype=float)

    ks = np.linspace(-60.0, -0.5, _FIT_GRID_POINTS)
    xs = 1.0 / (ts - ks[:, None])  # t − k > 0 since k < 0 and t > 0
    w_prime, cp, sse = _ols_grid(xs, ps)
    # Refusing every grid point is how this rejects garbage input (power rising
    # with duration, say) instead of reporting a fitted-looking number.
    i = _best_grid_point(sse, (cp > 0) & (w_prime > 0))
    if i is None:
        return None
    k = float(ks[i])
    return float(cp[i]), float(w_prime[i]), k, float(cp[i] - w_prime[i] / k)


def estimate_exponential(
    bests: dict[int, float],
) -> tuple[float, float, float] | None:
    """
    Fit the CP-anchored exponential model to the power bests.

    Model: ``P(t) = CP + (P_max − CP)·e^(−t/τ)`` — power decays from a maximal
    instantaneous ``P_max`` toward the ``CP`` asymptote with time constant
    ``τ``.  For a fixed ``τ`` the model is linear in ``CP`` and the amplitude
    ``A = P_max − CP`` (regress ``P`` on ``e^(−t/τ)``), so ``τ`` is found by grid
    search (over ``ln τ`` for scale invariance).

    Uses durations in ``EXP_FIT_DURATIONS``; needs at least 3 points.  Returns
    ``(cp, pmax, tau)`` or ``None`` if it cannot be fit sensibly.
    """
    durations = [d for d in EXP_FIT_DURATIONS if d in bests]
    if len(durations) < 3:
        return None
    ts = np.array(durations, dtype=float)
    ps = np.array([bests[d] for d in durations], dtype=float)

    # τ from ~5 s to ~1 h, gridded over ln τ for scale invariance.
    log_taus = np.linspace(math.log(5.0), math.log(3600.0), _FIT_GRID_POINTS)
    xs = np.exp(-ts / np.exp(log_taus)[:, None])
    amplitude, cp, sse = _ols_grid(xs, ps)  # slope = P_max − CP, intercept = CP
    i = _best_grid_point(sse, (cp > 0) & (amplitude > 0))
    if i is None:
        return None
    return float(cp[i]), float(cp[i] + amplitude[i]), float(math.exp(log_taus[i]))


def estimate_power_law(bests: dict[int, float]) -> tuple[float, float] | None:
    """
    Fit the power-law (Riegel) model ``P(t) = a·t^b`` (``b < 0``) to the bests.

    Linear in log space: ``ln P = ln a + b·ln t``, fit by OLS.  Uses durations
    in ``POWER_LAW_FIT_DURATIONS``; needs at least 2 points.  Returns
    ``(a, b)`` or ``None`` if it cannot be fit sensibly — including a curve flat
    enough that ``b`` is indistinguishable from zero.
    """
    durations = [d for d in POWER_LAW_FIT_DURATIONS if d in bests and bests[d] > 0]
    if len(durations) < 2:
        return None
    b, ln_a = np.polyfit(
        np.log(np.array(durations, dtype=float)),
        np.log(np.array([bests[d] for d in durations], dtype=float)),
        1,
    )
    if b >= -_POWER_LAW_MIN_DECAY:
        return None
    return float(math.exp(ln_a)), float(b)


def predict_power(model: str, params: tuple[float, ...], t: float) -> float:
    """
    Predict mean power (watts) for an effort of ``t`` seconds using the fitted
    ``params`` of ``model`` (``"cp2"``, ``"cp3"``, ``"exp"`` or ``"power_law"``).
    """
    t = float(t)
    if model == "cp2":
        cp, w_prime = params
        return cp + w_prime / t
    if model == "cp3":
        cp, w_prime, k, _pmax = params
        return cp + w_prime / (t - k)
    if model == "exp":
        cp, pmax, tau = params
        return cp + (pmax - cp) * math.exp(-t / tau)
    if model == "power_law":
        a, b = params
        return a * (t ** b)
    raise ValueError(f"unknown power model: {model!r}")


def sample_power_curve(
    model: str, params: tuple[float, ...], durations: list[int]
) -> list[tuple[int, float]]:
    """Return ``[(duration_s, predicted_power_w), …]`` for the given durations."""
    return [(d, predict_power(model, params, d)) for d in durations]


def model_rmse(
    model: str,
    params: tuple[float, ...],
    bests: dict[int, float],
    fit_durations: list[int],
) -> float | None:
    """
    Root-mean-square error (watts) of ``model`` against the actual bests over
    the durations it was fit on.  Returns ``None`` if no overlapping data.
    """
    residuals = [
        predict_power(model, params, d) - bests[d]
        for d in fit_durations
        if d in bests
    ]
    if not residuals:
        return None
    return (sum(r * r for r in residuals) / len(residuals)) ** 0.5


# Distance best durations in metres
DISTANCE_BEST_DISTANCES: list[int] = [
    1_000, 2_000, 3_000, 5_000, 8_000,
    10_000, 20_000, 30_000, 40_000, 50_000,
    60_000, 70_000, 80_000, 90_000, 100_000,
    110_000, 120_000, 130_000, 140_000, 150_000,
]


# Windows are matched against the distance with a relative tolerance.  A prefix
# sum over hours of samples carries ~1e-13 of accumulated rounding, so a window
# that covers the distance exactly in real arithmetic can land a fraction of a
# nanometre short of it in floating point — and the answer then jumps by a whole
# second.  No speed sensor resolves anything close to that.
_DISTANCE_MATCH_TOLERANCE = 1e-12


def _fastest_window(cum: np.ndarray, distance_m: int) -> int | None:
    """Narrowest window of the prefix sum ``cum`` spanning ``distance_m``."""
    n = cum.size - 1
    target = distance_m * (1.0 - _DISTANCE_MATCH_TOLERANCE)
    if n == 0 or cum[-1] < target:
        return None
    # For every window end, the latest start that still covers the distance —
    # the vectorised form of walking a second pointer forward.  -1 means no such
    # start exists, i.e. this end is unreachable.
    starts = np.searchsorted(cum, cum - target, side="right") - 1
    widths = np.where(starts >= 0, np.arange(cum.size) - starts, n + 1)
    best = int(widths.min())
    return best if best <= n else None


def best_time_for_distance(
    speed_stream: Sequence[float | None], distance_m: int
) -> int | None:
    """
    Return the minimum number of seconds to cover `distance_m` metres in
    `speed_stream` (m/s values at 1-second intervals).

    Returns None if the total distance in the stream is less than distance_m.

    Unlike the power bests, a gap counts as a second at zero speed: the answer is
    a *time*, so the seconds must be real — closing the window over a dropout
    would report a fastest kilometre that was never ridden.
    """
    return _fastest_window(_prefix_sum(streams.filled(speed_stream)), distance_m)


def compute_distance_bests(speed_stream: Sequence[float | None]) -> dict[int, int]:
    """
    Compute best_time_for_distance for every standard distance.
    Only returns entries where the stream covers that distance.
    """
    cum = _prefix_sum(streams.filled(speed_stream))
    return {
        d: t
        for d in DISTANCE_BEST_DISTANCES
        if (t := _fastest_window(cum, d)) is not None
    }


def weighted_power(power_series: Sequence[float | None]) -> float | None:
    """30-second rolling average → raise to 4th power → mean → 4th root.

    Gaps are dropped rather than read as zero watts, so the rolling window
    closes over the recorded samples — the same series this saw before streams
    carried gaps.  Reading a dropout as zero would let a failed sensor lower a
    rider's weighted power, and with it their Load for the day.
    """
    window = 30
    series = streams.present(power_series)
    if series.size < window:
        return None
    cum = _prefix_sum(series)
    rolling = (cum[window:] - cum[:-window]) / window
    return float((rolling**4).mean() ** 0.25)


def compute_torque_stream(
    power: Sequence[float | None], cadence: Sequence[float | None]
) -> list[float | None]:
    """Per-second crank torque (Nm) derived from power (W) and cadence (rpm).

    torque = power · 60 / (2π · cadence).  Returns 0.0 where cadence is 0 or
    negative (coasting / no pedalling).  Returns an empty list if either input
    is empty.

    A *paired* metric, so it reads the grid as it stands: a second where either
    channel has a gap yields a gap.  Both inputs are grid-length now, so the
    ``min`` below is a formality kept for streams stored before issue #76.
    """
    n = min(len(power), len(cadence))
    if n == 0:
        return []
    p = streams.as_array(power)[:n]
    c = streams.as_array(cadence)[:n]
    pedalling = c > 0
    # Mask the divisor rather than the result: dividing by the raw cadence would
    # warn (and produce inf) on the coasting samples before np.where drops them.
    torque = np.where(
        pedalling, p * (60.0 / (2.0 * math.pi)) / np.where(pedalling, c, 1.0), 0.0
    )
    # ``pedalling`` is False at a cadence gap, so the np.where above would have
    # quietly turned it into a confident 0.0 Nm.
    gaps = np.isnan(p) | np.isnan(c)
    return streams.to_json_stream(np.where(gaps, np.nan, torque))


# ---------------------------------------------------------------------------
# Aerobic response metrics (issue #37)
#
# Efficiency factor and variability index are pure ratios of values openkoutsi
# already stores per activity, so they are derived on read rather than
# persisted.  Aerobic decoupling and W' balance need the per-second streams.
# All are O(n); the stream scans are vectorised, except ``w_bal_stream``, whose
# [0, W'] clamp makes it a genuinely sequential nonlinear recurrence.
# ---------------------------------------------------------------------------


def efficiency_factor(
    weighted_power: float | None, avg_hr: float | None
) -> float | None:
    """
    Aerobic efficiency: weighted power (W) per beat of average heart rate.

    Rising over time at a constant training load means the athlete is producing
    more power for the same cardiac cost — aerobic progress the Fitness/Fatigue
    model cannot see.  Returns None unless both inputs are present and positive.
    """
    if not weighted_power or not avg_hr or weighted_power <= 0 or avg_hr <= 0:
        return None
    return weighted_power / avg_hr


def variability_index(
    weighted_power: float | None, avg_power: float | None
) -> float | None:
    """
    Weighted power / average power — how punchy the ride was.

    1.0 is perfectly steady; values above ~1.10 indicate interval or otherwise
    surging riding (the same threshold `categorization.classify_workout` uses).
    Returns None unless both inputs are present and positive.
    """
    if not weighted_power or not avg_power or weighted_power <= 0 or avg_power <= 0:
        return None
    return weighted_power / avg_power


def _half_power(segment: np.ndarray) -> float | None:
    """Representative power for one half of a decoupling split.

    Weighted power is the defensible choice on variable terrain, but it needs
    the 30-second rolling window; shorter halves fall back to the arithmetic
    mean.  Returns None for an empty or zero-power segment, or for one that is
    all gaps.
    """
    recorded = streams.present(segment)
    if recorded.size == 0:
        return None
    value = weighted_power(recorded) if recorded.size >= 30 else None
    if value is None:
        value = float(recorded.mean())
    return value if value > 0 else None


def aerobic_decoupling(
    power: Sequence[float | None], heartrate: Sequence[float | None]
) -> float | None:
    """
    Power:HR decoupling (Pw:HR drift) as a percentage.

    Splits the ride into two equal halves, takes the power-to-heart-rate ratio
    of each, and returns how far the second half drifted from the first:

        (ratio_first - ratio_second) / ratio_first * 100

    A positive number means heart rate climbed relative to power — the classic
    sign of fading aerobic durability.  Streams of unequal length are truncated
    to the shorter one; on an odd number of samples the middle sample is
    dropped so both halves stay the same length.

    Pairs power against the heart rate at the *same index*, so it is meaningful
    only because the streams share a clock (issue #76).  Within each half the
    channels are summarised over the samples they have, so a gap costs a second
    of evidence rather than pairing a wattage against the wrong heartbeat.

    Returns None if either half has no usable power or heart rate.  Raw math with
    no validity checks — see `decoupling_unavailable_reason`.
    """
    watts = streams.as_array(power)
    beats = streams.as_array(heartrate)
    n = min(watts.size, beats.size)
    half = n // 2
    if half == 0:
        return None

    def ratio(lo: int, hi: int) -> float | None:
        p = _half_power(watts[lo:hi])
        hr_slice = beats[lo:hi]
        hr_slice = hr_slice[hr_slice > 0]
        if p is None or hr_slice.size == 0:
            return None
        return p / float(hr_slice.mean())

    first = ratio(0, half)
    second = ratio(n - half, n)
    # A relative floor rather than `first == 0`: the ratio blows up continuously
    # as the first half's power approaches zero, so the function stays safe on
    # its own terms instead of relying on the caller's variability gate to have
    # already excluded the extremes.
    if first is None or second is None:
        return None
    if first < DECOUPLING_MIN_RATIO * max(second, first):
        return None
    return (first - second) / first * 100


# Decoupling over a short ride is noise: the conventional minimum is roughly an
# hour of steady riding.
DECOUPLING_MIN_DURATION_S = 3600

# Categories whose whole point is repeated hard efforts. A power:HR drift number
# over one of these describes the intervals, not the athlete's durability.
DECOUPLING_EXCLUDED_CATEGORIES = frozenset({"vo2max", "anaerobic", "sprint"})

# Above this variability index the ride was too surgy for the two-half split to
# mean anything (same threshold `classify_workout` uses to spot interval work).
DECOUPLING_MAX_VI = 1.10

# Above this relative difference between the two halves' mean power the ride was
# ridden as a ramp or a negative split. Pw:HR assumes steady output, and a rider
# who simply rode the back half harder produces a large drift number that reads
# as a durability verdict when it is really a pacing choice.
DECOUPLING_MAX_HALF_POWER_DELTA = 0.10

# The two streams are paired sample-for-sample, so the question is how much of
# the ride they can speak to *together*. Measured as the seconds carrying both
# channels over the seconds carrying the better-covered one: below this
# fraction, the two are describing different parts of the ride.
#
# This used to compare the two streams' *lengths*, which was the only symptom
# available before the streams shared a clock — and it missed the case that
# motivated issue #76: two channels each dropping a similar number of records at
# different points, ending up near-identical in length while internally
# misaligned. The overlap can now just be counted, and the same threshold covers
# both symptoms, including on pre-#76 streams where a short channel has no gaps
# and simply contributes nothing past where it stops.
DECOUPLING_MIN_PAIRED_COVERAGE = 0.95

# Smallest first-half ratio, relative to the larger of the two halves, that the
# percentage formula can be evaluated at without amplifying noise.
DECOUPLING_MIN_RATIO = 0.05


def _positive_in_both_halves(stream: np.ndarray, n: int) -> bool:
    """Does ``stream`` carry a positive sample in each half of the split?

    The whole-stream check isn't enough: a power meter that dies at halfway
    leaves a stream that is non-empty overall but unusable for a two-half
    comparison.
    """
    half = n // 2
    if half == 0:
        return False
    return bool((stream[:half] > 0).any() and (stream[n - half:n] > 0).any())


def decoupling_unavailable_reason(
    duration_s: int | None,
    power: Sequence[float | None] | None,
    heartrate: Sequence[float | None] | None,
    workout_category: str | None = None,
    vi: float | None = None,
) -> str | None:
    """
    Why a decoupling figure would be misleading for this activity, or None if
    it is worth computing.

    A decoupling number for a hard interval session is worse than none, so the
    caller stores NULL and surfaces the reason.  Reason codes are stable strings
    the API and web app key their explanations off:

    ``no_power``, ``no_hr``, ``stream_mismatch``, ``too_short``,
    ``degenerate_hr``, ``variable_effort``, ``uneven_pacing``.

    Data problems are reported before qualification problems, so the athlete is
    told the thing actually blocking the measurement.
    """
    watts = streams.as_array(power)
    hr = streams.as_array(heartrate)
    n = min(watts.size, hr.size)

    # Content-aware, not just emptiness: a paired-but-silent meter records a
    # full stream of zeros, and calling that a heart-rate problem would send the
    # athlete (and the LLM coach) after the wrong thing entirely.
    #
    # Whole-stream checks come first. The per-half checks below divide by the
    # *shared* length, so a missing heart-rate stream would otherwise make the
    # power half-check fail and misreport a fine power meter as absent.
    if not (watts > 0).any():
        return "no_power"
    if not (hr > 0).any():
        return "no_hr"
    if not _positive_in_both_halves(watts, n):
        return "no_power"
    if not _positive_in_both_halves(hr, n):
        return "no_hr"

    # Seconds where both channels have something to say — the only seconds this
    # metric can actually use.
    paired = streams.paired_count(watts, hr)
    covered = max(streams.present(watts).size, streams.present(hr).size)
    if covered and paired < DECOUPLING_MIN_PAIRED_COVERAGE * covered:
        return "stream_mismatch"

    # Both clocks matter: `duration_s` is elapsed time from the FIT header, while
    # the halves are split by position on the grid. A ride with four hours
    # elapsed but forty minutes recorded clears the elapsed check and then gets
    # split into two halves that are mostly gap, so the paired seconds are
    # counted rather than the width of the grid they are spread across.
    if (duration_s or 0) < DECOUPLING_MIN_DURATION_S:
        return "too_short"
    if paired < DECOUPLING_MIN_DURATION_S:
        return "too_short"

    beats = hr[hr > 0]
    if beats.size == 0 or beats.min() == beats.max():
        return "degenerate_hr"

    if workout_category in DECOUPLING_EXCLUDED_CATEGORIES:
        return "variable_effort"
    if vi is not None and vi > DECOUPLING_MAX_VI:
        return "variable_effort"

    # Variability index catches surging but is blind to a monotonic ramp, which
    # is precisely the shape that produces a large spurious drift number.
    half = n // 2
    first_recorded = streams.present(watts[:half])
    second_recorded = streams.present(watts[n - half:n])
    if first_recorded.size == 0 or second_recorded.size == 0:
        return "no_power"
    first_mean = float(first_recorded.mean())
    second_mean = float(second_recorded.mean())
    reference = max(first_mean, second_mean)
    if reference > 0 and abs(first_mean - second_mean) / reference > DECOUPLING_MAX_HALF_POWER_DELTA:
        return "uneven_pacing"

    return None


# Physiologically plausible bounds for a cycling CP/W' fit. The linear work-time
# model is an ordinary least-squares fit with no constraint on its intercept, so
# a rider whose short bests sit below the line — anyone who only ever rides
# steady — fits a negative or near-zero W' as a matter of course. A W' of a few
# hundred joules is not a small anaerobic reserve, it is a failed fit, and the
# balance curve it produces craters in seconds and means nothing.
CP_PLAUSIBLE_W = (50.0, 600.0)
W_PRIME_PLAUSIBLE_J = (5_000.0, 50_000.0)


def cp_wprime_plausible(cp: float | None, w_prime: float | None) -> bool:
    """Is this CP/W' pair physiologically believable for a cyclist?

    Used to reject a fit outright rather than persist numbers that look like
    measurements. See ``CP_PLAUSIBLE_W`` / ``W_PRIME_PLAUSIBLE_J``.
    """
    if cp is None or w_prime is None:
        return False
    return (
        CP_PLAUSIBLE_W[0] <= cp <= CP_PLAUSIBLE_W[1]
        and W_PRIME_PLAUSIBLE_J[0] <= w_prime <= W_PRIME_PLAUSIBLE_J[1]
    )


def w_bal_stream(
    power: list[float | None], cp: float | None, w_prime: float | None
) -> list[float]:
    """
    Per-second W' balance (joules remaining) from a power stream.

    Uses the differential form: above CP the athlete spends anaerobic capacity
    at (P - CP) joules per second; below CP it is reconstituted in proportion to
    how depleted the tank already is, which gives the exponential recovery of
    the Skiba model without its O(n^2) integral.

        P >  CP:  bal -= (P - CP)
        P <= CP:  bal += (CP - P) * (W' - bal) / W'

    Starts full at W' and is clamped to [0, W'] — a rider who empties the tank
    holds at zero rather than going negative.  Returns an empty list when there
    is no power stream, or when CP/W' are missing or implausible, since a W'
    curve built on a guessed W' would be fiction.

    **Requires a 1 Hz power stream.** The arithmetic is joules per *sample* and
    only equals joules per second at one sample per second, which the parser has
    guaranteed since issue #76. It is not self-checking, though: a stream from
    somewhere other than ``openkoutsi.streams`` (a pre-#76 row, a provider that
    skipped the resampler) can be at another rate, and deciding whether the
    sampling supports the integration is the caller's job — see
    ``services.aerobic_metrics._sampling_supports_integration``.

    Gaps count as seconds at zero watts, i.e. as recovery — the *time* reading of
    a gap rather than the *sample* one, since the integration is over the clock.
    A stream gappy enough for that to distort the curve should have been rejected
    by the caller's coverage check before reaching here.
    """
    if not power or not cp_wprime_plausible(cp, w_prime):
        return []

    balance = w_prime
    out: list[float] = []
    for p in streams.filled(power):
        if p > cp:
            balance -= p - cp
        else:
            balance += (cp - p) * (w_prime - balance) / w_prime
        balance = 0.0 if balance < 0 else (w_prime if balance > w_prime else balance)
        out.append(balance)
    return out


def calculate_load(
    duration_s: int,
    wp: float | None,
    avg_hr: float | None,
    ftp: int | None,
    max_hr: int | None,
) -> tuple[float | None, float | None]:
    """
    Returns (load, intensity).

    Priority: power-based Load if Weighted Power and FTP are available, otherwise
    HR-based TRIMP Load if avg_hr and max_hr are available.
    """
    if wp is not None and ftp:
        intensity = wp / ftp
        load = (duration_s * wp * intensity) / (ftp * 3600) * 100
        return load, intensity

    if avg_hr is not None and max_hr:
        lthr = 0.9 * max_hr
        if lthr == 0:
            return None, None
        intensity = avg_hr / lthr
        load = (duration_s / 3600) * intensity ** 2 * 100
        return load, None

    return None, None
