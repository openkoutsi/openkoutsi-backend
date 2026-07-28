"""
Shared training load calculations — peak power, weighted power, Load, distance bests.
"""

import math

POWER_BEST_DURATIONS: list[int] = [
    1, 3, 5, 10, 15, 30, 45, 60, 120, 180, 300, 480, 600,
    900, 1200, 1800, 2700, 3600, 7200, 10800, 14400,
    18000, 21600, 25200, 28800,
]


def peak_average_power(stream: list[float], duration_s: int) -> float | None:
    """
    Return the highest mean wattage over any contiguous `duration_s`-second
    window in `stream`.  Returns None if the stream is shorter than the window.
    Uses a sliding-window approach so it is O(n) per duration.
    """
    n = len(stream)
    if n < duration_s:
        return None
    window_sum = sum(stream[:duration_s])
    best = window_sum
    for i in range(duration_s, n):
        window_sum += stream[i] - stream[i - duration_s]
        if window_sum > best:
            best = window_sum
    return best / duration_s


def compute_power_bests(stream: list[float]) -> dict[int, float]:
    """
    Compute peak_average_power for every standard duration in POWER_BEST_DURATIONS.
    Only returns entries where the stream is long enough to cover the duration.
    """
    return {
        d: v
        for d in POWER_BEST_DURATIONS
        if (v := peak_average_power(stream, d)) is not None
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
    points = [
        (float(d), bests[d] * d)  # (t, work)
        for d in CP_FIT_DURATIONS
        if d in bests
    ]
    if len(points) < 2:
        return None, None

    n = len(points)
    sum_t = sum(t for t, _ in points)
    sum_w = sum(w for _, w in points)
    sum_tt = sum(t * t for t, _ in points)
    sum_tw = sum(t * w for t, w in points)

    denom = n * sum_tt - sum_t * sum_t
    if denom == 0:
        return None, None

    cp = (n * sum_tw - sum_t * sum_w) / denom
    w_prime = (sum_w - cp * sum_t) / n

    if cp <= 0:
        return None, None
    return cp, w_prime


# ---------------------------------------------------------------------------
# Advanced power–duration models
#
# Each model predicts mean power P(t) for an effort of ``t`` seconds and is fit
# to the athlete's rank-1 best power per duration (same input as the CP fit
# above).  There is no numpy/scipy available, so the two nonlinear models
# (3-parameter CP and exponential) are fit with a 1-D grid search over the
# single nonlinear parameter, solving the remaining parameters in closed form
# by ordinary least squares at each grid point.
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


def _linear_ols(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """
    Ordinary least squares fit of ``y = slope·x + intercept``.

    Returns ``(slope, intercept, sse)`` where ``sse`` is the residual sum of
    squares, or ``None`` if there are fewer than 2 points or ``x`` has no spread.
    """
    n = len(xs)
    if n < 2:
        return None
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return None
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    sse = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, sse


def _grid_search_1d(lo: float, hi: float, f, steps: int = 120):
    """
    Two-pass linear grid search minimising ``f(x)[0]`` over ``x ∈ [lo, hi]``.

    ``f`` returns a tuple whose first element is the objective (e.g. SSE), or
    ``None`` for an invalid ``x``.  A coarse scan is followed by a finer scan
    around the best point.  Returns ``(x, f(x))`` or ``None`` if ``f`` was never
    valid.
    """
    def scan(a: float, b: float):
        best = None
        for i in range(steps + 1):
            x = a + (b - a) * i / steps
            res = f(x)
            if res is not None and (best is None or res[0] < best[1][0]):
                best = (x, res)
        return best

    coarse = scan(lo, hi)
    if coarse is None:
        return None
    span = (hi - lo) / steps
    fine = scan(max(lo, coarse[0] - span), min(hi, coarse[0] + span))
    if fine is None:
        return coarse
    return fine if fine[1][0] <= coarse[1][0] else coarse


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
    points = [(float(d), bests[d]) for d in CP3_FIT_DURATIONS if d in bests]
    if len(points) < 3:
        return None
    ts = [t for t, _ in points]
    ps = [p for _, p in points]

    def objective(k: float):
        xs = [1.0 / (t - k) for t in ts]  # t − k > 0 since k < 0 and t > 0
        fit = _linear_ols(xs, ps)
        if fit is None:
            return None
        w_prime, cp, sse = fit
        if cp <= 0 or w_prime <= 0:
            return None
        return sse, cp, w_prime

    result = _grid_search_1d(-60.0, -0.5, objective)
    if result is None:
        return None
    k, (_, cp, w_prime) = result
    pmax = cp - w_prime / k
    return cp, w_prime, k, pmax


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
    points = [(float(d), bests[d]) for d in EXP_FIT_DURATIONS if d in bests]
    if len(points) < 3:
        return None
    ts = [t for t, _ in points]
    ps = [p for _, p in points]

    def objective(log_tau: float):
        tau = math.exp(log_tau)
        xs = [math.exp(-t / tau) for t in ts]
        fit = _linear_ols(xs, ps)
        if fit is None:
            return None
        amplitude, cp, sse = fit  # slope = P_max − CP, intercept = CP
        if cp <= 0 or amplitude <= 0:
            return None
        return sse, cp, cp + amplitude

    # τ from ~5 s to ~1 h.
    result = _grid_search_1d(math.log(5.0), math.log(3600.0), objective)
    if result is None:
        return None
    log_tau, (_, cp, pmax) = result
    return cp, pmax, math.exp(log_tau)


def estimate_power_law(bests: dict[int, float]) -> tuple[float, float] | None:
    """
    Fit the power-law (Riegel) model ``P(t) = a·t^b`` (``b < 0``) to the bests.

    Linear in log space: ``ln P = ln a + b·ln t``, fit by OLS.  Uses durations
    in ``POWER_LAW_FIT_DURATIONS``; needs at least 2 points.  Returns
    ``(a, b)`` or ``None`` if it cannot be fit sensibly.
    """
    points = [
        (float(d), bests[d])
        for d in POWER_LAW_FIT_DURATIONS
        if d in bests and bests[d] > 0
    ]
    if len(points) < 2:
        return None
    xs = [math.log(t) for t, _ in points]
    ys = [math.log(p) for _, p in points]
    fit = _linear_ols(xs, ys)
    if fit is None:
        return None
    b, ln_a, _ = fit
    if b >= 0:
        return None
    return math.exp(ln_a), b


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


def best_time_for_distance(speed_stream: list[float], distance_m: int) -> int | None:
    """
    Return the minimum number of seconds to cover `distance_m` metres in
    `speed_stream` (m/s values at 1-second intervals).

    Uses a two-pointer sliding-window approach: O(n).
    Returns None if the total distance in the stream is less than distance_m.
    """
    n = len(speed_stream)
    if n == 0:
        return None

    cum = [0.0] * (n + 1)
    for i, v in enumerate(speed_stream):
        cum[i + 1] = cum[i] + v

    if cum[n] < distance_m:
        return None

    best = n + 1
    j = 0
    for i in range(1, n + 1):
        while cum[i] - cum[j] >= distance_m:
            best = min(best, i - j)
            j += 1

    return best if best <= n else None


def compute_distance_bests(speed_stream: list[float]) -> dict[int, int]:
    """
    Compute best_time_for_distance for every standard distance.
    Only returns entries where the stream covers that distance.
    """
    return {
        d: t
        for d in DISTANCE_BEST_DISTANCES
        if (t := best_time_for_distance(speed_stream, d)) is not None
    }


def weighted_power(power_series: list[float]) -> float | None:
    """30-second rolling average → raise to 4th power → mean → 4th root."""
    if len(power_series) < 30:
        return None
    window = 30
    rolling = [
        sum(power_series[i - window + 1 : i + 1]) / window
        for i in range(window - 1, len(power_series))
    ]
    if not rolling:
        return None
    return (sum(v**4 for v in rolling) / len(rolling)) ** 0.25


def compute_torque_stream(power: list[float], cadence: list[float]) -> list[float]:
    """Per-second crank torque (Nm) derived from power (W) and cadence (rpm).

    torque = power · 60 / (2π · cadence).  Returns 0.0 where cadence is 0 or
    negative (coasting / no pedalling).  Returns an empty list if either input
    is empty; the result length is the shorter of the two inputs (FIT streams
    can differ in length).
    """
    if not power or not cadence:
        return []
    n = min(len(power), len(cadence))
    k = 60.0 / (2.0 * math.pi)
    out = [0.0] * n
    for i in range(n):
        c = cadence[i]
        out[i] = (power[i] * k / c) if c and c > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Aerobic response metrics (issue #37)
#
# Efficiency factor and variability index are pure ratios of values openkoutsi
# already stores per activity, so they are derived on read rather than
# persisted.  Aerobic decoupling and W' balance need the per-second streams.
# All are plain-Python and O(n) — no numpy/scipy in this module.
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


def _half_power(segment: list[float]) -> float | None:
    """Representative power for one half of a decoupling split.

    Weighted power is the defensible choice on variable terrain, but it needs
    the 30-second rolling window; shorter halves fall back to the arithmetic
    mean.  Returns None for an empty or zero-power segment.
    """
    if not segment:
        return None
    value = weighted_power(segment) if len(segment) >= 30 else None
    if value is None:
        value = sum(segment) / len(segment)
    return value if value > 0 else None


def aerobic_decoupling(
    power: list[float], heartrate: list[float]
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

    Returns None if either half has no usable power or heart rate.  This is raw
    math with no validity checks — see `decoupling_unavailable_reason` for
    whether the answer is meaningful at all.
    """
    n = min(len(power), len(heartrate))
    half = n // 2
    if half == 0:
        return None

    def ratio(lo: int, hi: int) -> float | None:
        p = _half_power(power[lo:hi])
        hr_slice = [v for v in heartrate[lo:hi] if v > 0]
        if p is None or not hr_slice:
            return None
        return p / (sum(hr_slice) / len(hr_slice))

    first = ratio(0, half)
    second = ratio(n - half, n)
    if first is None or second is None or first == 0:
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


def decoupling_unavailable_reason(
    duration_s: int | None,
    power: list[float] | None,
    heartrate: list[float] | None,
    workout_category: str | None = None,
    vi: float | None = None,
) -> str | None:
    """
    Why a decoupling figure would be misleading for this activity, or None if
    it is worth computing.

    Presenting a decoupling number for a hard interval session is worse than
    showing nothing, so the caller stores NULL and surfaces the reason instead
    of a figure the athlete would over-read.  Reason codes are stable strings
    the API and web app key their explanations off:

    ``too_short``, ``no_power``, ``no_hr``, ``degenerate_hr``, ``variable_effort``.
    """
    if not power:
        return "no_power"
    if not heartrate:
        return "no_hr"
    if (duration_s or 0) < DECOUPLING_MIN_DURATION_S:
        return "too_short"

    beats = [v for v in heartrate if v > 0]
    if not beats or min(beats) == max(beats):
        return "degenerate_hr"

    if workout_category in DECOUPLING_EXCLUDED_CATEGORIES:
        return "variable_effort"
    if vi is not None and vi > DECOUPLING_MAX_VI:
        return "variable_effort"

    return None


def w_bal_stream(
    power: list[float], cp: float | None, w_prime: float | None
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
    is no power stream or CP/W' could not be established, since a W' curve built
    on a guessed W' would be fiction.
    """
    if not power or not cp or not w_prime or cp <= 0 or w_prime <= 0:
        return []

    balance = w_prime
    out: list[float] = []
    for p in power:
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
