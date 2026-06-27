"""
Shared training load calculations — peak power, normalized power, TSS, distance bests.
"""

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


def normalized_power(power_series: list[float]) -> float | None:
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


def calculate_tss(
    duration_s: int,
    np: float | None,
    avg_hr: float | None,
    ftp: int | None,
    max_hr: int | None,
) -> tuple[float | None, float | None]:
    """
    Returns (tss, intensity_factor).

    Priority: power-based TSS if NP and FTP are available, otherwise
    HR-based TRIMP TSS if avg_hr and max_hr are available.
    """
    if np is not None and ftp:
        intensity_factor = np / ftp
        tss = (duration_s * np * intensity_factor) / (ftp * 3600) * 100
        return tss, intensity_factor

    if avg_hr is not None and max_hr:
        lthr = 0.9 * max_hr
        if lthr == 0:
            return None, None
        intensity_factor = avg_hr / lthr
        tss = (duration_s / 3600) * intensity_factor ** 2 * 100
        return tss, None

    return None, None
