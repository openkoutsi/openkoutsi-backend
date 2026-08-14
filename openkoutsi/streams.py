"""The stream contract: every channel on one clock, gaps marked explicitly (issue #76).

A **stream** is a list of length ``N``, where ``N`` is the activity's elapsed
seconds and index ``i`` is second ``i`` counted from the first record. Each
element is a float, or ``None`` (``NaN`` once it reaches numpy) meaning *this
channel had no sample at this second*.

That contract did not used to hold. ``fit.summarizeWorkout`` built each channel
by appending one sample per ``record`` frame that happened to carry that field,
never consulting the timestamp, so the channels were index-aligned only while
every channel appeared in every record. A heart-rate strap dropping ``n``
records did not leave a hole in the HR list — it shifted every later HR sample
``n`` positions *earlier* relative to power. The lists stayed dense and merely
ended up different lengths, which is why nothing ever looked broken.

Every metric that pairs two channels at the same index depends on this module
now doing what the old code assumed: aerobic decoupling multiplies power against
the heart rate recorded at the same moment, ``compute_interval_stats`` slices
``data[start_offset_s:end]`` as though the index were a clock, and
``w_bal_stream``'s joules-per-sample arithmetic is only joules-per-second at
exactly 1 Hz.

Two rules the rest of the codebase is written against:

* **A gap is not a zero.** Zero watts is coasting; a gap is the absence of a
  measurement. Nothing here fills one in — no interpolation, no carry-forward,
  not even across a single dropped sample — because a reconstruction that looks
  like data is exactly what issue #76 is about removing.
* **Consumers choose how to read a gap, and the choice is not uniform.**
  Aggregates that don't care about alignment (average power, the power bests,
  time-in-zone) read *present samples only*, which is byte-for-byte what they
  saw before this module existed. Metrics that are about the clock (distance
  bests, W' balance) read the grid with gaps as zero, because a fastest-1 km
  measured across a dropout is a fake PR. :func:`present` and :func:`filled`
  are the two sides of that split.

Streams stored before this existed are dense with no gaps, so every helper here
and every consumer downstream reads them exactly as it always did. There is no
migration and nothing rewrites an activity's streams after ingestion, so those
rides keep the old convention for good — which is why "handles both shapes" is a
permanent requirement of everything downstream rather than a transitional one.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

import numpy as np

# A ride cannot plausibly span more than a day, and the grid is allocated from
# the span between the first and last record — so a device that stamps one
# record with a garbage timestamp would otherwise size a list from it. Clamping
# costs a truncated tail on a file that is already broken; not clamping costs
# an allocation bounded by whatever the sensor happened to write.
MAX_STREAM_SECONDS = 24 * 60 * 60

# The public spelling of a gap. Stored as JSON ``null``; ``NaN`` would not
# survive ``json.dumps`` into a Postgres JSON column.
GAP: float | None = None

# One stream channel as collected during parsing: (second offset, value) pairs,
# in file order, not necessarily unique or sorted.
Samples = list[tuple[int, float]]


def second_offsets(
    timestamps: Sequence[datetime], *, max_seconds: int = MAX_STREAM_SECONDS
) -> tuple[list[int], int]:
    """Second offsets for ``timestamps`` relative to the first, and the grid length.

    Offsets are truncated to whole seconds, so a device recording faster than
    1 Hz lands several records on the same slot (see :func:`resample_1hz` for
    which one wins). Offsets beyond ``max_seconds`` are reported as ``-1`` and
    should be dropped by the caller; the returned length never exceeds the cap.

    Returns ``([], 0)`` for an empty input.
    """
    if not timestamps:
        return [], 0

    first = timestamps[0]
    offsets: list[int] = []
    length = 0
    for ts in timestamps:
        offset = int((ts - first).total_seconds())
        if offset < 0 or offset >= max_seconds:
            # Before the first record (clock stepped backwards mid-file) or past
            # the cap. Either way there is no slot for it.
            offsets.append(-1)
            continue
        offsets.append(offset)
        if offset >= length:
            length = offset + 1
    return offsets, length


def resample_1hz(channels: dict[str, Samples], length: int) -> dict[str, list[float | None]]:
    """Scatter each channel's ``(second, value)`` pairs onto a shared 1 Hz grid.

    Every returned channel has exactly ``length`` elements, so the channels are
    aligned by construction rather than by the coincidence of every record
    carrying every field. Seconds no sample landed on stay ``None``.

    Where several records share a second — a device recording at 2 Hz, or a
    duplicate frame — the last one wins, matching the "most recent reading"
    semantics a sensor has anyway.

    Channels with no samples at all come back as ``[]`` rather than a grid of
    ``None``: a ride with no power meter has *no power stream*, which is a
    different statement from a power stream that is entirely gaps, and callers
    key "did this activity record power" off the empty list.
    """
    out: dict[str, list[float | None]] = {}
    for name, samples in channels.items():
        if not samples:
            out[name] = []
            continue
        grid: list[float | None] = [None] * length
        for offset, value in samples:
            if 0 <= offset < length:
                grid[offset] = value
        out[name] = grid
    return out


def resample_from_time_stream(
    time_offsets: Sequence[float],
    channels: dict[str, Sequence[float]],
    *,
    max_seconds: int = MAX_STREAM_SECONDS,
) -> dict[str, list[float | None]]:
    """Put provider-supplied parallel arrays onto the same grid as a FIT file.

    Strava hands back one array per channel plus a ``time`` array of second
    offsets, all indexed together. That is internally consistent but it is not
    the contract the rest of openkoutsi is written against: the arrays are only
    1 Hz when the ride was recorded that way, so index ``i`` is not second ``i``
    and the same pairing bugs are available. Running them through here makes the
    provider path and the FIT path produce the same shape.

    A channel shorter than ``time_offsets`` contributes only the samples it has;
    a channel that is empty stays empty (see :func:`resample_1hz`).
    """
    if not len(time_offsets):
        return {name: [float(v) for v in values] for name, values in channels.items()}

    base = float(time_offsets[0])
    offsets = [int(float(t) - base) for t in time_offsets]
    length = 0
    for offset in offsets:
        if 0 <= offset < max_seconds and offset >= length:
            length = offset + 1

    collected: dict[str, Samples] = {}
    for name, values in channels.items():
        collected[name] = [
            (offsets[i], float(v))
            for i, v in enumerate(values)
            if i < len(offsets) and offsets[i] >= 0
        ]
    return resample_1hz(collected, length)


# ── numpy / JSON bridge ──────────────────────────────────────────────────────
#
# Streams cross three representations: JSON ``null`` in the database and on the
# wire, ``None`` in the Python lists, and ``NaN`` inside numpy. These four
# helpers are the only places that conversion should happen.


def as_array(stream: Iterable[float | None] | None) -> np.ndarray:
    """A stream as a float array, gaps as ``NaN``.

    ``np.asarray([1.0, None], dtype=float)`` already yields ``[1.0, nan]``, so
    this is mostly a named, None-tolerant entry point — but having one means
    callers never have to remember whether their stream came from a freshly
    parsed file or from a pre-#76 row.
    """
    if stream is None:
        return np.empty(0, dtype=float)
    return np.asarray(list(stream), dtype=float)


def present(values: np.ndarray | Iterable[float | None] | None) -> np.ndarray:
    """Just the recorded samples, gaps dropped — the *unaligned* reading.

    Use for aggregates where a gap should be ignored rather than counted: mean
    power, the power bests, time-in-zone. This is what those metrics saw before
    #76, when a dropout silently shortened the list, so switching them to it is
    a no-op on their numbers.
    """
    arr = values if isinstance(values, np.ndarray) else as_array(values)
    return arr[~np.isnan(arr)]


def filled(
    values: np.ndarray | Iterable[float | None] | None, fill: float = 0.0
) -> np.ndarray:
    """The full grid with gaps replaced by ``fill`` — the *time-aligned* reading.

    Use where position on the clock is the point and dropping a gap would
    silently shorten time: the distance bests (a fastest 1 km must not be
    stitched across a dropout) and the W' balance integration (a second with no
    power reading is a second the rider was not measurably above CP).
    """
    arr = values if isinstance(values, np.ndarray) else as_array(values)
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)


def present_ratio(stream: Iterable[float | None] | None) -> float:
    """Fraction of the grid that carries a sample, 0.0 for an empty stream.

    The honest version of "is this stream dense enough to integrate as one
    sample per second": before #76 that had to be guessed by comparing a dense
    list's length against the activity's elapsed seconds, which conflated a
    low-rate recording with a ride full of café stops.
    """
    arr = as_array(stream)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(~np.isnan(arr))) / arr.size


def paired_count(
    a: np.ndarray | Iterable[float | None] | None,
    b: np.ndarray | Iterable[float | None] | None,
) -> int:
    """Number of seconds where **both** channels carry a sample.

    What a pairing metric needs to know before it multiplies one channel against
    the other: not how long the two streams are, but how much of the ride they
    can actually speak to together. Compared over the overlapping prefix, so a
    channel that stops halfway contributes only the half it recorded.
    """
    arr_a = a if isinstance(a, np.ndarray) else as_array(a)
    arr_b = b if isinstance(b, np.ndarray) else as_array(b)
    n = min(arr_a.size, arr_b.size)
    if n == 0:
        return 0
    return int(np.count_nonzero(~np.isnan(arr_a[:n]) & ~np.isnan(arr_b[:n])))


def to_json_stream(values: Iterable[float | None] | np.ndarray | None) -> list[float | None]:
    """Prepare a stream for storage: ``NaN``/``inf`` become ``None``.

    ``json.dumps(float("nan"))`` emits a bare ``NaN`` token, which is not valid
    JSON — SQLite accepts it and Postgres rejects it, so a NaN reaching the ORM
    is a bug that only shows up in production. Everything persisted to
    ``ActivityStream.data`` goes through here.
    """
    if values is None:
        return []
    return [
        None if v is None or not np.isfinite(v) else float(v)
        for v in values
    ]
