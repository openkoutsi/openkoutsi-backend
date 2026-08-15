"""One place that knows which activity file formats exist and how to read them.

openkoutsi ingests three: FIT (what a head unit writes and what the platform was
built around), and TCX and GPX (what a Strava bulk export contains, alongside
FIT). All three are parsed into the same :class:`openkoutsi.workout.Profile`, so
everything downstream — Load, weighted power, zone snapshots, power bests,
torque, interval extraction — is written once and works on all of them.

Each format module exposes the same three functions, which is what makes the
dispatch below a lookup rather than a branch:

``summarizeWorkout(fileish) -> Profile``
    The whole activity, on the shared 1 Hz clock (:mod:`openkoutsi.streams`).
``getStartTime(fileish) -> datetime | None``
    Cheap enough to run before deciding whether the activity is a duplicate.
``extractIntervals(fileish) -> list[dict]``
    Device-recorded laps, or ``[]`` when the format has no such concept and the
    caller should auto-split.

What the formats do *not* share is how much they carry. A FIT file states its
own distance, ascent and timer time and usually carries power; a GPX states
coordinates and often nothing else. :func:`format_capabilities` is the honest
summary of that, for a UI that would otherwise present "no power data" as a
failed import.
"""
from __future__ import annotations

from types import ModuleType

#: Formats in descending order of how much of a ride they preserve. This is the
#: preference order for in-batch deduplication: a Strava export can contain the
#: same ride as FIT, TCX *and* GPX, and the FIT is the one to keep.
FORMATS: tuple[str, ...] = ("fit", "tcx", "gpx")

_FIT_MAGIC = b".FIT"

_EXTENSIONS = {
    ".fit": "fit",
    ".tcx": "tcx",
    ".gpx": "gpx",
}

_ROOT_TAGS = {
    "gpx": "gpx",
    "trainingcenterdatabase": "tcx",
}


class ActivityParseError(ValueError):
    """A file could not be read as an activity, with a reason a user can act on.

    Raised by the format parsers rather than returned, because every caller
    either imports the file or records why it could not: a bulk import writes
    ``str(exc)`` straight into the per-file result list, which is the difference
    between "53 files failed" and a list an athlete can do something about.
    """


def detect_format(data: bytes, filename: str | None = None) -> str | None:
    """Which of :data:`FORMATS` this file is, or ``None`` if it is none of them.

    Content first, filename second. Inside a Strava export the names are
    reliable, but a browser that renamed an upload, a zip entry called
    ``activity.xml``, or a ``.gpx`` that is really a TCX are all things that
    happen, and the bytes are not opinions.
    """
    if len(data) >= 12 and data[8:12] == _FIT_MAGIC:
        return "fit"

    from .xmlsafe import root_tag

    root = root_tag(data)
    if root is not None:
        by_root = _ROOT_TAGS.get(root.lower())
        if by_root is not None:
            return by_root

    if filename:
        lowered = filename.lower()
        for suffix, fmt in _EXTENSIONS.items():
            if lowered.endswith(suffix):
                return fmt
    return None


def strip_compression_suffix(filename: str) -> str:
    """``ride.fit.gz`` → ``ride.fit``. Leaves anything else alone."""
    return filename[:-3] if filename.lower().endswith(".gz") else filename


def parser_for(fmt: str) -> ModuleType:
    """The module implementing ``fmt``.

    Imported here rather than at module scope so that importing this registry
    does not drag in ``fitdecode`` for a caller that only wanted to know whether
    ``"gpx"`` is a format.
    """
    if fmt == "fit":
        from . import fit

        return fit
    if fmt == "gpx":
        from . import gpx

        return gpx
    if fmt == "tcx":
        from . import tcx

        return tcx
    raise ActivityParseError(f"Unsupported activity file format: {fmt}")


def format_priority(fmt: str) -> int:
    """Position in :data:`FORMATS`; lower keeps its data when two files collide."""
    try:
        return FORMATS.index(fmt)
    except ValueError:
        return len(FORMATS)


def format_capabilities(fmt: str) -> dict[str, bool]:
    """What a format can carry — not what a given file actually did.

    ``power`` is the one that matters to a caller: a GPX activity with no
    weighted power, no power bests and no power zone times is a complete import
    of a file that never had them, and the UI should say so rather than showing
    an empty chart or an error.
    """
    return {
        "power": fmt in ("fit", "tcx"),
        "laps": fmt in ("fit", "tcx"),
        # Only FIT states its own totals; the others are derived from the track.
        "stated_totals": fmt == "fit",
    }
