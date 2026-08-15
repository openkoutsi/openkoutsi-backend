"""Unpacking what an athlete actually has on disk (issue #36).

A Strava bulk export is a zip of ``.fit.gz``, ``.tcx.gz`` and ``.gpx.gz``, which
is three layers away from the single uncompressed ``.fit`` the upload endpoint
accepts. This module is the layer that closes that gap: hand it whatever was
uploaded and it yields the individual activity files, expanded onto disk.

Everything here treats the archive as hostile, because it is a file the user
downloaded from somewhere else:

* **Nothing is extracted to a name the archive chose.** Every member is written
  to a generated filename in the job's own directory, so a ``../../etc`` entry
  has nowhere to go. Entry names are *also* validated and rejected, because a
  malicious name is worth reporting rather than merely defusing.
* **Every size is a budget, not a promise.** A zip's stated uncompressed size is
  a number the archive author wrote, so it is used only as a cheap pre-check;
  the real limit is enforced while reading, against a running total shared by
  every member and every nesting level. That is what makes a zip bomb a failed
  import instead of a full disk.
* **Nesting is bounded.** Archives inside archives are walked, to a depth, so a
  zip quine terminates.

The 50 MB request cap the single-file upload relies on is *compressed* size and
does not bound any of this on its own.
"""
from __future__ import annotations

import gzip
import logging
import uuid
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

from openkoutsi.activity_formats import detect_format, strip_compression_suffix

log = logging.getLogger(__name__)

#: Most files one import job may contain, counting every archive member at every
#: level. A decade of daily training is ~4 000 activities; a Strava export of one
#: carries about that many files.
MAX_ENTRIES = 20_000
#: Largest single expanded file. Matches the single-upload cap: a FIT bigger
#: than this is not a training file.
MAX_MEMBER_BYTES = 50 * 1024 * 1024
#: Total expanded bytes across the whole job. A 500 MB upload of ordinary
#: activity files expands to a few times that; anything approaching this is
#: either a bomb or something that should be split into two imports.
MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
#: How many archives deep to walk. Strava's export is one level; two is room for
#: someone zipping their download.
MAX_DEPTH = 3

_ZIP_MAGIC = b"PK\x03\x04"
_GZIP_MAGIC = b"\x1f\x8b"
_CHUNK = 256 * 1024

# Metadata archivers add that is never activity data.
_IGNORED_PREFIXES = ("__MACOSX/", "._", ".DS_Store")


class ArchiveTooLarge(Exception):
    """The archive exceeds a whole-job budget, so the job cannot proceed.

    Distinct from a member that is individually too big, which is one file's
    failure and leaves the rest of the import running.
    """


@dataclass
class ExpandedFile:
    """One activity file recovered from the upload.

    ``path`` is ``None`` when the member could not be expanded or is not an
    activity file at all; ``error`` then says why, and the job records it as a
    failure against ``name`` — the name the archive used, so the athlete can
    find the file in their export.
    """

    name: str
    path: Path | None = None
    format: str | None = None
    error: str | None = None


class Budget:
    """Whole-job limits, shared across every member and nesting level."""

    def __init__(
        self, *, entries: int = MAX_ENTRIES, total_bytes: int = MAX_TOTAL_BYTES
    ) -> None:
        self.entries_left = entries
        self.bytes_left = total_bytes
        self.entries_seen = 0

    def take_entry(self) -> None:
        if self.entries_left <= 0:
            raise ArchiveTooLarge(
                f"Import contains more than {MAX_ENTRIES} files. "
                "Split the archive and import it in parts."
            )
        self.entries_left -= 1
        self.entries_seen += 1

    def take_bytes(self, count: int) -> None:
        self.bytes_left -= count
        if self.bytes_left < 0:
            raise ArchiveTooLarge(
                f"Import expands to more than {MAX_TOTAL_BYTES // (1024 ** 3)} GB. "
                "Split the archive and import it in parts."
            )


def is_safe_entry_name(name: str) -> bool:
    """Would extracting under this name escape the directory it belongs in?

    Nothing is ever extracted under an archive's own name — see the module
    docstring — so this is the check that lets a hostile entry be *reported*
    rather than silently renamed.
    """
    if not name or name.startswith("/") or name.startswith("\\"):
        return False
    normalised = name.replace("\\", "/")
    if ":" in normalised.split("/")[0]:  # C:\... from a Windows-built archive
        return False
    return not any(part == ".." for part in normalised.split("/"))


def _is_ignored(name: str) -> bool:
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    return any(
        name.startswith(prefix) or basename.startswith(prefix)
        for prefix in _IGNORED_PREFIXES
    )


def _staged_path(work_dir: Path) -> Path:
    return work_dir / f"{uuid.uuid4()}.bin"


def _write_capped(reader, destination: Path, budget: Budget, name: str) -> None:
    """Stream ``reader`` to ``destination``, enforcing both size limits.

    Raises :class:`ValueError` when this member alone is too large — one file's
    problem — and lets :class:`ArchiveTooLarge` propagate when the job's total
    budget is gone.
    """
    written = 0
    with destination.open("wb") as out:
        while True:
            chunk = reader.read(_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_MEMBER_BYTES:
                raise ValueError(
                    f"File is larger than {MAX_MEMBER_BYTES // (1024 * 1024)} MB"
                )
            budget.take_bytes(len(chunk))
            out.write(chunk)


def _classify(path: Path, name: str) -> ExpandedFile:
    """Decide what an expanded file is, by its bytes and then by its name."""
    try:
        with path.open("rb") as handle:
            head = handle.read(4096)
    except OSError as exc:
        return ExpandedFile(name=name, path=None, error=f"Could not read: {exc}")

    fmt = detect_format(head, name)
    if fmt is None:
        return ExpandedFile(
            name=name, path=None, error="Not a FIT, GPX or TCX activity file"
        )
    return ExpandedFile(name=name, path=path, format=fmt)


def _expand_zip(
    source: Path, name: str, work_dir: Path, budget: Budget, depth: int
) -> list[ExpandedFile]:
    out: list[ExpandedFile] = []
    try:
        archive = zipfile.ZipFile(source)
    except (zipfile.BadZipFile, OSError) as exc:
        return [ExpandedFile(name=name, path=None, error=f"Could not open archive: {exc}")]

    with archive:
        for info in archive.infolist():
            if info.is_dir() or _is_ignored(info.filename):
                continue

            entry_name = info.filename
            if not is_safe_entry_name(entry_name):
                out.append(
                    ExpandedFile(
                        name=entry_name,
                        path=None,
                        error="Archive entry name is unsafe and was not extracted",
                    )
                )
                continue

            budget.take_entry()

            # The archive's own claim about the expanded size. Cheap to check
            # and worth checking, but not trusted — `_write_capped` enforces the
            # limit against what actually comes out.
            if info.file_size > MAX_MEMBER_BYTES:
                out.append(
                    ExpandedFile(
                        name=entry_name,
                        path=None,
                        error=f"File is larger than {MAX_MEMBER_BYTES // (1024 * 1024)} MB",
                    )
                )
                continue

            destination = _staged_path(work_dir)
            try:
                with archive.open(info) as reader:
                    _write_capped(reader, destination, budget, entry_name)
            except ArchiveTooLarge:
                destination.unlink(missing_ok=True)
                raise
            except (ValueError, zipfile.BadZipFile, zlib.error, OSError, EOFError) as exc:
                destination.unlink(missing_ok=True)
                out.append(ExpandedFile(name=entry_name, path=None, error=str(exc)))
                continue

            out.extend(_walk(destination, entry_name, work_dir, budget, depth + 1))
    return out


def _expand_gzip(
    source: Path, name: str, work_dir: Path, budget: Budget, depth: int
) -> list[ExpandedFile]:
    # `ride.fit.gz` is reported as `ride.fit`: the athlete is looking for the
    # activity, and the compression was the exporter's choice, not theirs.
    inner_name = strip_compression_suffix(name)
    destination = _staged_path(work_dir)
    try:
        with gzip.open(source, "rb") as reader:
            _write_capped(reader, destination, budget, inner_name)
    except ArchiveTooLarge:
        destination.unlink(missing_ok=True)
        raise
    except (ValueError, OSError, EOFError, zlib.error) as exc:
        destination.unlink(missing_ok=True)
        return [ExpandedFile(name=name, path=None, error=f"Could not decompress: {exc}")]

    return _walk(destination, inner_name, work_dir, budget, depth + 1)


def _walk(
    source: Path, name: str, work_dir: Path, budget: Budget, depth: int
) -> list[ExpandedFile]:
    """Expand one file, recursing while it turns out to be another container."""
    if depth > MAX_DEPTH:
        return [ExpandedFile(name=name, path=None, error="Archive nesting is too deep")]

    try:
        with source.open("rb") as handle:
            head = handle.read(8)
    except OSError as exc:
        return [ExpandedFile(name=name, path=None, error=f"Could not read: {exc}")]

    if head[: len(_ZIP_MAGIC)] == _ZIP_MAGIC:
        return _expand_zip(source, name, work_dir, budget, depth)
    if head[: len(_GZIP_MAGIC)] == _GZIP_MAGIC:
        return _expand_gzip(source, name, work_dir, budget, depth)
    return [_classify(source, name)]


def expand(
    source: Path, name: str, work_dir: Path, budget: Budget | None = None
) -> list[ExpandedFile]:
    """Every activity file inside ``source``, expanded into ``work_dir``.

    A plain activity file yields itself; a gzip yields its contents; a zip yields
    the result of doing this to each of its entries. Anything that is none of
    those comes back as one :class:`ExpandedFile` carrying the reason, so an
    import can report a stray PDF in the archive rather than failing.
    """
    budget = budget if budget is not None else Budget()
    work_dir.mkdir(parents=True, exist_ok=True)
    budget.take_entry()  # the uploaded part itself
    return _walk(source, name, work_dir, budget, 0)


def expand_all(uploads: list[tuple[Path, str]], work_dir: Path) -> list[ExpandedFile]:
    """Expand every uploaded part into one flat list of activity files.

    The budget is shared across the parts: fifty archives in one request are one
    import job's worth of limits, not fifty.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    out: list[ExpandedFile] = []
    for path, name in uploads:
        out.extend(expand(path, name, work_dir, budget))
    return out
