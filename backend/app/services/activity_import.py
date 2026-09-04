"""Running a bulk import of activity files (issue #36).

The single-file upload is an interaction: pick a ride, wait a moment, see it.
Bulk import is a *job* — a Strava export is thousands of files and tens of minutes
of parsing, which no browser should hold a connection open for. The endpoint
stages the upload, creates an ``ImportJob`` row, and this module works through it
in the background while the client polls.

Three things here are not just "the upload path in a loop":

**One file's failure is not the job's.** Every file gets a result row — imported,
skipped as a duplicate, or failed with a reason — and the loop continues.

**Duplicates are expected.** A Strava export can hold one ride as FIT *and* TCX
*and* GPX, so deduplication happens within the batch before anything is written,
keeping the richest copy. Re-importing after a partial failure skips what is
already present and says so.

**The expensive work happens once, at the end.** ``recalculate_from`` over three
years of history run nine hundred times is quadratic and pointless, so metrics
and plan adherence are recomputed once every file has landed and achievements are
marked for the next reconcile.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.core.file_encryption import encrypt_file
from backend.app.db.user_session import get_user_session_factory
from backend.app.models.user_orm import (
    Activity,
    ActivitySource,
    Athlete,
    ImportJob,
)
from backend.app.services.achievements import mark_achievements_dirty
from backend.app.services.activity_archive import (
    ArchiveError,
    ArchiveTooLarge,
    ExpandedFile,
    expand_all,
)
from backend.app.services.activity_workout_matcher import find_and_link_workout
from backend.app.services.fit_processor import (
    parse_activity_file,
    process_activity_file,
    read_activity_start_time,
)
from backend.app.services.metrics_engine import recalculate_from
from backend.app.services.plan_adherence import catch_up_adherence
from backend.app.services.provider_sync import activity_create_guard
from openkoutsi.activity_formats import ActivityParseError, format_priority

log = logging.getLogger(__name__)

#: Same ±5 minutes the single upload and the provider syncs use to decide two
#: recordings are one ride. Imported here rather than restated so the three
#: paths cannot drift apart.
DUPLICATE_WINDOW = timedelta(minutes=5)

#: How often the job row is updated while it runs. Committing after every file
#: would make a 900-file import 900 extra write transactions on a database with
#: one write lock; committing only at the end would leave the progress bar
#: frozen. Every 25 files is fine for both.
_PROGRESS_EVERY = 25

#: How long a `running` job may go without touching its row before another
#: import may start anyway.
#:
#: The endpoint refuses a second import while one is in flight, but a job whose
#: process died cannot clear its own status — without this that athlete could
#: never import again. A healthy job commits at least every
#: :data:`_PROGRESS_EVERY` files, and the longest gap is the single end-of-job
#: recalculation, so an hour is far outside anything a live job does. Same shape
#: as `stranded_runs.pending_timed_out`: a crash-recovery bound, not a timeout.
STALE_JOB_AFTER = timedelta(hours=1)


def is_in_flight(job: ImportJob, now: datetime | None = None) -> bool:
    """Is this job actually still running, as opposed to merely still marked so?"""
    if job.status not in ("pending", "running"):
        return False
    touched = job.updated_at or job.created_at
    if touched is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - _normalise(touched)) <= STALE_JOB_AFTER


# Outcome codes in `ImportJob.results`. Stable strings — the web client maps
# them to translated text.
OUTCOME_IMPORTED = "imported"
OUTCOME_DUPLICATE = "skipped_duplicate"
OUTCOME_FAILED = "failed"


def _result(
    filename: str,
    outcome: str,
    *,
    reason: str | None = None,
    activity_id: str | None = None,
    fmt: str | None = None,
) -> dict:
    return {
        "filename": filename,
        "outcome": outcome,
        "reason": reason,
        "activity_id": activity_id,
        "format": fmt,
    }


class _Candidate:
    """An expanded file that parsed far enough to know when it happened."""

    __slots__ = ("file", "start_time", "duplicate_of")

    def __init__(self, file: ExpandedFile, start_time: datetime | None):
        self.file = file
        self.start_time = start_time
        # Set during in-batch deduplication to the filename that won, so the
        # skipped file's result can say what it lost to.
        self.duplicate_of: str | None = None


def _normalise(moment: datetime) -> datetime:
    """Naive-or-aware timestamps compared on one footing, as UTC."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _start_times(candidates: list[ExpandedFile]) -> list[datetime | None]:
    """Read every file's start timestamp in one pass off the event loop.

    Each parser stops at the first timestamped record, so this is cheap per file
    — but nine hundred of them is not, and it all has to happen before anything
    can be deduplicated.
    """
    out: list[datetime | None] = []
    for item in candidates:
        try:
            out.append(read_activity_start_time(str(item.path), item.format or "fit"))
        except Exception:  # noqa: BLE001 - a file too broken to date is still importable
            out.append(None)
    return out


def deduplicate_batch(candidates: list[_Candidate]) -> None:
    """Collapse files that describe the same ride, keeping the richest format.

    An export holding one ride as ``.fit``, ``.tcx`` and ``.gpx`` should produce
    one activity, and it should be the FIT: power, laps and the device's own
    totals, where the GPX has coordinates and a heart rate.

    Mutates the candidates in place — losers get ``duplicate_of`` set to the
    filename that won, so a skip is a result worth showing rather than a file
    silently vanishing.

    Grouping walks the batch in time order, opening a new group whenever a file
    starts more than :data:`DUPLICATE_WINDOW` after the one that opened the
    current group.
    """
    timed = sorted(
        (c for c in candidates if c.start_time is not None),
        key=lambda c: _normalise(c.start_time),
    )

    group: list[_Candidate] = []
    anchor: datetime | None = None

    def settle(members: list[_Candidate]) -> None:
        if len(members) < 2:
            return
        winner = min(members, key=lambda c: format_priority(c.file.format or ""))
        for member in members:
            if member is not winner:
                member.duplicate_of = winner.file.name

    for candidate in timed:
        moment = _normalise(candidate.start_time)
        if anchor is not None and moment - anchor > DUPLICATE_WINDOW:
            settle(group)
            group = []
            anchor = None
        if anchor is None:
            anchor = moment
        group.append(candidate)
    settle(group)


async def _existing_activity(session, athlete_id: str, start: datetime) -> Activity | None:
    result = await session.execute(
        select(Activity).where(
            Activity.athlete_id == athlete_id,
            Activity.start_time >= start - DUPLICATE_WINDOW,
            Activity.start_time <= start + DUPLICATE_WINDOW,
        )
    )
    return result.scalars().first()


def _store_original(source: Path, user_id: str, fmt: str) -> Path:
    """Move an expanded file into the athlete's upload directory, keeping its format.

    The original is kept as it arrived rather than converted to FIT, so what an
    athlete downloads later is the file they imported and a reprocess re-reads
    exactly what was parsed the first time.
    """
    storage_dir = settings.user_fit_dir(user_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    destination = storage_dir / f"{uuid.uuid4()}.{fmt}"
    shutil.move(str(source), destination)
    return destination


async def _import_one(
    candidate: _Candidate,
    athlete: Athlete,
    session,
    user_id: str,
) -> dict:
    """Import a single expanded file, returning its result row.

    Never raises for a file-level problem: a parse failure, a duplicate or an
    unreadable file all come back as a result the job records and moves on from.
    """
    expanded = candidate.file
    fmt = expanded.format or "fit"

    if candidate.duplicate_of is not None:
        return _result(
            expanded.name,
            OUTCOME_DUPLICATE,
            reason=f"The same activity is also in this import as {candidate.duplicate_of}",
            fmt=fmt,
        )

    # Cheap pre-check, outside the lease. Re-importing an archive is a normal
    # thing to do, and this is what keeps that from parsing nine hundred files
    # to discover it already has all of them. The authoritative check is the
    # one under the lease below.
    if candidate.start_time is not None:
        existing = await _existing_activity(session, athlete.id, candidate.start_time)
        if existing is not None:
            return _result(
                expanded.name,
                OUTCOME_DUPLICATE,
                reason="An activity starting at this time already exists",
                activity_id=existing.id,
                fmt=fmt,
            )

    # Parsing is synchronous and by far the most expensive step, so it runs off
    # the event loop — and outside the lease, which only has to cover the
    # database work it exists to serialise.
    try:
        parsed = await asyncio.to_thread(parse_activity_file, str(expanded.path), fmt)
    except ActivityParseError as exc:
        return _result(expanded.name, OUTCOME_FAILED, reason=str(exc), fmt=fmt)
    except Exception as exc:  # noqa: BLE001 - one bad file must not end the job
        log.warning("Unexpected failure parsing %s", expanded.name, exc_info=True)
        return _result(
            expanded.name, OUTCOME_FAILED, reason=f"Could not read the file: {exc}", fmt=fmt
        )

    # ── Find-or-create under the same two guards the provider syncs use ──────
    #
    # It matters more here than on the single upload it grew out of. An upload's
    # window is one request; an import holds the guard open once per file for the
    # length of the job, so a Strava webhook landing mid-import is an ordinary
    # Tuesday rather than a coincidence.
    #
    # `process_activity_file` commits inside the block, which is the invariant
    # the guard requires.
    async with activity_create_guard(session, user_id, athlete.id):
        if candidate.start_time is not None:
            existing = await _existing_activity(session, athlete.id, candidate.start_time)
            if existing is not None:
                # Something else created it between the pre-check and here —
                # exactly the race this block exists to lose safely.
                return _result(
                    expanded.name,
                    OUTCOME_DUPLICATE,
                    reason="An activity starting at this time already exists",
                    activity_id=existing.id,
                    fmt=fmt,
                )

        stored = _store_original(expanded.path, user_id, fmt)
        activity = Activity(id=str(uuid.uuid4()), athlete_id=athlete.id, status="pending")
        session.add(activity)
        await session.flush()

        source = ActivitySource(
            activity_id=activity.id,
            provider="upload",
            fit_file_path=str(stored),
            format=fmt,
        )
        session.add(source)
        await session.flush()

        try:
            await process_activity_file(
                str(stored), athlete, activity, session, fmt=fmt, parsed=parsed
            )
        except Exception as exc:  # noqa: BLE001 - as above
            log.warning("Failed to process imported file %s", expanded.name, exc_info=True)
            await session.rollback()
            # A rollback expires every object in the session regardless of
            # `expire_on_commit=False`, and the athlete is read again on the
            # next file — refreshing it here keeps that from becoming implicit
            # IO in a place that cannot await.
            await session.refresh(athlete)
            stored.unlink(missing_ok=True)
            return _result(
                expanded.name,
                OUTCOME_FAILED,
                reason=f"Could not process the file: {exc}",
                fmt=fmt,
            )

    # Read before anything that could expire it: a rollback below would turn
    # this into a lazy load, and a lazy load here is implicit IO in a place that
    # cannot await it.
    activity_id = activity.id

    try:
        encrypt_file(stored, user_id)
    except Exception:
        # Not fatal, and not a reason to roll anything back — the activity is
        # already committed and the file is readable, just not encrypted.
        log.warning("Failed to encrypt imported file %s — left in plaintext", stored, exc_info=True)
    else:
        source.fit_file_encrypted = True
        try:
            await session.commit()
        except Exception:
            # A failed commit leaves the session needing a rollback, and without
            # one every *later* file in the job fails on its first flush.
            log.exception("Could not record that %s was encrypted", stored)
            try:
                await session.rollback()
                await session.refresh(athlete)
            except Exception:
                log.exception("Could not recover the session after a failed commit")

    # Linking is per-activity by nature (a ride matches the workout planned for
    # its own day), unlike the metrics recalculation the job does once at the end.
    await find_and_link_workout(session, athlete.id, activity)

    return _result(
        expanded.name,
        OUTCOME_IMPORTED,
        activity_id=activity_id,
        fmt=fmt,
    )


async def _finalise(job: ImportJob, session, athlete_id: str, earliest: date | None) -> None:
    """The once-per-job work: metrics, adherence, achievements.

    Deliberately outside the per-file loop: ``recalculate_from`` walks every day
    from its start date to today, so calling it per activity is quadratic in the
    size of the import, and each answer is thrown away by the next call.

    Achievements are only *marked* here (issue #69). A multi-year import is where
    that single reconcile pass is most expensive, and the athlete's next read
    settles it anyway.
    """
    if earliest is None:
        return
    try:
        await recalculate_from(athlete_id, earliest, session)
        await catch_up_adherence(athlete_id, session)
        await mark_achievements_dirty(athlete_id, session)
    except Exception:
        # The activities are imported and correct; the derived metrics are not.
        # Recording that on the job beats failing an import that did land — but
        # the session has to be rolled back first, or the caller's final commit
        # raises and the job is left `running` forever, blocking the next one.
        log.exception("Post-import recalculation failed for athlete %s", athlete_id)
        try:
            await session.rollback()
        except Exception:
            log.exception("Rollback after a failed post-import recalculation failed")
        job.error = "Activities were imported, but recalculating metrics failed"


async def reap_stale_staging(session, athlete_id: str, user_id: str, keep: str) -> None:
    """Delete staging directories belonging to imports that are no longer running.

    A job's own directory is removed when it ends, whatever the outcome — but
    only by the task that runs it. If the process dies between the 202 and the
    task starting, up to the request cap is left on disk with nothing to reap
    it, and `is_in_flight` recovers the *job* while the bytes stay. Sweeping at
    the start of the next import is enough: imports are one at a time, so
    anything here that is neither the new job nor in flight is finished with.
    """
    root = settings.user_fit_dir(user_id) / "imports"
    if not root.is_dir():
        return

    live = {
        job.id
        for job in (
            await session.execute(
                select(ImportJob).where(
                    ImportJob.athlete_id == athlete_id,
                    ImportJob.status.in_(("pending", "running")),
                )
            )
        ).scalars()
        if is_in_flight(job)
    }
    live.add(keep)

    for directory in root.iterdir():
        if directory.is_dir() and directory.name not in live:
            log.info("Removing staging directory left behind by import %s", directory.name)
            shutil.rmtree(directory, ignore_errors=True)


async def run_import_job(
    job_id: str,
    athlete_id: str,
    user_id: str,
    uploads: list[tuple[Path, str]],
    work_dir: Path,
) -> None:
    """Work through one import job to completion, on its own database session.

    This is the entry point the endpoint schedules as a background task. The
    work itself is :func:`execute_import_job`, which takes a session, so tests
    can drive the real thing against their own database instead of a
    re-implementation of it.
    """
    async with get_user_session_factory(user_id)() as session:
        await execute_import_job(job_id, athlete_id, user_id, uploads, work_dir, session)


async def execute_import_job(
    job_id: str,
    athlete_id: str,
    user_id: str,
    uploads: list[tuple[Path, str]],
    work_dir: Path,
    session,
) -> None:
    """Work through one import job to completion, updating its row as it goes.

    ``uploads`` are the staged request parts as ``(path, original filename)``;
    ``work_dir`` is the job's scratch directory, removed when the job ends
    whatever the outcome.
    """
    job = (
        await session.execute(select(ImportJob).where(ImportJob.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        log.warning("Import job %s vanished before it could run", job_id)
        shutil.rmtree(work_dir, ignore_errors=True)
        return

    athlete = (
        await session.execute(select(Athlete).where(Athlete.id == athlete_id))
    ).scalar_one_or_none()
    if athlete is None:
        job.status = "failed"
        job.error = "Athlete not found"
        job.completed_at = datetime.now(timezone.utc)
        await session.commit()
        shutil.rmtree(work_dir, ignore_errors=True)
        return

    job.status = "running"
    await session.commit()

    results: list[dict] = []
    earliest: date | None = None
    # Counted here rather than on the row: a file whose processing fails
    # rolls the session back, which would discard increments made on `job`
    # since the last commit along with the half-written activity.
    counts = {OUTCOME_IMPORTED: 0, OUTCOME_DUPLICATE: 0, OUTCOME_FAILED: 0}

    def publish() -> None:
        job.imported = counts[OUTCOME_IMPORTED]
        job.skipped_duplicate = counts[OUTCOME_DUPLICATE]
        job.failed = counts[OUTCOME_FAILED]
        job.results = list(results)

    try:
        expanded = await asyncio.to_thread(expand_all, uploads, work_dir / "expanded")

        unusable = [item for item in expanded if item.path is None]
        usable = [item for item in expanded if item.path is not None]
        for item in unusable:
            results.append(
                _result(
                    item.name,
                    OUTCOME_FAILED,
                    reason=item.error or "Could not be read",
                    fmt=item.format,
                )
            )
            counts[OUTCOME_FAILED] += 1

        starts = await asyncio.to_thread(_start_times, usable)
        candidates = [
            _Candidate(item, start) for item, start in zip(usable, starts)
        ]
        deduplicate_batch(candidates)

        job.total_files = len(expanded)
        publish()
        await session.commit()

        for index, candidate in enumerate(candidates, start=1):
            outcome = await _import_one(candidate, athlete, session, user_id)
            results.append(outcome)
            counts[outcome["outcome"]] += 1

            if outcome["outcome"] == OUTCOME_IMPORTED and candidate.start_time:
                started = _normalise(candidate.start_time).date()
                earliest = started if earliest is None else min(earliest, started)

            if index % _PROGRESS_EVERY == 0:
                publish()
                await session.commit()

        publish()
        await session.commit()
        await _finalise(job, session, athlete_id, earliest)

        job.status = "completed"
    except ArchiveTooLarge as exc:
        job.status = "failed"
        job.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - the job row is the error channel
        log.exception("Import job %s failed", job_id)
        job.status = "failed"
        job.error = f"Import failed: {exc}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            publish()
            job.completed_at = datetime.now(timezone.utc)
            await session.commit()
        except Exception:
            log.exception("Could not record the outcome of import job %s", job_id)
