"""Wire shapes for bulk activity import (issue #36)."""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field, field_validator


class ImportFileResult(BaseModel):
    """What happened to one file in an import.

    ``outcome`` is one of ``imported``, ``skipped_duplicate`` or ``failed``.
    ``reason`` is prose written for the athlete — "An activity starting at this
    time already exists", "Not a FIT, GPX or TCX activity file" — because a
    count of failures is not something anyone can act on and a list of names
    with reasons is.
    """

    filename: str
    outcome: str
    reason: Optional[str] = None
    activity_id: Optional[str] = None
    format: Optional[str] = None


class ImportJobSummary(BaseModel):
    """An import job without its per-file detail, for the list view."""

    id: str
    status: str
    source_name: Optional[str] = None
    total_files: int = 0
    imported: int = 0
    skipped_duplicate: int = 0
    failed: int = 0
    error: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

    @computed_field
    @property
    def processed(self) -> int:
        """Files finished with, of ``total_files``.

        Progress is reported as a count rather than a percentage because
        ``total_files`` is 0 until the archives have been walked, and a
        percentage of an unknown total is a lie with a number on it.
        """
        return self.imported + self.skipped_duplicate + self.failed


class ImportJobResponse(ImportJobSummary):
    """One import job in full, including every file's outcome."""

    results: list[ImportFileResult] = Field(default_factory=list)

    @field_validator("results", mode="before")
    @classmethod
    def _no_results_yet(cls, value: Any) -> Any:
        """A job that has not started has a NULL result list; the client wants ``[]``.

        Keeping the wire shape a list from the first poll onwards means the
        client renders "nothing yet" the same way it renders "nothing left",
        instead of needing a null branch that only exists for a second.
        """
        return [] if value is None else value


class ImportJobListResponse(BaseModel):
    items: list[ImportJobSummary]
    total: int
