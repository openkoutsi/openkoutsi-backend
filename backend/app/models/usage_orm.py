"""ORM models for the dedicated LLM-usage database (issue #9).

One append-only row per **instance-paid** LLM call. BYOK calls (the user pays
their own provider) are never written here. Lives in its own database
(``UsageBase``) with no registry foreign keys, so its unbounded rows can be
pruned/rotated independently and a user-deletion sweep is a plain
``DELETE ... WHERE user_id = ?``.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import UsageBase


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LlmUsage(UsageBase):
    """A single instance-paid LLM call's token accounting.

    Input (``prompt_tokens``) and output (``completion_tokens``) tokens are
    stored separately and never merged — providers price them differently.
    Token columns are nullable: some servers (e.g. older Ollama) omit ``usage``
    even when asked, in which case the call is recorded with nulls rather than
    estimated.
    """

    __tablename__ = "llm_usage"
    __table_args__ = (Index("ix_llm_usage_user_created", "user_id", "created_at"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=_now, nullable=False
    )
    # chat | plan_generate | workout_generate | activity_analysis | training_status
    feature: Mapped[str] = mapped_column(String, nullable=False)
    # Which provider served the call — the resolved preset's label/name or the
    # base-URL host as a fallback. Recorded alongside ``model``, not merged.
    provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # instance | none — user/BYOK is never recorded; env no longer exists.
    key_source: Mapped[str] = mapped_column(String, nullable=False, default="instance")
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
