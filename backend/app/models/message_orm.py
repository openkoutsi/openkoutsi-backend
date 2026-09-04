import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import UserBase


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Message(UserBase):
    """An in-app message in a user's inbox.

    Lives in the per-user DB, so the file itself identifies the recipient — no
    recipient column is needed.

    Text is stored pre-rendered in `title` and `body` by
    `backend.app.services.message_text` at send time. Rendering from an i18n
    template in the web app meant a message could never say more than the
    template the frontend shipped — the achievement notification could count
    badges but not name them. `locale` records which language the text was
    rendered in, so translated rendering can be added without a migration.

    `type` and `data` are still stored, as machine-readable metadata (icon
    selection, deep links, the GDPR export) rather than as the source of the
    text. All three text columns are nullable, for messages written before this.
    """

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    type: Mapped[str] = mapped_column(String, nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    body: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    locale: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
