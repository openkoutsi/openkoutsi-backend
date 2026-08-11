"""Stored conversations with Koutsi (issue #44).

The first LLM surface in openkoutsi with anything to persist. Every other one is
a *generation*: ``analyze_training_status_bg`` builds a prompt, streams one
answer into a column on ``athletes``, and the message list it built dies with the
task. A dialogue has to survive the turn that produced it, so it gets tables.

Per-user DB, following the ``Message`` precedent in :mod:`.message_orm`: the
database file identifies the owner, so there is no owner column and no
``WHERE user_id = …`` to forget. Isolation is issue #42's per-user session, not
a predicate.

What is stored, and what is deliberately not
--------------------------------------------
Only the dialogue — the athlete's turns and Koutsi's prose. **Tool calls and
their results are not persisted**, though replaying them was the obvious design
and the issue left room for it. Three reasons, in order of weight:

1. They are almost all of the context. A single ``list_recent_activities`` result
   dwarfs the sentence that prompted it, and replaying every past result into
   every later turn is what would overflow a small model's window — the
   context-growth problem issue #44 predicted, created almost entirely by
   storing the one thing that does not need storing.
2. They go stale. A tool result from Tuesday describes Tuesday. Re-running the
   tool on Friday's turn is not merely cheaper than replaying it, it is *more
   correct*, and the tools are read-only so there is nothing to be idempotent
   about.
3. They are evidence, not dialogue. The athlete asked a question and Koutsi
   answered; the lookups in between are working, and the GDPR export is more
   honest for containing the conversation rather than the machinery.

``tool_names`` keeps only the names, for the "Koutsi looked at…" footer, drawn
from the same vocabulary as issue #43's progress codes.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import UserBase


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


#: A turn that has been accepted but has no agent slot yet (issue #44).
#:
#: Unique to chat. Background runs take :func:`~..services.llm_agent._run_slot`'s
#: immediate refusal and fall back to the blob prompt, which chat has no
#: equivalent of — so an interactive turn waits instead, and the wait is a state
#: the athlete can see rather than a spinner that means nothing.
STATUS_QUEUED = "queued"
#: The agent loop is running: gathering, then writing.
STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"

#: Settled states — nothing further will be written to the row.
TERMINAL_STATUSES = frozenset({STATUS_COMPLETE, STATUS_ERROR})

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


class ChatConversation(UserBase):
    """One thread of conversation with Koutsi.

    ``title`` is the first ~60 characters of the athlete's opening message, not
    a model-written summary. Chat is the first surface the athlete can trigger
    arbitrarily often and every turn is a full agent run, so spending an extra
    completion on a label for the sidebar is the wrong place to put the money.
    """

    __tablename__ = "chat_conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    #: Bumped on every turn, so the sidebar can sort by "last spoken to" rather
    #: than by when the thread was opened.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class ChatMessage(UserBase):
    """One turn: either something the athlete wrote or something Koutsi did.

    The assistant row is created **before** the answer exists, in
    :data:`STATUS_QUEUED`, and is then written into in place — which is what lets
    a reload mid-answer resume rather than lose the turn. It is the same
    stream-into-a-column shape ``training_status`` uses, moved off a singleton
    column on ``athletes`` and onto a row per turn, because a conversation can
    have more than one answer in flight over its life.

    User rows carry no status: nothing about them is pending.
    """

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    #: Grows as the answer streams. Empty on a queued or freshly-pending row.
    content: Mapped[str] = mapped_column(String, nullable=False, default="")
    #: One of the ``STATUS_*`` constants on assistant rows; NULL on user rows.
    status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    #: Issue #43's progress code — ``thinking`` or ``tool.<name>`` — while the
    #: run is gathering and has written no prose. Cleared when prose starts.
    progress: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    #: Why a run failed, as a machine key the web app localises: the athlete
    #: gets "Koutsi is busy finishing your daily check-in" rather than one
    #: generic apology for five quite different causes.
    error_code: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    #: Registry tool names this turn called, in call order, for the
    #: "Koutsi looked at…" footer. Never the arguments or the results.
    tool_names: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    #: Touched on every progress commit, so the stuck-run check means "no
    #: progress for N minutes" rather than "started N minutes ago" — the
    #: distinction issue #91 had to make for the daily card, for the same reason:
    #: a healthy run against a slow local model must not be declared dead
    #: underneath itself.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


#: The thread read is ``WHERE conversation_id = ? ORDER BY created_at`` on every
#: poll, and chat polls fast while a turn is live.
Index(
    "ix_chat_messages_conversation_created",
    ChatMessage.conversation_id,
    ChatMessage.created_at,
)
