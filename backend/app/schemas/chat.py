from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from backend.app.core.config import settings


class ChatMessageResponse(BaseModel):
    id: str
    #: ``user`` or ``assistant``.
    role: str
    #: Grows while an answer streams; empty on a queued turn.
    content: str
    #: ``queued`` / ``pending`` / ``complete`` / ``error`` on assistant turns,
    #: null on the athlete's own.
    status: Optional[str] = None
    #: Issue #43's progress code while the turn is still gathering.
    progress: Optional[str] = None
    #: Why a turn failed, as a key the web app localises — see
    #: ``services.llm_agent``'s ``CODE_*`` constants.
    error_code: Optional[str] = None
    #: Registry tool names this turn consulted, for the "Koutsi looked at…"
    #: footer. Never the arguments and never the results.
    tool_names: Optional[list[str]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatConversationSummary(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatConversationDetail(ChatConversationSummary):
    messages: list[ChatMessageResponse] = []


class ChatTurnBody(BaseModel):
    """One question from the athlete.

    ``message`` is the *only* thing the client sends. Everything the model
    actually receives — the system prompt, the scope policy, the replayed
    history — is built server-side in ``services.llm_chat``, which is the whole
    difference between this and the ``/api/llm/chat`` proxy issue #45 removed.
    """

    message: str = Field(min_length=1)
    #: Overrides the athlete's stored locale for this turn, as the training
    #: status trigger does — the browser knows which language the page is in.
    locale: Optional[str] = None


class ChatConversationCreate(BaseModel):
    #: Optional opening question. When given, the conversation is created and
    #: the first turn started in one round trip.
    message: Optional[str] = None
    locale: Optional[str] = None


class ChatAvailability(BaseModel):
    """What the chat surface may do right now, so the UI can say why not.

    Chat is the one LLM surface with no single-shot prompt to fall back on, so
    the reasons it might be unusable have to be answerable *before* the athlete
    types rather than discovered as a failed turn.
    """

    #: Has the athlete opted into the agentic coach? Chat is meaningless without
    #: tools, so it rides the same switch (``app_settings.agentic_koutsi``).
    enabled: bool
    #: Can the resolved model call tools at all? False means the nav entry is
    #: disabled: a settled property of the provider, not a transient failure, so
    #: inviting a retry would be a lie.
    tools_supported: bool
    #: False when the instance gate (issue #9) denies this user.
    entitled: bool
    turns_remaining_today: int
    max_turns_per_conversation: int = settings.chat_max_turns_per_conversation
    max_message_chars: int = settings.chat_max_message_chars
