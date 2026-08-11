"""Conversational Koutsi (issue #44).

The client sends one string: the question. Everything the model sees is built in
``services.llm_chat`` — this module owns storage, gating and the budgets, and
never lets a message array in from outside. That is the whole distinction from
the ``POST /api/llm/chat`` proxy issue #45 removed, which forwarded a
client-supplied array and therefore let any token holder replace the system
prompt the scope policy lives in.

Turns run as background tasks writing into their own row, the same shape the
daily training-status card uses, so a slow local model cannot time the request
out and a reload mid-answer resumes instead of losing the turn.
"""
import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import UserContext, get_current_user
from backend.app.core.config import settings
from backend.app.core.limiter import limiter
from backend.app.core.scopes import pat_forbidden
from backend.app.core.timezones import local_now
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.chat_orm import (
    ROLE_ASSISTANT,
    ROLE_USER,
    STATUS_ERROR,
    STATUS_PENDING,
    STATUS_QUEUED,
    ChatConversation,
    ChatMessage,
)
from backend.app.models.registry_orm import InstanceSettings
from backend.app.models.user_orm import Athlete
from backend.app.schemas.chat import (
    ChatAvailability,
    ChatConversationCreate,
    ChatConversationDetail,
    ChatConversationSummary,
    ChatMessageResponse,
    ChatRetryBody,
    ChatTurnBody,
)
from backend.app.services.llm_access import check_llm_access, subscription_required_error
from backend.app.services.llm_agent import agentic_enabled
from backend.app.services.llm_chat import (
    conversation_title,
    run_chat_turn_bg,
    settle_stuck_turns,
    turns_in_conversation,
    turns_used_today,
)

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[pat_forbidden()])

#: The athlete has not switched the agentic coach on, so there are no tools and
#: therefore nothing chat could usefully be.
CHAT_DISABLED = "chat_disabled"
#: This model cannot call tools. A settled property of the provider — the UI
#: disables the surface rather than inviting a retry that fails identically.
CHAT_TOOLS_UNSUPPORTED = "chat_tools_unsupported"
CHAT_DAILY_BUDGET = "chat_daily_budget"
CHAT_CONVERSATION_BUDGET = "chat_conversation_budget"
#: A turn is already running in this conversation.
CHAT_TURN_IN_FLIGHT = "chat_turn_in_flight"


async def get_chat_session(
    ctx: UserContext = Depends(get_current_user),
) -> AsyncGenerator[tuple[UserContext, AsyncSession], None]:
    await init_user_db(ctx.user_id)
    async with get_user_session_factory(ctx.user_id)() as session:
        yield ctx, session


async def _athlete(session: AsyncSession) -> Athlete:
    """The one athlete in this per-user DB."""
    athlete = (await session.execute(select(Athlete))).scalars().first()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")
    return athlete


def _budget_error(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=429, detail={"code": code, "message": message})


async def _require_chat_access(
    ctx: UserContext, athlete: Athlete, registry_session: AsyncSession
) -> None:
    """Everything that must be true before a turn may be started.

    Ordered cheapest-and-most-explanatory first: an athlete who never switched
    the agentic coach on should be told that, not told they need a subscription.
    """
    if not agentic_enabled(athlete):
        raise HTTPException(
            status_code=403,
            detail={
                "code": CHAT_DISABLED,
                "message": (
                    "Chatting with Koutsi needs the agentic coach switched on in "
                    "your profile."
                ),
            },
        )

    instance = (
        await registry_session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    access = await check_llm_access(ctx, athlete, instance, registry_session)
    if not access.allowed:
        raise subscription_required_error(access)


async def _enforce_daily_budget(session: AsyncSession, athlete: Athlete) -> None:
    now = local_now((athlete.app_settings or {}).get("timezone"))
    if await turns_used_today(session, now=now) >= settings.chat_max_turns_per_day:
        raise _budget_error(
            CHAT_DAILY_BUDGET,
            "You have used all of today's questions. More become available as "
            "the day rolls on.",
        )


async def _enforce_budgets(
    session: AsyncSession, athlete: Athlete, conversation_id: str
) -> None:
    await _enforce_daily_budget(session, athlete)
    if (
        await turns_in_conversation(session, conversation_id)
        >= settings.chat_max_turns_per_conversation
    ):
        raise _budget_error(
            CHAT_CONVERSATION_BUDGET,
            "This conversation has reached its length limit. Start a new one to "
            "carry on.",
        )


def _validate_message(message: str) -> str:
    text = (message or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Message must not be empty")
    if len(text) > settings.chat_max_message_chars:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Message must be at most {settings.chat_max_message_chars} "
                "characters"
            ),
        )
    return text


async def _get_conversation(
    conversation_id: str, session: AsyncSession
) -> ChatConversation:
    """Load a conversation from *this user's* DB.

    Isolation is the database file, not a predicate: a conversation id minted in
    another user's DB simply is not in this one, so cross-user access 404s
    without anything here having to remember to filter.
    """
    result = await session.execute(
        select(ChatConversation).where(ChatConversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


async def _messages_of(
    session: AsyncSession, conversation_id: str
) -> list[ChatMessage]:
    result = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc())
    )
    return list(result.scalars().all())


async def _start_turn(
    ctx: UserContext,
    session: AsyncSession,
    athlete: Athlete,
    conversation: ChatConversation,
    message: str,
    locale: str | None,
) -> ChatMessage:
    """Write the question and its answer-to-be, then spawn the run.

    Both rows are committed before the task starts. The assistant row exists in
    ``queued`` from the first moment precisely so the athlete's next poll shows
    the question *and* something happening under it, whether or not an agent
    slot was free — the wait is a state, not a gap.

    The one-turn-at-a-time check below is **best effort, not an invariant**: the
    read and the insert are separated by awaits with no uniqueness constraint
    underneath, so two simultaneous posts to the same conversation could both
    pass it. The web app disables the composer while a turn is live, so this is
    not reachable by accident, and the blast radius is one athlete's own thread —
    but do not build anything on it holding absolutely.
    """
    existing = await _messages_of(session, conversation.id)
    if any(
        m.role == ROLE_ASSISTANT and m.status in (STATUS_QUEUED, STATUS_PENDING)
        for m in existing
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": CHAT_TURN_IN_FLIGHT,
                "message": "Koutsi is still answering your last question.",
            },
        )

    now = datetime.now(timezone.utc)
    session.add(
        ChatMessage(
            conversation_id=conversation.id,
            role=ROLE_USER,
            content=message,
            created_at=now,
            updated_at=now,
        )
    )
    answer = ChatMessage(
        conversation_id=conversation.id,
        role=ROLE_ASSISTANT,
        content="",
        status=STATUS_QUEUED,
        created_at=now,
        updated_at=now,
    )
    session.add(answer)
    if not conversation.title:
        conversation.title = conversation_title(message)
    conversation.updated_at = now
    await session.commit()
    await session.refresh(answer)

    asyncio.create_task(
        run_chat_turn_bg(ctx.user_id, conversation.id, answer.id, locale)
    )
    return answer


# ── Routes ──────────────────────────────────────────────────────────────────


@router.get("/availability", response_model=ChatAvailability,
            operation_id="getChatAvailability", summary="Whether chat can be used")
async def get_availability(
    ctx_session=Depends(get_chat_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Why chat is or is not usable, before the athlete types anything.

    Chat is the only LLM surface with no single-shot prompt behind it, so the
    reasons it might not work have to be answerable up front: discovering
    "your model cannot call tools" as a failed turn, after composing a question,
    is a bad way to learn a permanent fact about your setup.
    """
    ctx, session = ctx_session
    athlete = await _athlete(session)

    instance = (
        await registry_session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    access = await check_llm_access(ctx, athlete, instance, registry_session)

    # Resolution only — no request is made to the provider, so this stays a
    # cheap call the chat page can make on every load.
    from backend.app.services.llm_streaming import resolve_stream_setup

    tools_supported = True
    try:
        setup = await resolve_stream_setup(athlete, ctx.user_id)
        tools_supported = bool(setup.cfg.tools_supported)
    except Exception:  # noqa: BLE001 - no LLM configured at all
        tools_supported = False

    now = local_now((athlete.app_settings or {}).get("timezone"))
    used = await turns_used_today(session, now=now)

    return ChatAvailability(
        enabled=agentic_enabled(athlete),
        tools_supported=tools_supported,
        entitled=access.allowed,
        turns_remaining_today=max(0, settings.chat_max_turns_per_day - used),
        max_turns_per_conversation=settings.chat_max_turns_per_conversation,
        max_message_chars=settings.chat_max_message_chars,
    )


@router.get("/conversations", response_model=list[ChatConversationSummary],
            operation_id="listChatConversations", summary="List conversations")
async def list_conversations(ctx_session=Depends(get_chat_session)):
    _, session = ctx_session
    result = await session.execute(
        select(ChatConversation).order_by(ChatConversation.updated_at.desc())
    )
    return [
        ChatConversationSummary.model_validate(c) for c in result.scalars().all()
    ]


@router.post("/conversations", response_model=ChatConversationDetail, status_code=201,
             operation_id="createChatConversation", summary="Start a conversation")
@limiter.limit("60/hour")
async def create_conversation(
    # slowapi reads the key off the request, so the parameter is required by
    # the decorator even though the handler never touches it.
    request: Request,
    body: ChatConversationCreate = ChatConversationCreate(),
    ctx_session=Depends(get_chat_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session = ctx_session
    athlete = await _athlete(session)
    await _require_chat_access(ctx, athlete, registry_session)

    # Refuse *before* the row exists. Committing the conversation first and then
    # validating leaves a titleless orphan in the athlete's rail for every
    # rejected opening message — and they accumulate fastest when the athlete is
    # already being told no, since a spent budget refuses every attempt.
    text = _validate_message(body.message) if body.message is not None else None
    if text is not None:
        # Only the daily cap can bind here; a conversation that does not exist
        # yet trivially satisfies the per-conversation one.
        await _enforce_daily_budget(session, athlete)

    now = datetime.now(timezone.utc)
    conversation = ChatConversation(created_at=now, updated_at=now)
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)

    if text is not None:
        await _start_turn(ctx, session, athlete, conversation, text, body.locale)

    messages = await _messages_of(session, conversation.id)
    return ChatConversationDetail(
        **ChatConversationSummary.model_validate(conversation).model_dump(),
        messages=[ChatMessageResponse.model_validate(m) for m in messages],
    )


@router.get("/conversations/{conversation_id}", response_model=ChatConversationDetail,
            operation_id="getChatConversation", summary="Read a conversation")
async def get_conversation(
    conversation_id: str, ctx_session=Depends(get_chat_session)
):
    """The thread, polled while a turn is live.

    Stuck turns are settled on the way through: ``stream_into_db`` settles its
    own row and ``failure_recovery`` covers a failure just outside it, but
    neither survives the process being restarted mid-turn, and this read is the
    only moment anybody cares that a row from a previous life is still
    ``pending``.
    """
    _, session = ctx_session
    conversation = await _get_conversation(conversation_id, session)
    athlete = await _athlete(session)

    if await settle_stuck_turns(
        session, now=local_now((athlete.app_settings or {}).get("timezone"))
    ):
        await session.commit()

    messages = await _messages_of(session, conversation_id)
    return ChatConversationDetail(
        **ChatConversationSummary.model_validate(conversation).model_dump(),
        messages=[ChatMessageResponse.model_validate(m) for m in messages],
    )


@router.delete("/conversations/{conversation_id}", status_code=204,
               operation_id="deleteChatConversation", summary="Delete a conversation")
async def delete_conversation(
    conversation_id: str, ctx_session=Depends(get_chat_session)
):
    """Remove a conversation and everything said in it.

    The messages are deleted explicitly rather than left to the foreign key:
    ``PRAGMA foreign_keys`` is off on these connections, so ``ON DELETE CASCADE``
    is documentation rather than behaviour, and relying on it here would orphan
    every row of a thread an athlete believed they had deleted — health-adjacent
    free text they wrote about their own body.

    A live turn is **not** a reason to refuse. This is a privacy action, and the
    confirm dialog says the whole thread goes; making the athlete wait out an
    answer they no longer want — possibly minutes on a slow local model — would
    be the wrong trade. The run notices on its next progress marker that its row
    has gone (``llm_chat._still_ours``), stands down and releases its slot,
    rather than finishing and writing into rows that no longer exist.
    """
    _, session = ctx_session
    conversation = await _get_conversation(conversation_id, session)
    for message in await _messages_of(session, conversation_id):
        await session.delete(message)
    await session.delete(conversation)
    await session.commit()


@router.post("/conversations/{conversation_id}/messages",
             response_model=ChatMessageResponse, status_code=202,
             operation_id="sendChatMessage", summary="Ask Koutsi something")
@limiter.limit("60/hour")
async def send_message(
    # slowapi reads the key off the request, so the parameter is required by
    # the decorator even though the handler never touches it.
    request: Request,
    conversation_id: str,
    body: ChatTurnBody,
    ctx_session=Depends(get_chat_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Accept a question and start the turn that answers it.

    202 rather than 200: the answer does not exist yet. What comes back is the
    assistant row in ``queued``, which the client polls through
    ``GET /conversations/{id}`` — the same DB-backed streaming the daily card
    uses, and for the same reason, that a local model may take minutes and a
    page reload must not lose the run.
    """
    ctx, session = ctx_session
    athlete = await _athlete(session)
    await _require_chat_access(ctx, athlete, registry_session)

    conversation = await _get_conversation(conversation_id, session)
    text = _validate_message(body.message)
    await _enforce_budgets(session, athlete, conversation_id)

    answer = await _start_turn(
        ctx, session, athlete, conversation, text, body.locale
    )
    return ChatMessageResponse.model_validate(answer)


@router.post("/conversations/{conversation_id}/messages/{message_id}/retry",
             response_model=ChatMessageResponse, status_code=202,
             operation_id="retryChatMessage", summary="Re-run a failed answer")
@limiter.limit("60/hour")
async def retry_message(
    # slowapi reads the key off the request, so the parameter is required by
    # the decorator even though the handler never touches it.
    request: Request,
    conversation_id: str,
    message_id: str,
    body: ChatRetryBody = ChatRetryBody(),
    ctx_session=Depends(get_chat_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Run a failed turn again **in place**, rather than asking anew.

    The obvious client-side retry — re-post the same text — is wrong in three
    ways at once, and they compound on exactly the setup most likely to need a
    retry (a local model that is flaky or not running). The athlete's question
    appears in the thread twice, verbatim, right after something has visibly
    gone wrong; the second attempt spends another turn of the daily budget; and
    the replayed history ends with the same question adjacent to itself, which
    several chat templates either reject or silently merge.

    Re-running the existing row avoids all three: one question, one answer slot,
    one charge. The failed row goes back to ``queued`` and the same background
    task picks it up, so the client polls exactly as it did the first time.

    Any failed row in the thread may be retried, not only the newest. That is
    safe because the run builds its history from the messages *before* the row it
    is answering (see ``llm_chat.run_chat_turn_bg``) rather than from everything
    else in the thread — so an older retry gets the question it is actually
    answering and nothing from after it. The web app only ever offers the newest,
    but this endpoint does not lean on that: two client-side guards are not where
    a server-side invariant belongs.
    """
    ctx, session = ctx_session
    athlete = await _athlete(session)
    await _require_chat_access(ctx, athlete, registry_session)
    await _get_conversation(conversation_id, session)

    result = await session.execute(
        select(ChatMessage).where(
            ChatMessage.id == message_id,
            ChatMessage.conversation_id == conversation_id,
        )
    )
    answer = result.scalar_one_or_none()
    if answer is None or answer.role != ROLE_ASSISTANT:
        raise HTTPException(status_code=404, detail="Message not found")
    if answer.status != STATUS_ERROR:
        # Only a failure can be retried. A completed answer would need a new
        # question, and a live one is already doing what the caller wants.
        raise HTTPException(
            status_code=409,
            detail={
                "code": CHAT_TURN_IN_FLIGHT,
                "message": "That answer is not waiting to be retried.",
            },
        )

    # Same one-at-a-time rule `_start_turn` applies: retrying an older failure
    # while a newer question is still being answered would put two runs on one
    # thread, each holding an agent slot.
    if any(
        m.status in (STATUS_QUEUED, STATUS_PENDING)
        for m in await _messages_of(session, conversation_id)
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": CHAT_TURN_IN_FLIGHT,
                "message": "Koutsi is still answering your last question.",
            },
        )

    # The daily cap still applies — a retry is a real run. It is the
    # *per-conversation* one that must not: the row already exists and is
    # already counted, so charging it again would make a thread at its limit
    # impossible to repair.
    await _enforce_daily_budget(session, athlete)

    now = datetime.now(timezone.utc)
    answer.status = STATUS_QUEUED
    answer.content = ""
    answer.progress = None
    answer.error_code = None
    answer.tool_names = None
    answer.prompt_tokens = None
    answer.completion_tokens = None
    answer.updated_at = now
    await session.commit()

    asyncio.create_task(
        run_chat_turn_bg(ctx.user_id, conversation_id, answer.id, body.locale)
    )
    return ChatMessageResponse.model_validate(answer)
