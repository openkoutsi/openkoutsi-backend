"""User-scoped in-app message inbox.

Endpoints operate on the current user's own mailbox (their per-user DB),
resolved from the access token. Any authenticated user may call these; the
frontend only surfaces the inbox UI for admins for now.
"""
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import UserContext, get_current_user
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.message_orm import Message
from backend.app.schemas.messages import MessageResponse, UnreadCountResponse
from backend.app.schemas.pagination import Page, PageParams, paginate_params

router = APIRouter(prefix="/messages", tags=["messages"])


async def get_user_message_session(
    ctx: UserContext = Depends(get_current_user),
) -> AsyncGenerator[tuple[UserContext, AsyncSession], None]:
    await init_user_db(ctx.user_id)
    async with get_user_session_factory(ctx.user_id)() as session:
        yield ctx, session


async def _get_message(message_id: str, session: AsyncSession) -> Message:
    result = await session.execute(select(Message).where(Message.id == message_id))
    message = result.scalar_one_or_none()
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


@router.get("", response_model=Page[MessageResponse],
            operation_id="listMessages", summary="List messages")
async def list_messages(
    ctx_session=Depends(get_user_message_session),
    params: PageParams = Depends(paginate_params),
):
    _, session = ctx_session
    total = (await session.execute(select(func.count()).select_from(Message))).scalar_one()
    result = await session.execute(
        select(Message)
        .order_by(Message.created_at.desc())
        .offset(params.offset)
        .limit(params.page_size)
    )
    items = [MessageResponse.model_validate(m) for m in result.scalars().all()]
    return Page.build(items, total, params.page, params.page_size)


@router.get("/unread-count", response_model=UnreadCountResponse)
async def unread_count(ctx_session=Depends(get_user_message_session)):
    _, session = ctx_session
    result = await session.execute(
        select(func.count()).select_from(Message).where(Message.read_at.is_(None))
    )
    return UnreadCountResponse(count=result.scalar_one())


@router.post("/read-all", status_code=204)
async def mark_all_read(ctx_session=Depends(get_user_message_session)):
    _, session = ctx_session
    now = datetime.now(timezone.utc)
    result = await session.execute(select(Message).where(Message.read_at.is_(None)))
    for message in result.scalars().all():
        message.read_at = now
    await session.commit()


@router.post("/{message_id}/read", response_model=MessageResponse)
async def mark_read(message_id: str, ctx_session=Depends(get_user_message_session)):
    _, session = ctx_session
    message = await _get_message(message_id, session)
    if message.read_at is None:
        message.read_at = datetime.now(timezone.utc)
        await session.commit()
    return message


@router.delete("/{message_id}", status_code=204)
async def delete_message(message_id: str, ctx_session=Depends(get_user_message_session)):
    _, session = ctx_session
    message = await _get_message(message_id, session)
    await session.delete(message)  # hard delete — really removed
    await session.commit()
