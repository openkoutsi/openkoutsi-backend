"""Shared FastAPI dependency helpers for token-scoped route handlers."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import UserContext, get_current_user
from backend.app.core.encryption import set_user_encryption_context
from backend.app.db.user_session import get_user_session_factory
from backend.app.models.user_orm import Athlete


class _UserSession:
    """Combined dependency: validates JWT, sets user encryption context, yields DB session.

    Usage in route handlers:
        async def endpoint(ctx_session = Depends(get_ctx_and_session)):
            ctx, session = ctx_session
            ...
    """

    async def __call__(
        self,
        ctx: UserContext = Depends(get_current_user),
    ) -> AsyncGenerator[tuple[UserContext, AsyncSession], None]:
        set_user_encryption_context(ctx.user_id)
        async with get_user_session_factory(ctx.user_id)() as session:
            yield ctx, session


get_ctx_and_session = _UserSession()


async def load_athlete(global_user_id: str, session: AsyncSession) -> Athlete:
    """The caller's athlete profile, or a 404 if they have none.

    Kept as a plain function alongside the dependency below for the handful of
    call sites that need it against a session they opened themselves (background
    tasks, the OAuth callbacks) rather than the request-scoped one.
    """
    result = await session.execute(
        select(Athlete).where(Athlete.global_user_id == global_user_id)
    )
    athlete = result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")
    return athlete


async def get_ctx_session_athlete(
    ctx_session: tuple[UserContext, AsyncSession] = Depends(get_ctx_and_session),
) -> tuple[UserContext, AsyncSession, Athlete]:
    """``get_ctx_and_session`` plus the caller's athlete profile.

    The overwhelmingly common shape for an athlete-facing route: every handler
    that used to open with its own ``_get_athlete`` lookup takes this instead.

    Usage in route handlers:
        async def endpoint(ctx_athlete = Depends(get_ctx_session_athlete)):
            ctx, session, athlete = ctx_athlete
            ...
    """
    ctx, session = ctx_session
    return ctx, session, await load_athlete(ctx.user_id, session)
