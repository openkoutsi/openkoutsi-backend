"""Shared FastAPI dependency helpers for token-scoped route handlers."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import UserContext, get_current_user
from backend.app.core.encryption import set_user_encryption_context
from backend.app.db.user_session import get_user_session_factory


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
