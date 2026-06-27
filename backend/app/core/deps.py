"""Shared FastAPI dependency helpers for team-scoped route handlers."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import TeamContext, get_current_user
from backend.app.core.encryption import set_team_encryption_context
from backend.app.db.team_session import get_team_session_factory


class _TeamSession:
    """Combined dependency: validates JWT, sets team encryption context, yields DB session.

    Usage in route handlers:
        async def endpoint(ctx_session = Depends(get_ctx_and_session)):
            ctx, session = ctx_session
            ...
    """

    async def __call__(
        self,
        ctx: TeamContext = Depends(get_current_user),
    ) -> AsyncGenerator[tuple[TeamContext, AsyncSession], None]:
        set_team_encryption_context(ctx.team_id)
        async with get_team_session_factory(ctx.team_id)() as session:
            yield ctx, session


get_ctx_and_session = _TeamSession()
