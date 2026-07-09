"""LLM subscription gating + per-user token-usage recording (issue #9).

Two concerns live here:

1. **The gate** (:func:`check_llm_access`): an opt-in, per-instance switch
   (``InstanceSettings.llm_requires_subscription``). When off (default), LLM
   features work exactly as before. When on, only users with an active LLM-access
   entitlement — or users running their own LLM (BYOK, #8) — may use the
   instance's LLM credentials; everyone else is denied with a machine-readable
   ``llm_subscription_required`` code the frontend turns into an upsell.

2. **Usage recording** (:func:`record_llm_usage`): fire-and-forget accounting of
   every **instance-paid** LLM call, written to the dedicated usage DB. BYOK
   calls (``ResolvedLlm.source == "user"``) are never recorded — the hoster pays
   nothing for them. Input and output tokens are stored separately.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple
from urllib.parse import urlparse

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.usage import usage_session_factory
from ..models.registry_orm import InstanceSettings, LlmEntitlement
from ..models.usage_orm import LlmUsage
from .llm_client import ResolvedLlm, preset_map

log = logging.getLogger(__name__)

# Machine-readable code carried in the 403 ``detail`` so every frontend consumer
# branches on a stable key, never on message text.
LLM_SUBSCRIPTION_REQUIRED = "llm_subscription_required"

# The five gated feature areas — also the ``llm_usage.feature`` column values.
Feature = Literal[
    "chat", "plan_generate", "workout_generate", "activity_analysis", "training_status"
]


class LlmAccess(NamedTuple):
    """The result of an access check.

    * ``ungated`` — the gate is off; resolution is unchanged (BYOK → instance).
    * ``byok`` — gated, but the user runs their own LLM; the instance credentials
      must never be touched (resolve with ``allow_instance_fallback=False``).
    * ``entitled`` — gated, no BYOK, but the user holds an active entitlement;
      instance presets are usable.
    * ``none`` — gated, no BYOK, no entitlement; denied (``reason`` is set).
    """

    allowed: bool
    mode: Literal["ungated", "byok", "entitled", "none"]
    reason: str | None = None


# ── Entitlement predicate ───────────────────────────────────────────────────


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def is_entitled(ent: LlmEntitlement | None, now: datetime | None = None) -> bool:
    """``status == active AND starts_at <= now AND (expires_at IS NULL OR expires_at > now)``."""
    if ent is None or ent.status != "active":
        return False
    now = now or datetime.now(timezone.utc)
    starts = _aware(ent.starts_at)
    if starts is not None and starts > now:
        return False
    expires = _aware(ent.expires_at)
    if expires is not None and expires <= now:
        return False
    return True


async def get_entitlement(
    user_id: str, registry_session: AsyncSession
) -> LlmEntitlement | None:
    result = await registry_session.execute(
        select(LlmEntitlement).where(LlmEntitlement.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def user_is_entitled(user_id: str, registry_session: AsyncSession) -> bool:
    return is_entitled(await get_entitlement(user_id, registry_session))


# ── The gate ────────────────────────────────────────────────────────────────


def byok_active(athlete: Any) -> bool:
    """Does this athlete run their own LLM server (the #8 no-mixing signal)?

    True when the single ``llm_base_url`` field is set, or any athlete-level
    preset carries its own ``base_url``. Mirrors the BYOK detection in
    :func:`resolve_llm` so the gate agrees with what the call will actually do.
    """
    settings = (getattr(athlete, "app_settings", None) or {}) if athlete is not None else {}
    if str(settings.get("llm_base_url") or "").strip():
        return True
    for entry in preset_map(settings.get("llm_models")).values():
        if str(entry.get("base_url") or "").strip():
            return True
    return False


async def check_llm_access(
    ctx: Any,
    athlete: Any,
    instance: InstanceSettings | None,
    registry_session: AsyncSession,
) -> LlmAccess:
    """Decide whether ``ctx``'s user may use LLM features on this instance.

    See :class:`LlmAccess` for the modes. Admins are **not** implicitly exempt —
    they can grant themselves an entitlement like anyone else.
    """
    gated = bool(getattr(instance, "llm_requires_subscription", False))
    if not gated:
        return LlmAccess(True, "ungated")

    if byok_active(athlete):
        return LlmAccess(True, "byok")

    if await user_is_entitled(ctx.user_id, registry_session):
        return LlmAccess(True, "entitled")

    return LlmAccess(False, "none", LLM_SUBSCRIPTION_REQUIRED)


async def auto_analysis_allowed(user_id: str, athlete: Any) -> bool:
    """Gate for the always-instance-paid background analyzers (issue #9).

    Self-contained (opens its own short-lived registry session) so it can be
    called from any spawn site — activity upload, provider sync, the auto
    training-status hook — before an ``analyze_*_bg`` task is created. Returns
    True when the gate is off, or the caller is entitled (BYOK users are
    "allowed" too; the analyzers always use instance credentials regardless).
    """
    from ..db.registry import _RegistrySessionLocal

    async with _RegistrySessionLocal() as reg:
        result = await reg.execute(select(InstanceSettings).limit(1))
        instance = result.scalar_one_or_none()
        ctx = SimpleNamespace(user_id=user_id)
        access = await check_llm_access(ctx, athlete, instance, reg)
        return access.allowed


def subscription_required_error() -> HTTPException:
    """The canonical 403 for a denied LLM request, with a structured ``detail``."""
    return HTTPException(
        status_code=403,
        detail={
            "code": LLM_SUBSCRIPTION_REQUIRED,
            "message": (
                "AI features on this server require a subscription. "
                "You can also connect your own AI model in Settings → AI / LLM."
            ),
        },
    )


# ── Usage recording ─────────────────────────────────────────────────────────


def parse_usage(usage: Any) -> tuple[int | None, int | None, int | None]:
    """Extract ``(prompt, completion, total)`` from an OpenAI ``usage`` object.

    Returns ``(None, None, None)`` when usage is absent — some servers (older
    Ollama) omit it even when asked; such calls are recorded with nulls, never
    estimated. ``total`` is derived from the parts only when the parts are known.
    """
    if not isinstance(usage, dict):
        return None, None, None

    def _int(key: str) -> int | None:
        val = usage.get(key)
        return int(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None

    prompt = _int("prompt_tokens")
    completion = _int("completion_tokens")
    total = _int("total_tokens")
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)
    return prompt, completion, total


def provider_label(cfg: ResolvedLlm) -> str | None:
    """Which provider served the call — the resolved base-URL host (issue #9).

    ``ResolvedLlm`` doesn't carry the preset's label, so the host is used as the
    provider identity (``api.openai.com``, ``localhost``, …), recorded alongside
    ``model``.
    """
    if not cfg.base_url:
        return None
    try:
        host = urlparse(cfg.base_url).hostname
    except ValueError:
        return None
    return host or None


def usage_from_sse_data(data: str) -> dict | None:
    """Return the ``usage`` object from one SSE ``data:`` payload, if non-null.

    The final chunk of a stream with ``stream_options.include_usage`` carries a
    populated ``usage``; earlier chunks carry ``null`` (or omit it). Malformed
    JSON is ignored.
    """
    data = data.strip()
    if not data or data == "[DONE]":
        return None
    try:
        chunk = json.loads(data)
    except (ValueError, TypeError):
        return None
    usage = chunk.get("usage") if isinstance(chunk, dict) else None
    return usage if isinstance(usage, dict) else None


async def record_llm_usage(
    *,
    user_id: str,
    feature: Feature | str,
    cfg: ResolvedLlm,
    usage: Any,
    duration_ms: int | None = None,
) -> None:
    """Record one **instance-paid** LLM call in the dedicated usage DB.

    Fire-and-forget: opens its own short-lived session (streaming generators
    finish outside the request session) and never raises into the caller — a
    failure logs a warning and is swallowed so it can't break the user's request.

    **BYOK calls are skipped entirely** (``cfg.source == "user"``): the user pays
    their own provider, so the hoster has no cost to account for.
    """
    if cfg.source == "user":
        return

    prompt_tokens, completion_tokens, total_tokens = parse_usage(usage)
    key_source = cfg.key_source if cfg.key_source in ("instance", "none") else "none"

    try:
        async with usage_session_factory()() as session:
            session.add(
                LlmUsage(
                    user_id=user_id,
                    feature=str(feature),
                    provider=provider_label(cfg),
                    model=cfg.model or None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    key_source=key_source,
                    duration_ms=duration_ms,
                )
            )
            await session.commit()
    except Exception:  # pragma: no cover - defensive; usage must never break a request
        log.warning(
            "Failed to record LLM usage (user=%s feature=%s)",
            user_id,
            feature,
            exc_info=True,
        )
