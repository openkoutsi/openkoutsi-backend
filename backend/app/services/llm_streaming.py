"""Shared plumbing for the streaming, DB-backed LLM analyses.

Three features stream a long-form coaching answer from an OpenAI-compatible
endpoint and persist it as it arrives — the activity analysis, the training
status, and the goal guidance. They differ only in the prompt they send and the
column they write; everything between is this module:

* :func:`stream_chat_completion` — the SSE transport, plus the config
  resolution and instance-context injection every athlete-facing call shares.
* :func:`stream_into_db` — the drain loop: buffer chunks, commit every 500 ms so
  a poll mid-generation shows live progress, settle the final status, and record
  instance-paid token usage whatever the outcome.
* :func:`failure_recovery` — the outer net that clears a stuck ``pending`` when
  the failure happened before the drain loop could own it.

Lives apart from ``llm_client`` because it needs ``llm_access`` (usage
recording), which imports ``llm_client`` in turn.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.ssrf import check_url_safe
from ..db.registry import _RegistrySessionLocal
from ..db.user_session import get_user_session_factory
from ..models.registry_orm import InstanceSettings
from ..models.user_orm import Athlete
from .llm_access import record_llm_usage, usage_from_sse_data
from .llm_client import (
    apply_body_extras,
    merge_llm_headers,
    raise_for_llm_status,
    resolve_llm_config,
    temperature_param,
)

log = logging.getLogger(__name__)

# Local models can take several minutes; generous but finite.
_STREAM_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

# How often the accumulated text is committed mid-stream. Short enough that the
# frontend's poll shows live progress, long enough not to hammer the DB.
_FLUSH_INTERVAL_S = 0.5


async def stream_chat_completion(
    athlete: Athlete,
    user_id: str,
    *,
    system_prompt: str,
    user_prompt: str,
    usage_out: dict | None = None,
) -> AsyncIterator[str]:
    """Yield assistant text chunks from an athlete-facing streaming completion.

    Resolves the athlete's effective LLM config in the usual priority order:
    their own BYOK server if configured, else their selected instance preset
    (``app_settings["llm_model"]``), else the instance default (first preset).
    The instance's ``llm_analysis_context`` — the hoster's house style — is
    injected as a second system message when set.

    When ``usage_out`` is provided it is populated with ``{"cfg", "usage"}`` so
    the caller can record instance-paid token usage (issue #9). ``usage`` is the
    trailing ``stream_options.include_usage`` chunk, and stays absent when the
    upstream omits it. BYOK calls resolve to ``source == "user"`` and are skipped
    by usage recording (the hoster pays nothing for them).
    """
    async with _RegistrySessionLocal() as reg:
        instance: InstanceSettings | None = (
            await reg.execute(select(InstanceSettings).limit(1))
        ).scalar_one_or_none()

    cfg = resolve_llm_config(athlete, instance, user_id)
    if usage_out is not None:
        usage_out["cfg"] = cfg

    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    check_url_safe(url)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    headers = merge_llm_headers(headers, cfg.extra_headers)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    analysis_context = getattr(instance, "llm_analysis_context", None)
    if analysis_context and analysis_context.strip():
        messages.append({"role": "system", "content": analysis_context.strip()})
    messages.append({"role": "user", "content": user_prompt})

    def _payload(include_usage: bool) -> dict:
        base: dict = {
            "model": cfg.model,
            "messages": messages,
            **temperature_param(),
            "stream": True,
        }
        if include_usage:
            base["stream_options"] = {"include_usage": True}
        return apply_body_extras(base, cfg.extra_body)

    async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
        # Ask for a trailing usage chunk; retry once without it if the upstream
        # rejects stream_options (Ollama-family tolerance).
        cm = client.stream("POST", url, json=_payload(True), headers=headers)
        resp = await cm.__aenter__()
        if getattr(resp, "is_error", False):
            await cm.__aexit__(None, None, None)
            cm = client.stream("POST", url, json=_payload(False), headers=headers)
            resp = await cm.__aenter__()
        try:
            await raise_for_llm_status(resp, url)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                if usage_out is not None:
                    usage = usage_from_sse_data(data)
                    if usage is not None:
                        usage_out["usage"] = usage
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        finally:
            await cm.__aexit__(None, None, None)


async def stream_into_db(
    session: AsyncSession,
    make_stream: Callable[[dict], AsyncIterator[str]],
    *,
    on_progress: Callable[[str], None],
    on_done: Callable[[str], None],
    on_error: Callable[[], None],
    user_id: str,
    feature: str,
    label: str,
) -> None:
    """Drain a chat-completion stream into the DB, settling the status either way.

    ``make_stream`` is handed the ``usage_out`` dict to pass down to
    :func:`stream_chat_completion`; the token usage it collects is recorded on
    the way out whether the stream succeeded, failed, or produced nothing.

    The three callbacks own the persistence: ``on_progress`` writes the partial
    text (committed every ~500 ms so a mid-generation poll shows real progress),
    ``on_done`` writes the final text and flips the status, and ``on_error``
    marks the failure. Each is followed by a commit here, so none of them should
    commit itself. ``label`` names the work in the log lines.
    """
    usage_out: dict = {}
    started = time.monotonic()
    buffer: list[str] = []
    accumulated = ""
    last_flush = time.monotonic()

    try:
        async for chunk in make_stream(usage_out):
            buffer.append(chunk)
            if time.monotonic() - last_flush >= _FLUSH_INTERVAL_S:
                accumulated += "".join(buffer)
                buffer.clear()
                last_flush = time.monotonic()
                on_progress(accumulated)
                await session.commit()

        accumulated += "".join(buffer)
        on_done(accumulated)
        await session.commit()
        log.info("%s complete", label)

    except Exception:
        log.exception("%s failed", label)
        on_error()
        await session.commit()
    finally:
        # Record instance-paid token usage (issue #9). Fire-and-forget; a
        # failure here never affects the analysis result.
        cfg = usage_out.get("cfg")
        if cfg is not None:
            await record_llm_usage(
                user_id=user_id,
                feature=feature,
                cfg=cfg,
                usage=usage_out.get("usage"),
                duration_ms=int((time.monotonic() - started) * 1000),
            )


@asynccontextmanager
async def failure_recovery(
    user_id: str,
    label: str,
    mark_error: Callable[[AsyncSession], Awaitable[None]],
):
    """Clear a stuck ``pending`` when the failure landed outside the drain loop.

    :func:`stream_into_db` settles the status itself, but it can only do that
    once it is running: if opening the session — or one of the context queries
    ahead of it — is what failed, nothing has cleared the ``pending`` the API
    wrote, and the user watches a spinner forever. A *fresh* session is opened
    here because the original one may be exactly what went wrong.
    """
    try:
        yield
    except Exception:
        log.exception("%s failed outside the inner handler", label)
        try:
            async with get_user_session_factory(user_id)() as recovery_session:
                await mark_error(recovery_session)
                await recovery_session.commit()
        except Exception:
            log.exception(
                "%s recovery session also failed — status may remain stuck", label
            )
