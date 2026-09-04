"""Shared plumbing for the streaming, DB-backed LLM analyses.

Three features stream a long-form coaching answer from an OpenAI-compatible
endpoint and persist it as it arrives — the activity analysis, the training
status, and the goal guidance. They differ only in the prompt they send and the
column they write; everything between is this module:

* :func:`stream_completion_events` — the SSE transport, one turn, in the
  provider's own vocabulary: text deltas and (issue #43) tool-call deltas.
* :func:`stream_chat_completion` — the single-shot wrapper over it, plus the
  config resolution and instance-context injection every athlete-facing call
  shares.
* :func:`stream_into_db` — the drain loop: buffer chunks, commit every 500 ms so
  a poll mid-generation shows live progress, settle the final status, and record
  instance-paid token usage whatever the outcome.
* :func:`failure_recovery` — the outer net that clears a stuck ``pending`` when
  the failure happened before the drain loop could own it.

The agent loop (issue #43) is layered *on* this rather than beside it:
:mod:`backend.app.services.llm_agent` drives many turns through
:func:`stream_completion_events` into the same :func:`stream_into_db`,
interleaved with :class:`AgentProgress` markers. The DB-backed
streaming-plus-polling design exists so a slow local model never dies on a
request timeout and a page reload never loses a generation.

Lives apart from ``llm_client`` because it needs ``llm_access`` (usage
recording), which imports ``llm_client`` in turn.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional, Union

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.ssrf import check_url_safe, guarded_async_client
from ..db.registry import _RegistrySessionLocal
from ..db.user_session import get_user_session_factory
from ..models.registry_orm import InstanceSettings
from ..models.user_orm import Athlete
from .llm_access import record_llm_usage, usage_from_sse_data
from .llm_client import (
    ResolvedLlm,
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


# ── One turn's events ───────────────────────────────────────────────────────
#
# The transport speaks in deltas because the provider does. Assembling them into
# whole tool calls is the agent loop's job (issue #43), not this module's: a
# partial `arguments` fragment is meaningless on its own and the reassembly rule
# is a property of the OpenAI chat-completions dialect, which is exactly the
# thing `llm_agent`'s adapter seam is there to isolate.


@dataclass(frozen=True)
class TextDelta:
    """A fragment of assistant prose."""

    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    """A fragment of a tool call, keyed by its position in the turn's list.

    Providers stream a tool call across several chunks: the first carries the
    ``index``, ``id`` and function name, later ones append to ``arguments``. Only
    ``index`` is reliably present on every fragment, which is why it — not the
    id — is what the reassembly keys on.
    """

    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None


@dataclass(frozen=True)
class AgentProgress:
    """What the agentic coach is doing, as a code from a fixed vocabulary.

    Yielded into :func:`stream_into_db` alongside text so the frontend's poll has
    something to show during the tool rounds, which produce no prose at all. See
    :mod:`backend.app.services.llm_agent` for the vocabulary and why it is codes
    rather than model-authored sentences.
    """

    code: Optional[str]


#: What a stream drained by :func:`stream_into_db` may yield. Plain ``str`` is
#: the single-shot case, unchanged; the agent loop adds progress markers.
StreamItem = Union[str, AgentProgress]


@dataclass(frozen=True)
class StreamSetup:
    """The resolved config plus the instance's house style, for one athlete.

    Split out of :func:`stream_chat_completion` so the agent loop resolves the
    athlete's LLM exactly the same way a single-shot call does — BYOK first,
    the no-mixing rule, the allow-list — instead of growing a second, subtly
    different resolution path.
    """

    cfg: ResolvedLlm
    #: The hoster's ``llm_analysis_context``, stripped, or ``None``. It is a
    #: *system* message, and the agent loop must resend it on every turn: an
    #: instance's house style is not something the model may forget three tool
    #: rounds in.
    analysis_context: Optional[str] = None

    def system_messages(self, system_prompt: str) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if self.analysis_context:
            messages.append({"role": "system", "content": self.analysis_context})
        return messages


async def resolve_stream_setup(
    athlete: Athlete,
    user_id: str,
    *,
    usage_out: dict | None = None,
) -> StreamSetup:
    """Resolve the athlete's effective LLM config and the instance house style.

    Priority is the usual one: the athlete's own BYOK server if configured, else
    their selected instance preset (``app_settings["llm_model"]``), else the
    instance default (the first preset). When ``usage_out`` is given, the config
    is stashed in it under ``"cfg"`` so usage recording can attribute the call.
    """
    async with _RegistrySessionLocal() as reg:
        instance: InstanceSettings | None = (
            await reg.execute(select(InstanceSettings).limit(1))
        ).scalar_one_or_none()

    cfg = resolve_llm_config(athlete, instance, user_id)
    if usage_out is not None:
        usage_out["cfg"] = cfg

    analysis_context = getattr(instance, "llm_analysis_context", None)
    analysis_context = analysis_context.strip() if analysis_context else None
    return StreamSetup(cfg=cfg, analysis_context=analysis_context or None)


async def stream_completion_events(
    cfg: ResolvedLlm,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: Any | None = None,
    usage_out: dict | None = None,
) -> AsyncIterator[Union[TextDelta, ToolCallDelta]]:
    """Stream one chat completion, yielding text and tool-call deltas.

    The transport and only the transport: no config resolution, no message
    building, no reassembly. ``messages`` is sent verbatim, which lets the agent
    loop replay a growing conversation on every turn.

    ``tools`` and ``tool_choice`` are sent only when given, so a provider that
    has never heard of function calling sees the pre-#43 request. One that
    rejects the ``tools`` *param* raises :class:`httpx.HTTPStatusError`, which
    :func:`~backend.app.services.llm_client.is_tool_calling_unsupported_error`
    recognises.

    With ``usage_out``, the trailing ``stream_options.include_usage`` chunk lands
    under ``"usage"`` — last non-null wins within one call. Summing across calls
    is the agent loop's job
    (:func:`~backend.app.services.llm_access.merge_usage`).
    """
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    check_url_safe(url)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    headers = merge_llm_headers(headers, cfg.extra_headers)

    def _payload(include_usage: bool) -> dict:
        base: dict = {
            "model": cfg.model,
            "messages": messages,
            **temperature_param(),
            "stream": True,
        }
        if tools:
            base["tools"] = tools
            if tool_choice is not None:
                base["tool_choice"] = tool_choice
        if include_usage:
            base["stream_options"] = {"include_usage": True}
        return apply_body_extras(base, cfg.extra_body)

    async with guarded_async_client(timeout=_STREAM_TIMEOUT) as client:
        # Ask for a trailing usage chunk; retry once without it if the upstream
        # rejects stream_options (Ollama-family tolerance). A provider refusing
        # `tools` fails both attempts and surfaces through raise_for_llm_status
        # with its own body intact, which is what the tool-support detection
        # needs to read.
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
                    delta = chunk["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if content:
                    yield TextDelta(content)
                for fragment in _tool_call_deltas(delta):
                    yield fragment
        finally:
            await cm.__aexit__(None, None, None)


def _tool_call_deltas(delta: dict) -> list[ToolCallDelta]:
    """Read the ``tool_calls`` fragments out of one streamed delta.

    Defensive throughout: a provider that emits function calling *badly* is one
    of the three populations issue #43 has to survive, and a malformed fragment
    must not kill the run. Anything unreadable is dropped here and simply never
    becomes a call.
    """
    raw = delta.get("tool_calls")
    if not isinstance(raw, list):
        return []
    out: list[ToolCallDelta] = []
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        if not isinstance(index, int):
            # Some servers omit `index` when they only ever emit one call per
            # chunk. Falling back to the position keeps those usable.
            index = position
        function = entry.get("function")
        function = function if isinstance(function, dict) else {}
        name = function.get("name")
        arguments = function.get("arguments")
        call_id = entry.get("id")
        out.append(
            ToolCallDelta(
                index=index,
                id=str(call_id) if isinstance(call_id, str) and call_id else None,
                name=str(name) if isinstance(name, str) and name else None,
                arguments=arguments if isinstance(arguments, str) else None,
            )
        )
    return out


async def stream_chat_completion(
    athlete: Athlete,
    user_id: str,
    *,
    system_prompt: str,
    user_prompt: str,
    usage_out: dict | None = None,
) -> AsyncIterator[str]:
    """Yield assistant text chunks from an athlete-facing streaming completion.

    Resolves the athlete's effective LLM config: their own BYOK server if
    configured, else their selected instance preset
    (``app_settings["llm_model"]``), else the instance default (first preset).
    The instance's ``llm_analysis_context`` is injected as a second system
    message when set.

    ``usage_out``, when given, is populated with ``{"cfg", "usage"}`` so the
    caller can record instance-paid usage (issue #9); it stays absent when the
    upstream omits the trailing chunk. BYOK calls resolve to ``source == "user"``
    and are skipped by usage recording.

    The **non-agentic path** — the blob prompt in one shot, and not legacy: it
    stays the answer for providers that cannot call tools and for bulk imports.
    """
    setup = await resolve_stream_setup(athlete, user_id, usage_out=usage_out)
    messages = setup.system_messages(system_prompt)
    messages.append({"role": "user", "content": user_prompt})

    async for event in stream_completion_events(
        setup.cfg, messages, usage_out=usage_out
    ):
        if isinstance(event, TextDelta):
            yield event.text


async def stream_into_db(
    session: AsyncSession,
    make_stream: Callable[[dict], AsyncIterator[StreamItem]],
    *,
    on_progress: Callable[[str], None],
    on_done: Callable[[str], None],
    on_error: Callable[[], None],
    on_step: Callable[[Optional[str]], None] | None = None,
    user_id: str,
    feature: str,
    label: str,
) -> None:
    """Drain a chat-completion stream into the DB, settling the status either way.

    ``make_stream`` is handed ``usage_out`` to pass down to
    :func:`stream_chat_completion`; the usage it collects is recorded on the way
    out whether the stream succeeded, failed or produced nothing. An agentic run
    sums several calls into that one dict
    (:func:`~backend.app.services.llm_access.merge_usage`), so the whole run's
    cost is recorded rather than its last turn's.

    The callbacks own the persistence: ``on_progress`` writes partial text
    (committed every ~500 ms so a mid-generation poll shows progress),
    ``on_done`` writes the final text and flips the status, ``on_error`` marks
    the failure. Each is followed by a commit here, so none should commit itself.

    ``on_step`` (issue #43) receives progress *codes* from
    :class:`AgentProgress` markers, committed as they arrive rather than on the
    500 ms text cadence. A ``None`` code clears the step, which the loop sends
    the moment real prose starts.
    """
    usage_out: dict = {}
    started = time.monotonic()
    buffer: list[str] = []
    accumulated = ""
    last_flush = time.monotonic()

    def _step(code: Optional[str]) -> None:
        if on_step is not None:
            on_step(code)

    try:
        async for item in make_stream(usage_out):
            if isinstance(item, AgentProgress):
                _step(item.code)
                await session.commit()
                continue
            buffer.append(item)
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

    :func:`stream_into_db` settles the status itself, but only once running: if
    opening the session — or a context query ahead of it — is what failed,
    nothing clears the ``pending`` the API wrote. A *fresh* session is opened
    here because the original may be exactly what went wrong.
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
