"""LLM proxy endpoint.

Security model
--------------
Users configure their own LLM endpoint (base URL, model, API key) in
Settings → AI / LLM.  The API key is encrypted server-side with AES-256
(Fernet) using a per-user HKDF-derived key — see file_encryption.py.  It is
stored in ``athlete.app_settings['llm_api_key_enc']`` and is **never**
returned to the browser after being saved.

When an LLM call is needed the server decrypts the key in-memory, adds it to
the outbound request headers, and proxies the OpenAI-compatible request to
the user's configured endpoint.  From the browser's perspective the request
goes to ``/api/llm/chat`` on the same origin, so:

* No API key is ever transmitted to the frontend.
* The browser's Content-Security-Policy (``connect-src 'self' ...``) already
  permits calls to the API origin — no extra CSP rules required.
* The LLM endpoint is called server-to-server, so mixed-content (HTTP ↔ HTTPS)
  restrictions in the browser do not apply.

SSRF mitigations
----------------
Because any authenticated user can set an arbitrary base URL, the server could
be used as a proxy to reach internal services.  The following defences are
applied:

1. Only ``http://`` and ``https://`` schemes are accepted.
2. The hostname is resolved to an IP address before the request is made.  If
   the resolved address is link-local (169.254.0.0/16, fe80::/10) — the range
   used by cloud-provider metadata services — the request is rejected.
   Loopback (127.x / ::1) and private RFC-1918 / RFC-4193 ranges are allowed
   so that Ollama running on localhost or a LAN machine works normally.
3. HTTP redirects are disabled so a redirect cannot bounce the server from a
   safe public host to an internal address.
4. The connection is made to the pre-resolved IP, not by passing the hostname
   to httpx again, to prevent trivial DNS rebinding.

Note: a single layer of DNS-based SSRF protection is not proof against all
DNS-rebinding attacks.  If your deployment is multi-tenant and users are not
fully trusted, consider restricting who can save an LLM base URL (e.g. admin
only) via an out-of-band policy.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import UserContext
from backend.app.core.config import settings
from backend.app.core.deps import get_ctx_and_session
from backend.app.core.file_encryption import decrypt_secret
from backend.app.core.limiter import limiter
from backend.app.core.ssrf import check_url_safe
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import InstanceSettings
from backend.app.models.user_orm import Athlete
from backend.app.services.llm_access import (
    LlmAccess,
    check_llm_access,
    get_entitlement,
    is_entitled,
    record_llm_usage,
    subscription_required_error,
    usage_from_sse_data,
)
from backend.app.services.llm_client import (
    LLM_ERROR_STATUS,
    LlmConfigError,
    apply_body_extras,
    merge_llm_headers,
    preset_map,
    resolve_llm,
    resolve_llm_config,
    temperature_param,
)


def _http_from_llm_config_error(exc: LlmConfigError) -> HTTPException:
    return HTTPException(status_code=LLM_ERROR_STATUS.get(exc.code, 400), detail=str(exc))


async def _load_instance_settings(registry_session: AsyncSession) -> InstanceSettings | None:
    result = await registry_session.execute(select(InstanceSettings).limit(1))
    return result.scalar_one_or_none()

log = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])

# Maximum bytes accepted from an upstream LLM response.
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024  # 32 MB

# ── Request schema ─────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class LlmChatRequest(BaseModel):
    messages: list[ChatMessage]
    # Optional: when omitted, the temperature parameter is left out of the
    # upstream request entirely so the model uses its own default. This keeps
    # thinking-enabled models — which reject any temperature other than 1 —
    # working through the proxy.
    temperature: Optional[float] = None
    stream: bool = False
    model: Optional[str] = None


# ── LLM config helper ──────────────────────────────────────────────────────


@router.get("/servers")
async def get_allowed_servers(ctx_session=Depends(get_ctx_and_session)):
    """Return the list of LLM base URLs the admin has allow-listed."""
    return {"servers": settings.llm_allowed_servers_list}


class LlmModelOption(BaseModel):
    name: str  # stable identifier / selection value
    label: str  # human-friendly display name


class LlmModelsResponse(BaseModel):
    models: list[LlmModelOption]
    selected: Optional[str] = None


@router.get("/models", response_model=LlmModelsResponse)
async def list_llm_models(
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Return the presets this user may select and their current selection.

    Each option carries the stable ``name`` (the selection value) and a
    human-friendly ``label`` for display. The list is the instance's presets
    overlaid by the user's personal presets; falls back to the single configured
    model name when no presets exist.
    """
    ctx, session = ctx_session

    result = await session.execute(select(Athlete).where(Athlete.global_user_id == ctx.user_id))
    athlete = result.scalar_one_or_none()
    athlete_settings = (athlete.app_settings if athlete else None) or {}

    instance = await _load_instance_settings(registry_session)

    presets: dict[str, dict] = {
        **preset_map(getattr(instance, "llm_models", None)),
        **preset_map(athlete_settings.get("llm_models")),
    }
    options: list[LlmModelOption] = [
        LlmModelOption(name=name, label=str(entry.get("label") or name))
        for name, entry in presets.items()
    ]

    # Mirror resolve_llm's selection order so `selected` reflects the model that
    # would actually be used: saved choice → first preset (the default).
    selected = (athlete_settings.get("llm_model") or "").strip()
    if not selected and presets:
        selected = next(iter(presets))

    # Make sure the effective selection is always offered, even if it is a lone
    # legacy single-model configuration not present in any preset list.
    if selected and selected not in presets:
        options.append(LlmModelOption(name=selected, label=selected))

    return LlmModelsResponse(models=options, selected=selected or None)


class LlmEntitlementSummary(BaseModel):
    status: str
    expires_at: Optional[str] = None


class LlmAccessResponse(BaseModel):
    """The frontend's single source of truth for LLM access state (issue #9).

    ``gated`` reflects the instance switch; ``mode`` is how the caller is (or
    would be) served: ``ungated`` (gate off), ``byok`` (own LLM), ``entitled``
    (active entitlement), or ``none`` (gated, no BYOK, no entitlement → the
    upsell). ``entitlement`` echoes the caller's own entitlement row if any.
    """

    gated: bool
    mode: str
    entitlement: Optional[LlmEntitlementSummary] = None


@router.get("/access", response_model=LlmAccessResponse)
async def get_llm_access(
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Report whether the caller may use LLM features, and by which route."""
    ctx, session = ctx_session

    result = await session.execute(select(Athlete).where(Athlete.global_user_id == ctx.user_id))
    athlete = result.scalar_one_or_none()

    instance = await _load_instance_settings(registry_session)
    access = await check_llm_access(ctx, athlete, instance, registry_session)

    ent = await get_entitlement(ctx.user_id, registry_session)
    ent_summary: Optional[LlmEntitlementSummary] = None
    if ent is not None:
        ent_summary = LlmEntitlementSummary(
            status="active" if is_entitled(ent) else ent.status,
            expires_at=ent.expires_at.isoformat() if ent.expires_at else None,
        )

    return LlmAccessResponse(
        gated=bool(getattr(instance, "llm_requires_subscription", False)),
        mode=access.mode,
        entitlement=ent_summary,
    )


class LlmTestResponse(BaseModel):
    ok: bool
    base_url: Optional[str] = None
    model_configured: Optional[str] = None
    prompt_sent: Optional[str] = None
    response_text: Optional[str] = None
    http_status: Optional[int] = None
    error: Optional[str] = None


# A minimal prompt used to verify the LLM answers. Kept tiny so the round-trip
# is cheap and fast regardless of the backing model.
_TEST_PROMPT = "Reply with a short greeting to confirm you are reachable."


async def _probe_llm_endpoint(
    base_url: str,
    model: str,
    api_key: str | None,
    *,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> LlmTestResponse:
    """Send a minimal chat completion and report whether the model replied.

    Shared by the admin (instance) and user (BYOK) test endpoints. Applies the
    SSRF guard, any extra headers and the selected model's body params, and maps
    connection/HTTP failures to a friendly :class:`LlmTestResponse`.
    """
    if not base_url:
        return LlmTestResponse(ok=False, error="No LLM base URL configured. Save a base URL first.")
    if not model:
        return LlmTestResponse(
            ok=False, base_url=base_url, error="No LLM model configured. Save a model first."
        )

    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        check_url_safe(chat_url)
    except HTTPException as exc:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=model, error=exc.detail)

    headers = merge_llm_headers({"Content-Type": "application/json"}, extra_headers)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = apply_body_extras(
        {
            "model": model,
            "messages": [{"role": "user", "content": _TEST_PROMPT}],
            "stream": False,
        },
        extra_body,
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            resp = await client.post(chat_url, headers=headers, json=payload, follow_redirects=False)
    except httpx.ConnectError as exc:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=model, prompt_sent=_TEST_PROMPT, error=f"Connection refused: {exc}")
    except httpx.TimeoutException:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=model, prompt_sent=_TEST_PROMPT, error="Connection timed out")
    except Exception as exc:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=model, prompt_sent=_TEST_PROMPT, error=str(exc))

    if resp.status_code == 401:
        return LlmTestResponse(
            ok=False, base_url=base_url, model_configured=model, prompt_sent=_TEST_PROMPT,
            http_status=resp.status_code, error="Authentication failed — check your API key",
        )
    if resp.status_code != 200:
        snippet = resp.text[:200] if resp.text else ""
        return LlmTestResponse(
            ok=False, base_url=base_url, model_configured=model, prompt_sent=_TEST_PROMPT,
            http_status=resp.status_code, error=f"HTTP {resp.status_code}: {snippet}",
        )

    try:
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        return LlmTestResponse(
            ok=False, base_url=base_url, model_configured=model, prompt_sent=_TEST_PROMPT,
            http_status=resp.status_code,
            error="The endpoint responded but the reply was not in the expected chat-completion format.",
        )

    reply_text = (reply or "").strip()
    if not reply_text:
        return LlmTestResponse(
            ok=False, base_url=base_url, model_configured=model, prompt_sent=_TEST_PROMPT,
            http_status=resp.status_code, error="The model returned an empty response.",
        )

    return LlmTestResponse(
        ok=True, base_url=base_url, model_configured=model, prompt_sent=_TEST_PROMPT,
        http_status=resp.status_code, response_text=reply_text[:500],
    )


@router.post("/test-connection", response_model=LlmTestResponse)
async def test_llm_connection(
    model: Optional[str] = None,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Test the instance's configured LLM connection. Admin-only.

    Sends a minimal "hello world" chat completion to the instance's saved LLM
    config and confirms the model replies with a usable response. Any configured
    extra headers and the selected model's body params are applied, so this also
    validates a zero-data-retention header or a thinking config. Pass ``model``
    to test a specific preset from the configured list instead of the default.
    """
    ctx, _ = ctx_session
    if not ctx.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    instance = await _load_instance_settings(registry_session)
    # Resolve the selected preset (its own base URL, model, key, headers, body).
    cfg = resolve_llm(instance=instance, requested_model=model)
    if not cfg.base_url:
        raise HTTPException(status_code=400, detail="No LLM base URL configured. Save a base URL first.")

    return await _probe_llm_endpoint(
        cfg.base_url, cfg.model, cfg.api_key,
        extra_headers=cfg.extra_headers, extra_body=cfg.extra_body,
    )


class LlmMyTestRequest(BaseModel):
    """Body for a user's own BYOK connection test.

    Any field may be given to override the saved athlete value so the Test
    button works before saving. When ``api_key`` is omitted (``None``) but a key
    is already saved, the saved key is used; pass an empty string to test
    keyless.
    """

    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None


@router.post("/test-my-connection", response_model=LlmTestResponse)
@limiter.limit("10/minute")
async def test_my_llm_connection(
    request: Request,
    body: LlmMyTestRequest = LlmMyTestRequest(),
    ctx_session=Depends(get_ctx_and_session),
):
    """Test the caller's own (BYOK) LLM connection. Any authenticated user.

    Body values override the athlete's saved config so the Test button works
    before saving; when ``api_key`` is omitted but a saved encrypted key exists
    it is decrypted and used. The tested URL must pass the SSRF guard and, when
    an allow-list is configured, be on it. Rate-limited since it triggers an
    outbound request on demand.
    """
    ctx, session = ctx_session

    result = await session.execute(select(Athlete).where(Athlete.global_user_id == ctx.user_id))
    athlete = result.scalar_one_or_none()
    saved = (athlete.app_settings if athlete else None) or {}

    base_url = (body.base_url if body.base_url is not None else saved.get("llm_base_url") or "").strip()
    model = (body.model if body.model is not None else saved.get("llm_model") or "").strip()

    if body.api_key is not None:
        api_key: str | None = body.api_key or None
    elif saved.get("llm_api_key_enc"):
        try:
            api_key = decrypt_secret(str(saved["llm_api_key_enc"]), ctx.user_id)
        except Exception:  # pragma: no cover - defensive; treat as no key
            api_key = None
    else:
        api_key = None

    if not base_url:
        raise HTTPException(status_code=400, detail="No LLM base URL configured. Enter a base URL first.")

    # BYOK URLs are subject to the admin allow-list (when set).
    allowed = settings.llm_allowed_servers_list
    if allowed and base_url.rstrip("/") not in {a.rstrip("/") for a in allowed}:
        raise HTTPException(
            status_code=403,
            detail="That LLM server is not in the server's allowed list.",
        )

    return await _probe_llm_endpoint(base_url, model, api_key)


# ── Endpoint ───────────────────────────────────────────────────────────────


@router.post("/chat")
async def llm_chat(
    body: LlmChatRequest,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Proxy an OpenAI-compatible chat completion to the user's LLM endpoint."""
    ctx, session = ctx_session

    result = await session.execute(select(Athlete).where(Athlete.global_user_id == ctx.user_id))
    athlete = result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")

    instance = await _load_instance_settings(registry_session)

    # Issue #9 gate: on a gated instance, deny non-entitled users without BYOK.
    access = await check_llm_access(ctx, athlete, instance, registry_session)
    if not access.allowed:
        raise subscription_required_error()

    try:
        cfg = resolve_llm_config(
            athlete,
            instance,
            ctx.user_id,
            requested_model=body.model,
            # In BYOK mode the instance credentials must never be touched.
            allow_instance_fallback=(access.mode != "byok"),
        )
    except LlmConfigError as exc:
        raise _http_from_llm_config_error(exc)
    upstream_url = f"{cfg.base_url.rstrip('/')}/chat/completions"

    check_url_safe(upstream_url)

    headers = merge_llm_headers({"Content-Type": "application/json"}, cfg.extra_headers)
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    def _build_payload(include_usage: bool) -> dict[str, Any]:
        base: dict[str, Any] = {
            "model": cfg.model,
            "messages": [{"role": m.role, "content": m.content} for m in body.messages],
            **temperature_param(body.temperature),
            "stream": body.stream,
        }
        # Ask the upstream to emit a trailing usage chunk so instance-paid token
        # counts can be recorded (issue #9). Some servers reject the option — the
        # caller retries once without it.
        if body.stream and include_usage:
            base["stream_options"] = {"include_usage": True}
        return apply_body_extras(base, cfg.extra_body)

    transport = httpx.AsyncHTTPTransport(retries=0)
    started = time.monotonic()

    if body.stream:
        client = httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(120.0),
        )

        async def _open_stream(include_usage: bool):
            req = client.build_request(
                "POST", upstream_url, headers=headers, json=_build_payload(include_usage)
            )
            return await client.send(req, stream=True)

        include_usage = True
        try:
            resp = await _open_stream(include_usage)
        except Exception as exc:
            await client.aclose()
            raise HTTPException(
                status_code=502,
                detail=f"Could not reach LLM endpoint: {exc}",
            )

        # Ollama-family tolerance: if stream_options was rejected, retry once
        # without it (usage is then simply never emitted → recorded as nulls).
        if resp.status_code != 200 and include_usage:
            await resp.aclose()
            include_usage = False
            try:
                resp = await _open_stream(include_usage)
            except Exception as exc:
                await client.aclose()
                raise HTTPException(
                    status_code=502,
                    detail=f"Could not reach LLM endpoint: {exc}",
                )

        if resp.status_code != 200:
            error_bytes = await resp.aread()
            await resp.aclose()
            await client.aclose()
            raise HTTPException(
                status_code=502,
                detail=f"LLM returned {resp.status_code}: {error_bytes[:512].decode(errors='replace')}",
            )

        async def _iter_upstream():
            total = 0
            captured_usage: dict | None = None
            text_buffer = ""
            try:
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    # Tee-parse the SSE text to capture the final usage chunk,
                    # passing the original bytes through untouched.
                    text_buffer += chunk.decode("utf-8", errors="ignore")
                    while "\n" in text_buffer:
                        line, text_buffer = text_buffer.split("\n", 1)
                        if line.startswith("data:"):
                            usage = usage_from_sse_data(line[5:])
                            if usage is not None:
                                captured_usage = usage
                    if total > _MAX_RESPONSE_BYTES:
                        log.warning("LLM streaming response exceeded %d bytes — aborting", _MAX_RESPONSE_BYTES)
                        yield b"data: [DONE]\n\n"
                        return
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()
                # Fire-and-forget; skips BYOK, writes to the dedicated usage DB.
                await record_llm_usage(
                    user_id=ctx.user_id,
                    feature="chat",
                    cfg=cfg,
                    usage=captured_usage,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )

        return StreamingResponse(_iter_upstream(), media_type="text/event-stream")

    else:
        async with httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(120.0),
        ) as client:
            try:
                resp = await client.post(upstream_url, headers=headers, json=_build_payload(False))
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Could not reach LLM endpoint: {exc}",
                )

            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"LLM returned {resp.status_code}: {resp.text[:512]}",
                )

            if len(resp.content) > _MAX_RESPONSE_BYTES:
                raise HTTPException(
                    status_code=502,
                    detail=f"LLM response exceeded the {_MAX_RESPONSE_BYTES // (1024*1024)} MB limit.",
                )

            data = resp.json()
            await record_llm_usage(
                user_id=ctx.user_id,
                feature="chat",
                cfg=cfg,
                usage=data.get("usage") if isinstance(data, dict) else None,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return data
