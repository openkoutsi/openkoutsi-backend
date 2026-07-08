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
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import UserContext
from backend.app.core.config import settings
from backend.app.core.deps import get_ctx_and_session
from backend.app.core.ssrf import check_url_safe
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import InstanceSettings
from backend.app.models.user_orm import Athlete
from backend.app.services.llm_client import (
    apply_body_extras,
    merge_llm_headers,
    preset_map,
    resolve_llm,
    temperature_param,
)


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
    # would actually be used: saved choice → instance default → first preset →
    # global env default.
    selected = (athlete_settings.get("llm_model") or "").strip()
    if not selected and instance and instance.llm_model:
        selected = instance.llm_model.strip()
    if not selected and presets:
        selected = next(iter(presets))
    if not selected:
        selected = (settings.llm_model or "").strip()

    # Make sure the effective selection is always offered, even if it is a lone
    # legacy single-model configuration not present in any preset list.
    if selected and selected not in presets:
        options.append(LlmModelOption(name=selected, label=selected))

    return LlmModelsResponse(models=options, selected=selected or None)


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
    to test a specific model from the configured list instead of the default.
    """
    ctx, _ = ctx_session
    if not ctx.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    instance = await _load_instance_settings(registry_session)
    # Resolve the selected preset (its own base URL, model, key, headers, body).
    cfg = resolve_llm(instance=instance, requested_model=model)

    base_url = cfg.base_url
    if not base_url:
        raise HTTPException(status_code=400, detail="No LLM base URL configured. Save a base URL first.")

    selected = cfg.model
    if not selected:
        return LlmTestResponse(
            ok=False,
            base_url=base_url,
            error="No LLM model configured. Save a model first.",
        )

    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        check_url_safe(chat_url)
    except HTTPException as exc:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=selected, error=exc.detail)

    headers = merge_llm_headers({"Content-Type": "application/json"}, cfg.extra_headers)
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    payload = apply_body_extras(
        {
            "model": selected,
            "messages": [{"role": "user", "content": _TEST_PROMPT}],
            "stream": False,
        },
        cfg.extra_body,
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            resp = await client.post(chat_url, headers=headers, json=payload, follow_redirects=False)
    except httpx.ConnectError as exc:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=selected, prompt_sent=_TEST_PROMPT, error=f"Connection refused: {exc}")
    except httpx.TimeoutException:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=selected, prompt_sent=_TEST_PROMPT, error="Connection timed out")
    except Exception as exc:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=selected, prompt_sent=_TEST_PROMPT, error=str(exc))

    if resp.status_code == 401:
        return LlmTestResponse(
            ok=False,
            base_url=base_url,
            model_configured=selected,
            prompt_sent=_TEST_PROMPT,
            http_status=resp.status_code,
            error="Authentication failed — check your API key",
        )
    if resp.status_code != 200:
        snippet = resp.text[:200] if resp.text else ""
        return LlmTestResponse(
            ok=False,
            base_url=base_url,
            model_configured=selected,
            prompt_sent=_TEST_PROMPT,
            http_status=resp.status_code,
            error=f"HTTP {resp.status_code}: {snippet}",
        )

    try:
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        return LlmTestResponse(
            ok=False,
            base_url=base_url,
            model_configured=selected,
            prompt_sent=_TEST_PROMPT,
            http_status=resp.status_code,
            error="The endpoint responded but the reply was not in the expected chat-completion format.",
        )

    reply_text = (reply or "").strip()
    if not reply_text:
        return LlmTestResponse(
            ok=False,
            base_url=base_url,
            model_configured=selected,
            prompt_sent=_TEST_PROMPT,
            http_status=resp.status_code,
            error="The model returned an empty response.",
        )

    return LlmTestResponse(
        ok=True,
        base_url=base_url,
        model_configured=selected,
        prompt_sent=_TEST_PROMPT,
        http_status=resp.status_code,
        response_text=reply_text[:500],
    )


async def _get_llm_config(
    athlete: Athlete,
    user_id: str,
    instance: InstanceSettings | None,
    requested_model: str | None = None,
):
    """Return a :class:`ResolvedLlm` for this athlete.

    Selection resolves ``requested_model`` (per-request override) → the athlete's
    saved model → the instance default; a selected preset supplies its own base
    URL, model id, API key, headers and body params. Raises ``HTTPException``
    (400/403) on the two API-facing failures: no base URL, or a resolved server
    outside the admin allow-list.
    """
    cfg = resolve_llm(
        instance=instance,
        athlete_settings=athlete.app_settings or {},
        user_id=user_id,
        requested_model=requested_model,
        default_model="llama3.2",
    )

    if not cfg.base_url:
        raise HTTPException(
            status_code=400,
            detail="LLM not configured. Set a base URL in Settings → AI / LLM.",
        )

    # Defense-in-depth: re-check the resolved server against the allow-list.
    # Normalise a trailing slash on both sides so it can't cause a false 403.
    allowed = settings.llm_allowed_servers_list
    if allowed and cfg.base_url.rstrip("/") not in {a.rstrip("/") for a in allowed}:
        raise HTTPException(
            status_code=403,
            detail="The configured LLM server is not in the server's allowed list. "
            "Update your LLM settings to use an allowed server.",
        )

    return cfg


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

    cfg = await _get_llm_config(athlete, ctx.user_id, instance, requested_model=body.model)
    upstream_url = f"{cfg.base_url.rstrip('/')}/chat/completions"

    check_url_safe(upstream_url)

    headers = merge_llm_headers({"Content-Type": "application/json"}, cfg.extra_headers)
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    payload: dict[str, Any] = apply_body_extras(
        {
            "model": cfg.model,
            "messages": [{"role": m.role, "content": m.content} for m in body.messages],
            **temperature_param(body.temperature),
            "stream": body.stream,
        },
        cfg.extra_body,
    )

    transport = httpx.AsyncHTTPTransport(retries=0)

    if body.stream:
        client = httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(120.0),
        )
        try:
            req = client.build_request("POST", upstream_url, headers=headers, json=payload)
            resp = await client.send(req, stream=True)
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
            try:
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > _MAX_RESPONSE_BYTES:
                        log.warning("LLM streaming response exceeded %d bytes — aborting", _MAX_RESPONSE_BYTES)
                        yield b"data: [DONE]\n\n"
                        return
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(_iter_upstream(), media_type="text/event-stream")

    else:
        async with httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(120.0),
        ) as client:
            try:
                resp = await client.post(upstream_url, headers=headers, json=payload)
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

            return resp.json()
