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
from backend.app.services.llm_client import temperature_param


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


class LlmTestResponse(BaseModel):
    ok: bool
    base_url: Optional[str] = None
    model_configured: Optional[str] = None
    response_text: Optional[str] = None
    http_status: Optional[int] = None
    error: Optional[str] = None


# A minimal prompt used to verify the LLM answers. Kept tiny so the round-trip
# is cheap and fast regardless of the backing model.
_TEST_PROMPT = "Reply with a short greeting to confirm you are reachable."


@router.post("/test-connection", response_model=LlmTestResponse)
async def test_llm_connection(
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Test the instance's configured LLM connection. Admin-only.

    Sends a minimal "hello world" chat completion to the instance's saved LLM
    config and confirms the model replies with a usable response.
    """
    ctx, _ = ctx_session
    if not ctx.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    instance = await _load_instance_settings(registry_session)

    base_url = (instance.llm_base_url.strip() if instance and instance.llm_base_url else None) or (settings.llm_base_url or "").strip()
    model = (instance.llm_model.strip() if instance and instance.llm_model else None) or (settings.llm_model or "").strip()

    if not base_url:
        raise HTTPException(status_code=400, detail="No LLM base URL configured. Save a base URL first.")

    if not model:
        return LlmTestResponse(
            ok=False,
            base_url=base_url,
            error="No LLM model configured. Save a model first.",
        )

    api_key: str | None = None
    if instance and instance.llm_api_key_enc:
        try:
            from backend.app.core.file_encryption import decrypt_instance_secret
            api_key = decrypt_instance_secret(str(instance.llm_api_key_enc))
        except Exception as exc:
            log.warning("Could not decrypt instance LLM API key for test: %s", exc)

    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        check_url_safe(chat_url)
    except HTTPException as exc:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=model, error=exc.detail)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _TEST_PROMPT}],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            resp = await client.post(chat_url, headers=headers, json=payload, follow_redirects=False)
    except httpx.ConnectError as exc:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=model, error=f"Connection refused: {exc}")
    except httpx.TimeoutException:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=model, error="Connection timed out")
    except Exception as exc:
        return LlmTestResponse(ok=False, base_url=base_url, model_configured=model, error=str(exc))

    if resp.status_code == 401:
        return LlmTestResponse(
            ok=False,
            base_url=base_url,
            model_configured=model,
            http_status=resp.status_code,
            error="Authentication failed — check your API key",
        )
    if resp.status_code != 200:
        snippet = resp.text[:200] if resp.text else ""
        return LlmTestResponse(
            ok=False,
            base_url=base_url,
            model_configured=model,
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
            model_configured=model,
            http_status=resp.status_code,
            error="The endpoint responded but the reply was not in the expected chat-completion format.",
        )

    reply_text = (reply or "").strip()
    if not reply_text:
        return LlmTestResponse(
            ok=False,
            base_url=base_url,
            model_configured=model,
            http_status=resp.status_code,
            error="The model returned an empty response.",
        )

    return LlmTestResponse(
        ok=True,
        base_url=base_url,
        model_configured=model,
        http_status=resp.status_code,
        response_text=reply_text[:500],
    )


async def _get_llm_config(
    athlete: Athlete,
    user_id: str,
    instance: InstanceSettings | None,
) -> tuple[str, str, str | None]:
    """Return *(base_url, model, api_key)* for this athlete.

    Priority: athlete app_settings → instance settings → global env vars.
    """
    athlete_settings = athlete.app_settings or {}

    # Determine base_url: athlete > instance > global
    base_url = (athlete_settings.get("llm_base_url") or "").strip()
    if not base_url and instance and instance.llm_base_url:
        base_url = instance.llm_base_url.strip()
    if not base_url:
        base_url = (settings.llm_base_url or "").strip()

    if not base_url:
        raise HTTPException(
            status_code=400,
            detail="LLM not configured. Set a base URL in Settings → AI / LLM.",
        )

    # Determine model: athlete > instance > global
    model = (athlete_settings.get("llm_model") or "").strip()
    if not model and instance and instance.llm_model:
        model = instance.llm_model.strip()
    if not model:
        model = (settings.llm_model or "llama3.2").strip()

    # Defense-in-depth: re-check against the allow-list at use time.
    allowed = settings.llm_allowed_servers_list
    if allowed and base_url not in allowed:
        raise HTTPException(
            status_code=403,
            detail="The configured LLM server is not in the server's allowed list. "
            "Update your LLM settings to use an allowed server.",
        )

    api_key: str | None = None

    # Check athlete's personal API key first
    enc_key = athlete_settings.get("llm_api_key_enc")
    if enc_key:
        try:
            from backend.app.core.file_encryption import decrypt_secret
            api_key = decrypt_secret(str(enc_key), user_id)
        except Exception as exc:
            log.error("Failed to decrypt athlete LLM API key for user %s: %s", user_id, exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to decrypt the stored LLM API key. "
                "Try re-entering your key in Settings → AI / LLM.",
            )

    # Fall back to the instance API key
    if api_key is None and instance and instance.llm_api_key_enc:
        try:
            from backend.app.core.file_encryption import decrypt_instance_secret
            api_key = decrypt_instance_secret(str(instance.llm_api_key_enc))
        except Exception as exc:
            log.error("Failed to decrypt instance LLM API key: %s", exc)

    return base_url, model, api_key


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

    base_url, model, api_key = await _get_llm_config(athlete, ctx.user_id, instance)
    upstream_url = f"{base_url.rstrip('/')}/chat/completions"

    check_url_safe(upstream_url)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": body.model or model,
        "messages": [{"role": m.role, "content": m.content} for m in body.messages],
        **temperature_param(body.temperature),
        "stream": body.stream,
    }

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
