"""
Shared helpers for calling an OpenAI-compatible chat completions API.

Both the training-plan generator (``llm_plan_generator``) and the structured
workout synthesizer (``llm_workout_generator``) reuse this module so the
LLM-config resolution, the HTTP call (with SSRF protection) and the JSON
extraction logic live in one place.

LLM settings are resolved with the same priority everywhere:
  athlete app_settings → instance settings → global env vars
  (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..core.config import settings
from ..core.ssrf import check_url_safe
from ..models.registry_orm import InstanceSettings
from ..models.user_orm import Athlete

log = logging.getLogger(__name__)


@dataclass
class ResolvedLlm:
    """Everything needed to make one outbound LLM request.

    ``extra_headers`` are added to every request (e.g. a zero-data-retention
    header); ``extra_body`` are extra chat-completion body params tied to the
    selected model (e.g. ``max_tokens`` or a thinking config).
    """

    base_url: str
    model: str
    api_key: str | None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


def _coerce_str_dict(value: Any) -> dict[str, str]:
    """Coerce a stored JSON value into a ``{str: str}`` header dict."""
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if str(k).strip()}


def model_body_map(models: Any) -> dict[str, dict[str, Any]]:
    """Turn a stored ``[{"name", "body"}]`` list into ``{name: body}``."""
    out: dict[str, dict[str, Any]] = {}
    if isinstance(models, list):
        for entry in models:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            body = entry.get("body")
            out[name] = body if isinstance(body, dict) else {}
    return out


def merge_llm_headers(base: dict[str, str], *extras: Any) -> dict[str, str]:
    """Overlay ``extras`` (later wins) onto ``base``; non-dicts are ignored."""
    headers = dict(base)
    for extra in extras:
        headers.update(_coerce_str_dict(extra))
    return headers


def apply_body_extras(payload: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    """Overlay ``extra`` under ``payload`` so core fields (model, messages,
    stream) always win but extras like ``max_tokens`` are added."""
    if not isinstance(extra, dict) or not extra:
        return payload
    return {**extra, **payload}


def resolve_instance_llm(instance: InstanceSettings | None) -> ResolvedLlm:
    """Instance-only resolution (instance settings → global env vars).

    Used by admin diagnostics and the automated analysers, which are not tied
    to a particular athlete's personal LLM overrides. ``base_url`` / ``model``
    may be empty strings when nothing is configured — callers should validate.
    """
    base_url = (instance.llm_base_url.strip() if instance and instance.llm_base_url else None) or (settings.llm_base_url or "").strip()

    bodies = model_body_map(getattr(instance, "llm_models", None)) if instance else {}
    model = (instance.llm_model.strip() if instance and instance.llm_model else "")
    if not model and bodies:
        model = next(iter(bodies))
    if not model:
        model = (settings.llm_model or "").strip()

    api_key: str | None = None
    if instance and instance.llm_api_key_enc:
        try:
            from ..core.file_encryption import decrypt_instance_secret
            api_key = decrypt_instance_secret(str(instance.llm_api_key_enc))
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Could not decrypt instance LLM API key: %s", exc)

    return ResolvedLlm(
        base_url=base_url,
        model=model,
        api_key=api_key,
        extra_headers=merge_llm_headers({}, getattr(instance, "llm_extra_headers", None) if instance else None),
        extra_body=bodies.get(model, {}),
    )


def temperature_param(override: float | None = None) -> dict[str, float]:
    """Return a ``{"temperature": X}`` payload fragment, or ``{}`` to omit it.

    By default the ``temperature`` field is left out entirely so each model
    applies its own default. This keeps thinking-enabled models (e.g. Claude
    with extended thinking, reached via Anthropic's OpenAI-compatible endpoint),
    which reject any temperature other than ``1``, working out of the box. A
    caller may still pass an explicit value to force one.
    """
    if override is None:
        return {}
    return {"temperature": override}


async def raise_for_llm_status(resp: httpx.Response, url: str) -> None:
    """Like ``resp.raise_for_status()`` but reads and surfaces the upstream body.

    ``httpx``'s built-in ``raise_for_status`` discards the response body, which
    is exactly where an OpenAI-compatible provider explains a 400/422 (e.g. an
    unsupported ``temperature`` for a thinking model). For streamed responses
    the body must be read explicitly before raising, so we always ``aread`` it
    and include a truncated copy in the log line and the raised error.
    """
    if not resp.is_error:
        return
    try:
        body = (await resp.aread())[:1000].decode(errors="replace")
    except Exception:  # pragma: no cover - body already consumed / unreadable
        body = "<response body unavailable>"
    log.error("LLM request to %s failed: HTTP %s — %s", url, resp.status_code, body)
    raise httpx.HTTPStatusError(
        f"LLM request to {url} failed with status {resp.status_code}: {body}",
        request=resp.request,
        response=resp,
    )


def extract_json(text: str) -> str:
    """Strip markdown code fences if present and return the raw JSON string."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def resolve_llm_config(
    athlete: Athlete,
    instance: InstanceSettings | None,
    user_id: str,
) -> ResolvedLlm:
    """Resolve the effective LLM config using athlete → instance → global priority.

    The available model list is the instance's curated list overlaid by any
    personal models the athlete configured; the selected model is the athlete's
    saved choice (``llm_model``) falling back to the instance default. Extra
    headers merge instance-then-athlete (athlete wins per key), and the body
    extras are those attached to the selected model.
    """
    athlete_settings = athlete.app_settings or {}

    base_url = (athlete_settings.get("llm_base_url") or "").strip()
    if not base_url and instance and instance.llm_base_url:
        base_url = instance.llm_base_url.strip()
    if not base_url:
        base_url = (settings.llm_base_url or "").strip()

    if not base_url:
        raise ValueError(
            "LLM not configured. Set a base URL in Settings → AI / LLM or ask your administrator."
        )

    # Available models: instance list overlaid by the athlete's personal list.
    bodies = {
        **model_body_map(getattr(instance, "llm_models", None)),
        **model_body_map(athlete_settings.get("llm_models")),
    }

    model = (athlete_settings.get("llm_model") or "").strip()
    if not model and instance and instance.llm_model:
        model = instance.llm_model.strip()
    if not model and bodies:
        model = next(iter(bodies))
    if not model:
        model = (settings.llm_model or "llama3.2").strip()

    api_key: str | None = None

    enc_key = athlete_settings.get("llm_api_key_enc")
    if enc_key:
        try:
            from ..core.file_encryption import decrypt_secret
            api_key = decrypt_secret(str(enc_key), user_id)
        except Exception as exc:
            log.error("Failed to decrypt athlete LLM API key for user %s: %s", user_id, exc)
            raise ValueError(
                "Failed to decrypt the stored LLM API key. Try re-entering it in Settings → AI / LLM."
            ) from exc

    if api_key is None and instance and instance.llm_api_key_enc:
        try:
            from ..core.file_encryption import decrypt_instance_secret
            api_key = decrypt_instance_secret(str(instance.llm_api_key_enc))
        except Exception as exc:
            log.error("Failed to decrypt instance LLM API key: %s", exc)

    extra_headers = merge_llm_headers(
        {},
        getattr(instance, "llm_extra_headers", None),
        athlete_settings.get("llm_extra_headers"),
    )

    return ResolvedLlm(
        base_url=base_url,
        model=model,
        api_key=api_key,
        extra_headers=extra_headers,
        extra_body=bodies.get(model, {}),
    )


async def call_llm(
    user_prompt: str,
    base_url: str,
    model: str,
    api_key: str | None,
    *,
    system_prompt: str,
    temperature: float | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Call the OpenAI-compatible chat completions endpoint, return raw text."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers = merge_llm_headers(headers, extra_headers)

    payload = apply_body_extras(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **temperature_param(temperature),
        },
        extra_body,
    )

    url = f"{base_url.rstrip('/')}/chat/completions"
    check_url_safe(url)
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        await raise_for_llm_status(resp, url)

    return resp.json()["choices"][0]["message"]["content"]
