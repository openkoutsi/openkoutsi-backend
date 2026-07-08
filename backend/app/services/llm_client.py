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


def preset_map(models: Any) -> dict[str, dict[str, Any]]:
    """Turn a stored preset list into ``{name: entry}`` keyed by trimmed name.

    A preset (an ``llm_models`` entry) is a full or partial connection:
    ``{"name", "base_url"?, "model"?, "api_key_enc"?, "headers"?, "body"?}``.
    Entries without a name are dropped; later duplicates win.
    """
    out: dict[str, dict[str, Any]] = {}
    if isinstance(models, list):
        for entry in models:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if name:
                out[name] = entry
    return out


def _try_decrypt(fn, *args) -> str | None:
    try:
        return fn(*args)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not decrypt an LLM API key: %s", exc)
        return None


def resolve_llm(
    *,
    instance: InstanceSettings | None,
    athlete_settings: dict | None = None,
    user_id: str | None = None,
    requested_model: str | None = None,
    default_model: str = "",
) -> ResolvedLlm:
    """Resolve one outbound LLM request from instance + optional athlete config.

    A *preset* (an ``llm_models`` entry) is a self-contained connection — its
    own base URL, upstream model id, API key, headers and body params. The
    selected preset is chosen by name: ``requested_model`` (a per-request
    override) → the athlete's saved ``llm_model`` → the instance default
    ``llm_model`` → the first available preset. When the selection names a
    preset, that preset's fields are used (falling back to the instance-level
    single values); otherwise the selection is treated as a bare model id on the
    instance endpoint (legacy single-model config).

    ``base_url`` / ``model`` may be empty strings when nothing is configured —
    callers validate and surface an error.
    """
    athlete_settings = athlete_settings or {}

    inst_presets = preset_map(getattr(instance, "llm_models", None))
    ath_presets = preset_map(athlete_settings.get("llm_models"))

    # ── Selection (by name) ────────────────────────────────────────────────
    name = (requested_model or "").strip() or (athlete_settings.get("llm_model") or "").strip()
    if not name and instance and instance.llm_model:
        name = instance.llm_model.strip()
    if not name:
        name = next(iter(ath_presets), "") or next(iter(inst_presets), "")
    if not name:
        name = (settings.llm_model or default_model).strip()

    ath_p = ath_presets.get(name)
    inst_p = inst_presets.get(name)
    has_preset = bool(ath_p or inst_p)

    # ── base_url: athlete single → preset → instance → global ──────────────
    base_url = (athlete_settings.get("llm_base_url") or "").strip()
    if not base_url and ath_p and ath_p.get("base_url"):
        base_url = str(ath_p["base_url"]).strip()
    if not base_url and inst_p and inst_p.get("base_url"):
        base_url = str(inst_p["base_url"]).strip()
    if not base_url and instance and instance.llm_base_url:
        base_url = instance.llm_base_url.strip()
    if not base_url:
        base_url = (settings.llm_base_url or "").strip()

    # ── model id sent upstream ─────────────────────────────────────────────
    if has_preset:
        model = str((ath_p or {}).get("model") or (inst_p or {}).get("model") or name).strip()
    else:
        model = name
    if not model and instance and instance.llm_model:
        model = instance.llm_model.strip()
    if not model:
        model = (settings.llm_model or default_model).strip()

    # ── api key: preset key (source-aware) → athlete single → instance ─────
    from ..core.file_encryption import decrypt_instance_secret, decrypt_secret

    api_key: str | None = None
    if ath_p and ath_p.get("api_key_enc") and user_id:
        api_key = _try_decrypt(decrypt_secret, str(ath_p["api_key_enc"]), user_id)
    elif inst_p and inst_p.get("api_key_enc"):
        api_key = _try_decrypt(decrypt_instance_secret, str(inst_p["api_key_enc"]))
    if api_key is None and athlete_settings.get("llm_api_key_enc") and user_id:
        api_key = _try_decrypt(decrypt_secret, str(athlete_settings["llm_api_key_enc"]), user_id)
    if api_key is None and instance and instance.llm_api_key_enc:
        api_key = _try_decrypt(decrypt_instance_secret, str(instance.llm_api_key_enc))

    # ── headers: instance → instance preset → athlete preset → athlete ─────
    extra_headers = merge_llm_headers(
        {},
        getattr(instance, "llm_extra_headers", None),
        (inst_p or {}).get("headers"),
        (ath_p or {}).get("headers"),
        athlete_settings.get("llm_extra_headers"),
    )

    # ── body: athlete preset overrides instance preset ─────────────────────
    body = (ath_p or {}).get("body") or (inst_p or {}).get("body") or {}
    extra_body = body if isinstance(body, dict) else {}

    return ResolvedLlm(
        base_url=base_url,
        model=model,
        api_key=api_key,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )


def resolve_instance_llm(instance: InstanceSettings | None) -> ResolvedLlm:
    """Instance-only resolution (instance presets/settings → global env vars).

    Used by admin diagnostics and the automated analysers, which are not tied
    to a particular athlete's personal LLM overrides.
    """
    return resolve_llm(instance=instance)


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
    """Resolve the effective LLM config using athlete → preset → instance → global.

    A selectable model is a full *preset* (its own base URL, model id, API key,
    headers and body params); the athlete's saved ``llm_model`` picks one. See
    :func:`resolve_llm` for the precedence rules. Raises ``ValueError`` when no
    base URL can be determined.
    """
    cfg = resolve_llm(
        instance=instance,
        athlete_settings=athlete.app_settings or {},
        user_id=user_id,
        default_model="llama3.2",
    )
    if not cfg.base_url:
        raise ValueError(
            "LLM not configured. Set a base URL in Settings → AI / LLM or ask your administrator."
        )
    return cfg


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
