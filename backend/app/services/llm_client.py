"""
Shared helpers for calling an OpenAI-compatible chat completions API.

Both the training-plan generator (``llm_plan_generator``) and the structured
workout synthesizer (``llm_workout_generator``) reuse this module so the
LLM-config resolution, the HTTP call (with SSRF protection) and the JSON
extraction logic live in one place.

LLM settings are resolved from preset lists everywhere — the athlete's own
BYOK config (``app_settings``) and the instance's ``llm_models`` (first entry =
default). There are no instance single-config or env-var fallbacks.

No-mixing rule (BYOK)
---------------------
As soon as the athlete configures their *own* base URL (the single
``llm_base_url`` field, or an athlete-level preset with a ``base_url``),
resolution uses **only** athlete-level values (model, key, headers) — instance
presets, the instance API key and instance headers are ignored entirely. This
guarantees the instance's (or the hoster's) API key can never be sent to a
user-chosen server. The resulting :class:`ResolvedLlm` carries a
``source``/``key_source`` signal so callers (and issue #9's gating) can tell
where the config came from.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from ..core.config import settings
from ..core.ssrf import check_url_safe
from ..models.registry_orm import InstanceSettings
from ..models.user_orm import Athlete

log = logging.getLogger(__name__)


ConfigSource = Literal["user", "instance", "env"]
KeySource = Literal["user", "instance", "env", "none"]


class LlmConfigError(ValueError):
    """Raised when an LLM config cannot be resolved for use.

    Carries a machine-readable ``code`` so API layers can map to the right HTTP
    status. Subclasses :class:`ValueError` so existing ``except ValueError``
    call sites keep working.

    Codes:
      * ``no_base_url`` — nothing resolves to a base URL (→ HTTP 400).
      * ``no_model`` — a base URL but no model id (→ HTTP 400).
      * ``server_not_allowed`` — a BYOK URL is outside ``LLM_ALLOWED_SERVERS``
        (→ HTTP 403).
      * ``instance_fallback_disabled`` — the user has no own config and the
        caller disallowed the instance fallback (hook for #9; → HTTP 403).
    """

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


# HTTP status each :class:`LlmConfigError` code maps to (used by API layers).
LLM_ERROR_STATUS: dict[str, int] = {
    "no_base_url": 400,
    "no_model": 400,
    "server_not_allowed": 403,
    "instance_fallback_disabled": 403,
}


@dataclass
class ResolvedLlm:
    """Everything needed to make one outbound LLM request.

    ``extra_headers`` are added to every request (e.g. a zero-data-retention
    header); ``extra_body`` are extra chat-completion body params tied to the
    selected model (e.g. ``max_tokens`` or a thinking config).

    ``source`` records where ``base_url`` came from and ``key_source`` where
    ``api_key`` came from. ``source == "user"`` is the canonical "BYOK active"
    signal (consumed by #9 for gating and usage attribution).
    """

    base_url: str
    model: str
    api_key: str | None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    source: ConfigSource = "env"
    key_source: KeySource = "none"
    # Whether to send a provider-side ``response_format`` (strict JSON schema) for
    # structured generation. On by default; a preset can opt out with
    # ``"structured_outputs": false`` (e.g. a server known not to support it).
    structured_outputs: bool = True


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


def _preset_structured_outputs(preset: dict | None) -> bool:
    """Whether ``preset`` allows provider-side structured outputs.

    Default is **on**: only an explicit ``"structured_outputs": false`` disables
    it (an absent or truthy flag ⇒ enabled).
    """
    return not (isinstance(preset, dict) and preset.get("structured_outputs") is False)


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
) -> ResolvedLlm:
    """Resolve one outbound LLM request from the configured presets.

    Every connection lives in a *preset* (an ``llm_models`` entry) — its own
    base URL, upstream model id, API key, headers and body params. There are no
    instance-level single-config or env-var fallbacks: the config is entirely
    the athlete's and the instance's preset lists. The selected preset is chosen
    by name: ``requested_model`` (a per-request override) → the athlete's saved
    ``llm_model`` → the **first preset in the list** (the default; athlete
    presets take precedence over instance presets).

    **No-mixing rule (BYOK):** if the athlete configured their own base URL
    (the single ``llm_base_url`` field, or a selected athlete-level preset with
    a ``base_url``), only athlete-level values are used — instance presets, the
    instance key and instance headers are all ignored, so the instance key can
    never leak to a user-chosen server.

    ``base_url`` / ``model`` may be empty strings when nothing is configured —
    callers validate and surface an error. The returned :class:`ResolvedLlm`
    records ``source``/``key_source``.
    """
    from ..core.file_encryption import decrypt_instance_secret, decrypt_secret

    athlete_settings = athlete_settings or {}

    inst_presets = preset_map(getattr(instance, "llm_models", None))
    ath_presets = preset_map(athlete_settings.get("llm_models"))

    # ── Selection (by name): request → athlete → first preset ──────────────
    name = (requested_model or "").strip() or (athlete_settings.get("llm_model") or "").strip()
    if not name:
        # The first preset in the list is the default (the instance default lives
        # at the head of ``instance.llm_models``; athlete presets take precedence).
        name = next(iter(ath_presets), "") or next(iter(inst_presets), "")

    ath_p = ath_presets.get(name)
    inst_p = inst_presets.get(name)

    # ── Is this a BYOK (user-owned) request? ───────────────────────────────
    byok_base = (athlete_settings.get("llm_base_url") or "").strip()
    byok_from_preset = False
    if not byok_base and ath_p and ath_p.get("base_url"):
        byok_base = str(ath_p["base_url"]).strip()
        byok_from_preset = True

    if byok_base:
        # No-mixing: athlete values only. Instance is ignored entirely.
        base_url = byok_base
        if byok_from_preset:
            model = str(ath_p.get("model") or name).strip()
        else:
            # BYOK uses the athlete's saved model. Per-request model selection is
            # not supported here (and never `name`, which may have fallen through
            # to an instance preset).
            model = (athlete_settings.get("llm_model") or "").strip()

        api_key: str | None = None
        key_source: KeySource = "none"
        if ath_p and ath_p.get("api_key_enc") and user_id:
            api_key = _try_decrypt(decrypt_secret, str(ath_p["api_key_enc"]), user_id)
        if api_key is None and athlete_settings.get("llm_api_key_enc") and user_id:
            api_key = _try_decrypt(decrypt_secret, str(athlete_settings["llm_api_key_enc"]), user_id)
        if api_key is not None:
            key_source = "user"

        extra_headers = merge_llm_headers({}, (ath_p or {}).get("headers"))
        body = (ath_p or {}).get("body") or {}
        return ResolvedLlm(
            base_url=base_url,
            model=model,
            api_key=api_key,
            extra_headers=extra_headers,
            extra_body=body if isinstance(body, dict) else {},
            source="user",
            key_source=key_source,
            structured_outputs=_preset_structured_outputs(ath_p),
        )

    # ── Non-BYOK: instance preset only ─────────────────────────────────────
    has_preset = bool(inst_p)

    base_url = ""
    source: ConfigSource = "instance"
    if inst_p and inst_p.get("base_url"):
        base_url = str(inst_p["base_url"]).strip()

    if has_preset:
        model = str((inst_p or {}).get("model") or name).strip()
    else:
        model = name

    api_key = None
    key_source = "none"
    if inst_p and inst_p.get("api_key_enc"):
        api_key = _try_decrypt(decrypt_instance_secret, str(inst_p["api_key_enc"]))
    if api_key is not None:
        key_source = "instance"

    extra_headers = merge_llm_headers({}, (inst_p or {}).get("headers"))
    body = (inst_p or {}).get("body") or {}

    return ResolvedLlm(
        base_url=base_url,
        model=model,
        api_key=api_key,
        extra_headers=extra_headers,
        extra_body=body if isinstance(body, dict) else {},
        source=source,
        key_source=key_source,
        structured_outputs=_preset_structured_outputs(inst_p),
    )


def resolve_instance_llm(instance: InstanceSettings | None) -> ResolvedLlm:
    """Instance-only resolution (the instance's preset list; first = default).

    Used by admin diagnostics that are not tied to a particular athlete's
    personal LLM overrides.
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


# Markers an OpenAI-compatible provider tends to put in the 400/422 body when it
# doesn't accept a ``response_format`` / json_schema param. Matched against the
# error text surfaced by ``raise_for_llm_status`` to drive the auto-fallback.
_RESPONSE_FORMAT_UNSUPPORTED_MARKERS = (
    "response_format",
    "response format",
    "json_schema",
    "json schema",
)


def is_response_format_unsupported_error(exc: httpx.HTTPStatusError) -> bool:
    """Is ``exc`` an upstream rejection of the ``response_format`` param?

    Recognises a 400/422 whose body (surfaced by :func:`raise_for_llm_status`)
    mentions ``response_format`` / ``json_schema`` — the signal that the provider
    doesn't support structured outputs, so the caller should drop the field and
    re-issue the prompt-instructed call.
    """
    resp = getattr(exc, "response", None)
    if resp is None or resp.status_code not in (400, 422):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _RESPONSE_FORMAT_UNSUPPORTED_MARKERS)


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
    *,
    requested_model: str | None = None,
    allow_instance_fallback: bool = True,
) -> ResolvedLlm:
    """Resolve the effective LLM config for an athlete-facing request.

    Wraps :func:`resolve_llm` and applies the use-time policy shared by the chat
    proxy and the plan/workout generators:

    * ``allow_instance_fallback=False`` (hook for #9): when the user has no own
      config, raise ``LlmConfigError("instance_fallback_disabled")`` instead of
      falling back to the instance/env config.
    * no resolvable base URL → ``LlmConfigError("no_base_url")``.
    * a base URL but no model → ``LlmConfigError("no_model")`` (so a BYOK user who
      forgot the model gets a clear 400 instead of an opaque upstream 502).
    * when the base URL is user-chosen (``source == "user"``) and an allow-list
      is configured, the URL must be on it, else
      ``LlmConfigError("server_not_allowed")``. The allow-list only ever
      restricts BYOK URLs; admin/instance config is not filtered.
    """
    cfg = resolve_llm(
        instance=instance,
        athlete_settings=athlete.app_settings or {},
        user_id=user_id,
        requested_model=requested_model,
    )

    if not allow_instance_fallback and cfg.source != "user":
        raise LlmConfigError(
            "instance_fallback_disabled",
            "This instance requires you to configure your own LLM server in Settings → AI / LLM.",
        )

    if not cfg.base_url:
        raise LlmConfigError(
            "no_base_url",
            "LLM not configured. Set a base URL in Settings → AI / LLM or ask your administrator.",
        )

    if not cfg.model:
        raise LlmConfigError(
            "no_model",
            "No LLM model configured. Set a model in Settings → AI / LLM or ask your administrator.",
        )

    if cfg.source == "user":
        allowed = settings.llm_allowed_servers_list
        if allowed and cfg.base_url.rstrip("/") not in {a.rstrip("/") for a in allowed}:
            raise LlmConfigError(
                "server_not_allowed",
                "The configured LLM server is not in the server's allowed list. "
                "Update your LLM settings to use an allowed server.",
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
    response_format: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Call the OpenAI-compatible chat completions endpoint.

    Returns ``(text, usage)`` where ``usage`` is the response's ``usage`` object
    (``{"prompt_tokens", "completion_tokens", "total_tokens"}``) or ``None`` when
    the upstream omits it. ``call_llm`` stays transport-only — the caller decides
    whether to record the usage (issue #9), since only instance-paid calls count.

    When ``response_format`` is given (a provider-side ``{"type": "json_schema",
    …}`` block), it is sent as a core payload field so a provider that supports
    structured outputs is constrained to that schema. It is kept distinct from the
    free-form ``extra_body`` so a caller's auto-fallback can drop just this field
    and re-issue the call when a provider rejects it.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers = merge_llm_headers(headers, extra_headers)

    core: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **temperature_param(temperature),
    }
    if response_format is not None:
        core["response_format"] = response_format
    payload = apply_body_extras(core, extra_body)

    url = f"{base_url.rstrip('/')}/chat/completions"
    check_url_safe(url)
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        await raise_for_llm_status(resp, url)

    data = resp.json()
    usage = data.get("usage") if isinstance(data, dict) else None
    return data["choices"][0]["message"]["content"], usage
