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

import httpx

from ..core.config import settings
from ..core.ssrf import check_url_safe
from ..models.registry_orm import InstanceSettings
from ..models.user_orm import Athlete

log = logging.getLogger(__name__)


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
) -> tuple[str, str, str | None]:
    """Return *(base_url, model, api_key)* using athlete → instance → global priority."""
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

    model = (athlete_settings.get("llm_model") or "").strip()
    if not model and instance and instance.llm_model:
        model = instance.llm_model.strip()
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

    return base_url, model, api_key


async def call_llm(
    user_prompt: str,
    base_url: str,
    model: str,
    api_key: str | None,
    *,
    system_prompt: str,
    temperature: float = 0.3,
) -> str:
    """Call the OpenAI-compatible chat completions endpoint, return raw text."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    check_url_safe(f"{base_url.rstrip('/')}/chat/completions")
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]
