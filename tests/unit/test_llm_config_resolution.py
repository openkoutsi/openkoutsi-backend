"""Unit tests for the shared LLM config resolution helpers.

Covers the extra-headers merge, body-extras overlay, and preset-based
resolution (each selectable model is a full connection: its own base URL,
model id, API key, headers and body).
"""
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.services.llm_client import (
    apply_body_extras,
    merge_llm_headers,
    preset_map,
    resolve_instance_llm,
    resolve_llm,
    resolve_llm_config,
)


def _instance(**kw):
    base = dict(
        llm_base_url="http://127.0.0.1:11434",
        llm_model="default-model",
        llm_api_key_enc=None,
        llm_models=None,
        llm_extra_headers=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _athlete(**settings):
    return SimpleNamespace(app_settings=settings or {})


class TestMergeHeaders:
    def test_later_wins_and_coerces_strings(self):
        merged = merge_llm_headers({"A": "1"}, {"B": 2}, {"A": "override"})
        assert merged == {"A": "override", "B": "2"}

    def test_ignores_non_dicts_and_blank_keys(self):
        assert merge_llm_headers({}, None, "nope", {"": "x", "K": "v"}) == {"K": "v"}


class TestApplyBodyExtras:
    def test_core_fields_win_over_extras(self):
        payload = {"model": "m", "messages": []}
        out = apply_body_extras(payload, {"model": "hacked", "max_tokens": 64})
        assert out["model"] == "m"
        assert out["max_tokens"] == 64

    def test_empty_extras_returns_payload_unchanged(self):
        payload = {"model": "m"}
        assert apply_body_extras(payload, None) is payload
        assert apply_body_extras(payload, {}) is payload


class TestPresetMap:
    def test_keys_by_name_and_drops_blank(self):
        models = [
            {"name": "gpt", "base_url": "https://x/v1"},
            {"name": ""},
            "garbage",
            {"no": "name"},
        ]
        m = preset_map(models)
        assert list(m.keys()) == ["gpt"]
        assert m["gpt"]["base_url"] == "https://x/v1"


class TestResolveInstanceLlm:
    def test_default_preset_supplies_full_connection(self):
        inst = _instance(
            llm_base_url="http://fallback/v1",
            llm_model="anthropic",
            llm_models=[
                {"name": "anthropic", "base_url": "https://api.anthropic.com/v1",
                 "model": "claude-x", "headers": {"X-ZDR": "true"}, "body": {"max_tokens": 8}},
                {"name": "mistral", "base_url": "https://api.mistral.ai/v1"},
            ],
            llm_extra_headers={"X-Global": "1"},
        )
        cfg = resolve_instance_llm(inst)
        assert cfg.base_url == "https://api.anthropic.com/v1"
        assert cfg.model == "claude-x"
        assert cfg.extra_body == {"max_tokens": 8}
        # Global header plus the preset header.
        assert cfg.extra_headers == {"X-Global": "1", "X-ZDR": "true"}

    def test_preset_without_base_url_falls_back_to_instance(self):
        inst = _instance(
            llm_base_url="http://fallback/v1",
            llm_model="gpt",
            llm_models=[{"name": "gpt", "model": "gpt-4o"}],
        )
        cfg = resolve_instance_llm(inst)
        assert cfg.base_url == "http://fallback/v1"
        assert cfg.model == "gpt-4o"

    def test_requested_model_selects_a_specific_preset(self):
        inst = _instance(
            llm_model="a",
            llm_models=[
                {"name": "a", "base_url": "https://a/v1"},
                {"name": "b", "base_url": "https://b/v1", "model": "b-model"},
            ],
        )
        cfg = resolve_llm(instance=inst, requested_model="b")
        assert cfg.base_url == "https://b/v1"
        assert cfg.model == "b-model"


class TestResolveLlmConfigPresetKeys:
    def test_selected_instance_preset_key_is_decrypted(self):
        inst = _instance(
            llm_model="anthropic",
            llm_models=[{"name": "anthropic", "base_url": "https://api.anthropic.com/v1",
                         "model": "claude-x", "api_key_enc": "ENC"}],
        )
        with patch(
            "backend.app.core.file_encryption.decrypt_instance_secret",
            return_value="sk-secret",
        ) as dec:
            cfg = resolve_llm_config(_athlete(llm_model="anthropic"), inst, "user-1")
        dec.assert_called_once_with("ENC")
        assert cfg.api_key == "sk-secret"
        assert cfg.base_url == "https://api.anthropic.com/v1"
        assert cfg.model == "claude-x"

    def test_athlete_selection_switches_preset(self):
        inst = _instance(
            llm_model="a",
            llm_models=[
                {"name": "a", "base_url": "https://a/v1", "model": "a1"},
                {"name": "b", "base_url": "https://b/v1", "model": "b1", "body": {"max_tokens": 3}},
            ],
        )
        cfg = resolve_llm_config(_athlete(llm_model="b"), inst, "user-1")
        assert cfg.base_url == "https://b/v1"
        assert cfg.model == "b1"
        assert cfg.extra_body == {"max_tokens": 3}

    def test_athlete_personal_base_url_overrides_preset(self):
        inst = _instance(
            llm_model="anthropic",
            llm_models=[{"name": "anthropic", "base_url": "https://api.anthropic.com/v1"}],
        )
        cfg = resolve_llm_config(
            _athlete(llm_model="anthropic", llm_base_url="http://my-own/v1"), inst, "user-1",
        )
        assert cfg.base_url == "http://my-own/v1"

    def test_falls_back_to_env_api_key(self):
        # No preset/athlete/instance key: the server-side default (LLM_API_KEY)
        # is used.
        inst = _instance(llm_api_key_enc=None)
        with patch("backend.app.services.llm_client.settings") as mock_settings:
            mock_settings.llm_api_key = "env-default-key"
            mock_settings.llm_model = ""
            cfg = resolve_llm(instance=inst)
        assert cfg.api_key == "env-default-key"
