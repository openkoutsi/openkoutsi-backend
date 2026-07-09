"""Unit tests for the shared LLM config resolution helpers.

Covers the extra-headers merge, body-extras overlay, and preset-based
resolution (each selectable model is a full connection: its own base URL,
model id, API key, headers and body).
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.app.services.llm_client import (
    LlmConfigError,
    apply_body_extras,
    merge_llm_headers,
    preset_map,
    resolve_instance_llm,
    resolve_llm,
    resolve_llm_config,
)


def _instance(**kw):
    # The instance's LLM config is entirely its preset list (first = default).
    base = dict(llm_models=None)
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
    def test_first_preset_is_the_default_full_connection(self):
        inst = _instance(
            llm_models=[
                {"name": "anthropic", "base_url": "https://api.anthropic.com/v1",
                 "model": "claude-x", "headers": {"X-ZDR": "true"}, "body": {"max_tokens": 8}},
                {"name": "mistral", "base_url": "https://api.mistral.ai/v1"},
            ],
        )
        cfg = resolve_instance_llm(inst)
        # The first preset in the list is the default selection.
        assert cfg.base_url == "https://api.anthropic.com/v1"
        assert cfg.model == "claude-x"
        assert cfg.extra_body == {"max_tokens": 8}
        assert cfg.extra_headers == {"X-ZDR": "true"}
        assert cfg.source == "instance"

    def test_preset_without_base_url_yields_empty_base_url(self):
        # No instance single-config fallback any more: a preset must carry its
        # own base URL, else there is none.
        inst = _instance(llm_models=[{"name": "gpt", "model": "gpt-4o"}])
        cfg = resolve_instance_llm(inst)
        assert cfg.base_url == ""
        assert cfg.model == "gpt-4o"

    def test_requested_model_selects_a_specific_preset(self):
        inst = _instance(
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
        assert cfg.source == "instance"
        assert cfg.key_source == "instance"

    def test_athlete_selection_switches_preset(self):
        inst = _instance(
            llm_models=[
                {"name": "a", "base_url": "https://a/v1", "model": "a1"},
                {"name": "b", "base_url": "https://b/v1", "model": "b1", "body": {"max_tokens": 3}},
            ],
        )
        cfg = resolve_llm_config(_athlete(llm_model="b"), inst, "user-1")
        assert cfg.base_url == "https://b/v1"
        assert cfg.model == "b1"
        assert cfg.extra_body == {"max_tokens": 3}

    def test_no_env_key_fallback(self):
        # No preset key and no instance/env fallback: api_key stays None.
        inst = _instance(llm_models=[{"name": "m", "base_url": "http://x/v1"}])
        cfg = resolve_llm(instance=inst)
        assert cfg.api_key is None
        assert cfg.key_source == "none"


class TestNoMixingRule:
    """BYOK: the athlete's own base URL means only athlete values are used."""

    def test_user_base_url_marks_source_user(self):
        inst = _instance(llm_models=[{"name": "inst", "base_url": "https://inst/v1"}])
        cfg = resolve_llm_config(
            _athlete(llm_base_url="http://my-own/v1", llm_model="my-model"), inst, "user-1",
        )
        assert cfg.base_url == "http://my-own/v1"
        assert cfg.model == "my-model"
        assert cfg.source == "user"

    def test_instance_key_never_leaks_to_user_server(self):
        # The instance's default preset carries an API key; the athlete brings
        # their own base URL but no key. The instance key must NOT be sent.
        inst = _instance(
            llm_models=[{"name": "inst", "base_url": "https://inst/v1", "api_key_enc": "ENC"}],
        )
        with patch(
            "backend.app.core.file_encryption.decrypt_instance_secret",
            return_value="sk-instance",
        ) as dec:
            cfg = resolve_llm_config(_athlete(llm_base_url="http://my-own/v1"), inst, "user-1")
        assert cfg.base_url == "http://my-own/v1"
        assert cfg.api_key is None
        assert cfg.key_source == "none"
        dec.assert_not_called()

    def test_user_own_key_is_used(self):
        inst = _instance(llm_models=[{"name": "inst", "base_url": "https://inst/v1"}])
        with patch(
            "backend.app.core.file_encryption.decrypt_secret",
            return_value="sk-user",
        ) as dec:
            cfg = resolve_llm_config(
                _athlete(llm_base_url="http://my-own/v1", llm_api_key_enc="UENC"),
                inst, "user-1",
            )
        assert cfg.api_key == "sk-user"
        assert cfg.key_source == "user"
        dec.assert_called_once_with("UENC", "user-1")


class TestUseTimeAllowlist:
    def test_user_url_on_allowlist_is_accepted(self):
        inst = _instance(llm_models=[])
        with patch("backend.app.services.llm_client.settings") as ms:
            ms.llm_allowed_servers_list = ["http://my-own/v1"]
            cfg = resolve_llm_config(_athlete(llm_base_url="http://my-own/v1"), inst, "user-1")
        assert cfg.base_url == "http://my-own/v1"

    def test_user_url_off_allowlist_is_rejected(self):
        inst = _instance(llm_models=[])
        with patch("backend.app.services.llm_client.settings") as ms:
            ms.llm_allowed_servers_list = ["http://allowed/v1"]
            with pytest.raises(LlmConfigError) as exc:
                resolve_llm_config(_athlete(llm_base_url="http://blocked/v1"), inst, "user-1")
        assert exc.value.code == "server_not_allowed"

    def test_allowlist_does_not_restrict_instance_config(self):
        # The allow-list only ever restricts BYOK (user) URLs.
        inst = _instance(llm_models=[{"name": "m", "base_url": "http://instance-only/v1"}])
        with patch("backend.app.services.llm_client.settings") as ms:
            ms.llm_allowed_servers_list = ["http://something-else/v1"]
            cfg = resolve_llm_config(_athlete(), inst, "user-1")
        assert cfg.base_url == "http://instance-only/v1"
        assert cfg.source == "instance"


class TestInstanceFallbackHook:
    def test_no_base_url_raises(self):
        inst = _instance(llm_models=[])
        with pytest.raises(LlmConfigError) as exc:
            resolve_llm_config(_athlete(), inst, "user-1")
        assert exc.value.code == "no_base_url"

    def test_disallowing_instance_fallback_raises_for_non_user(self):
        inst = _instance(llm_models=[{"name": "m", "base_url": "http://instance/v1"}])
        with pytest.raises(LlmConfigError) as exc:
            resolve_llm_config(_athlete(), inst, "user-1", allow_instance_fallback=False)
        assert exc.value.code == "instance_fallback_disabled"

    def test_user_config_passes_with_fallback_disabled(self):
        inst = _instance(llm_models=[{"name": "m", "base_url": "http://instance/v1"}])
        cfg = resolve_llm_config(
            _athlete(llm_base_url="http://my-own/v1"), inst, "user-1",
            allow_instance_fallback=False,
        )
        assert cfg.base_url == "http://my-own/v1"
        assert cfg.source == "user"
