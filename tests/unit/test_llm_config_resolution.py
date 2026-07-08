"""Unit tests for the shared LLM config resolution helpers.

Covers the extra-headers merge, the per-model body-extras lookup, and the
multi-model selection precedence used by resolve_llm_config / resolve_instance_llm.
"""
from types import SimpleNamespace

from backend.app.services.llm_client import (
    apply_body_extras,
    merge_llm_headers,
    model_body_map,
    resolve_instance_llm,
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


class TestModelBodyMap:
    def test_builds_name_to_body(self):
        models = [
            {"name": "gpt", "body": {"max_tokens": 10}},
            {"name": "claude", "body": {"thinking": {"type": "enabled"}}},
            {"name": "bare"},  # no body -> {}
            "garbage",         # ignored
            {"name": ""},      # blank name ignored
        ]
        assert model_body_map(models) == {
            "gpt": {"max_tokens": 10},
            "claude": {"thinking": {"type": "enabled"}},
            "bare": {},
        }


class TestResolveInstanceLlm:
    def test_picks_default_model_and_its_body(self):
        inst = _instance(
            llm_model="claude",
            llm_models=[
                {"name": "gpt", "body": {"max_tokens": 10}},
                {"name": "claude", "body": {"reasoning_effort": "high"}},
            ],
            llm_extra_headers={"X-ZDR": "true"},
        )
        cfg = resolve_instance_llm(inst)
        assert cfg.model == "claude"
        assert cfg.extra_body == {"reasoning_effort": "high"}
        assert cfg.extra_headers == {"X-ZDR": "true"}

    def test_falls_back_to_first_listed_when_no_default(self):
        inst = _instance(
            llm_model=None,
            llm_models=[{"name": "first", "body": {"a": 1}}, {"name": "second"}],
        )
        cfg = resolve_instance_llm(inst)
        assert cfg.model == "first"
        assert cfg.extra_body == {"a": 1}


class TestResolveLlmConfig:
    def test_athlete_selection_and_personal_headers_override(self):
        inst = _instance(
            llm_models=[{"name": "shared", "body": {"max_tokens": 100}}],
            llm_extra_headers={"X-ZDR": "true", "X-Env": "instance"},
        )
        ath = _athlete(
            llm_model="personal",
            llm_models=[{"name": "personal", "body": {"max_tokens": 5}}],
            llm_extra_headers={"X-Env": "athlete"},
        )
        cfg = resolve_llm_config(ath, inst, "user-1")
        assert cfg.model == "personal"
        assert cfg.extra_body == {"max_tokens": 5}
        # Instance header kept, athlete header wins on the shared key.
        assert cfg.extra_headers == {"X-ZDR": "true", "X-Env": "athlete"}

    def test_selected_model_body_comes_from_instance_list(self):
        inst = _instance(
            llm_model="a",
            llm_models=[
                {"name": "a", "body": {"max_tokens": 1}},
                {"name": "b", "body": {"max_tokens": 2}},
            ],
        )
        # Athlete selects model "b" from the instance list; body follows.
        cfg = resolve_llm_config(_athlete(llm_model="b"), inst, "user-1")
        assert cfg.model == "b"
        assert cfg.extra_body == {"max_tokens": 2}
