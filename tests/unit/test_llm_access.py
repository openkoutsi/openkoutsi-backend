"""Unit tests for the LLM access gate + usage helpers (issue #9)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from backend.app.models.registry_orm import LlmEntitlement
from backend.app.services.llm_access import (
    byok_active,
    is_entitled,
    merge_usage,
    parse_usage,
    provider_label,
    usage_from_sse_data,
)
from backend.app.services.llm_client import ResolvedLlm


def _ent(**kw) -> LlmEntitlement:
    now = datetime.now(timezone.utc)
    defaults = dict(status="active", source="manual", starts_at=now, expires_at=None)
    defaults.update(kw)
    return LlmEntitlement(user_id="u1", **defaults)


class TestIsEntitled:
    def test_none_is_not_entitled(self):
        assert is_entitled(None) is False

    def test_active_no_expiry(self):
        assert is_entitled(_ent()) is True

    def test_revoked(self):
        assert is_entitled(_ent(status="revoked")) is False

    def test_future_start(self):
        future = datetime.now(timezone.utc) + timedelta(days=1)
        assert is_entitled(_ent(starts_at=future)) is False

    def test_expired(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert is_entitled(_ent(expires_at=past)) is False

    def test_expiry_in_future_is_active(self):
        future = datetime.now(timezone.utc) + timedelta(days=1)
        assert is_entitled(_ent(expires_at=future)) is True

    def test_naive_datetimes_are_treated_as_utc(self):
        # Boundary: naive stored datetimes must not raise and compare as UTC.
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        assert is_entitled(_ent(starts_at=past, expires_at=None)) is True


class TestParseUsage:
    def test_none(self):
        assert parse_usage(None) == (None, None, None)

    def test_not_a_dict(self):
        assert parse_usage("nope") == (None, None, None)

    def test_full(self):
        assert parse_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        ) == (10, 5, 15)

    def test_total_derived_when_missing(self):
        assert parse_usage({"prompt_tokens": 10, "completion_tokens": 5}) == (10, 5, 15)

    def test_partial_only_prompt(self):
        assert parse_usage({"prompt_tokens": 7}) == (7, None, 7)

    def test_empty_dict(self):
        assert parse_usage({}) == (None, None, None)

    def test_bool_is_not_int(self):
        assert parse_usage({"prompt_tokens": True}) == (None, None, None)


class TestProviderLabel:
    def _cfg(self, base_url):
        return ResolvedLlm(base_url=base_url, model="m", api_key=None)

    def test_host_from_url(self):
        assert provider_label(self._cfg("https://api.openai.com/v1")) == "api.openai.com"

    def test_localhost(self):
        assert provider_label(self._cfg("http://localhost:11434/v1")) == "localhost"

    def test_empty(self):
        assert provider_label(self._cfg("")) is None


class TestUsageFromSse:
    def test_present(self):
        data = '{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}'
        assert usage_from_sse_data(data) == {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        }

    def test_null_usage(self):
        assert usage_from_sse_data('{"choices":[{"delta":{"content":"hi"}}],"usage":null}') is None

    def test_no_usage_key(self):
        assert usage_from_sse_data('{"choices":[{"delta":{"content":"hi"}}]}') is None

    def test_done_marker(self):
        assert usage_from_sse_data("[DONE]") is None

    def test_malformed(self):
        assert usage_from_sse_data("{broken") is None


class TestByokActive:
    def test_single_base_url(self):
        athlete = SimpleNamespace(app_settings={"llm_base_url": "http://my-ollama:11434/v1"})
        assert byok_active(athlete) is True

    def test_preset_base_url(self):
        athlete = SimpleNamespace(
            app_settings={"llm_models": [{"name": "mine", "base_url": "http://x/v1"}]}
        )
        assert byok_active(athlete) is True

    def test_no_byok(self):
        athlete = SimpleNamespace(app_settings={"llm_model": "gpt-4o"})
        assert byok_active(athlete) is False

    def test_none_athlete(self):
        assert byok_active(None) is False


class TestMergeUsage:
    """Summing a run's turns (issue #43).

    An agent loop is three to five completions, each billed separately. Keeping
    only the last would make ``GET /api/admin/llm-usage/summary`` under-report
    every agentic analysis by however many turns it took — the concrete
    accounting bug the issue names, so it gets its own tests rather than being
    assumed correct.
    """

    def test_sums_the_parts_across_two_calls(self):
        merged = merge_usage(
            {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            {"prompt_tokens": 400, "completion_tokens": 300, "total_tokens": 700},
        )
        assert merged == {
            "prompt_tokens": 500,
            "completion_tokens": 320,
            "total_tokens": 820,
        }

    def test_the_first_call_starts_from_nothing(self):
        addition = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert merge_usage(None, addition) == addition

    def test_a_turn_reporting_nothing_leaves_the_total_alone(self):
        total = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert merge_usage(total, None) == total

    def test_nothing_at_all_stays_nothing(self):
        # Recorded with nulls, never estimated — the same rule `parse_usage` follows.
        assert merge_usage(None, None) is None

    def test_absent_parts_stay_absent_rather_than_becoming_zero(self):
        # A provider that reports no completion_tokens has told us nothing; 0 is
        # a claim, and a wrong one.
        merged = merge_usage({"prompt_tokens": 10}, {"prompt_tokens": 30})
        assert merged["prompt_tokens"] == 40
        assert "completion_tokens" not in merged

    def test_a_total_can_never_be_smaller_than_its_own_parts(self):
        # One turn reported only a total, the next only the parts. Trusting the
        # summed `total_tokens` alone would report 50 for 50 + 400 + 300.
        merged = merge_usage(
            {"total_tokens": 50},
            {"prompt_tokens": 400, "completion_tokens": 300},
        )
        assert merged["total_tokens"] == 750

    def test_a_non_dict_addition_is_ignored(self):
        total = {"prompt_tokens": 10}
        assert merge_usage(total, "nonsense") == total
        assert merge_usage("nonsense", None) is None

    def test_booleans_are_not_token_counts(self):
        assert merge_usage({"prompt_tokens": True}, {"prompt_tokens": 5}) == {
            "prompt_tokens": 5,
            "total_tokens": 5,
        }
