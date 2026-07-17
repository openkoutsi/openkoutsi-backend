"""Tests for the LLM activity analyzer service."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.services.llm_activity_analyzer import (
    _build_system_prompt,
    _build_prompt,
    _stream_analysis,
    analyze_activity_bg,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_activity(**kwargs):
    act = MagicMock()
    act.sport_type = kwargs.get("sport_type", "Ride")
    act.start_time = kwargs.get("start_time", datetime(2025, 6, 1, 8, 0, tzinfo=timezone.utc))
    act.duration_s = kwargs.get("duration_s", 3600)
    act.distance_m = kwargs.get("distance_m", 50000)
    act.elevation_m = kwargs.get("elevation_m", 500)
    act.avg_power = kwargs.get("avg_power", 220)
    act.weighted_power = kwargs.get("weighted_power", 235)
    act.intensity = kwargs.get("intensity", 0.84)
    act.load = kwargs.get("load", 75.0)
    act.avg_hr = kwargs.get("avg_hr", 155)
    act.max_hr = kwargs.get("max_hr", 178)
    act.intervals = kwargs.get("intervals", [])
    act.labels = kwargs.get("labels", [])
    act.notes = kwargs.get("notes", None)
    act.analysis = None
    act.analysis_status = None
    return act


def _make_athlete(**kwargs):
    ath = MagicMock()
    ath.ftp = kwargs.get("ftp", 280)
    ath.max_hr = kwargs.get("max_hr", 185)
    ath.app_settings = kwargs.get("app_settings", {})
    return ath


def _make_fatigue(fitness=50.0, fatigue=55.0, form=-5.0):
    f = MagicMock()
    f.fitness = fitness
    f.fatigue = fatigue
    f.form = form
    return f


# ── _build_system_prompt ──────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    def test_no_locale_returns_base(self):
        prompt = _build_system_prompt(None)
        assert "Koutsi" in prompt
        assert "MOOD:" in prompt

    def test_known_locale_adds_language(self):
        prompt = _build_system_prompt("fi")
        assert "Finnish" in prompt

    def test_locale_with_region_tag(self):
        prompt = _build_system_prompt("de-DE")
        assert "German" in prompt

    def test_unknown_locale_no_language_appended(self):
        prompt_default = _build_system_prompt(None)
        prompt_unknown = _build_system_prompt("xx")
        assert prompt_default == prompt_unknown

    def test_all_supported_locales(self):
        for locale, lang in [("en", "English"), ("sv", "Swedish"), ("fr", "French"),
                              ("es", "Spanish"), ("pt", "Portuguese"), ("it", "Italian"),
                              ("nl", "Dutch"), ("pl", "Polish"), ("ru", "Russian"),
                              ("ja", "Japanese"), ("zh", "Chinese"), ("ko", "Korean")]:
            assert lang in _build_system_prompt(locale)

    def test_includes_experience_guidance(self):
        assert "experience level" in _build_system_prompt(None, "Ride")

    def test_cycling_sport_gets_detailed_prompt(self):
        prompt = _build_system_prompt(None, "Ride")
        assert "3-5 paragraphs" in prompt
        assert "experience level" in prompt
        assert "MOOD:" in prompt

    def test_cycling_variants_get_detailed_prompt(self):
        for sport in ("VirtualRide", "GravelRide", "MountainBikeRide", "EBikeRide"):
            prompt = _build_system_prompt(None, sport)
            assert "3-5 paragraphs" in prompt, sport

    def test_non_cycling_sport_gets_short_acknowledgement(self):
        for sport in ("Run", "Swim", "WeightTraining", "Yoga", "Walk"):
            prompt = _build_system_prompt(None, sport)
            assert "supplemental" in prompt.lower(), sport
            assert "3-5 paragraphs" not in prompt, sport
            assert "experience level" not in prompt, sport
            # Frontend/eval still require the MOOD contract.
            assert "MOOD:" in prompt, sport

    def test_missing_sport_defaults_to_short_acknowledgement(self):
        prompt = _build_system_prompt(None, None)
        assert "3-5 paragraphs" not in prompt
        assert "MOOD:" in prompt

    def test_supplemental_prompt_honours_locale(self):
        assert "Finnish" in _build_system_prompt("fi", "Yoga")


# ── _build_prompt ─────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_includes_sport_type(self):
        act = _make_activity(sport_type="Run")
        prompt = _build_prompt(act, _make_athlete())
        assert "Run" in prompt

    def test_includes_duration_hours_format(self):
        act = _make_activity(duration_s=5400)  # 1h30m
        prompt = _build_prompt(act, _make_athlete())
        assert "1h" in prompt

    def test_includes_duration_minutes_format(self):
        act = _make_activity(duration_s=45 * 60)  # 45m
        prompt = _build_prompt(act, _make_athlete())
        assert "45m" in prompt

    def test_includes_power_metrics(self):
        act = _make_activity()
        prompt = _build_prompt(act, _make_athlete())
        assert "220" in prompt  # avg power
        assert "235" in prompt  # Weighted Power

    def test_includes_fatigue_when_provided(self):
        prompt = _build_prompt(_make_activity(), _make_athlete(), _make_fatigue())
        assert "Fitness" in prompt
        assert "Fatigue" in prompt
        assert "Form" in prompt

    def test_no_fatigue_omits_fatigue_section(self):
        prompt = _build_prompt(_make_activity(), _make_athlete(), None)
        assert "Fitness" not in prompt

    def test_missing_optional_fields_dont_crash(self):
        act = _make_activity(distance_m=None, elevation_m=None, avg_power=None,
                              weighted_power=None, load=None, avg_hr=None, max_hr=None)
        prompt = _build_prompt(act, _make_athlete(ftp=None, max_hr=None))
        assert "Ride" in prompt  # at minimum sport type is present

    def test_includes_intervals(self):
        iv = MagicMock()
        iv.interval_number = 1
        iv.duration_s = 300
        iv.avg_hr = 165.0
        iv.avg_power = 280.0
        iv.avg_speed_ms = None
        iv.is_auto_split = False
        act = _make_activity(intervals=[iv])
        prompt = _build_prompt(act, _make_athlete())
        assert "Interval 1" in prompt

    def test_pr_section_present_when_badges_exist(self):
        act = _make_activity()
        prompt = _build_prompt(
            act,
            _make_athlete(),
            power_pr_badges={300: {"all_time": "gold", "12mo": "gold"}},
            distance_pr_badges={5000: {"6mo": "silver"}},
        )
        assert "Personal Records" in prompt
        assert "5min power" in prompt
        assert "all-time gold" in prompt
        assert "12-month gold" in prompt
        assert "5km distance" in prompt
        assert "6-month silver" in prompt

    def test_pr_section_absent_when_no_badges(self):
        act = _make_activity()
        prompt = _build_prompt(act, _make_athlete(), power_pr_badges={}, distance_pr_badges={})
        assert "Personal Records" not in prompt

    def test_pr_section_absent_when_badges_none(self):
        act = _make_activity()
        prompt = _build_prompt(act, _make_athlete())
        assert "Personal Records" not in prompt

    def test_labels_included_when_present(self):
        act = _make_activity()
        act.labels = ["race", "commute"]
        prompt = _build_prompt(act, _make_athlete())
        assert "Activity labels" in prompt
        assert "race" in prompt
        assert "commute" in prompt

    def test_labels_absent_when_empty(self):
        act = _make_activity()
        act.labels = []
        prompt = _build_prompt(act, _make_athlete())
        assert "Activity labels" not in prompt

    def test_labels_absent_when_none(self):
        act = _make_activity()
        act.labels = None
        prompt = _build_prompt(act, _make_athlete())
        assert "Activity labels" not in prompt

    def test_notes_included_when_present(self):
        act = _make_activity()
        act.notes = "Finished 3rd overall. Official time: 1:23:45."
        prompt = _build_prompt(act, _make_athlete())
        assert "Athlete notes" in prompt
        assert "Finished 3rd overall" in prompt

    def test_notes_absent_when_none(self):
        act = _make_activity()
        act.notes = None
        prompt = _build_prompt(act, _make_athlete())
        assert "Athlete notes" not in prompt

    def test_notes_absent_when_whitespace_only(self):
        act = _make_activity()
        act.notes = "   "
        prompt = _build_prompt(act, _make_athlete())
        assert "Athlete notes" not in prompt

    def test_experience_level_included_when_set(self):
        ath = _make_athlete(app_settings={"experience_level": "intermediate"})
        prompt = _build_prompt(_make_activity(), ath)
        assert "experience level: intermediate" in prompt

    def test_experience_level_absent_when_unset(self):
        prompt = _build_prompt(_make_activity(), _make_athlete(app_settings={}))
        assert "experience level" not in prompt


# ── _stream_analysis ──────────────────────────────────────────────────────────

def _make_streaming_lines(chunks):
    """Return an async generator of SSE lines from text chunks."""
    async def _gen():
        for chunk in chunks:
            data = f'{{"choices":[{{"delta":{{"content":"{chunk}"}}}}]}}'
            yield f"data: {data}"
        yield "data: [DONE]"
    return _gen()


def _make_mock_team(base_url="http://localhost:11434/v1", model="llama3.2", analysis_context=None):
    # The instance's LLM config is entirely its preset list (first = default).
    team = MagicMock()
    team.llm_models = [{"name": model, "base_url": base_url}]
    team.llm_analysis_context = analysis_context
    return team


@asynccontextmanager
async def _mock_registry_session(team):
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = team
    session.execute = AsyncMock(return_value=result)
    yield session


class TestStreamAnalysis:
    async def test_yields_text_chunks(self):
        team = _make_mock_team()
        mock_resp = AsyncMock()
        mock_resp.aiter_lines = MagicMock(return_value=_make_streaming_lines(["Hello", " world"]))
        mock_resp.is_error = False

        @asynccontextmanager
        async def _mock_stream(*args, **kwargs):
            yield mock_resp

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream

        @asynccontextmanager
        async def _mock_httpx(*args, **kwargs):
            yield mock_client

        with (
            patch("backend.app.services.llm_activity_analyzer._RegistrySessionLocal",
                  return_value=_mock_registry_session(team)),
            patch("httpx.AsyncClient", return_value=_mock_httpx()),
        ):
            collected = []
            async for chunk in _stream_analysis(_make_activity(), _make_athlete(), "team-1"):
                collected.append(chunk)

        assert "Hello" in collected
        assert " world" in collected

    async def test_raises_when_no_llm_config(self):
        team = MagicMock()
        team.llm_models = None

        with patch("backend.app.services.llm_activity_analyzer._RegistrySessionLocal",
                   return_value=_mock_registry_session(team)):
            with pytest.raises(ValueError, match="base URL"):
                async for _ in _stream_analysis(_make_activity(), _make_athlete(), "team-1"):
                    pass

    async def test_skips_malformed_sse_lines(self):
        team = _make_mock_team()

        async def _malformed_lines():
            yield "not-a-data-line"
            yield "data: {broken json"
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"

        mock_resp = AsyncMock()
        mock_resp.aiter_lines = MagicMock(return_value=_malformed_lines())
        mock_resp.is_error = False

        @asynccontextmanager
        async def _mock_stream(*args, **kwargs):
            yield mock_resp

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream

        @asynccontextmanager
        async def _mock_httpx(*args, **kwargs):
            yield mock_client

        with (
            patch("backend.app.services.llm_activity_analyzer._RegistrySessionLocal",
                  return_value=_mock_registry_session(team)),
            patch("httpx.AsyncClient", return_value=_mock_httpx()),
        ):
            collected = []
            async for chunk in _stream_analysis(_make_activity(), _make_athlete(), "team-1"):
                collected.append(chunk)

        assert collected == ["ok"]

    async def _run_capture_payload(self, team, activity=None):
        """Run _stream_analysis and return the captured messages list."""
        captured: dict = {}

        mock_resp = AsyncMock()
        mock_resp.aiter_lines = MagicMock(return_value=_make_streaming_lines(["ok"]))
        mock_resp.is_error = False

        @asynccontextmanager
        async def _mock_stream(method, url, json=None, **kwargs):
            captured.update(json or {})
            yield mock_resp

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream

        @asynccontextmanager
        async def _mock_httpx(*args, **kwargs):
            yield mock_client

        with (
            patch("backend.app.services.llm_activity_analyzer._RegistrySessionLocal",
                  return_value=_mock_registry_session(team)),
            patch("httpx.AsyncClient", return_value=_mock_httpx()),
        ):
            async for _ in _stream_analysis(activity or _make_activity(), _make_athlete(), "team-1"):
                pass

        return captured["messages"]

    async def test_cycling_activity_uses_detailed_system_prompt(self):
        team = _make_mock_team(analysis_context=None)
        messages = await self._run_capture_payload(team, _make_activity(sport_type="Ride"))
        assert "3-5 paragraphs" in messages[0]["content"]

    async def test_non_cycling_activity_uses_short_system_prompt(self):
        team = _make_mock_team(analysis_context=None)
        messages = await self._run_capture_payload(team, _make_activity(sport_type="Yoga"))
        assert "3-5 paragraphs" not in messages[0]["content"]
        assert "supplemental" in messages[0]["content"].lower()

    async def test_analysis_context_injected_as_second_system_message(self):
        team = _make_mock_team(analysis_context="Focus on running economy.")
        messages = await self._run_capture_payload(team)
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "system", "content": "Focus on running economy."}
        assert messages[2]["role"] == "user"

    async def test_no_analysis_context_omits_second_system_message(self):
        team = _make_mock_team(analysis_context=None)
        messages = await self._run_capture_payload(team)
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 1
        assert messages[-1]["role"] == "user"

    async def test_whitespace_only_context_not_injected(self):
        team = _make_mock_team(analysis_context="   \n  ")
        messages = await self._run_capture_payload(team)
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 1

    async def _run_capture_full(self, team, athlete):
        """Run _stream_analysis and return the full captured request (url + body)."""
        captured: dict = {}

        mock_resp = AsyncMock()
        mock_resp.aiter_lines = MagicMock(return_value=_make_streaming_lines(["ok"]))
        mock_resp.is_error = False

        @asynccontextmanager
        async def _mock_stream(method, url, json=None, **kwargs):
            captured["url"] = url
            captured.update(json or {})
            yield mock_resp

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream

        @asynccontextmanager
        async def _mock_httpx(*args, **kwargs):
            yield mock_client

        with (
            patch("backend.app.services.llm_activity_analyzer._RegistrySessionLocal",
                  return_value=_mock_registry_session(team)),
            patch("httpx.AsyncClient", return_value=_mock_httpx()),
        ):
            async for _ in _stream_analysis(_make_activity(), athlete, "team-1"):
                pass
        return captured

    def _multi_preset_team(self):
        team = MagicMock()
        # Two instance presets; the first is the default.
        team.llm_models = [
            {"name": "default", "base_url": "http://127.0.0.1:11434/v1", "model": "a-model"},
            {"name": "claude", "base_url": "http://127.0.0.1:11435/v1", "model": "b-model"},
        ]
        team.llm_analysis_context = None
        return team

    async def test_honours_athlete_selected_instance_preset(self):
        # The athlete picked the non-default "claude" preset — the analyzer must
        # use it, not the instance default (the reported bug).
        athlete = _make_athlete(app_settings={"llm_model": "claude"})
        captured = await self._run_capture_full(self._multi_preset_team(), athlete)
        assert captured["model"] == "b-model"
        assert captured["url"] == "http://127.0.0.1:11435/v1/chat/completions"

    async def test_falls_back_to_default_preset_without_selection(self):
        athlete = _make_athlete(app_settings={})
        captured = await self._run_capture_full(self._multi_preset_team(), athlete)
        assert captured["model"] == "a-model"
        assert captured["url"] == "http://127.0.0.1:11434/v1/chat/completions"

    async def test_honours_athlete_byok(self):
        # A BYOK athlete's own server is used even by the background analyzer.
        athlete = _make_athlete(
            app_settings={"llm_base_url": "http://127.0.0.1:9999/v1", "llm_model": "my-model"}
        )
        captured = await self._run_capture_full(self._multi_preset_team(), athlete)
        assert captured["model"] == "my-model"
        assert captured["url"] == "http://127.0.0.1:9999/v1/chat/completions"


# ── analyze_activity_bg ───────────────────────────────────────────────────────

class TestAnalyzeActivityBg:
    async def _run_with_mocked_stream(self, chunks, activity, athlete):
        """Run analyze_activity_bg with mocked DB sessions and a canned stream."""
        session = AsyncMock()

        act_result = MagicMock()
        act_result.scalar_one.return_value = activity
        ath_result = MagicMock()
        ath_result.scalar_one.return_value = athlete
        fat_result = MagicMock()
        fat_result.scalar_one_or_none.return_value = None

        session.execute = AsyncMock(side_effect=[act_result, ath_result, fat_result])
        session.commit = AsyncMock()

        @asynccontextmanager
        async def _factory():
            yield session

        async def _canned_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        with (
            patch("backend.app.services.llm_activity_analyzer.get_user_session_factory",
                  return_value=lambda: _factory()),
            patch("backend.app.services.llm_activity_analyzer._stream_analysis",
                  side_effect=_canned_stream),
            patch("backend.app.services.llm_activity_analyzer.detect_pr_badges",
                  new=AsyncMock(return_value=({}, {}))),
        ):
            await analyze_activity_bg("act-1", "ath-1", "team-1")

        return activity, session

    async def test_happy_path_sets_status_done(self):
        act = _make_activity()
        ath = _make_athlete()
        result_act, _ = await self._run_with_mocked_stream(["Good ", "work!"], act, ath)
        assert result_act.analysis_status == "done"
        assert "Good " in result_act.analysis or "work!" in result_act.analysis

    async def test_empty_stream_still_sets_done(self):
        act = _make_activity()
        ath = _make_athlete()
        result_act, _ = await self._run_with_mocked_stream([], act, ath)
        assert result_act.analysis_status == "done"

    async def test_stream_error_sets_status_error(self):
        session = AsyncMock()

        act = _make_activity()
        ath = _make_athlete()
        act_result = MagicMock()
        act_result.scalar_one.return_value = act
        ath_result = MagicMock()
        ath_result.scalar_one.return_value = ath
        fat_result = MagicMock()
        fat_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(side_effect=[act_result, ath_result, fat_result])
        session.commit = AsyncMock()

        @asynccontextmanager
        async def _factory():
            yield session

        async def _failing_stream(*args, **kwargs):
            raise RuntimeError("LLM connection refused")
            yield  # make it an async generator

        with (
            patch("backend.app.services.llm_activity_analyzer.get_user_session_factory",
                  return_value=lambda: _factory()),
            patch("backend.app.services.llm_activity_analyzer._stream_analysis",
                  side_effect=_failing_stream),
            patch("backend.app.services.llm_activity_analyzer.detect_pr_badges",
                  new=AsyncMock(return_value=({}, {}))),
        ):
            await analyze_activity_bg("act-1", "ath-1", "team-1")

        assert act.analysis_status == "error"
