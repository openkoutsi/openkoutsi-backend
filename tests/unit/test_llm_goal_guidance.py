"""Tests for the LLM goal-guidance service."""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

from backend.app.models.user_orm import Activity, Athlete, DailyMetric, Goal, TrainingPlan
from backend.app.services.llm_goal_guidance import (
    _FALLBACK_VERDICT,
    _build_goal_prompt,
    _build_system_prompt,
    _parse_verdict,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_athlete(**kwargs):
    ath = MagicMock(spec=Athlete)
    ath.ftp = kwargs.get("ftp", 250)
    ath.max_hr = kwargs.get("max_hr", 186)
    ath.app_settings = kwargs.get("app_settings", {})
    return ath


def _make_goal(**kwargs):
    goal = MagicMock(spec=Goal)
    goal.title = kwargs.get("title", "Reach FTP 300 W")
    goal.description = kwargs.get("description", "Push threshold power up over the winter block")
    goal.metric = kwargs.get("metric", "ftp")
    goal.target_value = kwargs.get("target_value", 300.0)
    goal.current_value = kwargs.get("current_value", 250.0)
    goal.target_date = kwargs.get("target_date", date(2026, 12, 1))
    goal.status = kwargs.get("status", "active")
    return goal


def _make_metric(ctl=64.0, atl=71.0, tsb=-7.0):
    m = MagicMock(spec=DailyMetric)
    m.ctl = ctl
    m.atl = atl
    m.tsb = tsb
    return m


def _make_activity(**kwargs):
    act = MagicMock(spec=Activity)
    act.sport_type = kwargs.get("sport_type", "Ride")
    act.start_time = kwargs.get("start_time", datetime(2026, 7, 8, 17, tzinfo=timezone.utc))
    act.duration_s = kwargs.get("duration_s", 3600)
    act.tss = kwargs.get("tss", 68.0)
    return act


_NOW = datetime(2026, 7, 13, 9, 0, tzinfo=timezone.utc)


# ── _build_system_prompt ──────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    def test_no_locale_returns_base(self):
        prompt = _build_system_prompt(None)
        assert "Koutsi" in prompt
        assert "REALISM:" in prompt

    def test_known_locale_adds_language(self):
        assert "Finnish" in _build_system_prompt("fi")

    def test_locale_with_region_tag(self):
        assert "German" in _build_system_prompt("de-DE")

    def test_unknown_locale_no_language_appended(self):
        assert _build_system_prompt(None) == _build_system_prompt("xx")

    def test_coaching_style_appended(self):
        prompt = _build_system_prompt(None, "stern")
        assert "strict" in prompt.lower() or "blunt" in prompt.lower()


# ── _build_goal_prompt ────────────────────────────────────────────────────────

class TestBuildGoalPrompt:
    def test_includes_goal_title_and_target(self):
        prompt = _build_goal_prompt(
            _make_athlete(), _make_goal(), [], _make_metric(), None, _NOW
        )
        assert "Reach FTP 300 W" in prompt
        assert "300" in prompt  # target value

    def test_includes_target_date_and_days_remaining(self):
        prompt = _build_goal_prompt(
            _make_athlete(), _make_goal(target_date=date(2026, 12, 1)), [], None, None, _NOW
        )
        assert "2026-12-01" in prompt
        # 2026-07-13 → 2026-12-01 is 141 days.
        assert "141 days remaining" in prompt

    def test_past_target_date_marked(self):
        prompt = _build_goal_prompt(
            _make_athlete(), _make_goal(target_date=date(2026, 1, 1)), [], None, None, _NOW
        )
        assert "already past" in prompt

    def test_no_target_date(self):
        prompt = _build_goal_prompt(
            _make_athlete(), _make_goal(target_date=None), [], None, None, _NOW
        )
        assert "none set" in prompt

    def test_includes_fitness_metrics(self):
        prompt = _build_goal_prompt(
            _make_athlete(), _make_goal(), [], _make_metric(ctl=64.0), None, _NOW
        )
        assert "CTL" in prompt
        assert "ATL" in prompt
        assert "TSB" in prompt

    def test_omits_fitness_without_metric(self):
        prompt = _build_goal_prompt(_make_athlete(), _make_goal(), [], None, None, _NOW)
        assert "CTL" not in prompt

    def test_includes_recent_activities(self):
        prompt = _build_goal_prompt(
            _make_athlete(), _make_goal(), [_make_activity()], None, None, _NOW
        )
        assert "Ride" in prompt

    def test_no_activities_message(self):
        prompt = _build_goal_prompt(_make_athlete(), _make_goal(), [], None, None, _NOW)
        assert "no activities recorded" in prompt

    def test_includes_active_plan(self):
        plan = MagicMock(spec=TrainingPlan)
        plan.name = "Base to Build"
        plan.start_date = date(2026, 6, 29)
        plan.end_date = date(2026, 8, 24)
        prompt = _build_goal_prompt(_make_athlete(), _make_goal(), [], None, plan, _NOW)
        assert "Base to Build" in prompt

    def test_no_plan_message(self):
        prompt = _build_goal_prompt(_make_athlete(), _make_goal(), [], None, None, _NOW)
        assert "No active training plan" in prompt

    def test_missing_optional_fields_dont_crash(self):
        goal = _make_goal(
            description=None, metric=None, target_value=None,
            current_value=None, target_date=None,
        )
        prompt = _build_goal_prompt(
            _make_athlete(ftp=None, max_hr=None), goal, [], None, None, _NOW
        )
        assert "Reach FTP 300 W" in prompt  # title always present


# ── _parse_verdict ────────────────────────────────────────────────────────────

class TestParseVerdict:
    def test_realistic(self):
        verdict, prose = _parse_verdict("REALISM:realistic\n\nYou've got this.\n\nStay consistent.")
        assert verdict == "realistic"
        assert prose == "You've got this.\n\nStay consistent."
        assert "REALISM" not in prose

    def test_ambitious_with_space(self):
        verdict, prose = _parse_verdict("REALISM: ambitious\n\nA real stretch.")
        assert verdict == "ambitious"
        assert prose == "A real stretch."

    def test_unrealistic(self):
        verdict, _ = _parse_verdict("REALISM:unrealistic\n\nNot on this timeline.")
        assert verdict == "unrealistic"

    def test_case_insensitive(self):
        verdict, _ = _parse_verdict("realism: REALISTIC\n\nGood.")
        assert verdict == "realistic"

    def test_leading_blank_lines_skipped(self):
        verdict, prose = _parse_verdict("\n\nREALISM:realistic\n\nGo.")
        assert verdict == "realistic"
        assert prose == "Go."

    def test_missing_tag_uses_fallback_and_keeps_prose(self):
        verdict, prose = _parse_verdict("You should train more base miles.")
        assert verdict == _FALLBACK_VERDICT
        assert "train more base miles" in prose

    def test_malformed_tag_uses_fallback(self):
        verdict, prose = _parse_verdict("REALISM:maybe\n\nUnsure.")
        assert verdict == _FALLBACK_VERDICT
        # The whole text is kept since the first line wasn't a valid tag.
        assert "Unsure." in prose
