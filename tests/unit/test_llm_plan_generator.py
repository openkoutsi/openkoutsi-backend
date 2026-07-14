"""Unit tests for the LLM training-plan prompt builder."""
from openkoutsi.plan_schema import DayConfig, PlanConfig

from backend.app.services.llm_plan_generator import _build_user_prompt


def _config() -> PlanConfig:
    return PlanConfig(
        days_per_week=2,
        day_configs=[
            DayConfig(day_of_week=2, workout_type="threshold"),
            DayConfig(day_of_week=6, workout_type="long"),
        ],
    )


class TestBuildUserPrompt:
    def test_includes_core_requirements(self):
        prompt = _build_user_prompt(_config(), "Spring gran fondo", 8, 250, 60.0)
        assert "8-week training plan" in prompt
        assert "Spring gran fondo" in prompt
        assert "250W" in prompt

    def test_includes_experience_level_when_set(self):
        prompt = _build_user_prompt(_config(), None, 8, 250, None, "experienced")
        assert "experience level: experienced" in prompt

    def test_omits_experience_level_when_none(self):
        prompt = _build_user_prompt(_config(), None, 8, 250, None)
        assert "experience level" not in prompt
