"""Unit tests for the shared athlete experience-level helper (issue #32)."""
from backend.app.services.athlete_experience import (
    EXPERIENCE_GUIDANCE,
    VALID_EXPERIENCE_LEVELS,
    experience_level,
)


class TestExperienceLevel:
    def test_returns_valid_level(self):
        for level in VALID_EXPERIENCE_LEVELS:
            assert experience_level({"experience_level": level}) == level

    def test_strips_whitespace(self):
        assert experience_level({"experience_level": " intermediate "}) == "intermediate"

    def test_unknown_level_returns_none(self):
        assert experience_level({"experience_level": "pro-tour"}) is None

    def test_empty_string_returns_none(self):
        assert experience_level({"experience_level": ""}) is None

    def test_missing_key_returns_none(self):
        assert experience_level({"locale": "fi"}) is None

    def test_none_settings_returns_none(self):
        assert experience_level(None) is None

    def test_non_dict_settings_returns_none(self):
        assert experience_level("intermediate") is None


class TestExperienceGuidance:
    def test_guidance_mentions_the_levels(self):
        assert "novice" in EXPERIENCE_GUIDANCE
        assert "elite" in EXPERIENCE_GUIDANCE
