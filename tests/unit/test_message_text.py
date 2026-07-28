"""Copy rendering for inbox messages.

Messages carry their own text now, so these tests are what stops a message
reaching a mailbox saying nothing useful — the failure mode the old
count-only achievement notification had.
"""
import pytest

from backend.app.services.message_text import (
    ACHIEVEMENT_NAMES,
    _COUNT_NOUN,
    achievement_label,
    render,
)
from openkoutsi.achievements import CATALOGUE, CATALOGUE_BY_ID


class TestCatalogueCoverage:
    def test_every_achievement_has_a_name(self):
        """A new badge must not ship a nameless inbox message."""
        assert set(ACHIEVEMENT_NAMES) == set(CATALOGUE_BY_ID)

    def test_every_multi_tier_count_achievement_says_what_it_counts(self):
        """"10" alone is meaningless — 10 activities, 10 races and 10 notes
        are all `count` tiers."""
        needs_noun = {
            d.id for d in CATALOGUE if d.unit == "count" and len(d.tiers) > 1
        }
        assert needs_noun <= set(_COUNT_NOUN)


class TestAchievementLabel:
    def test_names_the_badge_and_the_tier(self):
        assert achievement_label("single_ride_elevation", 1000) == "Climber — 1000 m"
        assert achievement_label("activity_count", 10) == "Getting started — 10 activities"
        assert achievement_label("long_activity", 6) == "Long hauler — 6 h"
        assert achievement_label("total_distance", 40075) == "The long way round — 40075 km"
        assert achievement_label("plan_adherence", 90) == "On the programme — 90%"
        assert achievement_label("total_load", 1000) == "Load bearer — 1000 Load"
        assert achievement_label("streak_active_weeks", 4) == "Consistency — 4 weeks"
        assert achievement_label("streak_active_months", 3) == "In it for the long run — 3 months"

    def test_a_tier_of_one_reads_as_singular(self):
        """The first tier of `activity_count` is the first badge anyone earns."""
        assert achievement_label("activity_count", 1) == "Getting started — 1 activity"
        assert achievement_label("race_day", 1) == "Race day — 1 race"
        assert achievement_label("goals_reached", 1) == "Goal getter — 1 goal"

    def test_single_tier_badges_are_named_only(self):
        """"Everesting — 8848 m" says the same thing twice."""
        assert achievement_label("everesting", 8848) == "Everesting"
        assert achievement_label("comeback", 1) == "Welcome back"

    def test_tiers_render_without_trailing_noise(self):
        """Tiers arrive as floats from the DB."""
        assert achievement_label("single_ride_elevation", 1000.0) == "Climber — 1000 m"

    def test_an_unknown_id_falls_back_to_the_id(self):
        assert achievement_label("not_a_badge", 1) == "not_a_badge"


class TestAchievementUnlocked:
    def test_a_single_unlock_is_named_in_a_sentence(self):
        rendered = render(
            "achievement_unlocked",
            {"count": 1, "achievements": [{"id": "activity_count", "tier": 1}]},
        )
        assert rendered.title == "Achievement unlocked"
        assert rendered.body == "You earned Getting started — 1 activity."

    def test_a_batch_lists_every_badge(self):
        rendered = render(
            "achievement_unlocked",
            {
                "count": 3,
                "achievements": [
                    {"id": "activity_count", "tier": 1},
                    {"id": "everesting", "tier": 8848},
                    {"id": "long_activity", "tier": 6},
                ],
            },
        )
        assert rendered.title == "3 achievements unlocked"
        lines = rendered.body.splitlines()
        assert lines[0] == "You earned:"
        assert lines[1:] == [
            "• Getting started — 1 activity",
            "• Everesting",
            "• Long hauler — 6 h",
        ]

    def test_a_payload_without_a_list_still_reads(self):
        """Messages written before the list existed, and any future caller that
        forgets it, must not produce an empty card."""
        rendered = render("achievement_unlocked", {"count": 12})
        assert rendered.title == "12 achievements unlocked"
        assert rendered.body == "You earned 12 new achievements."

    def test_malformed_entries_are_skipped(self):
        rendered = render(
            "achievement_unlocked",
            {"count": 2, "achievements": [{"id": "everesting", "tier": 8848}, {}, "junk"]},
        )
        assert rendered.body == "You earned Everesting."


class TestInviteUsed:
    def test_names_the_user(self):
        rendered = render("invite_used", {"username": "ana", "display_name": None})
        assert rendered.title == "Invite used"
        assert rendered.body == "ana joined via an invite link."

    def test_prefers_the_display_name_when_there_is_one(self):
        rendered = render("invite_used", {"username": "ana", "display_name": "Ana R"})
        assert rendered.body == "Ana R (ana) joined via an invite link."


class TestFallback:
    @pytest.mark.parametrize("payload", [{}, None, {"anything": 1}])
    def test_an_unknown_type_is_still_readable(self, payload):
        rendered = render("some_future_type", payload)
        assert rendered.title == "Notification"
        assert rendered.body == "You have a new notification."
