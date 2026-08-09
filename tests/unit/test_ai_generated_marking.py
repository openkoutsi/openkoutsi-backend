"""Machine-readable marking of language-model output (issue #41).

The EU AI Act's transparency rules ask that generated content be disclosed as
such. Our own web UI carries a visible notice, but it is not the only client of
these endpoints — personal access tokens, the MCP layer and third-party
integrations all read the same JSON. Every response that carries Koutsi's prose
therefore ships a boolean saying a model wrote it.

The flags are derived from the prose column rather than stored, so these tests
pin the property that makes that safe: the flag is true exactly when there is
generated text to mark, in every intermediate state the endpoints can return.
"""

from datetime import datetime, timezone

from backend.app.schemas.activities import ActivityDetailResponse
from backend.app.schemas.athlete import TrainingStatusResponse
from backend.app.schemas.goals import GoalGuidanceResponse


def activity(**overrides) -> ActivityDetailResponse:
    """An activity detail response with only its required identity filled in."""
    return ActivityDetailResponse(
        id="a1",
        athlete_id="ath1",
        status="processed",
        created_at=datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc),
        **overrides,
    )


class TestTrainingStatusMarking:
    def test_marked_when_feedback_is_present(self):
        resp = TrainingStatusResponse(status="done", feedback="Ease off this week.")
        assert resp.feedback_ai_generated is True

    def test_not_marked_before_anything_is_generated(self):
        assert TrainingStatusResponse().feedback_ai_generated is False

    def test_not_marked_while_still_pending(self):
        # The analysers stream into the column, so a pending response can hold
        # no text yet. Nothing to disclose until there is.
        resp = TrainingStatusResponse(status="pending", feedback=None)
        assert resp.feedback_ai_generated is False

    def test_marked_once_a_pending_stream_has_partial_text(self):
        resp = TrainingStatusResponse(status="pending", feedback="MOOD: knowing\n\nYour load")
        assert resp.feedback_ai_generated is True

    def test_not_marked_when_generation_failed_without_output(self):
        resp = TrainingStatusResponse(status="error", feedback=None)
        assert resp.feedback_ai_generated is False

    def test_flag_is_serialised(self):
        payload = TrainingStatusResponse(status="done", feedback="Go easy.").model_dump()
        assert payload["feedback_ai_generated"] is True


class TestGoalGuidanceMarking:
    def test_marked_when_guidance_is_present(self):
        resp = GoalGuidanceResponse(
            status="done", verdict="ambitious", guidance="A stretch, but reachable."
        )
        assert resp.guidance_ai_generated is True

    def test_not_marked_before_anything_is_generated(self):
        assert GoalGuidanceResponse().guidance_ai_generated is False

    def test_not_marked_while_still_pending(self):
        resp = GoalGuidanceResponse(status="pending", verdict=None, guidance=None)
        assert resp.guidance_ai_generated is False

    def test_flag_is_serialised(self):
        payload = GoalGuidanceResponse(status="done", guidance="Reachable.").model_dump()
        assert payload["guidance_ai_generated"] is True


class TestActivityAnalysisMarking:
    def test_marked_when_analysis_is_present(self):
        resp = activity(analysis="Solid tempo work.")
        assert resp.analysis_ai_generated is True

    def test_not_marked_before_anything_is_generated(self):
        assert activity().analysis_ai_generated is False

    def test_not_marked_while_still_pending(self):
        resp = activity(analysis_status="pending", analysis=None)
        assert resp.analysis_ai_generated is False

    def test_marking_is_scoped_to_the_analysis(self):
        # Everything else on an activity is measured or computed from the ride
        # file. A response with real metrics but no analysis must not claim to
        # be generated, or the flag would be useless for telling the two apart.
        resp = activity(avg_power=231.0, duration_s=3600, analysis=None)
        assert resp.analysis_ai_generated is False

    def test_flag_is_serialised(self):
        payload = activity(analysis="Solid tempo work.").model_dump()
        assert payload["analysis_ai_generated"] is True
