"""Integration tests for commute detection and suggestions (issue #63).

The invariant every test here circles: **a suggestion is not a label**.
``activities.labels`` is what the athlete has confirmed; ``label_suggestions``
is what we think. Getting that wrong shows up as achievement tiers minted off a
heuristic and as rides vanishing from the prompt that was meant to ask about
them, so the split is tested from both sides.
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from backend.app.models.user_orm import Activity

# A rule an athlete might plausibly write: a short weekday bike ride, morning or
# evening. Reused across the module so a test that changes behaviour changes it
# for one reason only.
MORNING_EVENING_RULE = {
    "id": "to-work",
    "name": "To work",
    "sport_types": ["Ride"],
    "min_distance_m": 4000,
    "max_distance_m": 8000,
    "windows": [{"start": "06:30", "end": "08:30"}, {"start": "15:30", "end": "18:00"}],
    "weekdays": [0, 1, 2, 3, 4],
}


async def _set_rules(client, auth_headers, rules):
    resp = await client.patch(
        "/api/athlete", json={"app_settings": {"commute_rules": rules}}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    return resp


async def _ride(
    client,
    auth_headers,
    *,
    start="2026-08-26T07:30:00Z",  # a Wednesday
    duration_s=1200,
    distance_m=5400.0,
    sport_type="Ride",
):
    resp = await client.post(
        "/api/activities",
        json={
            "sport_type": sport_type,
            "start_time": start,
            "duration_s": duration_s,
            "distance_m": distance_m,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _get(client, auth_headers, activity_id):
    resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
    assert resp.status_code == 200
    return resp.json()


def _commute_state(activity: dict):
    entry = (activity.get("label_suggestions") or {}).get("commute")
    return entry["state"] if entry else None


class TestSuggestionOnIngest:
    async def test_a_matching_ride_is_suggested_not_labelled(self, client, auth_headers):
        """The central invariant of the feature."""
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)

        fresh = await _get(client, auth_headers, ride["id"])
        assert _commute_state(fresh) == "pending"
        assert fresh["labels"] == []

    async def test_the_suggestion_records_which_rule_fired(self, client, auth_headers):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)

        fresh = await _get(client, auth_headers, ride["id"])
        assert fresh["label_suggestions"]["commute"]["source"] == "rule:to-work"

    async def test_a_non_matching_ride_gets_nothing(self, client, auth_headers):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers, distance_m=90000.0, duration_s=12000)

        fresh = await _get(client, auth_headers, ride["id"])
        assert fresh["label_suggestions"] == {}
        assert fresh["labels"] == []

    async def test_an_athlete_with_no_rules_gets_nothing(self, client, auth_headers):
        ride = await _ride(client, auth_headers)
        fresh = await _get(client, auth_headers, ride["id"])
        assert fresh["label_suggestions"] == {}

    async def test_auto_apply_writes_the_label_outright(self, client, auth_headers):
        """Opt-in per rule, for an athlete who has watched it get it right."""
        await _set_rules(
            client, auth_headers, [{**MORNING_EVENING_RULE, "auto_apply": True}]
        )
        ride = await _ride(client, auth_headers)

        fresh = await _get(client, auth_headers, ride["id"])
        assert fresh["labels"] == ["commute"]
        assert _commute_state(fresh) == "accepted"

    async def test_auto_apply_is_off_unless_asked_for(self, client, auth_headers):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)
        assert (await _get(client, auth_headers, ride["id"]))["labels"] == []


class TestAnsweringASuggestion:
    async def test_accepting_applies_the_label_and_records_the_answer(
        self, client, auth_headers
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)

        resp = await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"commute": "accepted"}},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["labels"] == ["commute"]
        assert _commute_state(resp.json()) == "accepted"

    async def test_dismissing_records_the_refusal_without_labelling(
        self, client, auth_headers
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)

        resp = await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"commute": "dismissed"}},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["labels"] == []
        assert _commute_state(resp.json()) == "dismissed"

    async def test_the_answer_keeps_the_rule_that_proposed_it_on_record(
        self, client, auth_headers
    ):
        """`rule_feedback` is built entirely on this provenance surviving."""
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"commute": "dismissed"}},
            headers=auth_headers,
        )
        assert resp.json()["label_suggestions"]["commute"]["source"] == "rule:to-work"

    async def test_labelling_by_hand_answers_a_pending_suggestion(
        self, client, auth_headers
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{ride['id']}",
            json={"labels": ["commute"]},
            headers=auth_headers,
        )
        assert _commute_state(resp.json()) == "accepted"

    async def test_removing_a_label_by_hand_dismisses_the_suggestion(
        self, client, auth_headers
    ):
        """Otherwise unticking it would leave the suggestion pending forever."""
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)
        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"labels": ["commute"]},
            headers=auth_headers,
        )
        resp = await client.patch(
            f"/api/activities/{ride['id']}", json={"labels": []}, headers=auth_headers
        )
        assert resp.json()["labels"] == []
        # Accepted, then un-applied: the ride is not suggested again.
        assert _commute_state(resp.json()) in ("accepted", "dismissed")

    @pytest.mark.parametrize("answer", ["maybe", "pending", "", "yes"])
    async def test_an_unknown_answer_is_rejected(self, client, auth_headers, answer):
        ride = await _ride(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"commute": answer}},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_an_unknown_label_is_rejected(self, client, auth_headers):
        ride = await _ride(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"groceries": "accepted"}},
            headers=auth_headers,
        )
        assert resp.status_code == 422


class TestDismissalIsDurable:
    """The single most important behaviour in the feature."""

    async def test_a_dismissal_survives_a_rule_re_evaluation(self, client, auth_headers):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)
        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"commute": "dismissed"}},
            headers=auth_headers,
        )

        # Saving the rules again re-evaluates everything unanswered.
        await _set_rules(
            client, auth_headers, [{**MORNING_EVENING_RULE, "max_distance_m": 9000}]
        )

        fresh = await _get(client, auth_headers, ride["id"])
        assert _commute_state(fresh) == "dismissed"
        assert fresh["labels"] == []

    async def test_a_dismissal_survives_a_history_scan(self, client, auth_headers):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)
        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"commute": "dismissed"}},
            headers=auth_headers,
        )

        resp = await client.post("/api/activities/commute/scan", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["suggested"] == 0
        assert _commute_state(await _get(client, auth_headers, ride["id"])) == "dismissed"

    async def test_force_is_the_explicit_way_to_look_again(self, client, auth_headers):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)
        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"commute": "dismissed"}},
            headers=auth_headers,
        )

        resp = await client.post(
            "/api/activities/commute/scan?force=true", headers=auth_headers
        )
        assert resp.json()["suggested"] == 1
        assert _commute_state(await _get(client, auth_headers, ride["id"])) == "pending"


class TestReevaluationOnRuleChange:
    """Issue #63 open question 5: re-evaluate, don't freeze."""

    async def test_narrowing_a_rule_withdraws_the_suggestions_it_dropped(
        self, client, auth_headers
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers, distance_m=7500.0)
        assert _commute_state(await _get(client, auth_headers, ride["id"])) == "pending"

        await _set_rules(
            client, auth_headers, [{**MORNING_EVENING_RULE, "max_distance_m": 6000}]
        )

        fresh = await _get(client, auth_headers, ride["id"])
        assert fresh["label_suggestions"] == {}

    async def test_widening_a_rule_picks_up_what_it_now_covers(self, client, auth_headers):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers, distance_m=11000.0)
        assert (await _get(client, auth_headers, ride["id"]))["label_suggestions"] == {}

        await _set_rules(
            client, auth_headers, [{**MORNING_EVENING_RULE, "max_distance_m": 12000}]
        )

        assert _commute_state(await _get(client, auth_headers, ride["id"])) == "pending"

    async def test_deleting_every_rule_withdraws_every_pending_suggestion(
        self, client, auth_headers
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)
        assert _commute_state(await _get(client, auth_headers, ride["id"])) == "pending"

        await _set_rules(client, auth_headers, [])

        assert (await _get(client, auth_headers, ride["id"]))["label_suggestions"] == {}

    async def test_an_accepted_suggestion_is_not_undone_by_a_narrowed_rule(
        self, client, auth_headers
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers, distance_m=7500.0)
        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"commute": "accepted"}},
            headers=auth_headers,
        )

        await _set_rules(
            client, auth_headers, [{**MORNING_EVENING_RULE, "max_distance_m": 6000}]
        )

        fresh = await _get(client, auth_headers, ride["id"])
        assert fresh["labels"] == ["commute"]
        assert _commute_state(fresh) == "accepted"


class TestRuleValidation:
    async def test_a_rule_with_no_criteria_is_rejected(self, client, auth_headers):
        """It would match nothing, silently, and the athlete would never know."""
        resp = await client.patch(
            "/api/athlete",
            json={"app_settings": {"commute_rules": [{"id": "empty"}]}},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "criterion" in resp.json()["detail"]

    async def test_a_rule_without_an_id_is_rejected(self, client, auth_headers):
        resp = await client.patch(
            "/api/athlete",
            json={"app_settings": {"commute_rules": [{"sport_types": ["Ride"]}]}},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    async def test_duplicate_ids_are_rejected(self, client, auth_headers):
        resp = await client.patch(
            "/api/athlete",
            json={
                "app_settings": {
                    "commute_rules": [
                        {"id": "a", "sport_types": ["Ride"]},
                        {"id": "a", "sport_types": ["Run"]},
                    ]
                }
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "unique" in resp.json()["detail"]

    async def test_a_non_list_is_rejected(self, client, auth_headers):
        resp = await client.patch(
            "/api/athlete",
            json={"app_settings": {"commute_rules": {"id": "a"}}},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    async def test_rules_are_stored_normalised(self, client, auth_headers):
        """What comes back is what the detector will actually use."""
        resp = await _set_rules(
            client,
            auth_headers,
            [{"id": "x", "sport_types": ["RIDE", "Ride"], "windows": [{"start": "7:05", "end": "08:30"}]}],
        )
        stored = resp.json()["app_settings"]["commute_rules"][0]
        assert stored["sport_types"] == ["ride"]
        assert stored["windows"] == [{"start": "07:05", "end": "08:30"}]
        assert stored["enabled"] is True and stored["auto_apply"] is False

    async def test_an_empty_list_is_a_state_the_athlete_can_choose(
        self, client, auth_headers
    ):
        """It must survive the None-stripping merge, or "delete all" is a no-op."""
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        resp = await _set_rules(client, auth_headers, [])
        assert resp.json()["app_settings"]["commute_rules"] == []


class TestTimezone:
    async def test_windows_are_the_athletes_local_clock(self, client, auth_headers):
        """05:30 UTC is 08:30 in Helsinki — inside the morning window."""
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"timezone": "Europe/Helsinki"}},
            headers=auth_headers,
        )
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers, start="2026-08-26T05:30:00Z")

        assert _commute_state(await _get(client, auth_headers, ride["id"])) == "pending"

    async def test_the_same_ride_misses_without_the_zone(self, client, auth_headers):
        """Proof the conversion is doing the work rather than luck."""
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers, start="2026-08-26T05:30:00Z")

        assert (await _get(client, auth_headers, ride["id"]))["label_suggestions"] == {}

    async def test_an_unusable_timezone_degrades_to_utc_rather_than_500(
        self, client, auth_headers
    ):
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"timezone": "Mars/Olympus_Mons"}},
            headers=auth_headers,
        )
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        resp = await client.post(
            "/api/activities",
            json={
                "sport_type": "Ride",
                "start_time": "2026-08-26T07:30:00Z",
                "duration_s": 1200,
                "distance_m": 5400.0,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201


class TestRpeQueueInteraction:
    """The ordering trap that makes suggestions a separate state from labels."""

    async def test_a_suggested_ride_stays_in_the_prompt_queue(self, client, auth_headers):
        """Writing the label early would delete it from the surface that asks."""
        await client.get("/api/activities/rpe-queue", headers=auth_headers)
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)

        resp = await client.get("/api/activities/rpe-queue", headers=auth_headers)
        items = {item["id"]: item for item in resp.json()["items"]}
        assert ride["id"] in items
        # And the prompt has what it needs to pre-tick its box.
        assert _commute_state(items[ride["id"]]) == "pending"

    async def test_accepting_takes_the_ride_out_of_the_queue(self, client, auth_headers):
        await client.get("/api/activities/rpe-queue", headers=auth_headers)
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)
        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"commute": "accepted"}},
            headers=auth_headers,
        )

        resp = await client.get("/api/activities/rpe-queue", headers=auth_headers)
        assert ride["id"] not in [item["id"] for item in resp.json()["items"]]

    async def test_an_auto_applied_ride_never_reaches_the_queue(self, client, auth_headers):
        await client.get("/api/activities/rpe-queue", headers=auth_headers)
        await _set_rules(
            client, auth_headers, [{**MORNING_EVENING_RULE, "auto_apply": True}]
        )
        ride = await _ride(client, auth_headers)

        resp = await client.get("/api/activities/rpe-queue", headers=auth_headers)
        assert ride["id"] not in [item["id"] for item in resp.json()["items"]]


class TestAchievements:
    """A pending suggestion is not an earned badge."""

    async def _commuter_progress(self, client, auth_headers):
        resp = await client.get("/api/achievements", headers=auth_headers)
        assert resp.status_code == 200
        return resp.json()["progress"].get("commuter", 0)

    async def test_a_pending_suggestion_does_not_count_toward_the_badge(
        self, client, auth_headers
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        for day in range(5):
            await _ride(
                client, auth_headers, start=f"2026-08-{24 + day:02d}T07:30:00Z"
            )
        assert await self._commuter_progress(client, auth_headers) == 0

    async def test_accepting_them_does(self, client, auth_headers):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ids = []
        for day in range(3):
            ride = await _ride(
                client, auth_headers, start=f"2026-08-{24 + day:02d}T07:30:00Z"
            )
            ids.append(ride["id"])
        for activity_id in ids:
            await client.patch(
                f"/api/activities/{activity_id}",
                json={"label_answers": {"commute": "accepted"}},
                headers=auth_headers,
            )
        assert await self._commuter_progress(client, auth_headers) == 3


class TestHistoryScan:
    async def test_scanning_suggests_across_the_back_catalogue(self, client, auth_headers):
        """The answer to an imported history the per-ingest hook never saw."""
        for day in range(5):
            await _ride(client, auth_headers, start=f"2026-08-{24 + day:02d}T07:30:00Z")
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])

        # Saving the rules already re-evaluated; scan reports the same set.
        resp = await client.post("/api/activities/commute/scan", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["scanned"] == 5

    async def test_scanning_without_rules_does_nothing(self, client, auth_headers):
        await _ride(client, auth_headers)
        resp = await client.post("/api/activities/commute/scan", headers=auth_headers)
        assert resp.json() == {"scanned": 0, "suggested": 0, "applied": 0}

    async def test_the_review_filter_returns_only_unanswered_suggestions(
        self, client, auth_headers
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        pending = await _ride(client, auth_headers, start="2026-08-26T07:30:00Z")
        answered = await _ride(client, auth_headers, start="2026-08-25T07:30:00Z")
        unrelated = await _ride(
            client, auth_headers, start="2026-08-24T12:00:00Z", distance_m=90000.0
        )
        await client.patch(
            f"/api/activities/{answered['id']}",
            json={"label_answers": {"commute": "dismissed"}},
            headers=auth_headers,
        )

        resp = await client.get(
            "/api/activities?suggested_label=commute", headers=auth_headers
        )
        ids = [item["id"] for item in resp.json()["items"]]
        assert pending["id"] in ids
        assert answered["id"] not in ids
        assert unrelated["id"] not in ids

    async def test_the_review_filter_rejects_an_unknown_label(self, client, auth_headers):
        resp = await client.get(
            "/api/activities?suggested_label=groceries", headers=auth_headers
        )
        assert resp.status_code == 422


class TestRuleProposal:
    async def test_nothing_is_proposed_below_ten_labelled_rides(self, client, auth_headers):
        """Issue #63: clustering three rides invents a rule out of nothing."""
        for day in range(1, 10):
            ride = await _ride(client, auth_headers, start=f"2026-08-{day:02d}T07:30:00Z")
            await client.patch(
                f"/api/activities/{ride['id']}",
                json={"labels": ["commute"]},
                headers=auth_headers,
            )

        resp = await client.get("/api/activities/commute/proposal", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["rule"] is None
        assert resp.json()["sample_count"] == 9
        assert resp.json()["min_samples"] == 10

    async def test_ten_labelled_rides_yield_a_usable_rule(self, client, auth_headers):
        for day in range(1, 11):
            ride = await _ride(client, auth_headers, start=f"2026-08-{day:02d}T07:40:00Z")
            await client.patch(
                f"/api/activities/{ride['id']}",
                json={"labels": ["commute"]},
                headers=auth_headers,
            )

        resp = await client.get("/api/activities/commute/proposal", headers=auth_headers)
        rule = resp.json()["rule"]
        assert rule is not None
        assert rule["windows"]
        assert rule["min_distance_m"] <= 5400.0 <= rule["max_distance_m"]

    async def test_the_proposal_is_built_from_labels_not_from_our_own_guesses(
        self, client, auth_headers
    ):
        """A proposal derived from suggestions would be circular."""
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        for day in range(1, 13):
            await _ride(client, auth_headers, start=f"2026-08-{day:02d}T07:40:00Z")

        resp = await client.get("/api/activities/commute/proposal", headers=auth_headers)
        assert resp.json()["sample_count"] == 0
        assert resp.json()["rule"] is None


class TestRuleFeedback:
    async def test_a_hand_labelled_ride_the_rules_missed_names_the_bound_to_widen(
        self, client, auth_headers
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers, distance_m=8600.0)  # just over the band
        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"labels": ["commute"]},
            headers=auth_headers,
        )

        resp = await client.get("/api/activities/commute/feedback", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["unmatched_manual_labels"] == 1
        assert data["widen"] == [{"rule_id": "to-work", "criteria": {"distance": 1}}]

    async def test_a_genuinely_different_ride_is_not_evidence_about_any_rule(
        self, client, auth_headers
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(
            client,
            auth_headers,
            start="2026-08-29T10:00:00Z",  # Saturday, midday
            distance_m=90000.0,
            duration_s=12000,
        )
        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"labels": ["commute"]},
            headers=auth_headers,
        )

        data = (
            await client.get("/api/activities/commute/feedback", headers=auth_headers)
        ).json()
        assert data["unmatched_manual_labels"] == 1
        assert data["widen"] == []

    async def test_repeated_dismissals_flag_the_rule_for_review(self, client, auth_headers):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        for day in range(24, 27):
            ride = await _ride(client, auth_headers, start=f"2026-08-{day:02d}T07:30:00Z")
            await client.patch(
                f"/api/activities/{ride['id']}",
                json={"label_answers": {"commute": "dismissed"}},
                headers=auth_headers,
            )

        data = (
            await client.get("/api/activities/commute/feedback", headers=auth_headers)
        ).json()
        assert data["review"] == [{"rule_id": "to-work", "dismissed": 3}]

    async def test_one_dismissal_is_a_slip_not_a_verdict(self, client, auth_headers):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)
        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"label_answers": {"commute": "dismissed"}},
            headers=auth_headers,
        )

        data = (
            await client.get("/api/activities/commute/feedback", headers=auth_headers)
        ).json()
        assert data["review"] == []


class TestAthleteIsolation:
    async def test_one_athletes_suggestions_are_invisible_to_another(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _set_rules(client, auth_headers, [MORNING_EVENING_RULE])
        ride = await _ride(client, auth_headers)

        other = Activity(
            id="other-athlete-ride",
            athlete_id="someone-else",
            name="Not yours",
            sport_type="Ride",
            start_time=datetime(2026, 8, 26, 7, 30, tzinfo=timezone.utc),
            duration_s=1200,
            distance_m=5400.0,
            status="processed",
        )
        session.add(other)
        await session.commit()

        resp = await client.get(
            "/api/activities?suggested_label=commute", headers=auth_headers
        )
        ids = [item["id"] for item in resp.json()["items"]]
        assert ids == [ride["id"]]

        resp = await client.get(
            f"/api/activities/{other.id}", headers=auth_headers
        )
        assert resp.status_code == 404


class TestStravaCommuteFlag:
    """The athlete's own tick in Strava, applied rather than suggested."""

    async def test_a_flagged_activity_is_labelled_outright(self, session, seeded_athlete):
        from backend.app.services.commute import adopt_provider_flag

        activity = Activity(
            id="strava-commute",
            athlete_id=seeded_athlete.id,
            sport_type="Ride",
            start_time=datetime(2026, 8, 26, 7, 30, tzinfo=timezone.utc),
            duration_s=1200,
            distance_m=5400.0,
            status="processed",
        )
        assert adopt_provider_flag(activity, True) is True
        assert activity.labels == ["commute"]
        assert activity.label_suggestions["commute"]["state"] == "accepted"
        assert activity.label_suggestions["commute"]["source"] == "strava"

    @pytest.mark.parametrize("flag", [False, None])
    async def test_an_absent_or_false_flag_asserts_nothing(
        self, session, seeded_athlete, flag
    ):
        """Most athletes never touch Strava's checkbox; a default is not a denial."""
        from backend.app.services.commute import adopt_provider_flag

        activity = Activity(
            id=f"strava-{flag}",
            athlete_id=seeded_athlete.id,
            sport_type="Ride",
            start_time=datetime(2026, 8, 26, 7, 30, tzinfo=timezone.utc),
            duration_s=1200,
            distance_m=5400.0,
            status="processed",
        )
        assert adopt_provider_flag(activity, flag) is False
        assert not activity.labels

    async def test_the_flag_does_not_overrule_a_dismissal(self, session, seeded_athlete):
        from backend.app.services.commute import (
            adopt_provider_flag,
            answer_suggestion,
            STATE_DISMISSED,
        )

        activity = Activity(
            id="strava-dismissed",
            athlete_id=seeded_athlete.id,
            sport_type="Ride",
            start_time=datetime(2026, 8, 26, 7, 30, tzinfo=timezone.utc),
            duration_s=1200,
            distance_m=5400.0,
            status="processed",
        )
        answer_suggestion(activity, "commute", STATE_DISMISSED)
        assert adopt_provider_flag(activity, True) is False
        assert not activity.labels

    async def test_the_strava_normalizer_reads_the_flag(self):
        from backend.app.services.providers.strava import _normalize_activity

        raw = {
            "id": 1,
            "start_date": "2026-08-26T07:30:00Z",
            "name": "Ride to work",
            "sport_type": "Ride",
            "moving_time": 1200,
            "distance": 5400.0,
            "commute": True,
        }
        assert _normalize_activity(raw).commute is True

    async def test_a_payload_without_the_key_normalizes_to_false(self):
        from backend.app.services.providers.strava import _normalize_activity

        raw = {
            "id": 1,
            "start_date": "2026-08-26T07:30:00Z",
            "sport_type": "Ride",
            "moving_time": 1200,
        }
        assert _normalize_activity(raw).commute is False
