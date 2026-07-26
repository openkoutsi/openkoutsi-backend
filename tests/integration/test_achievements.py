"""Integration tests for /api/achievements (issue #33)."""

from datetime import date, datetime, timedelta, timezone


def _iso(day: date, hour: int = 10) -> str:
    return datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).isoformat()


async def _log_ride(client, auth_headers, *, day, **fields):
    payload = {"sport_type": "Ride", "start_time": _iso(day), "duration_s": 3600}
    payload.update(fields)
    return await client.post("/api/activities", json=payload, headers=auth_headers)


class TestGetAchievements:
    async def test_new_athlete_gets_a_catalogue_and_no_unlocks(self, client, auth_headers):
        resp = await client.get("/api/achievements", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["unlocked"] == []
        assert body["disabled"] is False
        # The catalogue is always present so the UI can show what's reachable.
        assert any(d["id"] == "activity_count" for d in body["catalogue"])

    async def test_unauthenticated_returns_401(self, client):
        assert (await client.get("/api/achievements")).status_code == 401

    async def test_catalogue_carries_no_display_text(self, client, auth_headers):
        """Names live in the web app's i18n files, never in the API payload."""
        body = (await client.get("/api/achievements", headers=auth_headers)).json()
        for definition in body["catalogue"]:
            assert set(definition) == {"id", "category", "tiers", "unit", "requires"}

    async def test_logging_a_ride_unlocks_the_first_tier(self, client, auth_headers):
        await _log_ride(client, auth_headers, day=date.today())

        body = (await client.get("/api/achievements", headers=auth_headers)).json()

        unlocked = {(u["achievement_id"], u["tier"]) for u in body["unlocked"]}
        assert ("activity_count", 1.0) in unlocked
        assert body["progress"]["activity_count"] == 1.0

    async def test_deleting_the_ride_revokes_the_unlock(self, client, auth_headers):
        created = await _log_ride(client, auth_headers, day=date.today())
        activity_id = created.json()["id"]

        body = (await client.get("/api/achievements", headers=auth_headers)).json()
        assert body["unlocked"]

        assert (
            await client.delete(f"/api/activities/{activity_id}", headers=auth_headers)
        ).status_code == 204

        body = (await client.get("/api/achievements", headers=auth_headers)).json()
        assert body["unlocked"] == []

    async def test_elevation_badges_are_absent_without_elevation_data(
        self, client, auth_headers
    ):
        await _log_ride(client, auth_headers, day=date.today(), distance_m=40_000)

        body = (await client.get("/api/achievements", headers=auth_headers)).json()

        ids = {d["id"] for d in body["catalogue"]}
        assert "total_distance" in ids
        assert "everesting" not in ids

    async def test_unlocked_never_contains_an_id_the_catalogue_omits(
        self, client, auth_headers, monkeypatch
    ):
        """`unlocked` must be filtered by availability like its three siblings.

        With today's catalogue the property holds by coincidence: every gated
        achievement's lowest tier needs the very data that gates it, so an
        unavailable achievement has no tier dates and no row. That makes the
        happy path untestable — so this constructs the case the coincidence
        rules out, by gating an achievement on data its tiers don't need.

        Without the filter the athlete receives an `achievement_id` the frontend
        has no definition, tiers or unit for.
        """
        from dataclasses import replace

        from openkoutsi import achievements as catalogue_mod
        from backend.app.api import achievements as api_mod

        # activity_count is earned from any activity, but now claims to need
        # elevation — exactly the shape a future "created a plan" badge would have.
        gated = replace(catalogue_mod.CATALOGUE_BY_ID["activity_count"], requires="elevation")
        monkeypatch.setitem(catalogue_mod.CATALOGUE_BY_ID, "activity_count", gated)
        monkeypatch.setattr(
            api_mod,
            "CATALOGUE",
            tuple(gated if d.id == "activity_count" else d for d in catalogue_mod.CATALOGUE),
        )

        # No elevation anywhere, so "elevation" is unavailable — but the ride
        # still earns activity_count, and the reconcile still persists it.
        await _log_ride(client, auth_headers, day=date.today(), distance_m=20_000)

        body = (await client.get("/api/achievements", headers=auth_headers)).json()

        catalogue_ids = {d["id"] for d in body["catalogue"]}
        assert "activity_count" not in catalogue_ids
        unlocked_ids = {u["achievement_id"] for u in body["unlocked"]}
        assert unlocked_ids <= catalogue_ids, unlocked_ids - catalogue_ids
        assert set(body["progress"]) <= catalogue_ids
        assert {s["id"] for s in body["streaks"]} <= catalogue_ids

    async def test_the_response_stays_consistent_as_data_comes_and_goes(
        self, client, auth_headers
    ):
        """The same property over real data, before and after a climb is removed."""
        climb = await _log_ride(
            client, auth_headers, day=date.today(), distance_m=40_000, elevation_m=1_200,
        )
        await _log_ride(
            client, auth_headers, day=date.today() - timedelta(days=1), distance_m=20_000,
        )

        for _ in range(2):
            body = (await client.get("/api/achievements", headers=auth_headers)).json()
            catalogue_ids = {d["id"] for d in body["catalogue"]}
            assert {u["achievement_id"] for u in body["unlocked"]} <= catalogue_ids
            await client.delete(
                f"/api/activities/{climb.json()['id']}", headers=auth_headers
            )

    async def test_elevation_badges_appear_once_a_climb_is_recorded(
        self, client, auth_headers
    ):
        await _log_ride(
            client, auth_headers, day=date.today(), distance_m=40_000, elevation_m=900,
        )

        body = (await client.get("/api/achievements", headers=auth_headers)).json()

        assert "everesting" in {d["id"] for d in body["catalogue"]}


class TestStreaks:
    async def test_streaks_endpoint_reports_the_current_run(self, client, auth_headers):
        today = date.today()
        for weeks_ago in range(3):
            await _log_ride(client, auth_headers, day=today - timedelta(weeks=weeks_ago))

        resp = await client.get("/api/achievements/streaks", headers=auth_headers)

        assert resp.status_code == 200
        active = next(s for s in resp.json() if s["id"] == "streak_active_weeks")
        assert active["current"] == 3
        assert active["longest"] == 3

    async def test_streaks_are_included_in_the_main_response(self, client, auth_headers):
        await _log_ride(client, auth_headers, day=date.today())

        body = (await client.get("/api/achievements", headers=auth_headers)).json()

        assert any(s["id"] == "streak_active_weeks" for s in body["streaks"])


class TestOptOut:
    async def test_disabled_returns_an_empty_payload(self, client, auth_headers):
        await _log_ride(client, auth_headers, day=date.today())
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"gamification": False}},
            headers=auth_headers,
        )

        body = (await client.get("/api/achievements", headers=auth_headers)).json()

        assert body["disabled"] is True
        assert body["catalogue"] == []
        assert body["unlocked"] == []

    async def test_streaks_are_empty_when_disabled(self, client, auth_headers):
        await _log_ride(client, auth_headers, day=date.today())
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"gamification": False}},
            headers=auth_headers,
        )

        resp = await client.get("/api/achievements/streaks", headers=auth_headers)

        assert resp.json() == []

    async def test_re_enabling_brings_the_unlocks_back(self, client, auth_headers):
        await _log_ride(client, auth_headers, day=date.today())
        before = (await client.get("/api/achievements", headers=auth_headers)).json()

        await client.patch(
            "/api/athlete",
            json={"app_settings": {"gamification": False}},
            headers=auth_headers,
        )
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"gamification": True}},
            headers=auth_headers,
        )

        after = (await client.get("/api/achievements", headers=auth_headers)).json()

        assert {(u["achievement_id"], u["tier"]) for u in after["unlocked"]} == {
            (u["achievement_id"], u["tier"]) for u in before["unlocked"]
        }


class TestSeenMarker:
    async def test_marking_seen_flips_the_flag(self, client, auth_headers):
        await _log_ride(client, auth_headers, day=date.today())

        body = (await client.get("/api/achievements", headers=auth_headers)).json()
        assert all(u["seen"] is False for u in body["unlocked"])

        assert (
            await client.post("/api/achievements/seen", headers=auth_headers)
        ).status_code == 204

        body = (await client.get("/api/achievements", headers=auth_headers)).json()
        assert all(u["seen"] is True for u in body["unlocked"])

    async def test_seen_is_idempotent(self, client, auth_headers):
        await _log_ride(client, auth_headers, day=date.today())
        await client.post("/api/achievements/seen", headers=auth_headers)
        resp = await client.post("/api/achievements/seen", headers=auth_headers)
        assert resp.status_code == 204


class TestEngagementBadges:
    async def test_editing_rpe_and_notes_feeds_the_engagement_counters(
        self, client, auth_headers
    ):
        created = await _log_ride(client, auth_headers, day=date.today())
        activity_id = created.json()["id"]

        await client.patch(
            f"/api/activities/{activity_id}",
            json={"rpe": 6, "notes": "Felt strong"},
            headers=auth_headers,
        )

        body = (await client.get("/api/achievements", headers=auth_headers)).json()

        assert body["progress"]["rpe_recorded"] == 1.0
        assert body["progress"]["notes_written"] == 1.0

    async def test_race_label_counts_toward_race_day(self, client, auth_headers):
        created = await _log_ride(client, auth_headers, day=date.today())
        activity_id = created.json()["id"]

        await client.patch(
            f"/api/activities/{activity_id}",
            json={"labels": ["race"]},
            headers=auth_headers,
        )

        body = (await client.get("/api/achievements", headers=auth_headers)).json()

        unlocked = {(u["achievement_id"], u["tier"]) for u in body["unlocked"]}
        assert ("race_day", 1.0) in unlocked


class TestInboxNotification:
    async def test_unlocks_land_in_the_inbox(self, client, auth_headers):
        await _log_ride(client, auth_headers, day=date.today())

        messages = (await client.get("/api/messages", headers=auth_headers)).json()

        unlocked = [m for m in messages["items"] if m["type"] == "achievement_unlocked"]
        assert len(unlocked) == 1
        assert unlocked[0]["data"]["count"] >= 1

    async def test_a_batch_of_unlocks_produces_one_message_not_many(
        self, client, auth_headers
    ):
        """A season of back-filled history must not carpet-bomb the inbox."""
        await _log_ride(
            client, auth_headers, day=date.today(),
            duration_s=6 * 3600, distance_m=180_000, elevation_m=2_500,
        )

        messages = (await client.get("/api/messages", headers=auth_headers)).json()

        unlocked = [m for m in messages["items"] if m["type"] == "achievement_unlocked"]
        assert len(unlocked) == 1
        assert unlocked[0]["data"]["count"] > 1

    async def test_repeat_reads_do_not_re_notify(self, client, auth_headers):
        await _log_ride(client, auth_headers, day=date.today())
        for _ in range(3):
            await client.get("/api/achievements", headers=auth_headers)

        messages = (await client.get("/api/messages", headers=auth_headers)).json()

        unlocked = [m for m in messages["items"] if m["type"] == "achievement_unlocked"]
        assert len(unlocked) == 1


class TestExport:
    async def test_achievements_are_included_in_the_data_export(
        self, client, auth_headers
    ):
        import io
        import json
        import zipfile

        await _log_ride(client, auth_headers, day=date.today())

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            assert "achievements.json" in zf.namelist()
            payload = json.loads(zf.read("achievements.json"))

        assert any(u["achievement_id"] == "activity_count" for u in payload)


class TestGoalBadges:
    async def test_achieving_a_goal_unlocks_and_reverting_revokes(
        self, client, auth_headers
    ):
        created = await client.post(
            "/api/goals", json={"title": "First century"}, headers=auth_headers
        )
        goal_id = created.json()["id"]

        await client.put(
            f"/api/goals/{goal_id}", json={"status": "achieved"}, headers=auth_headers
        )
        body = (await client.get("/api/achievements", headers=auth_headers)).json()
        assert ("goals_reached", 1.0) in {
            (u["achievement_id"], u["tier"]) for u in body["unlocked"]
        }

        await client.put(
            f"/api/goals/{goal_id}", json={"status": "active"}, headers=auth_headers
        )
        body = (await client.get("/api/achievements", headers=auth_headers)).json()
        assert ("goals_reached", 1.0) not in {
            (u["achievement_id"], u["tier"]) for u in body["unlocked"]
        }
