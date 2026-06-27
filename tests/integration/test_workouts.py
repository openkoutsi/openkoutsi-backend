"""
Integration tests for /api/workouts endpoints.
"""


_STEP = {
    "kind": "step",
    "step_type": "active",
    "duration": {"type": "time", "seconds": 1800},
    "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 90.0}},
}

_WARMUP = {
    "kind": "step",
    "step_type": "warmup",
    "duration": {"type": "time", "seconds": 600},
}

_REPEAT = {
    "kind": "repeat",
    "repeat_count": 3,
    "steps": [
        {"kind": "step", "step_type": "active", "duration": {"type": "time", "seconds": 300},
         "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 105.0}}},
        {"kind": "step", "step_type": "recovery", "duration": {"type": "time", "seconds": 120}},
    ],
}

_WORKOUT_BODY = {
    "name": "Threshold Session",
    "description": "2x20 min threshold",
    "sport_type": "Ride",
    "steps": [_WARMUP, _STEP],
}


class TestListExportFormats:
    async def test_returns_known_formats(self, client, auth_headers):
        resp = await client.get("/api/workouts/export/formats", headers=auth_headers)
        assert resp.status_code == 200
        keys = {item["key"] for item in resp.json()}
        assert "zwift" in keys
        assert "json" in keys

    async def test_unauthenticated_still_works(self, client):
        # This endpoint has no auth requirement (no Depends)
        resp = await client.get("/api/workouts/export/formats")
        assert resp.status_code == 200


class TestListWorkouts:
    async def test_empty_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/workouts/", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_created_workout(self, client, auth_headers):
        await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        resp = await client.get("/api/workouts/", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["name"] == "Threshold Session"

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/workouts/")
        assert resp.status_code == 401


class TestCreateWorkout:
    async def test_creates_workout_with_correct_fields(self, client, auth_headers):
        resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Threshold Session"
        assert data["sport_type"] == "Ride"
        assert len(data["steps"]) == 2

    async def test_estimated_duration_calculated(self, client, auth_headers):
        resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        assert resp.status_code == 201
        # warmup: 600s + active: 1800s = 2400s
        assert resp.json()["estimated_duration_s"] == 2400

    async def test_creates_workout_with_repeat_block(self, client, auth_headers):
        body = {"name": "Intervals", "steps": [_REPEAT]}
        resp = await client.post("/api/workouts/", json=body, headers=auth_headers)
        assert resp.status_code == 201
        assert len(resp.json()["steps"]) == 1

    async def test_double_nested_repeat_returns_422(self, client, auth_headers):
        # max_depth > 1 means we need 3 levels: outer → mid → inner
        inner_repeat = {
            "kind": "repeat",
            "repeat_count": 2,
            "steps": [{"kind": "step", "step_type": "active",
                        "duration": {"type": "time", "seconds": 60}}],
        }
        mid_repeat = {"kind": "repeat", "repeat_count": 2, "steps": [inner_repeat]}
        outer_repeat = {"kind": "repeat", "repeat_count": 2, "steps": [mid_repeat]}
        body = {"name": "Deeply Nested", "steps": [outer_repeat]}
        resp = await client.post("/api/workouts/", json=body, headers=auth_headers)
        assert resp.status_code == 422

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post("/api/workouts/", json=_WORKOUT_BODY)
        assert resp.status_code == 401


class TestGetWorkout:
    async def test_returns_workout_by_id(self, client, auth_headers):
        create_resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        workout_id = create_resp.json()["id"]

        resp = await client.get(f"/api/workouts/{workout_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == workout_id

    async def test_nonexistent_returns_404(self, client, auth_headers):
        resp = await client.get("/api/workouts/no-such-id", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/workouts/some-id")
        assert resp.status_code == 401


class TestUpdateWorkout:
    async def test_update_name(self, client, auth_headers):
        create_resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        workout_id = create_resp.json()["id"]

        resp = await client.put(
            f"/api/workouts/{workout_id}",
            json={"name": "Renamed Workout"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Renamed Workout"

    async def test_update_steps_recalculates_duration(self, client, auth_headers):
        create_resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        workout_id = create_resp.json()["id"]

        new_step = {
            "kind": "step",
            "step_type": "active",
            "duration": {"type": "time", "seconds": 3600},
        }
        resp = await client.put(
            f"/api/workouts/{workout_id}",
            json={"steps": [new_step]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["estimated_duration_s"] == 3600

    async def test_nonexistent_returns_404(self, client, auth_headers):
        resp = await client.put("/api/workouts/no-such-id", json={"name": "X"}, headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.put("/api/workouts/some-id", json={"name": "X"})
        assert resp.status_code == 401


class TestDeleteWorkout:
    async def test_delete_returns_204(self, client, auth_headers):
        create_resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        workout_id = create_resp.json()["id"]

        resp = await client.delete(f"/api/workouts/{workout_id}", headers=auth_headers)
        assert resp.status_code == 204

    async def test_deleted_workout_returns_404(self, client, auth_headers):
        create_resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        workout_id = create_resp.json()["id"]
        await client.delete(f"/api/workouts/{workout_id}", headers=auth_headers)

        resp = await client.get(f"/api/workouts/{workout_id}", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.delete("/api/workouts/some-id")
        assert resp.status_code == 401


class TestExportWorkout:
    async def test_export_json_format(self, client, auth_headers):
        create_resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        workout_id = create_resp.json()["id"]

        resp = await client.get(f"/api/workouts/{workout_id}/export/json", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")

    async def test_export_zwift_format(self, client, auth_headers):
        create_resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        workout_id = create_resp.json()["id"]

        resp = await client.get(f"/api/workouts/{workout_id}/export/zwift", headers=auth_headers)
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]

    async def test_unknown_format_returns_404(self, client, auth_headers):
        create_resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
        workout_id = create_resp.json()["id"]

        resp = await client.get(
            f"/api/workouts/{workout_id}/export/nonexistent_format", headers=auth_headers
        )
        assert resp.status_code == 404

    async def test_nonexistent_workout_returns_404(self, client, auth_headers):
        resp = await client.get("/api/workouts/no-such-id/export/json", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/workouts/some-id/export/json")
        assert resp.status_code == 401
