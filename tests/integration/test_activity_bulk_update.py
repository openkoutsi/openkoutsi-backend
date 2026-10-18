"""Integration tests for ``POST /api/activities/bulk``.

The bulk edit is the single ``PATCH`` applied to a selection, and the tests hold
it to that: a label added in bulk answers a pending suggestion, a bike set in
bulk is stamped ``manual``, and nothing changes at all when any part of the
request is invalid.
"""
from __future__ import annotations

BULK = "/api/activities/bulk"

COMMUTE_RULE = {
    "id": "to-work",
    "name": "To work",
    "enabled": True,
    "auto_apply": False,
    "sport_types": ["Ride"],
    "max_distance_m": 10000,
    "windows": [{"start": "06:30", "end": "08:30"}],
    "weekdays": [0, 1, 2, 3, 4],
}


async def _ride(client, auth_headers, *, sport_type="Ride", start="2026-08-26T12:00:00Z"):
    resp = await client.post(
        "/api/activities",
        json={
            "sport_type": sport_type,
            "start_time": start,
            "duration_s": 1800,
            "distance_m": 5400.0,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _get(client, auth_headers, activity_id):
    resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _bike(client, auth_headers, **overrides):
    payload = {"name": "Road bike", "tyre_width_mm": 28, "riding_position": "hoods"}
    payload.update(overrides)
    resp = await client.post("/api/bikes", json=payload, headers=auth_headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


class TestLabels:
    async def test_add_keeps_the_labels_each_ride_already_has(self, client, auth_headers):
        a = await _ride(client, auth_headers)
        b = await _ride(client, auth_headers)
        await client.patch(
            f"/api/activities/{a['id']}", json={"labels": ["race"]}, headers=auth_headers
        )

        resp = await client.post(
            BULK,
            json={"activity_ids": [a["id"], b["id"]], "add_labels": ["commute"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"updated": 2, "not_found": []}
        assert (await _get(client, auth_headers, a["id"]))["labels"] == ["race", "commute"]
        assert (await _get(client, auth_headers, b["id"]))["labels"] == ["commute"]

    async def test_remove_takes_only_that_label_off(self, client, auth_headers):
        a = await _ride(client, auth_headers)
        await client.patch(
            f"/api/activities/{a['id']}",
            json={"labels": ["race", "commute"]},
            headers=auth_headers,
        )
        resp = await client.post(
            BULK,
            json={"activity_ids": [a["id"]], "remove_labels": ["commute"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        assert (await _get(client, auth_headers, a["id"]))["labels"] == ["race"]

    async def test_adding_answers_a_pending_suggestion(self, client, auth_headers):
        """Same meaning as labelling one ride by hand (issue #63)."""
        resp = await client.patch(
            "/api/athlete",
            json={"app_settings": {"commute_rules": [COMMUTE_RULE]}},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        ride = await _ride(client, auth_headers, start="2026-08-26T07:30:00Z")
        assert ride["label_suggestions"]["commute"]["state"] == "pending"

        await client.post(
            BULK,
            json={"activity_ids": [ride["id"]], "add_labels": ["commute"]},
            headers=auth_headers,
        )
        after = await _get(client, auth_headers, ride["id"])
        assert after["labels"] == ["commute"]
        assert after["label_suggestions"]["commute"]["state"] == "accepted"

    async def test_other_labels_pending_suggestions_are_left_waiting(
        self, client, auth_headers
    ):
        """Adding `race` decides nothing about `commute`."""
        resp = await client.patch(
            "/api/athlete",
            json={"app_settings": {"commute_rules": [COMMUTE_RULE]}},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        ride = await _ride(client, auth_headers, start="2026-08-26T07:30:00Z")
        assert ride["label_suggestions"]["commute"]["state"] == "pending"

        await client.post(
            BULK,
            json={"activity_ids": [ride["id"]], "add_labels": ["race"]},
            headers=auth_headers,
        )
        after = await _get(client, auth_headers, ride["id"])
        assert after["labels"] == ["race"]
        assert after["label_suggestions"]["commute"]["state"] == "pending"

    async def test_unknown_label_is_rejected(self, client, auth_headers):
        a = await _ride(client, auth_headers)
        resp = await client.post(
            BULK, json={"activity_ids": [a["id"]], "add_labels": ["nope"]}, headers=auth_headers
        )
        assert resp.status_code == 422

    async def test_adding_and_removing_the_same_label_is_rejected(self, client, auth_headers):
        a = await _ride(client, auth_headers)
        resp = await client.post(
            BULK,
            json={
                "activity_ids": [a["id"]],
                "add_labels": ["race"],
                "remove_labels": ["race"],
            },
            headers=auth_headers,
        )
        assert resp.status_code == 422


class TestBikeAndCategory:
    async def test_bike_is_stamped_manual(self, client, auth_headers):
        bike = await _bike(client, auth_headers)
        a = await _ride(client, auth_headers)
        b = await _ride(client, auth_headers)
        resp = await client.post(
            BULK,
            json={"activity_ids": [a["id"], b["id"]], "bike_id": bike["id"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        for ride in (a, b):
            after = await _get(client, auth_headers, ride["id"])
            assert after["bike_id"] == bike["id"]
            assert after["bike_source"] == "manual"

    async def test_null_bike_means_none_of_mine(self, client, auth_headers):
        await _bike(client, auth_headers, default_sports=["Ride"])
        a = await _ride(client, auth_headers)
        assert a["bike_source"] == "auto"
        await client.post(
            BULK, json={"activity_ids": [a["id"]], "bike_id": None}, headers=auth_headers
        )
        after = await _get(client, auth_headers, a["id"])
        assert after["bike_id"] is None
        assert after["bike_source"] == "manual"

    async def test_someone_elses_bike_is_404_and_changes_nothing(self, client, auth_headers):
        a = await _ride(client, auth_headers)
        resp = await client.post(
            BULK,
            json={"activity_ids": [a["id"]], "bike_id": "not-a-bike", "add_labels": ["race"]},
            headers=auth_headers,
        )
        assert resp.status_code == 404
        after = await _get(client, auth_headers, a["id"])
        assert after["labels"] in (None, [])
        assert after["bike_source"] is None

    async def test_category_is_set_and_cleared(self, client, auth_headers):
        a = await _ride(client, auth_headers)
        resp = await client.post(
            BULK,
            json={"activity_ids": [a["id"]], "workout_category": "endurance"},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        assert (await _get(client, auth_headers, a["id"]))["workout_category"] == "endurance"

        await client.post(
            BULK, json={"activity_ids": [a["id"]], "workout_category": None}, headers=auth_headers
        )
        assert (await _get(client, auth_headers, a["id"]))["workout_category"] is None

    async def test_unknown_category_is_rejected(self, client, auth_headers):
        a = await _ride(client, auth_headers)
        resp = await client.post(
            BULK,
            json={"activity_ids": [a["id"]], "workout_category": "napping"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_omitted_fields_are_left_alone(self, client, auth_headers):
        bike = await _bike(client, auth_headers)
        a = await _ride(client, auth_headers)
        await client.patch(
            f"/api/activities/{a['id']}",
            json={"bike_id": bike["id"], "workout_category": "tempo"},
            headers=auth_headers,
        )
        await client.post(
            BULK, json={"activity_ids": [a["id"]], "add_labels": ["race"]}, headers=auth_headers
        )
        after = await _get(client, auth_headers, a["id"])
        assert after["bike_id"] == bike["id"]
        assert after["workout_category"] == "tempo"


class TestSelection:
    async def test_unknown_ids_are_reported_not_fatal(self, client, auth_headers):
        a = await _ride(client, auth_headers)
        resp = await client.post(
            BULK,
            json={"activity_ids": [a["id"], "missing", a["id"]], "add_labels": ["race"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"updated": 1, "not_found": ["missing"]}

    async def test_an_empty_edit_is_rejected(self, client, auth_headers):
        a = await _ride(client, auth_headers)
        resp = await client.post(BULK, json={"activity_ids": [a["id"]]}, headers=auth_headers)
        assert resp.status_code == 422

    async def test_an_empty_selection_is_rejected(self, client, auth_headers):
        resp = await client.post(
            BULK, json={"activity_ids": [], "add_labels": ["race"]}, headers=auth_headers
        )
        assert resp.status_code == 422
