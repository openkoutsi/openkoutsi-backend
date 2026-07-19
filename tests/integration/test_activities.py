"""
Integration tests for /api/activities endpoints.

FIT upload tests call process_fit_file() and recalculate_from() directly
(bypassing the suppressed background task) to verify end-to-end behavior.
"""
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from backend.app.core.auth import create_access_token
from backend.app.models.user_orm import Activity, ActivitySource, ActivityStream, Athlete

from ._fit_fixtures import capabilities, fit_fixture_params


# ── Manual activity creation ───────────────────────────────────────────────────

class TestCreateManualActivity:
    async def test_explicit_tss_stored_as_is(self, client, auth_headers):
        resp = await client.post(
            "/api/activities",
            json={
                "sport_type": "Ride",
                "start_time": "2025-06-01T10:00:00Z",
                "duration_s": 3600,
                "load": 100.0,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["load"] == 100.0
        assert data["status"] == "processed"

    async def test_no_tss_inputs_gives_null_tss(self, client, auth_headers):
        resp = await client.post(
            "/api/activities",
            json={
                "sport_type": "Ride",
                "start_time": "2025-06-02T10:00:00Z",
                "duration_s": 3600,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["load"] is None

    async def test_rpe_based_tss(self, client, auth_headers):
        # load = (duration_s / 3600) * rpe^2 * 10
        resp = await client.post(
            "/api/activities",
            json={
                "sport_type": "Ride",
                "start_time": "2025-06-03T10:00:00Z",
                "duration_s": 3600,
                "rpe": 7,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        expected_tss = (3600 / 3600) * 49 * 10  # 490
        assert resp.json()["load"] == pytest.approx(expected_tss, rel=1e-6)

    async def test_hr_based_tss_requires_athlete_max_hr(self, client, auth_headers):
        # Set max_hr on the athlete first
        await client.patch("/api/athlete", json={"max_hr": 185}, headers=auth_headers)
        resp = await client.post(
            "/api/activities",
            json={
                "sport_type": "Ride",
                "start_time": "2025-06-04T10:00:00Z",
                "duration_s": 3600,
                "avg_hr": 150.0,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["load"] is not None
        assert resp.json()["load"] > 0

    async def test_empty_body_returns_422(self, client, auth_headers):
        # Every field is optional, but a completely empty submission is rejected.
        resp = await client.post("/api/activities", json={}, headers=auth_headers)
        assert resp.status_code == 422

    async def test_single_field_is_enough(self, client, auth_headers):
        # Only a sport type, nothing else — still a valid manual activity.
        resp = await client.post(
            "/api/activities",
            json={"sport_type": "Run"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["sport_type"] == "Run"
        assert data["duration_s"] is None
        assert data["load"] is None
        assert data["status"] == "processed"
        assert "manual" in data["sources"]

    async def test_power_hr_cadence_stored(self, client, auth_headers):
        resp = await client.post(
            "/api/activities",
            json={
                "sport_type": "Ride",
                "start_time": "2025-06-05T10:00:00Z",
                "duration_s": 3600,
                "avg_power": 220.0,
                "max_hr": 178.0,
                "avg_cadence": 88.0,
                "distance_m": 30000.0,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["avg_power"] == 220.0
        assert data["max_hr"] == 178.0
        assert data["avg_cadence"] == 88.0
        assert data["distance_m"] == 30000.0

    async def test_no_duration_gives_null_tss(self, client, auth_headers):
        # rpe/avg_hr can't derive Load without a duration to scale by.
        resp = await client.post(
            "/api/activities",
            json={"sport_type": "Run", "avg_hr": 150.0, "rpe": 7},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["load"] is None

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-06-01T10:00:00Z", "duration_s": 3600},
        )
        assert resp.status_code == 401


# ── Activity listing ───────────────────────────────────────────────────────────

class TestListActivities:
    async def test_empty_list_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/activities", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    async def test_returns_created_activities(self, client, auth_headers):
        for i in range(3):
            await client.post(
                "/api/activities",
                json={
                    "sport_type": "Ride",
                    "start_time": f"2025-06-0{i+1}T10:00:00Z",
                    "duration_s": 3600,
                },
                headers=auth_headers,
            )
        resp = await client.get("/api/activities", headers=auth_headers)
        assert resp.json()["total"] == 3

    async def test_sport_type_filter(self, client, auth_headers):
        await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-01-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        await client.post(
            "/api/activities",
            json={"sport_type": "Run", "start_time": "2025-01-02T10:00:00Z", "duration_s": 1800},
            headers=auth_headers,
        )
        resp = await client.get("/api/activities?sport_type=Ride", headers=auth_headers)
        assert resp.json()["total"] == 1
        assert resp.json()["items"][0]["sport_type"] == "Ride"

    async def test_date_range_filter(self, client, auth_headers):
        for month in [1, 3, 6]:
            await client.post(
                "/api/activities",
                json={
                    "sport_type": "Ride",
                    "start_time": f"2025-{month:02d}-15T10:00:00Z",
                    "duration_s": 3600,
                },
                headers=auth_headers,
            )
        resp = await client.get(
            "/api/activities?start=2025-02-01&end=2025-05-01",
            headers=auth_headers,
        )
        assert resp.json()["total"] == 1

    async def test_pagination(self, client, auth_headers):
        for i in range(5):
            await client.post(
                "/api/activities",
                json={
                    "sport_type": "Ride",
                    "start_time": f"2025-06-0{i+1}T10:00:00Z",
                    "duration_s": 3600,
                },
                headers=auth_headers,
            )
        resp = await client.get("/api/activities?page=2&page_size=2", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["page"] == 2

    async def _make_labeled(self, client, auth_headers, day, labels):
        resp = await client.post(
            "/api/activities",
            json={
                "sport_type": "Ride",
                "start_time": f"2025-06-{day:02d}T10:00:00Z",
                "duration_s": 3600,
            },
            headers=auth_headers,
        )
        activity_id = resp.json()["id"]
        await client.patch(
            f"/api/activities/{activity_id}",
            json={"labels": labels},
            headers=auth_headers,
        )
        return activity_id

    async def test_labels_filter_includes_only_matching(self, client, auth_headers):
        await self._make_labeled(client, auth_headers, 1, ["race"])
        await self._make_labeled(client, auth_headers, 2, ["commute"])
        await self._make_labeled(client, auth_headers, 3, [])

        resp = await client.get("/api/activities?labels=race", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["labels"] == ["race"]

    async def test_labels_filter_matches_any_of_several(self, client, auth_headers):
        await self._make_labeled(client, auth_headers, 1, ["race"])
        await self._make_labeled(client, auth_headers, 2, ["commute"])
        await self._make_labeled(client, auth_headers, 3, [])

        resp = await client.get(
            "/api/activities?labels=race&labels=commute", headers=auth_headers
        )
        assert resp.json()["total"] == 2

    async def test_exclude_labels_filters_out_matching(self, client, auth_headers):
        await self._make_labeled(client, auth_headers, 1, ["race"])
        await self._make_labeled(client, auth_headers, 2, ["commute"])
        await self._make_labeled(client, auth_headers, 3, [])

        resp = await client.get(
            "/api/activities?exclude_labels=commute", headers=auth_headers
        )
        data = resp.json()
        assert data["total"] == 2
        assert all("commute" not in item["labels"] for item in data["items"])

    async def test_labels_and_exclude_labels_combine(self, client, auth_headers):
        await self._make_labeled(client, auth_headers, 1, ["race"])
        await self._make_labeled(client, auth_headers, 2, ["race", "commute"])

        resp = await client.get(
            "/api/activities?labels=race&exclude_labels=commute",
            headers=auth_headers,
        )
        assert resp.json()["total"] == 1

    async def test_unknown_label_filter_returns_422(self, client, auth_headers):
        resp = await client.get("/api/activities?labels=bogus", headers=auth_headers)
        assert resp.status_code == 422

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/activities")
        assert resp.status_code == 401


# ── Activity detail ────────────────────────────────────────────────────────────

class TestGetActivity:
    async def test_returns_activity_detail(self, client, auth_headers):
        create_resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-01-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        activity_id = create_resp.json()["id"]
        resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == activity_id
        assert resp.json()["streams"] == {}

    async def test_nonexistent_activity_returns_404(self, client, auth_headers):
        resp = await client.get("/api/activities/nonexistent-id", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/activities/some-id")
        assert resp.status_code == 401


# ── Activity deletion ──────────────────────────────────────────────────────────

class TestDeleteActivity:
    async def test_delete_returns_204(self, client, auth_headers):
        create_resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-01-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        activity_id = create_resp.json()["id"]
        resp = await client.delete(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.status_code == 204

    async def test_deleted_activity_returns_404(self, client, auth_headers):
        create_resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-01-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        activity_id = create_resp.json()["id"]
        await client.delete(f"/api/activities/{activity_id}", headers=auth_headers)
        resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.delete("/api/activities/some-id")
        assert resp.status_code == 401


# ── FIT file upload ────────────────────────────────────────────────────────────

class TestFitUpload:
    @pytest.mark.parametrize("fit_path", fit_fixture_params())
    async def test_upload_fit_file_processes_correctly(self, fit_path, client, auth_headers, session):
        """Upload each FIT fixture and verify it's processed into a complete activity."""
        caps = capabilities(fit_path)
        # Set FTP so power-based Load can be calculated
        await client.patch("/api/athlete", json={"ftp": 280}, headers=auth_headers)

        with open(fit_path, "rb") as f:
            resp = await client.post(
                "/api/activities/upload",
                files={"file": ("test.fit", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        activity_id = data["id"]

        # Load the activity, its upload source, and the athlete from the test session
        act_result = await session.execute(
            select(Activity).where(Activity.id == activity_id)
        )
        activity = act_result.scalar_one()

        src_result = await session.execute(
            select(ActivitySource).where(
                ActivitySource.activity_id == activity_id,
                ActivitySource.provider == "upload",
            )
        )
        upload_src = src_result.scalar_one()

        ath_result = await session.execute(
            select(Athlete).where(Athlete.id == activity.athlete_id)
        )
        athlete = ath_result.scalar_one()

        # Process the FIT file directly (bypassing the suppressed background task)
        from backend.app.services.fit_processor import process_fit_file
        from backend.app.services.metrics_engine import recalculate_from

        await process_fit_file(upload_src.fit_file_path, athlete, activity, session)

        start_date = activity.start_time.date() if activity.start_time else None
        if start_date:
            await recalculate_from(athlete.id, start_date, session)

        await session.refresh(activity)
        assert activity.status == "processed"
        # Power-derived metrics only exist when the fixture actually has a power stream.
        if caps.has_power:
            assert activity.weighted_power is not None
            assert activity.load is not None
        else:
            assert activity.weighted_power is None

        # Torque is derived and stored whenever both power and cadence are present.
        streams_result = await session.execute(
            select(ActivityStream).where(ActivityStream.activity_id == activity_id)
        )
        stored_streams = {s.stream_type for s in streams_result.scalars()}
        if caps.has_power and caps.has_cadence:
            assert "torque" in stored_streams
        else:
            assert "torque" not in stored_streams

    async def test_has_fit_file_false_for_manual_activity(self, client, auth_headers):
        resp = await client.post(
            "/api/activities",
            json={
                "sport_type": "Ride",
                "start_time": "2025-01-01T10:00:00Z",
                "duration_s": 3600,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["has_fit_file"] is False

    @pytest.mark.parametrize("fit_path", fit_fixture_params())
    async def test_has_fit_file_true_for_uploaded_activity(self, fit_path, client, auth_headers):
        with open(fit_path, "rb") as f:
            resp = await client.post(
                "/api/activities/upload",
                files={"file": ("test.fit", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert resp.status_code == 201
        assert resp.json()["has_fit_file"] is True

    @pytest.mark.parametrize("fit_path", fit_fixture_params())
    async def test_duplicate_fit_upload_returns_409(self, fit_path, client, auth_headers, session):
        """Uploading a file whose start time matches an existing activity returns 409."""
        # First upload
        with open(fit_path, "rb") as f:
            resp1 = await client.post(
                "/api/activities/upload",
                files={"file": ("test.fit", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert resp1.status_code == 201
        activity_id = resp1.json()["id"]

        # Process the file so the activity gets a start_time (needed for duplicate detection)
        act_result = await session.execute(select(Activity).where(Activity.id == activity_id))
        activity = act_result.scalar_one()
        src_result = await session.execute(
            select(ActivitySource).where(
                ActivitySource.activity_id == activity_id,
                ActivitySource.provider == "upload",
            )
        )
        upload_src = src_result.scalar_one()
        ath_result = await session.execute(select(Athlete).where(Athlete.id == activity.athlete_id))
        athlete = ath_result.scalar_one()

        from backend.app.services.fit_processor import process_fit_file
        await process_fit_file(upload_src.fit_file_path, athlete, activity, session)

        # Second upload of the same file — should be rejected as duplicate
        with open(fit_path, "rb") as f:
            resp2 = await client.post(
                "/api/activities/upload",
                files={"file": ("test.fit", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert resp2.status_code == 409


# ── Individual FIT file download ───────────────────────────────────────────────

class TestDownloadFitFile:
    async def test_manual_activity_has_no_fit_file(self, client, auth_headers):
        create_resp = await client.post(
            "/api/activities",
            json={
                "sport_type": "Ride",
                "start_time": "2025-01-01T10:00:00Z",
                "duration_s": 3600,
            },
            headers=auth_headers,
        )
        activity_id = create_resp.json()["id"]
        resp = await client.get(f"/api/activities/{activity_id}/fit", headers=auth_headers)
        assert resp.status_code == 404

    async def test_nonexistent_activity_returns_404(self, client, auth_headers):
        resp = await client.get("/api/activities/nonexistent-id/fit", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/activities/some-id/fit")
        assert resp.status_code == 401

    @pytest.mark.parametrize("fit_path", fit_fixture_params())
    async def test_download_returns_fit_bytes(self, fit_path, client, auth_headers):
        with open(fit_path, "rb") as f:
            upload_resp = await client.post(
                "/api/activities/upload",
                files={"file": ("test.fit", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert upload_resp.status_code == 201
        activity_id = upload_resp.json()["id"]

        resp = await client.get(f"/api/activities/{activity_id}/fit", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert len(resp.content) == fit_path.stat().st_size

    @pytest.mark.parametrize("fit_path", fit_fixture_params())
    async def test_encrypted_download_returns_original_bytes(self, fit_path, client, auth_headers, session):
        """Download endpoint decrypts the file transparently when fit_file_encrypted=True."""
        test_key = Fernet.generate_key().decode()

        with open(fit_path, "rb") as f:
            upload_resp = await client.post(
                "/api/activities/upload",
                files={"file": ("test.fit", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert upload_resp.status_code == 201
        activity_id = upload_resp.json()["id"]

        act_result = await session.execute(select(Activity).where(Activity.id == activity_id))
        activity = act_result.scalar_one()
        src_result = await session.execute(
            select(ActivitySource).where(
                ActivitySource.activity_id == activity_id,
                ActivitySource.provider == "upload",
            )
        )
        upload_src = src_result.scalar_one()
        ath_result = await session.execute(select(Athlete).where(Athlete.id == activity.athlete_id))
        athlete = ath_result.scalar_one()

        original_bytes = fit_path.read_bytes()

        from backend.app.core import config as cfg
        from backend.app.core.file_encryption import encrypt_file

        with patch.object(cfg.settings, "encryption_key", test_key):
            encrypt_file(Path(upload_src.fit_file_path), athlete.global_user_id)
            upload_src.fit_file_encrypted = True
            await session.commit()

            resp = await client.get(f"/api/activities/{activity_id}/fit", headers=auth_headers)

        assert resp.status_code == 200
        assert resp.content == original_bytes


# ── Rename activity ────────────────────────────────────────────────────────────

class TestRenameActivity:
    async def _create(self, client, auth_headers) -> str:
        resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-01-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        return resp.json()["id"]

    async def test_rename_returns_updated_name(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{activity_id}",
            json={"name": "My renamed ride"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "My renamed ride"

    async def test_rename_persists_on_get(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        await client.patch(
            f"/api/activities/{activity_id}",
            json={"name": "Persisted name"},
            headers=auth_headers,
        )
        resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.json()["name"] == "Persisted name"

    async def test_empty_name_returns_422(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{activity_id}",
            json={"name": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.patch("/api/activities/some-id", json={"name": "x"})
        assert resp.status_code == 401


# ── Activity labels and notes ─────────────────────────────────────────────────

class TestActivityLabelsAndNotes:
    async def _create(self, client, auth_headers) -> str:
        resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-02-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        return resp.json()["id"]

    async def test_default_labels_empty(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.json()["labels"] == []

    async def test_set_valid_label(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{activity_id}",
            json={"labels": ["race"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["labels"] == ["race"]

    async def test_set_multiple_labels(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{activity_id}",
            json={"labels": ["race", "commute"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert sorted(resp.json()["labels"]) == ["commute", "race"]

    async def test_clear_labels(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        await client.patch(
            f"/api/activities/{activity_id}",
            json={"labels": ["race"]},
            headers=auth_headers,
        )
        resp = await client.patch(
            f"/api/activities/{activity_id}",
            json={"labels": []},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["labels"] == []

    async def test_unknown_label_returns_422(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{activity_id}",
            json={"labels": ["unknown_label"]},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_labels_persist_on_get(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        await client.patch(
            f"/api/activities/{activity_id}",
            json={"labels": ["commute"]},
            headers=auth_headers,
        )
        resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.json()["labels"] == ["commute"]

    async def test_default_notes_null(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.json()["notes"] is None

    async def test_set_notes(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{activity_id}",
            json={"notes": "Finished 3rd overall."},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["notes"] == "Finished 3rd overall."

    async def test_clear_notes(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        await client.patch(
            f"/api/activities/{activity_id}",
            json={"notes": "Some notes"},
            headers=auth_headers,
        )
        resp = await client.patch(
            f"/api/activities/{activity_id}",
            json={"notes": None},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["notes"] is None

    async def test_notes_persist_on_get(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        await client.patch(
            f"/api/activities/{activity_id}",
            json={"notes": "Great race, new PB!"},
            headers=auth_headers,
        )
        resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.json()["notes"] == "Great race, new PB!"


# ── Activity RPE (perceived effort) ────────────────────────────────────────────

class TestActivityRpe:
    async def _create(self, client, auth_headers) -> str:
        resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-03-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        return resp.json()["id"]

    async def test_default_rpe_null(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.json()["rpe"] is None

    async def test_set_rpe(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{activity_id}",
            json={"rpe": 8},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["rpe"] == 8

    async def test_rpe_persists_on_get(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        await client.patch(
            f"/api/activities/{activity_id}",
            json={"rpe": 6},
            headers=auth_headers,
        )
        resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.json()["rpe"] == 6

    async def test_clear_rpe(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        await client.patch(
            f"/api/activities/{activity_id}",
            json={"rpe": 5},
            headers=auth_headers,
        )
        resp = await client.patch(
            f"/api/activities/{activity_id}",
            json={"rpe": None},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["rpe"] is None

    async def test_rpe_out_of_range_returns_422(self, client, auth_headers):
        activity_id = await self._create(client, auth_headers)
        for bad in (0, 11, -3):
            resp = await client.patch(
                f"/api/activities/{activity_id}",
                json={"rpe": bad},
                headers=auth_headers,
            )
            assert resp.status_code == 422

    async def test_manual_entry_rpe_persisted(self, client, auth_headers):
        # RPE supplied on manual entry is persisted, not just used to derive load.
        resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "duration_s": 3600, "rpe": 7},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["rpe"] == 7


# ── RPE prompt queue ───────────────────────────────────────────────────────────

class TestRpeQueue:
    async def _create(self, client, auth_headers, sport_type="Ride", rpe=None) -> dict:
        body = {"sport_type": sport_type, "duration_s": 3600}
        if rpe is not None:
            body["rpe"] = rpe
        resp = await client.post("/api/activities", json=body, headers=auth_headers)
        assert resp.status_code == 201
        return resp.json()

    async def test_first_call_pins_head_and_returns_empty(self, client, auth_headers):
        # Pre-existing rides should NOT be surfaced on the first ever queue load.
        await self._create(client, auth_headers)
        await self._create(client, auth_headers)
        resp = await client.get("/api/activities/rpe-queue", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["rpe_head"] is not None

    async def test_new_cycling_ride_after_head_is_queued(self, client, auth_headers):
        # First call pins the cursor…
        await client.get("/api/activities/rpe-queue", headers=auth_headers)
        # …then a fresh ride lands and should appear in the queue.
        created = await self._create(client, auth_headers)
        resp = await client.get("/api/activities/rpe-queue", headers=auth_headers)
        data = resp.json()
        ids = [item["id"] for item in data["items"]]
        assert created["id"] in ids

    async def test_non_cycling_ride_not_queued(self, client, auth_headers):
        await client.get("/api/activities/rpe-queue", headers=auth_headers)
        run = await self._create(client, auth_headers, sport_type="Run")
        resp = await client.get("/api/activities/rpe-queue", headers=auth_headers)
        ids = [item["id"] for item in resp.json()["items"]]
        assert run["id"] not in ids

    async def test_rated_ride_not_queued(self, client, auth_headers):
        await client.get("/api/activities/rpe-queue", headers=auth_headers)
        rated = await self._create(client, auth_headers, rpe=7)
        resp = await client.get("/api/activities/rpe-queue", headers=auth_headers)
        ids = [item["id"] for item in resp.json()["items"]]
        assert rated["id"] not in ids

    async def test_commute_labelled_ride_not_queued(self, client, auth_headers):
        await client.get("/api/activities/rpe-queue", headers=auth_headers)
        ride = await self._create(client, auth_headers)
        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"labels": ["commute"]},
            headers=auth_headers,
        )
        resp = await client.get("/api/activities/rpe-queue", headers=auth_headers)
        ids = [item["id"] for item in resp.json()["items"]]
        assert ride["id"] not in ids

    async def test_advancing_head_removes_ride_from_queue(self, client, auth_headers):
        await client.get("/api/activities/rpe-queue", headers=auth_headers)
        ride = await self._create(client, auth_headers)
        # Advance the cursor past this ride via PATCH /api/athlete, mirroring
        # what the dashboard prompt does after Rate/Skip.
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"rpe_head": ride["created_at"]}},
            headers=auth_headers,
        )
        resp = await client.get("/api/activities/rpe-queue", headers=auth_headers)
        ids = [item["id"] for item in resp.json()["items"]]
        assert ride["id"] not in ids

    async def test_queue_requires_auth(self, client):
        resp = await client.get("/api/activities/rpe-queue")
        assert resp.status_code == 401


# ── Activity raw streams ───────────────────────────────────────────────────────

class TestGetActivityStreams:
    async def test_returns_streams_key(self, client, auth_headers):
        create_resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-01-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        activity_id = create_resp.json()["id"]
        resp = await client.get(f"/api/activities/{activity_id}/streams", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "streams" in data
        # manual activity has no streams
        assert isinstance(data["streams"], dict)

    async def test_nonexistent_activity_returns_404(self, client, auth_headers):
        resp = await client.get("/api/activities/nonexistent-id/streams", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/activities/some-id/streams")
        assert resp.status_code == 401

    async def test_returns_empty_streams_for_manual_activity(self, client, auth_headers):
        create_resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-02-01T10:00:00Z", "duration_s": 1800},
            headers=auth_headers,
        )
        activity_id = create_resp.json()["id"]
        resp = await client.get(f"/api/activities/{activity_id}/streams", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["streams"] == {}


# ── Reprocess intervals ────────────────────────────────────────────────────────

class TestReprocess:
    async def _create_processed(self, client, auth_headers) -> str:
        resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": "2025-01-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "processed"
        return resp.json()["id"]

    async def test_reprocess_returns_200_with_intervals(self, client, auth_headers):
        activity_id = await self._create_processed(client, auth_headers)
        resp = await client.post(
            f"/api/activities/{activity_id}/reprocess",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == activity_id
        assert isinstance(data["intervals"], list)
        assert len(data["intervals"]) > 0

    async def test_reprocess_updates_tss_and_fitness(self, client, auth_headers, session):
        from backend.app.models.user_orm import Activity, ActivityStream, DailyMetric
        from sqlalchemy import select as sa_select

        await client.patch("/api/athlete", json={"ftp": 250}, headers=auth_headers)
        activity_id = await self._create_processed(client, auth_headers)

        # Add power + cadence streams and clear Load so we can verify reprocess
        # sets it — and derives a torque stream from power + cadence.
        act_result = await session.execute(sa_select(Activity).where(Activity.id == activity_id))
        activity = act_result.scalar_one()
        activity.avg_power = 200.0
        activity.load = None
        session.add(ActivityStream(activity_id=activity_id, stream_type="power", data=[200] * 20))
        session.add(ActivityStream(activity_id=activity_id, stream_type="cadence", data=[90] * 20))
        await session.commit()

        resp = await client.post(f"/api/activities/{activity_id}/reprocess", headers=auth_headers)
        assert resp.status_code == 200

        # Reprocess derives and returns torque from the stored power + cadence.
        body = resp.json()
        assert "torque" in body["streams"]
        assert len(body["streams"]["torque"]) == 20

        await session.refresh(activity)
        assert activity.load is not None

        # Fitness should have been updated too
        resp2 = await client.get("/api/metrics/fitness/current", headers=auth_headers)
        assert resp2.status_code == 200

    async def test_reprocess_nonexistent_returns_404(self, client, auth_headers):
        resp = await client.post(
            "/api/activities/nonexistent-id/reprocess",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_reprocess_unprocessed_returns_400(self, client, auth_headers, session):
        from backend.app.models.user_orm import Activity
        from sqlalchemy import select as sa_select

        activity_id = await self._create_processed(client, auth_headers)
        result = await session.execute(sa_select(Activity).where(Activity.id == activity_id))
        activity = result.scalar_one()
        activity.status = "pending"
        await session.commit()

        resp = await client.post(
            f"/api/activities/{activity_id}/reprocess",
            headers=auth_headers,
        )
        assert resp.status_code == 400

    async def test_reprocess_unauthenticated_returns_401(self, client):
        resp = await client.post("/api/activities/some-id/reprocess")
        assert resp.status_code == 401


# ── Upload edge cases ──────────────────────────────────────────────────────────

class TestFitUploadEdgeCases:
    async def test_non_fit_file_returns_400(self, client, auth_headers):
        resp = await client.post(
            "/api/activities/upload",
            files={"file": ("test.fit", b"this is not a fit file at all!", "application/octet-stream")},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    async def test_too_short_file_returns_400(self, client, auth_headers):
        resp = await client.post(
            "/api/activities/upload",
            files={"file": ("test.fit", b"\x00\x01\x02", "application/octet-stream")},
            headers=auth_headers,
        )
        assert resp.status_code == 400


# ── Search & filter ────────────────────────────────────────────────────────────

class TestActivitySearchAndFilter:
    async def _create(self, client, auth_headers, **kwargs):
        defaults = {"sport_type": "Ride", "start_time": "2025-06-01T10:00:00Z", "duration_s": 3600}
        defaults.update(kwargs)
        resp = await client.post("/api/activities", json=defaults, headers=auth_headers)
        assert resp.status_code == 201
        return resp.json()

    async def test_search_by_name_matches(self, client, auth_headers):
        await self._create(client, auth_headers, name="Morning Ride", start_time="2025-06-01T08:00:00Z")
        await self._create(client, auth_headers, name="Evening Run", sport_type="Run", start_time="2025-06-02T18:00:00Z")
        resp = await client.get("/api/activities?q=morning", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["name"] == "Morning Ride"

    async def test_search_case_insensitive(self, client, auth_headers):
        await self._create(client, auth_headers, name="Zwift Race", start_time="2025-06-01T10:00:00Z")
        resp = await client.get("/api/activities?q=ZWIFT", headers=auth_headers)
        assert resp.json()["total"] == 1

    async def test_search_no_match_returns_empty(self, client, auth_headers):
        await self._create(client, auth_headers, name="Morning Ride", start_time="2025-06-01T08:00:00Z")
        resp = await client.get("/api/activities?q=nonexistent", headers=auth_headers)
        assert resp.json()["total"] == 0

    async def test_min_duration_filter(self, client, auth_headers):
        await self._create(client, auth_headers, duration_s=1800, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, duration_s=7200, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities?min_duration=3600", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["duration_s"] == 7200

    async def test_max_duration_filter(self, client, auth_headers):
        await self._create(client, auth_headers, duration_s=1800, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, duration_s=7200, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities?max_duration=3600", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["duration_s"] == 1800

    async def test_min_distance_filter(self, client, auth_headers):
        await self._create(client, auth_headers, distance_m=5000, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, distance_m=50000, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities?min_distance=10000", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["distance_m"] == 50000

    async def test_max_distance_filter(self, client, auth_headers):
        await self._create(client, auth_headers, distance_m=5000, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, distance_m=50000, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities?max_distance=10000", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["distance_m"] == 5000

    async def test_min_tss_filter(self, client, auth_headers):
        await self._create(client, auth_headers, load=40.0, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, load=120.0, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities?min_tss=100", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["load"] == 120.0

    async def test_max_tss_filter(self, client, auth_headers):
        await self._create(client, auth_headers, load=40.0, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, load=120.0, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities?max_tss=100", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["load"] == 40.0

    async def test_workout_category_filter(self, client, auth_headers):
        r1 = await self._create(client, auth_headers, start_time="2025-06-01T10:00:00Z")
        r2 = await self._create(client, auth_headers, start_time="2025-06-02T10:00:00Z")
        # Set categories directly via PATCH
        await client.patch(
            f"/api/activities/{r1['id']}",
            json={"workout_category": "endurance"},
            headers=auth_headers,
        )
        await client.patch(
            f"/api/activities/{r2['id']}",
            json={"workout_category": "recovery"},
            headers=auth_headers,
        )
        resp = await client.get("/api/activities?workout_category=endurance", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["workout_category"] == "endurance"

    async def test_has_power_true_filter(self, client, auth_headers):
        # Manual activities don't get avg_power set by default
        await self._create(client, auth_headers, start_time="2025-06-01T10:00:00Z")
        resp = await client.get("/api/activities?has_power=true", headers=auth_headers)
        assert resp.json()["total"] == 0

    async def test_has_power_false_filter(self, client, auth_headers):
        await self._create(client, auth_headers, start_time="2025-06-01T10:00:00Z")
        resp = await client.get("/api/activities?has_power=false", headers=auth_headers)
        assert resp.json()["total"] == 1

    async def test_combined_filters(self, client, auth_headers):
        await self._create(client, auth_headers, sport_type="Run", duration_s=1800, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, sport_type="Ride", duration_s=7200, start_time="2025-06-02T10:00:00Z")
        await self._create(client, auth_headers, sport_type="Ride", duration_s=1800, start_time="2025-06-03T10:00:00Z")
        resp = await client.get(
            "/api/activities?sport_type=Ride&min_duration=3600",
            headers=auth_headers,
        )
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["duration_s"] == 7200
