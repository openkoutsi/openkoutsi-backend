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
from backend.app.models.team_orm import Activity, ActivitySource, Athlete

TESTDATA = Path(__file__).parent.parent.parent / "testdata"
SAMPLE_FIT = TESTDATA / "Zwift_Aerobic_Foundation_Forge.fit"


# ── Manual activity creation ───────────────────────────────────────────────────

class TestCreateManualActivity:
    async def test_explicit_tss_stored_as_is(self, client, auth_headers):
        resp = await client.post(
            "/api/activities/",
            json={
                "sport_type": "Ride",
                "start_time": "2025-06-01T10:00:00Z",
                "duration_s": 3600,
                "tss": 100.0,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["tss"] == 100.0
        assert data["status"] == "processed"

    async def test_no_tss_inputs_gives_null_tss(self, client, auth_headers):
        resp = await client.post(
            "/api/activities/",
            json={
                "sport_type": "Ride",
                "start_time": "2025-06-02T10:00:00Z",
                "duration_s": 3600,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["tss"] is None

    async def test_rpe_based_tss(self, client, auth_headers):
        # tss = (duration_s / 3600) * rpe^2 * 10
        resp = await client.post(
            "/api/activities/",
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
        assert resp.json()["tss"] == pytest.approx(expected_tss, rel=1e-6)

    async def test_hr_based_tss_requires_athlete_max_hr(self, client, auth_headers):
        # Set max_hr on the athlete first
        await client.put("/api/athlete/", json={"max_hr": 185}, headers=auth_headers)
        resp = await client.post(
            "/api/activities/",
            json={
                "sport_type": "Ride",
                "start_time": "2025-06-04T10:00:00Z",
                "duration_s": 3600,
                "avg_hr": 150.0,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["tss"] is not None
        assert resp.json()["tss"] > 0

    async def test_missing_required_field_returns_422(self, client, auth_headers):
        resp = await client.post(
            "/api/activities/",
            json={"sport_type": "Ride"},  # missing start_time and duration_s
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post(
            "/api/activities/",
            json={"sport_type": "Ride", "start_time": "2025-06-01T10:00:00Z", "duration_s": 3600},
        )
        assert resp.status_code == 401


# ── Activity listing ───────────────────────────────────────────────────────────

class TestListActivities:
    async def test_empty_list_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/activities/", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    async def test_returns_created_activities(self, client, auth_headers):
        for i in range(3):
            await client.post(
                "/api/activities/",
                json={
                    "sport_type": "Ride",
                    "start_time": f"2025-06-0{i+1}T10:00:00Z",
                    "duration_s": 3600,
                },
                headers=auth_headers,
            )
        resp = await client.get("/api/activities/", headers=auth_headers)
        assert resp.json()["total"] == 3

    async def test_sport_type_filter(self, client, auth_headers):
        await client.post(
            "/api/activities/",
            json={"sport_type": "Ride", "start_time": "2025-01-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        await client.post(
            "/api/activities/",
            json={"sport_type": "Run", "start_time": "2025-01-02T10:00:00Z", "duration_s": 1800},
            headers=auth_headers,
        )
        resp = await client.get("/api/activities/?sport_type=Ride", headers=auth_headers)
        assert resp.json()["total"] == 1
        assert resp.json()["items"][0]["sport_type"] == "Ride"

    async def test_date_range_filter(self, client, auth_headers):
        for month in [1, 3, 6]:
            await client.post(
                "/api/activities/",
                json={
                    "sport_type": "Ride",
                    "start_time": f"2025-{month:02d}-15T10:00:00Z",
                    "duration_s": 3600,
                },
                headers=auth_headers,
            )
        resp = await client.get(
            "/api/activities/?start=2025-02-01&end=2025-05-01",
            headers=auth_headers,
        )
        assert resp.json()["total"] == 1

    async def test_pagination(self, client, auth_headers):
        for i in range(5):
            await client.post(
                "/api/activities/",
                json={
                    "sport_type": "Ride",
                    "start_time": f"2025-06-0{i+1}T10:00:00Z",
                    "duration_s": 3600,
                },
                headers=auth_headers,
            )
        resp = await client.get("/api/activities/?page=2&page_size=2", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["page"] == 2

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/activities/")
        assert resp.status_code == 401


# ── Activity detail ────────────────────────────────────────────────────────────

class TestGetActivity:
    async def test_returns_activity_detail(self, client, auth_headers):
        create_resp = await client.post(
            "/api/activities/",
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
            "/api/activities/",
            json={"sport_type": "Ride", "start_time": "2025-01-01T10:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        activity_id = create_resp.json()["id"]
        resp = await client.delete(f"/api/activities/{activity_id}", headers=auth_headers)
        assert resp.status_code == 204

    async def test_deleted_activity_returns_404(self, client, auth_headers):
        create_resp = await client.post(
            "/api/activities/",
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
    @pytest.mark.skipif(not SAMPLE_FIT.exists(), reason="FIT fixture not found")
    async def test_upload_fit_file_processes_correctly(self, client, auth_headers, session):
        """Upload a real FIT file and verify it's processed into a complete activity."""
        # Set FTP so power-based TSS can be calculated
        await client.put("/api/athlete/", json={"ftp": 280}, headers=auth_headers)

        with open(SAMPLE_FIT, "rb") as f:
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
        assert activity.normalized_power is not None
        assert activity.tss is not None

    async def test_has_fit_file_false_for_manual_activity(self, client, auth_headers):
        resp = await client.post(
            "/api/activities/",
            json={
                "sport_type": "Ride",
                "start_time": "2025-01-01T10:00:00Z",
                "duration_s": 3600,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["has_fit_file"] is False

    @pytest.mark.skipif(not SAMPLE_FIT.exists(), reason="FIT fixture not found")
    async def test_has_fit_file_true_for_uploaded_activity(self, client, auth_headers):
        with open(SAMPLE_FIT, "rb") as f:
            resp = await client.post(
                "/api/activities/upload",
                files={"file": ("test.fit", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert resp.status_code == 201
        assert resp.json()["has_fit_file"] is True

    @pytest.mark.skipif(not SAMPLE_FIT.exists(), reason="FIT fixture not found")
    async def test_duplicate_fit_upload_returns_409(self, client, auth_headers, session):
        """Uploading a file whose start time matches an existing activity returns 409."""
        # First upload
        with open(SAMPLE_FIT, "rb") as f:
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
        with open(SAMPLE_FIT, "rb") as f:
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
            "/api/activities/",
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

    @pytest.mark.skipif(not SAMPLE_FIT.exists(), reason="FIT fixture not found")
    async def test_download_returns_fit_bytes(self, client, auth_headers):
        with open(SAMPLE_FIT, "rb") as f:
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
        assert len(resp.content) == SAMPLE_FIT.stat().st_size

    @pytest.mark.skipif(not SAMPLE_FIT.exists(), reason="FIT fixture not found")
    async def test_encrypted_download_returns_original_bytes(self, client, auth_headers, session):
        """Download endpoint decrypts the file transparently when fit_file_encrypted=True."""
        test_key = Fernet.generate_key().decode()

        with open(SAMPLE_FIT, "rb") as f:
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

        original_bytes = SAMPLE_FIT.read_bytes()

        from backend.app.core import config as cfg
        from backend.app.core.file_encryption import encrypt_file
        from tests.conftest import _TEST_TEAM_ID

        with patch.object(cfg.settings, "encryption_key", test_key):
            encrypt_file(Path(upload_src.fit_file_path), _TEST_TEAM_ID, athlete.global_user_id)
            upload_src.fit_file_encrypted = True
            await session.commit()

            resp = await client.get(f"/api/activities/{activity_id}/fit", headers=auth_headers)

        assert resp.status_code == 200
        assert resp.content == original_bytes


# ── Rename activity ────────────────────────────────────────────────────────────

class TestRenameActivity:
    async def _create(self, client, auth_headers) -> str:
        resp = await client.post(
            "/api/activities/",
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
            "/api/activities/",
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


# ── Activity raw streams ───────────────────────────────────────────────────────

class TestGetActivityStreams:
    async def test_returns_streams_key(self, client, auth_headers):
        create_resp = await client.post(
            "/api/activities/",
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
            "/api/activities/",
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
            "/api/activities/",
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
        from backend.app.models.team_orm import Activity, ActivityStream, DailyMetric
        from sqlalchemy import select as sa_select

        await client.put("/api/athlete/", json={"ftp": 250}, headers=auth_headers)
        activity_id = await self._create_processed(client, auth_headers)

        # Add a power stream and clear TSS so we can verify reprocess sets it
        act_result = await session.execute(sa_select(Activity).where(Activity.id == activity_id))
        activity = act_result.scalar_one()
        activity.avg_power = 200.0
        activity.tss = None
        session.add(ActivityStream(activity_id=activity_id, stream_type="power", data=[200] * 20))
        await session.commit()

        resp = await client.post(f"/api/activities/{activity_id}/reprocess", headers=auth_headers)
        assert resp.status_code == 200

        await session.refresh(activity)
        assert activity.tss is not None

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
        from backend.app.models.team_orm import Activity
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
        resp = await client.post("/api/activities/", json=defaults, headers=auth_headers)
        assert resp.status_code == 201
        return resp.json()

    async def test_search_by_name_matches(self, client, auth_headers):
        await self._create(client, auth_headers, name="Morning Ride", start_time="2025-06-01T08:00:00Z")
        await self._create(client, auth_headers, name="Evening Run", sport_type="Run", start_time="2025-06-02T18:00:00Z")
        resp = await client.get("/api/activities/?q=morning", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["name"] == "Morning Ride"

    async def test_search_case_insensitive(self, client, auth_headers):
        await self._create(client, auth_headers, name="Zwift Race", start_time="2025-06-01T10:00:00Z")
        resp = await client.get("/api/activities/?q=ZWIFT", headers=auth_headers)
        assert resp.json()["total"] == 1

    async def test_search_no_match_returns_empty(self, client, auth_headers):
        await self._create(client, auth_headers, name="Morning Ride", start_time="2025-06-01T08:00:00Z")
        resp = await client.get("/api/activities/?q=nonexistent", headers=auth_headers)
        assert resp.json()["total"] == 0

    async def test_min_duration_filter(self, client, auth_headers):
        await self._create(client, auth_headers, duration_s=1800, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, duration_s=7200, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities/?min_duration=3600", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["duration_s"] == 7200

    async def test_max_duration_filter(self, client, auth_headers):
        await self._create(client, auth_headers, duration_s=1800, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, duration_s=7200, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities/?max_duration=3600", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["duration_s"] == 1800

    async def test_min_distance_filter(self, client, auth_headers):
        await self._create(client, auth_headers, distance_m=5000, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, distance_m=50000, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities/?min_distance=10000", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["distance_m"] == 50000

    async def test_max_distance_filter(self, client, auth_headers):
        await self._create(client, auth_headers, distance_m=5000, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, distance_m=50000, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities/?max_distance=10000", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["distance_m"] == 5000

    async def test_min_tss_filter(self, client, auth_headers):
        await self._create(client, auth_headers, tss=40.0, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, tss=120.0, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities/?min_tss=100", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["tss"] == 120.0

    async def test_max_tss_filter(self, client, auth_headers):
        await self._create(client, auth_headers, tss=40.0, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, tss=120.0, start_time="2025-06-02T10:00:00Z")
        resp = await client.get("/api/activities/?max_tss=100", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["tss"] == 40.0

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
        resp = await client.get("/api/activities/?workout_category=endurance", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["workout_category"] == "endurance"

    async def test_has_power_true_filter(self, client, auth_headers):
        # Manual activities don't get avg_power set by default
        await self._create(client, auth_headers, start_time="2025-06-01T10:00:00Z")
        resp = await client.get("/api/activities/?has_power=true", headers=auth_headers)
        assert resp.json()["total"] == 0

    async def test_has_power_false_filter(self, client, auth_headers):
        await self._create(client, auth_headers, start_time="2025-06-01T10:00:00Z")
        resp = await client.get("/api/activities/?has_power=false", headers=auth_headers)
        assert resp.json()["total"] == 1

    async def test_combined_filters(self, client, auth_headers):
        await self._create(client, auth_headers, sport_type="Run", duration_s=1800, start_time="2025-06-01T10:00:00Z")
        await self._create(client, auth_headers, sport_type="Ride", duration_s=7200, start_time="2025-06-02T10:00:00Z")
        await self._create(client, auth_headers, sport_type="Ride", duration_s=1800, start_time="2025-06-03T10:00:00Z")
        resp = await client.get(
            "/api/activities/?sport_type=Ride&min_duration=3600",
            headers=auth_headers,
        )
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["duration_s"] == 7200
