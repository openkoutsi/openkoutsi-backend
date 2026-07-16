"""
Integration tests for /api/athlete endpoints.
"""
import io
import json
from datetime import datetime
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from ._fit_fixtures import fit_fixture_params

# Minimal valid image headers for magic-byte detection tests
_JPEG = b"\xff\xd8\xff\xe0"
_PNG  = b"\x89PNG\r\n\x1a\n"
_WEBP = b"RIFF\x00\x00\x00\x00WEBP"
_GIF  = b"GIF89a"


class TestGetAthlete:
    async def test_returns_profile_after_registration(self, client, auth_headers):
        resp = await client.get("/api/athlete", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["hr_zones"] == []
        assert data["power_zones"] == []
        assert data["ftp_tests"] == []
        assert data["connected_providers"] == []

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/athlete")
        assert resp.status_code == 401


class TestUpdateAthlete:
    async def test_set_ftp_records_test_history(self, client, auth_headers):
        resp = await client.patch("/api/athlete", json={"ftp": 280}, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ftp"] == 280
        assert len(data["ftp_tests"]) == 1
        assert data["ftp_tests"][0]["ftp"] == 280
        assert data["ftp_tests"][0]["method"] == "manual"

    async def test_ftp_test_method_recorded_when_provided(self, client, auth_headers):
        resp = await client.patch(
            "/api/athlete",
            json={"ftp": 265, "ftp_test_method": "20min"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ftp_tests"][-1]["method"] == "20min"

    async def test_invalid_ftp_test_method_rejected(self, client, auth_headers):
        resp = await client.patch(
            "/api/athlete",
            json={"ftp": 265, "ftp_test_method": "xyz"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_ftp_test_method_ignored_without_ftp(self, client, auth_headers):
        resp = await client.patch(
            "/api/athlete",
            json={"max_hr": 190, "ftp_test_method": "cp"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["ftp_tests"] == []

    async def test_updating_ftp_twice_preserves_history(self, client, auth_headers):
        await client.patch("/api/athlete", json={"ftp": 250}, headers=auth_headers)
        resp = await client.patch("/api/athlete", json={"ftp": 280}, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ftp"] == 280
        assert len(data["ftp_tests"]) == 2

    async def test_partial_update_leaves_other_fields_unchanged(self, client, auth_headers):
        await client.patch("/api/athlete", json={"ftp": 300, "max_hr": 185}, headers=auth_headers)
        resp = await client.patch("/api/athlete", json={"name": "Test Rider"}, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Test Rider"
        assert data["ftp"] == 300
        assert data["max_hr"] == 185

    async def test_update_hr_zones(self, client, auth_headers):
        zones = [
            {"low": 0, "high": 130, "name": "Z1"},
            {"low": 130, "high": 155, "name": "Z2"},
        ]
        resp = await client.patch("/api/athlete", json={"hr_zones": zones}, headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["hr_zones"]) == 2

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.patch("/api/athlete", json={"ftp": 300})
        assert resp.status_code == 401


class TestPatchAthlete:
    async def test_patch_partial_update(self, client, auth_headers):
        await client.patch("/api/athlete", json={"ftp": 300, "max_hr": 185}, headers=auth_headers)
        resp = await client.patch("/api/athlete", json={"name": "Patched Rider"}, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Patched Rider"
        assert data["ftp"] == 300
        assert data["max_hr"] == 185

    async def test_patch_unauthenticated_returns_401(self, client):
        resp = await client.patch("/api/athlete", json={"ftp": 300})
        assert resp.status_code == 401


class TestExperienceLevel:
    """Self-reported experience level, stored in app_settings (see #18)."""

    async def test_valid_level_round_trips(self, client, auth_headers):
        resp = await client.patch(
            "/api/athlete",
            json={"app_settings": {"experience_level": "elite"}},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["app_settings"]["experience_level"] == "elite"

        get_resp = await client.get("/api/athlete", headers=auth_headers)
        assert get_resp.json()["app_settings"]["experience_level"] == "elite"

    async def test_invalid_level_rejected(self, client, auth_headers):
        resp = await client.patch(
            "/api/athlete",
            json={"app_settings": {"experience_level": "pro"}},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    async def test_empty_string_clears_level(self, client, auth_headers):
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"experience_level": "novice"}},
            headers=auth_headers,
        )
        resp = await client.patch(
            "/api/athlete",
            json={"app_settings": {"experience_level": ""}},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert "experience_level" not in resp.json()["app_settings"]


class TestLlmApiKeyHandling:
    """The LLM API key must be encrypted at rest and never returned to the client."""

    def _patch_enc_key(self):
        from cryptography.fernet import Fernet
        from backend.app.core import config
        return patch.object(config.settings, "encryption_key", Fernet.generate_key().decode())

    async def test_saving_key_does_not_return_plaintext(self, client, auth_headers):
        with self._patch_enc_key():
            resp = await client.patch(
                "/api/athlete",
                json={"app_settings": {"llm_base_url": "http://localhost:11434/v1", "llm_api_key": "sk-secret"}},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()
        settings = data["app_settings"]
        assert "llm_api_key" not in settings
        assert "llm_api_key_enc" not in settings

    async def test_key_set_indicator_is_true_after_saving(self, client, auth_headers):
        with self._patch_enc_key():
            resp = await client.patch(
                "/api/athlete",
                json={"app_settings": {"llm_base_url": "http://localhost:11434/v1", "llm_api_key": "sk-secret"}},
                headers=auth_headers,
            )
        assert resp.json()["app_settings"]["llm_api_key_set"] is True

    async def test_get_athlete_never_returns_encrypted_key(self, client, auth_headers):
        with self._patch_enc_key():
            await client.patch(
                "/api/athlete",
                json={"app_settings": {"llm_api_key": "sk-secret"}},
                headers=auth_headers,
            )
            resp = await client.get("/api/athlete", headers=auth_headers)
        settings = resp.json()["app_settings"]
        assert "llm_api_key" not in settings
        assert "llm_api_key_enc" not in settings

    async def test_clearing_key_sets_indicator_to_false(self, client, auth_headers):
        with self._patch_enc_key():
            await client.patch(
                "/api/athlete",
                json={"app_settings": {"llm_api_key": "sk-secret"}},
                headers=auth_headers,
            )
            resp = await client.patch(
                "/api/athlete",
                json={"app_settings": {"llm_api_key": None}},
                headers=auth_headers,
            )
        assert resp.json()["app_settings"]["llm_api_key_set"] is False

    async def test_no_key_indicator_is_false_by_default(self, client, auth_headers):
        resp = await client.get("/api/athlete", headers=auth_headers)
        assert resp.json()["app_settings"].get("llm_api_key_set") is False

    async def test_saving_key_without_encryption_key_returns_503(self, client, auth_headers):
        from backend.app.core import config
        with patch.object(config.settings, "encryption_key", None):
            resp = await client.patch(
                "/api/athlete",
                json={"app_settings": {"llm_api_key": "sk-secret"}},
                headers=auth_headers,
            )
        assert resp.status_code == 503


class TestExportAthlete:
    async def test_export_returns_zip(self, client, auth_headers):
        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200
        assert "application/zip" in resp.headers["content-type"]

    async def test_export_zip_contains_profile_json(self, client, auth_headers):
        await client.patch(
            "/api/athlete",
            json={"ftp": 280, "name": "Test Rider"},
            headers=auth_headers,
        )
        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            assert "profile.json" in zf.namelist()
            profile = json.loads(zf.read("profile.json"))
        assert profile["ftp"] == 280
        assert profile["name"] == "Test Rider"
        assert "username" in profile
        assert "exported_at" in profile

    async def test_export_zip_contains_activities_json(self, client, auth_headers):
        for i in range(2):
            await client.post(
                "/api/activities",
                json={
                    "sport_type": "Ride",
                    "start_time": f"2025-06-0{i + 1}T10:00:00Z",
                    "duration_s": 3600,
                },
                headers=auth_headers,
            )

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            assert "activities.json" in zf.namelist()
            activities = json.loads(zf.read("activities.json"))
        assert len(activities) == 2

    async def test_export_empty_activities_still_valid_zip(self, client, auth_headers):
        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
        assert "profile.json" in names
        assert "activities.json" in names

    async def test_export_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/athlete/export")
        assert resp.status_code == 401

    @pytest.mark.parametrize("fit_path", fit_fixture_params())
    async def test_export_decrypts_encrypted_fit_files(self, fit_path, client, auth_headers, session):
        """Exported zip contains valid (decrypted) FIT bytes even when files are encrypted at rest."""
        from backend.app.core import config as cfg
        from backend.app.core.file_encryption import encrypt_file
        from backend.app.models.user_orm import Activity, Athlete

        test_key = Fernet.generate_key().decode()

        with open(fit_path, "rb") as f:
            upload_resp = await client.post(
                "/api/activities/upload",
                files={"file": ("test.fit", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert upload_resp.status_code == 201
        activity_id = upload_resp.json()["id"]

        from backend.app.models.user_orm import ActivitySource
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

        with patch.object(cfg.settings, "encryption_key", test_key):
            encrypt_file(Path(upload_src.fit_file_path), athlete.global_user_id)
            upload_src.fit_file_encrypted = True
            await session.commit()

            resp = await client.get("/api/athlete/export", headers=auth_headers)

        assert resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            fit_names = [n for n in zf.namelist() if n.startswith("fit_files/")]
            assert len(fit_names) == 1
            exported_bytes = zf.read(fit_names[0])

        assert exported_bytes == original_bytes

    async def test_export_contains_all_sections(self, client, auth_headers):
        """Every data category has a JSON file, even for an otherwise-empty account."""
        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = set(zf.namelist())

        assert {
            "profile.json",
            "activities.json",
            "goals.json",
            "plans.json",
            "workout_definitions.json",
            "daily_metrics.json",
            "personal_records.json",
            "inbox.json",
            "weight_log.json",
        } <= names

    async def test_export_profile_redacts_llm_key_and_includes_settings(
        self, client, auth_headers, session, seeded_athlete
    ):
        """LLM analysis settings are exported, but the encrypted BYOK key never leaks."""
        seeded_athlete.app_settings = {
            "llm_api_key_enc": "super-secret-encrypted-token",
            "llm_model": "my-model",
            "experience_level": "intermediate",
        }
        seeded_athlete.availability = {"monday": True}
        seeded_athlete.training_status = "Building nicely."
        await session.commit()

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            profile = json.loads(zf.read("profile.json"))

        settings = profile["app_settings"]
        assert settings["llm_api_key_set"] is True
        assert "llm_api_key_enc" not in settings
        assert "super-secret-encrypted-token" not in json.dumps(profile)
        assert settings["llm_model"] == "my-model"
        assert settings["experience_level"] == "intermediate"
        assert profile["availability"] == {"monday": True}
        assert profile["training_status"] == "Building nicely."

    async def test_export_activity_includes_notes_labels_analysis(
        self, client, auth_headers, session, seeded_athlete
    ):
        from backend.app.models.user_orm import Activity

        session.add(
            Activity(
                id="act-notes-1",
                athlete_id=seeded_athlete.id,
                name="Tempo ride",
                sport_type="Ride",
                start_time=datetime(2025, 6, 1, 10, 0, 0),
                labels=["tempo", "endurance"],
                notes="Felt strong.",
                workout_category="tempo",
                analysis_status="done",
                analysis="Great aerobic session.",
            )
        )
        await session.commit()

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            activities = json.loads(zf.read("activities.json"))

        entry = next(a for a in activities if a["id"] == "act-notes-1")
        assert entry["labels"] == ["tempo", "endurance"]
        assert entry["notes"] == "Felt strong."
        assert entry["workout_category"] == "tempo"
        assert entry["analysis_status"] == "done"
        assert entry["analysis"] == "Great aerobic session."

    async def test_export_goals(self, client, auth_headers, session, seeded_athlete):
        from backend.app.models.user_orm import Goal

        session.add(
            Goal(
                id="goal-1",
                athlete_id=seeded_athlete.id,
                title="Sub-3 marathon",
                metric="time",
                target_value=10800.0,
                status="active",
            )
        )
        await session.commit()

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            goals = json.loads(zf.read("goals.json"))

        assert len(goals) == 1
        assert goals[0]["title"] == "Sub-3 marathon"
        assert goals[0]["target_value"] == 10800.0

    async def test_export_plans_with_nested_workouts(
        self, client, auth_headers, session, seeded_athlete
    ):
        from backend.app.models.user_orm import PlannedWorkout, TrainingPlan

        session.add(
            TrainingPlan(id="plan-1", athlete_id=seeded_athlete.id, name="Base build", weeks=4)
        )
        session.add(
            PlannedWorkout(
                id="pw-1",
                plan_id="plan-1",
                week_number=1,
                day_of_week=3,
                workout_type="endurance",
                description="90 min Z2",
            )
        )
        await session.commit()

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            plans = json.loads(zf.read("plans.json"))

        assert len(plans) == 1
        assert plans[0]["name"] == "Base build"
        assert len(plans[0]["planned_workouts"]) == 1
        assert plans[0]["planned_workouts"][0]["description"] == "90 min Z2"

    async def test_export_workout_definitions(
        self, client, auth_headers, session, seeded_athlete
    ):
        from backend.app.models.user_orm import WorkoutDefinition

        session.add(
            WorkoutDefinition(
                id="wd-1",
                athlete_id=seeded_athlete.id,
                name="4x4 VO2max",
                sport_type="Ride",
                steps=[{"type": "interval", "duration_s": 240, "target": 300}],
            )
        )
        await session.commit()

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            defs = json.loads(zf.read("workout_definitions.json"))

        assert len(defs) == 1
        assert defs[0]["name"] == "4x4 VO2max"
        assert defs[0]["steps"][0]["duration_s"] == 240

    async def test_export_daily_metrics(
        self, client, auth_headers, session, seeded_athlete
    ):
        from datetime import date

        from backend.app.models.user_orm import DailyMetric

        session.add(
            DailyMetric(
                athlete_id=seeded_athlete.id,
                date=date(2025, 6, 1),
                fitness=55.0,
                fatigue=40.0,
                form=15.0,
                load_day=80.0,
            )
        )
        await session.commit()

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            metrics = json.loads(zf.read("daily_metrics.json"))

        assert len(metrics) == 1
        assert metrics[0]["fitness"] == 55.0
        assert metrics[0]["load_day"] == 80.0

    async def test_export_personal_records(
        self, client, auth_headers, session, seeded_athlete
    ):
        from backend.app.models.user_orm import (
            Activity,
            ActivityDistanceBest,
            ActivityPowerBest,
        )

        start = datetime(2025, 6, 1, 10, 0, 0)
        session.add(
            Activity(
                id="act-pr-1",
                athlete_id=seeded_athlete.id,
                name="PR ride",
                sport_type="Ride",
                start_time=start,
            )
        )
        session.add(
            ActivityPowerBest(
                activity_id="act-pr-1",
                athlete_id=seeded_athlete.id,
                duration_s=300,
                power_w=280.0,
                activity_start_time=start,
            )
        )
        session.add(
            ActivityDistanceBest(
                activity_id="act-pr-1",
                athlete_id=seeded_athlete.id,
                distance_m=5000,
                time_s=600,
                activity_start_time=start,
            )
        )
        await session.commit()

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            records = json.loads(zf.read("personal_records.json"))

        assert records["power_bests"][0]["power_w"] == 280.0
        assert records["distance_bests"][0]["distance_m"] == 5000

    async def test_export_inbox(self, client, auth_headers, session):
        from backend.app.models.message_orm import Message

        session.add(Message(id="msg-1", type="welcome", data={"foo": "bar"}))
        await session.commit()

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            inbox = json.loads(zf.read("inbox.json"))

        assert len(inbox) == 1
        assert inbox[0]["type"] == "welcome"
        assert inbox[0]["data"] == {"foo": "bar"}

    async def test_export_weight_log(
        self, client, auth_headers, session, seeded_athlete
    ):
        from datetime import date

        from backend.app.models.user_orm import WeightLog

        session.add(
            WeightLog(
                id="wl-1",
                athlete_id=seeded_athlete.id,
                effective_date=date(2025, 6, 1),
                weight_kg=72.5,
            )
        )
        await session.commit()

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            weights = json.loads(zf.read("weight_log.json"))

        assert len(weights) == 1
        assert weights[0]["weight_kg"] == 72.5
        assert weights[0]["effective_date"] == "2025-06-01"


# ── Avatar fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def avatar_dir(tmp_path):
    """Redirect avatar storage to a temp directory for the duration of the test."""
    d = tmp_path / "avatars"
    with patch("backend.app.api.athlete.settings") as mock_settings:
        mock_settings.user_avatar_dir.return_value = d
        mock_settings.frontend_url = ""
        yield d


# ── Avatar tests ───────────────────────────────────────────────────────────────

class TestAvatar:
    async def test_avatar_url_is_null_by_default(self, client, auth_headers):
        resp = await client.get("/api/athlete", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["avatar_url"] is None

    async def test_upload_jpeg_returns_populated_avatar_url(self, client, auth_headers, avatar_dir):
        resp = await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("photo.jpg", _JPEG + b"fake-jpeg", "image/jpeg")},
        )
        assert resp.status_code == 200
        url = resp.json()["avatar_url"]
        assert url is not None
        assert "avatar" in url

    async def test_upload_png_accepted(self, client, auth_headers, avatar_dir):
        resp = await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("photo.png", _PNG + b"fake-png", "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["avatar_url"] is not None

    async def test_upload_webp_accepted(self, client, auth_headers, avatar_dir):
        resp = await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("photo.webp", _WEBP + b"fake-webp", "image/webp")},
        )
        assert resp.status_code == 200

    async def test_upload_rejects_non_image_content(self, client, auth_headers, avatar_dir):
        resp = await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("doc.pdf", b"pdf-bytes", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    async def test_upload_rejects_svg_with_spoofed_content_type(self, client, auth_headers, avatar_dir):
        svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        resp = await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("evil.svg", svg, "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    async def test_upload_too_large_returns_400(self, client, auth_headers, avatar_dir):
        big = b"x" * (5 * 1024 * 1024 + 1)
        resp = await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("big.jpg", big, "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "too large" in resp.json()["detail"].lower()

    async def test_upload_unauthenticated_returns_401(self, client, avatar_dir):
        resp = await client.post(
            "/api/athlete/avatar",
            files={"file": ("photo.jpg", b"bytes", "image/jpeg")},
        )
        assert resp.status_code == 401

    async def test_get_avatar_requires_auth(self, client, auth_headers, avatar_dir):
        await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("photo.jpg", _JPEG + b"image-data", "image/jpeg")},
        )
        athlete_id = (await client.get("/api/athlete", headers=auth_headers)).json()["id"]
        resp = await client.get(f"/api/athlete/{athlete_id}/avatar", headers=auth_headers)
        assert resp.status_code == 200

    async def test_get_avatar_returns_exact_uploaded_bytes(self, client, auth_headers, avatar_dir):
        image_bytes = b"\xff\xd8\xff\xe0fake-jpeg-content"
        await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("photo.jpg", image_bytes, "image/jpeg")},
        )
        athlete_id = (await client.get("/api/athlete", headers=auth_headers)).json()["id"]
        resp = await client.get(f"/api/athlete/{athlete_id}/avatar", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.content == image_bytes

    async def test_get_avatar_unknown_athlete_returns_404(self, client, auth_headers):
        resp = await client.get("/api/athlete/does-not-exist/avatar", headers=auth_headers)
        assert resp.status_code == 404

    async def test_get_avatar_when_none_set_returns_404(self, client, auth_headers):
        athlete_id = (await client.get("/api/athlete", headers=auth_headers)).json()["id"]
        resp = await client.get(f"/api/athlete/{athlete_id}/avatar", headers=auth_headers)
        assert resp.status_code == 404

    async def test_delete_avatar_clears_avatar_url(self, client, auth_headers, avatar_dir):
        await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("photo.jpg", _JPEG + b"bytes", "image/jpeg")},
        )
        resp = await client.delete("/api/athlete/avatar", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["avatar_url"] is None

    async def test_delete_avatar_removes_file_from_disk(self, client, auth_headers, avatar_dir):
        await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("photo.jpg", _JPEG + b"bytes", "image/jpeg")},
        )
        await client.delete("/api/athlete/avatar", headers=auth_headers)
        remaining = list(avatar_dir.glob("*")) if avatar_dir.exists() else []
        assert remaining == []

    async def test_delete_with_no_avatar_is_idempotent(self, client, auth_headers):
        resp = await client.delete("/api/athlete/avatar", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["avatar_url"] is None

    async def test_delete_avatar_unauthenticated_returns_401(self, client):
        resp = await client.delete("/api/athlete/avatar")
        assert resp.status_code == 401

    async def test_upload_replaces_old_file_on_extension_change(self, client, auth_headers, avatar_dir):
        await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("first.jpg", _JPEG + b"first", "image/jpeg")},
        )
        await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("second.png", _PNG + b"second", "image/png")},
        )
        files = list(avatar_dir.glob("*"))
        assert len(files) == 1
        assert files[0].suffix == ".png"

    async def test_avatar_url_includes_user_id(self, client, auth_headers, avatar_dir):
        user_id = (await client.get("/api/athlete", headers=auth_headers)).json()["user_id"]
        resp = await client.post(
            "/api/athlete/avatar",
            headers=auth_headers,
            files={"file": ("photo.jpg", _JPEG + b"bytes", "image/jpeg")},
        )
        assert user_id in resp.json()["avatar_url"]


class TestTrainingStatus:
    async def test_get_training_status_no_feedback(self, client, auth_headers):
        resp = await client.get("/api/athlete/training-status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] is None
        assert data["feedback"] is None
        assert data["generated_date"] is None

    async def test_trigger_training_status_sets_pending(self, client, auth_headers):
        from unittest.mock import AsyncMock
        with patch(
            "backend.app.services.llm_training_status_analyzer.analyze_training_status_bg",
            new_callable=AsyncMock,
        ):
            resp = await client.post(
                "/api/athlete/training-status", json={}, headers=auth_headers
            )
        assert resp.status_code == 202
        assert resp.json()["status"] == "pending"

        status_resp = await client.get("/api/athlete/training-status", headers=auth_headers)
        assert status_resp.json()["status"] == "pending"

    async def test_trigger_while_pending_returns_pending_immediately(self, client, auth_headers):
        from unittest.mock import AsyncMock
        with patch(
            "backend.app.services.llm_training_status_analyzer.analyze_training_status_bg",
            new_callable=AsyncMock,
        ):
            await client.post(
                "/api/athlete/training-status", json={}, headers=auth_headers
            )
            resp = await client.post(
                "/api/athlete/training-status", json={}, headers=auth_headers
            )
        assert resp.status_code == 202
        assert resp.json()["status"] == "pending"

    async def test_get_training_status_after_analysis(self, client, auth_headers, session):
        from datetime import date
        from backend.app.models.user_orm import Athlete
        from sqlalchemy import select

        result = await session.execute(
            select(Athlete).where(Athlete.global_user_id == "test-user-00000000")
        )
        athlete = result.scalar_one()
        athlete.training_status = "MOOD:knowing\n\nYou are training well."
        athlete.training_status_status = "done"
        athlete.training_status_date = date.today()
        await session.commit()

        resp = await client.get("/api/athlete/training-status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert "training well" in data["feedback"]
        assert data["generated_date"] is not None

    async def test_auto_training_status_first_access_triggers_analysis(
        self, client, auth_headers, session
    ):
        from datetime import date, timedelta
        from unittest.mock import AsyncMock
        from backend.app.models.user_orm import Athlete
        from sqlalchemy import select

        result = await session.execute(
            select(Athlete).where(Athlete.global_user_id == "test-user-00000000")
        )
        athlete = result.scalar_one()
        athlete.app_settings = {**(athlete.app_settings or {}), "auto_training_status": True}
        athlete.training_status_date = date.today() - timedelta(days=1)
        athlete.training_status_status = None
        await session.commit()

        with patch(
            "backend.app.services.llm_training_status_analyzer.analyze_training_status_bg",
            new_callable=AsyncMock,
        ):
            resp = await client.get("/api/athlete/training-status", headers=auth_headers)

        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    async def test_pending_with_null_updated_at_resets_to_error(self, client, auth_headers, session):
        from backend.app.models.user_orm import Athlete

        result = await session.execute(
            select(Athlete).where(Athlete.global_user_id == "test-user-00000000")
        )
        athlete = result.scalar_one()
        athlete.training_status_status = "pending"
        athlete.training_status_updated_at = None
        await session.commit()

        resp = await client.get("/api/athlete/training-status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    async def test_pending_older_than_timeout_resets_to_error(self, client, auth_headers, session):
        from datetime import timedelta, timezone
        from backend.app.models.user_orm import Athlete

        result = await session.execute(
            select(Athlete).where(Athlete.global_user_id == "test-user-00000000")
        )
        athlete = result.scalar_one()
        athlete.training_status_status = "pending"
        athlete.training_status_updated_at = datetime.now(timezone.utc) - timedelta(minutes=31)
        await session.commit()

        resp = await client.get("/api/athlete/training-status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"
