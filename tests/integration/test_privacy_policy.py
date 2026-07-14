"""Integration tests for GET /api/legal/privacy-policy and its rendering."""
import pytest

from backend.app.core.config import settings
from backend.app.services import privacy_policy


class TestPrivacyPolicyEndpoint:
    async def test_public_no_auth_required(self, client):
        resp = await client.get("/api/legal/privacy-policy")
        assert resp.status_code == 200

    async def test_returns_version_and_markdown(self, client):
        resp = await client.get("/api/legal/privacy-policy")
        data = resp.json()
        assert data["version"] == settings.privacy_policy_version
        assert "# Privacy & Data Processing Policy" in data["markdown"]
        assert "Article 9" in data["markdown"]

    async def test_reports_unconfigured_by_default(self, client):
        resp = await client.get("/api/legal/privacy-policy")
        data = resp.json()
        # The default test settings do not fill controller identity.
        assert data["configured"] is False
        assert "not been fully configured" in data["markdown"]
        assert "[Legal entity name]" in data["markdown"]


class TestPrivacyPolicyRendering:
    def test_configured_values_are_rendered(self, monkeypatch):
        privacy_policy._env.cache_clear()
        monkeypatch.setattr(settings, "privacy_controller_name", "Koutsi Oy", raising=False)
        monkeypatch.setattr(settings, "privacy_controller_country", "Finland", raising=False)
        monkeypatch.setattr(settings, "privacy_contact_email", "privacy@koutsi.dev", raising=False)
        monkeypatch.setattr(settings, "privacy_effective_date", "2026-07-14", raising=False)
        monkeypatch.setattr(settings, "privacy_hosting_provider", "Hetzner, Finland (EU)", raising=False)

        assert privacy_policy.is_configured() is True
        md = privacy_policy.render_privacy_policy()
        assert "Koutsi Oy" in md
        assert "privacy@koutsi.dev" in md
        assert "Hetzner, Finland (EU)" in md
        assert "not been fully configured" not in md
        assert "[Legal entity name]" not in md

    def test_allowed_llm_servers_listed(self, monkeypatch):
        privacy_policy._env.cache_clear()
        monkeypatch.setattr(
            settings, "llm_allowed_servers",
            "https://api.openai.com/v1,http://localhost:11434/v1", raising=False,
        )
        md = privacy_policy.render_privacy_policy()
        assert "https://api.openai.com/v1" in md
        assert "http://localhost:11434/v1" in md

    def test_no_allowed_servers_uses_generic_language(self, monkeypatch):
        privacy_policy._env.cache_clear()
        monkeypatch.setattr(settings, "llm_allowed_servers", "", raising=False)
        md = privacy_policy.render_privacy_policy()
        assert "does not restrict which LLM provider" in md


@pytest.fixture(autouse=True)
def _clear_env_cache():
    """Keep the Jinja env cache from leaking template state between tests."""
    yield
    privacy_policy._env.cache_clear()
