"""Render the instance privacy policy from a Jinja2 template.

The policy text lives in ``backend/app/templates/privacy_policy.md.j2`` and is
filled in from the ``PRIVACY_*`` settings (see ``backend.app.core.config``).
openkoutsi is self-hosted, so the operator of each instance is the GDPR data
controller and supplies the controller identity, retention periods, hosting
details, etc. Any unset field falls back to the template's ``[…]`` placeholder,
and the policy is flagged as not-yet-configured so an operator can tell at a
glance that it still needs completing.

The rendered Markdown is served (unauthenticated) at
``GET /api/legal/privacy-policy`` and is what the consent screen shows the user
before they accept. The policy version is kept in sync with the consent version
so bumping the policy forces re-consent.
"""
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from backend.app.core.config import settings

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_TEMPLATE_NAME = "privacy_policy.md.j2"

# Fields that must be filled for the policy to count as "configured". If any is
# blank the rendered document carries a visible not-configured warning.
_REQUIRED_FIELDS = (
    "privacy_controller_name",
    "privacy_controller_country",
    "privacy_contact_email",
    "privacy_effective_date",
)


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(enabled_extensions=(), default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def policy_version() -> str:
    """Current privacy-policy version (kept in sync with the consent version)."""
    return settings.privacy_policy_version


def is_configured() -> bool:
    """True when the operator has filled in the essential controller fields."""
    return all(getattr(settings, name) for name in _REQUIRED_FIELDS)


def _context() -> dict:
    return {
        "app_name": settings.privacy_app_name,
        "policy_version": settings.privacy_policy_version,
        "configured": is_configured(),
        "effective_date": settings.privacy_effective_date,
        "last_updated": settings.privacy_last_updated or settings.privacy_effective_date,
        "controller_name": settings.privacy_controller_name,
        "controller_address": settings.privacy_controller_address,
        "controller_country": settings.privacy_controller_country,
        "controller_registration_number": settings.privacy_controller_registration_number,
        "contact_email": settings.privacy_contact_email,
        "dpo": settings.privacy_dpo,
        "eu_representative": settings.privacy_eu_representative,
        "hosting_provider": settings.privacy_hosting_provider,
        "encryption_description": settings.privacy_encryption_description,
        "retention_period": settings.privacy_retention_period,
        "diagnostic_retention_period": settings.privacy_diagnostic_retention_period,
        "children_min_age": settings.privacy_children_min_age,
        "supervisory_authority": settings.privacy_supervisory_authority,
        "ai_servers": settings.llm_allowed_servers_list,
    }


def render_privacy_policy() -> str:
    """Render the privacy policy to Markdown using the current settings."""
    return _env().get_template(_TEMPLATE_NAME).render(**_context()).strip() + "\n"
