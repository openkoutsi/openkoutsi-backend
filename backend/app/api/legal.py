"""Public legal endpoints — the instance privacy policy.

The policy is rendered from a Jinja2 template filled in from the operator's
``PRIVACY_*`` configuration (see ``backend.app.services.privacy_policy``). It is
served unauthenticated so the consent screen, the login/register pages, and the
landing site can all link to and display the same current text. Its version is
kept in sync with the consent version so bumping the policy forces re-consent.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.services.privacy_policy import (
    is_configured,
    policy_version,
    render_privacy_policy,
)

router = APIRouter(prefix="/legal", tags=["legal"])


class PrivacyPolicyResponse(BaseModel):
    version: str
    configured: bool
    markdown: str


@router.get("/privacy-policy", response_model=PrivacyPolicyResponse,
            operation_id="getPrivacyPolicy", summary="Get the instance privacy policy")
async def get_privacy_policy() -> PrivacyPolicyResponse:
    return PrivacyPolicyResponse(
        version=policy_version(),
        configured=is_configured(),
        markdown=render_privacy_policy(),
    )
