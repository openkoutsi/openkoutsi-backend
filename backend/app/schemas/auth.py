from typing import Optional

from pydantic import BaseModel, EmailStr, field_validator


def _validate_password_strength(v: str) -> str:
    if len(v) < 12:
        raise ValueError("Password must be at least 12 characters")
    if not any(c.isupper() for c in v):
        raise ValueError("Password must contain at least one uppercase letter")
    if not any(c.isdigit() for c in v):
        raise ValueError("Password must contain at least one digit")
    return v


class RegisterRequest(BaseModel):
    username: str
    password: str
    invite_token: str
    display_name: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)


class SignupRequest(BaseModel):
    """Self-serve signup. Requires an email address and password only; remaining
    profile details are collected during onboarding after the email is verified.
    """
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)


class VerifyEmailRequest(BaseModel):
    token: str


class RequestPasswordResetRequest(BaseModel):
    """Request that a password-reset link be emailed to the given address."""
    email: EmailStr


class LoginRequest(BaseModel):
    # Accepts either a username or an email address as the login identifier.
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    """A generic, non-enumerating acknowledgement."""
    detail: str


class AdminResetTokenRequest(BaseModel):
    username: str


class DeleteAccountRequest(BaseModel):
    password: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)
