from typing import Optional

from pydantic import BaseModel, EmailStr, field_validator


# bcrypt hashes at most 72 bytes and, since 5.0, raises rather than silently
# truncating. Nothing caught that, so a longer password was an unhandled 500:
# on login an unauthenticated one, and on signup a server error in place of an
# explanation for anyone whose passphrase ran long (issue #102, F-07).
#
# The limit is on the UTF-8 *encoding*, not the character count — 72 emoji are
# 288 bytes — so this is checked in bytes, which is also the unit bcrypt's own
# error speaks in.
_MAX_PASSWORD_BYTES = 72


def _validate_password_length(v: str) -> str:
    """The one rule that applies to every password field, set or supplied.

    Checked when *verifying* a password too, not only when setting one: bcrypt
    raises on length before it hashes, so an over-long value fails the same way
    whether or not the account exists — but it fails with a 500 unless it is
    turned away here first.
    """
    if len(v.encode("utf-8")) > _MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password must be at most {_MAX_PASSWORD_BYTES} bytes "
            "(accented and non-Latin characters count as more than one)"
        )
    return v


def _validate_password_strength(v: str) -> str:
    _validate_password_length(v)
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

    # Length only — the strength rules belong to *setting* a password. Applying
    # them here would turn a wrong-password attempt into a lecture about
    # uppercase letters, and would lock out any account whose password predates
    # the current policy.
    @field_validator("password")
    @classmethod
    def password_length(cls, v: str) -> str:
        return _validate_password_length(v)


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

    @field_validator("password")
    @classmethod
    def password_length(cls, v: str) -> str:
        return _validate_password_length(v)


class ChangeEmailRequest(BaseModel):
    """Ask for the account's email address to be changed (or set) — issue #62.

    The current password is required for the same reason ``DeleteAccountRequest``
    requires it: a session alone must not be enough to move the login identifier
    and the password-reset target somewhere the holder controls.
    """
    new_email: EmailStr
    password: str

    # Length only, as on login and delete-account: this checks a password, it
    # does not set one, so the strength rules would only lecture an account
    # whose password predates them.
    @field_validator("password")
    @classmethod
    def password_length(cls, v: str) -> str:
        return _validate_password_length(v)


class ConfirmEmailChangeRequest(BaseModel):
    token: str


class AccountResponse(BaseModel):
    """The caller's own account identifiers.

    Nothing else exposed this: ``AthleteResponse`` carries the training profile
    and no login identity at all, so the web app had no way to show the address
    it is about to let you change.
    """
    username: Optional[str] = None
    email: Optional[str] = None
    email_verified: bool = False
    # Set while a change is awaiting confirmation at that address.
    pending_email: Optional[str] = None


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)
