from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator


class JoinRequestCreate(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    message: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class JoinRequestResponse(BaseModel):
    id: str
    team_slug: str
    username: str
    display_name: Optional[str] = None
    message: Optional[str] = None
    status: str
    created_at: datetime
