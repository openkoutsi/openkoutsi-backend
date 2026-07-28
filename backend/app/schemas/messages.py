from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class MessageResponse(BaseModel):
    id: str
    type: str
    #: Machine-readable metadata (deep links, icon selection) — not the text.
    data: dict
    #: Rendered at send time. Null on messages written before messages carried
    #: their own text.
    title: Optional[str] = None
    body: Optional[str] = None
    #: Which language `title`/`body` were rendered in.
    locale: Optional[str] = None
    read_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UnreadCountResponse(BaseModel):
    count: int
