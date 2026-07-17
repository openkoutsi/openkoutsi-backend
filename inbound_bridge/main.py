"""
Inbound Email Bridge — standalone FastAPI service (issue #38).

The optional "third bridge" alongside ``strava_bridge/`` and ``wahoo_bridge/``.
It is **off by default**: a self-hosted instance only runs it (via the
``inbound-email`` Docker Compose profile) when the operator opts in, so an
instance that omits it has zero inbound-email surface.

Flow: the email provider POSTs mail for the operator address here → the bridge
verifies the provider's webhook **signature** on the **raw body**, parses it,
and **holds** it in a small SQLite queue → the backend **polls** ``/events/pending``
on its own schedule, delivers each message to administrators, and claims it.

Holding (rather than immediately relaying) is the whole point: this public
receiver stays up independently of the backend, so mail is never lost while the
backend is down — it simply waits here until the next poll. It mirrors the
Strava/Wahoo bridge template exactly (queue + bearer-guarded polling endpoints).

The signature scheme mirrors the shared email module's ``EuromailProvider``
(``backend/app/services/email/euromail.py``): HMAC-SHA256 over
``{timestamp}.{body}`` carried in ``X-Euromail-Signature`` as
``t={ts},v1={hex}``. It is reimplemented inline (like the HMAC check in
``strava_bridge/main.py``) so the bridge stays a self-contained deployable that
does not import the backend package.

Endpoints:
  GET  /                    — health check / provider challenge handshake
  POST /webhook             — receive a provider inbound webhook, verify + queue
  GET  /events/pending      — return unclaimed messages (Bearer auth)
  POST /events/{id}/claim   — mark a message as claimed (Bearer auth)
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import DateTime, String, delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

log = logging.getLogger(__name__)


# ── Settings ──────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", secrets_dir="/run/secrets", extra="ignore"
    )

    database_path: str = "bridge.db"

    # Provider selection (only "euromail" is supported today; kept explicit so a
    # future provider is a one-line change here and in _verify_signature).
    email_provider: str = "euromail"

    # Secret for verifying the provider's inbound webhook signature.
    euromail_webhook_secret: str = ""

    # Bearer secret the backend presents when polling/claiming events.
    inbound_bridge_secret: str = "changeme"


settings = Settings()


# EuroMail signs webhook bodies as HMAC-SHA256 over ``{timestamp}.{body}`` and
# carries the result in ``X-Euromail-Signature``. Deliveries whose timestamp is
# outside this tolerance are rejected to blunt replay attacks.
_SIGNATURE_HEADER = "x-euromail-signature"
_SIGNATURE_TOLERANCE_SECONDS = 300


# ── Database ──────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class InboundEmailEvent(Base):
    """A verified inbound email held for the backend to poll."""

    __tablename__ = "inbound_email_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    from_addr: Mapped[str] = mapped_column(String, nullable=False)
    to_addr: Mapped[str] = mapped_column(String, nullable=False, default="")
    subject: Mapped[str] = mapped_column(String, nullable=False, default="")
    text: Mapped[str] = mapped_column(String, nullable=False, default="")
    message_id: Mapped[str] = mapped_column(String, nullable=False)
    received_at: Mapped[str] = mapped_column(String, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


engine = create_async_engine(f"sqlite+aiosqlite:///{settings.database_path}")
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


# ── Cleanup background task ───────────────────────────────────────────────


async def _cleanup_loop() -> None:
    """Delete events older than 7 days every hour."""
    while True:
        await asyncio.sleep(3600)
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(InboundEmailEvent).where(InboundEmailEvent.created_at < cutoff)
            )
            await session.commit()


# ── App factory ───────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="openkoutsi Inbound Email Bridge", lifespan=lifespan)


# ── Auth helper ───────────────────────────────────────────────────────────


def _require_bearer(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {settings.inbound_bridge_secret}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Signature verification ────────────────────────────────────────────────


def _header_value(headers, name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return ""


def _parse_signature_header(header: str) -> tuple[str, list[str]]:
    timestamp = ""
    signatures: list[str] = []
    for part in header.split(","):
        key, _, value = part.strip().partition("=")
        if key == "t":
            timestamp = value
        elif key == "v1":
            signatures.append(value)
    return timestamp, signatures


def _verify_signature(secret: str, raw_body: bytes, headers) -> bool:
    """Validate the provider's inbound signature over the raw body.

    Returns False (never raises) when the secret is unset, the header is
    missing/malformed, the timestamp is stale, or no signature matches.
    """
    if not secret:
        return False
    header = _header_value(headers, _SIGNATURE_HEADER)
    if not header:
        return False
    timestamp, signatures = _parse_signature_header(header)
    if not timestamp or not signatures:
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > _SIGNATURE_TOLERANCE_SECONDS:
        return False
    try:
        signed = f"{timestamp}.{raw_body.decode('utf-8')}".encode()
    except UnicodeDecodeError:
        return False
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(expected, sig) for sig in signatures)


# ── Inbound payload parsing ────────────────────────────────────────────────


def _as_address(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return _as_address(value[0]) if value else ""
    if isinstance(value, dict):
        return str(value.get("email") or value.get("address") or "")
    return str(value)


def _as_iso(value: Any) -> str:
    """Best-effort normalise a timestamp to an ISO-8601 string ("" if absent)."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    return str(value)


def _parse_inbound(payload: Any) -> dict:
    """Map a verified provider payload to the fields we hold.

    Mirrors ``EuromailProvider._parse_inbound_payload`` (the ``data``/``email``
    envelope and the field-name variants EuroMail uses across REST and webhook
    payloads).
    """
    if not isinstance(payload, dict):
        raise ValueError("Inbound payload is not an object")
    message = payload.get("data") or payload.get("email") or payload
    if not isinstance(message, dict):
        raise ValueError("Inbound payload has no email object")

    from_addr = _as_address(message.get("from_address") or message.get("from"))
    to_addr = _as_address(
        message.get("to_addresses")
        or message.get("to")
        or message.get("to_address")
    )
    subject = message.get("subject") or ""
    text = message.get("text_body") or message.get("text") or ""
    message_id = message.get("id") or message.get("message_id") or ""
    received_at = (
        message.get("created_at")
        or message.get("received_at")
        or payload.get("timestamp")
    )

    if not from_addr or not message_id:
        raise ValueError("Inbound payload missing sender or message id")

    return {
        "from_addr": from_addr,
        "to_addr": to_addr,
        "subject": subject,
        "text": text,
        "message_id": str(message_id),
        "received_at": _as_iso(received_at),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/")
async def health(request: Request):
    """Health check / provider challenge handshake.

    Some providers verify a new inbound route by GETting it with a challenge
    query parameter and expecting it echoed back; we support that while also
    serving as a plain liveness probe.
    """
    challenge = request.query_params.get("challenge")
    if challenge is not None:
        return {"challenge": challenge}
    return {"status": "ok"}


@app.post("/webhook", status_code=202)
async def receive_webhook(request: Request):
    """Verify a provider inbound webhook and queue it for the backend to poll."""
    raw_body = await request.body()

    if not _verify_signature(
        settings.euromail_webhook_secret, raw_body, request.headers
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        parsed = _parse_inbound(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Bad payload: {exc}")

    event = InboundEmailEvent(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        **parsed,
    )
    async with AsyncSessionLocal() as session:
        session.add(event)
        await session.commit()

    return Response(status_code=202)


# ── Polling endpoints (called by the backend) ─────────────────────────────


@app.get("/events/pending")
async def get_pending_events(request: Request):
    """Return all unclaimed messages. Auth: Bearer inbound_bridge_secret."""
    _require_bearer(request)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(InboundEmailEvent)
            .where(InboundEmailEvent.claimed_at.is_(None))
            .order_by(InboundEmailEvent.created_at)
            .limit(100)
        )
        events = result.scalars().all()
        return [
            {
                "id": e.id,
                "from_addr": e.from_addr,
                "to_addr": e.to_addr,
                "subject": e.subject,
                "text": e.text,
                "message_id": e.message_id,
                "received_at": e.received_at,
                "created_at": e.created_at.isoformat(),
            }
            for e in events
        ]


@app.post("/events/{event_id}/claim", status_code=200)
async def claim_event(event_id: str, request: Request):
    """Mark a message as claimed. Auth: Bearer inbound_bridge_secret."""
    _require_bearer(request)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(InboundEmailEvent).where(InboundEmailEvent.id == event_id)
        )
        event = result.scalar_one_or_none()
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        event.claimed_at = datetime.now(timezone.utc)
        await session.commit()
    return {"status": "claimed"}
