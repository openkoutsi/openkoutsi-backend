"""ORM models for the dedicated third-party API-usage database (issue #66).

One append-only row per **outbound** call to a third party: every HTTP request
to Strava or Wahoo, and every transactional email handed to the email provider.
Lives in its own database (``ApiUsageBase``) with no registry foreign keys, so
its rows can be pruned/rotated independently and a user-deletion sweep is a
plain ``DELETE ... WHERE user_id = ?``.

Two questions are served by the one table:

* **Headroom** — "can we start a big import right now?" Strava reports our
  standing authoritatively in the rate-limit headers of *every* response, so
  each row carries that reading and "where are we now" is the newest row for a
  service. No second table and no second write path.
* **Volume** — "are we trending toward the ceiling, and what does email cost?"
  Answered by counting our own rows, bucketed by day/week/month.
"""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import ApiUsageBase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ApiUsage(ApiUsageBase):
    """One outbound third-party call.

    The unit is the **HTTP request**, not the provider method: quotas are
    counted in requests, and one method call is not one request
    (``fetch_zones()`` issues two, a Wahoo FIT download may follow a CDN
    redirect). Email is the exception and is counted per message, which is how
    it is billed.
    """

    __tablename__ = "api_usage"
    __table_args__ = (
        Index("ix_api_usage_service_created", "service", "created_at"),
        Index("ix_api_usage_user_created", "user_id", "created_at"),
    )

    #: An INTEGER primary key, which SQLite makes an alias for the rowid — so it
    #: costs no index of its own. A random UUID would need a second B-tree and
    #: scatter its page splits, and buy nothing: this table is append-only,
    #: nothing joins to it, and it is only ever read in aggregate.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=_now, nullable=False
    )
    #: Not separately indexed: ``service`` is the prefix of the
    #: ``(service, created_at)`` composite below, so SQLite uses that for
    #: service-only predicates too. One less B-tree to maintain on what is the
    #: hottest write path in the system.
    service: Mapped[str] = mapped_column(String, nullable=False)
    #: A **normalised template**, never the raw URL — ``/activities/12345/streams``
    #: is stored as ``/activities/{id}/streams``. Raw URLs carry activity ids and,
    #: on some paths, query-string credentials. Headers are never stored at all.
    endpoint: Mapped[str] = mapped_column(String, nullable=False)
    method: Mapped[str] = mapped_column(String, nullable=False)
    #: None for a transport failure — the request never produced a response.
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # ok | client_error | server_error | transport_error | rate_limited
    outcome: Mapped[str] = mapped_column(String, nullable=False)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    #: Nullable: most calls are made on a user's behalf during sync, but bridge
    #: polling and some OAuth paths have no user.
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # ── Rate-limit reading taken from this response's headers ──────────────
    #
    # Null where the service sends none (email, Wahoo). ``short`` is the
    # provider's 15-minute window, ``daily`` the 24-hour one. A reading is only
    # meaningful *within the window it was observed in* — see
    # :mod:`backend.app.services.quota`, which is the only thing that should
    # interpret these.
    ratelimit_usage_short: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ratelimit_limit_short: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ratelimit_usage_daily: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ratelimit_limit_daily: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    #: Strava's separate, lower quota for *read* requests, kept apart from the
    #: overall one rather than merged into it. A backfill is all reads, and the
    #: read ceiling is roughly half the overall one — so it is the ceiling that
    #: binds first for exactly the question this feature exists to answer, and
    #: folding the two together would hide it.
    ratelimit_read_usage_short: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ratelimit_read_limit_short: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ratelimit_read_usage_daily: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ratelimit_read_limit_daily: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
