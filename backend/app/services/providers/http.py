"""The one place every outbound provider HTTP request passes through (issue #66).

There is no shared HTTP client in this package — :mod:`.base` is a pure ABC with
no transport, and each provider method opens its own ``httpx.AsyncClient``. This
module supplies the client those call sites build instead, so that every request
is counted exactly once.

**Counting happens at the HTTP-request layer, not the provider-method layer**,
because quotas are counted in requests and one method call is not one request:
``StravaProviderClient.fetch_zones()`` issues two (``/athlete`` and
``/athlete/zones``, gathered), and ``WahooClient.download_fit_file()`` may follow
a CDN redirect. Counting per method would under-report precisely where the
backfill spends the most.

A transport wrapper is used rather than ``event_hooks`` for three reasons, the
last decisive:

* a response hook never runs for a **transport failure** — DNS, TLS and connect
  errors produce no response at all, so those requests would vanish;
* redirects and connection-level retries pass through the transport
  individually, which is what makes the count match the provider's;
* it is the single point that sees **every response's headers**, which is where
  the rate-limit reading comes from.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import httpx

from backend.app.services.api_usage import parse_rate_limit, schedule_api_usage

#: Hosts we own the URL shape of, and the path prefix to strip from each so the
#: recorded template reads like the provider's documented endpoint rather than
#: its version scheme. Anything not listed is a redirect target (a CDN), whose
#: path is a signed blob and is never recorded — see :func:`normalise_endpoint`.
_KNOWN_HOSTS: dict[str, tuple[str, ...]] = {
    "www.strava.com": ("/api/v3",),
    "api.wahooligan.com": ("/v1",),
}

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                      r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{8,}$")


def _is_identifier(segment: str) -> bool:
    """Whether a path segment is an id rather than part of the route's name.

    Deliberately generous: a segment that is *probably* an identifier is
    replaced, because the cost of a false positive is a slightly coarser bucket
    in an admin table, while the cost of a false negative is an athlete's
    activity id in a database that exists to count things.
    """
    if not segment:
        return False
    if segment.isdigit():
        return True
    if _UUID_RE.match(segment) or _HEX_RE.match(segment):
        return True
    # An opaque token: long, and mixing digits in with the letters. Real route
    # names in these APIs are short words ("activities", "power_zones").
    return len(segment) >= 16 and any(c.isdigit() for c in segment)


def normalise_endpoint(url: httpx.URL) -> str:
    """A storable template for *url* — never the URL itself.

    ``/api/v3/activities/12345/streams`` becomes ``/activities/{id}/streams``.
    Raw URLs carry activity ids and, on some paths, query-string credentials, so
    the query is dropped whole and every segment that looks like an identifier is
    replaced. A host we do not recognise is a redirect target whose path is a
    signed blob: it is recorded as ``<host>/*`` — enough to attribute the request
    and to group CDN downloads together, and nothing more.
    """
    host = url.host
    prefixes = _KNOWN_HOSTS.get(host)
    if prefixes is None:
        return f"{host}/*"

    path = url.path or "/"
    for prefix in prefixes:
        if path.startswith(prefix):
            path = path[len(prefix):] or "/"
            break

    segments = [
        "{id}" if _is_identifier(seg) else seg
        for seg in path.split("/")
    ]
    return "/".join(segments) or "/"


class CountingTransport(httpx.AsyncBaseTransport):
    """Wraps a transport, recording one ``api_usage`` row per request.

    The write is **scheduled, not awaited**. Swallowing its failures is enough to
    stop accounting failing a sync, but not enough to stop it *slowing* one: the
    usage engine's pool and connect timeouts run to tens of seconds, so a
    database under an exclusive lock — what the documented retention ``VACUUM``
    takes — would add ~30s to every outbound provider request. Handing the
    response back first is what makes "never at the athlete's expense" true of
    latency as well as of failure.
    """

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        *,
        service: str,
        user_id: str | None = None,
    ) -> None:
        self._inner = inner
        self._service = service
        self._user_id = user_id

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        endpoint = normalise_endpoint(request.url)
        started = time.perf_counter()
        try:
            response = await self._inner.handle_async_request(request)
        except Exception:
            # No response was produced, so there is no status and no rate-limit
            # reading — but the request was still made, and a backfill that dies
            # on DNS is exactly the kind of thing this table should show.
            schedule_api_usage(
                service=self._service,
                endpoint=endpoint,
                method=request.method,
                status_code=None,
                duration_ms=_elapsed_ms(started),
                user_id=self._user_id,
            )
            raise

        schedule_api_usage(
            service=self._service,
            endpoint=endpoint,
            method=request.method,
            status_code=response.status_code,
            duration_ms=_elapsed_ms(started),
            user_id=self._user_id,
            rate_limit=parse_rate_limit(response.headers),
        )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def _elapsed_ms(started: float) -> int:
    """Milliseconds since *started*.

    Measured to response *headers*, not to the last byte of the body: that is
    where the transport hands back, and it is the figure that reflects the
    provider rather than the size of what we asked for.
    """
    return int((time.perf_counter() - started) * 1000)


def provider_client(
    service: str,
    *,
    timeout: Optional[httpx.Timeout] = None,
    follow_redirects: bool = False,
    user_id: str | None = None,
    **kwargs,
) -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` that counts every request it makes.

    The drop-in replacement for ``httpx.AsyncClient(...)`` at every provider call
    site. ``user_id`` is optional; when omitted the recorded row takes whatever
    :func:`~backend.app.services.api_usage.attribute_to_user` has established for
    the surrounding context, and NULL when nothing has.

    **The client is built first and its transports wrapped afterwards**, rather
    than constructed with ``transport=``. httpx reads ``HTTP_PROXY`` /
    ``HTTPS_PROXY`` / ``NO_PROXY`` only when no explicit transport is given
    (``allow_env_proxies = trust_env and transport is None``), so passing one
    drops the whole proxy map — which would have silently stopped provider sync
    on any instance whose egress goes through a proxy, and reported the result as
    a transport error rather than explaining it. Every mount is wrapped too, so a
    request that takes a proxy route is counted exactly like a direct one.
    """
    client = httpx.AsyncClient(
        timeout=timeout, follow_redirects=follow_redirects, **kwargs
    )

    def wrap(transport):
        return CountingTransport(transport, service=service, user_id=user_id)

    # Private attributes, because httpx offers no public hook for decorating the
    # transports it built. Every access is type-guarded rather than merely
    # present-guarded, so this degrades to wrapping less rather than raising:
    # an httpx upgrade that renames or reshapes them still yields a working
    # client, and a test that patches `httpx.AsyncClient` with a mock gets its
    # mock back untouched instead of a TypeError from iterating it.
    transport = getattr(client, "_transport", None)
    if isinstance(transport, httpx.AsyncBaseTransport):
        client._transport = wrap(transport)

    mounts = getattr(client, "_mounts", None)
    if isinstance(mounts, dict):
        client._mounts = {
            pattern: wrap(t) if isinstance(t, httpx.AsyncBaseTransport) else t
            for pattern, t in mounts.items()
        }
    return client
