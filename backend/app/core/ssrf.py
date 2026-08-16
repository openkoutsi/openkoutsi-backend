"""SSRF guard for outbound LLM requests."""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

from .config import settings

log = logging.getLogger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}

# Blocked unconditionally. These are the ranges this guard exists for: a
# self-hoster pointing at Ollama on the LAN has no reason to reach a cloud
# metadata service, so `llm_allow_private_networks` below does *not* re-open
# them. Order matters — `fd00:ec2::254` sits inside the ULA range that the
# opt-out unblocks, and this list is checked first.
_ALWAYS_BLOCKED = [
    ipaddress.ip_network("169.254.0.0/16"),    # IPv4 link-local (AWS/GCP/Azure metadata)
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
    ipaddress.ip_network("fd00:ec2::254/128"), # GCP internal metadata (IPv6)
    ipaddress.ip_network("224.0.0.0/4"),       # IPv4 multicast
    ipaddress.ip_network("240.0.0.0/4"),       # IPv4 reserved
    ipaddress.ip_network("ff00::/8"),          # IPv6 multicast
]

# Blocked unless the operator sets `LLM_ALLOW_PRIVATE_NETWORKS=true`, which is
# the self-hosted case: Ollama on localhost or a model server on the LAN. Left
# reachable by default, a user-supplied base URL is a read primitive against
# everything else running on the box and everything else on its network.
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),       # "this network"; 0.0.0.0 reaches loopback on Linux
    ipaddress.ip_network("127.0.0.0/8"),     # IPv4 loopback
    ipaddress.ip_network("10.0.0.0/8"),      # RFC 1918
    ipaddress.ip_network("172.16.0.0/12"),   # RFC 1918
    ipaddress.ip_network("192.168.0.0/16"),  # RFC 1918
    ipaddress.ip_network("100.64.0.0/10"),   # CGNAT (RFC 6598)
    ipaddress.ip_network("::/128"),          # IPv6 unspecified
    ipaddress.ip_network("::1/128"),         # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),        # IPv6 unique local
]


def _normalise(ip: ipaddress.IPv4Address | ipaddress.IPv6Address):
    """Fold an IPv4-mapped IPv6 address down to its IPv4 form.

    ``::ffff:127.0.0.1`` reaches loopback but matches none of the IPv4
    networks above while it is still wearing its IPv6 spelling.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _reject_if_blocked(ip_str: str, url: str) -> None:
    """Raise if *ip_str* is in a range this instance refuses to connect to."""
    try:
        ip = _normalise(ipaddress.ip_address(ip_str))
    except ValueError:
        raise HTTPException(
            status_code=502, detail="LLM hostname resolved to an unparseable address."
        )

    for blocked in _ALWAYS_BLOCKED:
        if ip in blocked:
            log.warning(
                "SSRF guard: blocked request to %s (resolved to %s, in blocked range %s)",
                url, ip, blocked,
            )
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Requests to {ip} are not permitted. "
                    "That address is a link-local, metadata or multicast range."
                ),
            )

    if settings.llm_allow_private_networks:
        return

    for blocked in _PRIVATE_NETWORKS:
        if ip in blocked:
            log.warning(
                "SSRF guard: blocked request to %s (resolved to %s, in private range %s)",
                url, ip, blocked,
            )
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Requests to {ip} are not permitted. That address is on a "
                    "private or loopback network. Set LLM_ALLOW_PRIVATE_NETWORKS=true "
                    "if this instance runs a self-hosted model."
                ),
            )


def check_url_safe(url: str) -> tuple[str, int]:
    """Validate *url* against SSRF risks.

    Returns *(resolved_host, port)* — the caller should connect to this IP
    directly rather than re-resolving the hostname, to prevent DNS rebinding.
    :func:`guarded_async_client` does that; prefer it over calling this and
    then handing the hostname to a plain client.

    Raises ``HTTPException(400/403/502)`` for disallowed schemes or blocked addresses.
    """
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise HTTPException(
            status_code=400,
            detail=f"LLM base URL scheme '{parsed.scheme}' is not allowed. Use http or https.",
        )

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="LLM base URL has no hostname.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not resolve LLM hostname '{hostname}': {exc}",
        )

    # Every answer, not just the first: a hostname that resolves to a public
    # address and a loopback address is judged on whichever the connection
    # happens to pick, so any blocked answer rejects the whole name.
    resolved: list[str] = []
    for info in addr_info:
        ip_str = info[4][0]
        _reject_if_blocked(ip_str, url)
        resolved.append(ip_str)

    if not resolved:
        raise HTTPException(
            status_code=502, detail=f"LLM hostname '{hostname}' resolved to no addresses."
        )

    return resolved[0], port


class _PinnedTransport(httpx.AsyncHTTPTransport):
    """Transport that connects only to an address :func:`check_url_safe` vetted.

    Checking a URL and then handing the *hostname* to httpx resolves it a
    second time, and an attacker-controlled record with a short TTL can answer
    differently on that second lookup — the check says 1.2.3.4, the connection
    goes to 127.0.0.1. So the request is rewritten to the vetted IP, with the
    Host header and the TLS SNI (and therefore certificate verification) left
    on the original hostname.

    Redirects re-enter the transport as fresh requests, so a 302 into a blocked
    range is caught the same way.

    Note that passing an explicit transport opts out of httpx's environment
    proxy autodetection. That is deliberate: a proxy would do its own name
    resolution and the pinning would mean nothing.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        hostname = request.url.host
        resolved_ip, _port = check_url_safe(str(request.url))

        if hostname == resolved_ip:
            # Already an IP literal — vetted above, nothing to rewrite.
            return await super().handle_async_request(request)

        host_header = request.headers.get("Host") or request.url.netloc.decode("ascii")
        request.extensions["sni_hostname"] = hostname
        request.url = request.url.copy_with(host=resolved_ip)
        request.headers["Host"] = host_header

        return await super().handle_async_request(request)


def guarded_async_client(**kwargs) -> httpx.AsyncClient:
    """An :class:`httpx.AsyncClient` that runs the SSRF guard on every request.

    Takes the same keyword arguments as ``httpx.AsyncClient``. Use this for any
    outbound call to a URL a user or admin supplied.
    """
    kwargs.setdefault("follow_redirects", False)
    return httpx.AsyncClient(transport=_PinnedTransport(), **kwargs)
