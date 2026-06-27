"""SSRF guard for outbound LLM requests."""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

from fastapi import HTTPException

log = logging.getLogger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("169.254.0.0/16"),   # IPv4 link-local (AWS/GCP/Azure metadata)
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
    ipaddress.ip_network("fd00:ec2::254/128"), # GCP internal metadata (IPv6)
]


def check_url_safe(url: str) -> tuple[str, int]:
    """Validate *url* against SSRF risks.

    Returns *(resolved_host, port)* — the caller should connect to this
    IP directly rather than re-resolving the hostname, to prevent DNS rebinding.

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

    resolved_ip_str = addr_info[0][4][0]
    try:
        resolved_ip = ipaddress.ip_address(resolved_ip_str)
    except ValueError:
        raise HTTPException(status_code=502, detail="LLM hostname resolved to an unparseable address.")

    for blocked in _BLOCKED_NETWORKS:
        if resolved_ip in blocked:
            log.warning(
                "SSRF guard: blocked request to %s (resolved to %s, in blocked range %s)",
                url, resolved_ip, blocked,
            )
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Requests to {resolved_ip} are not permitted. "
                    "That address is a cloud-provider metadata range."
                ),
            )

    return resolved_ip_str, port
