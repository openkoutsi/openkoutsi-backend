"""Unit tests for the LLM SSRF guard (check_url_safe)."""
import socket
from unittest.mock import patch

import pytest
from fastapi import HTTPException


def _check(url: str):
    from backend.app.core.ssrf import check_url_safe
    return check_url_safe(url)


class TestSchemeValidation:
    def test_http_scheme_accepted(self):
        # 127.0.0.1 resolves without DNS — loopback is allowed.
        _check("http://127.0.0.1:11434/v1")

    def test_https_scheme_accepted(self):
        _check("https://127.0.0.1:443/v1")

    def test_file_scheme_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _check("file:///etc/passwd")
        assert exc_info.value.status_code == 400
        assert "file" in exc_info.value.detail

    def test_ftp_scheme_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _check("ftp://127.0.0.1/resource")
        assert exc_info.value.status_code == 400

    def test_no_hostname_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _check("http:///path")
        assert exc_info.value.status_code == 400


class TestBlockedAddresses:
    def test_aws_metadata_ipv4_blocked(self):
        """169.254.169.254 is the AWS/GCP/Azure metadata service — must be blocked."""
        with pytest.raises(HTTPException) as exc_info:
            _check("http://169.254.169.254/latest/meta-data/")
        assert exc_info.value.status_code == 403

    def test_link_local_range_blocked(self):
        """Any 169.254.x.x address should be refused."""
        with pytest.raises(HTTPException) as exc_info:
            _check("http://169.254.1.1/")
        assert exc_info.value.status_code == 403

    def test_ipv6_link_local_blocked(self):
        """fe80:: is the IPv6 link-local prefix used for metadata on some platforms."""
        # Fake the resolution to return a link-local IPv6 address.
        fake_addrinfo = [(socket.AF_INET6, socket.SOCK_STREAM, 0, '', ('fe80::1', 80, 0, 0))]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://some-host/v1")
        assert exc_info.value.status_code == 403


class TestAllowedAddresses:
    def test_localhost_allowed(self):
        """Loopback is intentionally permitted — primary Ollama use case."""
        ip, port = _check("http://127.0.0.1:11434/v1")
        assert ip == "127.0.0.1"

    def test_private_rfc1918_allowed(self):
        """10.x LAN addresses are allowed so Ollama on a NAS etc. works."""
        fake_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, '', ('10.0.0.5', 11434))]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            ip, port = _check("http://ollama-box:11434/v1")
        assert ip == "10.0.0.5"

    def test_public_ip_allowed(self):
        fake_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, '', ('1.2.3.4', 443))]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            ip, _ = _check("https://my-llm.example.com/v1")
        assert ip == "1.2.3.4"


class TestDnsResolutionFailure:
    def test_unresolvable_host_returns_502(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name not found")):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://does-not-exist.invalid/v1")
        assert exc_info.value.status_code == 502
