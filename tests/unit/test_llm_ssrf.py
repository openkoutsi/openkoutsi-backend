"""Unit tests for the LLM SSRF guard (check_url_safe)."""
import socket
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from backend.app.core import config


def _check(url: str):
    from backend.app.core.ssrf import check_url_safe
    return check_url_safe(url)


def _resolving_to(*ips: str):
    """Patch getaddrinfo to answer with *ips* (IPv6 entries get a v6 family)."""
    infos = []
    for ip in ips:
        if ":" in ip:
            infos.append((socket.AF_INET6, socket.SOCK_STREAM, 0, '', (ip, 80, 0, 0)))
        else:
            infos.append((socket.AF_INET, socket.SOCK_STREAM, 0, '', (ip, 80)))
    return patch("socket.getaddrinfo", return_value=infos)


def _allow_private(enabled: bool = True):
    """Set the self-hosted opt-out explicitly.

    Every test here states the flag it means rather than inheriting one: the
    suite-wide conftest fixture turns the opt-out *on* so the LLM fixtures can
    use localhost, which is the opposite of the production default these tests
    are about.
    """
    return patch.object(config.settings, "llm_allow_private_networks", enabled)


def _deny_private():
    """The production default."""
    return _allow_private(False)


class TestSchemeValidation:
    def test_http_scheme_accepted(self):
        with _resolving_to("1.2.3.4"):
            _check("http://llm.example.com/v1")

    def test_https_scheme_accepted(self):
        with _resolving_to("1.2.3.4"):
            _check("https://llm.example.com/v1")

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


class TestAlwaysBlockedAddresses:
    """Metadata, link-local and multicast — refused even with the opt-out on."""

    @pytest.mark.parametrize("url", [
        "http://169.254.169.254/latest/meta-data/",  # AWS/GCP/Azure metadata
        "http://169.254.1.1/",                       # any link-local
    ])
    def test_link_local_blocked(self, url):
        with _deny_private(), pytest.raises(HTTPException) as exc_info:
            _check(url)
        assert exc_info.value.status_code == 403

    def test_ipv6_link_local_blocked(self):
        with _deny_private(), _resolving_to("fe80::1"):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://some-host/v1")
        assert exc_info.value.status_code == 403

    def test_gcp_ipv6_metadata_blocked(self):
        with _deny_private(), _resolving_to("fd00:ec2::254"):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://some-host/v1")
        assert exc_info.value.status_code == 403

    def test_multicast_blocked(self):
        with _deny_private(), _resolving_to("224.0.0.1"):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://some-host/v1")
        assert exc_info.value.status_code == 403

    @pytest.mark.parametrize("ip", ["169.254.169.254", "fe80::1", "fd00:ec2::254"])
    def test_opt_out_does_not_reopen_metadata(self, ip):
        """LLM_ALLOW_PRIVATE_NETWORKS is for Ollama, not for the metadata service.

        fd00:ec2::254 in particular sits inside the ULA range the opt-out
        unblocks, so it only stays refused because the always-blocked list is
        consulted first.
        """
        with _allow_private(), _resolving_to(ip):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://some-host/v1")
        assert exc_info.value.status_code == 403


class TestPrivateNetworksBlockedByDefault:
    """F-02: these were all reachable from a user-supplied base URL."""

    @pytest.mark.parametrize("ip", [
        "127.0.0.1",     # loopback
        "10.0.0.5",      # RFC 1918
        "192.168.1.1",   # RFC 1918
        "172.16.0.1",    # RFC 1918
        "100.64.0.1",    # CGNAT
        "0.0.0.0",       # "this network" — reaches loopback on Linux
        "::1",           # IPv6 loopback
        "fc00::1",       # IPv6 ULA
    ])
    def test_private_address_blocked(self, ip):
        with _deny_private(), _resolving_to(ip):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://some-host/v1")
        assert exc_info.value.status_code == 403

    def test_loopback_literal_blocked(self):
        """The reported PoC pointed base_url straight at 127.0.0.1."""
        with _deny_private(), pytest.raises(HTTPException) as exc_info:
            _check("http://127.0.0.1:34145/v1")
        assert exc_info.value.status_code == 403

    def test_ipv4_mapped_ipv6_loopback_blocked(self):
        """::ffff:127.0.0.1 reaches loopback while wearing an IPv6 spelling."""
        with _deny_private(), _resolving_to("::ffff:127.0.0.1"):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://some-host/v1")
        assert exc_info.value.status_code == 403

    def test_error_names_the_opt_out(self):
        """An operator hitting this needs to know which switch to flip."""
        with _deny_private(), pytest.raises(HTTPException) as exc_info:
            _check("http://127.0.0.1:11434/v1")
        assert "LLM_ALLOW_PRIVATE_NETWORKS" in exc_info.value.detail


class TestPrivateNetworksWithOptOut:
    """The self-hosted case: Ollama on localhost or a LAN box."""

    def test_loopback_allowed(self):
        with _allow_private():
            ip, port = _check("http://127.0.0.1:11434/v1")
        assert ip == "127.0.0.1"
        assert port == 11434

    def test_rfc1918_allowed(self):
        with _allow_private(), _resolving_to("10.0.0.5"):
            ip, _ = _check("http://ollama-box:11434/v1")
        assert ip == "10.0.0.5"


class TestAllAddressesChecked:
    def test_blocked_second_answer_rejects_the_name(self):
        """A name answering with a public *and* a loopback address is refused.

        The guard used to look only at addr_info[0], so the order the resolver
        happened to return decided whether the request went through.
        """
        with _deny_private(), _resolving_to("1.2.3.4", "127.0.0.1"):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://dual.example.com/v1")
        assert exc_info.value.status_code == 403

    def test_all_public_answers_allowed(self):
        with _resolving_to("1.2.3.4", "5.6.7.8"):
            ip, _ = _check("http://multi.example.com/v1")
        assert ip == "1.2.3.4"

    def test_public_ip_allowed(self):
        with _resolving_to("1.2.3.4"):
            ip, _ = _check("https://my-llm.example.com/v1")
        assert ip == "1.2.3.4"


class TestDnsResolutionFailure:
    def test_unresolvable_host_returns_502(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name not found")):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://does-not-exist.invalid/v1")
        assert exc_info.value.status_code == 502

    def test_empty_resolution_returns_502(self):
        with patch("socket.getaddrinfo", return_value=[]):
            with pytest.raises(HTTPException) as exc_info:
                _check("http://empty.example.com/v1")
        assert exc_info.value.status_code == 502
