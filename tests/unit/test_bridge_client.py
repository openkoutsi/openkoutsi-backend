"""The bridge client, and the deploy window it exists to survive (issue #50).

The backend and the two bridges are three separate images. `okdeploy-pull.sh`
recreates only the services whose digest moved, in whatever order Compose picks,
so a new backend can meet an old bridge that has no `/events/claim`. Without a
fallback that window is one in which nothing drains either queue, and the
backlog grows against a ceiling of 10 000 events.
"""
import httpx
import pytest

from backend.app.services import bridge_client
from backend.app.services.bridge_client import BridgeClient

BASE = "http://bridge.test"
SECRET = "s3cret"


@pytest.fixture(autouse=True)
def _forget_probes():
    """The legacy probe is cached per URL, so tests must not leak into each other."""
    bridge_client._LEGACY_BRIDGES.clear()
    yield
    bridge_client._LEGACY_BRIDGES.clear()


def _client(handler) -> BridgeClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE)
    return BridgeClient(http, BASE, SECRET)


class TestAModernBridge:
    async def test_claim_returns_the_batch(self):
        def handler(request):
            assert request.url.path == "/events/claim"
            assert request.headers["Authorization"] == f"Bearer {SECRET}"
            return httpx.Response(200, json=[{"id": "e1", "claim_token": "t1"}])

        events = await _client(handler).claim_batch()
        assert events == [{"id": "e1", "claim_token": "t1"}]

    async def test_ack_carries_the_token(self):
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, json={"status": "acked"})

        await _client(handler).ack("e1", "t1")
        assert seen == [f"{BASE}/events/e1/ack?claim_token=t1"]

    async def test_nack_carries_the_token(self):
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, json={"status": "released"})

        await _client(handler).nack("e1", "t1")
        assert seen == [f"{BASE}/events/e1/nack?claim_token=t1"]


class TestALegacyBridge:
    """New backend, old bridge — the mixed-version window during a deploy."""

    def _handler(self, calls):
        def handler(request):
            calls.append(request.url.path)
            if request.url.path == "/events/claim":
                return httpx.Response(404)
            if request.url.path == "/events/pending":
                return httpx.Response(200, json=[{"id": "e1"}])
            return httpx.Response(200, json={})
        return handler

    async def test_it_falls_back_to_the_old_flow(self):
        calls: list[str] = []
        events = await _client(self._handler(calls)).claim_batch()

        assert events == [{"id": "e1"}], "the fallback did not drain the queue"
        assert calls == ["/events/claim", "/events/pending"]

    async def test_the_probe_is_not_repeated_every_poll(self):
        """One 404 per bridge, not one per minute forever."""
        calls: list[str] = []
        client = _client(self._handler(calls))

        await client.claim_batch()
        await client.claim_batch()
        await client.claim_batch()

        assert calls.count("/events/claim") == 1, calls
        assert calls.count("/events/pending") == 3

    async def test_ack_uses_the_old_claim_verb(self):
        """On a legacy bridge `claim` is the only terminal verb there is."""
        calls: list[str] = []
        await _client(self._handler(calls)).ack("e1", None)
        assert calls == ["/events/e1/claim"]

    async def test_nack_is_a_no_op(self):
        """An unclaimed event is already deliverable there — nothing to release."""
        calls: list[str] = []
        await _client(self._handler(calls)).nack("e1", None)
        assert calls == []


class TestAnUnreachableBridge:
    async def test_claim_returns_nothing_rather_than_raising(self):
        """A poll that cannot reach its bridge is a skipped tick, not a crash."""
        def handler(request):
            raise httpx.ConnectError("bridge is down")

        assert await _client(handler).claim_batch() == []

    async def test_a_failed_ack_does_not_raise(self):
        """The visibility deadline covers it: the event comes back and replays."""
        def handler(request):
            raise httpx.ConnectError("bridge is down")

        await _client(handler).ack("e1", "t1")  # must not raise
