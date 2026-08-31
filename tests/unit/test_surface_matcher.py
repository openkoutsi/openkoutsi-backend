"""The OSM surface matcher client (issue #56).

No live engine anywhere — CI must never need 15 GB of routing tiles. The real
client is driven through ``httpx.MockTransport``, which exercises the request
it actually builds and the parsing it actually does, and the degradation paths
are the point rather than an afterthought:

* unreachable, timing out, or answering with an error → ``None``, and the
  caller keeps a Stage 1 course;
* partially matched, or matched with unmatched points → whatever was matched,
  with the rest reading as unknown;
* two overlapping chunks that disagree → the disputed point loses its claim to
  being a fact.
"""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from backend.app.services import surface_matcher as sm


@pytest.fixture(autouse=True)
def _forget_unavailable_sidecars():
    """The negative cache is module state; a test must not leak into the next."""
    sm._UNAVAILABLE_UNTIL.clear()
    yield
    sm._UNAVAILABLE_UNTIL.clear()


BASE = "http://valhalla:8002"


def _client(handler) -> sm.ValhallaSurfaceMatcher:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return sm.ValhallaSurfaceMatcher(http, BASE)


def _trace(surfaces: list[str | None]) -> dict:
    """A trace_attributes response matching each point to its own edge."""
    return {
        "edges": [{"surface": s} for s in surfaces if s is not None],
        "matched_points": [
            {"type": "unmatched"}
            if s is None
            else {
                "type": "matched",
                "edge_index": len([x for x in surfaces[:i] if x is not None]),
            }
            for i, s in enumerate(surfaces)
        ],
    }


def _responding(surfaces: list[str | None]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_trace(surfaces))

    return handler


_POINTS = [(61.5 + i * 1e-4, 20.5) for i in range(4)]


class TestTheRequestItBuilds:
    async def test_it_posts_a_bicycle_map_snap_trace(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = httpx.Request(
                "POST", request.url, content=request.content
            ).content
            import json

            captured["json"] = json.loads(request.content)
            return httpx.Response(200, json=_trace(["gravel"] * 4))

        await _client(handler).match(_POINTS)
        assert captured["url"] == f"{BASE}/trace_attributes"
        assert captured["json"]["costing"] == "bicycle"
        assert captured["json"]["shape_match"] == "map_snap"
        assert captured["json"]["shape"][0] == {"lat": _POINTS[0][0], "lon": _POINTS[0][1]}

    async def test_it_asks_for_the_surface_attributes(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["json"] = json.loads(request.content)
            return httpx.Response(200, json=_trace(["gravel"] * 4))

        await _client(handler).match(_POINTS)
        attributes = captured["json"]["filters"]["attributes"]
        assert "edge.surface" in attributes
        assert captured["json"]["filters"]["action"] == "include"


class TestParsing:
    async def test_each_point_gets_its_edge_surface(self):
        got = await _client(_responding(["paved_smooth", "gravel", "gravel", "dirt"])).match(
            _POINTS
        )
        assert got == ["paved_smooth", "gravel", "gravel", "dirt"]

    async def test_unmatched_points_come_back_as_none(self):
        """An ordinary outcome, not a failure — it classifies as unknown."""
        got = await _client(_responding(["paved_smooth", None, None, "dirt"])).match(_POINTS)
        assert got == ["paved_smooth", None, None, "dirt"]

    async def test_an_edge_index_out_of_range_is_ignored_rather_than_raising(self):
        """A malformed response is no answer, not an answer of "unknown"."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "edges": [{"surface": "gravel"}],
                    "matched_points": [{"type": "matched", "edge_index": 99}] * 4,
                },
            )

        assert await _client(handler).match(_POINTS) is None

    async def test_a_partial_match_keeps_what_was_matched(self):
        got = await _client(_responding(["gravel", None, "dirt", None])).match(_POINTS)
        assert got == ["gravel", None, "dirt", None]

    async def test_a_response_with_no_matches_at_all_reads_as_unreachable(self):
        """Nothing matched is nothing to store, so the course stays Stage 1."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"edges": [], "matched_points": []})

        assert await _client(handler).match(_POINTS) is None


class TestDegradation:
    """Unconfigured or unreachable, the feature is *absent* rather than broken."""

    async def test_an_unconfigured_matcher_matches_nothing(self):
        matcher = sm.NullSurfaceMatcher()
        assert matcher.is_configured is False
        assert await matcher.match(_POINTS) is None

    async def test_an_empty_base_url_is_not_configured(self):
        assert sm.ValhallaSurfaceMatcher(httpx.AsyncClient(), "").is_configured is False

    async def test_a_transport_failure_returns_none(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        assert await _client(handler).match(_POINTS) is None

    async def test_a_timeout_returns_none(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow")

        assert await _client(handler).match(_POINTS) is None

    async def test_an_http_error_returns_none_rather_than_an_empty_match(self):
        """Without raise_for_status a 500 would parse as "all unknown".

        That is the difference between "we could not ask" and "the road has no
        surface", and only one of them should reach the athlete.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="tiles not loaded")

        assert await _client(handler).match(_POINTS) is None

    async def test_a_non_json_body_returns_none(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway</html>")

        assert await _client(handler).match(_POINTS) is None

    async def test_a_failed_sidecar_is_not_hammered_on_the_next_course(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectError("down")

        matcher = _client(handler)
        await matcher.match(_POINTS)
        await matcher.match(_POINTS)
        assert calls["n"] == 1

    async def test_the_negative_cache_expires_rather_than_latching(self):
        """A sidecar restarted a second later must not stay written off."""
        import time

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_trace(["gravel"] * 4))

        sm._UNAVAILABLE_UNTIL[BASE] = time.monotonic() - 1.0
        assert await _client(handler).match(_POINTS) is not None


class TestChunking:
    def test_short_tracks_are_one_chunk(self):
        assert list(sm._chunk_indices(500)) == [(0, 500)]

    def test_chunks_overlap(self):
        chunks = list(sm._chunk_indices(2500))
        for before, after in zip(chunks, chunks[1:]):
            assert after[0] < before[1], "chunks must overlap, not abut"

    def test_chunks_cover_every_point(self):
        covered: set[int] = set()
        for start, end in sm._chunk_indices(2500):
            covered.update(range(start, end))
        assert covered == set(range(2500))

    async def test_a_course_cannot_ask_for_more_than_the_chunk_cap(self):
        """One upload must not be able to occupy the sidecar for ever."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=_trace(["gravel"] * sm.CHUNK_POINTS))

        huge = [(61.5, 20.5)] * (sm.CHUNK_POINTS * (sm.MAX_CHUNKS + 20))
        await _client(handler).match(huge)
        assert calls["n"] <= sm.MAX_CHUNKS

    async def test_one_failed_chunk_does_not_fail_the_course(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 2:
                return httpx.Response(500)
            return httpx.Response(200, json=_trace(["gravel"] * sm.CHUNK_POINTS))

        got = await _client(handler).match([(61.5, 20.5)] * 2500)
        assert got is not None
        assert any(v == "gravel" for v in got), "the matched part must survive"


class TestOverlapAgreement:
    async def test_chunks_that_disagree_downgrade_the_disputed_point(self):
        """The overlap is free evidence against a bad snap.

        The rule that lets a real 130 m sector survive is the one a spurious
        snap could exploit to invent one, so where two independent answers
        disagree the point stops claiming to be a fact.
        """
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            # Second chunk calls the overlap dirt where the first said gravel.
            value = "gravel" if calls["n"] == 1 else "dirt"
            return httpx.Response(200, json=_trace([value] * sm.CHUNK_POINTS))

        got = await _client(handler).match([(61.5, 20.5)] * 1500)
        overlap = range(sm.CHUNK_POINTS - sm.CHUNK_OVERLAP, sm.CHUNK_POINTS)
        assert all(got[i] is None for i in overlap), "disputed points must not assert"
        assert got[0] == "gravel", "the undisputed part is untouched"

    async def test_chunks_that_agree_keep_the_answer(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_trace(["gravel"] * sm.CHUNK_POINTS))

        got = await _client(handler).match([(61.5, 20.5)] * 1500)
        overlap = sm.CHUNK_POINTS - sm.CHUNK_OVERLAP
        assert got[overlap] == "gravel"


class TestDecimation:
    def test_it_thins_to_the_match_spacing(self):
        distances = [i * 8.0 for i in range(100)]
        kept = sm.decimate([(61.5, 20.5)] * 100, distances)
        gaps = [distances[b] - distances[a] for a, b in zip(kept, kept[1:])]
        assert all(gap >= sm.MATCH_SPACING_M or b == kept[-1] for gap, b in zip(gaps, kept[1:]))
        assert len(kept) < 100

    def test_the_endpoints_are_always_kept(self):
        distances = [i * 8.0 for i in range(100)]
        kept = sm.decimate([(61.5, 20.5)] * 100, distances)
        assert kept[0] == 0 and kept[-1] == 99

    def test_an_empty_track_decimates_to_nothing(self):
        assert sm.decimate([], []) == []

    def test_expanding_uses_the_nearest_preceding_sample(self):
        assert sm.expand(["gravel", "dirt"], [0, 4], 8) == ["gravel"] * 4 + ["dirt"] * 4

    def test_expanding_covers_every_point(self):
        out = sm.expand(["gravel", "dirt", "paved_smooth"], [0, 3, 7], 10)
        assert len(out) == 10 and all(v is not None for v in out)


class TestMatchTrack:
    """The whole-track entry point, in the shape the course row stores."""

    _TRACK = [(61.5 + i * 1e-4, 20.5) for i in range(40)]
    _DISTANCES = [i * 8.0 for i in range(40)]

    async def test_it_returns_raw_value_and_confidence_per_point(self):
        matcher = _client(_responding(["gravel"] * 40))
        stored = await sm.match_track(matcher, self._TRACK, self._DISTANCES)
        assert len(stored) == len(self._TRACK)
        assert stored[0] == ["gravel", "confirmed"]

    async def test_untagged_roads_store_as_inferred(self):
        matcher = _client(_responding(["paved_smooth"] * 40))
        stored = await sm.match_track(matcher, self._TRACK, self._DISTANCES)
        assert stored[0] == ["paved_smooth", "inferred"]

    async def test_an_absent_matcher_stores_nothing(self):
        assert (
            await sm.match_track(sm.NullSurfaceMatcher(), self._TRACK, self._DISTANCES)
            is None
        )

    async def test_an_unreachable_matcher_stores_nothing(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        assert await sm.match_track(_client(handler), self._TRACK, self._DISTANCES) is None

    async def test_an_empty_track_stores_nothing(self):
        assert await sm.match_track(_client(_responding([])), [], []) is None


class TestTheWallClockBudget:
    """Chunk count bounds the requests; this bounds the wall clock."""

    async def test_it_stops_once_the_budget_is_spent(self):
        """A slow sidecar must not hold a course open indefinitely.

        The clock is driven by the handler so the test is deterministic rather
        than actually slow: each request burns 100 s against a 150 s budget.
        """
        clock = {"t": 0.0}
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            clock["t"] += 100.0
            return httpx.Response(200, json=_trace(["gravel"] * sm.CHUNK_POINTS))

        with (
            patch.object(sm, "TOTAL_BUDGET_S", 150.0),
            patch.object(sm.time, "monotonic", lambda: clock["t"]),
        ):
            got = await _client(handler).match([(61.5, 20.5)] * 5000)

        assert calls["n"] < 6, "the budget must cut the run short"
        assert got is not None, "what was matched before the budget ran out survives"
        assert got[0] == "gravel"
        assert got[-1] is None, "the unreached tail reads as unknown, not as a guess"

    async def test_an_exhausted_budget_still_degrades_rather_than_failing(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_trace(["gravel"] * 4))

        with patch.object(sm, "TOTAL_BUDGET_S", -1.0):
            assert await _client(handler).match(_POINTS) is None

    async def test_an_empty_point_list_matches_nothing(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not reach the sidecar with no points")

        assert await _client(handler).match([]) is None


class TestTheFactory:
    def test_an_unconfigured_instance_gets_the_null_matcher(self, monkeypatch):
        monkeypatch.setattr(sm.settings, "valhalla_url", "")
        assert isinstance(sm.build_surface_matcher(), sm.NullSurfaceMatcher)

    def test_a_configured_one_gets_a_real_client(self, monkeypatch):
        monkeypatch.setattr(sm.settings, "valhalla_url", BASE)
        matcher = sm.build_surface_matcher()
        assert isinstance(matcher, sm.ValhallaSurfaceMatcher)
        assert matcher.is_configured is True

    def test_a_trailing_slash_does_not_produce_a_double_slash_url(self):
        assert sm.ValhallaSurfaceMatcher(httpx.AsyncClient(), BASE + "/")._base == BASE
