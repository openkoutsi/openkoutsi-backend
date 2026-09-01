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

import logging
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

    async def test_it_asks_for_matched_points_by_their_filter_names(self):
        """The engine's filter keys are not its response keys.

        Points come back under `matched_points`, but they are requested as
        `matched.…` — `kMatchedEdgeIndex = "matched.edge_index"` in
        valhalla/baldr/attributes_controller.h. Asking for
        `matched_points.edge_index` is rejected on the engine's own stdout and
        answered with a 200 that simply omits the points, so nothing in our
        logs or our response says the request was wrong.
        """
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["json"] = json.loads(request.content)
            return httpx.Response(200, json=_trace(["gravel"] * 4))

        await _client(handler).match(_POINTS)
        attributes = captured["json"]["filters"]["attributes"]
        assert "matched.edge_index" in attributes
        assert "matched.type" in attributes
        assert not any(a.startswith("matched_points.") for a in attributes), (
            "matched_points.* is the response shape, not the filter vocabulary"
        )


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


class TestOverlapSilence:
    """Silence from one chunk must not discard another chunk's real match.

    Map matching is least certain at the *head* of a trace, and an overlap is
    exactly the leading CHUNK_OVERLAP points of every chunk after the first.
    Counting `None` as disagreement threw those away at every boundary — and
    they landed as UNKNOWN, whose Crr is the pavement curve, silently
    re-solving them as the tarmac this feature exists to stop assuming.
    """

    async def test_an_unmatched_second_opinion_does_not_discard_a_good_match(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json=_trace(["gravel"] * sm.CHUNK_POINTS))
            # Chunk 2 fails to snap its leading overlap, then matches the rest.
            values = [None] * sm.CHUNK_OVERLAP + ["gravel"] * (
                sm.CHUNK_POINTS - sm.CHUNK_OVERLAP
            )
            return httpx.Response(200, json=_trace(values))

        got = await _client(handler).match([(61.5, 20.5)] * 1200)
        overlap = range(sm.CHUNK_POINTS - sm.CHUNK_OVERLAP, sm.CHUNK_POINTS)
        assert all(got[i] == "gravel" for i in overlap), (
            "chunk 1 matched these cleanly; chunk 2 merely said nothing"
        )

    async def test_a_real_disagreement_still_disputes(self):
        """The rule it must not weaken: value vs different value is unsettled."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            value = "gravel" if calls["n"] == 1 else "dirt"
            return httpx.Response(200, json=_trace([value] * sm.CHUNK_POINTS))

        got = await _client(handler).match([(61.5, 20.5)] * 1200)
        overlap = range(sm.CHUNK_POINTS - sm.CHUNK_OVERLAP, sm.CHUNK_POINTS)
        assert all(got[i] is None for i in overlap)


class TestTheCircuitBreaker:
    async def test_a_4xx_does_not_write_the_sidecar_off_for_other_courses(self):
        """A 400 is about this request — a shape the engine will not take.

        Valhalla returns 400 for a chunk over its configured max_trace_shape,
        so one athlete's awkward course must not give every other athlete
        `unavailable` for the next five minutes.
        """
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(400, text="Exceeded max shape count")
            return httpx.Response(200, json=_trace(["gravel"] * 4))

        matcher = _client(handler)
        assert await matcher.match(_POINTS) is None
        assert await matcher.match(_POINTS) is not None, "the breaker tripped on a 4xx"
        assert calls["n"] == 2

    @pytest.mark.parametrize("status", [500, 502, 503, 429])
    async def test_a_5xx_or_429_does_trip_it(self, status):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(status, text="unwell")

        matcher = _client(handler)
        await matcher.match(_POINTS)
        await matcher.match(_POINTS)
        assert calls["n"] == 1, "an unhealthy sidecar should not be re-probed at once"

    async def test_a_course_in_flight_stops_when_the_sidecar_dies(self):
        """Otherwise the breaker is only ever consulted at entry.

        A long course would spend the rest of its budget re-proving what the
        first failed chunk already established.
        """
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectError("down")

        await _client(handler).match([(61.5, 20.5)] * (sm.CHUNK_POINTS * 10))
        assert calls["n"] == 1, "kept hammering a sidecar already known to be down"


class TestAMisshapenPayload:
    """A 200 promises valid JSON, not the shape we asked for."""

    @pytest.mark.parametrize(
        "payload",
        [
            ["not", "an", "object"],
            {"edges": ["a string"], "matched_points": [{"edge_index": 0}]},
            {"edges": [{"surface": "gravel"}], "matched_points": ["a string"]},
        ],
    )
    async def test_it_degrades_instead_of_raising(self, payload):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        assert await _client(handler).match(_POINTS) is None

    async def test_it_marks_the_sidecar_unhealthy_like_any_other_non_answer(self):
        """Every other way of not getting an answer sets the breaker."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=["not an object"])

        await _client(handler).match(_POINTS)
        assert BASE in sm._UNAVAILABLE_UNTIL


class TestClientLifetime:
    """`is_configured` is a settings read; it must not allocate a pool."""

    def test_repeated_calls_share_one_client(self, monkeypatch):
        monkeypatch.setattr(sm.settings, "valhalla_url", BASE)
        monkeypatch.setattr(sm, "_client", None)
        clients = {id(sm.get_surface_matcher()._http) for _ in range(5)}
        assert len(clients) == 1

    def test_asking_whether_it_is_configured_builds_nothing(self, monkeypatch):
        monkeypatch.setattr(sm.settings, "valhalla_url", BASE)
        monkeypatch.setattr(sm, "_client", None)
        assert sm.surface_matching_configured() is True
        assert sm._client is None, "answering a settings question opened a pool"

    def test_an_unconfigured_instance_says_so_without_a_client(self, monkeypatch):
        monkeypatch.setattr(sm.settings, "valhalla_url", "")
        assert sm.surface_matching_configured() is False

    async def test_closing_releases_the_client(self, monkeypatch):
        monkeypatch.setattr(sm.settings, "valhalla_url", BASE)
        monkeypatch.setattr(sm, "_client", None)
        sm.get_surface_matcher()
        assert sm._client is not None
        await sm.close_surface_matcher()
        assert sm._client is None

    async def test_a_closed_client_is_rebuilt_rather_than_reused(self, monkeypatch):
        """Shutdown then use must not hand out a client that cannot connect."""
        monkeypatch.setattr(sm.settings, "valhalla_url", BASE)
        monkeypatch.setattr(sm, "_client", None)
        sm.get_surface_matcher()
        await sm.close_surface_matcher()
        assert sm.get_surface_matcher()._http.is_closed is False
        await sm.close_surface_matcher()


class TestAnEngineThatRejectedOurFilter:
    """The shape a live Valhalla returns when the filter keys are wrong.

    It answers 200 with the edges it understood and omits `matched_points`
    entirely, reporting `Invalid filter attribute` only on its own stdout. That
    is indistinguishable from a healthy answer unless something looks for it —
    which is how a whole deployment classified nothing while reporting no
    error anywhere in the backend.
    """

    _REJECTED = {
        "edges": [
            {"surface": "paved_smooth", "way_id": 1},
            {"surface": "gravel", "way_id": 2},
        ]
        # ...and no matched_points at all.
    }

    async def test_it_degrades_rather_than_claiming_an_all_unknown_course(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=self._REJECTED)

        assert await _client(handler).match(_POINTS) is None

    async def test_it_says_so_in_the_log(self):
        """The one thing that would have made this obvious in minutes.

        Captured with a handler attached straight to this module's logger
        rather than through `caplog`. Somewhere earlier in a full run Alembic's
        `fileConfig` fires with its default `disable_existing_loggers=True`,
        which sets `disabled` on every logger imported by then — so a
        caplog-based assertion here passes alone and fails in the suite. The
        app process never calls `fileConfig` (the entrypoint migrates in a
        separate process, and `_stamp_at_head` reads the script directory
        without executing `env.py`), so this is a test-process artefact; the
        test just should not depend on global logging state to prove a local
        thing.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=self._REJECTED)

        records: list[logging.LogRecord] = []

        class _Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger(sm.__name__)
        was_disabled, was_level = logger.disabled, logger.level
        collector = _Collect()
        logger.disabled = False
        logger.setLevel(logging.WARNING)
        logger.addHandler(collector)
        try:
            await _client(handler).match(_POINTS)
        finally:
            logger.removeHandler(collector)
            logger.disabled, logger.level = was_disabled, was_level

        messages = [record.getMessage() for record in records]
        assert any(
            "matched_points" in message and "filter" in message
            for message in messages
        ), f"a response with edges but no points must name why; got {messages}"

    async def test_a_genuinely_empty_answer_is_not_mistaken_for_it(self):
        """No edges *and* no points is the engine snapping nothing, which is
        an ordinary outcome and must not raise the filter warning."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"edges": [], "matched_points": []})

        assert await _client(handler).match(_POINTS) is None
