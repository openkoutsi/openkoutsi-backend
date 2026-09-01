"""Map-matching a course track against OSM (issue #56, Stage 2).

The engine is a **Valhalla sidecar the self-hoster runs themselves**, off by
default, reachable only from inside the deployment. Nothing here talks to a
third party: the coordinates of a stored course go to a container on the same
box and no further, which is the whole reason the sidecar exists rather than a
hosted routing API.

Behind an interface on purpose. :class:`SurfaceMatcher` is what the rest of the
backend depends on, so the engine is swappable and — more usefully — every test
in the suite runs against a double rather than 15 GB of tiles.

Three things this module is careful about:

* **Absent is not broken.** With no ``VALHALLA_URL`` configured, or with the
  sidecar unreachable, :meth:`match` returns ``None`` and the course keeps its
  Stage 1 analysis. A course without surface data is a complete and useful
  thing and must never be presented as an error.
* **A course cannot occupy the sidecar indefinitely.** Point count, chunk
  count and wall-clock are all bounded; past the budget the remainder is
  reported as unmatched rather than waited on. A 400 km upload degrades in
  resolution, not in availability.
* **The chunk overlap is evidence, not just seam glue.** A point matched in two
  chunks yields two independent answers, and disagreement is the signature of a
  bad snap — so it downgrades that point's confidence rather than being
  silently resolved in favour of whichever chunk came last.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Protocol, Sequence

import httpx

from openkoutsi import surface as surface_math

from backend.app.core.config import settings

log = logging.getLogger(__name__)

# ── bounds ────────────────────────────────────────────────────────────────────

# The stored track is ~8 m spaced, which would put 50 000 points on a 400 km
# course. Surface changes at road granularity, not at 8 m, so matching runs on
# a decimated copy and the answer is mapped back onto the full track.
MATCH_SPACING_M = 25.0

# Points per request, and how many are repeated from the previous chunk. The
# overlap exists so a chunk boundary does not sit in the middle of a snap
# decision; it doubles as the agreement check described above.
CHUNK_POINTS = 1000
CHUNK_OVERLAP = 50

# Hard ceiling on requests for one course. At 25 m spacing this covers ~1 500 km
# of route — generous for anything anyone rides in a day, and finite, which is
# the point: one upload must not be able to occupy the sidecar for ever.
MAX_CHUNKS = 60

# The sidecar is a container on the same host, so a connect that does not
# happen fast is not going to happen. The read budget is per chunk; a course is
# bounded by the total below regardless.
REQUEST_TIMEOUT = httpx.Timeout(15.0, connect=2.0)
TOTAL_BUDGET_S = 120.0

#: When to next probe a sidecar that failed, keyed by base URL. Expires rather
#: than latching, following ``bridge_client``: a sidecar restarted a second
#: after we gave up must not stay written off until the backend is redeployed.
_UNAVAILABLE_UNTIL: dict[str, float] = {}
_RE_PROBE_AFTER_S = 300.0

# The engine's *filter* keys are not its *response* keys. Matched points come
# back under `matched_points`, but they are requested as `matched.…` — see
# `kMatchedEdgeIndex = "matched.edge_index"` in valhalla/baldr/
# attributes_controller.h. Getting this wrong is quiet: the engine logs
# `Invalid filter attribute` on its own stdout, answers 200 with the edges it
# did understand, and simply omits `matched_points` — so every point comes back
# unmatched and the course degrades to "unavailable" with nothing in the
# backend log to say why. `_surfaces_from_trace` now names that case.
_ATTRIBUTES = [
    "edge.surface",
    "edge.road_class",
    "edge.use",
    "edge.way_id",
    "matched.edge_index",
    "matched.type",
]


class SurfaceMatcher(Protocol):
    """What the backend depends on. Implemented by the engine and by fakes."""

    @property
    def is_configured(self) -> bool:
        """Whether matching is possible at all on this instance.

        Callers check this to *degrade*, hiding the capability rather than
        letting a request fail — the same discipline as
        :attr:`EmailProvider.is_configured`.
        """
        ...

    async def match(
        self, points: Sequence[tuple[float, float]]
    ) -> Optional[list[Optional[str]]]:
        """One raw engine surface value per input point, or ``None``.

        ``None`` for the whole call means the matcher could not be reached and
        the caller should carry on without surface data. ``None`` for a single
        point means that point was not matched, which is an ordinary outcome
        and classifies as unknown.
        """
        ...


class NullSurfaceMatcher:
    """The matcher on an instance that has none. Absent, not broken."""

    @property
    def is_configured(self) -> bool:
        return False

    async def match(
        self, points: Sequence[tuple[float, float]]
    ) -> Optional[list[Optional[str]]]:
        return None


class ValhallaSurfaceMatcher:
    """Map matching through a self-hosted Valhalla ``trace_attributes``."""

    def __init__(self, http: httpx.AsyncClient, base_url: str):
        self._http = http
        self._base = base_url.rstrip("/")

    @property
    def is_configured(self) -> bool:
        return bool(self._base)

    async def match(
        self, points: Sequence[tuple[float, float]]
    ) -> Optional[list[Optional[str]]]:
        if not self.is_configured or not points:
            return None
        if _UNAVAILABLE_UNTIL.get(self._base, 0.0) > time.monotonic():
            return None

        # One answer per input point, plus the count of chunks that agreed, so
        # a boundary point matched twice can be checked rather than overwritten.
        answers: list[Optional[str]] = [None] * len(points)
        disputed = [False] * len(points)
        seen = [False] * len(points)

        deadline = time.monotonic() + TOTAL_BUDGET_S
        chunks = list(_chunk_indices(len(points)))
        if len(chunks) > MAX_CHUNKS:
            log.warning(
                "Course needs %d match chunks, capping at %d — the remainder "
                "is reported as unmatched rather than waited on",
                len(chunks),
                MAX_CHUNKS,
            )
            chunks = chunks[:MAX_CHUNKS]

        for start, end in chunks:
            if _UNAVAILABLE_UNTIL.get(self._base, 0.0) > time.monotonic():
                # The sidecar died mid-course. Without this the breaker is only
                # ever consulted at entry, so a course already in flight spends
                # the rest of its budget re-proving what the first failed chunk
                # established — up to MAX_CHUNKS connect timeouts.
                log.warning(
                    "Surface matcher at %s went away mid-course; stopping early",
                    self._base,
                )
                break
            if time.monotonic() >= deadline:
                log.warning(
                    "Surface matching hit its %.0f s budget; the rest of the "
                    "course keeps no surface data",
                    TOTAL_BUDGET_S,
                )
                break
            chunk = await self._match_chunk(points[start:end])
            if chunk is None:
                # A single failed chunk is not a failed course: keep what was
                # matched and let the rest read as unknown.
                continue
            for offset, value in enumerate(chunk):
                i = start + offset
                if i >= len(answers):
                    break
                if seen[i]:
                    # Second opinion on an overlap point. Two chunks that
                    # disagree mean the snap is not settled, whichever one
                    # spoke last — but a chunk that snapped *nothing* has not
                    # disagreed, it has only stayed silent, and silence must
                    # not discard a good answer. Map matching is least certain
                    # at the head of a trace, which is exactly what an overlap
                    # is, so treating None as a contradiction threw away
                    # CHUNK_OVERLAP points at every boundary — and they landed
                    # as UNKNOWN, whose Crr is the *pavement* curve, quietly
                    # re-solving them as the tarmac this feature exists to
                    # stop assuming.
                    if value is not None and answers[i] != value:
                        disputed[i] = True
                elif value is not None:
                    # Only a real match counts as having been seen. A chunk
                    # that snapped nothing has not answered the question, and
                    # a point another chunk later matches is not a
                    # contradiction of that silence.
                    answers[i] = value
                    seen[i] = True

        if not any(seen):
            # The engine answered and snapped nothing — a course outside the
            # region whose tiles this instance built, most likely. That is "no
            # surface data", not "the whole route is unknown": drawing a
            # full-length grey band would claim we had looked and found
            # something when we had only looked.
            return None
        # A disputed point keeps its class but loses its claim to being a fact;
        # `None` is how this layer says "unknown", which classifies as inferred.
        return [None if disputed[i] else answers[i] for i in range(len(answers))]

    async def _match_chunk(
        self, points: Sequence[tuple[float, float]]
    ) -> Optional[list[Optional[str]]]:
        body = {
            "shape": [{"lat": lat, "lon": lon} for lat, lon in points],
            "costing": "bicycle",
            "shape_match": "map_snap",
            "filters": {"attributes": _ATTRIBUTES, "action": "include"},
        }
        try:
            response = await self._http.post(
                f"{self._base}/trace_attributes", json=body, timeout=REQUEST_TIMEOUT
            )
            # httpx raises only for transport failures, so without this a 400
            # from a course the engine cannot snap would parse as an empty
            # match and read as "all unknown" instead of as a failure.
            response.raise_for_status()
            payload = response.json()
            # A 200 only promises *valid JSON*, not the shape we asked for. A
            # proxy answering with an error envelope, or an engine whose
            # response shape has moved, lands here — and reading it must fail
            # on this module's own degradation path rather than raise out of it.
            if not isinstance(payload, dict):
                raise ValueError("trace_attributes did not return an object")
            surfaces = _surfaces_from_trace(payload, len(points))
        except httpx.HTTPStatusError as exc:
            # A 4xx is a statement about *this request* — a shape the engine
            # will not take, a trace it cannot snap — not about the sidecar's
            # health. Failing the chunk is right; writing the whole sidecar off
            # on behalf of every other athlete's course is not, and it would
            # present as an intermittent five-minute outage nobody could
            # diagnose from `surface_status` alone.
            status = exc.response.status_code
            log.warning(
                "Surface matcher at %s refused a chunk with %s", self._base, status
            )
            if status >= 500 or status == 429:
                _UNAVAILABLE_UNTIL[self._base] = time.monotonic() + _RE_PROBE_AFTER_S
            return None
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            log.warning("Surface matcher at %s did not answer", self._base)
            _UNAVAILABLE_UNTIL[self._base] = time.monotonic() + _RE_PROBE_AFTER_S
            return None

        _UNAVAILABLE_UNTIL.pop(self._base, None)
        return surfaces


def _chunk_indices(total: int):
    """Overlapping ``[start, end)`` windows covering ``total`` points."""
    if total <= CHUNK_POINTS:
        yield (0, total)
        return
    step = CHUNK_POINTS - CHUNK_OVERLAP
    start = 0
    while start < total:
        yield (start, min(start + CHUNK_POINTS, total))
        if start + CHUNK_POINTS >= total:
            return
        start += step


def _surfaces_from_trace(payload: dict, count: int) -> list[Optional[str]]:
    """Pull one edge surface per matched point out of a trace response."""
    edges = payload.get("edges") or []
    surfaces = [edge.get("surface") for edge in edges]
    matched = payload.get("matched_points") or []

    if edges and not matched:
        # The engine understood the request well enough to return edges but
        # gave us no points to hang them on. Without the mapping there is
        # nothing to store, and the likeliest cause is that it rejected our
        # filter keys — which it reports on its *own* stdout and not in the
        # response, so say it here or nobody sees it.
        log.warning(
            "Surface matcher returned %d edges but no matched_points. The "
            "engine may have rejected the filter attributes (%s) — check its "
            "log for 'Invalid filter attribute'.",
            len(edges),
            ", ".join(a for a in _ATTRIBUTES if a.startswith("matched.")),
        )

    out: list[Optional[str]] = [None] * count
    for i, point in enumerate(matched[:count]):
        if point.get("type") == "unmatched":
            continue
        index = point.get("edge_index")
        if index is None or not isinstance(index, int) or not 0 <= index < len(surfaces):
            continue
        out[i] = surfaces[index]
    return out


def decimate(
    points: Sequence[tuple[float, float]],
    distances_m: Sequence[float],
    spacing_m: float = MATCH_SPACING_M,
) -> list[int]:
    """Indices of the points to actually send, at ``spacing_m`` apart.

    Returned as indices rather than points so the answer can be mapped back
    onto the full track without a second distance walk.
    """
    if not points:
        return []
    kept = [0]
    last = distances_m[0]
    for i in range(1, len(points)):
        if distances_m[i] - last >= spacing_m:
            kept.append(i)
            last = distances_m[i]
    if kept[-1] != len(points) - 1:
        kept.append(len(points) - 1)
    return kept


def expand(
    matched: Sequence[Optional[str]],
    kept: Sequence[int],
    total: int,
) -> list[Optional[str]]:
    """Spread a decimated answer back over every track point.

    Nearest-preceding: a point inherits the surface of the last sample matched
    at or before it, which is what "the road you are on right now" means.
    """
    out: list[Optional[str]] = [None] * total
    for position, index in enumerate(kept):
        end = kept[position + 1] if position + 1 < len(kept) else total
        value = matched[position] if position < len(matched) else None
        for j in range(index, min(end, total)):
            out[j] = value
    return out


async def match_track(
    matcher: SurfaceMatcher,
    points: Sequence[tuple[float, float]],
    distances_m: Sequence[float],
) -> Optional[list[list]]:
    """Match a whole stored track, in the form the course row stores.

    Returns ``[[raw_value, confidence], …]`` — one entry per track point — or
    ``None`` when the matcher is absent or unreachable, which the caller reads
    as "this course has no surface data" rather than as a failure.
    """
    if not matcher.is_configured or not points:
        return None
    kept = decimate(points, distances_m)
    answers = await matcher.match([points[i] for i in kept])
    if answers is None:
        return None
    raw = expand(answers, kept, len(points))
    return [[value, surface_math.confidence_for(value)] for value in raw]


# ── wiring ────────────────────────────────────────────────────────────────────


def surface_matching_configured() -> bool:
    """Whether a sidecar is wired up. A settings read, and nothing more.

    Most callers only want this — a course response saying whether the action
    can be offered — and building an HTTP client to answer it would allocate a
    connection pool per request on the hottest course route.
    """
    return bool(settings.valhalla_url)


#: The one client the matcher uses, built on first need and closed at shutdown.
#: Every other outbound client in this codebase is scoped by an ``async with``;
#: this one cannot be, because the background match outlives the request that
#: scheduled it. So it is owned here instead of by nobody: an
#: ``httpx.AsyncClient`` that is dropped rather than closed does not reliably
#: release the sockets in its pool, and this is a path a course page hits.
_client: httpx.AsyncClient | None = None


def build_surface_matcher() -> SurfaceMatcher:
    """The matcher this instance is configured for, or the absent one."""
    global _client
    if not surface_matching_configured():
        return NullSurfaceMatcher()
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            # Sized against the chunk loop, which is sequential: one live
            # connection is enough, and a small idle pool keeps the sidecar
            # from collecting sockets between courses.
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    return ValhallaSurfaceMatcher(_client, settings.valhalla_url)


async def close_surface_matcher() -> None:
    """Release the shared client. Called from the app's lifespan shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def get_surface_matcher() -> SurfaceMatcher:
    """Injectable accessor — overridden in tests, as the email provider is."""
    return build_surface_matcher()


__all__ = [
    "CHUNK_OVERLAP",
    "close_surface_matcher",
    "surface_matching_configured",
    "CHUNK_POINTS",
    "MATCH_SPACING_M",
    "MAX_CHUNKS",
    "NullSurfaceMatcher",
    "SurfaceMatcher",
    "TOTAL_BUDGET_S",
    "ValhallaSurfaceMatcher",
    "build_surface_matcher",
    "decimate",
    "expand",
    "get_surface_matcher",
    "match_track",
]
