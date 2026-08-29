from datetime import datetime
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from openkoutsi.training_math import efficiency_factor, variability_index


def _aerobic_ratios(activity) -> dict[str, float | None]:
    """Efficiency factor and variability index, derived from stored columns.

    Both are pure ratios of ``weighted_power`` / ``avg_hr`` / ``avg_power``, so
    they are computed on read rather than persisted: nothing can drift out of
    sync with its operands, and activities processed before these metrics
    existed carry them without needing a reprocess.
    """
    ef = efficiency_factor(activity.weighted_power, activity.avg_hr)
    vi = variability_index(activity.weighted_power, activity.avg_power)
    return {
        "efficiency_factor": round(ef, 3) if ef is not None else None,
        "variability_index": round(vi, 3) if vi is not None else None,
    }


class LabelSuggestion(BaseModel):
    """A label openkoutsi thinks applies, and what the athlete said about it.

    ``state`` is ``pending`` (awaiting an answer), ``accepted`` or ``dismissed``.
    ``source`` says what proposed it — ``rule:<id>`` for one of the athlete's own
    commute rules, ``strava`` for the provider's own flag — so a client can
    explain *why* the ride was picked out, and the athlete can go fix the rule
    rather than dismissing the same wrong suggestion forever.
    """

    state: str
    source: Optional[str] = None
    at: Optional[str] = None


def _clean_suggestions(raw: Any) -> dict[str, dict]:
    """The suggestion column, with anything unreadable dropped.

    Defensive despite this column having exactly one writer: a row written by a
    future version, or hand-edited during support, must not turn every activity
    read into a 500.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for label, entry in raw.items():
        if not isinstance(label, str) or not isinstance(entry, dict):
            continue
        state = entry.get("state")
        if not isinstance(state, str):
            continue
        cleaned: dict[str, str] = {"state": state}
        for key in ("source", "at"):
            value = entry.get(key)
            if isinstance(value, str):
                cleaned[key] = value
        out[label] = cleaned
    return out


class ActivityUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    workout_category: Optional[str] = None
    labels: Optional[list[str]] = None
    notes: Optional[str] = Field(None, max_length=5000)
    rpe: Optional[int] = Field(None, ge=1, le=10)
    # Answering a suggestion (issue #63): ``{"commute": "accepted"}`` applies the
    # label and records the answer in one write, so the label and its suggestion
    # can never drift apart; ``"dismissed"`` records the refusal durably, which
    # is what stops the same ride being suggested again after every reprocess.
    label_answers: Optional[dict[str, str]] = None


class FrontendAnalysisBody(BaseModel):
    analysis: str = Field(..., min_length=1)


class AnalyzeBody(BaseModel):
    locale: Optional[str] = None


class ManualActivityCreate(BaseModel):
    """Payload for logging a workout by hand (no device recording / FIT file).

    Every field is optional so users can jot down as much or as little as they
    remember, but a completely empty submission is rejected.
    """

    sport_type: Optional[str] = None
    start_time: Optional[datetime] = None
    duration_s: Optional[int] = Field(None, gt=0)
    name: Optional[str] = None
    # Load resolution (in priority order): explicit load > rpe > avg_hr.
    # rpe/avg_hr derivation additionally require duration_s to be set.
    load: Optional[float] = Field(None, ge=0)
    rpe: Optional[int] = Field(None, ge=1, le=10)
    avg_hr: Optional[float] = Field(None, gt=0)
    max_hr: Optional[float] = Field(None, gt=0)
    avg_power: Optional[float] = Field(None, gt=0)
    avg_cadence: Optional[float] = Field(None, ge=0)
    distance_m: Optional[float] = Field(None, ge=0)
    elevation_m: Optional[float] = None

    @model_validator(mode="after")
    def _require_at_least_one_field(self) -> "ManualActivityCreate":
        if not any(
            getattr(self, field) is not None for field in type(self).model_fields
        ):
            raise ValueError("At least one field must be provided")
        return self


class IntervalResponse(BaseModel):
    interval_number: int
    start_offset_s: int
    duration_s: int
    distance_m: Optional[float] = None
    avg_hr: Optional[float] = None
    avg_power: Optional[float] = None
    avg_speed_ms: Optional[float] = None
    avg_cadence: Optional[float] = None
    is_auto_split: bool

    model_config = {"from_attributes": True}


class ActivityResponse(BaseModel):
    id: str
    athlete_id: str
    # List of provider names that contributed data to this activity,
    # e.g. ["wahoo", "strava"] or ["upload"].
    sources: list[str] = []
    name: Optional[str] = None
    sport_type: Optional[str] = None
    start_time: Optional[datetime] = None
    duration_s: Optional[int] = None
    distance_m: Optional[float] = None
    elevation_m: Optional[float] = None
    avg_power: Optional[float] = None
    weighted_power: Optional[float] = None
    avg_hr: Optional[float] = None
    max_hr: Optional[float] = None
    avg_cadence: Optional[float] = None
    load: Optional[float] = None
    intensity: Optional[float] = None
    # Aerobic response metrics (issue #37). `efficiency_factor` (weighted power
    # per heartbeat) and `variability_index` (weighted / average power) are
    # derived on read from the columns above. `decoupling_pct` is the stored
    # power:HR drift over the ride; when it is null `decoupling_reason` says why
    # a figure would be misleading — one of `too_short`, `no_power`, `no_hr`,
    # `degenerate_hr`, `variable_effort`.
    efficiency_factor: Optional[float] = None
    variability_index: Optional[float] = None
    decoupling_pct: Optional[float] = None
    decoupling_reason: Optional[str] = None
    workout_category: Optional[str] = None
    labels: list[str] = []
    # Labels proposed but not (yet) applied — see `LabelSuggestion` and
    # `services.commute`. Kept strictly apart from `labels` above, which is only
    # ever what the athlete has confirmed.
    label_suggestions: dict[str, LabelSuggestion] = {}
    notes: Optional[str] = None
    rpe: Optional[int] = None
    has_fit_file: bool = False
    # Which format the stored original is in — `fit`, `gpx` or `tcx` (issue
    # #36). Null when there is no file. `has_fit_file` keeps its name and its
    # meaning ("there is an original to download"); this says what the download
    # will actually be, so the UI can label the button and explain why a
    # GPX-sourced ride has no power data.
    original_format: Optional[str] = None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}

    # Field names this schema shares with an ORM relationship of the same name.
    # `_orm_values` skips them so the caller's own value wins instead of a lazy
    # relationship load — see `ActivityDetailResponse`.
    _ORM_OVERRIDES: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def _orm_values(cls, activity) -> dict:
        """This schema's fields read off an ORM ``Activity``.

        Every field except the three derived ones is a straight column read, so
        they are pulled by name from the schema's own field list rather than
        restated here: declaring a field is enough to have it populated.
        """
        values = {
            name: getattr(activity, name)
            for name in cls.model_fields
            if name not in cls._ORM_OVERRIDES and hasattr(activity, name)
        }
        return {
            **values,
            "sources": [s.provider for s in (activity.sources or [])],
            "labels": activity.labels or [],
            "label_suggestions": _clean_suggestions(activity.label_suggestions),
            # Issue #43: a progress code only means anything while the run is in
            # flight. The analyzer clears it on the way out, but a task killed
            # between its last progress commit and settling would leave one
            # behind, and "Koutsi is checking your power curve…" printed under a
            # finished analysis reads as a bug. Gating on the status here means
            # a stale code can never reach a client.
            "analysis_progress": (
                getattr(activity, "analysis_progress", None)
                if getattr(activity, "analysis_status", None) == "pending"
                else None
            ),
            **_aerobic_ratios(activity),
        }

    @model_validator(mode="before")
    @classmethod
    def _extract_sources(cls, data: Any) -> Any:
        """Derive `sources` and the aerobic ratios when validating from an ORM object."""
        if hasattr(data, "sources"):
            return cls._orm_values(data)
        return data


class ActivityListResponse(BaseModel):
    items: list[ActivityResponse]
    total: int
    page: int
    page_size: int


class CommuteScanResponse(BaseModel):
    """Outcome of looking at the whole back catalogue (issue #63)."""

    scanned: int = 0
    suggested: int = 0
    applied: int = 0
    # Pending suggestions the scan *retracted*, because the rules no longer
    # stand behind them. Reported because a scan that only withdraws would
    # otherwise answer `{"scanned": N, "suggested": 0, "applied": 0}`, which
    # reads as "nothing happened" when something did.
    withdrawn: int = 0


class CommuteRuleProposal(BaseModel):
    """A rule derived from the athlete's own labelled commutes, if there are enough.

    ``rule`` is null when the athlete has fewer than the ten labelled rides the
    clustering needs — below that a "cluster" is a coincidence, and a
    confident-looking rule built from three rides is worse than no proposal.
    """

    rule: Optional[dict] = None
    sample_count: int = 0
    min_samples: int


class CommuteFeedback(BaseModel):
    """What the athlete's own answers say about their rules being wrong.

    Reported, never applied: silently widening a rule every time its output is
    accepted is a feedback loop with no brake, and only the athlete knows
    whether the 9 km ride was the commute or the long way round.
    """

    unmatched_manual_labels: int = 0
    widen: list[dict] = []
    review: list[dict] = []


class RpeQueueResponse(BaseModel):
    """Pending RPE-rating queue for the dashboard/post-upload prompt (issue #28).

    ``items`` are qualifying cycling activities ingested after the athlete's
    ``rpe_head`` cursor that still lack an RPE, oldest-first. ``rpe_head`` is the
    server-side cursor (an activity ``created_at`` ISO timestamp) marking the
    boundary between already-handled and new activities.
    """

    items: list[ActivityResponse] = []
    rpe_head: Optional[str] = None


# Every stream is a 1 Hz series on one shared clock: index i is second i of the
# activity, in every channel, and ``null`` means that channel had no sample at
# that second (a sensor dropout, or a stretch the device did not record at all).
# A consumer must not read a null as a zero — zero watts is coasting, a null is
# the absence of a measurement — and must not assume the streams are dense.
#
# Activities ingested before this convention landed carry dense lists with no
# nulls, whose index is a sample rather than a second. Nothing rewrites an
# activity's streams afterwards, so both shapes are served indefinitely and a
# client has to handle either. See ``openkoutsi.streams``.
StreamMap = dict[str, list[Optional[float]]]


class ActivityStreamsResponse(BaseModel):
    streams: StreamMap = {}


class ActivityDetailResponse(ActivityResponse):
    streams: StreamMap = {}
    power_bests: dict[int, float] = {}
    distance_bests: dict[int, int] = {}
    power_pr_badges: dict[int, dict[str, str]] = {}
    distance_pr_badges: dict[int, dict[str, str]] = {}
    intervals: list[IntervalResponse] = []
    # CP (watts) and W' (joules) the `w_bal` stream was integrated with, frozen
    # at processing time from the athlete's power bests as of this activity's
    # date. Both null — and no `w_bal` in `streams` — when CP couldn't be fit.
    cp_w: Optional[float] = None
    w_prime_j: Optional[float] = None
    # How many duration bests the CP fit used. Low values mean the fit was made
    # against a thin power profile — typically an old ride processed early in a
    # provider backlog import, before the surrounding history existed.
    cp_fit_points: Optional[int] = None
    analysis_status: Optional[str] = None
    analysis: Optional[str] = None
    # Issue #43. The agentic coach's current step while `analysis_status` is
    # "pending" — a code from a fixed vocabulary (`thinking`, `tool.<name>`) the
    # client localises, never a sentence. Null once the prose starts and null
    # for the whole non-agentic path; an unrecognised code means a tool the
    # client predates, so fall back to generic "thinking" copy.
    analysis_progress: Optional[str] = None

    # Issue #41. Scoped to the analysis rather than the whole response on
    # purpose: every other field here is measured or computed from the ride
    # file, and a blanket flag would wrongly mark those as generated too.
    # The prose reaches this column from either the instance's model or, on the
    # BYOK path, the athlete's own via `PATCH /{id}/analysis` — both are model
    # output, so deriving the flag from the column covers both.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def analysis_ai_generated(self) -> bool:
        """True when `analysis` holds text generated by a language model."""
        return bool(self.analysis)

    # `Activity` carries relationships under these names too; the caller passes
    # the already-shaped values, so never read them off the ORM object.
    _ORM_OVERRIDES: ClassVar[frozenset[str]] = frozenset(
        {"streams", "power_bests", "distance_bests", "intervals"}
    )

    @classmethod
    def from_orm_and_streams(
        cls,
        activity,
        streams: dict[str, list],
        power_bests: dict[int, float] | None = None,
        distance_bests: dict[int, int] | None = None,
        intervals: list[IntervalResponse] | None = None,
        power_pr_badges: dict[int, dict[str, str]] | None = None,
        distance_pr_badges: dict[int, dict[str, str]] | None = None,
    ) -> "ActivityDetailResponse":
        return cls.model_validate({
            **cls._orm_values(activity),
            "streams": streams,
            "power_bests": power_bests or {},
            "distance_bests": distance_bests or {},
            "power_pr_badges": power_pr_badges or {},
            "distance_pr_badges": distance_pr_badges or {},
            "intervals": intervals or [],
        })
