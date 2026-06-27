from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class TimeDuration(BaseModel):
    type: Literal["time"]
    seconds: int = Field(gt=0)


class DistanceDuration(BaseModel):
    type: Literal["distance"]
    meters: int = Field(gt=0)


class OpenDuration(BaseModel):
    type: Literal["open"]


Duration = Annotated[
    Union[TimeDuration, DistanceDuration, OpenDuration],
    Field(discriminator="type"),
]


class ZoneSpec(BaseModel):
    type: Literal["zone"]
    zone_number: int = Field(ge=1)


class PctFtpSpec(BaseModel):
    type: Literal["pct_ftp"]
    pct: float = Field(gt=0)


class AbsoluteSpec(BaseModel):
    type: Literal["absolute"]
    value: float = Field(gt=0)


class RangeSpec(BaseModel):
    type: Literal["range"]
    low: float
    high: float


TargetSpec = Annotated[
    Union[ZoneSpec, PctFtpSpec, AbsoluteSpec, RangeSpec],
    Field(discriminator="type"),
]


class WorkoutTarget(BaseModel):
    metric: Literal["power", "hr", "cadence", "pace"]
    spec: TargetSpec


class WorkoutStep(BaseModel):
    kind: Literal["step"]
    step_type: Literal["warmup", "active", "recovery", "cooldown", "rest", "other"]
    duration: Duration
    target: Optional[WorkoutTarget] = None
    notes: Optional[str] = None


class RepeatBlock(BaseModel):
    kind: Literal["repeat"]
    repeat_count: int = Field(ge=2)
    steps: list[WorkoutStepOrRepeat] = Field(min_length=1)

    def max_depth(self) -> int:
        depths = []
        for s in self.steps:
            if isinstance(s, RepeatBlock):
                depths.append(1 + s.max_depth())
            else:
                depths.append(0)
        return max(depths) if depths else 0


WorkoutStepOrRepeat = Annotated[
    Union[WorkoutStep, RepeatBlock],
    Field(discriminator="kind"),
]

RepeatBlock.model_rebuild()
