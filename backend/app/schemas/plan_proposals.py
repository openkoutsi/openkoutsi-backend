"""The preview an athlete says yes or no to (issue #72).

One shape, three readers, which is why it lives here rather than beside any of
them:

* the **model** reads it back as a tool result and describes it in prose;
* the **card** in the chat thread renders it under the turn that offered it;
* the **apply path** compares part of it against the database again, because a
  preview drafted yesterday may be describing a world that has moved.

Every field is described with its unit, because the first of those readers is a
language model and ``tests/unit/test_mcp_registry.py`` holds the whole tool
layer to it.

The one field that is not decoration is :attr:`PlanProposalSummary.archives`.
Creating a plan archives every active plan whose dates overlap it, and a yes/no
that does not say so is uninformed consent — so the plans an approval would file
away are named here, by name and date range, and the apply path refuses if that
set has changed since the athlete was shown it.
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field

#: What a proposal would do. ``update_workout`` covers skipping and unskipping
#: too: both are edits to one planned session.
KIND_CREATE_PLAN = "create_plan"
KIND_UPDATE_PLAN = "update_plan"
KIND_UPDATE_WORKOUT = "update_workout"

#: Where a proposal is in its life. Only ``pending`` can be applied.
STATUS_PENDING = "pending"
STATUS_APPLIED = "applied"
STATUS_DECLINED = "declined"
STATUS_EXPIRED = "expired"
STATUS_SUPERSEDED = "superseded"

#: Whether the draft's weeks came from a model or from the deterministic
#: builder. Recorded rather than inferred, so a fallback is visible.
BUILT_BY_LLM = "llm"
BUILT_BY_RULES = "rule_based"


class ArchivedPlanPreview(BaseModel):
    """A plan an approval would file away."""

    plan_id: str = Field(..., description="Identifier of the plan that would be archived.")
    name: str = Field(..., description="Plan name as the athlete sees it.")
    start_date: Optional[date] = Field(None, description="First calendar date of that plan.")
    end_date: Optional[date] = Field(None, description="Last calendar date of that plan.")


class ProposedWeek(BaseModel):
    """One week of a drafted plan, summarised."""

    week_number: int = Field(..., description="Which week of the plan this is, 1-based (count).")
    week_type: Optional[str] = Field(
        None,
        description=(
            "'build', 'recovery' or 'taper', where the draft says. A machine key "
            "rather than a sentence: the card localises it, and the preview has "
            "to read the same in Finnish as in English."
        ),
    )
    sessions: int = Field(
        0, description="Training sessions this week, rest days excluded (count)."
    )
    total_load: Optional[int] = Field(
        None, description="Prescribed Load for the whole week (unitless Load points)."
    )
    total_duration_min: Optional[int] = Field(
        None, description="Prescribed riding time for the whole week, in minutes (min)."
    )


class ProposedSession(BaseModel):
    """One day of a drafted plan."""

    day_of_week: int = Field(..., description="1 = Monday through 7 = Sunday.")
    workout_type: Optional[str] = Field(
        None, description="Session type, e.g. endurance, threshold, rest."
    )
    description: Optional[str] = Field(
        None, description="What the session asks for, in the draft's own words."
    )
    duration_min: Optional[int] = Field(
        None, description="Prescribed duration in minutes (min). Null on a rest day."
    )
    target_load: Optional[int] = Field(
        None, description="Prescribed Load (unitless Load points). Null on a rest day."
    )


class ProposedChange(BaseModel):
    """One field a change would move, and what it would move from and to."""

    field: str = Field(..., description="Which field would change, e.g. start_date, duration_min.")
    before: Optional[str] = Field(None, description="What it is now, rendered as text.")
    after: Optional[str] = Field(None, description="What it would become, rendered as text.")


class PlanProposalSummary(BaseModel):
    """Everything the athlete needs in order to answer yes or no."""

    kind: str = Field(
        ...,
        description=(
            "'create_plan' for a new plan, 'update_plan' for whole-plan fields, "
            "'update_workout' for one planned session (including skip and unskip)."
        ),
    )
    built_by: str = Field(
        ...,
        description=(
            "'llm' when a model wrote the weeks, 'rule_based' when the "
            "deterministic builder did — which happens when the model call "
            "failed or could not be parsed."
        ),
    )
    fallback_reason: Optional[str] = Field(
        None,
        description=(
            "Why the deterministic builder was used, when it was. Null on the "
            "ordinary path."
        ),
    )

    plan_name: Optional[str] = Field(None, description="Name of the plan being created or changed.")
    goal: Optional[str] = Field(None, description="What the plan is built towards, in words.")
    start_date: Optional[date] = Field(None, description="First calendar date the plan would run.")
    end_date: Optional[date] = Field(None, description="Last calendar date the plan would run.")
    weeks: Optional[int] = Field(None, description="Planned length in weeks (count).")

    weekly: list[ProposedWeek] = Field(
        default_factory=list, description="Week-by-week summary of a drafted plan."
    )
    first_week: list[ProposedSession] = Field(
        default_factory=list,
        description="The seven days of week 1, so the shape of the plan is readable.",
    )
    remaining_weeks: int = Field(
        0,
        description=(
            "How many further weeks the draft holds beyond the one listed day by "
            "day (count). Their totals are in 'weekly'."
        ),
    )

    changes: list[ProposedChange] = Field(
        default_factory=list, description="The fields a change would move. Empty for a new plan."
    )
    target_plan_id: Optional[str] = Field(
        None, description="Identifier of the plan a change applies to."
    )
    target_workout_id: Optional[str] = Field(
        None, description="Identifier of the planned session a change applies to."
    )
    target_date: Optional[date] = Field(
        None, description="Calendar date the targeted session falls on."
    )
    target_label: Optional[str] = Field(
        None, description="How to name the targeted session in a sentence, e.g. 'Thursday's threshold'."
    )
    reopens_plan: bool = Field(
        False,
        description=(
            "True when approving would reactivate a finished or filed-away plan, "
            "which is itself something the athlete should be told."
        ),
    )

    archives: list[ArchivedPlanPreview] = Field(
        default_factory=list,
        description=(
            "Plans an approval would archive, because their dates overlap. "
            "Creating a plan files away every active plan it overlaps — say so "
            "in your reply; a yes that did not know this is not consent. "
            "Archiving is reversible from the plan page."
        ),
    )


class PlanProposalResult(BaseModel):
    """What a propose tool hands back to the model."""

    proposal_id: str = Field(..., description="Identifier of the drafted proposal.")
    status: str = Field(
        ...,
        description="Always 'pending' from a propose call: nothing has been applied.",
    )
    expires_at: datetime = Field(
        ..., description="When the offer lapses if the athlete has not answered."
    )
    summary: PlanProposalSummary = Field(
        ..., description="The preview the athlete is being shown."
    )
    note: str = Field(
        ...,
        description=(
            "A reminder of what just happened, for the reply you are about to "
            "write. Read it."
        ),
    )
