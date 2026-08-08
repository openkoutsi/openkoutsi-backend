"""``get_goal_progress`` — what the athlete is training *for* (issue #42).

Load and Form describe a state; a goal is what makes that state good or bad news.
Fitness sliding in October is a problem before a November event and exactly right
after one, and the only way for a model to tell those apart is to know the goal
and its date.

So this tool leads with the arithmetic that distinguishes them: days remaining,
progress against target, and whether the deadline has already gone by. A goal
whose date has passed while still marked active is reported as overdue rather
than quietly listed as active — that is usually a goal the athlete forgot to
close, and saying so is more useful than treating it as live.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from backend.app.mcp.dispatch import ToolRun
from backend.app.mcp.registry import ToolArgs, tool
from backend.app.mcp.shaping import page, progress_pct, round_or_none
from backend.app.models.user_orm import Goal


class GoalProgressArgs(ToolArgs):
    status: Literal["active", "achieved", "abandoned", "all"] = Field(
        "active",
        description=(
            "Which goals to return. 'active' is what the athlete is working "
            "towards now; 'all' includes finished and abandoned ones, which is "
            "history rather than direction."
        ),
    )
    limit: int = Field(10, ge=1, le=25, description="How many goals to return (count).")


class GoalRow(BaseModel):
    goal_id: str = Field(..., description="Identifier of the goal.")
    title: str = Field(..., description="What the athlete called the goal.")
    description: Optional[str] = Field(None, description="Longer description, in the athlete's words.")
    status: str = Field(..., description="'active', 'achieved' or 'abandoned' as stored on the goal.")
    metric: Optional[str] = Field(
        None,
        description=(
            "What the target is measured in, in the athlete's own words, e.g. "
            "'FTP (W)' or 'weekly hours'. Null for a goal with no number."
        ),
    )
    target_value: Optional[float] = Field(
        None, description="The number being aimed at, in the goal's own 'metric' unit."
    )
    current_value: Optional[float] = Field(
        None, description="Latest recorded value, in the same unit as 'target_value'."
    )
    progress_pct: Optional[float] = Field(
        None,
        description=(
            "Current ÷ target as a percentage (%). Null when the goal has no "
            "target number — that means unmeasurable, not zero progress."
        ),
    )
    target_date: Optional[date] = Field(None, description="Calendar date the goal is aimed at.")
    days_remaining: Optional[int] = Field(
        None,
        description=(
            "Days from today until the target date (days). Negative means the "
            "date has passed. Null when the goal has no date."
        ),
    )
    overdue: bool = Field(
        False,
        description=(
            "True when the goal is still active but its target date has passed — "
            "usually a goal the athlete never closed, worth asking about."
        ),
    )
    outcome_note: Optional[str] = Field(
        None, description="What the athlete recorded on finishing the goal, when they finished it."
    )
    guidance_verdict: Optional[str] = Field(
        None,
        description=(
            "A previous AI judgement of how realistic the goal is: realistic, "
            "ambitious or unrealistic. Null when never assessed. It is a prior "
            "opinion, not evidence — re-judge from the data."
        ),
    )


class GoalProgress(BaseModel):
    items: list[GoalRow] = Field(default_factory=list, description="Matching goals, soonest deadline first.")
    returned: int = Field(0, description="How many goals are in this response (count).")
    total: int = Field(0, description="How many matched in total (count).")
    truncated: bool = Field(False, description="True when results were cut off by 'limit'.")


@tool(
    name="get_goal_progress",
    title="Goal progress",
    scopes={"goals:read"},
    arguments=GoalProgressArgs,
    returns=GoalProgress,
)
async def get_goal_progress(run: ToolRun, args: GoalProgressArgs) -> GoalProgress:
    """The athlete's goals with their targets, deadlines, current values and
    progress: what they are training towards, how far along they are, and how
    long is left.

    Call this before judging whether a training state is good. Falling Fitness
    six weeks out from an event and falling Fitness the week after one are the
    same number and opposite news, and only the goal tells you which you are
    looking at.

    Goals are returned soonest-deadline-first, with dated goals ahead of undated
    ones. A goal still marked active past its target date is flagged 'overdue'.
    """
    query = select(Goal).where(Goal.athlete_id == run.athlete.id)
    if args.status != "all":
        query = query.where(Goal.status == args.status)

    total = (
        await run.session.execute(select(func.count()).select_from(query.subquery()))
    ).scalar_one()

    goals = (
        await run.session.execute(
            # Undated goals sort last: a deadline is what makes a goal urgent, and
            # a goal without one should never displace a goal with one.
            query.order_by(
                Goal.target_date.is_(None), Goal.target_date, Goal.created_at.desc()
            ).limit(args.limit)
        )
    ).scalars().all()

    today = run.today
    rows = []
    for goal in goals:
        days_remaining = (goal.target_date - today).days if goal.target_date else None
        rows.append(
            GoalRow(
                goal_id=goal.id,
                title=goal.title,
                description=goal.description,
                status=goal.status,
                metric=goal.metric,
                target_value=round_or_none(goal.target_value, 2),
                current_value=round_or_none(goal.current_value, 2),
                progress_pct=progress_pct(goal.current_value, goal.target_value),
                target_date=goal.target_date,
                days_remaining=days_remaining,
                overdue=goal.status == "active"
                and days_remaining is not None
                and days_remaining < 0,
                outcome_note=goal.outcome_note,
                guidance_verdict=goal.guidance_verdict,
            )
        )

    return GoalProgress(**page(rows, int(total)))
