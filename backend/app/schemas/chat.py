from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from backend.app.schemas.plan_proposals import PlanProposalSummary
from backend.app.schemas.plans import TrainingPlanResponse


class ChatProposal(BaseModel):
    """A training-plan change Koutsi has offered on this turn (issue #72).

    Present on the assistant turn that drafted it and nowhere else. The card the
    web app renders from this **is** the prompt: the athlete answers with a
    button, never by typing "yes", because a typed yes would put the decision
    back inside the thing being gated.

    Nothing here has been applied. ``status`` says where the offer stands, and
    only ``pending`` has anything left to answer.
    """

    id: str
    #: ``create_plan`` | ``update_plan`` | ``update_workout``.
    kind: str
    #: ``pending`` | ``applied`` | ``declined`` | ``expired`` | ``superseded``.
    status: str
    #: ``llm`` or ``rule_based`` — whether a model wrote the weeks or the
    #: deterministic builder did. Surfaced rather than hidden: a fallback is a
    #: different thing to be offered, not a worse version of the same thing.
    built_by: Optional[str] = None
    summary: PlanProposalSummary
    created_at: datetime
    expires_at: Optional[datetime] = None
    decided_at: Optional[datetime] = None
    #: The plan an approval created, once there is one — where to send the
    #: athlete next.
    applied_plan_id: Optional[str] = None

    model_config = {"from_attributes": True}


class ChatProposalDecision(BaseModel):
    """What came of the athlete answering an offer (issue #72).

    The proposal in its settled state, and — on an approval — the plan that now
    exists, in exactly the shape the plan page serves. A decision deliberately
    spends no chat turn and asks no model anything: the outcome the athlete
    needs is *your plan is live, here it is*, which is a fact the backend knows
    exactly.
    """

    proposal: ChatProposal
    #: The plan the approval created or changed, in the same shape
    #: ``GET /api/plans/{id}`` serves. Null on a decline.
    plan: Optional[TrainingPlanResponse] = None


class ChatMessageResponse(BaseModel):
    id: str
    #: ``user`` or ``assistant``.
    role: str
    #: Grows while an answer streams; empty on a queued turn.
    content: str
    #: ``queued`` / ``pending`` / ``complete`` / ``error`` on assistant turns,
    #: null on the athlete's own.
    status: Optional[str] = None
    #: Issue #43's progress code while the turn is still gathering.
    progress: Optional[str] = None
    #: Why a turn failed, as a key the web app localises — see
    #: ``services.llm_agent``'s ``CODE_*`` constants.
    error_code: Optional[str] = None
    #: Registry tool names this turn consulted, in call order, for the steps the
    #: thread shows ahead of the answer. Present and growing while the turn is
    #: still gathering. Never the arguments and never the results.
    tool_names: Optional[list[str]] = None
    #: The plan Koutsi offered on this turn, if it offered one (issue #72).
    #: Null on every other turn, which is almost all of them.
    proposal: Optional[ChatProposal] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatConversationSummary(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatConversationDetail(ChatConversationSummary):
    messages: list[ChatMessageResponse] = []


class ChatTurnBody(BaseModel):
    """One question from the athlete.

    ``message`` is the *only* thing the client sends. Everything the model
    actually receives — the system prompt, the scope policy, the replayed
    history — is built server-side in ``services.llm_chat``, which is the whole
    difference between this and the ``/api/llm/chat`` proxy issue #45 removed.
    """

    message: str = Field(min_length=1)
    #: Overrides the athlete's stored locale for this turn, as the training
    #: status trigger does — the browser knows which language the page is in.
    locale: Optional[str] = None


class ChatRetryBody(BaseModel):
    """Nothing but the locale — a retry re-runs a question already stored.

    Deliberately not :class:`ChatTurnBody`: taking a message here would invite a
    client to change the question while claiming to retry it, which is the very
    thing running the turn in place exists to prevent.
    """

    locale: Optional[str] = None


class ChatConversationCreate(BaseModel):
    #: Optional opening question. When given, the conversation is created and
    #: the first turn started in one round trip.
    message: Optional[str] = None
    locale: Optional[str] = None


class ChatAvailability(BaseModel):
    """What the chat surface may do right now, so the UI can say why not.

    Chat is the one LLM surface with no single-shot prompt to fall back on, so
    the reasons it might be unusable have to be answerable *before* the athlete
    types rather than discovered as a failed turn.
    """

    #: Has the athlete opted into the agentic coach? Chat is meaningless without
    #: tools, so it rides the same switch (``app_settings.agentic_koutsi``).
    enabled: bool
    #: Can the resolved model call tools at all? False means the nav entry is
    #: disabled: a settled property of the provider, not a transient failure, so
    #: inviting a retry would be a lie.
    tools_supported: bool
    #: False when the instance gate (issue #9) denies this user.
    entitled: bool
    turns_remaining_today: int
    # Filled by the handler, not defaulted from `settings` here: a default is
    # evaluated once at import and baked into the class, so it would report a
    # snapshot of the config rather than what the API is currently enforcing.
    # The web app gates its composer on both, so a divergence means the UI and
    # the API disagree about the limit.
    max_turns_per_conversation: int
    max_message_chars: int
