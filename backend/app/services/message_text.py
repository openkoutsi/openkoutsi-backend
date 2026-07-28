"""Human-readable copy for inbox messages.

Inbox messages carry their own ``title`` and ``body``, rendered here at write
time and stored on the row. That is deliberately different from how the
achievements *page* works: the catalogue served by ``GET /api/achievements``
stays free of display text (see ``openkoutsi.achievements``), and the web app
localises it from the achievement id. Messages cannot work that way — a message
type that says something new would need a matching i18n template shipped in the
web app before it could say anything at all, which is why the achievement
notification could only ever interpolate a count.

So this module is the one place inbox copy lives. Adding a message type means
adding a branch here and nothing else.

Rendering is English-only for now. ``render`` still takes a ``locale`` and the
caller records it on the message row, so a per-locale table can be slotted in
later without a migration and without touching any call site.
"""
from dataclasses import dataclass
from typing import Optional

from openkoutsi.achievements import CATALOGUE_BY_ID

DEFAULT_LOCALE = "en"


@dataclass(frozen=True)
class RenderedMessage:
    title: str
    body: str


# ── Achievement copy ─────────────────────────────────────────────────────────
# Kept in step with `app.achievements.items.<id>.name` in the web app's English
# catalogue, so a badge is called the same thing in the inbox and on the
# achievements page.
ACHIEVEMENT_NAMES: dict[str, str] = {
    # Volume
    "activity_count": "Getting started",
    "long_activity": "Long hauler",
    "total_hours": "Time in the saddle",
    "single_ride_distance": "Distance rider",
    "total_distance": "The long way round",
    "total_load": "Load bearer",
    # Climbing
    "single_ride_elevation": "Climber",
    "total_elevation": "Mountain goat",
    "everesting": "Everesting",
    # Variety & character
    "multisport": "All-rounder",
    "race_day": "Race day",
    "commuter": "Commuter",
    "comeback": "Welcome back",
    # Engagement
    "rpe_recorded": "In tune",
    "notes_written": "Training diary",
    # Plans & goals
    "plans_completed": "Plan finisher",
    "plan_flawless": "Perfect plan",
    "plan_adherence": "On the programme",
    "goals_reached": "Goal getter",
    # Streaks
    "streak_active_weeks": "Consistency",
    "streak_volume_weeks": "Steady volume",
    "streak_multisport_weeks": "Mixing it up",
    "streak_distance_weeks": "Rolling on",
    "streak_climbing_weeks": "Always climbing",
    "streak_active_months": "In it for the long run",
}

# What a `count` tier is counting, as (singular, plural). Every other unit names
# itself (`hours` → h, `km`, `metres` → m, `percent`, `weeks`, `months`,
# `load`), but "10" on its own says nothing — 10 activities, 10 races and 10
# notes are all `count` tiers. The singular is not decoration: the first tier of
# `activity_count`, `race_day`, `plans_completed` and `goals_reached` is 1, and
# the first of those is the very first badge every athlete earns.
_COUNT_NOUN: dict[str, tuple[str, str]] = {
    "activity_count": ("activity", "activities"),
    "multisport": ("sport", "sports"),
    "race_day": ("race", "races"),
    "commuter": ("commute", "commutes"),
    "rpe_recorded": ("ride rated", "rides rated"),
    "notes_written": ("note", "notes"),
    "plans_completed": ("plan", "plans"),
    "goals_reached": ("goal", "goals"),
}


def _number(value: float) -> str:
    """Render a tier without trailing noise — 8848, not 8848.0."""
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _tier_phrase(achievement_id: str, tier: float) -> Optional[str]:
    """How far the athlete got, or None when the tier adds nothing.

    Single-tier badges are fully described by their name: "Everesting — 8848 m"
    says the same thing twice.
    """
    definition = CATALOGUE_BY_ID.get(achievement_id)
    if definition is None or len(definition.tiers) <= 1:
        return None

    n = _number(tier)
    unit = definition.unit
    if unit == "count":
        forms = _COUNT_NOUN.get(achievement_id)
        if forms is None:
            return n
        return f"{n} {forms[0] if tier == 1 else forms[1]}"
    if unit == "hours":
        return f"{n} h"
    if unit == "metres":
        return f"{n} m"
    if unit == "percent":
        return f"{n}%"
    if unit == "load":
        return f"{n} Load"
    # km, weeks, months — the unit is already the word we want.
    return f"{n} {unit}"


def achievement_label(achievement_id: str, tier: float) -> str:
    """One badge as a line of prose, e.g. "Climber — 1000 m"."""
    name = ACHIEVEMENT_NAMES.get(achievement_id, achievement_id)
    phrase = _tier_phrase(achievement_id, tier)
    return f"{name} — {phrase}" if phrase else name


def _render_achievement_unlocked(data: dict) -> RenderedMessage:
    earned = [
        (entry.get("id"), entry.get("tier"))
        for entry in data.get("achievements") or []
        if isinstance(entry, dict) and entry.get("id") is not None
    ]
    labels = [achievement_label(str(a_id), float(tier or 0)) for a_id, tier in earned]

    if not labels:
        # Defensive: a payload without a list still deserves a sensible message.
        count = int(data.get("count") or 0)
        plural = "achievement" if count == 1 else "achievements"
        return RenderedMessage(
            title="Achievement unlocked" if count == 1 else f"{count} {plural} unlocked",
            body=f"You earned {count} new {plural}.",
        )

    if len(labels) == 1:
        return RenderedMessage(
            title="Achievement unlocked",
            body=f"You earned {labels[0]}.",
        )

    listed = "\n".join(f"• {label}" for label in labels)
    return RenderedMessage(
        title=f"{len(labels)} achievements unlocked",
        body=f"You earned:\n{listed}",
    )


def _render_invite_used(data: dict) -> RenderedMessage:
    username = str(data.get("username") or "Someone")
    display_name = data.get("display_name")
    who = f"{display_name} ({username})" if display_name else username
    return RenderedMessage(
        title="Invite used",
        body=f"{who} joined via an invite link.",
    )


def render(type: str, data: dict, locale: str = DEFAULT_LOCALE) -> RenderedMessage:
    """Render an inbox message's title and body.

    Unknown types get a generic notice rather than an exception or an empty
    card: a message that reached a mailbox should always be readable, even if
    this module has not caught up with the type that produced it.
    """
    data = data or {}
    if type == "achievement_unlocked":
        return _render_achievement_unlocked(data)
    if type == "invite_used":
        return _render_invite_used(data)
    return RenderedMessage(
        title="Notification",
        body="You have a new notification.",
    )
