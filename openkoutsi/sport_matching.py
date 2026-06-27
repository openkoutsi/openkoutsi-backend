"""Sport type matching for linking activities to planned workouts."""

from typing import Optional

# Maps activity sport_type values to canonical sport categories
_ACTIVITY_SPORT_TO_CATEGORY: dict[str, str] = {
    "Ride": "cycling",
    "VirtualRide": "cycling",
    "GravelRide": "cycling",
    "MountainBikeRide": "cycling",
    "EBikeRide": "cycling",
    "EBikeSport": "cycling",
    "Handcycle": "cycling",
    "Run": "running",
    "TrailRun": "running",
    "VirtualRun": "running",
    "Swim": "swimming",
    "OpenWaterSwim": "swimming",
    "Walk": "walking",
    "Hike": "hiking",
    "Yoga": "yoga",
    "WeightTraining": "strength",
    "Workout": "strength",
    "Crossfit": "strength",
    "Rowing": "rowing",
    "Kayaking": "paddling",
    "Canoeing": "paddling",
    "Skiing": "skiing",
    "NordicSki": "skiing",
    "AlpineSki": "skiing",
    "Snowboard": "skiing",
    "Soccer": "team_sport",
    "Tennis": "racket_sport",
    "Badminton": "racket_sport",
}

# Sport types that count as cycling, derived from the category map above.
CYCLING_SPORT_TYPES = frozenset(
    sport for sport, category in _ACTIVITY_SPORT_TO_CATEGORY.items() if category == "cycling"
)

# Workout types that are sport-agnostic (apply to whatever sport the plan is for)
_GENERIC_WORKOUT_TYPES = {
    "easy",
    "recovery",
    "endurance",
    "tempo",
    "threshold",
    "vo2max",
    "interval",
    "long",
    "race",
    "rest",
}

# Workout types that imply a specific sport category
_WORKOUT_TYPE_TO_CATEGORY: dict[str, str] = {
    "swim": "swimming",
    "run": "running",
    "ride": "cycling",
    "bike": "cycling",
    "cycling": "cycling",
    "running": "running",
    "swimming": "swimming",
    "strength": "strength",
    "yoga": "yoga",
    "cross-training": None,  # matches any sport
}


def _activity_category(sport_type: Optional[str]) -> Optional[str]:
    if not sport_type:
        return None
    return _ACTIVITY_SPORT_TO_CATEGORY.get(sport_type)


def _workout_category(workout_type: Optional[str]) -> Optional[str]:
    """Return the sport category implied by a workout type, or None if generic/unknown."""
    if not workout_type:
        return None
    lower = workout_type.lower()
    if lower in _GENERIC_WORKOUT_TYPES:
        return None  # generic — matches any sport
    return _WORKOUT_TYPE_TO_CATEGORY.get(lower, "unknown")


def sports_match(activity_sport: Optional[str], workout_type: Optional[str]) -> bool:
    """Return True if the activity sport loosely matches the planned workout type.

    Generic workout types (easy, threshold, vo2max, etc.) match any endurance sport
    (cycling, running, swimming).  Sport-specific workout types only match their sport.
    Non-endurance activities (yoga, walking, strength) never match generic types.
    """
    act_cat = _activity_category(activity_sport)
    wo_cat = _workout_category(workout_type)

    if act_cat is None:
        # Unknown activity sport — don't match
        return False

    if wo_cat is None:
        # Generic workout type — matches cycling, running, swimming only
        return act_cat in {"cycling", "running", "swimming", "rowing"}

    if wo_cat == "unknown":
        # Unrecognised workout type — don't match
        return False

    return act_cat == wo_cat
