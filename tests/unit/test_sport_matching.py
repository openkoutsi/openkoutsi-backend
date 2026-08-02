import pytest
from openkoutsi.sport_matching import is_rest_workout, sports_match


# Generic workout types (threshold, easy, etc.) match endurance sports
@pytest.mark.parametrize("sport,workout,expected", [
    # Cycling sports match generic types
    ("Ride", "threshold", True),
    ("Ride", "easy", True),
    ("Ride", "vo2max", True),
    ("Ride", "recovery", True),
    ("Ride", "endurance", True),
    ("Ride", "long", True),
    ("Ride", "tempo", True),
    ("Ride", "interval", True),
    ("VirtualRide", "threshold", True),
    ("GravelRide", "vo2max", True),
    ("MountainBikeRide", "endurance", True),
    ("EBikeRide", "easy", True),
    # Running sports match generic types
    ("Run", "threshold", True),
    ("Run", "easy", True),
    ("TrailRun", "vo2max", True),
    ("VirtualRun", "long", True),
    # Swimming matches generic types
    ("Swim", "endurance", True),
    ("OpenWaterSwim", "easy", True),
    # Non-endurance sports do NOT match generic types
    ("Walk", "easy", False),
    ("Hike", "endurance", False),
    ("Yoga", "recovery", False),
    ("WeightTraining", "strength", True),  # explicit strength-to-strength match is valid
    # Explicit sport-specific workout types
    ("Swim", "swim", True),
    ("Run", "run", True),
    ("Ride", "ride", True),
    ("Ride", "cycling", True),
    # Cross-sport mismatches
    ("Run", "swim", False),
    ("Swim", "run", False),
    ("Ride", "swim", False),
    # Unknown / None sport
    (None, "threshold", False),
    ("UnknownSport123", "threshold", False),
])
def test_sports_match(sport, workout, expected):
    assert sports_match(sport, workout) == expected


# ``rest`` stays a generic type for grading purposes, so sports_match still
# accepts it — which is exactly why the matcher needs is_rest_workout as a
# separate, explicit gate (issue #40).
def test_rest_is_still_a_generic_type_for_sports_match():
    assert sports_match("Ride", "rest") is True


@pytest.mark.parametrize("workout_type,expected", [
    ("rest", True),
    ("Rest", True),
    ("  REST  ", True),
    ("", True),        # untyped placeholder row
    (None, True),
    ("recovery", False),
    ("easy", False),
    ("endurance", False),
    ("strength", False),
])
def test_is_rest_workout(workout_type, expected):
    assert is_rest_workout(workout_type) is expected
