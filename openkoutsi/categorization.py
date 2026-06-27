from enum import Enum


class WorkoutCategory(str, Enum):
    recovery = "recovery"
    endurance = "endurance"
    tempo = "tempo"
    threshold = "threshold"
    vo2max = "vo2max"
    anaerobic = "anaerobic"
    sprint = "sprint"
    strength = "strength"
    yoga = "yoga"
    cross_training = "cross_training"


# Categories that can be auto-assigned from power data (Coggan zone model)
AUTO_CATEGORIES = {
    WorkoutCategory.recovery,
    WorkoutCategory.endurance,
    WorkoutCategory.tempo,
    WorkoutCategory.threshold,
    WorkoutCategory.vo2max,
    WorkoutCategory.anaerobic,
    WorkoutCategory.sprint,
}


def classify_workout(
    intensity_factor: float | None,
    variability_index: float | None,
) -> WorkoutCategory | None:
    """
    Classify a workout using Coggan's 7-zone power model.

    intensity_factor: NP / FTP
    variability_index: NP / avg_power — high VI (>1.10) indicates interval/punchy riding
    """
    if intensity_factor is None:
        return None

    vi = variability_index or 1.0

    if intensity_factor >= 1.20:
        return WorkoutCategory.sprint

    if intensity_factor >= 1.10:
        return WorkoutCategory.anaerobic

    if intensity_factor >= 1.00:
        return WorkoutCategory.vo2max

    if intensity_factor >= 0.90:
        if vi > 1.10:
            return WorkoutCategory.vo2max
        return WorkoutCategory.threshold

    if intensity_factor >= 0.78:
        if vi > 1.10:
            return WorkoutCategory.threshold
        return WorkoutCategory.tempo

    if intensity_factor >= 0.65:
        return WorkoutCategory.endurance

    return WorkoutCategory.recovery
