from openkoutsi.workout_formats.base import AbstractWorkoutExporter
from openkoutsi.workout_formats.zwift import ZwiftExporter
from openkoutsi.workout_formats.fit_workout import FitWorkoutExporter
from openkoutsi.workout_formats.json_export import JsonExporter

EXPORTERS: dict[str, type[AbstractWorkoutExporter]] = {
    "zwift": ZwiftExporter,
    "fit_workout": FitWorkoutExporter,
    "json": JsonExporter,
}
