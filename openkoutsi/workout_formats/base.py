from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ExporterMeta:
    key: str
    label: str
    file_extension: str
    mime_type: str


class AbstractWorkoutExporter(ABC):
    meta: ExporterMeta

    @abstractmethod
    def export(
        self,
        steps: list[dict],
        workout_name: str,
        workout_description: str | None,
        athlete_ftp: int | None,
        athlete_power_zones: list[dict] | None,
    ) -> bytes:
        """Convert internal workout steps to format-specific bytes."""
