"""Export workout definitions to JSON format."""

from __future__ import annotations

import json

from openkoutsi.workout_formats.base import AbstractWorkoutExporter, ExporterMeta


class JsonExporter(AbstractWorkoutExporter):
    meta = ExporterMeta(
        key="json",
        label="JSON (.json)",
        file_extension="json",
        mime_type="application/json",
    )

    def export(
        self,
        steps: list[dict],
        workout_name: str,
        workout_description: str | None,
        athlete_ftp: int | None,
        athlete_power_zones: list[dict] | None,
    ) -> bytes:
        payload = {
            "name": workout_name,
            "description": workout_description,
            "steps": steps,
        }
        return json.dumps(payload, indent=2).encode("utf-8")
