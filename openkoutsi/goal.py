from dataclasses import dataclass
from datetime import date as d
import json


@dataclass
class Goal:
    goalType: str
    description: str
    date: d
    target: str

    @classmethod
    def from_json(cls, value: str | dict) -> "Goal":
        data = json.loads(value) if isinstance(value, str) else value
        return cls(
            goalType=data["goalType"],
            description=data["description"],
            date=d.fromisoformat(data["date"]),
            target=data["target"],
        )

    @classmethod
    def from_string(cls, value: str) -> "Goal":
        # Backward-compatible alias; prefer from_json.
        return cls.from_json(value)

    def to_json(self) -> str:
        return json.dumps(
            {
                "goalType": self.goalType,
                "description": self.description,
                "date": self.date.isoformat(),
                "target": self.target,
            }
        )
