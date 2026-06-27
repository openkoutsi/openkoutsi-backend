#!/usr/bin/env python3
"""CLI tool to inspect an exported FIT workout file without a physical device.

Usage:
    python scripts/inspect_fit_workout.py path/to/workout.fit
"""
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <workout.fit>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    from openkoutsi.workout_formats.fit_debug import describe_fit_workout
    print(describe_fit_workout(path.read_bytes()))


if __name__ == "__main__":
    main()
