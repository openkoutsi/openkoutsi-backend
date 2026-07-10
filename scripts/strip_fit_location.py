#!/usr/bin/env python3
"""Strip GPS/location data from a FIT activity file.

Usage:
    uv run python scripts/strip_fit_location.py INPUT.fit OUTPUT.fit

Removes latitude/longitude and GPS-accuracy fields from every message, and
drops any message type that exists only to carry location. All other data
(power, heart rate, cadence, speed, distance, altitude, timestamps, …) is
preserved, so downstream processing and analysis are unaffected.

This is intended for preparing personal ride files before committing them as
test fixtures under ``testdata/fixtures/``. Note that a stripped file still
carries timestamps, heart rate, and power, which can be somewhat identifying.
"""
import sys

from fit_tool.data_message import DataMessage
from fit_tool.fit_file import FitFile
from fit_tool.fit_file_builder import FitFileBuilder

# Exact field names that encode a location. NB: names like
# "avg_power_position" / "max_cadence_position" are NOT location — they are
# cycling posture metrics — so we match an explicit set, not "position".
LOCATION_FIELDS = {
    "position_lat", "position_long", "gps_accuracy",
    "start_position_lat", "start_position_long",
    "end_position_lat", "end_position_long",
    "nec_lat", "nec_long", "swc_lat", "swc_long",  # session bounding box
}
# Whole messages that exist only to carry location.
LOCATION_MESSAGES = {"gps_metadata", "course_point", "segment_point"}


def strip(in_path: str, out_path: str) -> None:
    src = FitFile.from_file(in_path)
    builder = FitFileBuilder(auto_define=True)  # regenerates definitions to match
    kept = removed_fields = dropped_msgs = 0
    for record in src.records:
        msg = record.message
        if not isinstance(msg, DataMessage):
            continue  # skip old definition messages; the builder makes fresh ones
        if msg.name in LOCATION_MESSAGES:
            dropped_msgs += 1
            continue
        for field_id in [f.field_id for f in msg.fields if f.name in LOCATION_FIELDS]:
            msg.remove_field(field_id)
            removed_fields += 1
        builder.add(msg)
        kept += 1
    builder.build().to_file(out_path)
    print(
        f"kept {kept} messages, removed {removed_fields} location fields, "
        f"dropped {dropped_msgs} location-only messages -> {out_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    strip(sys.argv[1], sys.argv[2])
