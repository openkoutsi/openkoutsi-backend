"""Export workout definitions to Zwift .zwo XML format."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from openkoutsi.workout_formats.base import AbstractWorkoutExporter, ExporterMeta


def _zone_midpoint_pct(zone_number: int, power_zones: list[dict] | None, ftp: int) -> float:
    """Return the midpoint of a power zone as a fraction of FTP."""
    if power_zones and 1 <= zone_number <= len(power_zones):
        z = power_zones[zone_number - 1]
        low = z.get("low", 0)
        high = z.get("high", low)
        mid = (low + high) / 2.0
        return mid / ftp
    fallback = {1: 0.55, 2: 0.65, 3: 0.80, 4: 0.92, 5: 1.05, 6: 1.20, 7: 1.50}
    return fallback.get(zone_number, 0.75)


def _spec_to_pct(spec: dict, ftp: int, power_zones: list[dict] | None) -> tuple[float, float]:
    """Return (low_pct, high_pct) for a power spec, both as FTP fractions."""
    t = spec.get("type")
    if t == "pct_ftp":
        p = spec["pct"] / 100.0
        return p, p
    if t == "absolute":
        p = spec["value"] / ftp
        return p, p
    if t == "range":
        return spec["low"] / ftp, spec["high"] / ftp
    if t == "zone":
        p = _zone_midpoint_pct(spec["zone_number"], power_zones, ftp)
        return p, p
    return 0.5, 0.5


def _step_to_element(step: dict, ftp: int, power_zones: list[dict] | None) -> ET.Element | None:
    """Convert a single WorkoutStep dict to a Zwift XML element, or None if skippable."""
    dur = step.get("duration", {})
    if dur.get("type") == "time":
        duration_s = dur["seconds"]
    else:
        duration_s = 60

    target = step.get("target")
    has_power = target and target.get("metric") == "power"
    step_type = step.get("step_type", "active")

    if step_type == "warmup":
        el = ET.Element("Warmup", Duration=str(duration_s))
        if has_power and ftp:
            lo, hi = _spec_to_pct(target["spec"], ftp, power_zones)
            el.set("PowerLow", f"{lo:.3f}")
            el.set("PowerHigh", f"{hi:.3f}")
        else:
            el.set("PowerLow", "0.500")
            el.set("PowerHigh", "0.750")
        return el

    if step_type == "cooldown":
        el = ET.Element("Cooldown", Duration=str(duration_s))
        if has_power and ftp:
            lo, hi = _spec_to_pct(target["spec"], ftp, power_zones)
            el.set("PowerLow", f"{lo:.3f}")
            el.set("PowerHigh", f"{hi:.3f}")
        else:
            el.set("PowerLow", "0.750")
            el.set("PowerHigh", "0.400")
        return el

    if step_type in ("rest", "recovery") and not has_power:
        return ET.Element("FreeRide", Duration=str(duration_s), FlatRoad="1")

    if has_power and ftp:
        lo, hi = _spec_to_pct(target["spec"], ftp, power_zones)
        if abs(hi - lo) < 0.001:
            return ET.Element("SteadyState", Duration=str(duration_s), Power=f"{lo:.3f}")
        else:
            el = ET.Element("Ramp", Duration=str(duration_s))
            el.set("PowerLow", f"{lo:.3f}")
            el.set("PowerHigh", f"{hi:.3f}")
            return el

    return ET.Element("FreeRide", Duration=str(duration_s), FlatRoad="1")


def _repeat_to_elements(block: dict, ftp: int, power_zones: list[dict] | None) -> list[ET.Element]:
    count = block.get("repeat_count", 1)
    children = block.get("steps", [])

    if len(children) == 2:
        work, rest = children[0], children[1]
        if (
            work.get("kind") == "step" and rest.get("kind") == "step"
            and work.get("step_type") in ("active", "other")
            and rest.get("step_type") in ("recovery", "rest")
        ):
            work_dur = work.get("duration", {})
            rest_dur = rest.get("duration", {})
            on_s = work_dur.get("seconds", 60) if work_dur.get("type") == "time" else 60
            off_s = rest_dur.get("seconds", 60) if rest_dur.get("type") == "time" else 60
            el = ET.Element("IntervalsT", Repeat=str(count), OnDuration=str(on_s), OffDuration=str(off_s))
            work_target = work.get("target")
            rest_target = rest.get("target")
            if work_target and work_target.get("metric") == "power" and ftp:
                lo, hi = _spec_to_pct(work_target["spec"], ftp, power_zones)
                el.set("OnPower", f"{lo:.3f}")
            else:
                el.set("OnPower", "1.000")
            if rest_target and rest_target.get("metric") == "power" and ftp:
                lo, hi = _spec_to_pct(rest_target["spec"], ftp, power_zones)
                el.set("OffPower", f"{lo:.3f}")
            else:
                el.set("OffPower", "0.500")
            return [el]

    elements = []
    for _ in range(count):
        for child in children:
            elements.extend(_steps_to_elements([child], ftp, power_zones))
    return elements


def _steps_to_elements(steps: list[dict], ftp: int, power_zones: list[dict] | None) -> list[ET.Element]:
    elements = []
    for step in steps:
        if step.get("kind") == "repeat":
            elements.extend(_repeat_to_elements(step, ftp, power_zones))
        elif step.get("kind") == "step":
            el = _step_to_element(step, ftp, power_zones)
            if el is not None:
                elements.append(el)
    return elements


class ZwiftExporter(AbstractWorkoutExporter):
    meta = ExporterMeta(
        key="zwift",
        label="Zwift (.zwo)",
        file_extension="zwo",
        mime_type="application/xml",
    )

    def export(
        self,
        steps: list[dict],
        workout_name: str,
        workout_description: str | None,
        athlete_ftp: int | None,
        athlete_power_zones: list[dict] | None,
    ) -> bytes:
        root = ET.Element("workout_file")

        name_el = ET.SubElement(root, "name")
        name_el.text = workout_name

        if workout_description:
            desc_el = ET.SubElement(root, "description")
            desc_el.text = workout_description

        ET.SubElement(root, "sportType").text = "bike"

        workout_el = ET.SubElement(root, "workout")

        ftp = athlete_ftp or 0
        elements = _steps_to_elements(steps, ftp, athlete_power_zones)
        for el in elements:
            workout_el.append(el)

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=True).encode("utf-8")
