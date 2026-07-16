"""Guard against reintroducing TrainingPeaks/Peaksware trademarked metric names.

NP, TSS, IF, CTL, ATL and TSB (and their long forms) were replaced with the
generic weighted power / load / intensity / fitness / fatigue / form vocabulary.
This test fails if any of the trademarked identifiers reappear on the public API
surface (schema field names) so the regression is caught before it ships.
"""
import re

from backend.main import create_app

# Field-name tokens that must never appear in the OpenAPI schema again.
FORBIDDEN_FIELD_NAMES = {
    "tss",
    "tss_day",
    "daily_tss",
    "target_tss",
    "estimated_tss",
    "ctl",
    "atl",
    "tsb",
    "normalized_power",
    "intensity_factor",
}

_TRADEMARK_PHRASES = re.compile(
    r"normalized power|training stress|intensity factor|"
    r"acute training load|chronic training load|training stress balance",
    re.IGNORECASE,
)


def _iter_property_names(schema: dict):
    """Yield every property name declared anywhere in the OpenAPI components."""
    components = schema.get("components", {}).get("schemas", {})
    for model in components.values():
        for prop in (model.get("properties") or {}):
            yield prop


def test_openapi_has_no_trademarked_field_names():
    schema = create_app().openapi()
    offenders = {p for p in _iter_property_names(schema) if p in FORBIDDEN_FIELD_NAMES}
    assert not offenders, f"Trademarked field names leaked into the API: {offenders}"


def test_openapi_descriptions_have_no_trademarked_phrases():
    import json

    text = json.dumps(create_app().openapi())
    matches = set(m.lower() for m in _TRADEMARK_PHRASES.findall(text))
    assert not matches, f"Trademarked phrases leaked into the API schema: {matches}"
