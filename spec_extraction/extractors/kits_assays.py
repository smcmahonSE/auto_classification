"""Deterministic extraction rules for Kits and Assays."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, first_regex_match, iter_text_sources, missing_spec, vocabulary_matches


SUB_TYPES = {
    "ELISA Kit": ("elisa kit", "elisa assay", "enzyme-linked immunosorbent"),
    "Extraction Kit": ("extraction kit", "isolation kit", "purification kit", "miniprep", "midiprep"),
    "Detection Kit": ("detection kit", "detection assay", "staining kit", "imaging kit"),
    "Substrate": ("substrate", "tmb", "ecl", "chemiluminescent substrate", "chromogenic substrate"),
    "Blocking Buffer": ("blocking buffer", "blocker", "blocking reagent"),
    "Diluent": ("diluent", "sample diluent", "assay diluent"),
    "Stabilizer": ("stabilizer", "stabiliser"),
    "Signal Enhancer": ("signal enhancer", "enhancer"),
    "Lateral Flow": ("lateral flow", "lfa", "rapid test"),
}

APPLICATIONS = {
    "ELISA": ("elisa", "enzyme-linked immunosorbent"),
    "Western Blot": ("western blot", "wb", "immunoblot"),
    "IHC": ("ihc", "immunohistochemistry"),
    "Lateral Flow": ("lateral flow", "lfa"),
    "Multiplex": ("multiplex", "luminex"),
    "Cell Isolation": ("cell isolation", "cell separation", "cell enrichment"),
}

DETECTION_METHODS = {
    "Chromogenic": ("chromogenic", "colorimetric", "colorimetric detection", "tmb"),
    "Chemiluminescent": ("chemiluminescent", "chemiluminescence", "ecl"),
    "Fluorescent": ("fluorescent", "fluorescence", "fluorometric", "fluorimetric"),
    "Electrochemical": ("electrochemical",),
}

TARGET_ENZYMES = {
    "HRP": ("hrp", "horseradish peroxidase", "peroxidase"),
    "AP": ("ap", "alkaline phosphatase"),
    "Universal": ("universal",),
}

PHYSICAL_STATES = {
    "Liquid": ("liquid", "solution"),
    "Lyophilized": ("lyophilized", "lyophilised", "freeze-dried", "freeze dried"),
    "Powder": ("powder", "powdered"),
}

STORAGE_FIELD_PATTERN = re.compile(
    r"(?:^|\|\s*)(?:Storage(?: Conditions)?|Store(?: at)?|Temperature)\s*:\s*([^|;,]+)",
    re.IGNORECASE,
)
STORAGE_PATTERNS = (
    ("2-8°C", re.compile(r"\b(2\s*[-–]\s*8\s*(?:°\s*)?C|2\s*to\s*8\s*(?:°\s*)?C|4\s*(?:°\s*)?C)\b", re.IGNORECASE)),
    ("-20°C", re.compile(r"\b(-20\s*(?:°\s*)?C)\b", re.IGNORECASE)),
    ("-80°C", re.compile(r"\b(-80\s*(?:°\s*)?C)\b", re.IGNORECASE)),
    ("Room Temperature", re.compile(r"\b(room temperature|ambient|rt)\b", re.IGNORECASE)),
)


def normalize_storage(value: str) -> str:
    for normalized, pattern in STORAGE_PATTERNS:
        if pattern.search(value):
            return normalized
    return re.sub(r"\s+", " ", value).strip(" .")


def first_vocabulary_match(row: Mapping[str, object], field_name: str, vocabulary: dict[str, tuple[str, ...]], method: str) -> ExtractedSpec:
    for source_field, text in iter_text_sources(row):
        for normalized, aliases in vocabulary.items():
            for alias in aliases:
                pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    return ExtractedSpec(
                        field_name=field_name,
                        value=normalized,
                        status="matched",
                        method=method,
                        evidence=match.group(0),
                        source_field=source_field,
                        confidence=0.85,
                    )

    return missing_spec(field_name, method)


def extract_storage_conditions(row: Mapping[str, object]) -> ExtractedSpec:
    labeled = first_regex_match(
        row=row,
        field_name="Storage Conditions",
        pattern=STORAGE_FIELD_PATTERN,
        method="storage_field_regex",
        normalizer=normalize_storage,
    )
    if labeled.status == "matched":
        return labeled

    for normalized, pattern in STORAGE_PATTERNS:
        match = first_regex_match(
            row=row,
            field_name="Storage Conditions",
            pattern=pattern,
            method="storage_temperature_regex",
            normalizer=lambda _raw, value=normalized: value,
        )
        if match.status == "matched":
            return match

    return labeled


def extract_kits_assays_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract SME-proposed fields for Kits and Assays."""
    return [
        first_vocabulary_match(row, "Sub-Type", SUB_TYPES, "kit_subtype_dictionary"),
        vocabulary_matches(row, "Application", APPLICATIONS, "kit_application_dictionary", multi_select=True),
        vocabulary_matches(row, "Detection Method", DETECTION_METHODS, "detection_method_dictionary"),
        vocabulary_matches(row, "Target Enzyme", TARGET_ENZYMES, "target_enzyme_dictionary"),
        vocabulary_matches(row, "Physical State", PHYSICAL_STATES, "kit_physical_state_dictionary"),
        extract_storage_conditions(row),
    ]
