"""Deterministic extraction rules for Kits and Assays."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, vocabulary_matches
from spec_extraction.extractors.standards import extract_storage_conditions, first_vocabulary_match, physical_state_vocabulary


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

PHYSICAL_STATES = physical_state_vocabulary(["Liquid", "Lyophilized", "Powder"])


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
