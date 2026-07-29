"""Deterministic extraction rules for Controls, Calibrators and Standards."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, first_regex_match, vocabulary_matches
from spec_extraction.extractors.standards import REGULATORY_STATUS, normalize_text


SUB_TYPES = {
    "QC Control": ("qc control", "quality control", "assay control", "positive control", "negative control", "control material"),
    "Calibrator": ("calibrator", "calibration standard", "calibrator set", "master calibrator"),
    "Reference Standard": ("reference standard", "certified reference", "crm", "standard solution", "nist"),
    "Proficiency Testing": ("proficiency", "proficiency testing", "pt material", "eqa", "external quality assurance"),
}

ANALYTE_FIELD_PATTERN = re.compile(
    r"(?:^|\|\s*)(?:Analyte(?: / Parameter)?|Parameter|Target|Marker|Assay)\s*:\s*([^|;,]+)",
    re.IGNORECASE,
)
ANALYTES = {
    "DNA Ladder": ("dna ladder",),
    "Protein Ladder": ("protein ladder", "molecular weight marker", "mw marker"),
    "Glucose": ("glucose",),
    "Hemoglobin": ("hemoglobin", "haemoglobin", "hba1c"),
    "Cholesterol": ("cholesterol",),
    "pH": ("ph",),
    "Conductivity": ("conductivity",),
    "Osmolality": ("osmolality",),
    "Hematopoietic Progenitor Cells": ("hematopoietic progenitor", "hpc", "cd34"),
    "CFU": ("cfu", "colony forming unit"),
}

MATRICES = {
    "Human Blood": ("human blood", "whole blood"),
    "Serum": ("serum",),
    "Plasma": ("plasma",),
    "Urine": ("urine",),
    "Synthetic": ("synthetic", "artificial"),
    "Latex Particle": ("latex particle", "latex particles"),
    "Buffer": ("buffer",),
    "Aqueous": ("aqueous", "water"),
}


def extract_analyte_parameter(row: Mapping[str, object]) -> ExtractedSpec:
    labeled = first_regex_match(
        row=row,
        field_name="Analyte / Parameter",
        pattern=ANALYTE_FIELD_PATTERN,
        method="analyte_field_regex",
        normalizer=normalize_text,
    )
    if labeled.status == "matched":
        return labeled

    return vocabulary_matches(row, "Analyte / Parameter", ANALYTES, "analyte_dictionary", multi_select=True)


def extract_controls_calibrators_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract SME-proposed fields for Controls, Calibrators and Standards."""
    return [
        vocabulary_matches(row, "Sub-Type", SUB_TYPES, "control_subtype_dictionary"),
        extract_analyte_parameter(row),
        vocabulary_matches(row, "Matrix", MATRICES, "matrix_dictionary"),
        vocabulary_matches(row, "Regulatory Status", REGULATORY_STATUS, "regulatory_status_dictionary", multi_select=True),
    ]
