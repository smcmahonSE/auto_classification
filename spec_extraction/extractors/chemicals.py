"""Deterministic extraction rules for Chemicals & Solvents."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, first_regex_match, vocabulary_matches


CAS_PATTERN = re.compile(r"\b(\d{2,7}-\d{2}-\d)\b")
PURITY_PERCENT_PATTERN = re.compile(
    r"((?:>=|>|~|approx\.?\s*)?\s*\d{1,3}(?:\.\d+)?\s*%)(?:\s*(?:pure|purity|assay)?)",
    re.IGNORECASE,
)

PURITY_GRADES = {
    "ACS Grade": ("acs grade", "acs reagent"),
    "HPLC Grade": ("hplc grade", "hplc"),
    "LC-MS Grade": ("lc-ms grade", "lc/ms grade", "lcms grade"),
    "Molecular Biology Grade": ("molecular biology grade",),
    "Reagent Grade": ("reagent grade",),
    "Analytical Grade": ("analytical grade",),
    "Anhydrous": ("anhydrous",),
}


def is_valid_cas(cas_number: str) -> bool:
    """Validate CAS checksum."""
    parts = cas_number.split("-")
    if len(parts) != 3:
        return False

    body = "".join(parts[:2])
    check_digit = int(parts[2])
    checksum = sum(int(digit) * multiplier for multiplier, digit in enumerate(reversed(body), start=1))
    return checksum % 10 == check_digit


def normalize_purity(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def extract_cas_number(row: Mapping[str, object]) -> ExtractedSpec:
    return first_regex_match(
        row=row,
        field_name="CAS Number",
        pattern=CAS_PATTERN,
        method="cas_regex_checksum",
        validator=is_valid_cas,
    )


def extract_purity(row: Mapping[str, object]) -> ExtractedSpec:
    percent_match = first_regex_match(
        row=row,
        field_name="Purity",
        pattern=PURITY_PERCENT_PATTERN,
        method="purity_percent_regex",
        normalizer=normalize_purity,
    )
    if percent_match.status == "matched":
        return percent_match

    return vocabulary_matches(
        row=row,
        field_name="Purity",
        vocabulary=PURITY_GRADES,
        method="purity_grade_dictionary",
    )


def extract_chemical_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract SME-required fields for Chemicals & Solvents."""
    return [
        extract_cas_number(row),
        extract_purity(row),
    ]
