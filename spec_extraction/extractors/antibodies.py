"""Deterministic extraction rules for Antibodies."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.common import (
    ExtractedSpec,
    first_regex_match,
    missing_spec,
    vocabulary_matches,
)


HOST_PATTERN = re.compile(
    r"\b(mouse|rabbit|goat|rat|donkey|chicken|sheep)\b(?:\s+(?:monoclonal|polyclonal|recombinant))?\s+(?:anti[-\s])",
    re.IGNORECASE,
)
TARGET_PATTERN = re.compile(
    r"\banti[-\s]([A-Za-z0-9][A-Za-z0-9./_+\- ]{1,40}?)(?:\s+(?:antibody|mab|pab|clone|fitc|pe|apc|hrp|biotin)|[,;()]|$)",
    re.IGNORECASE,
)

HOST_SPECIES = {
    "Mouse": ("mouse", "murine"),
    "Rabbit": ("rabbit",),
    "Goat": ("goat",),
    "Rat": ("rat",),
    "Donkey": ("donkey",),
    "Chicken": ("chicken",),
    "Sheep": ("sheep", "ovine"),
}

CLONALITY = {
    "Recombinant Monoclonal": ("recombinant monoclonal", "recombinant mAb"),
    "Monoclonal": ("monoclonal", "mAb"),
    "Polyclonal": ("polyclonal", "pAb"),
}

REACTIVITY = {
    "Human": ("human", "homo sapiens"),
    "Mouse": ("mouse", "murine"),
    "Rat": ("rat",),
    "Rabbit": ("rabbit",),
    "Monkey": ("monkey", "non-human primate", "primate"),
    "Dog": ("dog", "canine"),
    "Pig": ("pig", "porcine"),
    "Cow": ("cow", "bovine"),
}

APPLICATIONS = {
    "FC": ("flow cytometry", "facs", "fc"),
    "WB": ("western blot", "western blotting", "wb", "immunoblot"),
    "IHC": ("ihc", "immunohistochemistry"),
    "IF": ("immunofluorescence", "if"),
    "ELISA": ("elisa",),
    "IP": ("immunoprecipitation", "ip"),
    "ChIP": ("chip", "chromatin immunoprecipitation"),
    "ICC": ("icc", "immunocytochemistry"),
    "Neutralization": ("neutralization", "neutralising", "neutralizing"),
}


def normalize_target(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" -_/")
    return value.strip()


def extract_target(row: Mapping[str, object]) -> ExtractedSpec:
    target = first_regex_match(
        row=row,
        field_name="Target / Specificity",
        pattern=TARGET_PATTERN,
        method="anti_target_regex",
        normalizer=normalize_target,
    )
    if target.status == "matched" and target.value:
        return target

    return missing_spec("Target / Specificity", "anti_target_regex")


def extract_host_species(row: Mapping[str, object]) -> ExtractedSpec:
    host = first_regex_match(
        row=row,
        field_name="Host Species",
        pattern=HOST_PATTERN,
        method="host_before_anti_regex",
        normalizer=lambda value: value.title(),
    )
    if host.status == "matched":
        return host

    return vocabulary_matches(
        row=row,
        field_name="Host Species",
        vocabulary=HOST_SPECIES,
        method="host_species_dictionary",
    )


def extract_antibody_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract SME-required fields for Antibodies."""
    return [
        extract_target(row),
        extract_host_species(row),
        vocabulary_matches(row, "Clonality", CLONALITY, "clonality_dictionary"),
        vocabulary_matches(row, "Reactivity", REACTIVITY, "reactivity_dictionary", multi_select=True),
        vocabulary_matches(row, "Application", APPLICATIONS, "application_dictionary", multi_select=True),
    ]
