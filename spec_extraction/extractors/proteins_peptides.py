"""Deterministic extraction rules for Proteins and Peptides."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.chemicals import extract_purity
from spec_extraction.extractors.common import ExtractedSpec, first_regex_match, vocabulary_matches


SOURCE_ORGANISMS = {
    "Human": ("human", "homo sapiens", "recombinant human"),
    "Mouse": ("mouse", "murine", "mice", "mus musculus", "recombinant mouse"),
    "Rat": ("rat", "rattus norvegicus", "recombinant rat"),
    "Rabbit": ("rabbit",),
    "Bovine": ("bovine", "cow", "bos taurus"),
    "Porcine": ("porcine", "pig", "swine"),
    "E. coli": ("e. coli", "escherichia coli", "ecoli"),
    "HEK293": ("hek293", "hek 293", "293 cells", "293-cell", "293 cell"),
    "CHO": ("cho", "cho cells", "cho-cell", "cho cell"),
    "Baculovirus": ("baculovirus", "insect cells", "sf9", "sf21"),
    "Yeast": ("yeast", "saccharomyces cerevisiae", "pichia pastoris"),
    "Wheat Germ": ("wheat germ",),
}

FORMS = {
    "Lyophilized": ("lyophilized", "lyophilised", "lyo", "freeze-dried", "freeze dried"),
    "Liquid": ("liquid",),
    "Frozen": ("frozen",),
}

PHYSICAL_STATES = {
    "Solid": ("solid",),
    "Liquid": ("liquid",),
    "Powder": ("powder", "powdered"),
    "Solution": ("solution",),
}

ACTIVITY_FIELD_PATTERN = re.compile(
    r"(?:^|\|\s*)(?:Specific Activity|Biological Activity|Bioactivity|Activity|ED50|EC50|IC50)\s*:\s*([^|;,]+)",
    re.IGNORECASE,
)
ACTIVITY_VALUE_PATTERN = re.compile(
    r"\b((?:ED50|EC50|IC50)\s*(?:<|<=|>|>=|=|~|approx\.?)?\s*\d+(?:\.\d+)?\s*(?:pg|ng|ug|µg|mg)?/?mL|"
    r"\d+(?:\.\d+)?\s*(?:U|units|IU|kU)\s*/\s*(?:mg|ug|µg))\b",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .")


def extract_activity(row: Mapping[str, object]) -> ExtractedSpec:
    labeled = first_regex_match(
        row=row,
        field_name="Activity",
        pattern=ACTIVITY_FIELD_PATTERN,
        method="activity_field_regex",
        normalizer=normalize_text,
    )
    if labeled.status == "matched":
        return labeled

    return first_regex_match(
        row=row,
        field_name="Activity",
        pattern=ACTIVITY_VALUE_PATTERN,
        method="activity_value_regex",
        normalizer=normalize_text,
    )


def extract_proteins_peptides_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract SME-proposed fields for Proteins and Peptides."""
    return [
        vocabulary_matches(row, "Source Organism", SOURCE_ORGANISMS, "source_organism_dictionary", multi_select=True),
        vocabulary_matches(row, "Form", FORMS, "protein_form_dictionary"),
        vocabulary_matches(row, "Physical State", PHYSICAL_STATES, "physical_state_dictionary"),
        extract_purity(row),
        extract_activity(row),
    ]
