"""Deterministic extraction rules for Proteins and Peptides."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.chemicals import extract_purity
from spec_extraction.extractors.common import ExtractedSpec, first_regex_match, vocabulary_matches
from spec_extraction.extractors.standards import physical_state_vocabulary, species_vocabulary

FORMS = {
    "Lyophilized": ("lyophilized", "lyophilised", "lyo", "freeze-dried", "freeze dried"),
    "Liquid": ("liquid",),
    "Frozen": ("frozen",),
}

SOURCE_ORGANISMS = species_vocabulary(
    ["Human", "Mouse", "Rat", "Rabbit", "Bovine", "Porcine", "E. coli", "HEK293", "CHO", "Baculovirus", "Yeast", "Wheat Germ"]
)
PHYSICAL_STATES = physical_state_vocabulary(["Solid", "Liquid", "Powder", "Solution"])

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
