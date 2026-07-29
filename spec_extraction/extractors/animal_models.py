"""Deterministic extraction rules for Animal Models."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, first_regex_match, vocabulary_matches
from spec_extraction.extractors.standards import species_vocabulary


ANIMAL_SPECIES = species_vocabulary(
    ["Mouse", "Rat", "Zebrafish", "Guinea Pig", "Hamster", "Non-Human Primate", "Rabbit", "Porcine", "Canine", "Drosophila", "C. elegans", "Yeast"]
)

GENETIC_MODIFICATIONS = {
    "Wild-type": ("wild-type", "wild type", "wt"),
    "Knockout": ("knockout", "knock-out", "ko", "null mutant"),
    "Transgenic": ("transgenic", "tg"),
    "Knock-in": ("knock-in", "knockin", "ki"),
    "Humanized": ("humanized", "humanised"),
    "CRISPR-Modified": ("crispr", "crispr-modified", "crispr modified", "gene edited", "gene-edited"),
}

STRAIN_PATTERN = re.compile(
    r"\b(C57BL/6\w*|BALB/c\w*|DBA/2\w*|FVB/N\w*|129S\w*|CD-1|Sprague[-\s]?Dawley|Wistar|Long[-\s]?Evans|"
    r"NOD(?:[-/\s]?SCID)?|NSG|SCID|nude|Rag[12]|db/db|ob/ob|5xFAD|APP/PS1|3xTg-AD|SOD1|"
    r"cynomolgus|rhesus|marmoset|New Zealand White|Hartley|Dunkin[-\s]?Hartley|Yucatan|Göttingen|Gottingen|Beagle)\b",
    re.IGNORECASE,
)


def normalize_strain(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .")


def extract_animal_model_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract SME-proposed fields for Animal Models."""
    return [
        vocabulary_matches(row, "Species", ANIMAL_SPECIES, "animal_species_dictionary", multi_select=True),
        first_regex_match(row, "Strain", STRAIN_PATTERN, "strain_regex", normalizer=normalize_strain),
        vocabulary_matches(row, "Genetic Modification", GENETIC_MODIFICATIONS, "genetic_modification_dictionary"),
    ]
