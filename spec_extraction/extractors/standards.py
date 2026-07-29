"""Shared normalization standards for deterministic spec extraction."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from spec_extraction.extractors.common import ExtractedSpec, first_regex_match, iter_text_sources, missing_spec


SPECIES = {
    "Human": ("human", "homo sapiens", "hsapiens", "h. sapiens"),
    "Mouse": ("mouse", "mice", "murine", "mus musculus", "mmusculus", "m. musculus"),
    "Rat": ("rat", "rattus norvegicus", "rnorvegicus", "r. norvegicus"),
    "Rabbit": ("rabbit", "oryctolagus cuniculus"),
    "Bovine": ("bovine", "cow", "bos taurus", "calf"),
    "Porcine": ("porcine", "pig", "swine", "sus scrofa"),
    "Canine": ("canine", "dog", "beagle"),
    "Feline": ("feline", "cat"),
    "Sheep": ("sheep", "ovine"),
    "Goat": ("goat", "caprine"),
    "Chicken": ("chicken", "gallus gallus"),
    "Guinea Pig": ("guinea pig",),
    "Hamster": ("hamster", "syrian hamster", "chinese hamster"),
    "Non-Human Primate": ("non-human primate", "non human primate", "nhp", "macaque", "cynomolgus", "rhesus", "marmoset", "baboon"),
    "Zebrafish": ("zebrafish", "danio rerio", "d. rerio"),
    "Drosophila": ("drosophila", "fruit fly", "d. melanogaster", "drosophila melanogaster"),
    "C. elegans": ("c. elegans", "caenorhabditis elegans"),
    "Yeast": ("yeast", "saccharomyces cerevisiae", "s. cerevisiae", "pichia pastoris"),
    "E. coli": ("e. coli", "escherichia coli", "ecoli"),
    "HEK293": ("hek293", "hek 293", "293 cells", "293-cell", "293 cell"),
    "CHO": ("cho", "cho cells", "cho-cell", "cho cell"),
    "Baculovirus": ("baculovirus", "insect cells", "sf9", "sf21"),
    "Wheat Germ": ("wheat germ",),
}

PHYSICAL_STATE_ALIASES = {
    "Solid": ("solid",),
    "Liquid": ("liquid", "fluid"),
    "Gas": ("gas", "gaseous"),
    "Powder": ("powder", "powdered"),
    "Crystal": ("crystal", "crystalline"),
    "Solution": ("solution", "aqueous solution"),
    "Gel": ("gel",),
    "Paste": ("paste",),
    "Lyophilized": ("lyophilized", "lyophilised", "freeze-dried", "freeze dried"),
}

STERILITY = {
    "Non-Sterile": ("non-sterile", "non sterile", "not sterile", "unsterile"),
    "Sterilizable": ("sterilizable", "sterilisable", "autoclavable", "autoclaveable"),
    "Sterile": ("sterile", "sterilized", "sterilised", "aseptic"),
}

STORAGE_CONDITIONS = {
    "2-8°C": (
        "2-8c",
        "2-8 c",
        "2-8°c",
        "2 to 8c",
        "2 to 8 c",
        "2 to 8°c",
        "4c",
        "4 c",
        "4°c",
        "refrigerated",
    ),
    "-20°C": ("-20c", "-20 c", "-20°c"),
    "-80°C": ("-80c", "-80 c", "-80°c"),
    "Room Temperature": ("room temperature", "ambient", "rt"),
    "Desiccated": ("desiccated", "dry place", "keep dry"),
    "Protected from Light": ("protected from light", "protect from light", "light sensitive"),
}

UNIT_ALIASES = {
    "nL": ("nl", "nanoliter", "nanoliters", "nanolitre", "nanolitres"),
    "uL": ("ul", "µl", "μl", "microliter", "microliters", "microlitre", "microlitres"),
    "mL": ("ml", "milliliter", "milliliters", "millilitre", "millilitres"),
    "L": ("l", "liter", "liters", "litre", "litres"),
    "ug": ("ug", "µg", "μg", "mcg", "microgram", "micrograms"),
    "mg": ("mg", "milligram", "milligrams"),
    "g": ("g", "gram", "grams"),
    "kg": ("kg", "kilogram", "kilograms"),
    "mm": ("mm", "millimeter", "millimeters", "millimetre", "millimetres"),
    "cm": ("cm", "centimeter", "centimeters", "centimetre", "centimetres"),
    "in": ("in", "inch", "inches"),
    "well": ("well", "wells"),
}
UNIT_LOOKUP = {alias: unit for unit, aliases in UNIT_ALIASES.items() for alias in aliases}

REGULATORY_STATUS = {
    "FDA Cleared": ("fda cleared", "fda-cleared", "510(k)", "510k"),
    "CE Marked": ("ce marked", "ce-marked", "ce mark"),
    "RUO": ("ruo", "research use only", "for research use only"),
    "IVD": ("ivd", "in vitro diagnostic", "in-vitro diagnostic"),
    "NIST Traceable": ("nist traceable", "nist-traceable"),
    "ISO Certified": ("iso certified", "iso-certified"),
}


def allowed_vocabulary(vocabulary: Mapping[str, Sequence[str]], allowed_values: Sequence[str]) -> dict[str, Sequence[str]]:
    return {value: vocabulary[value] for value in allowed_values if value in vocabulary}


def physical_state_vocabulary(allowed_values: Sequence[str]) -> dict[str, Sequence[str]]:
    return allowed_vocabulary(PHYSICAL_STATE_ALIASES, allowed_values)


def species_vocabulary(allowed_values: Sequence[str] | None = None) -> dict[str, Sequence[str]]:
    if allowed_values is None:
        return SPECIES
    return allowed_vocabulary(SPECIES, allowed_values)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .")


def normalize_amount(value: str) -> str:
    value = value.strip()
    return value[:-2] if value.endswith(".0") else value


def normalize_unit(value: str) -> str:
    return UNIT_LOOKUP.get(value.lower(), value)


def first_vocabulary_match(
    row: Mapping[str, object],
    field_name: str,
    vocabulary: Mapping[str, Sequence[str]],
    method: str,
) -> ExtractedSpec:
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


def extract_storage_conditions(row: Mapping[str, object], field_name: str = "Storage Conditions") -> ExtractedSpec:
    storage_field_pattern = re.compile(
        r"(?:^|\|\s*)(?:Storage(?: Conditions)?|Store(?: at)?|Temperature)\s*:\s*([^|;,]+)",
        re.IGNORECASE,
    )
    labeled = first_regex_match(
        row=row,
        field_name=field_name,
        pattern=storage_field_pattern,
        method="storage_field_regex",
        normalizer=normalize_storage_condition,
    )
    if labeled.status == "matched":
        return labeled

    return first_vocabulary_match(row, field_name, STORAGE_CONDITIONS, "storage_condition_dictionary")


def normalize_storage_condition(value: str) -> str:
    text = value.lower()
    for normalized, aliases in STORAGE_CONDITIONS.items():
        for alias in aliases:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text, re.IGNORECASE):
                return normalized
    return normalize_text(value)
