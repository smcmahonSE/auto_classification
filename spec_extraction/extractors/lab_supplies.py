"""Deterministic extraction rules for Lab Supplies and Consumables."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, first_regex_match, missing_spec, vocabulary_matches


MATERIALS = {
    "Polypropylene": ("polypropylene",),
    "Polystyrene": ("polystyrene",),
    "Polyethylene": ("polyethylene", "hdpe", "ldpe"),
    "Glass": ("glass",),
    "Borosilicate Glass": ("borosilicate", "borosilicate glass", "pyrex"),
    "PTFE": ("ptfe", "teflon"),
    "Nitrile": ("nitrile",),
    "Latex": ("latex",),
    "Stainless Steel": ("stainless steel",),
    "PVC": ("pvc", "polyvinyl chloride"),
    "Silicone": ("silicone",),
    "Nylon": ("nylon",),
    "Polycarbonate": ("polycarbonate",),
    "Cellulose": ("cellulose", "cellulose acetate"),
}

COLORS = {
    "Black": ("black",),
    "Blue": ("blue",),
    "Brown": ("brown",),
    "Clear": ("clear", "transparent"),
    "Green": ("green",),
    "Orange": ("orange",),
    "Pink": ("pink",),
    "Purple": ("purple", "violet"),
    "Red": ("red",),
    "White": ("white",),
    "Yellow": ("yellow",),
    "Amber": ("amber",),
    "Natural": ("natural",),
    "Assorted": ("assorted", "mixed colors", "multicolor", "multi-color"),
}

CAPACITY_VOLUME_SIZE_PATTERNS = (
    re.compile(
        r"(?:^|\|\s*)(?:Size|Capacity|Volume|Product Size)\s*:\s*"
        r"([^|;,]+?(?:\d+(?:\.\d+)?\s*(?:uL|µL|mL|L|nL|g|mg|kg|oz|mm|cm|in|inch|well|wells)|\d+\s*-\s*well)[^|;,]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d+(?:\.\d+)?\s*(?:uL|µL|mL|L|nL|g|mg|kg|oz|mm|cm|in|inch))\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d+\s*-\s*well|\d+\s*well)\b", re.IGNORECASE),
)

PACK_SIZE_PATTERNS = (
    re.compile(
        r"(?:^|\|\s*)(?:Pack Size|Pack Qty|Pack Quantity|Quantity|Size Quantity|Unit Count)\s*:\s*([^|;,]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d+\s*/\s*(?:pk|pack|cs|case|box|bag|ea|each))\b", re.IGNORECASE),
    re.compile(r"\b((?:pack|case|box|bag)\s+of\s+\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+\s*(?:pack|case|box|bag|pk|cs))\b", re.IGNORECASE),
)

NON_STERILE_PATTERN = re.compile(r"\b(non[-\s]?sterile|not sterile|unsterile)\b", re.IGNORECASE)
STERILIZABLE_PATTERN = re.compile(r"\b(sterilizable|sterilisable|autoclavable|autoclaveable)\b", re.IGNORECASE)
STERILE_PATTERN = re.compile(r"(?<!non-)(?<!non\s)\b(sterile|sterilized|sterilised|aseptic)\b", re.IGNORECASE)


def normalize_text_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .")


def extract_sterility(row: Mapping[str, object]) -> ExtractedSpec:
    for value, pattern, method in (
        ("Non-Sterile", NON_STERILE_PATTERN, "non_sterile_regex"),
        ("Sterilizable", STERILIZABLE_PATTERN, "sterilizable_regex"),
        ("Sterile", STERILE_PATTERN, "sterile_regex"),
    ):
        match = first_regex_match(
            row=row,
            field_name="Sterility",
            pattern=pattern,
            method=method,
            normalizer=lambda _raw, normalized=value: normalized,
        )
        if match.status == "matched":
            return match

    return missing_spec("Sterility", "sterility_regex")


def extract_first_pattern(
    row: Mapping[str, object],
    field_name: str,
    patterns: tuple[re.Pattern[str], ...],
    method: str,
) -> ExtractedSpec:
    for pattern in patterns:
        match = first_regex_match(
            row=row,
            field_name=field_name,
            pattern=pattern,
            method=method,
            normalizer=normalize_text_value,
        )
        if match.status == "matched":
            return match

    return missing_spec(field_name, method)


def extract_lab_supplies_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract team-selected fields for Lab Supplies and Consumables."""
    return [
        vocabulary_matches(row, "Material", MATERIALS, "material_dictionary"),
        extract_sterility(row),
        extract_first_pattern(
            row,
            "Capacity / Volume / Size",
            CAPACITY_VOLUME_SIZE_PATTERNS,
            "capacity_volume_size_regex",
        ),
        extract_first_pattern(row, "Pack Size", PACK_SIZE_PATTERNS, "pack_size_regex"),
        vocabulary_matches(row, "Color", COLORS, "color_dictionary"),
    ]
