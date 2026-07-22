"""Deterministic extraction rules for Lab Supplies and Consumables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from spec_extraction.extractors.common import (
    ExtractedSpec,
    first_regex_match,
    iter_text_sources,
    missing_spec,
    vocabulary_matches,
)


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
UNIT_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(UNIT_LOOKUP, key=len, reverse=True)
)
SIZE_PATTERN = re.compile(
    rf"\b(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_PATTERN})\b",
    re.IGNORECASE,
)
WELL_PATTERN = re.compile(r"\b(?P<amount>\d+)\s*-?\s*(?P<unit>wells?)\b", re.IGNORECASE)
LABELED_VALUE_PATTERN = re.compile(
    r"(?:^|\|\s*)(?P<label>[A-Za-z /-]+?)\s*:\s*(?P<value>[^|;,]+)",
    re.IGNORECASE,
)
PRODUCT_SIZE_COUNT_OF_PATTERN = re.compile(
    r"\b(?P<count>\d+(?:\.\d+)?)\s*(?:items?|pcs?|pieces?|racks?|plates?|tubes?|tips?|vials?|bottles?|units?)\s+of\b",
    re.IGNORECASE,
)
COUNT_VALUE_PATTERN = re.compile(
    r"\b(?P<count>\d+(?:\.\d+)?)\s*(?P<unit>items?|pcs?|pieces?|racks?|plates?|tubes?|tips?|vials?|bottles?|units?)\b",
    re.IGNORECASE,
)
PACK_CODE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:/|-)\s*(?:pk|pack|cs|case|box|bag|ea|each))\b", re.IGNORECASE)
PACK_OF_PATTERN = re.compile(r"\b((?:pack|case|box|bag)\s+of\s+\d+(?:\.\d+)?)\b", re.IGNORECASE)
PACK_COUNT_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:pack|case|box|bag|pk|cs))\b", re.IGNORECASE)

NON_STERILE_PATTERN = re.compile(r"\b(non[-\s]?sterile|not sterile|unsterile)\b", re.IGNORECASE)
STERILIZABLE_PATTERN = re.compile(r"\b(sterilizable|sterilisable|autoclavable|autoclaveable)\b", re.IGNORECASE)
STERILE_PATTERN = re.compile(r"(?<!non-)(?<!non\s)\b(sterile|sterilized|sterilised|aseptic)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SizeMatch:
    amount: str
    unit: str
    value: str
    evidence: str
    source_field: str
    method: str


def normalize_text_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .")


def normalize_amount(value: str) -> str:
    value = value.strip()
    return value[:-2] if value.endswith(".0") else value


def normalize_unit(value: str) -> str:
    return UNIT_LOOKUP.get(value.lower(), value)


def build_size_match(match: re.Match[str], source_field: str, method: str) -> SizeMatch:
    amount = normalize_amount(match.group("amount"))
    unit = normalize_unit(match.group("unit"))
    value = f"{amount}-well" if unit == "well" else f"{amount} {unit}"
    return SizeMatch(
        amount=amount,
        unit=unit,
        value=value,
        evidence=match.group(0).strip(),
        source_field=source_field,
        method=method,
    )


def iter_labeled_values(row: Mapping[str, object]):
    for source_field, text in iter_text_sources(row):
        for match in LABELED_VALUE_PATTERN.finditer(text):
            label = normalize_text_value(match.group("label")).lower()
            value = normalize_text_value(match.group("value"))
            yield source_field, label, value, match.group(0).strip()


def parse_size_value(value: str, source_field: str, method: str) -> SizeMatch | None:
    match = SIZE_PATTERN.search(value)
    if match:
        return build_size_match(match, source_field, method)

    match = WELL_PATTERN.search(value)
    if match:
        return build_size_match(match, source_field, method)

    return None


def find_capacity_volume_size(row: Mapping[str, object]) -> SizeMatch | None:
    for source_field, label, value, evidence in iter_labeled_values(row):
        if label not in {"capacity", "volume", "size", "product size"}:
            continue

        size_match = parse_size_value(value, source_field, "capacity_volume_size_labeled_regex")
        if size_match:
            return SizeMatch(
                amount=size_match.amount,
                unit=size_match.unit,
                value=size_match.value,
                evidence=evidence,
                source_field=source_field,
                method=size_match.method,
            )

    for source_field, text in iter_text_sources(row):
        size_match = parse_size_value(text, source_field, "capacity_volume_size_regex")
        if size_match:
            return size_match

    return None


def size_spec(size_match: SizeMatch | None, field_name: str, attribute: str) -> ExtractedSpec:
    if not size_match:
        return missing_spec(field_name, "capacity_volume_size_regex")

    return ExtractedSpec(
        field_name=field_name,
        value=getattr(size_match, attribute),
        status="matched",
        method=size_match.method,
        evidence=size_match.evidence,
        source_field=size_match.source_field,
        confidence=0.95,
    )


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


def normalize_pack_count(match: re.Match[str]) -> str:
    return normalize_amount(match.group("count"))


def normalize_pack_size_value(value: str) -> str:
    match = re.search(r"\d+", value)
    return match.group(0) if match else normalize_text_value(value)


def pack_spec(value: str, evidence: str, source_field: str, method: str) -> ExtractedSpec:
    return ExtractedSpec(
        field_name="Pack Size",
        value=normalize_pack_size_value(value),
        status="matched",
        method=method,
        evidence=evidence,
        source_field=source_field,
        confidence=0.9,
    )


def extract_pack_size(row: Mapping[str, object]) -> ExtractedSpec:
    for source_field, label, value, evidence in iter_labeled_values(row):
        if label != "product size":
            continue

        count_of_match = PRODUCT_SIZE_COUNT_OF_PATTERN.search(value)
        if count_of_match:
            return pack_spec(
                normalize_amount(count_of_match.group("count")),
                evidence,
                source_field,
                "product_size_count_of_regex",
            )

        count_match = COUNT_VALUE_PATTERN.search(value)
        if count_match:
            return pack_spec(
                normalize_pack_count(count_match),
                evidence,
                source_field,
                "product_size_count_regex",
            )

    for source_field, label, value, evidence in iter_labeled_values(row):
        if label not in {"pack size", "package size", "pack qty", "pack quantity", "quantity", "size quantity", "unit count"}:
            continue

        return pack_spec(value, evidence, source_field, "pack_size_labeled_regex")

    for source_field, text in iter_text_sources(row):
        for pattern, method in (
            (PACK_CODE_PATTERN, "pack_code_regex"),
            (PACK_OF_PATTERN, "pack_of_regex"),
            (PACK_COUNT_PATTERN, "pack_count_regex"),
        ):
            match = pattern.search(text)
            if match:
                return pack_spec(match.group(1), match.group(0), source_field, method)

    return missing_spec("Pack Size", "pack_size_regex")


def extract_lab_supplies_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract team-selected fields for Lab Supplies and Consumables."""
    size_match = find_capacity_volume_size(row)
    return [
        vocabulary_matches(row, "Material", MATERIALS, "material_dictionary"),
        extract_sterility(row),
        size_spec(size_match, "Capacity Volume Size", "value"),
        size_spec(size_match, "Size Amount", "amount"),
        size_spec(size_match, "Size Unit", "unit"),
        extract_pack_size(row),
        vocabulary_matches(row, "Color", COLORS, "color_dictionary"),
    ]
