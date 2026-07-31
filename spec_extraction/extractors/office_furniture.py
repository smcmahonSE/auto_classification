"""Deterministic extraction rules for furniture and office supply categories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, iter_text_sources, missing_spec, vocabulary_matches
from spec_extraction.extractors.standards import (
    COLORS,
    MATERIALS,
    first_vocabulary_match,
    normalize_amount,
    normalize_text,
    normalize_unit,
)


FURNITURE_SUB_TYPES = {
    "Chair Accessory": ("caster", "casters", "chair mat", "chair accessory", "arm rest", "backrest"),
    "Stool": ("stool",),
    "Seating": ("seating", "seat", "chair"),
    "Desk / Workstation": ("desk", "workstation", "work station"),
    "Table / Bench": ("table", "bench", "workbench", "lab bench"),
    "Cabinet": ("cabinet", "locker", "cupboard"),
    "Shelving": ("shelf", "shelving", "shelves"),
    "Rack": ("rack", "holder", "stand"),
    "Cart": ("cart", "trolley"),
    "Dispenser": ("dispenser",),
    "Storage": ("storage", "box", "container", "bin", "case"),
}

OFFICE_SUB_TYPES = {
    "Writing Instrument": ("pen", "pens", "marker", "markers", "pencil", "pencils", "highlighter", "highlighters"),
    "Paper / Notebook": ("paper", "notebook", "notepad", "pad", "copy paper", "sticky note", "post-it"),
    "Label / Tag": ("label", "labels", "tag", "tags"),
    "Tape / Adhesive": ("tape", "adhesive", "glue", "glue stick"),
    "Binder / Folder": ("binder", "binders", "folder", "folders"),
    "Envelope / Mailer": ("envelope", "envelopes", "mailer", "mailers"),
    "Fastener / Clip / Staple": ("clip", "clips", "staple", "staples", "stapler", "brad", "fastener"),
    "Desk Accessory": ("desk organizer", "organizer", "tray", "sorter", "holder"),
    "Shipping / Packaging": ("box", "carton", "shipping", "packing", "bubble mailer", "packing slip"),
}
FURNITURE_MATERIALS = {
    "Stainless Steel": ("stainless steel",),
    "Steel": ("steel",),
    "Aluminum": ("aluminum", "aluminium"),
    "Plastic": ("plastic",),
    "Polypropylene": ("polypropylene",),
    "Polyethylene": ("polyethylene", "hdpe", "ldpe"),
    "Wood": ("wood",),
    "Laminate": ("laminate", "laminated"),
    "Cardboard": ("cardboard", "corrugated"),
    "Acrylic": ("acrylic",),
    "Vinyl": ("vinyl",),
    "Fabric": ("fabric", "cloth", "upholstered"),
    "Leather": ("leather",),
    "Mesh": ("mesh",),
    "Metal": ("metal",),
}

LABEL_PATTERN = re.compile(r"(?:^|\|\s*)(?P<label>[A-Za-z /-]+?)\s*:\s*(?P<value>[^|;,]+)", re.IGNORECASE)
MEASURE = r"\d+(?:\.\d+)?(?:\s+\d+/\d+)?"
DIMENSION_PATTERN = re.compile(
    rf"\b(?P<d1>{MEASURE})\s*(?:\"|in(?:ches?)?|cm|mm|ft|feet|')?\s*[x×]\s*"
    rf"(?P<d2>{MEASURE})\s*(?:\"|in(?:ches?)?|cm|mm|ft|feet|')?\s*"
    rf"(?:[x×]\s*(?P<d3>{MEASURE})\s*)?"
    rf"(?P<unit>\"|in(?:ches?)?|cm|mm|ft|feet|')?\b",
    re.IGNORECASE,
)
LENGTH_PATTERN = re.compile(r"(?:^|\|\s*)(?:Length|Depth)\s*:\s*([^|;,]+)", re.IGNORECASE)
WIDTH_PATTERN = re.compile(r"(?:^|\|\s*)Width\s*:\s*([^|;,]+)", re.IGNORECASE)
HEIGHT_PATTERN = re.compile(r"(?:^|\|\s*)Height\s*:\s*([^|;,]+)", re.IGNORECASE)
SIZE_LABELS = {"size", "dimensions", "dimension", "size volume dimension", "l x w x h", "w x l"}
PACK_LABELS = {"package size", "pack size", "pack qty", "pack quantity", "quantity", "unit count"}
PACK_VALUE_PATTERN = re.compile(r"\b(?:pack\s+of\s+)?(?P<count>\d+(?:\.\d+)?)\s*(?:/|-|\s*)?(?:pk|pack|pcs?|items?|ea|each|cs|case|box)?\b", re.IGNORECASE)
LOAD_RATING_PATTERNS = (
    re.compile(r"\b(\d+(?:\.\d+)?\s*(?:lb|lbs|pounds?|kg)\s*(?:cap|capacity|load rating|load))\b", re.IGNORECASE),
    re.compile(r"\b((?:cap|capacity|load rating|load)\s*:?\s*\d+(?:\.\d+)?\s*(?:lb|lbs|pounds?|kg))\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class DimensionMatch:
    length: str | None
    width: str | None
    height: str | None
    value: str
    evidence: str
    source_field: str
    method: str


def iter_labeled_values(row: Mapping[str, object]):
    for source_field, text in iter_text_sources(row):
        for match in LABEL_PATTERN.finditer(text):
            label = normalize_text(match.group("label")).lower()
            value = normalize_text(match.group("value"))
            yield source_field, label, value, match.group(0).strip()


def normalize_dimension_unit(value: str | None, evidence: str = "") -> str:
    if not value:
        if '"' in evidence:
            return "in"
        if "'" in evidence:
            return "ft"
        if re.search(r"\bcm\b", evidence, re.IGNORECASE):
            return "cm"
        if re.search(r"\bmm\b", evidence, re.IGNORECASE):
            return "mm"
        if re.search(r"\bin(?:ches?)?\b", evidence, re.IGNORECASE):
            return "in"
        return ""
    if value in {'"', "in", "inch", "inches"}:
        return "in"
    if value in {"'", "ft", "feet"}:
        return "ft"
    return normalize_unit(value)


def with_unit(value: str | None, unit: str) -> str | None:
    if not value:
        return None
    value = normalize_amount(value)
    return f"{value} {unit}".strip() if unit else value


def dimension_from_match(match: re.Match[str], source_field: str, method: str) -> DimensionMatch:
    unit = normalize_dimension_unit(match.group("unit"), match.group(0))
    length = with_unit(match.group("d1"), unit)
    width = with_unit(match.group("d2"), unit)
    height = with_unit(match.group("d3"), unit)
    parts = [part for part in (length, width, height) if part]
    return DimensionMatch(
        length=length,
        width=width,
        height=height,
        value=" x ".join(parts),
        evidence=match.group(0).strip(),
        source_field=source_field,
        method=method,
    )


def find_dimensions(row: Mapping[str, object]) -> DimensionMatch | None:
    for source_field, label, value, evidence in iter_labeled_values(row):
        if label not in SIZE_LABELS:
            continue
        match = DIMENSION_PATTERN.search(value)
        if match:
            dimension = dimension_from_match(match, source_field, "dimension_labeled_regex")
            return DimensionMatch(
                length=dimension.length,
                width=dimension.width,
                height=dimension.height,
                value=dimension.value,
                evidence=evidence,
                source_field=source_field,
                method=dimension.method,
            )

    for source_field, text in iter_text_sources(row):
        match = DIMENSION_PATTERN.search(text)
        if match:
            return dimension_from_match(match, source_field, "dimension_regex")

    return None


def dimension_spec(dimension: DimensionMatch | None, field_name: str, attribute: str) -> ExtractedSpec:
    if not dimension:
        return missing_spec(field_name, "dimension_regex")
    value = getattr(dimension, attribute)
    if not value:
        return missing_spec(field_name, dimension.method)
    return ExtractedSpec(
        field_name=field_name,
        value=value,
        status="matched",
        method=dimension.method,
        evidence=dimension.evidence,
        source_field=dimension.source_field,
        confidence=0.85,
    )


def normalize_pack_size(value: str) -> str:
    match = re.search(r"\d+", value)
    return match.group(0) if match else normalize_text(value)


def extract_pack_size(row: Mapping[str, object]) -> ExtractedSpec:
    for source_field, label, value, evidence in iter_labeled_values(row):
        if label in PACK_LABELS or label in {"packaging uom", "uom"}:
            match = PACK_VALUE_PATTERN.search(value)
            if match:
                return ExtractedSpec(
                    field_name="Pack Size",
                    value=normalize_pack_size(match.group("count")),
                    status="matched",
                    method="pack_size_labeled_regex",
                    evidence=evidence,
                    source_field=source_field,
                    confidence=0.9,
                )

    for source_field, text in iter_text_sources(row):
        match = PACK_VALUE_PATTERN.search(text)
        if match and re.search(r"\b(pk|pack|pcs?|items?|ea|each|case|box)\b", match.group(0), re.IGNORECASE):
            return ExtractedSpec(
                field_name="Pack Size",
                value=normalize_pack_size(match.group("count")),
                status="matched",
                method="pack_size_regex",
                evidence=match.group(0),
                source_field=source_field,
                confidence=0.85,
            )

    return missing_spec("Pack Size", "pack_size_regex")


def extract_capacity_load_rating(row: Mapping[str, object]) -> ExtractedSpec:
    for source_field, text in iter_text_sources(row):
        for pattern in LOAD_RATING_PATTERNS:
            match = pattern.search(text)
            if match:
                return ExtractedSpec(
                    field_name="Capacity / Load Rating",
                    value=normalize_text(match.group(1)),
                    status="matched",
                    method="capacity_load_rating_regex",
                    evidence=match.group(0),
                    source_field=source_field,
                    confidence=0.9,
                )
    return missing_spec("Capacity / Load Rating", "capacity_load_rating_regex")


def extract_furniture_storage_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract recommended fields for Furniture and Storage."""
    dimensions = find_dimensions(row)
    return [
        first_vocabulary_match(row, "Sub-Type", FURNITURE_SUB_TYPES, "furniture_subtype_dictionary"),
        first_vocabulary_match(row, "Material", FURNITURE_MATERIALS, "material_dictionary"),
        vocabulary_matches(row, "Color", COLORS, "color_dictionary"),
        dimension_spec(dimensions, "Length", "length"),
        dimension_spec(dimensions, "Width", "width"),
        dimension_spec(dimensions, "Height", "height"),
        extract_capacity_load_rating(row),
        extract_pack_size(row),
    ]


def extract_general_office_supplies_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract recommended fields for General Office Supplies."""
    dimensions = find_dimensions(row)
    return [
        first_vocabulary_match(row, "Sub-Type", OFFICE_SUB_TYPES, "office_subtype_dictionary"),
        vocabulary_matches(row, "Color", COLORS, "color_dictionary"),
        dimension_spec(dimensions, "Size / Dimensions", "value"),
        extract_pack_size(row),
        vocabulary_matches(row, "Material", MATERIALS, "material_dictionary"),
    ]
