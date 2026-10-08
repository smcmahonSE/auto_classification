"""Generic regex-based extractors for the v2 spec model's simple numeric/text fields.

These are intentionally lighter-weight than the bespoke per-field logic in
extractors/lab_supplies.py (ported instead into v2/overrides/ for the fields that
need it, like Capacity / Volume / Size). Most L1-universal fields and many L3 fields
just need "find a labeled value, else look for a bare number+unit in free text" —
this module implements that once, driven by a small per-pattern-key config, rather
than writing ~30 near-identical bespoke regexes.

Coverage is expected to be uneven across these fields (some, like Category UNSPSC,
have no real source signal at all and will show as mostly "missing" in the coverage
report by design — that's honest, not a bug).
"""

from __future__ import annotations

import re
from typing import Mapping, NamedTuple

from spec_extraction.extractors.common import ExtractedSpec, missing_spec
from spec_extraction.extractors.lab_supplies import iter_labeled_values, normalize_text_value

URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
IMAGE_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+\.(?:png|jpe?g|gif|webp|svg)\b", re.IGNORECASE)


class NumericPattern(NamedTuple):
    labels: tuple[str, ...]           # labeled-value keys to look for, e.g. ("lead time", "lead time days")
    bare_regex: re.Pattern | None     # fallback free-text pattern if no labeled value found


def _pct_pattern(*label_words: str) -> re.Pattern:
    words = "|".join(re.escape(w) for w in label_words)
    return re.compile(rf"\b(?:{words})\D{{0,10}}?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


NUMERIC_PATTERNS: dict[str, NumericPattern] = {
    "lead_time_days": NumericPattern(
        labels=("lead time", "lead time days", "lead-time"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:business\s+)?days?\s+lead\s*time\b", re.IGNORECASE),
    ),
    "unit_count": NumericPattern(labels=("unit", "units per pack", "count per unit"), bare_regex=None),
    "quantity": NumericPattern(labels=("quantity", "qty"), bare_regex=None),
    "inventory": NumericPattern(labels=("inventory", "in stock", "stock", "units available"), bare_regex=None),
    "ship_from": NumericPattern(labels=("ship from", "ships from", "origin"), bare_regex=None),
    "neck_finish": NumericPattern(
        labels=("neck finish", "neck", "thread"),
        bare_regex=re.compile(r"\bGL\s?\d{2,3}\b|\b\d{2}-\d{3}\b", re.IGNORECASE),
    ),
    "temperature_range": NumericPattern(
        labels=("temperature range", "temperature", "storage temperature"),
        bare_regex=re.compile(
            r"-?\d+(?:\.\d+)?\s*(?:to|-|–)\s*-?\d+(?:\.\d+)?\s*°?C\b"
            r"|-?\d+(?:\.\d+)?\s*°C\b",
            re.IGNORECASE,
        ),
    ),
    "compatibility": NumericPattern(labels=("compatibility", "compatible with"), bare_regex=None),
    "well_number": NumericPattern(
        labels=("well number", "wells", "well count"),
        bare_regex=re.compile(r"\b(6|12|24|48|96|384|1536)\s*-?\s*wells?\b", re.IGNORECASE),
    ),
    "pore_size": NumericPattern(
        labels=("pore size",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:µm|um|micron)\b", re.IGNORECASE),
    ),
    "needle_gauge_length": NumericPattern(
        labels=("needle gauge", "gauge"),
        bare_regex=re.compile(r"\b(\d{1,2}\s*G)\b", re.IGNORECASE),
    ),
    "inner_diameter": NumericPattern(
        labels=("inner diameter", "id", "i.d."),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|in|inch(?:es)?)\s*(?:id|inner diameter)\b", re.IGNORECASE),
    ),
    "outer_diameter": NumericPattern(
        labels=("outer diameter", "od", "o.d."),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|in|inch(?:es)?)\s*(?:od|outer diameter)\b", re.IGNORECASE),
    ),
    "wall_thickness": NumericPattern(labels=("wall thickness",), bare_regex=None),
    "length_ft_m": NumericPattern(
        labels=("length",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(ft|feet|m|meters?)\b", re.IGNORECASE),
    ),
    "durometer": NumericPattern(
        labels=("durometer", "shore a", "shore hardness"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*shore\s*a\b", re.IGNORECASE),
    ),
    "max_pressure": NumericPattern(
        labels=("max pressure", "maximum pressure"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*psig?\b", re.IGNORECASE),
    ),
    "active_ingredient": NumericPattern(labels=("active ingredient", "active ingredients"), bare_regex=None),
    "width": NumericPattern(
        labels=("width",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|in|inch(?:es)?)\s*wide\b", re.IGNORECASE),
    ),
    "roll_length": NumericPattern(
        labels=("roll length",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*m\s*roll\b", re.IGNORECASE),
    ),
    "max_rcf": NumericPattern(
        labels=("max rcf", "rcf", "g-force", "max speed"),
        bare_regex=re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+)\s*×?\s*g\b", re.IGNORECASE),
    ),
    "dimensions_mm": NumericPattern(
        labels=("dimensions", "size"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE),
    ),
}

PERCENTAGE_PATTERNS: dict[str, re.Pattern] = {
    "crude_protein": _pct_pattern("crude protein", "protein"),
    "fat_pct": _pct_pattern("fat", "crude fat"),
    "crude_fiber": _pct_pattern("crude fiber", "fiber"),
}

SINGLE_LABEL_PATTERNS: dict[str, NumericPattern] = {
    "energy_density": NumericPattern(
        labels=("energy density", "kcal/g"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*kcal\s*/\s*g\b", re.IGNORECASE),
    ),
    "shelf_life_months": NumericPattern(
        labels=("shelf life",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*months?\s*shelf\s*life\b|shelf\s*life\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*months?\b", re.IGNORECASE),
    ),
}

LABELED_TEXT_PATTERNS: dict[str, tuple[str, ...]] = {
    "labeled_sku": ("sku", "supplier sku", "item number", "catalog number", "catalog no", "cat no"),
    "labeled_mfr_sku": ("manufacturer sku", "mfr sku", "mfr part number", "manufacturer part number", "mpn"),
    "labeled_manufacturer": ("manufacturer", "mfr", "brand", "made by"),
}


def _find_labeled_number(row: Mapping[str, object], labels: tuple[str, ...]) -> tuple[str, str, str] | None:
    """Return (value, evidence, source_field) for the first labeled value whose
    label matches, extracting a leading number if present, else the raw text."""
    for source_field, label, value, evidence in iter_labeled_values(row):
        if label not in labels:
            continue
        number_match = re.search(r"-?\d+(?:\.\d+)?", value)
        extracted = number_match.group(0) if number_match else normalize_text_value(value)
        return extracted, evidence, source_field
    return None


def extract_numeric_or_labeled(row: Mapping[str, object], field_name: str, pattern_key: str, method_prefix: str) -> ExtractedSpec:
    spec = NUMERIC_PATTERNS.get(pattern_key)
    if spec is None:
        return missing_spec(field_name, f"{method_prefix}_unconfigured")

    labeled = _find_labeled_number(row, spec.labels)
    if labeled:
        value, evidence, source_field = labeled
        return ExtractedSpec(
            field_name=field_name, value=value, status="matched",
            method=f"{method_prefix}_labeled_regex", evidence=evidence,
            source_field=source_field, confidence=0.9,
        )

    if spec.bare_regex is not None:
        for source_field, label, value, evidence in iter_labeled_values(row):
            pass  # labeled values already checked above; fall through to free text
        from spec_extraction.extractors.common import iter_text_sources
        for source_field, text in iter_text_sources(row):
            match = spec.bare_regex.search(text)
            if match:
                raw = match.group(1) if match.groups() else match.group(0)
                return ExtractedSpec(
                    field_name=field_name, value=normalize_text_value(raw), status="matched",
                    method=f"{method_prefix}_regex", evidence=match.group(0).strip(),
                    source_field=source_field, confidence=0.8,
                )

    return missing_spec(field_name, f"{method_prefix}_regex")


def extract_percentage(row: Mapping[str, object], field_name: str, pattern_key: str) -> ExtractedSpec:
    pattern = PERCENTAGE_PATTERNS.get(pattern_key)
    if pattern is None:
        return missing_spec(field_name, "percentage_unconfigured")

    from spec_extraction.extractors.common import iter_text_sources
    for source_field, text in iter_text_sources(row):
        match = pattern.search(text)
        if match:
            return ExtractedSpec(
                field_name=field_name, value=match.group(1), status="matched",
                method="percentage_regex", evidence=match.group(0).strip(),
                source_field=source_field, confidence=0.85,
            )
    return missing_spec(field_name, "percentage_regex")


def extract_labeled_text(row: Mapping[str, object], field_name: str, pattern_key: str) -> ExtractedSpec:
    labels = LABELED_TEXT_PATTERNS.get(pattern_key, ())
    for source_field, label, value, evidence in iter_labeled_values(row):
        if label in labels:
            return ExtractedSpec(
                field_name=field_name, value=normalize_text_value(value), status="matched",
                method=f"{pattern_key}_labeled_regex", evidence=evidence,
                source_field=source_field, confidence=0.85,
            )
    return missing_spec(field_name, f"{pattern_key}_labeled_regex")


def extract_url(row: Mapping[str, object], field_name: str, image_only: bool) -> ExtractedSpec:
    from spec_extraction.extractors.common import iter_text_sources
    pattern = IMAGE_URL_PATTERN if image_only else URL_PATTERN
    for source_field, text in iter_text_sources(row):
        match = pattern.search(text)
        if match:
            return ExtractedSpec(
                field_name=field_name, value=match.group(0).strip(), status="matched",
                method="url_regex", evidence=match.group(0).strip(),
                source_field=source_field, confidence=0.9,
            )
    return missing_spec(field_name, "url_regex")


def dispatch_regex_field(row: Mapping[str, object], field_name: str, pattern_key: str | None) -> ExtractedSpec:
    if pattern_key is None:
        return missing_spec(field_name, "no_pattern_configured")
    if pattern_key in LABELED_TEXT_PATTERNS:
        return extract_labeled_text(row, field_name, pattern_key)
    if pattern_key in PERCENTAGE_PATTERNS:
        return extract_percentage(row, field_name, pattern_key)
    if pattern_key == "url_generic":
        return extract_url(row, field_name, image_only=False)
    if pattern_key == "image_url":
        return extract_url(row, field_name, image_only=True)
    if pattern_key in NUMERIC_PATTERNS:
        return extract_numeric_or_labeled(row, field_name, pattern_key, pattern_key)
    if pattern_key in SINGLE_LABEL_PATTERNS:
        spec = SINGLE_LABEL_PATTERNS[pattern_key]
        labeled = _find_labeled_number(row, spec.labels)
        if labeled:
            value, evidence, source_field = labeled
            return ExtractedSpec(
                field_name=field_name, value=value, status="matched",
                method=f"{pattern_key}_labeled_regex", evidence=evidence,
                source_field=source_field, confidence=0.9,
            )
        if spec.bare_regex is not None:
            from spec_extraction.extractors.common import iter_text_sources
            for source_field, text in iter_text_sources(row):
                match = spec.bare_regex.search(text)
                if match:
                    raw = next((g for g in match.groups() if g), match.group(0))
                    return ExtractedSpec(
                        field_name=field_name, value=normalize_text_value(raw), status="matched",
                        method=f"{pattern_key}_regex", evidence=match.group(0).strip(),
                        source_field=source_field, confidence=0.8,
                    )
        return missing_spec(field_name, f"{pattern_key}_regex")

    return missing_spec(field_name, f"{pattern_key}_unconfigured")
