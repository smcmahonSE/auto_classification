"""Type/extractor-strategy-driven dispatch for fields that are purely declarative
(no bespoke per-field code needed) — the common case for the v2 spec model.
Handles: passthrough, derived, controlled_vocabulary, regex. "override"-strategy
fields are handled one level up in extractor_dispatch.py, via each L2's
v2/overrides/ module.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from spec_extraction.config.loader import FieldDef
from spec_extraction.extractors.common import ExtractedSpec, missing_spec, vocabulary_matches
from spec_extraction.v2.regex_patterns import dispatch_regex_field
from spec_extraction.v2.vocabularies import VOCABULARIES


def _passthrough(row: Mapping[str, object], field: FieldDef) -> ExtractedSpec:
    source = field.source_column or ""
    # "A / B" means "try A, fall back to B" — used for Category (L2/L3), which
    # needs both the L2 and L3 labels concatenated, not a simple single-column copy.
    if " / " in source:
        parts = [row.get(c.strip()) for c in source.split(" / ")]
        values = [str(p).strip() for p in parts if p is not None and pd.notna(p) and str(p).strip()]
        if not values:
            return missing_spec(field.name, "passthrough")
        return ExtractedSpec(
            field_name=field.name, value=" > ".join(values), status="matched",
            method="passthrough", evidence=None, source_field=source, confidence=1.0,
        )

    value = row.get(source)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return missing_spec(field.name, "passthrough")
    return ExtractedSpec(
        field_name=field.name, value=str(value).strip(), status="matched",
        method="passthrough", evidence=None, source_field=source, confidence=1.0,
    )


def _derived(row: Mapping[str, object], field: FieldDef) -> ExtractedSpec:
    if field.name == "Pricing model":
        pricing_status = row.get("PRICING_STATUS_C")
        if pricing_status is None or pd.isna(pricing_status):
            return missing_spec(field.name, "derived_pricing_status")
        value = "Priced" if str(pricing_status).strip().upper() == "PRICED" else "Quoted"
        return ExtractedSpec(
            field_name=field.name, value=value, status="matched",
            method="derived_pricing_status", evidence=str(pricing_status),
            source_field="PRICING_STATUS_C", confidence=0.7,
        )

    if field.name == "Listing type":
        # Diagnostic guess only (LISTING_TYPE_GUESS) — no reliable listing-type
        # column exists upstream yet. Never used to gate extraction; see
        # spec_extraction/README.md and schemas/spec_model_v2/l1_universal.json.
        pricing_status = row.get("PRICING_STATUS_C")
        if pricing_status is None or pd.isna(pricing_status):
            return missing_spec(field.name, "derived_listing_type_guess")
        value = "Product" if str(pricing_status).strip().upper() == "PRICED" else "Service"
        return ExtractedSpec(
            field_name=field.name, value=value, status="matched",
            method="derived_listing_type_guess", evidence=str(pricing_status),
            source_field="PRICING_STATUS_C", confidence=0.4,
        )

    return missing_spec(field.name, "derived_unconfigured")


def _controlled_vocabulary(row: Mapping[str, object], field: FieldDef) -> ExtractedSpec:
    vocab = VOCABULARIES.get(field.vocabulary_ref or "")
    if vocab is None:
        return missing_spec(field.name, "controlled_vocabulary_unconfigured")
    return vocabulary_matches(
        row, field.name, vocab,
        method=f"{field.vocabulary_ref}_dictionary",
        multi_select=field.multi_select,
    )


def extract_generic_field(row: Mapping[str, object], field: FieldDef) -> ExtractedSpec:
    if field.extractor == "passthrough":
        return _passthrough(row, field)
    if field.extractor == "derived":
        return _derived(row, field)
    if field.extractor == "controlled_vocabulary":
        return _controlled_vocabulary(row, field)
    if field.extractor == "regex":
        return dispatch_regex_field(row, field.name, field.pattern)
    return missing_spec(field.name, f"{field.extractor}_unconfigured")
