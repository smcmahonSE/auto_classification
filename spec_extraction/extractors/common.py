"""Shared helpers for deterministic specification extraction."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


TEXT_FIELDS = ("PRODUCT_NAME", "DESCRIPTION", "SPECIFICATION_ASSIGNMENTS", "SPECIFICATION_ASSIGNMENTS_C")


@dataclass(frozen=True)
class ExtractedSpec:
    """A single extracted product specification with review-friendly evidence."""

    field_name: str
    value: str | None
    status: str
    method: str
    evidence: str | None = None
    source_field: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def iter_text_sources(row: Mapping[str, object], fields: Sequence[str] = TEXT_FIELDS) -> Iterable[tuple[str, str]]:
    """Yield non-empty text fields in priority order."""
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            yield field, text


def first_regex_match(
    row: Mapping[str, object],
    field_name: str,
    pattern: re.Pattern[str],
    method: str,
    normalizer=None,
    validator=None,
) -> ExtractedSpec:
    """Return the first regex match across product text fields."""
    for source_field, text in iter_text_sources(row):
        match = pattern.search(text)
        if not match:
            continue

        raw_value = match.group(1) if match.groups() else match.group(0)
        value = normalizer(raw_value) if normalizer else raw_value.strip()
        valid = validator(value) if validator else True

        return ExtractedSpec(
            field_name=field_name,
            value=value,
            status="matched" if valid else "invalid",
            method=method,
            evidence=match.group(0).strip(),
            source_field=source_field,
            confidence=0.95 if valid else 0.25,
        )

    return missing_spec(field_name, method)


def vocabulary_matches(
    row: Mapping[str, object],
    field_name: str,
    vocabulary: Mapping[str, Sequence[str]],
    method: str,
    multi_select: bool = False,
) -> ExtractedSpec:
    """Find controlled vocabulary terms across product text fields."""
    matches: list[tuple[str, str, str]] = []

    for source_field, text in iter_text_sources(row):
        for normalized, aliases in vocabulary.items():
            for alias in aliases:
                pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    matches.append((normalized, match.group(0), source_field))
                    break

    if not matches:
        return missing_spec(field_name, method)

    seen = []
    for normalized, _, _ in matches:
        if normalized not in seen:
            seen.append(normalized)

    if not multi_select and len(seen) > 1:
        return ExtractedSpec(
            field_name=field_name,
            value="; ".join(seen),
            status="ambiguous",
            method=method,
            evidence="; ".join(evidence for _, evidence, _ in matches[:5]),
            source_field=matches[0][2],
            confidence=0.5,
        )

    value = "; ".join(seen) if multi_select else seen[0]
    return ExtractedSpec(
        field_name=field_name,
        value=value,
        status="matched",
        method=method,
        evidence="; ".join(evidence for _, evidence, _ in matches[:5]),
        source_field=matches[0][2],
        confidence=0.85,
    )


def missing_spec(field_name: str, method: str) -> ExtractedSpec:
    return ExtractedSpec(
        field_name=field_name,
        value=None,
        status="missing",
        method=method,
    )
