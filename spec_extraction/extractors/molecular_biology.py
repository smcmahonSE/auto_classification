"""Deterministic extraction rules for Molecular Biology Reagents."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, first_regex_match, vocabulary_matches


SUB_TYPES = {
    "Primer": ("primer", "primers", "oligo", "oligonucleotide"),
    "Probe": ("probe", "probes", "taqman"),
    "siRNA": ("sirna", "small interfering rna"),
    "Enzyme": (
        "polymerase",
        "ligase",
        "reverse transcriptase",
        "restriction enzyme",
        "endonuclease",
        "nuclease",
        "kinase",
    ),
    "Nucleotide": ("nucleotide", "nucleotides", "dntp", "dntps", "rntp", "rntps"),
    "gRNA": ("grna", "guide rna", "sgrna"),
    "Cas Nuclease": ("cas9", "cas12", "cas nuclease", "crispr nuclease"),
    "shRNA": ("shrna", "short hairpin rna"),
    "miRNA": ("mirna", "microrna", "micro rna"),
}

TARGET_FIELD_PATTERN = re.compile(
    r"(?:^|\|\s*)(?:Target(?: Gene| Region)?|Gene(?: Symbol)?|Symbol|Locus)\s*:\s*([^|;,]+)",
    re.IGNORECASE,
)
TARGET_NAME_PATTERNS = (
    re.compile(r"\b(?:siRNA|shRNA|miRNA|gRNA|sgRNA)\s+(?:for|against|targeting)\s+([A-Za-z0-9_.-]+)", re.IGNORECASE),
    re.compile(r"\b(?:human|mouse|rat)\s+([A-Za-z0-9_.-]+)\s+(?:qPCR|PCR)?\s*(?:primer|probe)\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9_.-]+)\s+(?:siRNA|shRNA|miRNA|gRNA|sgRNA|primer|probe)\b", re.IGNORECASE),
)


def normalize_target(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" -_/")
    value = re.sub(r"\s+\([^)]*\)$", "", value).strip()
    if value.upper() in {"PCR", "QPCR", "RT-PCR", "RT QPCR"}:
        return ""
    return value


def extract_target_gene_region(row: Mapping[str, object]) -> ExtractedSpec:
    structured = first_regex_match(
        row=row,
        field_name="Target Gene / Region",
        pattern=TARGET_FIELD_PATTERN,
        method="target_gene_field_regex",
        normalizer=normalize_target,
    )
    if structured.status == "matched" and structured.value:
        return structured

    for pattern in TARGET_NAME_PATTERNS:
        name_match = first_regex_match(
            row=row,
            field_name="Target Gene / Region",
            pattern=pattern,
            method="target_gene_name_regex",
            normalizer=normalize_target,
        )
        if name_match.status == "matched" and name_match.value:
            return name_match

    return structured


def extract_molecular_biology_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract SME-required fields for Molecular Biology Reagents."""
    return [
        vocabulary_matches(row, "Sub-Type", SUB_TYPES, "mol_bio_subtype_dictionary"),
        extract_target_gene_region(row),
    ]
