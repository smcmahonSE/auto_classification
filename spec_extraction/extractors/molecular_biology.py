"""Deterministic extraction rules for Molecular Biology Reagents."""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, first_regex_match, missing_spec, vocabulary_matches


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

TARGET_SPECIES = {
    "Human": ("human", "homo sapiens", "hsapiens", "h. sapiens"),
    "Mouse": ("mouse", "mice", "mus musculus", "mmusculus", "m. musculus"),
    "Rat": ("rat", "rattus norvegicus", "rnorvegicus", "r. norvegicus"),
    "Zebrafish": ("zebrafish", "danio rerio", "d. rerio"),
    "Drosophila": ("drosophila", "fruit fly", "d. melanogaster", "drosophila melanogaster"),
    "C. elegans": ("c. elegans", "caenorhabditis elegans"),
    "Yeast": ("yeast", "saccharomyces cerevisiae", "s. cerevisiae"),
    "E. coli": ("e. coli", "escherichia coli"),
    "Bovine": ("bovine", "cow", "bos taurus"),
    "Porcine": ("porcine", "pig", "swine", "sus scrofa"),
    "Chicken": ("chicken", "gallus gallus"),
    "Rabbit": ("rabbit", "oryctolagus cuniculus"),
}

TARGET_FIELD_PATTERN = re.compile(
    r"(?:^|\|\s*)(?:Target(?: Gene| Region)?|Gene(?: Symbol)?|Symbol|Locus)\s*:\s*([^|;,]+)",
    re.IGNORECASE,
)
TARGET_SPECIES_FIELD_PATTERN = re.compile(
    r"(?:^|\|\s*)(?:Target Species|Species|Organism|Target Organism)\s*:\s*([^|;,]+)",
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


def normalize_species(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .")
    for normalized, aliases in TARGET_SPECIES.items():
        for alias in aliases:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", value, re.IGNORECASE):
                return normalized
    return ""


def extract_target_species(row: Mapping[str, object]) -> ExtractedSpec:
    structured = first_regex_match(
        row=row,
        field_name="Target Species",
        pattern=TARGET_SPECIES_FIELD_PATTERN,
        method="target_species_field_regex",
        normalizer=normalize_species,
        validator=bool,
    )
    if structured.status == "matched":
        return structured

    vocab = vocabulary_matches(
        row,
        "Target Species",
        TARGET_SPECIES,
        "target_species_dictionary",
        multi_select=True,
    )
    if vocab.status == "matched":
        return vocab

    return missing_spec("Target Species", "target_species_dictionary")


def extract_molecular_biology_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract SME-required fields for Molecular Biology Reagents."""
    return [
        vocabulary_matches(row, "Sub-Type", SUB_TYPES, "mol_bio_subtype_dictionary"),
        extract_target_gene_region(row),
        extract_target_species(row),
    ]
