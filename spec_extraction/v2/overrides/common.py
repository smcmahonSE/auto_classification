"""Cross-category override logic shared by multiple v2 L2 configs — ported from the
old flat extractors/ modules (chemicals.py, antibodies.py, molecular_biology.py,
animal_models.py, equipment_instruments.py) where real working regex/vocab logic
already existed, generalized to take an explicit field_name parameter so each L2's
config can reuse the same engine under whatever field label the new spec doc uses
(e.g. "Target / Antigen" vs the old "Target / Specificity").
"""

from __future__ import annotations

import re
from typing import Mapping

from spec_extraction.extractors.common import (
    ExtractedSpec,
    first_regex_match,
    iter_text_sources,
    missing_spec,
    vocabulary_matches,
)
from spec_extraction.extractors.lab_supplies import find_capacity_volume_size, size_spec
from spec_extraction.extractors.standards import SPECIES
from spec_extraction.v2.vocabularies import VOCABULARIES


def extract_capacity_volume_size(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    """Amount+unit parser (e.g. '5 mL', '10 mL' volumes) — same engine Lab Supplies
    uses for 'Capacity / Volume / Size', reused here for any new-category field that's
    structurally the same (e.g. Blood & Biofluids 'Volume', Peptides 'Scale')."""
    return size_spec(find_capacity_volume_size(row), field_name, "value")

# ── CAS Number (checksum-validated) ───────────────────────────────────────────
CAS_PATTERN = re.compile(r"\b(\d{2,7}-\d{2}-\d)\b")


def is_valid_cas(cas_number: str) -> bool:
    parts = cas_number.split("-")
    if len(parts) != 3:
        return False
    body = "".join(parts[:2])
    check_digit = int(parts[2])
    checksum = sum(int(digit) * multiplier for multiplier, digit in enumerate(reversed(body), start=1))
    return checksum % 10 == check_digit


def extract_cas_number(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    return first_regex_match(
        row=row, field_name=field_name, pattern=CAS_PATTERN,
        method="cas_regex_checksum", validator=is_valid_cas,
    )


# ── Purity (percent regex, falls back to grade vocabulary) ───────────────────
PURITY_PERCENT_PATTERN = re.compile(
    r"((?:>=|>|~|approx\.?\s*)?\s*\d{1,3}(?:\.\d+)?\+?\s*%)(?:\s*(?:pure|purity|assay)?)",
    re.IGNORECASE,
)


def normalize_purity(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def extract_purity(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    percent_match = first_regex_match(
        row=row, field_name=field_name, pattern=PURITY_PERCENT_PATTERN,
        method="purity_percent_regex", normalizer=normalize_purity,
    )
    if percent_match.status == "matched":
        return percent_match
    return vocabulary_matches(row, field_name, VOCABULARIES["purity_grade"], "purity_grade_dictionary")


# ── Molecular Weight / Molecular Formula (generic labeled-value lookups) ─────
MOLECULAR_WEIGHT_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:g/mol|da|kda)\b", re.IGNORECASE)
MOLECULAR_FORMULA_PATTERN = re.compile(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*){2,})\b")


def extract_molecular_weight(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    return first_regex_match(row, field_name, MOLECULAR_WEIGHT_PATTERN, "molecular_weight_regex")


def extract_molecular_formula(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    return first_regex_match(row, field_name, MOLECULAR_FORMULA_PATTERN, "molecular_formula_regex")


# ── Target Gene / Region + Target Species (Molecular Biology) ────────────────
TARGET_GENE_FIELD_PATTERN = re.compile(
    r"(?:^|\|\s*)(?:Target(?: Gene| Region)?|Gene(?: Symbol)?|Symbol|Locus)\s*:\s*([^|;,]+)",
    re.IGNORECASE,
)
TARGET_GENE_NAME_PATTERNS = (
    re.compile(r"\b(?:siRNA|shRNA|miRNA|gRNA|sgRNA)\s+(?:for|against|targeting)\s+([A-Za-z0-9_.-]+)", re.IGNORECASE),
    re.compile(r"\b(?:human|mouse|rat)\s+([A-Za-z0-9_.-]+)\s+(?:qPCR|PCR)?\s*(?:primer|probe)\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9_.-]+)\s+(?:siRNA|shRNA|miRNA|gRNA|sgRNA|primer|probe)\b", re.IGNORECASE),
)
TARGET_SPECIES_FIELD_PATTERN = re.compile(
    r"(?:^|\|\s*)(?:Target Species|Species|Organism|Target Organism)\s*:\s*([^|;,]+)",
    re.IGNORECASE,
)


def normalize_gene_target(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" -_/")
    value = re.sub(r"\s+\([^)]*\)$", "", value).strip()
    if value.upper() in {"PCR", "QPCR", "RT-PCR", "RT QPCR"}:
        return ""
    return value


def extract_target_gene_region(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    structured = first_regex_match(row, field_name, TARGET_GENE_FIELD_PATTERN, "target_gene_field_regex", normalizer=normalize_gene_target)
    if structured.status == "matched" and structured.value:
        return structured
    for pattern in TARGET_GENE_NAME_PATTERNS:
        name_match = first_regex_match(row, field_name, pattern, "target_gene_name_regex", normalizer=normalize_gene_target)
        if name_match.status == "matched" and name_match.value:
            return name_match
    return structured


def normalize_species_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .")
    for normalized, aliases in SPECIES.items():
        for alias in aliases:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", value, re.IGNORECASE):
                return normalized
    return ""


def extract_target_species(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    structured = first_regex_match(row, field_name, TARGET_SPECIES_FIELD_PATTERN, "target_species_field_regex", normalizer=normalize_species_text, validator=bool)
    if structured.status == "matched":
        return structured
    vocab = vocabulary_matches(row, field_name, SPECIES, "target_species_dictionary", multi_select=True)
    if vocab.status == "matched":
        return vocab
    return missing_spec(field_name, "target_species_dictionary")


# ── Antibody Target / Antigen + Host Species ──────────────────────────────────
ANTIBODY_HOST_PATTERN = re.compile(
    r"\b(mouse|rabbit|goat|rat|donkey|chicken|sheep)\b(?:\s+(?:monoclonal|polyclonal|recombinant))?\s+(?:anti[-\s])",
    re.IGNORECASE,
)
ANTIBODY_TARGET_PATTERN = re.compile(
    r"\banti[-\s]([A-Za-z0-9][A-Za-z0-9./_+\- ]{1,40}?)(?:\s+(?:antibody|mab|pab|clone|fitc|pe|apc|hrp|biotin)|[,;()]|$)",
    re.IGNORECASE,
)
ANTIBODY_TARGET_FIELD_PATTERN = re.compile(
    r"(?:^|\|\s*)(?:Target(?: Name)?|Symbol)\s*:\s*([^|;,]+)",
    re.IGNORECASE,
)
ANTIBODY_NAME_TARGET_PATTERN = re.compile(
    r"^\s*(?:Anti[-\s])?(.+?)\s+(?:(?:Mouse|Rabbit|Goat|Rat|Human)\s+)?"
    r"(?:(?:Recombinant|Mouse|Rabbit|Goat|Rat)\s+)?"
    r"(?:(?:Monoclonal|Polyclonal)\s+)?(?:Antibody|mAb|pAb)\b",
    re.IGNORECASE,
)


def normalize_antibody_target(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" -_/")
    value = re.sub(r"\s+(?:FITC|PE|APC|HRP|Biotin)\s+conjugated$", "", value, flags=re.IGNORECASE)
    return value.strip()


def _target_match_from_text(row, source_field, pattern, method, field_name) -> ExtractedSpec | None:
    raw_text = row.get(source_field)
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    if not text or text.lower() == "nan":
        return None
    match = pattern.search(text)
    if not match:
        return None
    value = normalize_antibody_target(match.group(1))
    if not value:
        return None
    return ExtractedSpec(field_name=field_name, value=value, status="matched", method=method, evidence=match.group(0).strip(), source_field=source_field, confidence=0.85)


def extract_antibody_target(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    for source_field, pattern, method in (
        ("DESCRIPTION", ANTIBODY_TARGET_FIELD_PATTERN, "target_field_regex"),
        ("PRODUCT_NAME", ANTIBODY_TARGET_PATTERN, "anti_target_regex"),
        ("PRODUCT_NAME", ANTIBODY_NAME_TARGET_PATTERN, "name_target_regex"),
        ("DESCRIPTION", ANTIBODY_TARGET_PATTERN, "anti_target_regex"),
    ):
        target = _target_match_from_text(row, source_field, pattern, method, field_name)
        if target is not None:
            return target
    target = first_regex_match(row, field_name, ANTIBODY_TARGET_PATTERN, "anti_target_regex", normalizer=normalize_antibody_target)
    if target.status == "matched" and target.value:
        return target
    return missing_spec(field_name, "anti_target_regex")


def extract_antibody_host_species(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    host = first_regex_match(row, field_name, ANTIBODY_HOST_PATTERN, "host_before_anti_regex", normalizer=lambda v: v.title())
    if host.status == "matched":
        return host
    return vocabulary_matches(row, field_name, VOCABULARIES["host_species_antibody"], "host_species_dictionary")


# ── Animal strain ─────────────────────────────────────────────────────────────
ANIMAL_STRAIN_PATTERN = re.compile(
    r"\b(C57BL/6\w*|BALB/c\w*|DBA/2\w*|FVB/N\w*|129S\w*|CD-1|Sprague[-\s]?Dawley|Wistar|Long[-\s]?Evans|"
    r"NOD(?:[-/\s]?SCID)?|NSG|SCID|nude|Rag[12]|db/db|ob/ob|5xFAD|APP/PS1|3xTg-AD|SOD1|"
    r"cynomolgus|rhesus|marmoset|New Zealand White|Hartley|Dunkin[-\s]?Hartley|Yucatan|Göttingen|Gottingen|Beagle)\b",
    re.IGNORECASE,
)


def normalize_strain(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .")


def extract_animal_strain(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    return first_regex_match(row, field_name, ANIMAL_STRAIN_PATTERN, "strain_regex", normalizer=normalize_strain)


# ── Equipment type (free-text, broad vocabulary) ──────────────────────────────
EQUIPMENT_TYPES = {
    "Centrifuge": ("centrifuge", "microcentrifuge", "ultracentrifuge"),
    "Pipettor": ("pipettor", "pipette", "multichannel pipette", "electronic pipette"),
    "Liquid Handler": ("liquid handler", "liquid handling", "robotic pipetting", "plate washer"),
    "Microscope": ("microscope", "confocal", "fluorescence microscope"),
    "Spectrophotometer": ("spectrophotometer", "uv-vis", "nanodrop"),
    "Plate Reader": ("plate reader", "microplate reader"),
    "Flow Cytometer": ("flow cytometer", "facs", "cell sorter"),
    "PCR Machine": ("pcr machine", "thermal cycler", "qpcr", "real-time pcr", "digital pcr"),
    "Electrophoresis System": ("electrophoresis", "gel system"),
    "Incubator": ("incubator", "co2 incubator"),
    "Shaker": ("shaker", "orbital shaker", "tube rotator"),
    "Vortexer": ("vortex", "vortexer"),
    "Homogenizer": ("homogenizer", "homogeniser", "bead mill"),
    "Sonicator": ("sonicator", "ultrasonic"),
    "Freeze Dryer": ("freeze dryer", "lyophilizer"),
    "Water Bath": ("water bath", "circulator"),
    "Heating Block": ("heating block", "heat block", "dry bath"),
    "Stirrer": ("magnetic stirrer", "stirrer", "hot plate"),
    "Cell Counter": ("cell counter", "hemocytometer", "vi-cell", "countess"),
    "Autoclave": ("autoclave", "sterilizer"),
    "Biosafety Cabinet": ("biosafety cabinet", "bsc", "laminar flow hood"),
    "Fume Hood": ("fume hood",),
    "Chromatography System": ("chromatography", "hplc", "uhplc", "fplc"),
    "Mass Spectrometer": ("mass spectrometer", "mass spec", "lc-ms", "gc-ms"),
    "Balance": ("balance", "analytical balance"),
    "pH Meter": ("ph meter",),
    "Sequencer": ("sequencer", "ngs platform", "sanger sequencer"),
    "Bioreactor": ("bioreactor", "single-use bioreactor"),
    "Liquid Handler Automation": ("automated workstation", "robotic workstation"),
}


def extract_equipment_type(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    for source_field, text in iter_text_sources(row):
        for normalized, aliases in EQUIPMENT_TYPES.items():
            for alias in aliases:
                pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    return ExtractedSpec(field_name=field_name, value=normalized, status="matched", method="equipment_type_dictionary", evidence=match.group(0), source_field=source_field, confidence=0.8)
    return missing_spec(field_name, "equipment_type_dictionary")


# Registry the v2 dispatcher looks up by override_key (see schemas/spec_model_v2/l2/*.json).
# Shared across every new L2 added in this pass — each config's fields reference these
# keys directly rather than each L2 needing its own overrides module.
FIELD_OVERRIDES = {
    "cas_number": extract_cas_number,
    "purity": extract_purity,
    "molecular_weight": extract_molecular_weight,
    "molecular_formula": extract_molecular_formula,
    "target_gene_region": extract_target_gene_region,
    "target_species": extract_target_species,
    "antibody_target": extract_antibody_target,
    "antibody_host_species": extract_antibody_host_species,
    "animal_strain": extract_animal_strain,
    "equipment_type": extract_equipment_type,
    "capacity_volume_size": extract_capacity_volume_size,
}
