"""Deterministic category-specific specification extractors."""

from spec_extraction.extractors.antibodies import extract_antibody_specs
from spec_extraction.extractors.chemicals import extract_chemical_specs
from spec_extraction.extractors.lab_supplies import extract_lab_supplies_specs
from spec_extraction.extractors.molecular_biology import extract_molecular_biology_specs

__all__ = [
    "extract_antibody_specs",
    "extract_chemical_specs",
    "extract_lab_supplies_specs",
    "extract_molecular_biology_specs",
]
