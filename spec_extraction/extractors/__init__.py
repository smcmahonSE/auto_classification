"""Deterministic category-specific specification extractors."""

from spec_extraction.extractors.antibodies import extract_antibody_specs
from spec_extraction.extractors.chemicals import extract_chemical_specs

__all__ = ["extract_antibody_specs", "extract_chemical_specs"]
