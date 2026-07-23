"""Deterministic category-specific specification extractors."""

from spec_extraction.extractors.antibodies import extract_antibody_specs
from spec_extraction.extractors.chemicals import extract_chemical_specs
from spec_extraction.extractors.kits_assays import extract_kits_assays_specs
from spec_extraction.extractors.lab_supplies import extract_lab_supplies_specs
from spec_extraction.extractors.molecular_biology import extract_molecular_biology_specs
from spec_extraction.extractors.proteins_peptides import extract_proteins_peptides_specs

__all__ = [
    "extract_antibody_specs",
    "extract_chemical_specs",
    "extract_kits_assays_specs",
    "extract_lab_supplies_specs",
    "extract_molecular_biology_specs",
    "extract_proteins_peptides_specs",
]
