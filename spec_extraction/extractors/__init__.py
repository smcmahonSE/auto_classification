"""Deterministic category-specific specification extractors."""

from spec_extraction.extractors.antibodies import extract_antibody_specs
from spec_extraction.extractors.animal_models import extract_animal_model_specs
from spec_extraction.extractors.biospecimens import extract_biospecimen_specs
from spec_extraction.extractors.cell_tissue_culture import extract_cell_tissue_culture_specs
from spec_extraction.extractors.chemicals import extract_chemical_specs
from spec_extraction.extractors.controls_calibrators import extract_controls_calibrators_specs
from spec_extraction.extractors.equipment_instruments import extract_equipment_instruments_specs
from spec_extraction.extractors.kits_assays import extract_kits_assays_specs
from spec_extraction.extractors.lab_supplies import extract_lab_supplies_specs
from spec_extraction.extractors.molecular_biology import extract_molecular_biology_specs
from spec_extraction.extractors.office_furniture import (
    extract_furniture_storage_specs,
    extract_general_office_supplies_specs,
)
from spec_extraction.extractors.proteins_peptides import extract_proteins_peptides_specs

__all__ = [
    "extract_antibody_specs",
    "extract_animal_model_specs",
    "extract_biospecimen_specs",
    "extract_cell_tissue_culture_specs",
    "extract_chemical_specs",
    "extract_controls_calibrators_specs",
    "extract_equipment_instruments_specs",
    "extract_kits_assays_specs",
    "extract_lab_supplies_specs",
    "extract_molecular_biology_specs",
    "extract_furniture_storage_specs",
    "extract_general_office_supplies_specs",
    "extract_proteins_peptides_specs",
]
