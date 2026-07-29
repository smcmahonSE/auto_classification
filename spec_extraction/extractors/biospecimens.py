"""Deterministic extraction rules for Biospecimens."""

from __future__ import annotations

from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, vocabulary_matches
from spec_extraction.extractors.standards import species_vocabulary


SPECIMEN_TYPES = {
    "Blood": ("blood", "whole blood"),
    "Serum": ("serum",),
    "Plasma": ("plasma",),
    "Tissue": ("tissue", "tissues", "biopsy", "biopsies", "ffpe", "fresh frozen tissue"),
    "DNA": ("dna", "gdna", "genomic dna", "cfdna"),
    "RNA": ("rna", "mrna", "total rna", "small rna"),
    "PBMCs": ("pbmc", "pbmcs", "peripheral blood mononuclear"),
    "Bone Marrow Cells": ("bone marrow", "bone marrow cells"),
    "Cord Blood Cells": ("cord blood", "cord blood cells"),
    "Primary Cells": ("primary cell", "primary cells", "primary hepatocyte", "primary neuron"),
    "Urine": ("urine",),
    "CSF": ("csf", "cerebrospinal fluid"),
    "Saliva": ("saliva",),
}

DISEASE_STATES = {
    "Normal / Healthy / Control": ("normal", "healthy", "control donor", "non-diseased", "non diseased"),
    "Cancer / Oncology": ("cancer", "oncology", "tumor", "tumour", "carcinoma", "melanoma", "leukemia", "lymphoma", "sarcoma"),
    "Diabetes": ("diabetes", "diabetic", "type 1 diabetes", "type 2 diabetes", "t1d", "t2d"),
    "Cardiovascular Disease": ("cardiovascular", "hypertension", "coronary artery disease", "cad", "heart disease"),
    "Autoimmune Disease": ("autoimmune", "rheumatoid arthritis", "lupus", "sle", "ibd", "crohn", "ulcerative colitis"),
    "Infectious Disease": ("infectious disease", "covid", "sars-cov-2", "hiv", "hepatitis", "hbv", "hcv"),
    "Neurological Disease": ("neurological", "alzheimer", "parkinson", "multiple sclerosis", "ms"),
    "Respiratory Disease": ("respiratory", "asthma", "copd", "chronic obstructive pulmonary"),
    "Kidney Disease": ("kidney disease", "renal disease", "ckd"),
    "Liver Disease": ("liver disease", "hepatic disease", "cirrhosis"),
}


def extract_biospecimen_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract SME-proposed fields for Biospecimens."""
    return [
        vocabulary_matches(row, "Specimen Type", SPECIMEN_TYPES, "specimen_type_dictionary"),
        vocabulary_matches(row, "Species", species_vocabulary(), "species_dictionary", multi_select=True),
        vocabulary_matches(row, "Disease State", DISEASE_STATES, "disease_state_dictionary", multi_select=True),
    ]
