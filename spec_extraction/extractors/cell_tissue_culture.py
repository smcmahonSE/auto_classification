"""Deterministic extraction rules for Cells and Tissue Culture."""

from __future__ import annotations

from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, vocabulary_matches
from spec_extraction.extractors.standards import species_vocabulary


SUB_TYPES = {
    "Media": ("media", "medium", "dmem", "rpmi", "mem", "imdm", "f-12", "mccoy"),
    "Supplement": ("supplement", "sera", "serum", "fbs", "fetal bovine serum", "b-27", "n-2", "glutamax"),
    "Growth Factor": ("growth factor", "egf", "fgf", "vegf", "tgf", "bmp", "cytokine"),
    "Cryopreservation Reagent": ("cryopreservation", "freezing medium", "freezing media", "cryostor", "cell banker", "bambanker"),
    "Matrix / Substrate": ("matrix", "substrate", "matrigel", "collagen", "fibronectin", "laminin", "vitronectin", "gelatin"),
    "Dissociation Reagent": ("dissociation", "trypsin", "accutase", "accumax", "collagenase", "dispase", "trypzean", "tryple"),
    "Cell Line": ("cell line", "hela", "hek293", "cho", "jurkat", "u2os"),
    "Primary Cell": ("primary cell", "primary cells", "pbmc", "pbmcs", "hepatocyte", "endothelial cell", "fibroblast"),
    "Stem Cell": ("stem cell", "stem cells", "ipsc", "ipscs", "esc", "escs", "msc", "mscs", "hspc", "hspcs"),
    "Organoid": ("organoid", "organoids", "spheroid", "spheroids"),
    "Tissue": ("tissue", "tissues", "tissue slice", "tissue slices"),
}


def extract_cell_tissue_culture_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract first-pass fields for Cells and Tissue Culture."""
    return [
        vocabulary_matches(row, "Sub-Type", SUB_TYPES, "cell_tissue_subtype_dictionary"),
        vocabulary_matches(row, "Species", species_vocabulary(), "species_dictionary", multi_select=True),
    ]
