"""Top-level per-row extraction dispatch for the v2 spec model: given an L2 id,
L3 node id, the effective field set (from config.loader.get_effective_field_set),
and a source row, returns the list of ExtractedSpec results in field order.

For each field: check that L2's override registry first (bespoke logic that can't
be expressed as pure config), fall back to the generic type-driven dispatcher
otherwise. This is the seam that lets most fields be metadata-driven while a
handful of genuinely complex ones (e.g. Capacity / Volume / Size) get real
hand-written logic.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Mapping

from spec_extraction.config.loader import FieldDef
from spec_extraction.extractors.common import ExtractedSpec
from spec_extraction.v2.generic_field_extractors import extract_generic_field

# l2_id -> {override_key: callable(row, field_name) -> ExtractedSpec}
# New L2s all share the common overrides module (cas_number, purity, target_gene_region,
# antibody_target, animal_strain, equipment_type, capacity_volume_size, etc.) rather than
# each needing its own thin module — see v2/overrides/common.py's FIELD_OVERRIDES dict.
_OVERRIDE_MODULES: dict[str, str] = {
    "lab_supplies_consumables": "spec_extraction.v2.overrides.lab_supplies_consumables",
    "molecular_biology_reagents": "spec_extraction.v2.overrides.common",
    "chemicals_solvents": "spec_extraction.v2.overrides.common",
    "antibodies": "spec_extraction.v2.overrides.common",
    "proteins_peptides": "spec_extraction.v2.overrides.common",
    "cell_tissue_culture": "spec_extraction.v2.overrides.common",
    "animal_biospecimens": "spec_extraction.v2.overrides.common",
    "human_biospecimens": "spec_extraction.v2.overrides.common",
    "animal_models": "spec_extraction.v2.overrides.common",
    "equipment_instruments_parts": "spec_extraction.v2.overrides.common",
    "histology_pathology": "spec_extraction.v2.overrides.common",
    "analytical_reference_standards": "spec_extraction.v2.overrides.common",
    "dmpk_adme": "spec_extraction.v2.overrides.common",
    "bioanalysis": "spec_extraction.v2.overrides.common",
    "cmc_analytical_qc": "spec_extraction.v2.overrides.common",
    "regulatory_affairs": "spec_extraction.v2.overrides.common",
    "heor_rwe_market_access": "spec_extraction.v2.overrides.common",
    "phase_iv_post_marketing_studies": "spec_extraction.v2.overrides.common",
    "toxicology_safety_assessment": "spec_extraction.v2.overrides.common",
    "pharmacovigilance_drug_safety": "spec_extraction.v2.overrides.common",
}


@lru_cache(maxsize=None)
def _load_overrides(l2_id: str) -> dict[str, Callable[[Mapping[str, object], str], ExtractedSpec]]:
    module_path = _OVERRIDE_MODULES.get(l2_id)
    if module_path is None:
        return {}
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, "FIELD_OVERRIDES", {})


def extract_row(l2_id: str, l3_id: str, fields: list[FieldDef], row: Mapping[str, object]) -> list[ExtractedSpec]:
    overrides = _load_overrides(l2_id)
    results: list[ExtractedSpec] = []
    for field in fields:
        if field.extractor == "override" and field.override_key in overrides:
            results.append(overrides[field.override_key](row, field.name))
        else:
            results.append(extract_generic_field(row, field))
    return results
