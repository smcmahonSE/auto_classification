"""Generic regex-based extractors for the v2 spec model's simple numeric/text fields.

These are intentionally lighter-weight than the bespoke per-field logic in
extractors/lab_supplies.py (ported instead into v2/overrides/ for the fields that
need it, like Capacity / Volume / Size). Most L1-universal fields and many L3 fields
just need "find a labeled value, else look for a bare number+unit in free text" —
this module implements that once, driven by a small per-pattern-key config, rather
than writing ~30 near-identical bespoke regexes.

Coverage is expected to be uneven across these fields (some, like Category UNSPSC,
have no real source signal at all and will show as mostly "missing" in the coverage
report by design — that's honest, not a bug).
"""

from __future__ import annotations

import re
from typing import Mapping, NamedTuple

from spec_extraction.extractors.common import ExtractedSpec, missing_spec
from spec_extraction.extractors.lab_supplies import iter_labeled_values, normalize_text_value

URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
IMAGE_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+\.(?:png|jpe?g|gif|webp|svg)\b", re.IGNORECASE)


class NumericPattern(NamedTuple):
    labels: tuple[str, ...]           # labeled-value keys to look for, e.g. ("lead time", "lead time days")
    bare_regex: re.Pattern | None     # fallback free-text pattern if no labeled value found


def _pct_pattern(*label_words: str) -> re.Pattern:
    words = "|".join(re.escape(w) for w in label_words)
    return re.compile(rf"\b(?:{words})\D{{0,10}}?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


NUMERIC_PATTERNS: dict[str, NumericPattern] = {
    "lead_time_days": NumericPattern(
        labels=("lead time", "lead time days", "lead-time"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:business\s+)?days?\s+lead\s*time\b", re.IGNORECASE),
    ),
    "unit_count": NumericPattern(labels=("unit", "units per pack", "count per unit"), bare_regex=None),
    "quantity": NumericPattern(labels=("quantity", "qty"), bare_regex=None),
    "inventory": NumericPattern(labels=("inventory", "in stock", "stock", "units available"), bare_regex=None),
    "ship_from": NumericPattern(labels=("ship from", "ships from", "origin"), bare_regex=None),
    "neck_finish": NumericPattern(
        labels=("neck finish", "neck", "thread"),
        bare_regex=re.compile(r"\bGL\s?\d{2,3}\b|\b\d{2}-\d{3}\b", re.IGNORECASE),
    ),
    "temperature_range": NumericPattern(
        labels=("temperature range", "temperature", "storage temperature"),
        bare_regex=re.compile(
            r"-?\d+(?:\.\d+)?\s*(?:to|-|–)\s*-?\d+(?:\.\d+)?\s*°?C\b"
            r"|-?\d+(?:\.\d+)?\s*°C\b",
            re.IGNORECASE,
        ),
    ),
    "compatibility": NumericPattern(labels=("compatibility", "compatible with"), bare_regex=None),
    "well_number": NumericPattern(
        labels=("well number", "wells", "well count"),
        bare_regex=re.compile(r"\b(6|12|24|48|96|384|1536)\s*-?\s*wells?\b", re.IGNORECASE),
    ),
    "pore_size": NumericPattern(
        labels=("pore size",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:µm|um|micron)\b", re.IGNORECASE),
    ),
    "needle_gauge_length": NumericPattern(
        labels=("needle gauge", "gauge"),
        bare_regex=re.compile(r"\b(\d{1,2}\s*G)\b", re.IGNORECASE),
    ),
    "inner_diameter": NumericPattern(
        labels=("inner diameter", "id", "i.d."),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|in|inch(?:es)?)\s*(?:id|inner diameter)\b", re.IGNORECASE),
    ),
    "outer_diameter": NumericPattern(
        labels=("outer diameter", "od", "o.d."),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|in|inch(?:es)?)\s*(?:od|outer diameter)\b", re.IGNORECASE),
    ),
    "wall_thickness": NumericPattern(labels=("wall thickness",), bare_regex=None),
    "length_ft_m": NumericPattern(
        labels=("length",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(ft|feet|m|meters?)\b", re.IGNORECASE),
    ),
    "durometer": NumericPattern(
        labels=("durometer", "shore a", "shore hardness"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*shore\s*a\b", re.IGNORECASE),
    ),
    "max_pressure": NumericPattern(
        labels=("max pressure", "maximum pressure"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*psig?\b", re.IGNORECASE),
    ),
    "active_ingredient": NumericPattern(labels=("active ingredient", "active ingredients"), bare_regex=None),
    "width": NumericPattern(
        labels=("width",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|in|inch(?:es)?)\s*wide\b", re.IGNORECASE),
    ),
    "roll_length": NumericPattern(
        labels=("roll length",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*m\s*roll\b", re.IGNORECASE),
    ),
    "max_rcf": NumericPattern(
        labels=("max rcf", "rcf", "g-force", "max speed"),
        bare_regex=re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+)\s*×?\s*g\b", re.IGNORECASE),
    ),
    "dimensions_mm": NumericPattern(
        labels=("dimensions", "size"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE),
    ),
    "turnaround": NumericPattern(
        labels=("turnaround", "turnaround time", "tat"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:business\s+)?(?:days?|weeks?)\s*turnaround\b", re.IGNORECASE),
    ),
    "molecular_weight": NumericPattern(
        labels=("molecular weight", "mw"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:g/mol|da|kda)\b", re.IGNORECASE),
    ),
    "ph": NumericPattern(labels=("ph",), bare_regex=re.compile(r"\bph\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)),
    "length_nt_aa_bp": NumericPattern(
        labels=("length",),
        bare_regex=re.compile(r"\b(\d+)\s*(?:nt|aa|bp)\b", re.IGNORECASE),
    ),
    "viability_pct": NumericPattern(
        labels=("viability",),
        bare_regex=re.compile(r"\b(?:>=|≥)?\s*(\d+(?:\.\d+)?)\s*%\s*(?:viability|viable|post-thaw)\b", re.IGNORECASE),
    ),
    "cell_count": NumericPattern(labels=("cell count", "cells per vial"), bare_regex=None),
    "donor_age": NumericPattern(
        labels=("donor age", "age", "animal age", "age at collection"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:years?|yrs?|weeks?|months?)\s*(?:old)?\b", re.IGNORECASE),
    ),
    "passage_number": NumericPattern(
        labels=("passage", "passage number"),
        bare_regex=re.compile(r"\bP(\d{1,2})\b"),
    ),
    "yield_amount": NumericPattern(labels=("yield",), bare_regex=None),
    "titer": NumericPattern(
        labels=("titer",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?(?:e\d+)?)\s*(?:gc/ml|tu/ml)\b", re.IGNORECASE),
    ),
    "read_depth": NumericPattern(
        labels=("read depth", "reads", "coverage", "million reads"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:(?:×|x)\s*coverage|m(?:illion)?\s*reads)\b", re.IGNORECASE),
    ),
    "full_cure_time": NumericPattern(
        labels=("full cure time", "cure time", "cures in"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*h(?:ours?)?\b(?=.{0,20}cure|.{0,20}\bat\b)", re.IGNORECASE),
    ),
    "work_life": NumericPattern(
        labels=("work life", "pot life", "open time"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*min(?:utes?)?\b", re.IGNORECASE),
    ),
    "viscosity_cp": NumericPattern(
        labels=("viscosity",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*c[Pp]\b"),
    ),
    "operating_temp_range": NumericPattern(
        labels=("operating temperature", "service temperature"),
        bare_regex=re.compile(r"-?\d+(?:\.\d+)?\s*(?:to|-|–)\s*-?\d+(?:\.\d+)?\s*°?C\b", re.IGNORECASE),
    ),
    "shear_strength": NumericPattern(
        labels=("shear strength",),
        bare_regex=re.compile(r"\b([\d,]+(?:\.\d+)?)\s*psi\b", re.IGNORECASE),
    ),
    "cross_linkage_pct": NumericPattern(
        labels=("cross-linkage", "cross linkage", "dvb"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*dvb\b", re.IGNORECASE),
    ),
    "loading_capacity": NumericPattern(
        labels=("loading", "capacity", "ion exchange capacity"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:meq/ml|mmol/g)\b", re.IGNORECASE),
    ),
    "moisture_content_pct": NumericPattern(
        labels=("moisture content", "moisture"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*moisture\b", re.IGNORECASE),
    ),
    "water_content_pct": NumericPattern(
        labels=("water content",),
        bare_regex=re.compile(r"\b(?:<|≤)?\s*(\d+(?:\.\d+)?)\s*%\s*water\b", re.IGNORECASE),
    ),
    "specific_activity": NumericPattern(
        labels=("specific activity",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*mci\s*/\s*mmol\b", re.IGNORECASE),
    ),
    "concentration_molarity": NumericPattern(
        labels=("concentration", "molarity"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:M|×|x)\b"),
    ),
    "library_collection_size": NumericPattern(
        labels=("library size", "collection size", "compounds"),
        bare_regex=re.compile(r"\b([\d,]+)\s*k?\s*(?:compounds|building blocks)\b", re.IGNORECASE),
    ),
    "endotoxin_level": NumericPattern(
        labels=("endotoxin", "endotoxin level"),
        bare_regex=re.compile(r"\b(?:<|≤)?\s*(\d+(?:\.\d+)?)\s*eu\s*/\s*(?:µg|ug|mg)\b", re.IGNORECASE),
    ),
    "specific_activity_protein": NumericPattern(
        labels=("specific activity",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*units?\s*/\s*mg\b", re.IGNORECASE),
    ),
    "bioactivity_ed50": NumericPattern(
        labels=("ed50", "bioactivity"),
        bare_regex=re.compile(r"\bed50\D{0,10}?(\d+(?:\.\d+)?)\s*ng\s*/\s*ml\b", re.IGNORECASE),
    ),
    "peptide_scale": NumericPattern(labels=("scale",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|g|kg)\b", re.IGNORECASE)),
    "endotoxin_eu_ml": NumericPattern(
        labels=("endotoxin",),
        bare_regex=re.compile(r"\b(?:<|≤)?\s*(\d+(?:\.\d+)?)\s*eu\s*/\s*ml\b", re.IGNORECASE),
    ),
    "hemoglobin_mg_dl": NumericPattern(
        labels=("hemoglobin",),
        bare_regex=re.compile(r"\b(?:<|≤)?\s*(\d+(?:\.\d+)?)\s*mg\s*/\s*dl\b", re.IGNORECASE),
    ),
    "cell_number_per_vial": NumericPattern(labels=("cell number", "cells per vial"), bare_regex=None),
    "population_doublings": NumericPattern(
        labels=("population doublings", "pd"),
        bare_regex=re.compile(r"\b(?:>|≥)?\s*(\d+(?:\.\d+)?)\s*pd\b", re.IGNORECASE),
    ),
    "sample_size_amount": NumericPattern(
        labels=("sample size", "amount"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:mg|sections?)\b", re.IGNORECASE),
    ),
    "volume_ml": NumericPattern(
        labels=("volume",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*ml\b", re.IGNORECASE),
    ),
    "rin": NumericPattern(
        labels=("rin", "rna integrity"),
        bare_regex=re.compile(r"\brin\D{0,5}?(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    ),
    "tumor_content_pct": NumericPattern(
        labels=("tumor content", "tumor %", "tumor cellularity", "tumor nuclei", "cancer %"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*tumor\b", re.IGNORECASE),
    ),
    "necrosis_pct": NumericPattern(
        labels=("necrosis",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*necrosis\b", re.IGNORECASE),
    ),
    "fresh_delivery_time_hours": NumericPattern(labels=("fresh delivery time",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*hours?\b", re.IGNORECASE)),
    "time_to_start_days": NumericPattern(labels=("time to start",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:-\s*\d+(?:\.\d+)?\s*)?days?\b", re.IGNORECASE)),
    "collection_to_processing_hours": NumericPattern(labels=("collection-to-processing", "collection to processing"), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*hours?\b", re.IGNORECASE)),
    "retrieval_turnaround_hours": NumericPattern(labels=("retrieval turnaround",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*hours?\b", re.IGNORECASE)),
    "accessioning_turnaround_days": NumericPattern(labels=("accessioning turnaround",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*days?\b", re.IGNORECASE)),
    "age_weight_range": NumericPattern(
        labels=("age", "weight", "age / weight range"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:weeks?|g|grams?)\b", re.IGNORECASE),
    ),
    "adult_age_months": NumericPattern(labels=("adult age",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*months?\b", re.IGNORECASE)),
    "age_weight_years_kg": NumericPattern(
        labels=("age / weight", "age", "weight"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:years?|kg)\b", re.IGNORECASE),
    ),
    "times_outcrossed": NumericPattern(labels=("times outcrossed", "outcrosses"), bare_regex=re.compile(r"\bx(\d+)\b", re.IGNORECASE)),
    "max_pressure_bar": NumericPattern(
        labels=("max pressure",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*bar\b", re.IGNORECASE),
    ),
    "flow_rate_range": NumericPattern(labels=("flow rate",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*ml\s*/\s*min\b", re.IGNORECASE)),
    "sample_capacity_vials": NumericPattern(labels=("sample capacity",), bare_regex=re.compile(r"\b(\d+)\s*vials?\b", re.IGNORECASE)),
    "mass_range_mz": NumericPattern(labels=("mass range",), bare_regex=re.compile(r"\b(\d+(?:,\d+)?)\s*m/z\b", re.IGNORECASE)),
    "scan_speed_mrm": NumericPattern(labels=("scan speed",), bare_regex=re.compile(r"\b(\d+(?:,\d+)?)\s*mrm\s*/\s*sec\b", re.IGNORECASE)),
    "magnification_x": NumericPattern(labels=("magnification",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*[xX]\b")),
    "fluorescence_channels": NumericPattern(labels=("fluorescence channels", "channels"), bare_regex=re.compile(r"\b(\d+)\s*\+?\s*channels?\b", re.IGNORECASE)),
    "volume_range_ul": NumericPattern(labels=("volume range",), bare_regex=re.compile(r"\b(\d+(?:,\d+)?)\s*(?:µl|ul)\b", re.IGNORECASE)),
    "vessel_size_l": NumericPattern(labels=("vessel size", "working volume", "nominal volume", "bag size", "capacity"), bare_regex=re.compile(r"\b(\d+(?:,\d+)?(?:\.\d+)?)\s*L\b")),
    "noise_level_db": NumericPattern(labels=("noise level",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*db", re.IGNORECASE)),
    "dimensions_wdh_cm": NumericPattern(
        labels=("dimensions",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*cm\b", re.IGNORECASE),
    ),
    "weight_kg": NumericPattern(labels=("weight",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*kg\b", re.IGNORECASE)),
    "max_read_length_bp": NumericPattern(labels=("max read length", "read length"), bare_regex=re.compile(r"\b(\d+)\s*bp\b", re.IGNORECASE)),
    "output_per_flowcell_gb": NumericPattern(labels=("output per flow cell", "output"), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*gb\b", re.IGNORECASE)),
    "reads_per_run": NumericPattern(labels=("reads per run",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*billion\b", re.IGNORECASE)),
    "run_time_hours": NumericPattern(labels=("run time",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:hr|hours?)\b", re.IGNORECASE)),
    "samples_per_flowcell": NumericPattern(labels=("samples per flow cell",), bare_regex=None),
    "parameters_detectors": NumericPattern(labels=("parameters", "detectors"), bare_regex=re.compile(r"\b(\d+)\s*parameters?\b", re.IGNORECASE)),
    "acquisition_rate_events_sec": NumericPattern(labels=("acquisition rate",), bare_regex=re.compile(r"\b([\d,]+)\s*events?\s*/\s*sec\b", re.IGNORECASE)),
    "section_thickness_nm": NumericPattern(labels=("section thickness",), bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*nm\b", re.IGNORECASE)),
    "sections_per_slide": NumericPattern(labels=("sections per slide",), bare_regex=None),
    "capacity_slides_per_month": NumericPattern(labels=("capacity",), bare_regex=re.compile(r"\b([\d,]+)\s*slides?\s*/\s*month\b", re.IGNORECASE)),

    # --- Analytical & Reference Standards ---
    "concentration_ug_mg_per_ml": NumericPattern(labels=("concentration",), bare_regex=None),
    "minimum_sample_size": NumericPattern(labels=("minimum sample size", "sample size"), bare_regex=None),
    "number_of_analytes": NumericPattern(labels=("number of analytes", "analytes"), bare_regex=None),
    "open_vial_stability_days": NumericPattern(labels=("open-vial stability", "open vial stability"), bare_regex=None),
    "languages_supported_count": NumericPattern(labels=("languages supported", "languages"), bare_regex=None),
    # --- DMPK / ADME ---
    "turnaround_business_days": NumericPattern(labels=("turnaround", "turnaround time", "tat",), bare_regex=None),
    "compound_amount_required_mg": NumericPattern(labels=("compound amount required", "compound amount",), bare_regex=None),
    "donor_pool_size": NumericPattern(labels=("donor pool size", "donor pool",), bare_regex=None),
    "test_concentration_um": NumericPattern(labels=("test concentration",), bare_regex=None),
    "replicates_count": NumericPattern(labels=("replicates",), bare_regex=None),
    "max_compounds_per_cassette": NumericPattern(labels=("max compounds per cassette", "compounds per cassette",), bare_regex=None),
    "animals_per_group": NumericPattern(labels=("animals per group",), bare_regex=None),
    "max_infusion_duration_hours": NumericPattern(labels=("max infusion duration", "infusion duration",), bare_regex=None),
    "hepatocyte_donors": NumericPattern(labels=("hepatocyte donors",), bare_regex=None),
    "incubation_time_hours": NumericPattern(labels=("incubation time",), bare_regex=None),
    "tissue_dilution_factor": NumericPattern(labels=("tissue dilution factor", "dilution factor",), bare_regex=None),
    "incubation_time_min": NumericPattern(labels=("incubation time",), bare_regex=None),
    "monolayer_age_days": NumericPattern(labels=("monolayer age",), bare_regex=None),

    # --- Bioanalysis ---
    "validated_assay_library_size": NumericPattern(labels=("validated assay library size", "assay library size",), bare_regex=None),
    "method_development_time_weeks": NumericPattern(labels=("method development time",), bare_regex=None),
    "matrix_lots_for_matrix_effect": NumericPattern(labels=("matrix lots for matrix effect", "matrix lots",), bare_regex=None),
    "calibration_standard_levels": NumericPattern(labels=("calibration standard levels",), bare_regex=None),
    "minimum_sample_volume_ul": NumericPattern(labels=("minimum sample volume",), bare_regex=None),
    "nonglp_turnaround_days": NumericPattern(labels=("non-glp sample analysis turnaround", "non-glp turnaround",), bare_regex=None),
    "glp_turnaround_days": NumericPattern(labels=("glp sample analysis turnaround", "glp turnaround",), bare_regex=None),
    "gene_therapy_detection_limit": NumericPattern(labels=("gene therapy detection limit", "detection limit",), bare_regex=None),
    "ada_assay_sensitivity_ng_ml": NumericPattern(labels=("ada assay sensitivity", "assay sensitivity",), bare_regex=None),
    "ada_assay_development_time_weeks": NumericPattern(labels=("ada assay development time", "assay development time",), bare_regex=None),
    "max_analytes_per_sample": NumericPattern(labels=("max analytes per sample", "analytes per sample",), bare_regex=None),

    # --- CMC, Analytical & QC ---
    "standard_turnaround_days": NumericPattern(labels=("standard turnaround",), bare_regex=None),
    "sterility_incubation_period_days": NumericPattern(labels=("sterility incubation period", "incubation period",), bare_regex=None),
    "nitrosamine_loq_ppb": NumericPattern(labels=("nitrosamine loq", "loq",), bare_regex=None),

    # --- Regulatory Affairs ---
    "markets_supported_for_dispatch": NumericPattern(labels=("markets supported for dispatch", "markets supported",), bare_regex=None),

    # --- Phase IV / Post-Marketing Studies ---
    "number_of_countries_phase_iv": NumericPattern(labels=("number of countries",), bare_regex=None),
    "target_sample_size": NumericPattern(labels=("target sample size", "sample size",), bare_regex=None),
    "number_of_data_sources_linked": NumericPattern(labels=("number of data sources linked", "data sources linked",), bare_regex=None),
    "registry_duration_years": NumericPattern(labels=("registry duration",), bare_regex=None),
    "infant_follow_up_period_months": NumericPattern(labels=("infant follow-up period", "infant follow-up",), bare_regex=None),
    "follow_up_duration_years": NumericPattern(labels=("follow-up duration",), bare_regex=None),

    # --- Toxicology & Safety Assessment ---
    "study_start_lead_time_weeks": NumericPattern(labels=("study start lead time",), bare_regex=None),
    "time_to_final_report_weeks": NumericPattern(labels=("time to final report",), bare_regex=None),
    "method_validation_lead_time_weeks": NumericPattern(labels=("method validation lead time",), bare_regex=None),

    # --- Pharmacovigilance & Drug Safety ---
    "languages_supported_pv": NumericPattern(labels=("languages supported", "languages",), bare_regex=None),
    "on_time_submission_compliance_pct": NumericPattern(labels=("on-time submission compliance", "submission compliance",), bare_regex=None),
    "program_languages_pv": NumericPattern(labels=("program languages",), bare_regex=None),

}

PERCENTAGE_PATTERNS: dict[str, re.Pattern] = {
    "crude_protein": _pct_pattern("crude protein", "protein"),
    "fat_pct": _pct_pattern("fat", "crude fat"),
    "crude_fiber": _pct_pattern("crude fiber", "fiber"),
    "quality_score_pct": _pct_pattern("quality score", "q30"),
}

SINGLE_LABEL_PATTERNS: dict[str, NumericPattern] = {
    "energy_density": NumericPattern(
        labels=("energy density", "kcal/g"),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*kcal\s*/\s*g\b", re.IGNORECASE),
    ),
    "shelf_life_months": NumericPattern(
        labels=("shelf life",),
        bare_regex=re.compile(r"\b(\d+(?:\.\d+)?)\s*months?\s*shelf\s*life\b|shelf\s*life\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*months?\b", re.IGNORECASE),
    ),
}

LABELED_TEXT_PATTERNS: dict[str, tuple[str, ...]] = {
    "labeled_sku": ("sku", "supplier sku", "item number", "catalog number", "catalog no", "cat no"),
    "labeled_mfr_sku": ("manufacturer sku", "mfr sku", "mfr part number", "manufacturer part number", "mpn"),
    "labeled_manufacturer": ("manufacturer", "mfr", "brand", "made by"),
    "labeled_protocol_scope": ("protocol", "scope", "protocol/scope", "study design"),
    "labeled_capacity": ("capacity", "throughput"),
    "labeled_deliverable": ("deliverable", "deliverables", "report"),
    "labeled_clone_id": ("clone", "clone id"),
    "labeled_rrid": ("rrid",),
    "labeled_dilution": ("dilution", "recommended dilution"),
    "labeled_part_number": ("part number", "part no", "p/n"),
    "labeled_compatible_instruments": ("compatible instruments", "compatible with", "fits"),
    "labeled_vector": ("vector", "vector/system", "backbone"),
    "labeled_cell_type": ("cell type",),
    "labeled_diagnosis": ("diagnosis", "disease state", "disease", "indication", "condition", "pathology diagnosis", "icd-10 code"),
    "labeled_anatomical_site": ("anatomical site", "tissue type", "organ"),
    "labeled_appearance_color": ("appearance", "color", "appearance/color"),
    "labeled_target_pathway": ("target", "pathway", "target/pathway"),
    "labeled_mix_ratio": ("mix ratio", "ratio"),
    "labeled_kit_contents": ("kit contents", "contents"),
    "labeled_matrix": ("matrix",),
    "labeled_functional_group": ("functional group",),
    "labeled_ionic_form": ("ionic form",),
    "labeled_particle_mesh_size": ("particle size", "mesh size", "mesh"),
    "labeled_polymerization_time": ("polymerization time", "cure schedule"),
    "labeled_physicochemical_descriptors": ("mw", "logp", "clogp", "hbd", "hba", "psa", "fsp3"),
    "labeled_solution_concentration": ("solution concentration", "stock concentration"),
    "labeled_sequence_region": ("sequence", "sequence/region", "region", "accession"),
    "labeled_cofactors_substrate": ("cofactor", "cofactors", "substrate"),
    "labeled_fraction_purity": ("fraction", "fraction/purity"),
    "labeled_target_antigen": ("target", "antigen", "target/antigen"),
    "labeled_kit_components": ("kit components", "components"),
    "labeled_intended_cell_type": ("intended cell type", "application", "intended cell type / application"),
    "labeled_validated_cell_types": ("validated cell types",),
    "labeled_tissue_organ": ("tissue", "organ", "tissue / organ"),
    "labeled_disease": ("disease",),
    "labeled_contamination_virus_status": ("contamination", "virus status", "contamination / virus status"),
    "labeled_licensing_permitted_use": ("licensing", "permitted use", "license"),
    "labeled_recommended_medium_conditions": ("recommended medium", "conditions", "recommended medium & conditions"),
    "labeled_subculturing_ratio": ("subculturing ratio", "split ratio"),
    "labeled_donor_age_sex": ("donor age", "donor sex", "age / sex"),
    "labeled_karyotype": ("karyotype",),
    "labeled_tissue_source": ("tissue", "source", "tissue / source", "tissue/source", "source tissue"),
    "labeled_characterization_markers": ("characterization markers", "markers", "pluripotency markers", "phenotype", "qc markers"),
    "labeled_recommended_medium": ("recommended medium",),
    "labeled_parental_cell_origin": ("parental cell", "tissue origin", "parental cell / tissue origin"),
    "labeled_donor_age_sex_disease": ("donor age", "sex", "disease"),
    "labeled_derived_cell_type_purity": ("derived cell type", "purity"),
    "labeled_disease_model_state": ("disease", "model state", "disease / model state"),
    "labeled_iacuc_consent": ("iacuc", "animal-welfare consent", "protocol number"),
    "labeled_disease_stage": ("disease stage", "stage", "tumor stage", "tnm stage", "clinical stage"),
    "labeled_clinical_data": ("clinical data", "clinical info", "lab data", "patient data", "annotations"),
    "labeled_medications": ("medications", "meds", "concomitant medications", "drug history"),
    "labeled_hla_type": ("hla type", "hla"),
    "labeled_facility_locations": ("facility locations", "locations"),
    "labeled_courier_partners": ("courier partners", "couriers"),
    "labeled_fixation_requirements": ("fixation requirements", "fixation"),
    "labeled_genotype_alleles": ("genotype", "alleles", "genotype / alleles"),
    "labeled_genetic_background": ("genetic background",),
    "labeled_mhc_haplotype": ("mhc haplotype", "mhc"),
    "labeled_known_background_traits": ("known background traits", "background traits"),
    "labeled_cancer_type_indication": ("cancer type", "indication"),
    "labeled_supplier_stock_code": ("stock number", "strain code", "supplier stock"),
    "labeled_production_location": ("production location", "origin"),
    "labeled_licensing_conditions": ("licensing", "conditions of use"),
    "labeled_line_strain_name": ("line", "strain name"),
    "labeled_mutant_allele": ("mutant allele", "allele designation"),
    "labeled_gene_symbol_locus": ("gene symbol", "locus"),
    "labeled_stock_strain_number": ("stock number", "strain number"),
    "labeled_genotype": ("genotype",),
    "labeled_chromosomes_affected": ("chromosomes affected", "chromosomes"),
    "labeled_balancer": ("balancer",),
    "labeled_mutagen": ("mutagen",),
    "labeled_donor_made_by": ("donor", "made by"),
    "labeled_compatible_lc_front_end": ("compatible lc front end", "lc front end"),
    "labeled_model_configuration": ("model", "configuration"),
    "labeled_software": ("software",),
    "labeled_warranty_service_plan": ("warranty", "service plan"),
    "labeled_camera_sensor": ("camera", "sensor"),
    "labeled_deck_worktable_size": ("deck size", "worktable size", "deck / worktable size"),
    "labeled_bag_consumable_type": ("bag", "consumable type"),
    "labeled_control_system": ("control system",),
    "labeled_scalability": ("scalability",),
    "labeled_material_equipment": ("material",),
    "labeled_performance_rating": ("performance rating",),
    "labeled_oem_cross_reference": ("replaces", "oem part cross-reference"),
    "labeled_mating_part": ("mating part",),
    "labeled_compatible_platform": ("compatible platform",),
    "labeled_size_capacity_consumables": ("size", "capacity"),
    "labeled_power_requirements": ("power requirements", "power"),
    "labeled_flow_cell_types": ("flow cell types", "flow cell"),
    "labeled_capacity_general_equip": ("capacity",),
    "labeled_image_analysis_platform": ("image analysis platform",),
    "labeled_scanner": ("scanner",),

    # --- Analytical & Reference Standards ---
    "labeled_analyte_parameter": ("analyte", "analytes", "parameter", "analyte / parameter", "analyte(s)"),
    "labeled_matrix_solvent": ("matrix", "solvent", "matrix / solvent"),
    "labeled_expiration_validity": ("expiration", "expiry", "validity", "valid until"),
    "labeled_certified_assigned_value": ("certified value", "assigned value", "certified / assigned value"),
    "labeled_current_lot_valid_use_date": ("current lot", "valid use date", "lot"),
    "labeled_homogeneity": ("homogeneity",),
    "labeled_variants_allele_frequency": ("variants", "allele frequency"),
    "labeled_peer_data_program": ("peer data program", "peer group"),
    "labeled_study_schedule": ("study schedule", "schedule"),
    # --- DMPK / ADME ---
    "labeled_time_points": ("time points", "timepoints",),
    "labeled_strain_dmpk": ("strain",),
    "labeled_inhibitor_concentration_range": ("inhibitor concentration range", "concentration range",),
    "labeled_dynamic_range": ("dynamic range",),
    "labeled_buffer_ph": ("buffer ph", "ph",),
    "labeled_probe_substrate": ("probe substrate",),

    # --- Bioanalysis ---
    "labeled_mass_spectrometer_model": ("mass spectrometer model", "mass spec model", "instrument model",),
    "labeled_default_assay_range": ("default assay range", "assay range",),
    "labeled_drug_tolerance_target": ("drug tolerance target",),

    # --- HEOR, RWE & Market Access ---
    "labeled_proprietary_modeling_platform": ("proprietary modeling platform", "modeling platform",),
    "labeled_proprietary_data_asset_used": ("proprietary data asset used", "data asset",),
    "labeled_slr_platform_tooling": ("slr platform", "tooling", "platform or tooling",),

    # --- Toxicology & Safety Assessment ---
    "labeled_pk_analysis_software": ("pk analysis software",),
    "labeled_data_acquisition_system": ("data acquisition system",),

    # --- Pharmacovigilance & Drug Safety ---
    "labeled_literature_tool": ("literature tool",),

}


def _find_labeled_number(row: Mapping[str, object], labels: tuple[str, ...]) -> tuple[str, str, str] | None:
    """Return (value, evidence, source_field) for the first labeled value whose
    label matches, extracting a leading number if present, else the raw text."""
    for source_field, label, value, evidence in iter_labeled_values(row):
        if label not in labels:
            continue
        number_match = re.search(r"-?\d+(?:\.\d+)?", value)
        extracted = number_match.group(0) if number_match else normalize_text_value(value)
        return extracted, evidence, source_field
    return None


def extract_numeric_or_labeled(row: Mapping[str, object], field_name: str, pattern_key: str, method_prefix: str) -> ExtractedSpec:
    spec = NUMERIC_PATTERNS.get(pattern_key)
    if spec is None:
        return missing_spec(field_name, f"{method_prefix}_unconfigured")

    labeled = _find_labeled_number(row, spec.labels)
    if labeled:
        value, evidence, source_field = labeled
        return ExtractedSpec(
            field_name=field_name, value=value, status="matched",
            method=f"{method_prefix}_labeled_regex", evidence=evidence,
            source_field=source_field, confidence=0.9,
        )

    if spec.bare_regex is not None:
        for source_field, label, value, evidence in iter_labeled_values(row):
            pass  # labeled values already checked above; fall through to free text
        from spec_extraction.extractors.common import iter_text_sources
        for source_field, text in iter_text_sources(row):
            match = spec.bare_regex.search(text)
            if match:
                raw = match.group(1) if match.groups() else match.group(0)
                return ExtractedSpec(
                    field_name=field_name, value=normalize_text_value(raw), status="matched",
                    method=f"{method_prefix}_regex", evidence=match.group(0).strip(),
                    source_field=source_field, confidence=0.8,
                )

    return missing_spec(field_name, f"{method_prefix}_regex")


def extract_percentage(row: Mapping[str, object], field_name: str, pattern_key: str) -> ExtractedSpec:
    pattern = PERCENTAGE_PATTERNS.get(pattern_key)
    if pattern is None:
        return missing_spec(field_name, "percentage_unconfigured")

    from spec_extraction.extractors.common import iter_text_sources
    for source_field, text in iter_text_sources(row):
        match = pattern.search(text)
        if match:
            return ExtractedSpec(
                field_name=field_name, value=match.group(1), status="matched",
                method="percentage_regex", evidence=match.group(0).strip(),
                source_field=source_field, confidence=0.85,
            )
    return missing_spec(field_name, "percentage_regex")


def extract_labeled_text(row: Mapping[str, object], field_name: str, pattern_key: str) -> ExtractedSpec:
    labels = LABELED_TEXT_PATTERNS.get(pattern_key, ())
    for source_field, label, value, evidence in iter_labeled_values(row):
        if label in labels:
            return ExtractedSpec(
                field_name=field_name, value=normalize_text_value(value), status="matched",
                method=f"{pattern_key}_labeled_regex", evidence=evidence,
                source_field=source_field, confidence=0.85,
            )
    return missing_spec(field_name, f"{pattern_key}_labeled_regex")


def extract_url(row: Mapping[str, object], field_name: str, image_only: bool) -> ExtractedSpec:
    from spec_extraction.extractors.common import iter_text_sources
    pattern = IMAGE_URL_PATTERN if image_only else URL_PATTERN
    for source_field, text in iter_text_sources(row):
        match = pattern.search(text)
        if match:
            return ExtractedSpec(
                field_name=field_name, value=match.group(0).strip(), status="matched",
                method="url_regex", evidence=match.group(0).strip(),
                source_field=source_field, confidence=0.9,
            )
    return missing_spec(field_name, "url_regex")


def dispatch_regex_field(row: Mapping[str, object], field_name: str, pattern_key: str | None) -> ExtractedSpec:
    if pattern_key is None:
        return missing_spec(field_name, "no_pattern_configured")
    if pattern_key in LABELED_TEXT_PATTERNS:
        return extract_labeled_text(row, field_name, pattern_key)
    if pattern_key in PERCENTAGE_PATTERNS:
        return extract_percentage(row, field_name, pattern_key)
    if pattern_key == "url_generic":
        return extract_url(row, field_name, image_only=False)
    if pattern_key == "image_url":
        return extract_url(row, field_name, image_only=True)
    if pattern_key in NUMERIC_PATTERNS:
        return extract_numeric_or_labeled(row, field_name, pattern_key, pattern_key)
    if pattern_key in SINGLE_LABEL_PATTERNS:
        spec = SINGLE_LABEL_PATTERNS[pattern_key]
        labeled = _find_labeled_number(row, spec.labels)
        if labeled:
            value, evidence, source_field = labeled
            return ExtractedSpec(
                field_name=field_name, value=value, status="matched",
                method=f"{pattern_key}_labeled_regex", evidence=evidence,
                source_field=source_field, confidence=0.9,
            )
        if spec.bare_regex is not None:
            from spec_extraction.extractors.common import iter_text_sources
            for source_field, text in iter_text_sources(row):
                match = spec.bare_regex.search(text)
                if match:
                    raw = next((g for g in match.groups() if g), match.group(0))
                    return ExtractedSpec(
                        field_name=field_name, value=normalize_text_value(raw), status="matched",
                        method=f"{pattern_key}_regex", evidence=match.group(0).strip(),
                        source_field=source_field, confidence=0.8,
                    )
        return missing_spec(field_name, f"{pattern_key}_regex")

    return missing_spec(field_name, f"{pattern_key}_unconfigured")
