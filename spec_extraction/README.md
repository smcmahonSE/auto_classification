# Specification Extraction Pilot

This folder contains the deterministic first pass for extracting SME-proposed
category-specific product specifications after L3 classification.

The pilot intentionally avoids LLM calls. It uses regexes, controlled
vocabularies, normalization, and evidence capture so we can measure how far
rules alone can get before considering model-based extraction.

## Structure

- `extract_product_specs.py` orchestrates extraction over classified products.
- `schemas/l3_required_fields.json` records the SME-required fields for pilot L3s.
- `extractors/common.py` contains shared matching and output helpers.
- `extractors/chemicals.py` extracts `CAS Number` and `Purity`.
- `extractors/antibodies.py` extracts antibody-specific required fields.
- `outputs/` is reserved for generated pilot CSVs and coverage reports.

## Running

The runner reads classified products from Snowflake using `--env`:

- `stage` reads `NEW_CLASSIFICATIONS_STAGE`.
- `prod` reads `NEW_CLASSIFICATIONS_PROD`.

Use `--mode sample` for local review artifacts and `--mode full` for Snowflake
publishing.

```bash
# Sample/review run
/Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python extract_product_specs.py \
  --env stage \
  --mode sample \
  --category lab_supplies_consumables

# Full publish run
/Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python extract_product_specs.py \
  --env prod \
  --mode full \
  --category lab_supplies_consumables
```

Sample mode writes local CSVs under `outputs/<env>/sample/<category>/`:

- `spec_extraction_pilot_details.csv`
- `spec_extraction_pilot_coverage.csv`
- `spec_extraction_filter_summary.csv`
- `manual_review_sample.csv`

Full mode prints the same summary report in the terminal and writes a value-only
Snowflake table named `{CATEGORY}_SPEC_{ENV}`, for example
`LAB_SUPPLIES_CONSUMABLES_SPEC_STAGE`.

## Pilot Categories

The initial categories are:

- `Chemicals & Solvents`, because its required fields are highly pattern-based.
- `Antibodies`, because it tests deterministic extraction on a harder biology
  category with controlled vocabulary and multi-select fields.
