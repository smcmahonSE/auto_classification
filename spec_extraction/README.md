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

## Pilot Categories

The initial categories are:

- `Chemicals & Solvents`, because its required fields are highly pattern-based.
- `Antibodies`, because it tests deterministic extraction on a harder biology
  category with controlled vocabulary and multi-select fields.
