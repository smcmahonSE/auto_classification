# auto_classification

Automated L3 + L4 product taxonomy classification using Amazon Titan embeddings and cosine similarity.

## Repository structure

```
auto_classification/
├── classification_pipeline/               # Active classification pipeline — start here
│   ├── classify_products.py
│   ├── product_classifier_utils.py
│   ├── seed_anchor_tables.py
│   ├── taxonomy/
│   │   ├── l3_taxonomy_anchors.json       # L3 category definitions
│   │   └── l4_taxonomy_anchors.json       # L4 subcategory definitions
│   └── README.md                          # Run instructions
├── spec_extraction/        # Deterministic spec extraction by L3 category
├── analysis/               # Exploratory notebooks and reference data
├── artifacts/              # Cache files, classification results, model artifacts
│   ├── cache/              # Embedding caches (large — not committed to git)
│   └── analysis/           # Phase output CSVs and parquets
├── archive/                # Superseded scripts from prior approaches
├── requirements.txt
└── README.md
```

## Getting started

See [`classification_pipeline/README.md`](classification_pipeline/README.md) for full run instructions, phase descriptions, environment configs, and output column definitions.

## Setup

```bash
source /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/activate
pip install -r requirements.txt
```

## Auth

```bash
# AWS SSO (Bedrock) — required for embedding and anchor seeding
aws sso login --profile staging.admin
```

Snowflake auth (Okta SSO) triggers automatically via browser on first use.

## How it works

Products are classified by embedding their text (name + description + price fields) using Amazon Titan (`amazon.titan-embed-text-v1`) and computing cosine similarity against pre-embedded anchor descriptions for each L3 category and L4 subcategory. Anchors are stored in Snowflake (`EMBEDDED_L3_DESCRIPTIONS`, `EMBEDDED_L4_DESCRIPTIONS`) and loaded at runtime — no Bedrock calls needed for classification itself.

A margin-based confidence threshold (top-1 minus top-2 similarity ≥ 0.05) flags low-confidence assignments for human review.

## Taxonomy

13 L3 categories, 76 L4 subcategories. Definitions live in `classification_pipeline/taxonomy/`. Re-run `classification_pipeline/seed_anchor_tables.py` after any taxonomy changes to update the Snowflake anchor tables.

## Prior approach

The `archive/` folder contains the original LightGBM-based classifier and the intermediate per-source scripts (`classify_lcg.py`, `classify_services_batched.py`, etc.) that preceded the unified `classify_products.py` pipeline.
