# Classification Pipeline

End-to-end L3 + L4 product classification using cosine similarity against pre-embedded anchor descriptions. Supports both staging and prod environments via a single `--env` flag.

## Files

| File | Purpose |
|---|---|
| `classify_products.py` | Main pipeline — 5 phases covering cache lookup, vector extraction, embedding, and Snowflake publish |
| `product_classifier_utils.py` | Shared utilities: Snowflake session, Bedrock/Titan embedding, text hashing, cache helpers |
| `seed_anchor_tables.py` | Re-embed L3/L4 anchor descriptions and write to Snowflake — re-run when taxonomy changes |
| `taxonomy/l3_taxonomy_anchors.json` | L3 category anchor descriptions (13 categories) |
| `taxonomy/l4_taxonomy_anchors.json` | L4 subcategory anchor descriptions (76 subcategories across all L3s) |

## Auth

Both AWS (Bedrock) and Snowflake (Okta SSO) are required for phases that embed or publish.

```bash
# AWS — tokens last ~8 hours; re-run if you see UnauthorizedSSOTokenError
aws sso login --profile staging.admin
```

Snowflake auth triggers automatically on first use — a browser window opens for Okta SSO.

## Running the pipeline

All commands are run from the `classification_pipeline/` directory:

```bash
cd /Users/stephanie.mcmahon/smcmahon_repo/auto_classification/classification_pipeline
```

### Full run order

```bash
# Phase A — classify v2 cache + env cache hits (~30-60 min)
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase a

# Phase extract — extract v1 (32GB) vectors to memmap (~30 min)
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase extract

# Phase B — classify v1 cache hits from memmap (~30-40 min)
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase b

# Phase embed — embed net-new products via Bedrock, then classify (~2-4 hrs)
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase embed

# Phase publish — merge results and write to Snowflake (~10 min)
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase publish
```

Replace `--env stage` with `--env prod` to run against prod (requires `PRODUCTS_PROD` table to exist).

### Why 5 phases?

The prod embedding caches (v1: ~32GB, v2: ~13GB) cannot be loaded simultaneously. Phases a/extract/b stagger their memory usage. Phase embed only touches net-new products not found in any cache. Each phase is independently resumable.

### Resuming after interruption

Every phase writes its output before exiting. The embed phase checkpoints the env cache every 1,000 embeddings. To resume:

```bash
# Clean up any partial checkpoint file
rm -f /Users/stephanie.mcmahon/smcmahon_repo/auto_classification/artifacts/cache/embedding_cache_stage.tmp

# Re-authenticate if needed
aws sso login --profile staging.admin

# Re-run the interrupted phase — already-done work is skipped automatically
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase embed
```

## Environment configs

Defined in `ENV_CONFIGS` at the top of `classify_products.py`:

| | stage | prod |
|---|---|---|
| Input table | `PRODUCTS_STAGE` | `PRODUCTS_PROD` |
| Output table | `NEW_CLASSIFICATIONS_STAGE` | `NEW_CLASSIFICATIONS_PROD` |
| Env cache | `embedding_cache_stage.pkl` | `embedding_cache_prod_new.pkl` |
| Artifacts dir | `artifacts/analysis/stage_classification/` | `artifacts/analysis/prod_classification/` |

Shared read-only caches (`embedding_cache.pkl`, `embedding_cache_new.pkl`, `embedding_cache_keys.pkl`) are used by both environments.

## Output columns

Written to the Snowflake output table:

| Column | Description |
|---|---|
| `PRODUCT_ID` | Product identifier |
| `PRODUCT_NAME` | Product name |
| `DESCRIPTION` | Product description |
| `PRICING_STATUS_C` | Pricing status |
| `LIST_PRICE_C` | List price |
| `SOURCE` | Segment: LCG, LEI, or Services |
| `ASSIGNED_NEW_L3_ID` | L3 category snake_case id |
| `ASSIGNED_NEW_L3_LABEL` | L3 category display label |
| `L3_CONFIDENCE` | Cosine similarity score to winning L3 anchor |
| `L3_CONFIDENCE_MARGIN` | Gap between top-1 and top-2 L3 scores |
| `L3_IS_LOW_CONFIDENCE` | True if margin < 0.05 |
| `ASSIGNED_L4_ID` | L4 subcategory snake_case id |
| `ASSIGNED_L4_LABEL` | L4 subcategory display label |
| `L4_CONFIDENCE` | Cosine similarity score to winning L4 anchor |
| `L4_CONFIDENCE_MARGIN` | Gap between top-1 and top-2 L4 scores |
| `L4_IS_LOW_CONFIDENCE` | True if margin < 0.05 |

## Re-seeding anchors

Run `seed_anchor_tables.py` any time the taxonomy JSON files change. This re-embeds the anchor descriptions and overwrites the Snowflake anchor tables used by the pipeline.

```bash
aws sso login --profile staging.admin
/Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 seed_anchor_tables.py
```
