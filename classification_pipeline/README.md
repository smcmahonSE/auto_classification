# Classification Pipeline

End-to-end L3 + L4 product classification using cosine similarity against pre-embedded anchor descriptions. Supports both staging and prod environments via a single `--env` flag.

## Files

| File | Purpose |
|---|---|
| `classify_products.py` | Product pipeline — 5 phases covering cache lookup, vector extraction, embedding, and Snowflake publish |
| `classify_services.py` | Services pipeline — 2 phases (embed, publish); classifies quoted services against the same taxonomy anchors |
| `product_classifier_utils.py` | Shared utilities: Snowflake session, listing loader, anchor loading, classification math, Bedrock/Titan embedding, text hashing, cache helpers |
| `seed_anchor_tables.py` | Re-embed L3/L4 anchor descriptions and write to Snowflake — re-run when taxonomy changes |
| `taxonomy/l3_taxonomy_anchors.json` | L3 category anchor descriptions (13 categories) |
| `taxonomy/l4_taxonomy_anchors.json` | L4 subcategory anchor descriptions (76 subcategories across all L3s) |

Both pipelines classify against the same anchor tables (`EMBEDDED_L3_DESCRIPTIONS`/`EMBEDDED_L4_DESCRIPTIONS`), loaded via the shared `load_anchors_from_snowflake()` in `product_classifier_utils.py`. Re-running `seed_anchor_tables.py` after a taxonomy change affects both — re-run each pipeline's `--phase embed`/`a` afterward to reclassify against the updated anchors.

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

Every phase writes its output before exiting. The embed phase checkpoints every 1,000
embeddings by appending just the new-since-last-checkpoint entries to a small
`<cache>.pkl.delta` log next to the main cache file (`append_cache_delta` /
`load_pickle_cache_with_delta` / `consolidate_cache_delta` in
`product_classifier_utils.py`) — the full multi-GB cache file itself is only
rewritten once, at the very end of a run, so checkpoint cost stays cheap and constant
regardless of how large the cache has grown. To resume:

```bash
# Re-authenticate if needed
aws sso login --profile staging.admin

# Re-run the interrupted phase — already-done work (including anything sitting in an
# unconsolidated .delta log from the interrupted run) is picked up automatically
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase embed
```

A `.delta` log left behind by an interrupted run is safe to leave in place — the next
run's `load_pickle_cache_with_delta` replays it automatically, and discards a
truncated trailing chunk (from a kill mid-write) rather than failing, at the cost of
re-embedding at most one checkpoint's worth of entries.

## Environment configs

Defined in `ENV_CONFIGS` at the top of `classify_products.py`:

| | stage | prod | stage_backfill |
|---|---|---|---|
| Input table | `PRODUCTS_STAGE` | `PRODUCTS_PROD` | `PRODUCTS_STAGE_BACKFILL` |
| Output table | `NEW_CLASSIFICATIONS_STAGE` | `NEW_CLASSIFICATIONS_PROD` | `NEW_CLASSIFICATIONS_STAGE` (same as stage) |
| Env cache | `embedding_cache_stage.pkl` | `embedding_cache_prod_new.pkl` | `embedding_cache_stage.pkl` (shared with stage) |
| Artifacts dir | `artifacts/analysis/stage_classification/` | `artifacts/analysis/prod_classification/` | `artifacts/analysis/stage_backfill_classification/` |
| Publish mode | overwrite | overwrite | **append** (delete-matching-PRODUCT_IDs, then insert) |

Shared read-only caches (`embedding_cache.pkl`, `embedding_cache_new.pkl`, `embedding_cache_keys.pkl`) are used by all environments.

### Backfill runs

Any `ENV_CONFIGS` entry with `"append_mode": True` (like `stage_backfill`) publishes into its
target table via upsert instead of overwrite: `phase_publish` stages the run's distinct
`PRODUCT_ID`s to a Snowflake temp table, deletes any matching rows from the output table, then
appends all of this run's rows. This makes it safe to point a backfill's `output_table` at an
existing, already-published table (e.g. `stage_backfill` → `NEW_CLASSIFICATIONS_STAGE`) without
wiping out prior results, and safe to re-run a backfill's `--phase publish` if it's interrupted
mid-write.

`stage_backfill` deliberately shares its `cache_path` with `stage` (embeddings are keyed by text
content hash, not `PRODUCT_ID`, so overlapping text is a free cache hit either way) — **but do
not run `--phase embed` for `stage` and `stage_backfill` at the same time**, since the pickle
cache read/mutate/write cycle isn't safe for concurrent writers.

To set up a future backfill: add a new `ENV_CONFIGS` entry with its own `input_table` and
`out_dir`, point `output_table` at whichever table it should land in, and set `"append_mode":
True` if that table already has data you don't want overwritten.

## Running the services pipeline

`classify_services.py` classifies quoted-service listings using the same taxonomy anchors as products, but does not touch the product v1/v2 caches — it maintains its own dedicated, incrementally-growing cache instead.

```bash
cd /Users/stephanie.mcmahon/smcmahon_repo/auto_classification/classification_pipeline

# Phase embed — embed net-new services via Bedrock, then classify
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_services.py --env stage --phase embed

# Phase publish — write results to Snowflake
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_services.py --env stage --phase publish
```

| | stage | prod |
|---|---|---|
| Input table | `SERVICES_V1_STAGE` | `SERVICES_PROD_V1` (placeholder — table doesn't exist yet) |
| Output table | `CLASSIFICATIONS_SERVICES_V1_STAGE` | `CLASSIFICATIONS_SERVICES_V1_PROD` (placeholder) |
| Cache | `embedding_cache_services_stage.pkl` | `embedding_cache_services_prod.pkl` |
| Artifacts dir | `artifacts/analysis/stage_services_classification/` | `artifacts/analysis/prod_services_classification/` |

Re-running `--phase embed` after a taxonomy update skips Bedrock calls for already-cached hashes and just reclassifies + re-publishes against the refreshed anchors.

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
