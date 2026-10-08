# Classification Pipeline

End-to-end L2 + L3 listing classification using cosine similarity against pre-embedded anchor descriptions. Classifies both products and services in a single pass, since the taxonomy's L2/L3 anchor set already covers both.

## Files

| File | Purpose |
|---|---|
| `classify_listings.py` | The pipeline — 3 phases (classify, embed, publish) covering cache lookup, embedding, and Snowflake publish. Replaces the former `classify_products.py`/`classify_services.py` pair (archived under `auto_classification/archive/`) now that a single input view covers both. |
| `product_classifier_utils.py` | Shared utilities: Snowflake session, listing loader, anchor loading, classification math, upsert-publish, Bedrock/Titan embedding, text hashing, cache helpers |
| `freeze_cache.py` | Manual override: freezes an active cache into a dated read-only volume and resets it to empty. Normally unnecessary — the pipeline auto-freezes at ~6GB (see "Cache volumes & freezing") |
| `seed_anchor_tables.py` | Re-embed L2/L3 anchor descriptions and write to Snowflake — re-run when taxonomy changes |
| `taxonomy/l2_taxonomy_anchors.json` | L2 category anchor descriptions (27 categories) |
| `taxonomy/l3_taxonomy_anchors.json` | L3 subcategory anchor descriptions (190 subcategories across all L2s) |

Anchors are loaded via `load_anchors_from_snowflake()` in `product_classifier_utils.py`, from `EMBEDDED_L2_DESCRIPTIONS`/`EMBEDDED_L3_DESCRIPTIONS`. Re-running `seed_anchor_tables.py` after a taxonomy change affects classification immediately — re-run `--phase classify`/`embed` afterward to reclassify against the updated anchors.

## Why one pipeline instead of two

Products and services used to be classified by two separate scripts, reading two separate source tables (`PRODUCTS_STAGE`/`SERVICES_STAGE`), and had to coordinate publish order carefully: products published via full table overwrite, which would silently wipe out services' rows from the shared output table unless services immediately re-published afterward via upsert. That fragility is gone — `COMPLETE_STAGE` is a Snowflake view that unions products + services directly off the raw platform tables (deduped, active-only), and the taxonomy's L2/L3 anchor set already has dedicated categories for both product and service listings. One input table, one classification pass, one overwrite publish.

## Auth

Both AWS (Bedrock) and Snowflake (Okta SSO) are required for phases that embed or publish.

```bash
# AWS — tokens last anywhere from ~8 to 17+ hours; re-run if you see UnauthorizedSSOTokenError
aws sso login --profile bedrock
```

Snowflake auth triggers automatically on first use — a browser window opens for Okta SSO.

## Running the pipeline

All commands are run from the `classification_pipeline/` directory:

```bash
cd /Users/stephanie.mcmahon/smcmahon_repo/auto_classification/classification_pipeline
```

### Full run order

```bash
# Phase classify — classify cache hits: active cache, then each frozen volume in
# artifacts/cache/frozen_volumes/ (one at a time)
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_listings.py --env stage --phase classify

# Phase embed — embed net-new listings via Bedrock, then classify
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_listings.py --env stage --phase embed

# Phase publish — merge results and write to Snowflake (overwrite — this is the only job
# writing to NEW_CLASSIFICATIONS_STAGE, so no ordering dependency to manage)
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_listings.py --env stage --phase publish
```

`--env prod` isn't wired up yet — add a `"prod"` entry to `ENV_CONFIGS` pointing at a `COMPLETE_PROD` view, once one exists, following the same shape as `"stage"`.

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
aws sso login --profile bedrock

# Re-run the interrupted phase — already-done work (including anything sitting in an
# unconsolidated .delta log from the interrupted run) is picked up automatically
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_listings.py --env stage --phase embed
```

A `.delta` log left behind by an interrupted run is safe to leave in place — the next
run's `load_pickle_cache_with_delta` replays it automatically, and discards a
truncated trailing chunk (from a kill mid-write) rather than failing, at the cost of
re-embedding at most one checkpoint's worth of entries.

### Incremental embedding batches

When the net-new volume is huge (millions of rows) and the taxonomy may still change,
embed in capped batches rather than committing to the full run up front:

```bash
# Embed a capped batch
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_listings.py --env stage --phase embed --limit 1000000

# Reclassify everything currently cached (this batch + all prior batches) and publish
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_listings.py --env stage --phase classify
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_listings.py --env stage --phase publish
```

No dedup/"already published" bookkeeping is needed between batches — publish always does
a full overwrite, replacing the table with the complete current cache-hit set. Embeddings
accumulate permanently in the cache regardless of batch size.

**If the taxonomy changes between batches**: re-run `seed_anchor_tables.py`, then just
repeat the reclassify+publish step above (no re-embedding needed — the cache is
independent of the taxonomy). Then continue embedding new batches, increasing `--limit`
over time as confidence in the results grows.

`--limit` only affects `--phase embed`; the other phases are unaffected and always
process everything currently available to them.

To classify only what's already embedded (no Bedrock calls at all — useful for a fast
preview against a freshly-updated taxonomy before committing to embedding a large
net-new volume), run `--phase classify` then `--phase publish`, skipping `--phase embed`
entirely.

## Cache volumes & freezing

The active cache (`embedding_cache_listings_stage.pkl`) grows over time as `phase embed` runs add new entries. Left unbounded, it eventually gets too big to comfortably hold in memory alongside everything else a phase needs (source dataframe, anchor vectors) — this machine has 24GB RAM.

**This is automatic.** At the end of `phase embed`, once the active cache is saved, `maybe_auto_freeze()` (in `product_classifier_utils.py`) checks its on-disk size and — if it's crossed `FREEZE_THRESHOLD_BYTES` (6GB, set at the top of the script) — freezes it into a new dated volume and resets the active cache to `{}`, automatically, no manual step required. You'll see this in the `phase embed` output:

```
*** Active cache is 6.02 GB (>= 6.0 GB auto-freeze threshold) ***
*** Froze to embedding_cache_listings_stage_frozen_20261008_010203.pkl — active cache reset to empty ***
```

`phase_classify` automatically discovers every `*.pkl` in `artifacts/cache/frozen_volumes/` at runtime — no code changes needed after a freeze, ever. It checks the active cache first, then each frozen volume **one at a time** (load, check membership, classify hits, release before loading the next), so peak memory stays bounded regardless of how many volumes accumulate.

`freeze_cache.py` still exists for manual use (e.g. freezing early, on purpose, before a batch you know will cross the threshold):

```bash
python freeze_cache.py artifacts/cache/embedding_cache_listings_stage.pkl artifacts/cache/frozen_volumes
```

### Historical volumes

The original v1/v2 product embedding volumes and pre-migration caches were archived to `artifacts/cache/archive/` after an earlier source-table schema migration made them permanently stale — no longer read by any pipeline, kept purely for reference/audit.

The separate products and services caches that existed prior to the products/services pipeline collapse (October 2026) have been consolidated into the single cache this pipeline now uses: their frozen volumes were moved (not re-embedded) into the unified `frozen_volumes/` directory, and the two (empty, at time of collapse) active caches were merged into `embedding_cache_listings_stage.pkl`. The old active cache files are backed up at `artifacts/cache/_pre_collapse_backup/` pending one clean verified run, after which they can be deleted.

## Environment configs

Defined in `ENV_CONFIGS` at the top of `classify_listings.py`:

| | stage |
|---|---|
| Input table | `COMPLETE_STAGE` (view unioning products + services off the raw platform tables) |
| Output table | `NEW_CLASSIFICATIONS_STAGE` |
| Cache | `embedding_cache_listings_stage.pkl` |
| Artifacts dir | `artifacts/analysis/stage_listings_classification/` |
| Publish mode | overwrite (sole writer — no ordering dependency) |

Frozen volumes under `artifacts/cache/frozen_volumes/` are shared/read-only, checked by `phase classify`. See "Cache volumes & freezing" above.

## Output columns

Written to the Snowflake output table:

| Column | Description |
|---|---|
| `PRODUCT_ID` | Product identifier (being phased out in favor of `PRODUCT_VARIANT_ID`) |
| `PRODUCT_NAME` | Product name |
| `PRODUCT_VARIANT_ID` | Durable product variant identifier |
| `PRODUCT_VARIANT_NAME` | Variant-level name (ride-along; not yet used in classification) |
| `CATEGORY_NAME` | Source-provided category (ride-along; not yet used in classification) |
| `DESCRIPTION` | Product description |
| `SPECIFICATION_ASSIGNMENTS_C` | Structured spec assignments (ride-along; feeds `spec_extraction`'s text-mining fallback) |
| `PRICING_STATUS_C` | Pricing status |
| `LIST_PRICE_C` | List price |
| `PRODUCT_SOURCE_C` | Row origin (products vs. services) |
| `ASSIGNED_L2_ID` | L2 category snake_case id |
| `ASSIGNED_L2_LABEL` | L2 category display label |
| `L2_CONFIDENCE` | Cosine similarity score to winning L2 anchor |
| `L2_CONFIDENCE_MARGIN` | Gap between top-1 and top-2 L2 scores |
| `L2_IS_LOW_CONFIDENCE` | True if margin < 0.05 |
| `ASSIGNED_L3_ID` | L3 subcategory snake_case id |
| `ASSIGNED_L3_LABEL` | L3 subcategory display label |
| `L3_CONFIDENCE` | Cosine similarity score to winning L3 anchor |
| `L3_CONFIDENCE_MARGIN` | Gap between top-1 and top-2 L3 scores |
| `L3_IS_LOW_CONFIDENCE` | True if margin < 0.05 |

## Re-seeding anchors

Run `seed_anchor_tables.py` any time the taxonomy JSON files change. This re-embeds the anchor descriptions and overwrites the Snowflake anchor tables used by the pipeline.

```bash
aws sso login --profile bedrock
/Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 seed_anchor_tables.py
```
