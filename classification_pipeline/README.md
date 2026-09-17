# Classification Pipeline

End-to-end L3 + L4 product classification using cosine similarity against pre-embedded anchor descriptions. Supports both staging and prod environments via a single `--env` flag.

## Files

| File | Purpose |
|---|---|
| `classify_products.py` | Product pipeline — 3 phases (a, embed, publish) covering cache lookup, embedding, and Snowflake publish |
| `classify_services.py` | Services pipeline — 3 phases (cached, embed, publish); classifies quoted services against the same taxonomy anchors, publishes into the same output table as products |
| `product_classifier_utils.py` | Shared utilities: Snowflake session, listing loader, anchor loading, classification math, upsert-publish, Bedrock/Titan embedding, text hashing, cache helpers |
| `freeze_cache.py` | Manual override: freezes an active cache into a dated read-only volume and resets it to empty. Normally unnecessary — both pipelines auto-freeze at ~6GB (see "Cache volumes & freezing") |
| `seed_anchor_tables.py` | Re-embed L3/L4 anchor descriptions and write to Snowflake — re-run when taxonomy changes |
| `taxonomy/l3_taxonomy_anchors.json` | L3 category anchor descriptions (27 categories) |
| `taxonomy/l4_taxonomy_anchors.json` | L4 subcategory anchor descriptions (190 subcategories across all L3s) |

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
# Phase A — classify cache hits: active cache, then each frozen volume in
# artifacts/cache/frozen_volumes/ (one at a time)
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase a

# Phase embed — embed net-new products via Bedrock, then classify
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase embed

# Phase publish — merge results and write to Snowflake
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase publish
```

Replace `--env stage` with `--env prod` to run against prod (requires `PRODUCTS_PROD` table to exist).

### Why only 3 phases?

Historically this pipeline also had `extract`/`b` phases, needed because the shared v1/v2 product embedding volumes (~32GB and ~13GB) couldn't be loaded simultaneously, and v1 alone was too big to hold in memory alongside anything else — hence extracting just the needed vectors to a memmapped `.npy` file rather than loading the whole thing. Those volumes were retired in September 2026 after a source-table schema migration (see "Cache volumes & freezing" below) made them permanently stale (0 cache hits against the new data). The replacement design — small, individually-sized frozen volumes checked one at a time — never needs that workaround, so `extract`/`b` were removed. `phase a` now does classification directly as it goes, whichever layer (active cache or a given frozen volume) currently holds the vectors.

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

### Incremental embedding batches

When the net-new volume is huge (millions of rows) and the taxonomy may still change,
embed in capped batches rather than committing to the full run up front:

```bash
# Embed a capped batch (products: --limit; services: same flag, same meaning)
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase embed --limit 300000
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_services.py --env stage --phase embed --limit 100000
# (safe to run these two concurrently — separate cache files, no shared-write risk)

# Reclassify everything currently cached (this batch + all prior batches) and publish
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase a
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_products.py --env stage --phase publish
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_services.py --env stage --phase cached
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_services.py --env stage --phase publish
```

No dedup/"already published" bookkeeping is needed between batches: products publishes
via full overwrite (each publish replaces the table with the complete current cache-hit
set) and services publishes via upsert (safe to re-publish overlapping rows). Embeddings
accumulate permanently in each pipeline's cache regardless of batch size.

**If the taxonomy changes between batches**: re-run `seed_anchor_tables.py`, then just
repeat the reclassify+publish step above (no re-embedding needed — the cache is
independent of the taxonomy). Then continue embedding new batches, increasing `--limit`
over time (e.g. 300K → 1M) as confidence in the results grows.

`--limit` only affects `--phase embed`; the other phases are unaffected and always
process everything currently available to them.

## Cache volumes & freezing

Each active per-env cache (`embedding_cache_stage.pkl`, `embedding_cache_services_stage.pkl`, etc.) grows over time as `phase embed`/`phase cached`+`phase embed` runs add new entries. Left unbounded, an active cache eventually gets too big to comfortably hold in memory alongside everything else a phase needs (source dataframe, anchor vectors) — this machine has 24GB RAM, and the original v1 volume (34.5GB) already exceeded that on its own.

**This is now automatic.** At the end of `phase embed` (both `classify_products.py` and `classify_services.py`), once the active cache is saved, `maybe_auto_freeze()` (in `product_classifier_utils.py`) checks its on-disk size and — if it's crossed `FREEZE_THRESHOLD_BYTES` (6GB, set at the top of each script) — freezes it into a new dated volume and resets the active cache to `{}`, automatically, no manual step required. You'll see this in the `phase embed` output:

```
*** Active cache is 6.02 GB (>= 6.0 GB auto-freeze threshold) ***
*** Froze to embedding_cache_stage_frozen_20260917_143210.pkl — active cache reset to empty ***
```

`classify_products.py`'s `phase_a` and `classify_services.py`'s `phase_cached` both automatically discover every `*.pkl` in their respective frozen-volumes directory (`frozen_volumes/` for products, `frozen_volumes_services/` for services) at runtime — no code changes needed after a freeze, ever. Each checks the active cache first, then each frozen volume **one at a time** (load, check membership, classify hits, release before loading the next), so peak memory stays bounded regardless of how many volumes accumulate. This only works because each volume is kept small by the freeze threshold — if one were ever allowed to grow to v1's old size, it would need the old memmap-extraction treatment again.

`freeze_cache.py` still exists for manual use (e.g. freezing early, on purpose, before a batch you know will cross the threshold), and now takes an explicit target directory since products and services keep separate frozen-volume directories:

```bash
python freeze_cache.py artifacts/cache/embedding_cache_stage.pkl artifacts/cache/frozen_volumes
python freeze_cache.py artifacts/cache/embedding_cache_services_stage.pkl artifacts/cache/frozen_volumes_services
```

### Historical volumes (retired September 2026)

The original v1 (~32GB) and v2 (~13.8GB) product embedding volumes, plus the pre-migration `embedding_cache_stage.pkl` (2.19M entries) and the entire pre-migration `embedding_cache_services_stage.pkl` (535,184 entries, 100% unused), were archived — not deleted — to `artifacts/cache/archive/` after a `PRODUCTS_STAGE`/`SERVICES_STAGE` schema migration made them permanently stale (confirmed 0 cache hits against the new data; the underlying `DESCRIPTION` content and `PRICING_STATUS_C` casing changed). Only the subset confirmed still useful (665,204 unique hashes, 668,315 rows) was carried forward into a fresh active cache. These are no longer read by any pipeline — kept purely for reference/audit.

## Environment configs

Defined in `ENV_CONFIGS` at the top of `classify_products.py`:

| | stage | prod | stage_backfill |
|---|---|---|---|
| Input table | `PRODUCTS_STAGE` | `PRODUCTS_PROD` | `PRODUCTS_STAGE_BACKFILL` |
| Output table | `NEW_CLASSIFICATIONS_STAGE` | `NEW_CLASSIFICATIONS_PROD` | `NEW_CLASSIFICATIONS_STAGE` (same as stage) |
| Env cache | `embedding_cache_stage.pkl` | `embedding_cache_prod_new.pkl` | `embedding_cache_stage.pkl` (shared with stage) |
| Artifacts dir | `artifacts/analysis/stage_classification/` | `artifacts/analysis/prod_classification/` | `artifacts/analysis/stage_backfill_classification/` |
| Publish mode | overwrite | overwrite | **append** (delete-matching-PRODUCT_IDs, then insert) |

Frozen volumes under `artifacts/cache/frozen_volumes/` are shared/read-only and checked by all environments that share a cache lineage. See "Cache volumes & freezing" above.

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

`classify_services.py` classifies quoted-service listings using the same taxonomy anchors as products, but does not touch the product v1/v2 caches — it maintains its own dedicated, incrementally-growing cache instead. It **publishes into the same output table as `classify_products.py`** (`NEW_CLASSIFICATIONS_STAGE`/`PROD`), always via upsert (never overwrite) — so it must always be run *after* the corresponding products run, whose overwrite would otherwise wipe services' rows.

```bash
cd /Users/stephanie.mcmahon/smcmahon_repo/auto_classification/classification_pipeline

# Phase cached — classify whatever's already in the services cache, stage the rest
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_services.py --env stage --phase cached

# Phase embed — embed net-new services via Bedrock, then classify
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_services.py --env stage --phase embed

# Phase publish — upsert results into Snowflake
caffeinate -dims /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/python3 classify_services.py --env stage --phase publish
```

| | stage | prod |
|---|---|---|
| Input table | `SERVICES_STAGE` | `SERVICES_PROD` (placeholder — table doesn't exist yet) |
| Output table | `NEW_CLASSIFICATIONS_STAGE` (shared with products) | `NEW_CLASSIFICATIONS_PROD` (shared with products) |
| Cache | `embedding_cache_services_stage.pkl` | `embedding_cache_services_prod.pkl` |
| Artifacts dir | `artifacts/analysis/stage_services_classification/` | `artifacts/analysis/prod_services_classification/` |

To classify only what's already embedded (no Bedrock calls at all — useful for a fast preview against a freshly-updated taxonomy before committing to embedding a large net-new volume), run `--phase cached` then `--phase publish`, skipping `--phase embed` entirely. This works the same way for `classify_products.py`: run `--phase a` and `--phase publish`, skipping `--phase embed`.

Re-running `--phase embed` after a taxonomy update skips Bedrock calls for already-cached hashes and just reclassifies + re-publishes against the refreshed anchors.

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
