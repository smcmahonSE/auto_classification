"""
Classify product listings into L3 + L4 taxonomy in a single pipeline.

Supports both staging and prod environments via --env. All environment-specific
table names, cache paths, and artifact directories are resolved from ENV_CONFIGS.

Key design:
  - Anchor vectors loaded from Snowflake (no Bedrock calls for anchors)
  - L3 + L4 classification happen in the same pass
  - Cache hits are checked in two layers: the active/growing per-env cache, then
    each frozen (read-only) volume in artifacts/cache/frozen_volumes/, loaded one
    at a time and classified immediately before moving to the next — bounded peak
    memory no matter how many frozen volumes accumulate over time. Historical v1/v2
    volumes (from before the products/services schema migration) are archived under
    artifacts/cache/archive/ and no longer checked — see freeze_cache.py to create
    new frozen volumes as the active cache grows.
  - Net-new products embedded with parallel Bedrock workers (max_workers=10)
  - New embeddings saved to a per-env incremental cache, checkpointed every 1,000
    (checkpoints append only new-since-last-checkpoint entries to a small delta
    log — see append_cache_delta/consolidate_cache_delta in
    product_classifier_utils.py — so cost stays cheap regardless of cache size;
    the full cache file is only rewritten once, at the very end of a run)
  - Results written to Snowflake in 500K-row chunks

Run order (full run):
    python classify_products.py --env stage --phase a        # classify cache hits (active + frozen volumes)
    python classify_products.py --env stage --phase embed    # embed & classify net-new products
    python classify_products.py --env stage --phase publish  # write to Snowflake

For incremental batching (--limit on phase embed) and the freeze workflow, see
README.md.
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "classification_pipeline"))

from product_classifier_utils import (
    append_cache_delta,
    attach_classifications,
    build_product_text,
    classify_l3_and_l4,
    consolidate_cache_delta,
    embed_texts_from_cache,
    get_bedrock_client,
    get_products_session,
    load_anchors_from_snowflake,
    load_listings,
    load_pickle_cache_with_delta,
    stable_text_hash,
    upsert_publish,
)

# ── Static config (shared across all environments) ────────────────────────────
AWS_PROFILE      = "staging.admin"
AWS_REGION       = "us-east-1"
MODEL_ID         = "amazon.titan-embed-text-v1"
EMBED_WORKERS    = 10        # parallel Bedrock workers for net-new products
EMBED_CHECKPOINT = 1_000     # save env cache every N new embeddings
PUBLISH_CHUNK    = 500_000   # rows per Snowflake append

FROZEN_VOLUMES_DIR = PROJECT_ROOT / "artifacts/cache/frozen_volumes"

# ── Environment configs ───────────────────────────────────────────────────────
ENV_CONFIGS = {
    "stage": {
        "input_table":  "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_STAGE",
        "output_table": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_CLASSIFICATIONS_STAGE",
        "cache_path":   PROJECT_ROOT / "artifacts/cache/embedding_cache_stage.pkl",
        "out_dir":      PROJECT_ROOT / "artifacts/analysis/stage_classification",
        "append_mode":  False,
    },
    "prod": {
        "input_table":  "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_PROD",
        "output_table": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_CLASSIFICATIONS_PROD",
        "cache_path":   PROJECT_ROOT / "artifacts/cache/embedding_cache_prod_new.pkl",
        "out_dir":      PROJECT_ROOT / "artifacts/analysis/prod_classification",
        "append_mode":  False,
    },
    # Backfill: products missing from the original PRODUCTS_STAGE run. Publishes into the
    # SAME output table as "stage" via append_mode (delete-matching-PRODUCT_IDs, then insert)
    # instead of the default overwrite-the-whole-table behavior. cache_path is deliberately
    # shared with "stage" since embeddings are keyed by text content hash, not PRODUCT_ID —
    # but do not run `phase embed` for "stage" and "stage_backfill" concurrently, since
    # both would append checkpoint chunks to the same shared delta log file and could
    # interleave writes; the two also race to consolidate it back into the main cache.
    "stage_backfill": {
        "input_table":  "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_STAGE_BACKFILL",
        "output_table": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_CLASSIFICATIONS_STAGE",
        "cache_path":   PROJECT_ROOT / "artifacts/cache/embedding_cache_stage.pkl",
        "out_dir":      PROJECT_ROOT / "artifacts/analysis/stage_backfill_classification",
        "append_mode":  True,
    },
}

# ── Runtime globals (set in __main__ after --env is resolved) ─────────────────
# These are referenced by the phase functions below.
INPUT_TABLE      = None
OUTPUT_TABLE     = None
CACHE_ENV_PATH   = None
OUT_DIR          = None
APPEND_MODE      = None
PHASE_A_RESULTS  = None
EMBED_WORK       = None
PHASE_EMBED_RESULTS = None
EMBED_LIMIT      = None


# ── Phase A ───────────────────────────────────────────────────────────────────

def phase_a():
    print("\n=== PHASE A: Classify cache hits (active cache + frozen volumes) ===")

    sf = get_products_session()
    l3_anchors, l4_by_l3 = load_anchors_from_snowflake(sf)
    df = load_listings(sf, INPUT_TABLE)

    texts  = build_product_text(df).tolist()
    hashes = [stable_text_hash(t) for t in texts]

    BATCH = 250_000
    records = []

    def classify_and_record(cache_layer, idx_list, label):
        for start in range(0, len(idx_list), BATCH):
            idx_batch    = idx_list[start:start + BATCH]
            batch_hashes = [hashes[i] for i in idx_batch]
            vecs    = np.array([cache_layer[h] for h in batch_hashes], dtype=np.float32)
            results = classify_l3_and_l4(vecs, l3_anchors, l4_by_l3)
            records.append(attach_classifications(df.iloc[idx_batch], results))
            pct = (start + len(idx_batch)) / max(len(idx_list), 1) * 100
            hi  = (~results[4]).sum()
            print(f"  {label} batch {start:,}–{start+len(idx_batch):,} ({pct:.0f}%) — L3 high-conf: {hi:,}/{len(idx_batch):,}")
            del vecs

    # Active/growing cache first — cheapest, and most likely to match recently
    # embedded batches.
    cache_env  = load_pickle_cache_with_delta(CACHE_ENV_PATH)
    remaining  = list(range(len(hashes)))
    active_idx = [i for i in remaining if hashes[i] in cache_env]
    print(f"In active cache: {len(active_idx):,}")
    if active_idx:
        classify_and_record(cache_env, active_idx, "active")
        active_set = set(active_idx)
        remaining  = [i for i in remaining if i not in active_set]
    del cache_env

    # Frozen (read-only) volumes, one at a time — bounded peak memory no matter how
    # many accumulate, since each is loaded fully, checked/classified, then released
    # before the next loads. See freeze_cache.py for how these get created; each is
    # kept to a size (~1M entries / ~6GB) that's safe to hold on its own.
    FROZEN_VOLUMES_DIR.mkdir(parents=True, exist_ok=True)
    frozen_paths = sorted(FROZEN_VOLUMES_DIR.glob("*.pkl"))
    for i, volume_path in enumerate(frozen_paths):
        if not remaining:
            break
        print(f"Loading frozen volume {i+1}/{len(frozen_paths)} ({volume_path.name}, "
              f"{volume_path.stat().st_size/1e9:.2f} GB)...")
        with open(volume_path, "rb") as f:
            volume = pickle.load(f)
        vol_idx = [i for i in remaining if hashes[i] in volume]
        print(f"  {len(vol_idx):,} hits in this volume")
        if vol_idx:
            classify_and_record(volume, vol_idx, volume_path.stem)
            vol_set   = set(vol_idx)
            remaining = [i for i in remaining if i not in vol_set]
        del volume

    miss_idx = remaining
    print(f"\nCache misses: {len(miss_idx):,}  ← will be embedded in phase embed")

    if records:
        phase_a_df = pd.concat(records, ignore_index=True)
    else:
        print("Phase A: no cache hits.")
        empty_results = classify_l3_and_l4(np.empty((0, 1536), dtype=np.float32), l3_anchors, l4_by_l3)
        phase_a_df = attach_classifications(df.iloc[0:0], empty_results)

    phase_a_df.to_csv(PHASE_A_RESULTS, index=False)
    print(f"\nPhase A saved: {PHASE_A_RESULTS} ({len(phase_a_df):,} rows)")
    if len(phase_a_df):
        hi = (~phase_a_df["L3_IS_LOW_CONFIDENCE"]).sum()
        print(f"L3 high-confidence: {hi:,}/{len(phase_a_df):,} ({hi/len(phase_a_df)*100:.1f}%)")

    embed_work = df.iloc[miss_idx].copy()
    embed_work["_HASH"] = [hashes[i] for i in miss_idx]
    embed_work.to_parquet(EMBED_WORK, index=False)
    print(f"Embed work file:   {EMBED_WORK} ({len(embed_work):,} rows)")


# ── Phase embed ───────────────────────────────────────────────────────────────

def phase_embed():
    print("\n=== PHASE EMBED: Embed & classify net-new products ===")
    if not EMBED_WORK.exists():
        print("ERROR: Run phase a first.")
        sys.exit(1)

    embed_work = pd.read_parquet(EMBED_WORK)
    print(f"Net-new products to embed: {len(embed_work):,}")

    sf = get_products_session()
    l3_anchors, l4_by_l3 = load_anchors_from_snowflake(sf)

    cache_env = load_pickle_cache_with_delta(CACHE_ENV_PATH)
    bedrock   = get_bedrock_client(profile_name=AWS_PROFILE, region=AWS_REGION)

    hashes = embed_work["_HASH"].tolist()

    already_done = [h for h in hashes if h in cache_env]
    still_needed = sorted({h for h in hashes if h not in cache_env})
    print(f"Already in env cache: {len(already_done):,} (resuming from prior run)")
    print(f"Not yet embedded: {len(still_needed):,}")

    if EMBED_LIMIT is not None and len(still_needed) > EMBED_LIMIT:
        print(f"--limit {EMBED_LIMIT:,}: embedding only {EMBED_LIMIT:,} of {len(still_needed):,} remaining this run; "
              f"{len(still_needed) - EMBED_LIMIT:,} deferred to a future run")
        still_needed = still_needed[:EMBED_LIMIT]

    if still_needed:
        print(f"\nEmbedding {len(still_needed):,} texts with {EMBED_WORKERS} parallel workers...")

        all_texts    = build_product_text(embed_work).tolist()
        hash_to_text = {h: t for h, t in zip(hashes, all_texts)}

        # Checkpoints append only the entries added since the last checkpoint to a
        # small delta log (append_cache_delta), instead of re-pickling the entire
        # (multi-GB, ever-growing) cache dict every time. The full cache file is only
        # rewritten once, at the very end (consolidate_cache_delta below).
        seen_keys = set(cache_env.keys())

        def on_checkpoint(cache, processed):
            nonlocal seen_keys
            current_keys = set(cache.keys())
            new_keys = current_keys - seen_keys
            delta = {k: cache[k] for k in new_keys}
            print(f"  Checkpoint: {processed:,} embedded — appending {len(delta):,} new entries to delta log...")
            append_cache_delta(delta, CACHE_ENV_PATH)
            seen_keys = current_keys

        embed_texts_from_cache(
            texts            = [hash_to_text[h] for h in still_needed],
            text_hashes      = still_needed,
            cache            = cache_env,
            client           = bedrock,
            model_id         = MODEL_ID,
            show_progress    = True,
            max_workers      = EMBED_WORKERS,
            checkpoint_every = EMBED_CHECKPOINT,
            on_checkpoint    = on_checkpoint,
        )
        print("Saving final env cache...")
        consolidate_cache_delta(cache_env, CACHE_ENV_PATH)

    # Classify whatever is *currently* cache-hit within embed_work — this run's batch
    # plus anything embedded in an earlier --limit'd run — not the full embed_work set,
    # since a capped run leaves most of it still uncached (cache_env[h] would KeyError
    # on those). An uncapped run ends up classifying everything anyway, since the whole
    # set becomes cache-hit.
    cached_mask = [h in cache_env for h in hashes]
    n_ready = sum(cached_mask)
    ready_df = embed_work[cached_mask].reset_index(drop=True)
    ready_hashes = [h for h, m in zip(hashes, cached_mask) if m]
    print(f"\nClassifying {n_ready:,} of {len(embed_work):,} net-new products now embedded/cached "
          f"({len(embed_work) - n_ready:,} still awaiting embedding)...")

    BATCH = 100_000
    records = []
    for start in range(0, len(ready_df), BATCH):
        end          = min(start + BATCH, len(ready_df))
        batch_hashes = ready_hashes[start:end]
        vecs    = np.array([cache_env[h] for h in batch_hashes], dtype=np.float32)
        results = classify_l3_and_l4(vecs, l3_anchors, l4_by_l3)
        batch_df = ready_df.iloc[start:end].drop(columns=["_HASH"], errors="ignore")
        records.append(attach_classifications(batch_df, results))
        hi  = (~results[4]).sum()
        pct = end / max(len(ready_df), 1) * 100
        print(f"  batch {start:,}–{end:,} ({pct:.0f}%) — L3 high-conf: {hi:,}/{end-start:,}")

    if records:
        embed_results = pd.concat(records, ignore_index=True)
    else:
        embed_results = ready_df.drop(columns=["_HASH"], errors="ignore")

    embed_results.to_csv(PHASE_EMBED_RESULTS, index=False)
    print(f"\nPhase embed saved: {PHASE_EMBED_RESULTS} ({len(embed_results):,} rows)")
    if len(embed_results):
        hi = (~embed_results["L3_IS_LOW_CONFIDENCE"]).sum()
        print(f"L3 high-confidence: {hi:,}/{len(embed_results):,} ({hi/len(embed_results)*100:.1f}%)")


# ── Phase publish ─────────────────────────────────────────────────────────────

def phase_publish():
    print("\n=== PHASE PUBLISH: Write to Snowflake ===")

    # A phase's result file can be a stale leftover from a much earlier run (e.g. a
    # full run from weeks ago) sitting in the same artifacts dir, silently picked up
    # here even though this run never regenerated it — bit us once already. Flag any
    # result file whose age differs from Phase A's by more than a day so it doesn't
    # happen silently again; a real multi-hour/overnight run between phases is normal
    # and won't trip this.
    reference_mtime = PHASE_A_RESULTS.stat().st_mtime if PHASE_A_RESULTS.exists() else None
    STALE_THRESHOLD_SECONDS = 24 * 3600

    parts = []
    for label, path in [
        ("Phase A",     PHASE_A_RESULTS),
        ("Phase embed", PHASE_EMBED_RESULTS),
    ]:
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            stale_flag = ""
            if reference_mtime is not None and abs(path.stat().st_mtime - reference_mtime) > STALE_THRESHOLD_SECONDS:
                stale_flag = "  *** POSSIBLY STALE — modified >24h apart from Phase A's result. Verify this wasn't left over from an earlier run before trusting this publish. ***"
            df = pd.read_csv(path, low_memory=False)
            parts.append(df)
            print(f"  {label}: {len(df):,} rows (file modified {mtime:%Y-%m-%d %H:%M}){stale_flag}")
        else:
            print(f"  WARNING: {path.name} not found — skipping")

    if not parts:
        print("ERROR: No phase results found.")
        sys.exit(1)

    # Drop genuinely-empty (0-row) parts before concatenating — mixing them in
    # triggers a pandas dtype-inference quirk on boolean columns (see the
    # concat FutureWarning) that broke the confidence-summary prints below.
    parts = [df for df in parts if len(df) > 0]

    combined = pd.concat(parts, ignore_index=True)
    # PRODUCT_ID is being phased out in favor of PRODUCT_VARIANT_ID as the durable
    # identifier — some rows now have a null PRODUCT_ID with a valid, unique
    # PRODUCT_VARIANT_ID instead. Dedup on whichever identifier is present so
    # multiple such rows aren't all treated as "the same" null and collapsed
    # down to one (pandas' drop_duplicates considers NaN == NaN).
    combined["_DEDUP_KEY"] = combined["PRODUCT_VARIANT_ID"].fillna(combined["PRODUCT_ID"])
    combined = combined.drop_duplicates(subset="_DEDUP_KEY", keep="last").drop(columns="_DEDUP_KEY")
    print(f"\nCombined: {len(combined):,} rows, {len(combined.columns)} columns")

    total = len(combined)
    hi_l3 = (~combined["L3_IS_LOW_CONFIDENCE"]).sum()
    hi_l4 = (combined["L4_IS_LOW_CONFIDENCE"] == False).sum()  # noqa: E712
    no_l4 = combined["ASSIGNED_L4_LABEL"].isna().sum()
    print(f"L3 high-confidence:  {hi_l3:,} ({hi_l3/total*100:.1f}%)")
    print(f"L4 assigned:         {total - no_l4:,} ({(total-no_l4)/total*100:.1f}%)")
    print(f"L4 high-confidence:  {hi_l4:,} ({hi_l4/total*100:.1f}%)")

    print("\nL3 distribution:")
    print(combined["ASSIGNED_NEW_L3_LABEL"].value_counts().to_string())

    combined.columns = [c.upper() for c in combined.columns]

    print(f"\nConnecting to Snowflake...")
    sf = get_products_session()

    if APPEND_MODE:
        upsert_publish(sf, combined, OUTPUT_TABLE, PUBLISH_CHUNK)
    else:
        n_chunks = (len(combined) + PUBLISH_CHUNK - 1) // PUBLISH_CHUNK
        print(f"Writing {len(combined):,} rows to {OUTPUT_TABLE} in {n_chunks} chunk(s)...")
        for i, start in enumerate(range(0, len(combined), PUBLISH_CHUNK)):
            chunk = combined.iloc[start:start + PUBLISH_CHUNK]
            mode  = "overwrite" if i == 0 else "append"
            sf.create_dataframe(chunk).write.mode(mode).save_as_table(OUTPUT_TABLE)
            print(f"  chunk {i+1}/{n_chunks}: {len(chunk):,} rows written ({mode})")
        print(f"\nDone. {OUTPUT_TABLE} updated with {len(combined):,} rows.")

    print("\nColumns written:")
    for col in combined.columns:
        print(f"  {col}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env",   choices=list(ENV_CONFIGS.keys()), required=True,
                        help="Which environment to classify (see ENV_CONFIGS)")
    parser.add_argument("--phase", choices=["a", "embed", "publish"], required=True,
                        help="Which phase to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Phase embed only: cap how many net-new rows get embedded via Bedrock "
                             "this run, for incremental batching. Omit to embed everything still needed.")
    args = parser.parse_args()

    # Resolve environment config into module-level globals so phase functions pick them up
    cfg = ENV_CONFIGS[args.env]
    INPUT_TABLE   = cfg["input_table"]
    OUTPUT_TABLE  = cfg["output_table"]
    CACHE_ENV_PATH = cfg["cache_path"]
    OUT_DIR        = cfg["out_dir"]
    APPEND_MODE    = cfg.get("append_mode", False)
    EMBED_LIMIT    = args.limit
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PHASE_A_RESULTS     = OUT_DIR / "phase_a_results.csv"
    EMBED_WORK          = OUT_DIR / "phase_embed_work.parquet"
    PHASE_EMBED_RESULTS = OUT_DIR / "phase_embed_results.csv"

    print(f"Environment: {args.env.upper()}")
    print(f"  Input:  {INPUT_TABLE}")
    print(f"  Output: {OUTPUT_TABLE}")
    print(f"  Cache:  {CACHE_ENV_PATH}")
    print(f"  Artifacts: {OUT_DIR}")
    print(f"  Mode:   {'APPEND (delete+insert upsert)' if APPEND_MODE else 'OVERWRITE'}")

    if args.phase == "a":
        phase_a()
    elif args.phase == "embed":
        phase_embed()
    elif args.phase == "publish":
        phase_publish()
