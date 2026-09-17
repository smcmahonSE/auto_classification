"""
Classify quoted-service listings into L3 + L4 taxonomy, using the same
taxonomy anchors as classify_products.py.

Services are a distinct, actively-changing corpus with their own dedicated
embedding cache — this pipeline does NOT check the frozen product embedding
volumes (v1/v2) used by classify_products.py.

Key design:
  - Anchor vectors loaded from Snowflake (shared with classify_products.py)
  - L3 + L4 classification happen in the same pass
  - Phase 'cached' classifies whatever's already in the dedicated services cache
    and stages the rest — mirrors classify_products.py's phase_a, minus the v1/v2
    volume split (services has only one cache layer, no giant frozen volumes to
    memmap around)
  - Phase 'embed' embeds and classifies only the net-new subset phase 'cached' staged
  - New embeddings saved to a per-env incremental cache, checkpointed every 1,000
    (checkpoints append only new-since-last-checkpoint entries to a small delta
    log — see append_cache_delta/consolidate_cache_delta in
    product_classifier_utils.py — so cost stays cheap regardless of cache size;
    the full cache file is only rewritten once, at the very end of a run)
  - Results published via upsert (delete-matching-then-insert) — NEVER an
    overwrite, since this shares NEW_CLASSIFICATIONS_STAGE/PROD with
    classify_products.py's full-overwrite publish

Run order:
    python classify_services.py --env stage --phase cached   # classify cache hits, stage the rest
    python classify_services.py --env stage --phase embed    # embed & classify net-new services
    python classify_services.py --env stage --phase publish  # upsert into Snowflake

To classify ONLY what's already embedded (no Bedrock calls at all), run
'cached' then 'publish' — do not run 'embed'. Must run AFTER
classify_products.py's full ('stage'/'prod') publish, since that overwrites
the shared output table.
"""

import argparse
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
EMBED_WORKERS    = 10        # parallel Bedrock workers for net-new services
EMBED_CHECKPOINT = 1_000     # save cache every N new embeddings
PUBLISH_CHUNK    = 500_000   # rows per Snowflake append
CLASSIFY_BATCH   = 100_000   # rows per classification batch

# ── Environment configs ───────────────────────────────────────────────────────
# NOTE: the "prod" entry is a naming placeholder — no prod services table
# exists yet. Confirm/adjust these names once it's created.
ENV_CONFIGS = {
    "stage": {
        "input_table":  "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES_STAGE",
        "output_table": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_CLASSIFICATIONS_STAGE",
        "cache_path":   PROJECT_ROOT / "artifacts/cache/embedding_cache_services_stage.pkl",
        "out_dir":      PROJECT_ROOT / "artifacts/analysis/stage_services_classification",
    },
    "prod": {
        "input_table":  "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES_PROD",
        "output_table": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_CLASSIFICATIONS_PROD",
        "cache_path":   PROJECT_ROOT / "artifacts/cache/embedding_cache_services_prod.pkl",
        "out_dir":      PROJECT_ROOT / "artifacts/analysis/prod_services_classification",
    },
}

# ── Runtime globals (set in __main__ after --env is resolved) ─────────────────
INPUT_TABLE         = None
OUTPUT_TABLE        = None
CACHE_PATH          = None
OUT_DIR             = None
CACHED_RESULTS      = None
EMBED_WORK          = None
PHASE_EMBED_RESULTS = None
EMBED_LIMIT         = None


# ── Phase cached ──────────────────────────────────────────────────────────────

def phase_cached():
    print("\n=== PHASE CACHED: Classify cache hits, stage the rest ===")

    sf = get_products_session()
    l3_anchors, l4_by_l3 = load_anchors_from_snowflake(sf)
    df = load_listings(sf, INPUT_TABLE)

    texts  = build_product_text(df).tolist()
    hashes = [stable_text_hash(t) for t in texts]

    cache = load_pickle_cache_with_delta(CACHE_PATH)

    in_cache   = [h in cache for h in hashes]
    cached_idx = [i for i, m in enumerate(in_cache) if m]
    miss_idx   = [i for i, m in enumerate(in_cache) if not m]

    print(f"\nIn cache:    {len(cached_idx):,}")
    print(f"Not cached:  {len(miss_idx):,}  ← will be embedded in phase embed")

    if cached_idx:
        records = []
        for start in range(0, len(cached_idx), CLASSIFY_BATCH):
            idx_batch    = cached_idx[start:start + CLASSIFY_BATCH]
            batch_hashes = [hashes[i] for i in idx_batch]
            vecs    = np.array([cache[h] for h in batch_hashes], dtype=np.float32)
            results = classify_l3_and_l4(vecs, l3_anchors, l4_by_l3)
            records.append(attach_classifications(df.iloc[idx_batch], results))
            hi  = (~results[4]).sum()
            pct = (start + len(idx_batch)) / len(cached_idx) * 100
            print(f"  batch {start:,}–{start+len(idx_batch):,} ({pct:.0f}%) — L3 high-conf: {hi:,}/{len(idx_batch):,}")
        cached_df = pd.concat(records, ignore_index=True)
        cached_df.to_csv(CACHED_RESULTS, index=False)
        hi = (~cached_df["L3_IS_LOW_CONFIDENCE"]).sum()
        print(f"\nPhase cached saved: {CACHED_RESULTS} ({len(cached_df):,} rows)")
        print(f"L3 high-confidence: {hi:,}/{len(cached_df):,} ({hi/len(cached_df)*100:.1f}%)")
    else:
        print("Phase cached: no cache hits.")

    embed_work = df.iloc[miss_idx].copy()
    embed_work["_HASH"] = [hashes[i] for i in miss_idx]
    embed_work.to_parquet(EMBED_WORK, index=False)
    print(f"Embed work file: {EMBED_WORK} ({len(embed_work):,} rows)")


# ── Phase embed ───────────────────────────────────────────────────────────────

def phase_embed():
    print("\n=== PHASE EMBED: Embed & classify net-new services ===")
    if not EMBED_WORK.exists():
        print("ERROR: Run phase cached first.")
        sys.exit(1)

    embed_work = pd.read_parquet(EMBED_WORK)
    print(f"Net-new services to embed: {len(embed_work):,}")

    sf = get_products_session()
    l3_anchors, l4_by_l3 = load_anchors_from_snowflake(sf)

    cache   = load_pickle_cache_with_delta(CACHE_PATH)
    bedrock = get_bedrock_client(profile_name=AWS_PROFILE, region=AWS_REGION)

    hashes = embed_work["_HASH"].tolist()

    already_done = [h for h in hashes if h in cache]
    still_needed = sorted({h for h in hashes if h not in cache})
    print(f"Already cached: {len(already_done):,} (resuming from prior run)")
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
        # (ever-growing) cache dict every time. The full cache file is only
        # rewritten once, at the very end (consolidate_cache_delta below).
        seen_keys = set(cache.keys())

        def on_checkpoint(c, processed):
            nonlocal seen_keys
            current_keys = set(c.keys())
            new_keys = current_keys - seen_keys
            delta = {k: c[k] for k in new_keys}
            print(f"  Checkpoint: {processed:,} embedded — appending {len(delta):,} new entries to delta log...")
            append_cache_delta(delta, CACHE_PATH)
            seen_keys = current_keys

        embed_texts_from_cache(
            texts            = [hash_to_text[h] for h in still_needed],
            text_hashes      = still_needed,
            cache            = cache,
            client           = bedrock,
            model_id         = MODEL_ID,
            show_progress    = True,
            max_workers      = EMBED_WORKERS,
            checkpoint_every = EMBED_CHECKPOINT,
            on_checkpoint    = on_checkpoint,
        )
        print("Saving final cache...")
        consolidate_cache_delta(cache, CACHE_PATH)

    # Classify whatever is *currently* cache-hit within embed_work — this run's batch
    # plus anything embedded in an earlier --limit'd run — not the full embed_work set,
    # since a capped run leaves most of it still uncached (cache[h] would KeyError on
    # those). An uncapped run ends up classifying everything anyway, since the whole
    # set becomes cache-hit.
    cached_mask = [h in cache for h in hashes]
    n_ready = sum(cached_mask)
    ready_df = embed_work[cached_mask].reset_index(drop=True)
    ready_hashes = [h for h, m in zip(hashes, cached_mask) if m]
    print(f"\nClassifying {n_ready:,} of {len(embed_work):,} net-new services now embedded/cached "
          f"({len(embed_work) - n_ready:,} still awaiting embedding)...")

    records = []
    for start in range(0, len(ready_df), CLASSIFY_BATCH):
        end          = min(start + CLASSIFY_BATCH, len(ready_df))
        batch_hashes = ready_hashes[start:end]
        vecs    = np.array([cache[h] for h in batch_hashes], dtype=np.float32)
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
    print("\n=== PHASE PUBLISH: Upsert into Snowflake ===")

    # See classify_products.py's phase_publish for why this check exists — a stale
    # result file from a much earlier run can otherwise get silently picked up here.
    reference_mtime = CACHED_RESULTS.stat().st_mtime if CACHED_RESULTS.exists() else None
    STALE_THRESHOLD_SECONDS = 24 * 3600

    parts = []
    for label, path in [
        ("Phase cached", CACHED_RESULTS),
        ("Phase embed",  PHASE_EMBED_RESULTS),
    ]:
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            stale_flag = ""
            if reference_mtime is not None and abs(path.stat().st_mtime - reference_mtime) > STALE_THRESHOLD_SECONDS:
                stale_flag = "  *** POSSIBLY STALE — modified >24h apart from Phase cached's result. Verify this wasn't left over from an earlier run before trusting this publish. ***"
            df = pd.read_csv(path, low_memory=False)
            parts.append(df)
            print(f"  {label}: {len(df):,} rows (file modified {mtime:%Y-%m-%d %H:%M}){stale_flag}")
        else:
            print(f"  WARNING: {path.name} not found — skipping")

    if not parts:
        print("ERROR: No phase results found.")
        sys.exit(1)

    # Drop genuinely-empty (0-row) parts before concatenating — mixing them in
    # triggers a pandas dtype-inference quirk on boolean columns.
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
    upsert_publish(sf, combined, OUTPUT_TABLE, PUBLISH_CHUNK)

    print("\nColumns written:")
    for col in combined.columns:
        print(f"  {col}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env",   choices=["stage", "prod"], required=True,
                        help="Which environment to classify (stage or prod)")
    parser.add_argument("--phase", choices=["cached", "embed", "publish"], required=True,
                        help="Which phase to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Phase embed only: cap how many net-new rows get embedded via Bedrock "
                             "this run, for incremental batching. Omit to embed everything still needed.")
    args = parser.parse_args()

    cfg = ENV_CONFIGS[args.env]
    INPUT_TABLE  = cfg["input_table"]
    OUTPUT_TABLE = cfg["output_table"]
    CACHE_PATH   = cfg["cache_path"]
    OUT_DIR      = cfg["out_dir"]
    EMBED_LIMIT  = args.limit
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    CACHED_RESULTS      = OUT_DIR / "phase_cached_results.csv"
    EMBED_WORK          = OUT_DIR / "phase_embed_work.parquet"
    PHASE_EMBED_RESULTS = OUT_DIR / "phase_embed_results.csv"

    print(f"Environment: {args.env.upper()}")
    print(f"  Input:  {INPUT_TABLE}")
    print(f"  Output: {OUTPUT_TABLE}")
    print(f"  Cache:  {CACHE_PATH}")
    print(f"  Artifacts: {OUT_DIR}")

    if args.phase == "cached":
        phase_cached()
    elif args.phase == "embed":
        phase_embed()
    elif args.phase == "publish":
        phase_publish()
