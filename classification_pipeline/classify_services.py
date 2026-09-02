"""
Classify quoted-service listings into L3 + L4 taxonomy, using the same
taxonomy anchors as classify_products.py.

Services are a distinct, actively-changing corpus with their own dedicated
embedding cache — this pipeline does NOT check the frozen product embedding
volumes (v1/v2) used by classify_products.py.

Key design:
  - Anchor vectors loaded from Snowflake (shared with classify_products.py)
  - L3 + L4 classification happen in the same pass
  - All listings looked up against one dedicated services cache; net-new
    services embedded with parallel Bedrock workers
  - New embeddings saved to a per-env incremental cache, checkpointed every 1,000
    (checkpoints append only new-since-last-checkpoint entries to a small delta
    log — see append_cache_delta/consolidate_cache_delta in
    product_classifier_utils.py — so cost stays cheap regardless of cache size;
    the full cache file is only rewritten once, at the very end of a run)
  - Results written to Snowflake in 500K-row chunks

Run order:
    python classify_services.py --env stage --phase embed    # embed & classify services
    python classify_services.py --env stage --phase publish  # write to Snowflake
"""

import argparse
import sys
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
)

# ── Static config (shared across all environments) ────────────────────────────
AWS_PROFILE      = "staging.admin"
AWS_REGION       = "us-east-1"
MODEL_ID         = "amazon.titan-embed-text-v1"
EMBED_WORKERS    = 5         # parallel Bedrock workers for net-new services
EMBED_CHECKPOINT = 1_000     # save cache every N new embeddings
PUBLISH_CHUNK    = 500_000   # rows per Snowflake append
CLASSIFY_BATCH   = 100_000   # rows per classification batch

# ── Environment configs ───────────────────────────────────────────────────────
# NOTE: the "prod" entry is a naming placeholder — no prod services table
# exists yet. Confirm/adjust these names once it's created.
ENV_CONFIGS = {
    "stage": {
        "input_table":  "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES_V1_STAGE",
        "output_table": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.CLASSIFICATIONS_SERVICES_V1_STAGE",
        "cache_path":   PROJECT_ROOT / "artifacts/cache/embedding_cache_services_stage.pkl",
        "out_dir":      PROJECT_ROOT / "artifacts/analysis/stage_services_classification",
    },
    "prod": {
        "input_table":  "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES_PROD_V1",
        "output_table": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.CLASSIFICATIONS_SERVICES_V1_PROD",
        "cache_path":   PROJECT_ROOT / "artifacts/cache/embedding_cache_services_prod.pkl",
        "out_dir":      PROJECT_ROOT / "artifacts/analysis/prod_services_classification",
    },
}

# ── Runtime globals (set in __main__ after --env is resolved) ─────────────────
INPUT_TABLE         = None
OUTPUT_TABLE        = None
CACHE_PATH          = None
OUT_DIR             = None
PHASE_EMBED_RESULTS = None


# ── Phase embed ───────────────────────────────────────────────────────────────

def phase_embed():
    print("\n=== PHASE EMBED: Embed & classify service listings ===")

    sf = get_products_session()
    l3_anchors, l4_by_l3 = load_anchors_from_snowflake(sf)
    df = load_listings(sf, INPUT_TABLE)

    texts  = build_product_text(df).tolist()
    hashes = [stable_text_hash(t) for t in texts]

    cache   = load_pickle_cache_with_delta(CACHE_PATH)
    bedrock = get_bedrock_client(profile_name=AWS_PROFILE, region=AWS_REGION)

    already_done = [h for h in hashes if h in cache]
    still_needed = sorted({h for h in hashes if h not in cache})
    print(f"Already cached: {len(already_done):,}")
    print(f"Need embedding: {len(still_needed):,}")

    if still_needed:
        print(f"\nEmbedding {len(still_needed):,} texts with {EMBED_WORKERS} parallel workers...")

        hash_to_text = {h: t for h, t in zip(hashes, texts)}

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

    print(f"\nClassifying {len(df):,} services...")
    records = []
    for start in range(0, len(df), CLASSIFY_BATCH):
        end          = min(start + CLASSIFY_BATCH, len(df))
        batch_hashes = hashes[start:end]
        vecs    = np.array([cache[h] for h in batch_hashes], dtype=np.float32)
        results = classify_l3_and_l4(vecs, l3_anchors, l4_by_l3)
        records.append(attach_classifications(df.iloc[start:end], results))
        hi  = (~results[4]).sum()
        pct = end / len(df) * 100
        print(f"  batch {start:,}–{end:,} ({pct:.0f}%) — L3 high-conf: {hi:,}/{end-start:,}")

    results_df = pd.concat(records, ignore_index=True)
    results_df.to_csv(PHASE_EMBED_RESULTS, index=False)
    hi = (~results_df["L3_IS_LOW_CONFIDENCE"]).sum()
    print(f"\nPhase embed saved: {PHASE_EMBED_RESULTS} ({len(results_df):,} rows)")
    print(f"L3 high-confidence: {hi:,}/{len(results_df):,} ({hi/len(results_df)*100:.1f}%)")


# ── Phase publish ─────────────────────────────────────────────────────────────

def phase_publish():
    print("\n=== PHASE PUBLISH: Write to Snowflake ===")
    if not PHASE_EMBED_RESULTS.exists():
        print("ERROR: Run phase embed first.")
        sys.exit(1)

    combined = pd.read_csv(PHASE_EMBED_RESULTS, low_memory=False)
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
    parser.add_argument("--env",   choices=["stage", "prod"], required=True,
                        help="Which environment to classify (stage or prod)")
    parser.add_argument("--phase", choices=["embed", "publish"], required=True,
                        help="Which phase to run")
    args = parser.parse_args()

    cfg = ENV_CONFIGS[args.env]
    INPUT_TABLE  = cfg["input_table"]
    OUTPUT_TABLE = cfg["output_table"]
    CACHE_PATH   = cfg["cache_path"]
    OUT_DIR      = cfg["out_dir"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PHASE_EMBED_RESULTS = OUT_DIR / "phase_embed_results.csv"

    print(f"Environment: {args.env.upper()}")
    print(f"  Input:  {INPUT_TABLE}")
    print(f"  Output: {OUTPUT_TABLE}")
    print(f"  Cache:  {CACHE_PATH}")
    print(f"  Artifacts: {OUT_DIR}")

    if args.phase == "embed":
        phase_embed()
    elif args.phase == "publish":
        phase_publish()
