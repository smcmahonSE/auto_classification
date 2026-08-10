"""
Classify product listings into L3 + L4 taxonomy in a single pipeline.

Supports both staging and prod environments via --env. All environment-specific
table names, cache paths, and artifact directories are resolved from ENV_CONFIGS.

Key design:
  - Anchor vectors loaded from Snowflake (no Bedrock calls for anchors)
  - L3 + L4 classification happen in the same pass
  - Reuses existing prod embedding caches (v1, v2) for overlapping products
  - Net-new products embedded with parallel Bedrock workers (max_workers=10)
  - New embeddings saved to a per-env incremental cache, checkpointed every 500
  - Results written to Snowflake in 500K-row chunks

Run order:
    python classify_products.py --env stage --phase a        # classify v2 + env cache hits
    python classify_products.py --env stage --phase extract  # extract v1 vectors to .npy
    python classify_products.py --env stage --phase b        # classify v1 cache hits
    python classify_products.py --env stage --phase embed    # embed & classify net-new products
    python classify_products.py --env stage --phase publish  # write to Snowflake
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "classification_pipeline"))

from product_classifier_utils import (
    attach_classifications,
    build_product_text,
    classify_l3_and_l4,
    embed_texts_from_cache,
    get_bedrock_client,
    get_products_session,
    load_anchors_from_snowflake,
    load_listings,
    load_pickle_cache,
    save_pickle_cache,
    stable_text_hash,
)

# ── Static config (shared across all environments) ────────────────────────────
AWS_PROFILE      = "staging.admin"
AWS_REGION       = "us-east-1"
MODEL_ID         = "amazon.titan-embed-text-v1"
EMBED_WORKERS    = 5         # parallel Bedrock workers for net-new products
EMBED_CHECKPOINT = 1_000     # save env cache every N new embeddings
PUBLISH_CHUNK    = 500_000   # rows per Snowflake append

CACHE_V1_PATH   = PROJECT_ROOT / "artifacts/cache/embedding_cache.pkl"
CACHE_V2_PATH   = PROJECT_ROOT / "artifacts/cache/embedding_cache_new.pkl"
CACHE_KEYS_PATH = PROJECT_ROOT / "artifacts/cache/embedding_cache_keys.pkl"

# ── Environment configs ───────────────────────────────────────────────────────
ENV_CONFIGS = {
    "stage": {
        "input_table":  "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_STAGE",
        "output_table": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_CLASSIFICATIONS_STAGE",
        "cache_path":   PROJECT_ROOT / "artifacts/cache/embedding_cache_stage.pkl",
        "out_dir":      PROJECT_ROOT / "artifacts/analysis/stage_classification",
    },
    "prod": {
        "input_table":  "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_PROD",
        "output_table": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_CLASSIFICATIONS_PROD",
        "cache_path":   PROJECT_ROOT / "artifacts/cache/embedding_cache_prod_new.pkl",
        "out_dir":      PROJECT_ROOT / "artifacts/analysis/prod_classification",
    },
}

# ── Runtime globals (set in __main__ after --env is resolved) ─────────────────
# These are referenced by the phase functions below.
INPUT_TABLE      = None
OUTPUT_TABLE     = None
CACHE_ENV_PATH   = None
OUT_DIR          = None
PHASE_A_RESULTS  = None
V1_WORK          = None
V1_VECTORS       = None
PHASE_B_RESULTS  = None
EMBED_WORK       = None
PHASE_EMBED_RESULTS = None


# ── Phase A ───────────────────────────────────────────────────────────────────

def phase_a():
    print("\n=== PHASE A: Classify v2 + env cache hits ===")

    sf = get_products_session()
    l3_anchors, l4_by_l3 = load_anchors_from_snowflake(sf)
    df = load_listings(sf, INPUT_TABLE)

    texts  = build_product_text(df).tolist()
    hashes = [stable_text_hash(t) for t in texts]

    print(f"\nLoading volume 2 ({CACHE_V2_PATH.stat().st_size/1e9:.1f} GB)...")
    with open(CACHE_V2_PATH, "rb") as f:
        cache_v2 = pickle.load(f)
    print(f"Volume 2: {len(cache_v2):,} entries")

    print("Loading volume 1 key index...")
    with open(CACHE_KEYS_PATH, "rb") as f:
        cache_v1_keys = pickle.load(f)

    cache_env = load_pickle_cache(CACHE_ENV_PATH)

    in_v2      = [h in cache_v2                                                              for h in hashes]
    in_v1_only = [h in cache_v1_keys and not in_v2[i]                                       for i, h in enumerate(hashes)]
    in_env     = [h in cache_env and not in_v2[i] and not in_v1_only[i]                     for i, h in enumerate(hashes)]
    in_none    = [not in_v2[i] and not in_v1_only[i] and not in_env[i]                      for i, h in enumerate(hashes)]

    v2_idx    = [i for i, m in enumerate(in_v2)      if m]
    v1_idx    = [i for i, m in enumerate(in_v1_only) if m]
    env_idx   = [i for i, m in enumerate(in_env)     if m]
    miss_idx  = [i for i, m in enumerate(in_none)    if m]

    print(f"\nIn volume 2:    {len(v2_idx):,}")
    print(f"In volume 1:    {len(v1_idx):,}")
    print(f"In env cache:   {len(env_idx):,}")
    print(f"In neither:     {len(miss_idx):,}  ← will be embedded in phase embed")

    BATCH = 250_000
    records = []

    for start in range(0, len(v2_idx), BATCH):
        idx_batch    = v2_idx[start:start + BATCH]
        batch_hashes = [hashes[i] for i in idx_batch]
        vecs    = np.array([cache_v2[h] for h in batch_hashes], dtype=np.float32)
        results = classify_l3_and_l4(vecs, l3_anchors, l4_by_l3)
        records.append(attach_classifications(df.iloc[idx_batch], results))
        pct = (start + len(idx_batch)) / max(len(v2_idx), 1) * 100
        hi  = (~results[4]).sum()
        print(f"  v2 batch {start:,}–{start+len(idx_batch):,} ({pct:.0f}%) — L3 high-conf: {hi:,}/{len(idx_batch):,}")
        del vecs
    del cache_v2

    if env_idx:
        for start in range(0, len(env_idx), BATCH):
            idx_batch    = env_idx[start:start + BATCH]
            batch_hashes = [hashes[i] for i in idx_batch]
            vecs    = np.array([cache_env[h] for h in batch_hashes], dtype=np.float32)
            results = classify_l3_and_l4(vecs, l3_anchors, l4_by_l3)
            records.append(attach_classifications(df.iloc[idx_batch], results))
            pct = (start + len(idx_batch)) / max(len(env_idx), 1) * 100
            hi  = (~results[4]).sum()
            print(f"  env batch {start:,}–{start+len(idx_batch):,} ({pct:.0f}%) — L3 high-conf: {hi:,}/{len(idx_batch):,}")
            del vecs
    del cache_env

    if records:
        phase_a_df = pd.concat(records, ignore_index=True)
        phase_a_df.to_csv(PHASE_A_RESULTS, index=False)
        hi = (~phase_a_df["L3_IS_LOW_CONFIDENCE"]).sum()
        print(f"\nPhase A saved: {PHASE_A_RESULTS} ({len(phase_a_df):,} rows)")
        print(f"L3 high-confidence: {hi:,}/{len(phase_a_df):,} ({hi/len(phase_a_df)*100:.1f}%)")
    else:
        print("Phase A: no v2 or env cache hits.")

    v1_work = df.iloc[v1_idx].copy()
    v1_work["_HASH"] = [hashes[i] for i in v1_idx]
    v1_work.to_parquet(V1_WORK, index=False)
    print(f"Phase B work file: {V1_WORK} ({len(v1_work):,} rows)")

    embed_work = df.iloc[miss_idx].copy()
    embed_work["_HASH"] = [hashes[i] for i in miss_idx]
    embed_work.to_parquet(EMBED_WORK, index=False)
    print(f"Embed work file:   {EMBED_WORK} ({len(embed_work):,} rows)")


# ── Phase extract ─────────────────────────────────────────────────────────────

def phase_extract():
    print("\n=== PHASE EXTRACT: Extract v1 vectors to memmap ===")
    if not V1_WORK.exists():
        print("ERROR: Run phase a first.")
        sys.exit(1)

    v1_work = pd.read_parquet(V1_WORK)
    hashes  = v1_work["_HASH"].tolist()
    print(f"Need vectors for {len(hashes):,} listings from volume 1")

    print(f"Loading volume 1 ({CACHE_V1_PATH.stat().st_size/1e9:.1f} GB)...")
    with open(CACHE_V1_PATH, "rb") as f:
        cache_v1 = pickle.load(f)
    print(f"Volume 1: {len(cache_v1):,} entries")

    v1_key_set = set(cache_v1.keys())
    not_found  = [h for h in hashes if h not in v1_key_set]
    if not_found:
        print(f"WARNING: {len(not_found):,} hashes not found in volume 1 — dropping")
        valid   = np.array([h in v1_key_set for h in hashes])
        hashes  = [h for h, m in zip(hashes, valid) if m]
        v1_work = v1_work[valid].reset_index(drop=True)
        v1_work.to_parquet(V1_WORK, index=False)

    n, dim = len(hashes), 1536
    print(f"Extracting {n:,} vectors ({n * dim * 4 / 1e9:.2f} GB)...")
    mmap  = np.lib.format.open_memmap(str(V1_VECTORS), mode="w+", dtype=np.float32, shape=(n, dim))
    CHUNK = 100_000
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        for i, h in enumerate(hashes[start:end]):
            mmap[start + i] = cache_v1[h]
        mmap.flush()
        print(f"  wrote {end:,}/{n:,} ({end/n*100:.0f}%)")

    del cache_v1, mmap
    print(f"Saved: {V1_VECTORS} ({V1_VECTORS.stat().st_size/1e9:.2f} GB)")


# ── Phase B ───────────────────────────────────────────────────────────────────

def phase_b():
    print("\n=== PHASE B: Classify from v1 vectors ===")
    for p in [V1_WORK, V1_VECTORS]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run phase a and extract first.")
            sys.exit(1)

    sf = get_products_session()
    l3_anchors, l4_by_l3 = load_anchors_from_snowflake(sf)

    vectors = np.load(V1_VECTORS, mmap_mode="r")
    v1_work = pd.read_parquet(V1_WORK)
    print(f"Vectors: {vectors.shape}")

    BATCH = 250_000
    records = []
    for start in range(0, len(v1_work), BATCH):
        end     = min(start + BATCH, len(v1_work))
        vecs    = np.array(vectors[start:end], dtype=np.float32)
        results = classify_l3_and_l4(vecs, l3_anchors, l4_by_l3)
        batch_df = v1_work.iloc[start:end].drop(columns=["_HASH"], errors="ignore")
        records.append(attach_classifications(batch_df, results))
        hi  = (~results[4]).sum()
        pct = end / len(v1_work) * 100
        print(f"  batch {start:,}–{end:,} ({pct:.0f}%) — L3 high-conf: {hi:,}/{end-start:,}")

    v1_results = pd.concat(records, ignore_index=True)
    v1_results.to_csv(PHASE_B_RESULTS, index=False)
    hi = (~v1_results["L3_IS_LOW_CONFIDENCE"]).sum()
    print(f"\nPhase B saved: {PHASE_B_RESULTS} ({len(v1_results):,} rows)")
    print(f"L3 high-confidence: {hi:,}/{len(v1_results):,} ({hi/len(v1_results)*100:.1f}%)")


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

    cache_env = load_pickle_cache(CACHE_ENV_PATH)
    bedrock   = get_bedrock_client(profile_name=AWS_PROFILE, region=AWS_REGION)

    hashes = embed_work["_HASH"].tolist()

    already_done = [h for h in hashes if h in cache_env]
    still_needed = [h for h in hashes if h not in cache_env]
    print(f"Already in env cache: {len(already_done):,} (resuming from prior run)")
    print(f"Still need embedding: {len(still_needed):,}")

    if still_needed:
        print(f"\nEmbedding {len(still_needed):,} texts with {EMBED_WORKERS} parallel workers...")

        all_texts    = build_product_text(embed_work).tolist()
        hash_to_text = {h: t for h, t in zip(hashes, all_texts)}

        def on_checkpoint(cache, processed):
            print(f"  Checkpoint: {processed:,} embedded — saving env cache...")
            save_pickle_cache(cache, CACHE_ENV_PATH)

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
        save_pickle_cache(cache_env, CACHE_ENV_PATH)

    print(f"\nClassifying {len(embed_work):,} net-new products...")
    BATCH = 100_000
    records = []
    for start in range(0, len(embed_work), BATCH):
        end          = min(start + BATCH, len(embed_work))
        batch_hashes = [hashes[i] for i in range(start, end)]
        vecs    = np.array([cache_env[h] for h in batch_hashes], dtype=np.float32)
        results = classify_l3_and_l4(vecs, l3_anchors, l4_by_l3)
        batch_df = embed_work.iloc[start:end].drop(columns=["_HASH"], errors="ignore")
        records.append(attach_classifications(batch_df, results))
        hi  = (~results[4]).sum()
        pct = end / len(embed_work) * 100
        print(f"  batch {start:,}–{end:,} ({pct:.0f}%) — L3 high-conf: {hi:,}/{end-start:,}")

    embed_results = pd.concat(records, ignore_index=True)
    embed_results.to_csv(PHASE_EMBED_RESULTS, index=False)
    hi = (~embed_results["L3_IS_LOW_CONFIDENCE"]).sum()
    print(f"\nPhase embed saved: {PHASE_EMBED_RESULTS} ({len(embed_results):,} rows)")
    print(f"L3 high-confidence: {hi:,}/{len(embed_results):,} ({hi/len(embed_results)*100:.1f}%)")


# ── Phase publish ─────────────────────────────────────────────────────────────

def phase_publish():
    print("\n=== PHASE PUBLISH: Write to Snowflake ===")

    parts = []
    for label, path in [
        ("Phase A",    PHASE_A_RESULTS),
        ("Phase B",    PHASE_B_RESULTS),
        ("Phase embed", PHASE_EMBED_RESULTS),
    ]:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            parts.append(df)
            print(f"  {label}: {len(df):,} rows")
        else:
            print(f"  WARNING: {path.name} not found — skipping")

    if not parts:
        print("ERROR: No phase results found.")
        sys.exit(1)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset="PRODUCT_ID", keep="last")
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
    parser.add_argument("--phase", choices=["a", "extract", "b", "embed", "publish"], required=True,
                        help="Which phase to run")
    args = parser.parse_args()

    # Resolve environment config into module-level globals so phase functions pick them up
    cfg = ENV_CONFIGS[args.env]
    INPUT_TABLE   = cfg["input_table"]
    OUTPUT_TABLE  = cfg["output_table"]
    CACHE_ENV_PATH = cfg["cache_path"]
    OUT_DIR        = cfg["out_dir"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PHASE_A_RESULTS     = OUT_DIR / "phase_a_results.csv"
    V1_WORK             = OUT_DIR / "phase_b_v1_work.parquet"
    V1_VECTORS          = OUT_DIR / "phase_b_v1_vectors.npy"
    PHASE_B_RESULTS     = OUT_DIR / "phase_b_results.csv"
    EMBED_WORK          = OUT_DIR / "phase_embed_work.parquet"
    PHASE_EMBED_RESULTS = OUT_DIR / "phase_embed_results.csv"

    print(f"Environment: {args.env.upper()}")
    print(f"  Input:  {INPUT_TABLE}")
    print(f"  Output: {OUTPUT_TABLE}")
    print(f"  Cache:  {CACHE_ENV_PATH}")
    print(f"  Artifacts: {OUT_DIR}")

    if args.phase == "a":
        phase_a()
    elif args.phase == "extract":
        phase_extract()
    elif args.phase == "b":
        phase_b()
    elif args.phase == "embed":
        phase_embed()
    elif args.phase == "publish":
        phase_publish()
