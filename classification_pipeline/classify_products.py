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
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "classification_pipeline"))

from product_classifier_utils import (
    build_product_text,
    embed_texts_from_cache,
    get_bedrock_client,
    get_snowflake_session,
    stable_text_hash,
)

# ── Static config (shared across all environments) ────────────────────────────
AWS_PROFILE      = "staging.admin"
AWS_REGION       = "us-east-1"
MODEL_ID         = "amazon.titan-embed-text-v1"
MARGIN_THRESHOLD = 0.05
EMBED_WORKERS    = 5         # parallel Bedrock workers for net-new products
EMBED_CHECKPOINT = 1_000     # save env cache every N new embeddings
PUBLISH_CHUNK    = 500_000   # rows per Snowflake append

L3_ANCHOR_TABLE = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L3_DESCRIPTIONS"
L4_ANCHOR_TABLE = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L4_DESCRIPTIONS"

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


# ── Anchor loading from Snowflake ─────────────────────────────────────────────

def load_anchors_from_snowflake(sf):
    """
    Load pre-embedded L3 and L4 anchor vectors from Snowflake.
    Returns:
        l3_anchors: (ids, labels, normed_matrix)
        l4_by_l3:   dict of l3_id -> (ids, labels, normed_matrix)
    """
    print("Loading L3 anchor vectors from Snowflake...")
    l3_df = sf.sql(
        f"SELECT ASSIGNED_NEW_L3_ID, ASSIGNED_NEW_L3_LABEL, L3_EMBED FROM {L3_ANCHOR_TABLE}"
    ).to_pandas()
    l3_ids    = l3_df["ASSIGNED_NEW_L3_ID"].tolist()
    l3_labels = l3_df["ASSIGNED_NEW_L3_LABEL"].tolist()
    l3_vecs   = np.array([json.loads(e) for e in l3_df["L3_EMBED"]], dtype=np.float32)
    l3_norms  = np.linalg.norm(l3_vecs, axis=1, keepdims=True)
    l3_normed = l3_vecs / np.clip(l3_norms, 1e-10, None)
    print(f"  L3 anchors: {len(l3_ids)} categories")

    print("Loading L4 anchor vectors from Snowflake...")
    l4_df = sf.sql(f"""
        SELECT ASSIGNED_NEW_L3_ID, ASSIGNED_L4_ID, ASSIGNED_L4_LABEL, L4_EMBED
        FROM {L4_ANCHOR_TABLE}
        ORDER BY ASSIGNED_NEW_L3_ID, L4_ID
    """).to_pandas()

    l4_by_l3 = {}
    for l3_id, grp in l4_df.groupby("ASSIGNED_NEW_L3_ID"):
        ids    = grp["ASSIGNED_L4_ID"].tolist()
        labels = grp["ASSIGNED_L4_LABEL"].tolist()
        vecs   = np.array([json.loads(e) for e in grp["L4_EMBED"]], dtype=np.float32)
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / np.clip(norms, 1e-10, None)
        l4_by_l3[l3_id] = (ids, labels, normed)
    print(f"  L4 anchors: {sum(len(v[0]) for v in l4_by_l3.values())} subcategories across {len(l4_by_l3)} L3s")

    return (l3_ids, l3_labels, l3_normed), l4_by_l3


# ── Classification ────────────────────────────────────────────────────────────

def classify_against_anchors(vecs, anchor_ids, anchor_labels, anchor_normed):
    """Cosine similarity classification. Returns (ids, labels, scores, margins, low_conf)."""
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.clip(norms, 1e-10, None)
    sim    = vecs_n @ anchor_normed.T
    top1   = sim.argmax(axis=1)
    top1_s = sim[np.arange(len(sim)), top1]
    sim2   = sim.copy()
    sim2[np.arange(len(sim)), top1] = -1
    margin = top1_s - sim2.max(axis=1)
    return (
        [anchor_ids[i]    for i in top1],
        [anchor_labels[i] for i in top1],
        top1_s.round(4),
        margin.round(4),
        margin < MARGIN_THRESHOLD,
    )


def classify_l3_and_l4(vecs, l3_anchors, l4_by_l3):
    """
    Run L3 then L4 classification in one pass.
    Returns all 10 result arrays (L3 + L4 each: ids, labels, scores, margins, low_conf).
    """
    l3_ids, l3_labels, l3_scores, l3_margins, l3_low_conf = classify_against_anchors(vecs, *l3_anchors)

    n = len(vecs)
    l4_ids      = [None] * n
    l4_labels   = [None] * n
    l4_scores   = np.zeros(n, dtype=np.float32)
    l4_margins  = np.zeros(n, dtype=np.float32)
    l4_low_conf = np.ones(n, dtype=bool)

    for unique_l3_id in set(l3_ids):
        if unique_l3_id not in l4_by_l3:
            continue
        idx      = [i for i, lid in enumerate(l3_ids) if lid == unique_l3_id]
        sub_vecs = vecs[np.array(idx)]
        s_ids, s_labels, s_scores, s_margins, s_low_conf = classify_against_anchors(
            sub_vecs, *l4_by_l3[unique_l3_id]
        )
        for pos, i in enumerate(idx):
            l4_ids[i]      = s_ids[pos]
            l4_labels[i]   = s_labels[pos]
            l4_scores[i]   = s_scores[pos]
            l4_margins[i]  = s_margins[pos]
            l4_low_conf[i] = s_low_conf[pos]

    return (
        l3_ids, l3_labels, l3_scores, l3_margins, l3_low_conf,
        l4_ids, l4_labels, l4_scores, l4_margins, l4_low_conf,
    )


def attach_classifications(batch_df, results):
    """Attach L3 + L4 classification columns to a DataFrame copy."""
    (l3_ids, l3_labels, l3_scores, l3_margins, l3_low_conf,
     l4_ids, l4_labels, l4_scores, l4_margins, l4_low_conf) = results

    out = batch_df.copy()
    out["ASSIGNED_NEW_L3_ID"]    = l3_ids
    out["ASSIGNED_NEW_L3_LABEL"] = l3_labels
    out["L3_CONFIDENCE"]         = l3_scores
    out["L3_CONFIDENCE_MARGIN"]  = l3_margins
    out["L3_IS_LOW_CONFIDENCE"]  = l3_low_conf
    out["ASSIGNED_L4_ID"]        = l4_ids
    out["ASSIGNED_L4_LABEL"]     = l4_labels
    out["L4_CONFIDENCE"]         = l4_scores
    out["L4_CONFIDENCE_MARGIN"]  = l4_margins
    out["L4_IS_LOW_CONFIDENCE"]  = l4_low_conf
    return out


# ── Env cache helpers ─────────────────────────────────────────────────────────

def load_env_cache():
    if CACHE_ENV_PATH.exists():
        print(f"Loading env cache ({CACHE_ENV_PATH.stat().st_size/1e9:.2f} GB)...")
        with open(CACHE_ENV_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"  Env cache: {len(cache):,} entries")
        return cache
    print("Env cache not found — starting fresh.")
    return {}


def save_env_cache(cache):
    """Atomic write: write to temp file then rename."""
    tmp = CACHE_ENV_PATH.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    tmp.rename(CACHE_ENV_PATH)


# ── Load listings from Snowflake ──────────────────────────────────────────────

def load_listings(sf):
    print(f"Loading listings from {INPUT_TABLE}...")
    df = sf.sql(f"""
        SELECT PRODUCT_ID, PRODUCT_VARIANT_ID, PRODUCT_NAME, DESCRIPTION, PRICING_STATUS_C, LIST_PRICE_C, PRODUCT_SEGMENT
        FROM {INPUT_TABLE}
    """).to_pandas()
    df["PRODUCT_ID"] = df["PRODUCT_ID"].astype(str)
    df = df.rename(columns={"PRODUCT_SEGMENT": "SOURCE"})
    print(f"Loaded: {len(df):,} rows")

    has_text = df["PRODUCT_NAME"].notna() | df["DESCRIPTION"].notna()
    df = df[has_text].copy().reset_index(drop=True)
    print(f"Rows with usable text: {len(df):,}")
    return df


def _sf_session():
    sf = get_snowflake_session()
    sf.sql("USE ROLE \"DEPT-ENGINEERING\"").collect()
    sf.sql("USE DATABASE SNOWFLAKE_LEARNING_DB").collect()
    sf.sql("USE SCHEMA SMCMAHON_PRODUCTS").collect()
    return sf


# ── Phase A ───────────────────────────────────────────────────────────────────

def phase_a():
    print("\n=== PHASE A: Classify v2 + env cache hits ===")

    sf = _sf_session()
    l3_anchors, l4_by_l3 = load_anchors_from_snowflake(sf)
    df = load_listings(sf)

    texts  = build_product_text(df).tolist()
    hashes = [stable_text_hash(t) for t in texts]

    print(f"\nLoading volume 2 ({CACHE_V2_PATH.stat().st_size/1e9:.1f} GB)...")
    with open(CACHE_V2_PATH, "rb") as f:
        cache_v2 = pickle.load(f)
    print(f"Volume 2: {len(cache_v2):,} entries")

    print("Loading volume 1 key index...")
    with open(CACHE_KEYS_PATH, "rb") as f:
        cache_v1_keys = pickle.load(f)

    cache_env = load_env_cache()

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

    sf = _sf_session()
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

    sf = _sf_session()
    l3_anchors, l4_by_l3 = load_anchors_from_snowflake(sf)

    cache_env = load_env_cache()
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
            save_env_cache(cache)

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
        save_env_cache(cache_env)

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
    sf = _sf_session()

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
