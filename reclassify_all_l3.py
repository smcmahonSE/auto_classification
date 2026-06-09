"""
Re-classify ALL listings against updated L3 anchor descriptions.

Run this whenever l4_taxonomy_anchors.json is updated to regenerate
L3 assignments for the full 7.8M listing catalog.

Same two-phase memory approach as reclassify_residual.py:
  Phase A — classify listings in volume 2 cache (13GB, loads fine)
  Phase B — classify listings in volume 1 cache (32GB, extract first)

Run order:
    python reclassify_all_l3.py --phase a        # classify v2 listings
    python reclassify_all_l3.py --phase extract  # extract v1 vectors to .npy
    python reclassify_all_l3.py --phase b        # classify from .npy
    python reclassify_all_l3.py --phase combine  # merge into new master CSV

Then run:
    python classify_l4.py --phase a  (and extract, b, combine)
    python publish_l4.py
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from product_classifier_utils import (
    build_product_text,
    embed_texts_from_cache,
    get_bedrock_client,
    get_snowflake_session,
    stable_text_hash,
)

# ── Config ─────────────────────────────────────────────────────────────────────
AWS_PROFILE      = "staging.admin"
AWS_REGION       = "us-east-1"
MODEL_ID         = "amazon.titan-embed-text-v1"
MARGIN_THRESHOLD = 0.05

ANCHORS_PATH    = PROJECT_ROOT / "analysis/data/l4_taxonomy_anchors.json"
CACHE_V1_PATH   = PROJECT_ROOT / "artifacts/cache/embedding_cache.pkl"
CACHE_V2_PATH   = PROJECT_ROOT / "artifacts/cache/embedding_cache_new.pkl"
CACHE_KEYS_PATH = PROJECT_ROOT / "artifacts/cache/embedding_cache_keys.pkl"

OUT_DIR         = PROJECT_ROOT / "artifacts/analysis/l3_reclassification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V2_RESULTS      = OUT_DIR / "phase_a_v2_results.csv"
V1_VECTORS      = OUT_DIR / "phase_b_v1_vectors.npy"
V1_WORK         = OUT_DIR / "phase_b_v1_work.parquet"
V1_RESULTS      = OUT_DIR / "phase_b_v1_results.csv"

NEW_MASTER_CSV  = PROJECT_ROOT / "artifacts/analysis/master_classification_results.csv"
NEW_RESIDUAL    = PROJECT_ROOT / "artifacts/analysis/master_residual_for_clustering.csv"

SOURCE_TABLES = {
    "LEI":      "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LEI",
    "SERVICES": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES",
    "LCG":      "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LCG",
}

# ── Load anchors ────────────────────────────────────────────────────────────────

def load_anchors_and_embed():
    print("Embedding updated L3 anchor descriptions...")
    with open(ANCHORS_PATH) as f:
        data = json.load(f)
    anchors       = data["anchors"]
    anchor_ids    = [a["id"]          for a in anchors]
    anchor_labels = [a["label"]       for a in anchors]
    anchor_texts  = [a["description"] for a in anchors]

    bedrock = get_bedrock_client(profile_name=AWS_PROFILE, region=AWS_REGION)
    fresh   = {}
    hashes  = [stable_text_hash(t) for t in anchor_texts]
    matrix  = embed_texts_from_cache(
        texts=anchor_texts, text_hashes=hashes, cache=fresh,
        client=bedrock, model_id=MODEL_ID, show_progress=False, max_workers=1,
    )
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / np.clip(norms, 1e-10, None)
    print(f"Anchors ready: {normed.shape}")
    return anchor_ids, anchor_labels, normed


# ── Load all listings from Snowflake ───────────────────────────────────────────

def load_all_listings():
    """Load all listings from the three source tables."""
    print("Loading all listings from Snowflake source tables...")
    sf = get_snowflake_session()
    frames = []
    for source, table in SOURCE_TABLES.items():
        print(f"  {source}: loading from {table}...")
        src = sf.sql(f"""
            SELECT PRODUCT_ID, PRODUCT_NAME, DESCRIPTION, PRICING_STATUS_C, LIST_PRICE_C
            FROM {table}
        """).to_pandas()
        src["PRODUCT_ID"] = src["PRODUCT_ID"].astype(str)
        src["SOURCE"] = source
        frames.append(src)
        print(f"    {len(src):,} rows")

    df = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(df):,} listings")
    return df


# ── Cosine similarity classification ───────────────────────────────────────────

def classify_batch(vecs, anchor_ids, anchor_labels, anchor_normed):
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


# ── Phase A ─────────────────────────────────────────────────────────────────────

def phase_a():
    print("\n=== PHASE A: Classify v2 listings ===")
    anchor_ids, anchor_labels, anchor_normed = load_anchors_and_embed()

    df = load_all_listings()
    has_text = df["PRODUCT_NAME"].notna() | df["DESCRIPTION"].notna()
    df = df[has_text].copy().reset_index(drop=True)
    print(f"Rows with usable text: {len(df):,}")

    texts  = build_product_text(df).tolist()
    hashes = [stable_text_hash(t) for t in texts]

    print(f"\nLoading volume 2 ({CACHE_V2_PATH.stat().st_size/1e9:.1f} GB)...")
    with open(CACHE_V2_PATH, "rb") as f:
        cache_v2 = pickle.load(f)
    print(f"Volume 2: {len(cache_v2):,} entries")

    print("Loading volume 1 key index...")
    with open(CACHE_KEYS_PATH, "rb") as f:
        cache_v1_keys = pickle.load(f)

    in_v2      = [h in cache_v2      for h in hashes]
    in_v1_only = [h in cache_v1_keys and not in_v2[i] for i, h in enumerate(hashes)]

    v2_indices = [i for i, m in enumerate(in_v2)      if m]
    v1_indices = [i for i, m in enumerate(in_v1_only) if m]
    neither    = len(hashes) - len(v2_indices) - len(v1_indices)
    print(f"In volume 2:      {len(v2_indices):,}")
    print(f"In volume 1 only: {len(v1_indices):,}")
    print(f"In neither:       {neither:,}")

    BATCH = 250_000
    records = []
    for start in range(0, len(v2_indices), BATCH):
        idx_batch    = v2_indices[start:start + BATCH]
        batch_hashes = [hashes[i] for i in idx_batch]
        vecs = np.array([cache_v2[h] for h in batch_hashes], dtype=np.float32)
        ids, labels, scores, margins, low_conf = classify_batch(
            vecs, anchor_ids, anchor_labels, anchor_normed
        )
        batch_df = df.iloc[idx_batch].copy()
        batch_df["ASSIGNED_L4_ID"]        = ids
        batch_df["ASSIGNED_L4_LABEL"]     = labels
        batch_df["CONFIDENCE"]            = scores
        batch_df["CONFIDENCE_MARGIN"]     = margins
        batch_df["IS_LOW_CONFIDENCE"]     = low_conf
        records.append(batch_df)
        high = (~low_conf).sum()
        pct  = (start + len(idx_batch)) / len(v2_indices) * 100
        print(f"  batch {start:,}–{start+len(idx_batch):,} ({pct:.0f}%) — high-conf: {high:,}/{len(idx_batch):,}")
        del vecs

    del cache_v2
    v2_results = pd.concat(records, ignore_index=True)
    v2_results.to_csv(V2_RESULTS, index=False)
    print(f"Phase A saved: {V2_RESULTS} ({len(v2_results):,} rows)")

    # Save v1 work
    v1_work = df.iloc[v1_indices].copy()
    v1_work["_HASH"] = [hashes[i] for i in v1_indices]
    v1_work.to_parquet(V1_WORK, index=False)
    print(f"Phase B work file: {V1_WORK} ({len(v1_work):,} rows)")

    high = (~v2_results["IS_LOW_CONFIDENCE"]).sum()
    print(f"\nPhase A: {high:,}/{len(v2_results):,} high-confidence ({high/len(v2_results)*100:.1f}%)")


# ── Phase extract ───────────────────────────────────────────────────────────────

def phase_extract():
    print("\n=== PHASE EXTRACT: Extract v1 vectors ===")
    if not V1_WORK.exists():
        print("ERROR: Run phase A first.")
        sys.exit(1)

    v1_work = pd.read_parquet(V1_WORK)
    hashes  = v1_work["_HASH"].tolist()
    print(f"Need vectors for {len(hashes):,} listings from volume 1")

    print(f"Loading volume 1 ({CACHE_V1_PATH.stat().st_size/1e9:.1f} GB)...")
    with open(CACHE_V1_PATH, "rb") as f:
        cache_v1 = pickle.load(f)
    print(f"Volume 1: {len(cache_v1):,} entries")

    cache_v1_keys = set(cache_v1.keys())
    not_found = [h for h in hashes if h not in cache_v1_keys]
    if not_found:
        print(f"WARNING: {len(not_found):,} hashes not in volume 1 — skipping")
        valid = np.array([h in cache_v1_keys for h in hashes])
        hashes  = [h for h, m in zip(hashes, valid) if m]
        v1_work = v1_work[valid].reset_index(drop=True)
        v1_work.to_parquet(V1_WORK, index=False)

    n, dim = len(hashes), 1536
    print(f"Extracting {n:,} vectors ({n * dim * 4 / 1e9:.2f} GB)...")
    mmap = np.lib.format.open_memmap(str(V1_VECTORS), mode="w+", dtype=np.float32, shape=(n, dim))
    CHUNK = 100_000
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        for i, h in enumerate(hashes[start:end]):
            mmap[start + i] = cache_v1[h]
        mmap.flush()
        print(f"  wrote {end:,}/{n:,} ({end/n*100:.0f}%)")
    del cache_v1, mmap
    print(f"Saved: {V1_VECTORS} ({V1_VECTORS.stat().st_size/1e9:.2f} GB)")


# ── Phase B ─────────────────────────────────────────────────────────────────────

def phase_b():
    print("\n=== PHASE B: Classify from v1 vectors ===")
    for p in [V1_WORK, V1_VECTORS]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run phase A and extract first.")
            sys.exit(1)

    anchor_ids, anchor_labels, anchor_normed = load_anchors_and_embed()

    vectors = np.load(V1_VECTORS, mmap_mode="r")
    v1_work = pd.read_parquet(V1_WORK)
    print(f"Vectors: {vectors.shape}")

    BATCH = 250_000
    records = []
    for start in range(0, len(v1_work), BATCH):
        end  = min(start + BATCH, len(v1_work))
        vecs = np.array(vectors[start:end], dtype=np.float32)
        ids, labels, scores, margins, low_conf = classify_batch(
            vecs, anchor_ids, anchor_labels, anchor_normed
        )
        batch_df = v1_work.iloc[start:end].drop(columns=["_HASH"], errors="ignore").copy()
        batch_df["ASSIGNED_L4_ID"]    = ids
        batch_df["ASSIGNED_L4_LABEL"] = labels
        batch_df["CONFIDENCE"]        = scores
        batch_df["CONFIDENCE_MARGIN"] = margins
        batch_df["IS_LOW_CONFIDENCE"] = low_conf
        records.append(batch_df)
        high = (~low_conf).sum()
        pct  = end / len(v1_work) * 100
        print(f"  batch {start:,}–{end:,} ({pct:.0f}%) — high-conf: {high:,}/{end-start:,}")

    v1_results = pd.concat(records, ignore_index=True)
    v1_results.to_csv(V1_RESULTS, index=False)
    print(f"Phase B saved: {V1_RESULTS} ({len(v1_results):,} rows)")
    high = (~v1_results["IS_LOW_CONFIDENCE"]).sum()
    print(f"Phase B: {high:,}/{len(v1_results):,} high-confidence ({high/len(v1_results)*100:.1f}%)")


# ── Combine ──────────────────────────────────────────────────────────────────────

def phase_combine():
    print("\n=== COMBINE: Build new master CSV ===")
    parts = []
    for p in [V2_RESULTS, V1_RESULTS]:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            parts.append(df)
            print(f"  {p.name}: {len(df):,} rows")
        else:
            print(f"  WARNING: {p.name} not found — skipping")

    if not parts:
        print("ERROR: No phase results found.")
        sys.exit(1)

    master = pd.concat(parts, ignore_index=True)
    master = master.drop_duplicates(subset="PRODUCT_ID", keep="last")

    high = (~master["IS_LOW_CONFIDENCE"]).sum()
    low  =   master["IS_LOW_CONFIDENCE"].sum()
    print(f"\nNew master: {len(master):,} rows")
    print(f"High-confidence: {high:,} ({high/len(master)*100:.1f}%)")
    print(f"Residual:        {low:,} ({low/len(master)*100:.1f}%)")

    print("\nL3 distribution:")
    print(master["ASSIGNED_L4_LABEL"].value_counts().to_string())

    master.to_csv(NEW_MASTER_CSV, index=False)
    print(f"\nMaster CSV saved: {NEW_MASTER_CSV}")

    residual = master[master["IS_LOW_CONFIDENCE"]].copy()
    residual.to_csv(NEW_RESIDUAL, index=False)
    print(f"Residual CSV saved: {NEW_RESIDUAL} ({len(residual):,} rows)")
    print("\nNext: run classify_l4.py --phase a (and extract, b, combine), then publish_l4.py")


# ── Entry point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a", "extract", "b", "combine"],
                        required=True, help="Which phase to run")
    args = parser.parse_args()

    if args.phase == "a":
        phase_a()
    elif args.phase == "extract":
        phase_extract()
    elif args.phase == "b":
        phase_b()
    elif args.phase == "combine":
        phase_combine()
