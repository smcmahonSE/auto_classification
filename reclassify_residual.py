"""
Re-classify the 2.65M low-confidence residual listings using improved anchor descriptions.

Uses the same two-phase memory approach as services/LCG classification:
  Phase A — residuals whose embeddings are in volume 2 (13GB, loads fine)
  Phase B — residuals whose embeddings are in volume 1 (32GB, extract vectors first)

Run order:
    Step 1 (this script, phase A):  python reclassify_residual.py --phase a
    Step 2 (extract v1 vectors):    python reclassify_residual.py --phase extract
    Step 3 (this script, phase B):  python reclassify_residual.py --phase b
    Step 4 (combine + republish):   python reclassify_residual.py --phase combine

Or run all phases sequentially (loads 32GB in phase B — may be slow):
    python reclassify_residual.py --phase all
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

ANCHORS_PATH     = PROJECT_ROOT / "analysis/data/l4_taxonomy_anchors.json"
ANCHOR_CACHE     = PROJECT_ROOT / "artifacts/cache/anchor_cache.pkl"
CACHE_V1_PATH    = PROJECT_ROOT / "artifacts/cache/embedding_cache.pkl"
CACHE_V2_PATH    = PROJECT_ROOT / "artifacts/cache/embedding_cache_new.pkl"
CACHE_KEYS_PATH  = PROJECT_ROOT / "artifacts/cache/embedding_cache_keys.pkl"
MASTER_CSV       = PROJECT_ROOT / "artifacts/analysis/master_classification_results.csv"
RESIDUAL_CSV     = PROJECT_ROOT / "artifacts/analysis/master_residual_for_clustering.csv"

OUT_DIR          = PROJECT_ROOT / "artifacts/analysis/residual_reclassification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V2_RESULTS       = OUT_DIR / "phase_a_v2_results.csv"
V1_VECTORS       = OUT_DIR / "phase_b_v1_vectors.npy"
V1_WORK          = OUT_DIR / "phase_b_v1_work.parquet"
V1_RESULTS       = OUT_DIR / "phase_b_v1_results.csv"
COMBINED_RESULTS = OUT_DIR / "residual_reclassified.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_anchors_and_embed():
    print("Loading updated anchor descriptions...")
    with open(ANCHORS_PATH) as f:
        data = json.load(f)
    anchors       = data["anchors"]
    anchor_ids    = [a["id"]          for a in anchors]
    anchor_labels = [a["label"]       for a in anchors]
    anchor_texts  = [a["description"] for a in anchors]

    bedrock = get_bedrock_client(profile_name=AWS_PROFILE, region=AWS_REGION)

    # Always re-embed anchors fresh (descriptions changed — don't use old cache)
    fresh_cache = {}
    anchor_hashes = [stable_text_hash(t) for t in anchor_texts]
    anchor_matrix = embed_texts_from_cache(
        texts=anchor_texts, text_hashes=anchor_hashes, cache=fresh_cache,
        client=bedrock, model_id=MODEL_ID, show_progress=False, max_workers=1,
    )
    # Save updated anchor cache
    with open(ANCHOR_CACHE, "wb") as f:
        pickle.dump(fresh_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    norms  = np.linalg.norm(anchor_matrix, axis=1, keepdims=True)
    normed = anchor_matrix / np.clip(norms, 1e-10, None)
    print(f"Anchors re-embedded: {normed.shape}")
    return anchor_ids, anchor_labels, normed


def classify_batch(vecs, anchor_normed, anchor_ids, anchor_labels):
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.clip(norms, 1e-10, None)
    sim    = vecs_n @ anchor_normed.T
    top1   = sim.argmax(axis=1)
    top1_s = sim[np.arange(len(sim)), top1]
    sim2   = sim.copy(); sim2[np.arange(len(sim)), top1] = -1
    margin = top1_s - sim2.max(axis=1)
    return (
        [anchor_ids[i]    for i in top1],
        [anchor_labels[i] for i in top1],
        top1_s.round(4),
        margin.round(4),
        margin < MARGIN_THRESHOLD,
    )


# ── Load residual + product text from Snowflake ────────────────────────────────

SOURCE_TABLES = {
    "LEI":      "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LEI",
    "SERVICES": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES",
    "LCG":      "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LCG",
}


def load_residual_with_text():
    """Load residual CSV and enrich it with product text from Snowflake."""
    print(f"Loading residual CSV ({RESIDUAL_CSV})...")
    residual = pd.read_csv(RESIDUAL_CSV, dtype={"PRODUCT_ID": str}, low_memory=False)
    print(f"Residual: {len(residual):,} rows")

    print("Reloading product text from Snowflake by SOURCE...")
    sf = get_snowflake_session()

    text_frames = []
    for source, table in SOURCE_TABLES.items():
        src_ids = residual.loc[residual["SOURCE"] == source, "PRODUCT_ID"].tolist()
        if not src_ids:
            continue
        print(f"  {source}: {len(src_ids):,} rows from {table}...")
        src_df = sf.sql(f"""
            SELECT PRODUCT_ID, PRODUCT_NAME, DESCRIPTION, PRICING_STATUS_C, LIST_PRICE_C
            FROM {table}
        """).to_pandas()
        src_df["PRODUCT_ID"] = src_df["PRODUCT_ID"].astype(str)
        src_df["SOURCE"] = source
        text_frames.append(src_df)

    text_df = pd.concat(text_frames, ignore_index=True)
    residual = residual.merge(text_df, on=["PRODUCT_ID", "SOURCE"], how="left")
    missing_text = residual[["PRODUCT_NAME"]].isna().all(axis=1).sum()
    if missing_text:
        print(f"  WARNING: {missing_text:,} residual rows have no product text after join — will be skipped")
    print(f"Product text joined. Working set: {len(residual):,} rows")
    return residual


# ── Phase A — classify v2 residuals ───────────────────────────────────────────

def phase_a():
    print("\n=== PHASE A: Classify residuals in volume 2 ===")
    anchor_ids, anchor_labels, anchor_normed = load_anchors_and_embed()

    residual = load_residual_with_text()

    # Drop rows with no product text (can't compute hash)
    has_text = residual["PRODUCT_NAME"].notna() | residual["DESCRIPTION"].notna()
    residual = residual[has_text].copy().reset_index(drop=True)
    print(f"Rows with usable text: {len(residual):,}")

    texts  = build_product_text(residual).tolist()
    hashes = [stable_text_hash(t) for t in texts]

    print(f"Loading volume 2 ({CACHE_V2_PATH.stat().st_size/1e9:.1f} GB)...")
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

    # Classify v2 in batches
    BATCH = 250_000
    records = []
    for start in range(0, len(v2_indices), BATCH):
        idx_batch    = v2_indices[start:start + BATCH]
        batch_hashes = [hashes[i] for i in idx_batch]
        vecs = np.array([cache_v2[h] for h in batch_hashes], dtype=np.float32)
        ids, labels, scores, margins, low_conf = classify_batch(
            vecs, anchor_normed, anchor_ids, anchor_labels
        )
        batch_df = residual.iloc[idx_batch].copy()
        batch_df["ASSIGNED_NEW_L3_ID"]    = ids
        batch_df["ASSIGNED_NEW_L3_LABEL"] = labels
        batch_df["CONFIDENCE"]            = scores
        batch_df["CONFIDENCE_MARGIN"]     = margins
        batch_df["IS_LOW_CONFIDENCE"]     = low_conf
        records.append(batch_df)
        high = (~low_conf).sum()
        pct  = (start + len(idx_batch)) / len(v2_indices) * 100
        print(f"  v2 batch {start:,}–{start+len(idx_batch):,} ({pct:.0f}%) — high-conf: {high:,}/{len(idx_batch):,}")
        del vecs

    del cache_v2
    v2_results = pd.concat(records, ignore_index=True)
    v2_results.to_csv(V2_RESULTS, index=False)
    print(f"Phase A saved: {V2_RESULTS} ({len(v2_results):,} rows)")

    # Save v1 work file
    v1_work = residual.iloc[v1_indices].copy()
    v1_work["_HASH"] = [hashes[i] for i in v1_indices]
    v1_work.to_parquet(V1_WORK, index=False)
    print(f"Phase B work file saved: {V1_WORK} ({len(v1_work):,} rows)")

    high_a = (~v2_results["IS_LOW_CONFIDENCE"]).sum()
    print(f"\nPhase A result: {high_a:,}/{len(v2_results):,} newly high-confidence ({high_a/len(v2_results)*100:.1f}%)")


# ── Phase B extract — pull v1 vectors ─────────────────────────────────────────

def phase_extract():
    print("\n=== PHASE EXTRACT: Extract v1 residual vectors ===")
    if not V1_WORK.exists():
        print("ERROR: Run phase A first.")
        sys.exit(1)

    v1_work = pd.read_parquet(V1_WORK)
    hashes  = v1_work["_HASH"].tolist()
    print(f"Need vectors for {len(hashes):,} residual listings from volume 1")

    print(f"Loading volume 1 ({CACHE_V1_PATH.stat().st_size/1e9:.1f} GB)...")
    with open(CACHE_V1_PATH, "rb") as f:
        cache_v1 = pickle.load(f)
    print(f"Volume 1: {len(cache_v1):,} entries")

    cache_v1_keys = set(cache_v1.keys())
    not_found = [h for h in hashes if h not in cache_v1_keys]
    if not_found:
        print(f"WARNING: {len(not_found):,} hashes not found in volume 1 — skipping")
        valid_mask = np.array([h in cache_v1_keys for h in hashes])
        hashes  = [h for h, m in zip(hashes, valid_mask) if m]
        v1_work = v1_work[valid_mask].reset_index(drop=True)
        v1_work.to_parquet(V1_WORK, index=False)

    n    = len(hashes)
    dim  = 1536
    print(f"Extracting {n:,} vectors into memory-mapped file ({n * dim * 4 / 1e9:.2f} GB)...")

    # Write into a memory-mapped .npy so we never hold the full array in RAM
    # alongside the cache.
    mmap = np.lib.format.open_memmap(
        str(V1_VECTORS), mode="w+", dtype=np.float32, shape=(n, dim)
    )
    CHUNK = 100_000
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        for i, h in enumerate(hashes[start:end]):
            mmap[start + i] = cache_v1[h]
        mmap.flush()
        print(f"  wrote {end:,}/{n:,} ({end/n*100:.0f}%)")

    del cache_v1
    del mmap
    print(f"Saved: {V1_VECTORS}  ({V1_VECTORS.stat().st_size/1e9:.2f} GB)")


# ── Phase B classify ───────────────────────────────────────────────────────────

def phase_b():
    print("\n=== PHASE B: Classify residuals from volume 1 vectors ===")
    for p in [V1_WORK, V1_VECTORS]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run phase A and extract first.")
            sys.exit(1)

    anchor_ids, anchor_labels, anchor_normed = load_anchors_and_embed()

    print(f"Loading pre-extracted vectors ({V1_VECTORS}) as memory-map...")
    vectors = np.load(V1_VECTORS, mmap_mode="r")  # reads slices from disk, no full RAM load
    v1_work = pd.read_parquet(V1_WORK)
    print(f"Vectors: {vectors.shape}")

    BATCH = 250_000
    records = []
    for start in range(0, len(v1_work), BATCH):
        end  = min(start + BATCH, len(v1_work))
        vecs = vectors[start:end].copy()
        ids, labels, scores, margins, low_conf = classify_batch(
            vecs, anchor_normed, anchor_ids, anchor_labels
        )
        batch_df = v1_work.iloc[start:end].drop(columns=["_HASH"], errors="ignore").copy()
        batch_df["ASSIGNED_NEW_L3_ID"]    = ids
        batch_df["ASSIGNED_NEW_L3_LABEL"] = labels
        batch_df["CONFIDENCE"]            = scores
        batch_df["CONFIDENCE_MARGIN"]     = margins
        batch_df["IS_LOW_CONFIDENCE"]     = low_conf
        records.append(batch_df)
        high = (~low_conf).sum()
        pct  = end / len(v1_work) * 100
        print(f"  v1 batch {start:,}–{end:,} ({pct:.0f}%) — high-conf: {high:,}/{end-start:,}")

    v1_results = pd.concat(records, ignore_index=True)
    v1_results.to_csv(V1_RESULTS, index=False)
    print(f"Phase B saved: {V1_RESULTS} ({len(v1_results):,} rows)")
    high_b = (~v1_results["IS_LOW_CONFIDENCE"]).sum()
    print(f"Phase B result: {high_b:,}/{len(v1_results):,} newly high-confidence ({high_b/len(v1_results)*100:.1f}%)")


# ── Combine phases ─────────────────────────────────────────────────────────────

def phase_combine():
    print("\n=== COMBINE: Merge reclassified residuals into master ===")
    parts = []
    for p in [V2_RESULTS, V1_RESULTS]:
        if p.exists():
            df = pd.read_csv(p)
            parts.append(df)
            print(f"  {p.name}: {len(df):,} rows")
        else:
            print(f"  WARNING: {p.name} not found — skipping")

    if not parts:
        print("ERROR: No phase results found.")
        sys.exit(1)

    reclassified = pd.concat(parts, ignore_index=True)
    reclassified.to_csv(COMBINED_RESULTS, index=False)
    print(f"Combined reclassified: {COMBINED_RESULTS} ({len(reclassified):,} rows)")

    high = (~reclassified["IS_LOW_CONFIDENCE"]).sum()
    low  =   reclassified["IS_LOW_CONFIDENCE"].sum()
    print(f"\nAfter reclassification:")
    print(f"  Newly high-confidence: {high:,} ({high/len(reclassified)*100:.1f}%)")
    print(f"  Still residual:        {low:,} ({low/len(reclassified)*100:.1f}%)")

    print("\nMerging into master CSV...")
    master = pd.read_csv(MASTER_CSV, dtype={"PRODUCT_ID": str}, low_memory=False)
    reclassified["PRODUCT_ID"] = reclassified["PRODUCT_ID"].astype(str)
    print(f"Master before: {len(master):,} rows, residual: {master['IS_LOW_CONFIDENCE'].sum():,}")

    # Build a clean output from reclassified: overwrite the assignment/confidence
    # columns with the new values, keep everything else from the original residual row.
    reclass_out = reclassified.copy()

    # Drop the old assignment columns (will be replaced by NEW_L3 equivalents)
    reclass_out = reclass_out.drop(
        columns=["ASSIGNED_L4_ID", "ASSIGNED_L4_LABEL"], errors="ignore"
    )
    # Rename NEW_L3 → L4 (internal master naming)
    reclass_out = reclass_out.rename(columns={
        "ASSIGNED_NEW_L3_ID":    "ASSIGNED_L4_ID",
        "ASSIGNED_NEW_L3_LABEL": "ASSIGNED_L4_LABEL",
    })

    # Drop any extra columns not in master, and align column order
    common_cols = [c for c in master.columns if c in reclass_out.columns]
    reclass_out = reclass_out[common_cols]

    # Drop old residual rows from master, append updated versions
    reclass_ids = set(reclass_out["PRODUCT_ID"].tolist())
    master_keep = master[~master["PRODUCT_ID"].isin(reclass_ids)].copy()
    master_keep = master_keep[common_cols]  # same column set

    master_updated = pd.concat([master_keep, reclass_out], ignore_index=True)
    print(f"Master after:  {len(master_updated):,} rows, residual: {master_updated['IS_LOW_CONFIDENCE'].sum():,}")

    master_updated.to_csv(MASTER_CSV, index=False)
    print(f"Master CSV updated: {MASTER_CSV}")
    print("\nRun combine_and_publish.py to push updates to Snowflake.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a", "extract", "b", "combine", "all"],
                        default="a", help="Which phase to run")
    args = parser.parse_args()

    if args.phase in ("a", "all"):
        phase_a()
    if args.phase in ("extract", "all"):
        phase_extract()
    if args.phase in ("b", "all"):
        phase_b()
    if args.phase in ("combine", "all"):
        phase_combine()
