"""
Classify listings into L4 subcategories using cosine similarity against subcategory anchors.

Each listing is classified ONLY within its assigned L3 bucket — subcategories from other L3s
are never considered. L3s without defined L4 subcategories receive ASSIGNED_L4_LABEL = None.

Uses the same two-phase memory approach as reclassify_residual.py:
  Phase A — classify listings whose embeddings are in volume 2 (13GB, loads fine)
  Phase B — classify listings whose embeddings are in volume 1 (32GB, extract first)

Run order:
    python classify_l4.py --phase a        # classify v2 listings
    python classify_l4.py --phase extract  # extract v1 vectors to .npy
    python classify_l4.py --phase b        # classify from .npy
    python classify_l4.py --phase combine  # merge and save final CSV
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

L3_ANCHORS_PATH  = PROJECT_ROOT / "analysis/data/l4_taxonomy_anchors.json"
L4_ANCHORS_PATH  = PROJECT_ROOT / "analysis/data/l4_subcategory_anchors.json"
CACHE_V1_PATH    = PROJECT_ROOT / "artifacts/cache/embedding_cache.pkl"
CACHE_V2_PATH    = PROJECT_ROOT / "artifacts/cache/embedding_cache_new.pkl"
CACHE_KEYS_PATH  = PROJECT_ROOT / "artifacts/cache/embedding_cache_keys.pkl"
MASTER_CSV       = PROJECT_ROOT / "artifacts/analysis/master_classification_results.csv"

OUT_DIR          = PROJECT_ROOT / "artifacts/analysis/l4_classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V2_RESULTS  = OUT_DIR / "phase_a_v2_results.csv"
V1_VECTORS  = OUT_DIR / "phase_b_v1_vectors.npy"
V1_WORK     = OUT_DIR / "phase_b_v1_work.parquet"
V1_RESULTS  = OUT_DIR / "phase_b_v1_results.csv"
L4_RESULTS  = OUT_DIR / "l4_results.csv"

SOURCE_TABLES = {
    "LEI":      "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LEI",
    "SERVICES": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES",
    "LCG":      "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LCG",
}

# ── Load anchor data ────────────────────────────────────────────────────────────

def load_l4_anchors():
    """Return dict: l3_label → list of (id, label, description) tuples."""
    with open(L4_ANCHORS_PATH) as f:
        data = json.load(f)

    # Map L3 IDs to labels from the L3 anchor file for joining
    with open(L3_ANCHORS_PATH) as f:
        l3_data = json.load(f)
    l3_id_to_label = {a["id"]: a["label"] for a in l3_data["anchors"]}

    l4_by_l3 = {}
    for l3_id, subcats in data["l3_subcategories"].items():
        l3_label = l3_id_to_label.get(l3_id, l3_id)
        l4_by_l3[l3_label] = [
            (s["id"], s["label"], s["description"]) for s in subcats
        ]
    return l4_by_l3


def embed_l4_anchors(l4_by_l3):
    """Embed all L4 anchor descriptions; return dict l3_label → (ids, labels, normed_matrix)."""
    bedrock = get_bedrock_client(profile_name=AWS_PROFILE, region=AWS_REGION)
    result  = {}
    fresh   = {}

    all_texts   = []
    all_indices = []  # (l3_label, position_in_l3)
    for l3_label, subcats in l4_by_l3.items():
        for i, (sid, slabel, sdesc) in enumerate(subcats):
            all_texts.append(sdesc)
            all_indices.append((l3_label, i))

    hashes = [stable_text_hash(t) for t in all_texts]
    matrix = embed_texts_from_cache(
        texts=all_texts, text_hashes=hashes, cache=fresh,
        client=bedrock, model_id=MODEL_ID, show_progress=True, max_workers=1,
    )

    # Split back into per-L3 matrices
    per_l3_rows = {}
    for idx, (l3_label, pos) in enumerate(all_indices):
        per_l3_rows.setdefault(l3_label, []).append((pos, matrix[idx]))

    for l3_label, subcats in l4_by_l3.items():
        rows = sorted(per_l3_rows.get(l3_label, []), key=lambda x: x[0])
        mat  = np.array([r for _, r in rows], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        normed = mat / np.clip(norms, 1e-10, None)
        ids    = [s[0] for s in subcats]
        labels = [s[1] for s in subcats]
        result[l3_label] = (ids, labels, normed)
        print(f"  {l3_label}: {len(ids)} subcategories embedded")

    return result


# ── Load master + product text ──────────────────────────────────────────────────

def load_master_with_text():
    """Load master classification CSV and enrich with product text from Snowflake."""
    print(f"Loading master CSV ({MASTER_CSV})...")
    master = pd.read_csv(MASTER_CSV, dtype={"PRODUCT_ID": str}, low_memory=False)
    print(f"Master: {len(master):,} rows, columns: {list(master.columns)}")

    label_col = "ASSIGNED_NEW_L3_LABEL" if "ASSIGNED_NEW_L3_LABEL" in master.columns else "ASSIGNED_L4_LABEL"
    print(f"L3 label column: {label_col}")
    print("L3 distribution:")
    print(master[label_col].value_counts().to_string())

    text_cols = ["PRODUCT_NAME", "DESCRIPTION", "PRICING_STATUS_C", "LIST_PRICE_C"]
    if all(c in master.columns for c in text_cols):
        print("\nText columns already present in master CSV — skipping Snowflake reload.")
    else:
        print("\nReloading product text from Snowflake by SOURCE...")
        sf = get_snowflake_session()
        text_frames = []
        for source, table in SOURCE_TABLES.items():
            if source not in master["SOURCE"].values:
                continue
            print(f"  {source}: loading from {table}...")
            src = sf.sql(f"""
                SELECT PRODUCT_ID, PRODUCT_NAME, DESCRIPTION, PRICING_STATUS_C, LIST_PRICE_C
                FROM {table}
            """).to_pandas()
            src["PRODUCT_ID"] = src["PRODUCT_ID"].astype(str)
            src["SOURCE"] = source
            text_frames.append(src)

        # Drop any text cols already in master to avoid _x/_y suffix conflicts
        master = master.drop(columns=[c for c in text_cols if c in master.columns], errors="ignore")
        text_df = pd.concat(text_frames, ignore_index=True)
        master  = master.merge(text_df, on=["PRODUCT_ID", "SOURCE"], how="left")

    print(f"Product text ready. Shape: {master.shape}")
    return master, label_col


# ── Per-L3 cosine similarity classification ─────────────────────────────────────

def classify_l3_bucket(vecs, l3_label, l4_embeddings):
    """
    Classify a batch of vectors within one L3 bucket.
    Returns (ids, labels, scores, margins, is_low_conf) or None if no L4s defined.
    """
    if l3_label not in l4_embeddings:
        n = len(vecs)
        return (
            [None] * n,
            [None] * n,
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.ones(n, dtype=bool),
        )

    anchor_ids, anchor_labels, anchor_normed = l4_embeddings[l3_label]
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


def classify_df(df, label_col, l4_embeddings, cache_dict, desc):
    """Classify all rows in df using cache_dict for embeddings."""
    results = []
    for l3_label, grp in df.groupby(label_col):
        texts  = build_product_text(grp).tolist()
        hashes = [stable_text_hash(t) for t in texts]

        # Get vectors for rows found in cache
        found_mask = np.array([h in cache_dict for h in hashes])
        if found_mask.sum() == 0:
            continue

        found_idx  = np.where(found_mask)[0]
        found_hash = [hashes[i] for i in found_idx]
        vecs       = np.array([cache_dict[h] for h in found_hash], dtype=np.float32)

        ids, labels, scores, margins, low_conf = classify_l3_bucket(
            vecs, l3_label, l4_embeddings
        )
        batch_df = grp.iloc[found_idx].copy()
        batch_df["ASSIGNED_L4_ID"]           = ids
        batch_df["ASSIGNED_L4_LABEL"]         = labels
        batch_df["L4_CONFIDENCE"]             = scores
        batch_df["L4_CONFIDENCE_MARGIN"]      = margins
        batch_df["L4_IS_LOW_CONFIDENCE"]      = low_conf
        results.append(batch_df)

        high = (~low_conf).sum()
        pct  = high / len(found_idx) * 100
        print(f"  [{desc}] {l3_label}: {len(found_idx):,} rows, {high:,} high-conf ({pct:.0f}%)")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ── Phase A ─────────────────────────────────────────────────────────────────────

def phase_a():
    print("\n=== PHASE A: Classify v2 listings ===")
    l4_by_l3     = load_l4_anchors()
    print(f"\nEmbedding {sum(len(v) for v in l4_by_l3.values())} L4 anchors across {len(l4_by_l3)} L3s...")
    l4_embeddings = embed_l4_anchors(l4_by_l3)

    master, label_col = load_master_with_text()

    # Drop rows with no text
    has_text = master["PRODUCT_NAME"].notna() | master["DESCRIPTION"].notna()
    master   = master[has_text].copy().reset_index(drop=True)

    print(f"\nLoading volume 2 ({CACHE_V2_PATH.stat().st_size/1e9:.1f} GB)...")
    with open(CACHE_V2_PATH, "rb") as f:
        cache_v2 = pickle.load(f)
    print(f"Volume 2: {len(cache_v2):,} entries")

    print("Loading volume 1 key index...")
    with open(CACHE_KEYS_PATH, "rb") as f:
        cache_v1_keys = pickle.load(f)

    # Determine which rows are in v2, which need v1
    texts  = build_product_text(master).tolist()
    hashes = [stable_text_hash(t) for t in texts]
    in_v2      = [h in cache_v2      for h in hashes]
    in_v1_only = [h in cache_v1_keys and not in_v2[i] for i, h in enumerate(hashes)]

    v2_indices = [i for i, m in enumerate(in_v2)      if m]
    v1_indices = [i for i, m in enumerate(in_v1_only) if m]
    print(f"In volume 2:      {len(v2_indices):,}")
    print(f"In volume 1 only: {len(v1_indices):,}")

    master_v2 = master.iloc[v2_indices].copy().reset_index(drop=True)
    v2_results = classify_df(master_v2, label_col, l4_embeddings, cache_v2, "v2")
    del cache_v2

    v2_results.to_csv(V2_RESULTS, index=False)
    print(f"\nPhase A saved: {V2_RESULTS} ({len(v2_results):,} rows)")

    # Save v1 work file with hashes for extraction
    v1_work = master.iloc[v1_indices].copy()
    v1_work["_HASH"] = [hashes[i] for i in v1_indices]
    v1_work.to_parquet(V1_WORK, index=False)
    print(f"Phase B work file saved: {V1_WORK} ({len(v1_work):,} rows)")

    high = (~v2_results["L4_IS_LOW_CONFIDENCE"]).sum()
    print(f"\nPhase A: {high:,}/{len(v2_results):,} high-confidence L4 assignments ({high/len(v2_results)*100:.1f}%)")


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
    print(f"Extracting {n:,} vectors to memory-mapped file ({n * dim * 4 / 1e9:.2f} GB)...")
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

    l4_by_l3      = load_l4_anchors()
    print(f"\nEmbedding {sum(len(v) for v in l4_by_l3.values())} L4 anchors...")
    l4_embeddings = embed_l4_anchors(l4_by_l3)

    v1_work  = pd.read_parquet(V1_WORK)
    vectors  = np.load(V1_VECTORS, mmap_mode="r")
    label_col = "ASSIGNED_NEW_L3_LABEL" if "ASSIGNED_NEW_L3_LABEL" in v1_work.columns else "ASSIGNED_L4_LABEL"
    print(f"Vectors: {vectors.shape}")

    # Build a hash→vector lookup from the memmap (by position)
    hashes = v1_work["_HASH"].tolist()
    hash_to_idx = {h: i for i, h in enumerate(hashes)}

    # Classify per L3 bucket using the memmap
    results = []
    for l3_label, grp in v1_work.groupby(label_col):
        grp_hashes = grp["_HASH"].tolist()
        grp_idx    = [hash_to_idx[h] for h in grp_hashes]
        vecs       = np.array(vectors[grp_idx], dtype=np.float32)

        ids, labels, scores, margins, low_conf = classify_l3_bucket(
            vecs, l3_label, l4_embeddings
        )
        batch_df = grp.drop(columns=["_HASH"], errors="ignore").copy()
        batch_df["ASSIGNED_L4_ID"]        = ids
        batch_df["ASSIGNED_L4_LABEL"]     = labels
        batch_df["L4_CONFIDENCE"]         = scores
        batch_df["L4_CONFIDENCE_MARGIN"]  = margins
        batch_df["L4_IS_LOW_CONFIDENCE"]  = low_conf
        results.append(batch_df)

        high = (~low_conf).sum()
        print(f"  [v1] {l3_label}: {len(grp):,} rows, {high:,} high-conf ({high/len(grp)*100:.0f}%)")

    v1_results = pd.concat(results, ignore_index=True)
    v1_results.to_csv(V1_RESULTS, index=False)
    print(f"\nPhase B saved: {V1_RESULTS} ({len(v1_results):,} rows)")
    high = (~v1_results["L4_IS_LOW_CONFIDENCE"]).sum()
    print(f"Phase B: {high:,}/{len(v1_results):,} high-confidence ({high/len(v1_results)*100:.1f}%)")


# ── Combine ──────────────────────────────────────────────────────────────────────

def phase_combine():
    print("\n=== COMBINE: Merge L4 results ===")
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

    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(L4_RESULTS, index=False)
    print(f"\nL4 results saved: {L4_RESULTS} ({len(combined):,} rows)")

    # Summary
    total   = len(combined)
    high    = (~combined["L4_IS_LOW_CONFIDENCE"]).sum()
    no_l4   = combined["ASSIGNED_L4_LABEL"].isna().sum()
    print(f"High-confidence L4:   {high:,} ({high/total*100:.1f}%)")
    print(f"Low-confidence L4:    {total-high-no_l4:,} ({(total-high-no_l4)/total*100:.1f}%)")
    print(f"No L4 defined (deferred L3s): {no_l4:,} ({no_l4/total*100:.1f}%)")

    print("\nL4 distribution:")
    dist = combined["ASSIGNED_L4_LABEL"].value_counts()
    print(dist.to_string())
    print("\nRun publish_l4.py to push results to Snowflake.")


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
