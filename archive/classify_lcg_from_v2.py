"""
Classify LCG listings using embeddings in volume 2 (embedding_cache_new.pkl).

The overnight embedding run added ~440K LCG embeddings to volume 2.
v1_work.parquet was accidentally overwritten, so we reload LCG from Snowflake,
recompute hashes, and classify directly from volume 2.

Run this instead of classify_lcg.py:
    python classify_lcg_from_v2.py

Then run Steps 7-10 in the notebook with MODE = "classify_lcg".
"""

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

MODE              = "classify_lcg"
AWS_PROFILE       = "staging.admin"
AWS_REGION        = "us-east-1"
MODEL_ID          = "amazon.titan-embed-text-v1"
MARGIN_THRESHOLD  = 0.05
BATCH_SIZE        = 100_000

LCG_TABLE         = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LCG"
CACHE_V1_PATH     = PROJECT_ROOT / "artifacts/cache/embedding_cache.pkl"
CACHE_V2_PATH     = PROJECT_ROOT / "artifacts/cache/embedding_cache_new.pkl"
CACHE_KEYS_PATH   = PROJECT_ROOT / "artifacts/cache/embedding_cache_keys.pkl"
ANCHOR_CACHE_PATH = PROJECT_ROOT / "artifacts/cache/anchor_cache.pkl"
ANCHORS_PATH      = PROJECT_ROOT / "analysis/data/l3_taxonomy_anchors.json"
OUTPUT_DIR        = PROJECT_ROOT / f"artifacts/analysis/l4_classification_{MODE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH      = OUTPUT_DIR / "classification_results_v1.csv"

# ── Load anchors ───────────────────────────────────────────────────────────────
print("Loading anchors...")
with open(ANCHORS_PATH) as f:
    anchors_data = json.load(f)
anchors       = anchors_data["anchors"]
anchor_ids    = [a["id"]          for a in anchors]
anchor_labels = [a["label"]       for a in anchors]
anchor_texts  = [a["description"] for a in anchors]

bedrock = get_bedrock_client(profile_name=AWS_PROFILE, region=AWS_REGION)
with open(ANCHOR_CACHE_PATH, "rb") as f:
    anchor_cache = pickle.load(f)

anchor_hashes = [stable_text_hash(t) for t in anchor_texts]
anchor_matrix = embed_texts_from_cache(
    texts=anchor_texts, text_hashes=anchor_hashes, cache=anchor_cache,
    client=bedrock, model_id=MODEL_ID, show_progress=False, max_workers=1,
)
anchor_norms  = np.linalg.norm(anchor_matrix, axis=1, keepdims=True)
anchor_normed = anchor_matrix / np.clip(anchor_norms, 1e-10, None)
print(f"Anchors ready: {anchor_normed.shape}")

# ── Load LCG from Snowflake ────────────────────────────────────────────────────
print(f"\nLoading LCG listings from Snowflake ({LCG_TABLE})...")
sf = get_snowflake_session()
df = sf.sql(f"""
    SELECT PRODUCT_ID, PRODUCT_NAME, DESCRIPTION,
           PRICING_STATUS_C, LIST_PRICE_C,
           PARENT_3_CATEGORY AS CURRENT_L3,
           PARENT_4_CATEGORY AS CURRENT_L4,
           PARENT_5_CATEGORY AS CURRENT_L5
    FROM {LCG_TABLE}
""").to_pandas()
print(f"Loaded {len(df):,} LCG listings")

texts  = build_product_text(df).tolist()
hashes = [stable_text_hash(t) for t in texts]

# ── Load volume 2 (services + LCG embeddings) ─────────────────────────────────
print(f"\nLoading volume 2 ({CACHE_V2_PATH.stat().st_size/1e9:.1f} GB)...")
with open(CACHE_V2_PATH, "rb") as f:
    cache_v2 = pickle.load(f)
print(f"Volume 2 entries: {len(cache_v2):,}")

# ── Load volume 1 key index ────────────────────────────────────────────────────
print(f"Loading volume 1 key index...")
with open(CACHE_KEYS_PATH, "rb") as f:
    cache_v1_keys = pickle.load(f)
print(f"Volume 1 keys: {len(cache_v1_keys):,}")

# ── Determine coverage ─────────────────────────────────────────────────────────
in_v2   = [h in cache_v2      for h in hashes]
in_v1   = [h in cache_v1_keys and not in_v2[i] for i, h in enumerate(hashes)]
missing = [not in_v2[i] and not in_v1[i]       for i in range(len(hashes))]

print(f"\nCoverage:")
print(f"  In volume 2:  {sum(in_v2):,}")
print(f"  In volume 1:  {sum(in_v1):,}")
print(f"  Missing:      {sum(missing):,}")

# ── Embed any remaining missing ────────────────────────────────────────────────
if any(missing):
    print(f"\nEmbedding {sum(missing):,} missing listings...")
    missing_hashes = [h for h, m in zip(hashes, missing) if m]
    missing_texts  = [t for t, m in zip(texts,  missing) if m]

    embed_texts_from_cache(
        texts=missing_texts, text_hashes=missing_hashes,
        cache=cache_v2, client=bedrock, model_id=MODEL_ID,
        show_progress=True, max_workers=8,
    )
    # Save new embeddings back to volume 2
    tmp = CACHE_V2_PATH.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache_v2, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE_V2_PATH)
    print("Volume 2 updated with newly embedded listings.")

# ── Batched classification ─────────────────────────────────────────────────────
output_cols = ["PRODUCT_ID", "ASSIGNED_L4_ID", "ASSIGNED_L4_LABEL",
               "CONFIDENCE", "CONFIDENCE_MARGIN", "IS_LOW_CONFIDENCE"]
for col in ["CURRENT_L3", "CURRENT_L4", "CURRENT_L5"]:
    if col in df.columns:
        output_cols.insert(2, col)

# Only classify listings whose hash is in volume 2
# (the ~530 in volume 1 only are already in classification_results_v1.csv)
v2_indices = [i for i, h in enumerate(hashes) if h in cache_v2]
print(f"\nListings with embeddings in volume 2: {len(v2_indices):,}")
print(f"Listings in volume 1 only (already classified): {len(hashes)-len(v2_indices):,}")
print(f"Classifying {len(v2_indices):,} listings in batches of {BATCH_SIZE:,}...")
results = []

for batch_start in range(0, len(v2_indices), BATCH_SIZE):
    idx_batch    = v2_indices[batch_start:batch_start + BATCH_SIZE]
    start, end   = idx_batch[0], idx_batch[-1] + 1
    batch_hashes = [hashes[i] for i in idx_batch]
    batch_df     = df.iloc[idx_batch]

    vecs = np.array([cache_v2[h] for h in batch_hashes], dtype=np.float32)
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.clip(norms, 1e-10, None)
    del vecs

    sim    = vecs_n @ anchor_normed.T
    del vecs_n
    top1   = sim.argmax(axis=1)
    top1_s = sim[np.arange(len(sim)), top1]
    sim2   = sim.copy(); sim2[np.arange(len(sim)), top1] = -1
    margin = top1_s - sim2.max(axis=1)
    del sim, sim2

    rec = batch_df.copy()
    rec["ASSIGNED_L4_ID"]    = [anchor_ids[i]    for i in top1]
    rec["ASSIGNED_L4_LABEL"] = [anchor_labels[i] for i in top1]
    rec["CONFIDENCE"]        = top1_s.round(4)
    rec["CONFIDENCE_MARGIN"] = margin.round(4)
    rec["IS_LOW_CONFIDENCE"] = margin < MARGIN_THRESHOLD
    results.append(rec[[c for c in output_cols if c in rec.columns]])

    high = (margin >= MARGIN_THRESHOLD).sum()
    pct  = (batch_start + len(idx_batch)) / len(v2_indices) * 100
    print(f"  {batch_start:,}–{batch_start+len(idx_batch):,} ({pct:.0f}%) — high-conf: {high:,}/{len(idx_batch):,}")

all_results = pd.concat(results, ignore_index=True)
all_results.to_csv(RESULTS_PATH, index=False)

high = (~all_results["IS_LOW_CONFIDENCE"]).sum()
low  =   all_results["IS_LOW_CONFIDENCE"].sum()
print(f"\nSaved: {RESULTS_PATH} ({len(all_results):,} rows)")
print(f"High-confidence: {high:,} ({high/len(all_results)*100:.1f}%)")
print(f"Residual:        {low:,} ({low/len(all_results)*100:.1f}%)")
print("\nDone. Return to the notebook and run Steps 7-10.")
