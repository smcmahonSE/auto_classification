"""
Batched cosine similarity classification for 7M service listings.
Runs as a standalone script (NOT inside Jupyter) to control memory precisely.

Loads volume 1 + volume 2 caches, processes listings in BATCH_SIZE chunks,
writes L4 assignments to CSV. Never builds a 43GB full matrix.

Usage:
    python classify_services_batched.py

Prerequisites:
    - embedding_cache.pkl (volume 1) — main cache
    - embedding_cache_new.pkl (volume 2) — new services embeddings
    - analysis/data/l4_taxonomy_anchors.json — anchor definitions
    - Snowflake connection (to reload listing metadata for PRODUCT_ID mapping)
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
    get_bedrock_client,
    get_snowflake_session,
    stable_text_hash,
)

# ── Config ─────────────────────────────────────────────────────────────────────
CACHE_V1_PATH   = PROJECT_ROOT / "artifacts/cache/embedding_cache.pkl"
CACHE_V2_PATH   = PROJECT_ROOT / "artifacts/cache/embedding_cache_new.pkl"
ANCHORS_PATH    = PROJECT_ROOT / "analysis/data/l4_taxonomy_anchors.json"
SERVICES_TABLE  = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES"
AWS_PROFILE     = "staging.admin"
AWS_REGION      = "us-east-1"
MODEL_ID        = "amazon.titan-embed-text-v1"
CONFIDENCE_THRESHOLD = 0.50
BATCH_SIZE      = 250_000
OUTPUT_DIR      = PROJECT_ROOT / "artifacts/analysis/l4_classification_classify_services"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load anchors and embed ─────────────────────────────────────────────────────
print("Loading anchors...")
with open(ANCHORS_PATH) as f:
    anchors_data = json.load(f)
anchors = anchors_data["anchors"]
anchor_ids     = [a["id"]    for a in anchors]
anchor_labels  = [a["label"] for a in anchors]
anchor_texts   = [a["description"] for a in anchors]

print("Embedding anchors (13 calls)...")
bedrock = get_bedrock_client(profile_name=AWS_PROFILE, region=AWS_REGION)

# Load minimal cache for anchor embeddings only
anchor_cache = {}
if CACHE_V2_PATH.exists():
    with open(CACHE_V2_PATH, "rb") as f:
        anchor_cache = pickle.load(f)

from product_classifier_utils import embed_texts_from_cache
anchor_hashes = [stable_text_hash(t) for t in anchor_texts]
anchor_matrix = embed_texts_from_cache(
    texts=anchor_texts,
    text_hashes=anchor_hashes,
    cache=anchor_cache,
    client=bedrock,
    model_id=MODEL_ID,
    show_progress=False,
    max_workers=1,
)
anchor_norms  = np.linalg.norm(anchor_matrix, axis=1, keepdims=True)
anchor_normed = anchor_matrix / np.clip(anchor_norms, 1e-10, None)
print(f"Anchor matrix: {anchor_normed.shape}")
del anchor_cache

# ── Load both caches ───────────────────────────────────────────────────────────
print(f"\nLoading volume 2 (new embeddings)...")
with open(CACHE_V2_PATH, "rb") as f:
    cache_v2 = pickle.load(f)
print(f"Volume 2: {len(cache_v2):,} entries")

print(f"Loading volume 1 (main cache, ~32GB — this takes a few minutes)...")
with open(CACHE_V1_PATH, "rb") as f:
    cache_v1 = pickle.load(f)
print(f"Volume 1: {len(cache_v1):,} entries")

def lookup(h):
    if h in cache_v2:
        return cache_v2[h]
    return cache_v1[h]

# ── Load listings from Snowflake ───────────────────────────────────────────────
print("\nLoading service listings from Snowflake...")
sf = get_snowflake_session()
query = f"""
SELECT
    PRODUCT_ID,
    PRODUCT_NAME,
    DESCRIPTION,
    PARENT_3_CATEGORY AS CURRENT_L3,
    PARENT_4_CATEGORY AS CURRENT_L4,
    PARENT_5_CATEGORY AS CURRENT_L5
FROM {SERVICES_TABLE}
"""
df = sf.sql(query).to_pandas()
print(f"Loaded {len(df):,} listings")

texts  = build_product_text(df).tolist()
hashes = [stable_text_hash(t) for t in texts]
del texts  # free memory

# ── Batched classification ─────────────────────────────────────────────────────
print(f"\nClassifying in batches of {BATCH_SIZE:,}...")
results = []

for batch_start in range(0, len(df), BATCH_SIZE):
    batch_end    = min(batch_start + BATCH_SIZE, len(df))
    batch_hashes = hashes[batch_start:batch_end]
    batch_df     = df.iloc[batch_start:batch_end]

    # Build embedding matrix for this batch only
    batch_vecs = np.array([lookup(h) for h in batch_hashes], dtype=np.float32)
    norms      = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
    batch_normed = batch_vecs / np.clip(norms, 1e-10, None)
    del batch_vecs

    # Cosine similarity against anchors
    sim = batch_normed @ anchor_normed.T  # (batch, 13)
    del batch_normed

    top1_idx    = sim.argmax(axis=1)
    top1_scores = sim[np.arange(len(sim)), top1_idx]

    # Confidence margin (gap to second best)
    sim_copy = sim.copy()
    sim_copy[np.arange(len(sim)), top1_idx] = -1
    top2_scores = sim_copy.max(axis=1)
    margin = top1_scores - top2_scores
    del sim, sim_copy

    batch_results = pd.DataFrame({
        "PRODUCT_ID":         batch_df["PRODUCT_ID"].values,
        "CURRENT_L3":         batch_df["CURRENT_L3"].values if "CURRENT_L3" in batch_df else "",
        "CURRENT_L4":         batch_df["CURRENT_L4"].values if "CURRENT_L4" in batch_df else "",
        "ASSIGNED_L4_ID":     [anchor_ids[i]    for i in top1_idx],
        "ASSIGNED_L4_LABEL":  [anchor_labels[i] for i in top1_idx],
        "CONFIDENCE":         top1_scores.round(4),
        "CONFIDENCE_MARGIN":  margin.round(4),
        "IS_LOW_CONFIDENCE":  top1_scores < CONFIDENCE_THRESHOLD,
    })
    results.append(batch_results)

    pct = batch_end / len(df) * 100
    high = (top1_scores >= CONFIDENCE_THRESHOLD).sum()
    print(f"  Batch {batch_start:,}–{batch_end:,} ({pct:.0f}%) — high-conf: {high:,}/{len(batch_results):,}")
    del batch_results

print("\nCombining results...")
all_results = pd.concat(results, ignore_index=True)
del results

# ── Save ───────────────────────────────────────────────────────────────────────
results_path  = OUTPUT_DIR / "classification_results.csv"
residual_path = OUTPUT_DIR / "residual_for_clustering.csv"

all_results.to_csv(results_path, index=False)
print(f"Results saved: {results_path} ({len(all_results):,} rows)")

residual = all_results[all_results["IS_LOW_CONFIDENCE"]]
residual.to_csv(residual_path, index=False)
print(f"Residual saved: {residual_path} ({len(residual):,} rows)")

# Summary
high = (~all_results["IS_LOW_CONFIDENCE"]).sum()
low  = all_results["IS_LOW_CONFIDENCE"].sum()
print(f"\nThreshold: {CONFIDENCE_THRESHOLD}")
print(f"High-confidence: {high:,} ({high/len(all_results)*100:.1f}%)")
print(f"Low-confidence:  {low:,} ({low/len(all_results)*100:.1f}%)")
print("\nL4 Distribution:")
print(all_results["ASSIGNED_L4_LABEL"].value_counts().to_string())
print("\nDone.")
