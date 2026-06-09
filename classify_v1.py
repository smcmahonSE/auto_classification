"""
Phase B classification: volume 1 listings (main embedding cache).

Loads v1_work.parquet (saved by the notebook after Phase A), loads the main
32GB embedding cache, classifies in batches, saves v1 results CSV.

Run AFTER the notebook has completed Phase A (Step 6):
    python classify_v1.py

Then return to the notebook and run the combine cell.
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
    embed_texts_from_cache,
    get_bedrock_client,
    stable_text_hash,
)

# ── Config — must match the notebook ──────────────────────────────────────────
MODE               = "classify_lcg"
AWS_PROFILE        = "staging.admin"
AWS_REGION         = "us-east-1"
MODEL_ID           = "amazon.titan-embed-text-v1"
CONFIDENCE_THRESHOLD = 0.50
BATCH_SIZE         = 250_000

CACHE_V1_PATH      = PROJECT_ROOT / "artifacts/cache/embedding_cache.pkl"
ANCHOR_CACHE_PATH  = PROJECT_ROOT / "artifacts/cache/anchor_cache.pkl"
ANCHORS_PATH       = PROJECT_ROOT / "analysis/data/l4_taxonomy_anchors.json"
OUTPUT_DIR         = PROJECT_ROOT / f"artifacts/analysis/l4_classification_{MODE}"
V1_WORK_PATH       = OUTPUT_DIR / "v1_work.parquet"
V1_RESULTS_PATH    = OUTPUT_DIR / "classification_results_v1.csv"

# ── Load anchors ───────────────────────────────────────────────────────────────
print("Loading anchors...")
with open(ANCHORS_PATH) as f:
    anchors_data = json.load(f)
anchors       = anchors_data["anchors"]
anchor_ids    = [a["id"]          for a in anchors]
anchor_labels = [a["label"]       for a in anchors]
anchor_texts  = [a["description"] for a in anchors]

print("Embedding anchors (using anchor_cache.pkl)...")
bedrock = get_bedrock_client(profile_name=AWS_PROFILE, region=AWS_REGION)

with open(ANCHOR_CACHE_PATH, "rb") as f:
    anchor_cache = pickle.load(f)

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

# ── Load v1 work file ─────────────────────────────────────────────────────────
if not V1_WORK_PATH.exists():
    print(f"ERROR: {V1_WORK_PATH} not found. Run the notebook Step 6 (Phase A) first.")
    sys.exit(1)

print(f"\nLoading v1 work file: {V1_WORK_PATH}")
v1_work = pd.read_parquet(V1_WORK_PATH)
hashes  = v1_work["_HASH"].tolist()
print(f"V1 listings: {len(v1_work):,}")

# ── Load volume 1 cache ───────────────────────────────────────────────────────
print(f"\nLoading volume 1 cache ({CACHE_V1_PATH.stat().st_size/1e9:.1f} GB) ...")
print("This will take several minutes and may use swap on machines with < 40GB RAM.")
with open(CACHE_V1_PATH, "rb") as f:
    cache_v1 = pickle.load(f)
print(f"Volume 1 loaded: {len(cache_v1):,} entries")

# Filter out any hashes not actually present in cache_v1 (edge cases from key index drift)
cache_v1_keys = set(cache_v1.keys())
not_found_mask = [h not in cache_v1_keys for h in hashes]
n_missing = sum(not_found_mask)
if n_missing:
    print(f"WARNING: {n_missing:,} hashes not found in volume 1 — saving to needs_retry.parquet")
    not_found_indices = [i for i, m in enumerate(not_found_mask) if m]
    v1_work.iloc[not_found_indices].to_parquet(OUTPUT_DIR / "needs_retry.parquet", index=False)
    valid_indices = [i for i, m in enumerate(not_found_mask) if not m]
    hashes   = [hashes[i] for i in valid_indices]
    v1_work  = v1_work.iloc[valid_indices].reset_index(drop=True)
    print(f"Proceeding with {len(v1_work):,} classifiable listings.")

# ── Batched classification ────────────────────────────────────────────────────
output_cols = ["PRODUCT_ID", "ASSIGNED_L4_ID", "ASSIGNED_L4_LABEL",
               "CONFIDENCE", "CONFIDENCE_MARGIN", "IS_LOW_CONFIDENCE"]
for col in ["CURRENT_L3", "CURRENT_L4", "CURRENT_L5"]:
    if col in v1_work.columns:
        output_cols.insert(2, col)

print(f"\nClassifying {len(v1_work):,} listings in batches of {BATCH_SIZE:,}...")
results = []

for start in range(0, len(v1_work), BATCH_SIZE):
    end   = min(start + BATCH_SIZE, len(v1_work))
    batch_hashes = hashes[start:end]
    batch_work   = v1_work.iloc[start:end]

    vecs  = np.array([cache_v1[h] for h in batch_hashes], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.clip(norms, 1e-10, None)
    del vecs

    sim    = vecs_n @ anchor_normed.T
    del vecs_n

    top1   = sim.argmax(axis=1)
    top1_s = sim[np.arange(len(sim)), top1]
    sim2   = sim.copy(); sim2[np.arange(len(sim)), top1] = -1
    margin = top1_s - sim2.max(axis=1)
    del sim, sim2

    rec = batch_work.drop(columns=["_HASH"]).copy()
    rec["ASSIGNED_L4_ID"]    = [anchor_ids[i]    for i in top1]
    rec["ASSIGNED_L4_LABEL"] = [anchor_labels[i] for i in top1]
    rec["CONFIDENCE"]        = top1_s.round(4)
    rec["CONFIDENCE_MARGIN"] = margin.round(4)
    rec["IS_LOW_CONFIDENCE"] = top1_s < CONFIDENCE_THRESHOLD
    results.append(rec[[c for c in output_cols if c in rec.columns]])

    pct  = end / len(v1_work) * 100
    high = (top1_s >= CONFIDENCE_THRESHOLD).sum()
    print(f"  {start:,}–{end:,} ({pct:.0f}%) — high-conf: {high:,}/{end-start:,}")
    del rec

print("\nCombining batches...")
all_v1 = pd.concat(results, ignore_index=True)
del results

all_v1.to_csv(V1_RESULTS_PATH, index=False)
print(f"V1 results saved: {V1_RESULTS_PATH} ({len(all_v1):,} rows)")

high = (~all_v1["IS_LOW_CONFIDENCE"]).sum()
low  =   all_v1["IS_LOW_CONFIDENCE"].sum()
print(f"\nThreshold: {CONFIDENCE_THRESHOLD}")
print(f"High-confidence: {high:,} ({high/len(all_v1)*100:.1f}%)")
print(f"Low-confidence:  {low:,} ({low/len(all_v1)*100:.1f}%)")
print("\nDone. Return to the notebook and run the combine cell.")
