"""
Classify LCG listings using pre-extracted vectors (no full cache load needed).

Run AFTER extract_lcg_vectors.py:
    python classify_lcg.py

Then return to the notebook and run Steps 7-10 with MODE = "classify_lcg".
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from product_classifier_utils import embed_texts_from_cache, get_bedrock_client, stable_text_hash

MODE              = "classify_lcg"
AWS_PROFILE       = "staging.admin"
AWS_REGION        = "us-east-1"
MODEL_ID          = "amazon.titan-embed-text-v1"
MARGIN_THRESHOLD  = 0.05
BATCH_SIZE        = 250_000

ANCHOR_CACHE_PATH = PROJECT_ROOT / "artifacts/cache/anchor_cache.pkl"
ANCHORS_PATH      = PROJECT_ROOT / "analysis/data/l3_taxonomy_anchors.json"
OUTPUT_DIR        = PROJECT_ROOT / f"artifacts/analysis/l4_classification_{MODE}"
V1_WORK_PATH      = OUTPUT_DIR / "v1_work.parquet"
VECTORS_PATH      = OUTPUT_DIR / "lcg_v1_vectors.npy"
HASHES_PATH       = OUTPUT_DIR / "lcg_v1_hashes.pkl"
V1_RESULTS_PATH   = OUTPUT_DIR / "classification_results_v1.csv"

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

# ── Load pre-extracted vectors ─────────────────────────────────────────────────
for p in [V1_WORK_PATH, VECTORS_PATH, HASHES_PATH]:
    if not p.exists():
        print(f"ERROR: {p} not found. Run extract_lcg_vectors.py first.")
        sys.exit(1)

print(f"\nLoading pre-extracted vectors from {VECTORS_PATH}...")
vectors = np.load(VECTORS_PATH)
v1_work = pd.read_parquet(V1_WORK_PATH)
print(f"Vectors: {vectors.shape} ({vectors.nbytes/1e9:.2f} GB)")

output_cols = ["PRODUCT_ID", "ASSIGNED_L4_ID", "ASSIGNED_L4_LABEL",
               "CONFIDENCE", "CONFIDENCE_MARGIN", "IS_LOW_CONFIDENCE"]
for col in ["CURRENT_L3", "CURRENT_L4", "CURRENT_L5"]:
    if col in v1_work.columns:
        output_cols.insert(2, col)

# ── Batched classification ─────────────────────────────────────────────────────
print(f"\nClassifying {len(v1_work):,} listings in batches of {BATCH_SIZE:,}...")
results = []

for start in range(0, len(v1_work), BATCH_SIZE):
    end   = min(start + BATCH_SIZE, len(v1_work))
    vecs  = vectors[start:end].astype(np.float32)
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

    rec = v1_work.iloc[start:end].drop(columns=["_HASH"], errors="ignore").copy()
    rec["ASSIGNED_L4_ID"]    = [anchor_ids[i]    for i in top1]
    rec["ASSIGNED_L4_LABEL"] = [anchor_labels[i] for i in top1]
    rec["CONFIDENCE"]        = top1_s.round(4)
    rec["CONFIDENCE_MARGIN"] = margin.round(4)
    rec["IS_LOW_CONFIDENCE"] = margin < MARGIN_THRESHOLD
    results.append(rec[[c for c in output_cols if c in rec.columns]])

    pct  = end / len(v1_work) * 100
    high = (margin >= MARGIN_THRESHOLD).sum()
    print(f"  {start:,}–{end:,} ({pct:.0f}%) — high-conf: {high:,}/{end-start:,}")

all_v1 = pd.concat(results, ignore_index=True)
all_v1.to_csv(V1_RESULTS_PATH, index=False)

high = (~all_v1["IS_LOW_CONFIDENCE"]).sum()
low  =   all_v1["IS_LOW_CONFIDENCE"].sum()
print(f"\nV1 results saved: {V1_RESULTS_PATH} ({len(all_v1):,} rows)")
print(f"High-confidence: {high:,} ({high/len(all_v1)*100:.1f}%)")
print(f"Residual:        {low:,} ({low/len(all_v1)*100:.1f}%)")
print("\nDone. Return to the notebook and run Steps 7-10.")
