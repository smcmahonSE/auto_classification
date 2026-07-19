"""
Extract only the LCG listing vectors from the main cache into a small numpy file.

Loads the full 32GB cache once, pulls out only the hashes needed for LCG
classification (~450K vectors = ~2.8GB), saves to disk, then exits.

Run BEFORE classify_lcg.py:
    python extract_lcg_vectors.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_V1_PATH  = PROJECT_ROOT / "artifacts/cache/embedding_cache.pkl"
MODE           = "classify_lcg"
OUTPUT_DIR     = PROJECT_ROOT / f"artifacts/analysis/l4_classification_{MODE}"
V1_WORK_PATH   = OUTPUT_DIR / "v1_work.parquet"
VECTORS_PATH   = OUTPUT_DIR / "lcg_v1_vectors.npy"
HASHES_PATH    = OUTPUT_DIR / "lcg_v1_hashes.pkl"

if not V1_WORK_PATH.exists():
    print(f"ERROR: {V1_WORK_PATH} not found. Run Step 6 in the notebook first.")
    sys.exit(1)

print(f"Loading v1 work file...")
v1_work = pd.read_parquet(V1_WORK_PATH)
hashes  = v1_work["_HASH"].tolist()
print(f"Need vectors for {len(hashes):,} LCG listings")

print(f"\nLoading main cache ({CACHE_V1_PATH.stat().st_size/1e9:.1f} GB) ...")
print("This is the last time we need to load it for LCG.")
with open(CACHE_V1_PATH, "rb") as f:
    cache_v1 = pickle.load(f)
print(f"Cache loaded: {len(cache_v1):,} entries")

# Extract only the needed vectors
cache_v1_keys = set(cache_v1.keys())
not_found = [h for h in hashes if h not in cache_v1_keys]
if not_found:
    print(f"WARNING: {len(not_found):,} hashes not found in cache — will be skipped")
    valid_mask = [h in cache_v1_keys for h in hashes]
    hashes  = [h for h, m in zip(hashes, valid_mask) if m]
    v1_work = v1_work[[m for m in valid_mask]].reset_index(drop=True)

print(f"Extracting {len(hashes):,} vectors...")
vectors = np.array([cache_v1[h] for h in hashes], dtype=np.float32)
del cache_v1  # free 32GB immediately
print(f"Vectors shape: {vectors.shape} ({vectors.nbytes/1e9:.2f} GB)")

np.save(VECTORS_PATH, vectors)
del vectors

with open(HASHES_PATH, "wb") as f:
    pickle.dump(hashes, f)

# Save filtered work file if any were skipped
v1_work.to_parquet(V1_WORK_PATH, index=False)

print(f"\nSaved:")
print(f"  {VECTORS_PATH} ({VECTORS_PATH.stat().st_size/1e9:.2f} GB)")
print(f"  {HASHES_PATH}")
print("\nRun classify_lcg.py to complete classification.")
