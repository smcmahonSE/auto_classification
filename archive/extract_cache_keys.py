"""
Extract just the hash keys from the main embedding cache into a small index file.
Run this as a standalone script (NOT inside Jupyter) so the 32GB cache is loaded
and freed in a separate process — the notebook never needs to hold it in RAM.

Usage:
    python extract_cache_keys.py

Output:
    artifacts/cache/embedding_cache_keys.pkl   (~350MB set of SHA-256 strings)
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_PATH   = PROJECT_ROOT / "artifacts/cache/embedding_cache.pkl"
KEYS_PATH    = PROJECT_ROOT / "artifacts/cache/embedding_cache_keys.pkl"

if not CACHE_PATH.exists():
    print(f"ERROR: Cache not found at {CACHE_PATH}")
    sys.exit(1)

print(f"Loading {CACHE_PATH} ({CACHE_PATH.stat().st_size / 1e9:.1f} GB) ...")
print("This will use ~32GB of RAM temporarily — the notebook will NOT need to do this.")

with open(CACHE_PATH, "rb") as f:
    cache = pickle.load(f)

keys = set(cache.keys())
n = len(keys)
print(f"Loaded {n:,} entries. Extracting keys ...")

del cache  # free 32GB before writing

tmp_path = KEYS_PATH.with_suffix(".pkl.tmp")
with open(tmp_path, "wb") as f:
    pickle.dump(keys, f, protocol=pickle.HIGHEST_PROTOCOL)
tmp_path.replace(KEYS_PATH)

size_mb = KEYS_PATH.stat().st_size / 1e6
print(f"Keys saved: {n:,} hashes → {KEYS_PATH} ({size_mb:.0f} MB)")
print("You can now load this key set in the notebook instead of the full cache.")
