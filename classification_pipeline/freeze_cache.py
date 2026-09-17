"""
Freeze an active, growing embedding cache into a dated, read-only volume under
artifacts/cache/frozen_volumes/, and reset the active cache to empty.

Run this manually whenever an active cache (embedding_cache_stage.pkl,
embedding_cache_services_stage.pkl, etc.) approaches the point where holding it in
memory alongside everything else phase_a/phase_cached needs gets risky — roughly
~1M entries / ~6GB on this machine (24GB RAM). Frozen volumes are picked up
automatically by classify_products.py's phase_a (and would be by any pipeline using
the same pattern) — no code changes needed after freezing.

Usage:
    python freeze_cache.py artifacts/cache/embedding_cache_stage.pkl
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FROZEN_VOLUMES_DIR = PROJECT_ROOT / "artifacts/cache/frozen_volumes"


def freeze(active_cache_path: Path) -> None:
    if not active_cache_path.exists():
        print(f"ERROR: {active_cache_path} does not exist.")
        sys.exit(1)

    print(f"Loading active cache ({active_cache_path.stat().st_size/1e9:.2f} GB)...")
    with open(active_cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache):,} entries")

    FROZEN_VOLUMES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frozen_path = FROZEN_VOLUMES_DIR / f"{active_cache_path.stem}_frozen_{timestamp}.pkl"

    print(f"Writing frozen volume: {frozen_path}")
    with open(frozen_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  {frozen_path.stat().st_size/1e9:.2f} GB written")

    print(f"Resetting active cache to empty: {active_cache_path}")
    tmp = active_cache_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({}, f)
    tmp.rename(active_cache_path)

    print("\nDone. Next phase_a run will pick up the new frozen volume automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_path", type=Path, help="Path to the active cache file to freeze")
    args = parser.parse_args()
    freeze(args.cache_path)
