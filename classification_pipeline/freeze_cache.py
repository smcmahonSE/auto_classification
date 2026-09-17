"""
Manually freeze an active, growing embedding cache into a dated, read-only
volume, and reset the active cache to empty.

Normally you shouldn't need this — both classify_products.py's phase_embed and
classify_services.py's phase_embed auto-freeze their active cache once it
crosses FREEZE_THRESHOLD_BYTES (~6GB). Use this script only to freeze early
(e.g. before a big batch you know will push it over the line) or to freeze a
cache that isn't wired into that auto-freeze logic yet.

Usage:
    python freeze_cache.py artifacts/cache/embedding_cache_stage.pkl artifacts/cache/frozen_volumes
    python freeze_cache.py artifacts/cache/embedding_cache_services_stage.pkl artifacts/cache/frozen_volumes_services
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "classification_pipeline"))

from product_classifier_utils import freeze_active_cache


def freeze(active_cache_path: Path, frozen_volumes_dir: Path) -> None:
    if not active_cache_path.exists():
        print(f"ERROR: {active_cache_path} does not exist.")
        sys.exit(1)

    print(f"Loading active cache ({active_cache_path.stat().st_size/1e9:.2f} GB)...")
    frozen_path = freeze_active_cache(active_cache_path, frozen_volumes_dir)
    print(f"Wrote frozen volume: {frozen_path} ({frozen_path.stat().st_size/1e9:.2f} GB)")
    print(f"Active cache reset to empty: {active_cache_path}")
    print("\nDone. Next phase_a/phase_cached run will pick up the new frozen volume automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_path", type=Path, help="Path to the active cache file to freeze")
    parser.add_argument("frozen_volumes_dir", type=Path,
                        help="Directory to write the frozen volume into (e.g. artifacts/cache/frozen_volumes "
                             "for products, artifacts/cache/frozen_volumes_services for services)")
    args = parser.parse_args()
    freeze(args.cache_path, args.frozen_volumes_dir)
