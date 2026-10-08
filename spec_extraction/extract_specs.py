"""Unified spec-extraction CLI — dispatches to the old flat system (unchanged, for
~24 not-yet-migrated L2 categories) or the new hierarchical L1/L2/L3 system (for
categories in config.registry.MIGRATED_L2_IDS), without the caller needing to know
which system a given category belongs to.

--category accepts an old-style flat category id (e.g. chemicals_solvents), a
new-style L2 id (e.g. lab_supplies_consumables), or a specific L3 node id (e.g.
bottles_containers) to run just one of a migrated L2's tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TAXONOMY_L3_PATH = PROJECT_ROOT / "classification_pipeline/taxonomy/l3_taxonomy_anchors.json"


def find_parent_l2(l3_node_id: str) -> str | None:
    with open(TAXONOMY_L3_PATH) as f:
        data = json.load(f)
    for l2_id, subcats in data["l2_subcategories"].items():
        if any(s["id"] == l3_node_id for s in subcats):
            return l2_id
    return None


def resolve_system(category: str) -> tuple[str, str, str | None]:
    """Return (system, l2_or_old_category_id, l3_node_id_or_None)."""
    from spec_extraction.config import registry
    from spec_extraction.extract_product_specs import EXTRACTORS

    if category in registry.MIGRATED_L2_IDS:
        return "new", category, None
    if category in EXTRACTORS:
        return "old", category, None

    l2_id = find_parent_l2(category)
    if l2_id is not None:
        if l2_id in registry.MIGRATED_L2_IDS:
            return "new", l2_id, category
        raise SystemExit(
            f"'{category}' is an L3 node under L2 '{l2_id}', but '{l2_id}' hasn't been "
            f"migrated to the new spec model yet (see config/registry.py). It has no "
            f"old-system flat category either, so there's nothing to run for it yet."
        )

    raise SystemExit(f"Unknown category/L2/L3 id: '{category}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("stage", "prod"), default="stage")
    parser.add_argument("--mode", choices=("sample", "full"), default="sample")
    parser.add_argument("--category", required=True,
                         help="Old flat category id, new L2 id, or a specific L3 node id")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--review-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system, resolved_id, l3_node_id = resolve_system(args.category)

    if system == "old":
        from spec_extraction.extract_product_specs import (
            DEFAULT_OUTPUT_DIR, DEFAULT_REVIEW_SIZE, DEFAULT_SAMPLE_SIZE, run_full, run_sample,
        )
        print(f"[legacy system] category={resolved_id}")
        if args.mode == "sample":
            paths = run_sample(
                env=args.env, category=resolved_id,
                output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
                sample_size=args.sample_size or DEFAULT_SAMPLE_SIZE,
                review_size=args.review_size or DEFAULT_REVIEW_SIZE,
            )
            for p in paths:
                print(f"Wrote: {p}")
        else:
            table = run_full(env=args.env, category=resolved_id)
            print(f"Wrote full output table: {table}")
        return

    # system == "new"
    from spec_extraction.v2.runner import DEFAULT_OUTPUT_DIR, DEFAULT_REVIEW_SIZE, DEFAULT_SAMPLE_SIZE, run_full_v2, run_sample_v2
    print(f"[v2 system] l2={resolved_id}" + (f" l3={l3_node_id}" if l3_node_id else " (all L3 nodes)"))
    if args.mode == "sample":
        results = run_sample_v2(
            env=args.env, l2_id=resolved_id, l3_id=l3_node_id,
            output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
            sample_size=args.sample_size or DEFAULT_SAMPLE_SIZE,
            review_size=args.review_size or DEFAULT_REVIEW_SIZE,
        )
        for node_id, paths in results.items():
            print(f"-- {node_id} --")
            for label, p in paths.items():
                print(f"  {label}: {p}")
    else:
        written = run_full_v2(env=args.env, l2_id=resolved_id, l3_id=l3_node_id)
        for table, count in written:
            print(f"Wrote {count:,} rows to {table}")


if __name__ == "__main__":
    main()
