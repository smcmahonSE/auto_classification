"""Triggers an immediate refresh of one SPECIFICATION_COVERAGE_SHARD_<L2_ID>_<ENV>.

TARGET_LAG on each shard is a loose backstop (see generate_coverage_sql.py) —
real freshness comes from calling this right after a full-mode spec-extraction
publish actually changes the data that (L2, env) shard reads. Only the
specific (L2, env) that was just published needs refreshing — the sharded
design (one dynamic table per (L2, env), unioned by the
SPECIFICATION_COVERAGE_SUMMARY view) means other shards, and the view
itself, don't need to be touched.

Called from spec_extraction/extract_specs.py's main(), not meant to be run
standalone (though `python3 -m sql.refresh <l2_id> <env>` works for a manual
nudge).
"""

from __future__ import annotations

import sys

from classification_pipeline.product_classifier_utils import get_products_session
from sql.generate_coverage_sql import DATABASE, TARGET_SCHEMA, shard_name


def refresh_specification_coverage_shard(l2_id: str, env: str) -> None:
    sf = get_products_session()
    try:
        sf.sql(f"ALTER DYNAMIC TABLE {DATABASE}.{TARGET_SCHEMA}.{shard_name(l2_id, env)} REFRESH").collect()
    finally:
        sf.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 -m sql.refresh <l2_id> <env>")
    l2_id, env = sys.argv[1], sys.argv[2]
    refresh_specification_coverage_shard(l2_id, env)
    print(f"Refreshed {DATABASE}.{TARGET_SCHEMA}.{shard_name(l2_id, env)}.")
