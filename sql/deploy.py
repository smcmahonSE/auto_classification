#!/usr/bin/env python3
"""Deploys a .sql file in this folder to Snowflake.

A file here holds one *or more* statements, separated by the sentinel comment
line generate_coverage_sql.STATEMENT_SEPARATOR uses (not bare ';' — a quoted
field label could in principle contain one). Each statement is executed in
order via a single session.

WAREHOUSE clauses use the literal placeholder {{WAREHOUSE}} instead of a
hardcoded name. The default used to be CURRENT_WAREHOUSE() ("whatever the
deploying session's context already is"), but that turned out to be
unreliable in practice: it returned SNOWPARK_WH — a real, enabled warehouse,
just one DEPT-ENGINEERING (the role get_products_session() uses) has no USAGE
grant on — because that was a stale per-user session default that survived
the USE ROLE switch without actually being usable under the new role.
Confirmed via `SHOW GRANTS TO ROLE "DEPT-ENGINEERING"` that it only has USAGE
on COMPUTE_WH, DBT_WH, DEV_WH, and SNOWPARK_XL_WH (X-Large, suspended — too
expensive for a routine full-scan refresh). DEFAULT_WAREHOUSE below is
COMPUTE_WH, chosen as the generic/cheapest confirmed-usable option; override
with --warehouse for anything that should run elsewhere.

Usage:
    python3 deploy.py specification_coverage_summary.sql
    python3 deploy.py specification_coverage_summary.sql --warehouse DEV_WH
    python3 deploy.py specification_coverage_summary.sql --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from classification_pipeline.product_classifier_utils import get_products_session  # noqa: E402
from sql.generate_coverage_sql import STATEMENT_SEPARATOR  # noqa: E402

WAREHOUSE_PLACEHOLDER = "{{WAREHOUSE}}"
DEFAULT_WAREHOUSE = "COMPUTE_WH"


def split_statements(sql_text: str) -> list[str]:
    """Strip the leading file-level comment header, then split on the sentinel."""
    # Everything before the first sentinel (if the file starts with one) is a
    # header comment, not a statement — drop empty/whitespace-only chunks.
    return [chunk.strip() for chunk in sql_text.split(STATEMENT_SEPARATOR.strip("\n")) if chunk.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sql_file", type=Path, help="Path to a .sql file in this folder")
    parser.add_argument(
        "--warehouse",
        default=DEFAULT_WAREHOUSE,
        help=f"Warehouse name to substitute for {{{{WAREHOUSE}}}} (default: {DEFAULT_WAREHOUSE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved statements without executing them",
    )
    args = parser.parse_args()

    sql_text = args.sql_file.read_text()
    statements = split_statements(sql_text)
    resolved_statements = [s.replace(WAREHOUSE_PLACEHOLDER, args.warehouse) for s in statements]

    if args.dry_run:
        for i, stmt in enumerate(resolved_statements, 1):
            print(f"-- statement {i}/{len(resolved_statements)} --")
            print(stmt)
            print()
        return

    print(f"Deploying {args.sql_file.name} ({len(resolved_statements)} statement(s), WAREHOUSE = {args.warehouse}) ...")
    sf = get_products_session()
    try:
        for i, stmt in enumerate(resolved_statements, 1):
            # First line (before whitespace/newline) is enough to identify the object.
            first_line = stmt.strip().splitlines()[0][:100]
            print(f"[{i}/{len(resolved_statements)}] {first_line}")
            sf.sql(stmt).collect()
        print("Done.")
    finally:
        sf.close()


if __name__ == "__main__":
    main()
