#!/usr/bin/env python3
"""Generates specification_coverage_summary.sql from the live v2 spec configs.

SPECIFICATION_COVERAGE_SUMMARY reports, per (L2, L3, specification), what % of
rows in that L3's published Snowflake table have a non-null extracted value
for that field. This script is the single source of truth for its DDL — the
same role spec_extraction/schemas/spec_model_v2 plays for extraction itself.
specification_coverage_summary.sql is *derived output*; never hand-edit it.
Re-run this script (and redeploy via deploy.py) any time an L2/L3 config
changes: new category, new L3 node, new/renamed/removed field.

Architecture — one dynamic table per (L2, env), unioned by a plain view:
A single dynamic table unioning all 181 L3 tables was tried first and
rejected: it compiled, but execution consistently hit a Snowflake-side
internal error (XX000, error 300002, reproduced identically across repeated
attempts — not transient) once that many deeply-nested UNION ALL branches
were combined into one object. A 12-branch (2-L2) version of the exact same
per-table query pattern deployed and ran cleanly, confirming it's a scale
limit on one monolithic object, not a flaw in the per-table query itself. We
never bisected the exact ceiling between 12 (fine) and 181 (fails).
So instead: one small dynamic table *per (L2, env)* (SPECIFICATION_COVERAGE_
SHARD_<L2_ID>_<ENV>, 3-17 branches each today — comfortably inside the
proven-working range), then SPECIFICATION_COVERAGE_SUMMARY as a plain VIEW
unioning all the shards. Sharding by (L2, env) rather than just L2 matters
specifically because we don't know that ceiling: it means adding "prod" to
ENVS_TO_INCLUDE creates 25 new small shards rather than doubling the branch
count inside each existing one, so every shard stays exactly the size it is
today regardless of how many envs get added later. Views aren't materialized,
so growing the view from 25 to 50 trivial `SELECT * FROM shard` branches
costs nothing — that's not the kind of complexity that caused the original
failure. Per-shard sharding also means a single category's publish only
needs to refresh *its own* shard (see sql/refresh.py), not recompute all 181
tables like the monolithic version would have.

Scope: only L2/L3-tier fields are tracked (tier == "l1" fields are the
universal platform columns — Name, Price, Category, etc. — which are set
directly by the platform, not extracted from supplier text, so "coverage"
isn't a meaningful concept for them).

Currently STAGE-only, since no L2/L3 spec tables exist in PROD yet. Once they
do, add "prod" to ENVS_TO_INCLUDE below — each L2 then gets a second,
separate shard (SPECIFICATION_COVERAGE_SHARD_<L2_ID>_PROD) with
SOURCE = 'PROD' rows, unioned into the same summary view. The per-table
block logic doesn't need to change.

Usage:
    python3 generate_coverage_sql.py
    # writes specification_coverage_summary.sql next to this script
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from spec_extraction.config.loader import FieldDef, get_effective_field_set, load_l2  # noqa: E402
from spec_extraction.config.registry import MIGRATED_L2_IDS  # noqa: E402
from spec_extraction.extract_product_specs import normalize_field_name  # noqa: E402
from spec_extraction.v2.runner import BASE_COLUMNS, ENV_SUFFIX  # noqa: E402

DATABASE = "SNOWFLAKE_LEARNING_DB"
TARGET_SCHEMA = "SMCMAHON_PRODUCTS"  # same schema NEW_CLASSIFICATIONS_STAGE lives in
TARGET_TABLE = "SPECIFICATION_COVERAGE_SUMMARY"  # the public-facing view
SHARD_PREFIX = "SPECIFICATION_COVERAGE_SHARD_"  # internal, one dynamic table per L2
TARGET_LAG = "7 days"  # loose staleness backstop; real freshness comes from the
# explicit ALTER DYNAMIC TABLE ... REFRESH triggered after each full-mode publish
# (see sql/refresh.py) for the specific L2 that was just published.
WAREHOUSE_PLACEHOLDER = "{{WAREHOUSE}}"  # deploy.py substitutes its DEFAULT_WAREHOUSE (or --warehouse) in
ENVS_TO_INCLUDE = ["stage"]  # add "prod" once PROD spec tables exist

RESERVED_COLUMNS = {c.upper() for c in BASE_COLUMNS} | {"LISTING_TYPE_GUESS"}

OUTPUT_PATH = Path(__file__).resolve().parent / "specification_coverage_summary.sql"

# Statements in the output file are separated by this sentinel line so
# deploy.py can split and execute them one at a time (can't just split on
# bare ';' — nothing stops a quoted field label from containing one).
STATEMENT_SEPARATOR = "\n-- @@STATEMENT_BREAK@@\n"


def sql_quote(s: str) -> str:
    return s.replace("'", "''")


def shard_name(l2_id: str, env: str) -> str:
    # Sharded by (L2, env), not just L2: adding "prod" to ENVS_TO_INCLUDE creates
    # new, separate shards rather than doubling the block count inside existing
    # ones. We don't know the exact ceiling that caused the monolithic-table
    # failure (12 blocks worked, 181 didn't; never bisected further) — this
    # keeps every shard permanently the same size it is today, so there's
    # nothing to re-discover a limit against later.
    return f"{SHARD_PREFIX}{l2_id.upper()}_{ENV_SUFFIX[env]}"


def build_table_block(
    l2_id: str,
    l2_label: str,
    schema_name: str,
    l3_id: str,
    l3_label: str,
    table_name: str,
    fields: list[FieldDef],
    env: str,
) -> str | None:
    """One UNION ALL branch covering every L2/L3-tier field in a single L3 table."""
    seen_cols: set[str] = set()
    rows: list[tuple[str, str, bool]] = []  # (column, human label, required)

    for f in fields:
        if f.tier == "l1":
            continue
        col = normalize_field_name(f.name).upper()
        if col in RESERVED_COLUMNS:
            print(
                f"WARNING: {l2_id}.{l3_id} field {f.name!r} -> column {col} collides "
                f"with a base column; skipping (would not reflect the extracted value)",
                file=sys.stderr,
            )
            continue
        if col in seen_cols:
            print(
                f"WARNING: {l2_id}.{l3_id} field {f.name!r} -> column {col} duplicates "
                f"another field's column; skipping",
                file=sys.stderr,
            )
            continue
        seen_cols.add(col)
        rows.append((col, f.name, bool(f.required)))

    if not rows:
        print(f"NOTE: {l2_id}.{l3_id} has no L2/L3-tier fields; skipping entirely", file=sys.stderr)
        return None

    suffix = ENV_SUFFIX[env]
    source_label = suffix  # "STAGE" / "PROD" — matches the SOURCE column values requested
    full_table = f'{DATABASE}."{schema_name}".{table_name}_{suffix}'

    # Always double-quote: some field names normalize to identifiers starting with
    # a digit (e.g. "24-hour coverage" -> 24_HOUR_COVERAGE, "21 CFR Part 11
    # validated" -> 21_CFR_PART_11_VALIDATED), which aren't valid bare unquoted
    # SQL identifiers — Snowflake's parser reads the leading digits as a number
    # literal. The real table's Snowpark-created column is quoted for the same
    # reason, so quoting here matches it exactly regardless of the name's shape.
    count_exprs = ",\n    ".join(f'COUNT("{col}") AS "{col}"' for col, _, _ in rows)
    values_rows = ",\n    ".join(
        f"('{col}', '{sql_quote(label)}', {str(required).upper()})" for col, label, required in rows
    )

    return f"""(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      {count_exprs}
    FROM {full_table}
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    {values_rows}
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    '{sql_quote(l2_id)}' AS ASSIGNED_L2_ID,
    '{sql_quote(l2_label)}' AS ASSIGNED_L2_LABEL,
    '{sql_quote(l3_id)}' AS ASSIGNED_L3_ID,
    '{sql_quote(l3_label)}' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    '{source_label}' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- {l2_label} > {l3_label}
)"""


def build_shard_statement(l2_id: str, env: str) -> tuple[str, int, int] | None:
    """One dynamic table per (L2, env), unioning that L2's L3 table blocks for
    just that one env. Returns (statement, l3_table_count, skipped_count), or
    None if the L2 contributes no L3 tables at all for this env."""
    l2 = load_l2(l2_id)
    blocks: list[str] = []
    skipped = 0

    for l3_id, node in l2.l3_nodes.items():
        fields = get_effective_field_set(l2_id, l3_id)
        block = build_table_block(
            l2_id=l2_id,
            l2_label=l2.l2_label,
            schema_name=l2.schema_name,
            l3_id=l3_id,
            l3_label=node.label,
            table_name=node.table_name,
            fields=fields,
            env=env,
        )
        if block is None:
            skipped += 1
        else:
            blocks.append(block)

    if not blocks:
        return None

    body = "\nUNION ALL\n".join(blocks)
    statement = f"""CREATE OR REPLACE DYNAMIC TABLE {DATABASE}.{TARGET_SCHEMA}.{shard_name(l2_id, env)}
  TARGET_LAG = '{TARGET_LAG}'
  WAREHOUSE = {WAREHOUSE_PLACEHOLDER}
AS
{body}
;"""
    return statement, len(blocks), skipped


def build_summary_view_statement(shard_keys: list[tuple[str, str]]) -> str:
    union_branches = "\nUNION ALL\n".join(
        f"SELECT * FROM {DATABASE}.{TARGET_SCHEMA}.{shard_name(l2_id, env)}" for l2_id, env in shard_keys
    )
    return f"""CREATE OR REPLACE VIEW {DATABASE}.{TARGET_SCHEMA}.{TARGET_TABLE} AS
{union_branches}
;"""


def main() -> None:
    statements: list[str] = []
    shard_keys: list[tuple[str, str]] = []  # (l2_id, env) pairs that got a shard
    table_count = 0
    skipped_count = 0

    for l2_id in sorted(MIGRATED_L2_IDS):
        for env in ENVS_TO_INCLUDE:
            result = build_shard_statement(l2_id, env)
            if result is None:
                print(f"NOTE: L2 '{l2_id}' ({env}) contributed no L3 tables; no shard created", file=sys.stderr)
                continue
            statement, n_tables, n_skipped = result
            statements.append(statement)
            shard_keys.append((l2_id, env))
            table_count += n_tables
            skipped_count += n_skipped

    statements.append(build_summary_view_statement(shard_keys))

    header = f"""-- GENERATED FILE — do not hand-edit.
-- Regenerate with: python3 sql/generate_coverage_sql.py
-- Source of truth: spec_extraction/schemas/spec_model_v2/l2/*.json via
-- spec_extraction/config/loader.py, enumerated through
-- spec_extraction/config/registry.py's MIGRATED_L2_IDS.
--
-- {table_count} L3 table(s) covered across {len(shard_keys)} (L2, env) shard(s)
-- ({skipped_count} skipped — see stderr from the generator run for why).
-- {len(statements) - 1} shard dynamic table(s) + 1 summary view. Sharded by
-- (L2, env), not just L2 — see this file's module docstring for why.
--
-- WAREHOUSE is a placeholder token; deploy.py substitutes its configured
-- default warehouse (or --warehouse override) before executing each shard
-- statement (the summary view doesn't need one — views aren't materialized).
--
-- Statements are separated by a sentinel comment line, not bare ';', since a
-- quoted field label could in principle contain one; deploy.py splits on
-- that sentinel and executes each statement in order.
"""
    sql = header + STATEMENT_SEPARATOR.join(statements) + "\n"
    OUTPUT_PATH.write_text(sql)
    print(
        f"Wrote {OUTPUT_PATH} — {table_count} table(s) across {len(shard_keys)} shard(s) "
        f"+ 1 summary view, {skipped_count} table(s) skipped.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
