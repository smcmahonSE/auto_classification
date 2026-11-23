# sql/

Repo-governed Snowflake object definitions. This folder didn't exist before
`SPECIFICATION_COVERAGE_SUMMARY` — it's the first place in the repo that
tracks Snowflake DDL as code instead of leaving it to live only in Snowflake.

## Convention

- One `.sql` file per logical feature, named after its main object
  (`specification_coverage_summary.sql` → `SPECIFICATION_COVERAGE_SUMMARY`).
  A file may hold **multiple statements** if the feature needs more than one
  object (see below) — each generated and executed independently, never one
  giant combined statement.
- Multi-statement files separate statements with the sentinel comment line
  `generate_coverage_sql.STATEMENT_SEPARATOR` defines (`@@STATEMENT_BREAK@@`),
  not a bare `;` — a quoted field label could in principle contain one.
  `deploy.py` splits on that sentinel and executes each statement in order.
- A file is either **hand-written** or **generated** — generated files start
  with a `-- GENERATED FILE — do not hand-edit.` header naming the generator
  script and source of truth. Edit the generator (or the config it reads
  from) and re-run it, never the `.sql` output directly.
- No hardcoded warehouse name exists anywhere in this repo (see
  `classification_pipeline/product_classifier_utils.get_snowflake_session()` —
  it's purely env-var/session-default driven). Any object needing a
  `WAREHOUSE = ...` clause (dynamic tables require one; views/plain tables
  don't) should use the literal placeholder `{{WAREHOUSE}}` in its `.sql`
  file and let `deploy.py` resolve it at deploy time — see below.

## Objects

| File | Object(s) | Kind | Generator |
|---|---|---|---|
| `specification_coverage_summary.sql` | 25× `SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_<L2_ID>_<ENV>` (one per migrated L2, per env — currently STAGE only) | Dynamic table | `generate_coverage_sql.py` |
| ″ | `SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SUMMARY` | View (unions all shards) | ″ |

Not yet tracked here, but should be eventually (see `classification_pipeline/README.md`):
`COMPLETE_STAGE` (exists in Snowflake, no repo-tracked DDL) and `COMPLETE_PROD`
(doesn't exist yet). Pulling these in is future work, not blocking.

## SPECIFICATION_COVERAGE_SUMMARY

One row per (L2, L3, specification): what % of that L3's published Snowflake
rows have a non-null extracted value for that field.

| Column | Meaning |
|---|---|
| `ASSIGNED_L2_ID` / `ASSIGNED_L2_LABEL` | L2 category id / human label |
| `ASSIGNED_L3_ID` / `ASSIGNED_L3_LABEL` | L3 node id / human label |
| `SPECIFICATION` | The field's human-readable name (from the JSON config) |
| `COVERAGE_PCT` | `% of rows with a non-null value for this field` (0–100) |
| `REQUIRED` | `TRUE`/`FALSE`, from the field's `"required"` key in its config |
| `SOURCE` | `'STAGE'` or `'PROD'` |

**Scope**: only L2/L3-tier fields are included — not the L1 universal
platform fields (Name, Price, Category, etc.), since those are set directly
from the source listing, not extracted from supplier text, so "coverage"
isn't meaningful for them. See `generate_coverage_sql.py`'s module docstring.

**Currently STAGE-only.** No L2/L3 spec tables exist in PROD yet. Once they
do, add `"prod"` to `ENVS_TO_INCLUDE` in `generate_coverage_sql.py` and
regenerate — each table then gets a second branch with `SOURCE = 'PROD'`; no
other logic changes.

### Architecture: sharded, not one monolithic dynamic table

The obvious design — one dynamic table unioning all 181 L3 tables — was
tried first and **rejected**: the DDL compiled fine, but execution
consistently hit a Snowflake-side internal error (`XX000`, error `300002`,
reproduced identically — a hard, repeatable limit, not a transient blip)
once that many deeply-nested `UNION ALL` branches landed in one object. A
12-branch (2-category) version of the exact same per-table query deployed
and ran cleanly, which confirmed it's a scale limit on one object, not a
flaw in the per-table query pattern itself.

So instead: **one small dynamic table per (L2, env)** (`SPECIFICATION_COVERAGE_
SHARD_<L2_ID>_<ENV>`, 3–17 L3 branches each — well inside the proven-working
range), and `SPECIFICATION_COVERAGE_SUMMARY` is a plain **view** that unions
all the shards (25 today, one per L2 × the one env currently in
`ENVS_TO_INCLUDE`). Views aren't materialized, so querying the view costs
nothing beyond reading the already-computed shards. This also means a single
category's publish only needs to refresh *its own* shard — see Refresh below
— not recompute all 181 tables, which the monolithic design would have
required on every refresh regardless of which one category actually changed.

Sharding by **(L2, env)**, not just L2, is deliberate: we never bisected the
exact ceiling between 12 branches (fine) and 181 (fails), so when PROD is
added to `ENVS_TO_INCLUDE`, it must not grow any *existing* shard's branch
count — it should create 25 new, separately-sized shards instead. That's
exactly what the `_<ENV>` suffix gets you: each shard stays permanently the
same size it is today no matter how many envs exist, and the view just grows
from 25 to 50 trivial `SELECT * FROM shard` branches, which costs nothing
(that's not the kind of complexity that caused the original failure).

(Also worth knowing if you touch the generator: `OBJECT_CONSTRUCT(*)` is only
allowed directly in a `SELECT` clause, not inline as a table-function
argument — it has to be materialized in its own CTE first, then flattened via
that CTE's column. And `UNPIVOT`, the more obvious tool for this column→row
pivot, is explicitly **not supported inside a dynamic table definition** at
all — Snowflake's own compilation error says so outright. Both are documented
inline in `generate_coverage_sql.py`'s `build_table_block()`.)

**Refresh**: each shard's `TARGET_LAG = '7 days'` (a constant in
`generate_coverage_sql.py`, adjust and regenerate for a different cadence) is
a loose staleness backstop only — real freshness comes from
`sql/refresh.py`'s `refresh_specification_coverage_shard(l2_id, env)`, which
`spec_extraction/extract_specs.py`'s `main()` calls automatically, for just
the (L2, env) that was published, as the last step of every `--mode full`
publish of a migrated (v2) category in `--env stage` — a refresh failure
there only prints a warning, it never fails the publish itself. The summary
view needs no refresh of its own; it always reflects whatever its shards
currently hold. To force a shard refresh manually:
```bash
python3 -m sql.refresh <l2_id> <env>
```
or directly in Snowflake:
```sql
ALTER DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_<L2_ID>_<ENV> REFRESH;
```

### Regenerating

Re-run any time an L2/L3 config under `spec_extraction/schemas/spec_model_v2/l2/`
changes — new category, new L3 node, new/renamed/removed field:

```bash
python3 sql/generate_coverage_sql.py
```

This overwrites `specification_coverage_summary.sql` (all 25 shard statements
plus the summary view statement). Diff it before committing — the diff is a
readable summary of what actually changed (a renamed field, a newly-added L3
node, etc.) and will localize to just the affected L2's shard statement.

### Deploying

```bash
python3 sql/deploy.py specification_coverage_summary.sql                # uses COMPUTE_WH (DEFAULT_WAREHOUSE)
python3 sql/deploy.py specification_coverage_summary.sql --warehouse DEV_WH
python3 sql/deploy.py specification_coverage_summary.sql --dry-run       # print resolved statements, don't run them
```

Executes all 26 statements (25 shards + 1 view) in order over one session,
printing `[i/26] ...` progress as it goes.

Requires Snowflake role `DEPT-ENGINEERING` (same role every other write in
this repo uses — see `get_products_session()` in
`classification_pipeline/product_classifier_utils.py`) to have
`CREATE DYNAMIC TABLE` and `CREATE VIEW` privilege on
`SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS`, `SELECT` on all 181 underlying
`*_SPEC_STAGE` tables, and `USAGE` on whichever warehouse is used for shard
refreshes. Confirmed via `SHOW GRANTS TO ROLE "DEPT-ENGINEERING"` that it has
`USAGE` on `COMPUTE_WH`, `DBT_WH`, `DEV_WH`, and `SNOWPARK_XL_WH` — notably
**not** on `SNOWPARK_WH`, even though that name can surface as a stale
`CURRENT_WAREHOUSE()` session default; see `deploy.py`'s module docstring for
the full story of why `DEFAULT_WAREHOUSE` is hardcoded to `COMPUTE_WH` rather
than auto-detected.
