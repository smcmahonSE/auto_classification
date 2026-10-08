"""Sample and full run modes for the v2 (L1/L2/L3) spec model — the new-system
counterpart to extract_product_specs.py's run_sample/run_full, but driven by
config.loader's hierarchical field sets instead of the old flat EXTRACTORS dict,
and writing to a schema-per-L2/table-per-L3 Snowflake layout instead of one flat
per-category table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from spec_extraction.config.loader import FieldDef, get_effective_field_set, load_l2
from spec_extraction.extract_product_specs import normalize_field_name, quote_sql
from spec_extraction.extractors.common import ExtractedSpec
from spec_extraction.session import close_spec_session, get_spec_session
from spec_extraction.v2.extractor_dispatch import extract_row

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "spec_extraction/outputs_v2"
DEFAULT_SAMPLE_SIZE = 20_000
DEFAULT_REVIEW_SIZE = 200

ENV_INPUT_TABLES = {
    "stage": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_CLASSIFICATIONS_STAGE",
    "prod": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_CLASSIFICATIONS_PROD",
}
ENV_SUFFIX = {"stage": "STAGE", "prod": "PROD"}

BASE_COLUMNS = [
    "PRODUCT_ID",
    "PRODUCT_VARIANT_ID",
    "PRODUCT_VARIANT_NAME",
    "CATEGORY_NAME",
    "PRODUCT_SOURCE_C",
    "PRODUCT_NAME",
    "DESCRIPTION",
    "SPECIFICATION_ASSIGNMENTS_C",
    "PRICING_STATUS_C",
    "LIST_PRICE_C",
    "ASSIGNED_L2_ID",
    "ASSIGNED_L2_LABEL",
    "ASSIGNED_L3_ID",
    "ASSIGNED_L3_LABEL",
]
DIAGNOSTIC_SUFFIXES = ["value", "status", "method", "evidence", "source_field", "confidence"]


def build_l3_query(env: str, l3_id: str, mode: str, sample_size: int) -> str:
    table = ENV_INPUT_TABLES[env]
    sample_clause = "ORDER BY RANDOM()" if mode == "sample" else ""
    limit_clause = f"\nLIMIT {int(sample_size)}" if mode == "sample" and sample_size > 0 else ""
    return f"""
        SELECT *
        FROM {table}
        WHERE ASSIGNED_L3_ID = '{quote_sql(l3_id)}'
        {sample_clause}
        {limit_clause}
    """


def listing_type_guess(row: Mapping[str, object]) -> str:
    pricing_status = row.get("PRICING_STATUS_C")
    if pricing_status is None or pd.isna(pricing_status):
        return ""
    return "Product" if str(pricing_status).strip().upper() == "PRICED" else "Service"


def get_detail_columns(fields: list[FieldDef]) -> list[str]:
    # Diagnostic columns are always suffixed (_value, _status, ...), so they can't
    # collide with BASE_COLUMNS — no dedup needed here, unlike get_full_output_columns.
    columns = list(BASE_COLUMNS) + ["LISTING_TYPE_GUESS"]
    for f in fields:
        prefix = normalize_field_name(f.name)
        columns.extend(f"{prefix}_{suffix}" for suffix in DIAGNOSTIC_SUFFIXES)
    return columns


def _dedup_preserve_order(columns: list[str]) -> list[str]:
    """A configured field's normalized name can collide with a raw BASE_COLUMNS
    name (e.g. L1 field "Description" -> DESCRIPTION, same as the raw source
    column) — keep the first occurrence only. In that specific case it's the same
    underlying data anyway (Description is a passthrough of DESCRIPTION), so
    nothing is lost; for any other future collision, the base/raw column wins."""
    seen: set[str] = set()
    result = []
    for c in columns:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def flatten_specs(specs: list[ExtractedSpec]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for spec in specs:
        prefix = normalize_field_name(spec.field_name)
        flat[f"{prefix}_value"] = spec.value
        flat[f"{prefix}_status"] = spec.status
        flat[f"{prefix}_method"] = spec.method
        flat[f"{prefix}_evidence"] = spec.evidence
        flat[f"{prefix}_source_field"] = spec.source_field
        flat[f"{prefix}_confidence"] = spec.confidence
    return flat


def flatten_values(specs: list[ExtractedSpec]) -> dict[str, object]:
    values: dict[str, object] = {}
    for spec in specs:
        column = normalize_field_name(spec.field_name).upper()
        values[column] = spec.value if spec.status == "matched" else None
    return values


def get_full_output_columns(fields: list[FieldDef]) -> list[str]:
    value_columns = [normalize_field_name(f.name).upper() for f in fields]
    return _dedup_preserve_order([*BASE_COLUMNS, "LISTING_TYPE_GUESS", *value_columns])


def update_coverage(
    coverage: dict[tuple[str, str], dict[str, int]],
    l3_id: str,
    specs: list[ExtractedSpec],
) -> None:
    for spec in specs:
        key = (l3_id, spec.field_name)
        counts = coverage.setdefault(
            key, {"total_rows": 0, "matched": 0, "missing": 0, "ambiguous": 0, "invalid": 0}
        )
        counts["total_rows"] += 1
        counts[spec.status] = counts.get(spec.status, 0) + 1


def build_coverage(coverage: dict[tuple[str, str], dict[str, int]]) -> pd.DataFrame:
    summaries = []
    for (l3_id, field_name), counts in sorted(coverage.items()):
        total = counts["total_rows"]
        matched = counts.get("matched", 0)
        summaries.append({
            "l3_id": l3_id,
            "field": field_name,
            "total_rows": total,
            "matched_rows": matched,
            "matched_pct": round((matched / total * 100), 2) if total else 0.0,
            "missing_rows": counts.get("missing", 0),
            "ambiguous_rows": counts.get("ambiguous", 0),
            "invalid_rows": counts.get("invalid", 0),
        })
    return pd.DataFrame(summaries)


def build_manual_review_sample(rows_df: pd.DataFrame, l2_id: str, l3_id: str, fields: list[FieldDef], review_size: int) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df

    samples = []
    per_field = max(1, review_size // max(1, len(fields)))
    for f in fields:
        prefix = normalize_field_name(f.name)
        status_col = f"{prefix}_status"
        if status_col not in rows_df.columns:
            continue
        subset = rows_df[rows_df[status_col].eq("matched")].copy()
        if subset.empty:
            continue
        sample = subset.sample(n=min(per_field, len(subset)), random_state=42).copy()
        sample.insert(0, "REVIEW_BUCKET", f"{l2_id}_{l3_id}_{prefix}_matched")
        samples.append(sample)

    if not samples:
        review = rows_df.sample(n=min(review_size, len(rows_df)), random_state=42).copy()
        review.insert(0, "REVIEW_BUCKET", f"{l2_id}_{l3_id}_random")
    else:
        review = pd.concat(samples, ignore_index=True)

    for column in ["IS_CORRECT", "CORRECTION", "NOTES"]:
        review[column] = ""

    review_columns = ["REVIEW_BUCKET", *get_detail_columns(fields), "IS_CORRECT", "CORRECTION", "NOTES"]
    return review.reindex(columns=review_columns)


def extract_chunk(df: pd.DataFrame, l2_id: str, l3_id: str, fields: list[FieldDef],
                   coverage: dict[tuple[str, str], dict[str, int]], full_mode: bool) -> list[dict[str, object]]:
    output_rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        specs = extract_row(l2_id, l3_id, fields, row_dict)
        update_coverage(coverage, l3_id, specs)
        base_values = {column: row_dict.get(column) for column in BASE_COLUMNS}
        base_values["LISTING_TYPE_GUESS"] = listing_type_guess(row_dict)
        if full_mode:
            output_rows.append({**base_values, **flatten_values(specs)})
        else:
            output_rows.append({**base_values, **flatten_specs(specs)})
    return output_rows


def run_sample_for_node(env: str, l2_id: str, l3_id: str, output_dir: Path, sample_size: int, review_size: int) -> dict[str, Path]:
    fields = get_effective_field_set(l2_id, l3_id)
    query = build_l3_query(env, l3_id, "sample", sample_size)
    print(f"Loading {l3_id} (under {l2_id}) from {ENV_INPUT_TABLES[env]} (sample)...")
    sf = get_spec_session()
    try:
        df = sf.sql(query).to_pandas()
    finally:
        close_spec_session(sf)
    print(f"Loaded {len(df):,} classified rows.")

    coverage: dict[tuple[str, str], dict[str, int]] = {}
    rows = extract_chunk(df, l2_id, l3_id, fields, coverage, full_mode=False)

    node_dir = output_dir / env / "sample" / l2_id / l3_id
    node_dir.mkdir(parents=True, exist_ok=True)
    detail_path = node_dir / "spec_extraction_pilot_details.csv"
    coverage_path = node_dir / "spec_extraction_pilot_coverage.csv"
    review_path = node_dir / "manual_review_sample.csv"

    detail_columns = get_detail_columns(fields)
    rows_df = pd.DataFrame(rows).reindex(columns=detail_columns)
    rows_df.to_csv(detail_path, index=False)
    build_coverage(coverage).to_csv(coverage_path, index=False)
    build_manual_review_sample(rows_df, l2_id, l3_id, fields, review_size).to_csv(review_path, index=False)

    print(f"  Coverage for {l3_id}:")
    cov_df = build_coverage(coverage)
    if not cov_df.empty:
        print(cov_df.to_string(index=False))

    return {"detail": detail_path, "coverage": coverage_path, "review": review_path}


def run_sample_v2(env: str, l2_id: str, l3_id: str | None, output_dir: Path, sample_size: int, review_size: int) -> dict[str, dict[str, Path]]:
    l2 = load_l2(l2_id)
    targets = [l3_id] if l3_id else list(l2.l3_nodes)
    results = {}
    for node_id in targets:
        results[node_id] = run_sample_for_node(env, l2_id, node_id, output_dir, sample_size, review_size)
    return results


def run_full_for_node(sf, env: str, l2_id: str, l3_id: str) -> tuple[str, int]:
    l2 = load_l2(l2_id)
    node = l2.l3_nodes[l3_id]
    fields = get_effective_field_set(l2_id, l3_id)
    table = f'"{l2.schema_name}".{node.table_name}_{ENV_SUFFIX[env]}'
    query = build_l3_query(env, l3_id, "full", 0)

    print(f"Loading {l3_id} (under {l2_id}) from {ENV_INPUT_TABLES[env]} (full)...")
    coverage: dict[tuple[str, str], dict[str, int]] = {}
    total_written = 0
    write_mode = "overwrite"
    columns = get_full_output_columns(fields)

    for batch_number, batch in enumerate(sf.sql(query).to_pandas_batches(), start=1):
        rows = extract_chunk(batch, l2_id, l3_id, fields, coverage, full_mode=True)
        if rows:
            output_df = pd.DataFrame(rows).reindex(columns=columns)
            output_df = output_df.astype("string").where(pd.notna(output_df), None)
            sf.create_dataframe(output_df).write.mode(write_mode).save_as_table(table)
            total_written += len(output_df)
            write_mode = "append"
        print(f"  {l3_id} batch {batch_number}: wrote {len(rows):,} rows")

    cov_df = build_coverage(coverage)
    if not cov_df.empty:
        print(f"  Coverage for {l3_id}:")
        print(cov_df.to_string(index=False))
    print(f"Done. Wrote {total_written:,} rows to {table}.")
    return table, total_written


def run_full_v2(env: str, l2_id: str, l3_id: str | None) -> list[tuple[str, int]]:
    l2 = load_l2(l2_id)
    targets = [l3_id] if l3_id else list(l2.l3_nodes)
    written: list[tuple[str, int]] = []
    sf = get_spec_session(schema=l2.schema_name, create_schema_if_missing=True)
    try:
        for node_id in targets:
            written.append(run_full_for_node(sf, env, l2_id, node_id))
    finally:
        close_spec_session(sf)
    return written
