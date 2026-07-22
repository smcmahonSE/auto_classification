"""Run deterministic specification extraction for classified products."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from classification_pipeline.product_classifier_utils import get_snowflake_session
from spec_extraction.extractors import (
    extract_antibody_specs,
    extract_chemical_specs,
    extract_lab_supplies_specs,
    extract_molecular_biology_specs,
)
from spec_extraction.extractors.common import ExtractedSpec


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "spec_extraction/outputs"
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_REVIEW_SIZE = 200
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)

BASE_TABLE = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS"
SNOWFLAKE_DATABASE = "SNOWFLAKE_LEARNING_DB"
SNOWFLAKE_SCHEMA = "SMCMAHON_PRODUCTS"

ENV_CONFIGS = {
    "stage": {
        "input_table": f"{BASE_TABLE}.NEW_CLASSIFICATIONS_STAGE",
        "output_suffix": "STAGE",
    },
    "prod": {
        "input_table": f"{BASE_TABLE}.NEW_CLASSIFICATIONS_PROD",
        "output_suffix": "PROD",
    },
}

PRODUCT_COLUMNS = [
    "PRODUCT_ID",
    "PRODUCT_VARIANT_ID",
    "SOURCE",
    "PRODUCT_NAME",
    "DESCRIPTION",
    "ASSIGNED_NEW_L3_ID",
    "ASSIGNED_NEW_L3_LABEL",
    "ASSIGNED_L4_ID",
    "ASSIGNED_L4_LABEL",
]

OPTIONAL_SOURCE_COLUMNS = [
    "PRICING_STATUS_C",
    "LIST_PRICE_C",
]

DIAGNOSTIC_SUFFIXES = ["value", "status", "method", "evidence", "source_field", "confidence"]

EXTRACTORS: dict[str, Callable[[Mapping[str, object]], list[ExtractedSpec]]] = {
    "chemicals_solvents": extract_chemical_specs,
    "molecular_biology_reagents": extract_molecular_biology_specs,
    "lab_supplies_consumables": extract_lab_supplies_specs,
    "antibodies": extract_antibody_specs,
}

LABEL_TO_ID = {
    "Chemicals and Solvents": "chemicals_solvents",
    "Chemicals & Solvents": "chemicals_solvents",
    "Molecular Biology Reagents": "molecular_biology_reagents",
    "Lab Supplies and Consumables": "lab_supplies_consumables",
    "Lab Supplies & Consumables": "lab_supplies_consumables",
    "Antibodies": "antibodies",
}

CATEGORY_LABELS = {
    "chemicals_solvents": "Chemicals and Solvents",
    "molecular_biology_reagents": "Molecular Biology Reagents",
    "lab_supplies_consumables": "Lab Supplies and Consumables",
    "antibodies": "Antibodies",
}

CATEGORY_FIELDS = {
    "chemicals_solvents": ["CAS Number", "Purity"],
    "molecular_biology_reagents": ["Sub-Type", "Target Gene / Region", "Target Species"],
    "lab_supplies_consumables": [
        "Material",
        "Sterility",
        "Capacity Volume Size",
        "Size Amount",
        "Size Unit",
        "Pack Size",
        "Color",
    ],
    "antibodies": [
        "Target / Specificity",
        "Host Species",
        "Clonality",
        "Reactivity",
        "Application",
    ],
}

BASE_DETAIL_COLUMNS = [
    "PRODUCT_ID",
    "PRODUCT_VARIANT_ID",
    "SOURCE",
    "PRODUCT_NAME",
    "DESCRIPTION",
    "PRICING_STATUS_C",
    "LIST_PRICE_C",
    "ASSIGNED_NEW_L3_ID",
    "ASSIGNED_NEW_L3_LABEL",
    "ASSIGNED_L4_ID",
    "ASSIGNED_L4_LABEL",
]


def first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def normalize_field_name(field_name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", field_name).strip("_").lower()
    return value


def get_detail_columns(category: str) -> list[str]:
    columns = list(BASE_DETAIL_COLUMNS)
    for field in CATEGORY_FIELDS[category]:
        prefix = normalize_field_name(field)
        columns.extend(f"{prefix}_{suffix}" for suffix in DIAGNOSTIC_SUFFIXES)
    return columns


def get_l3_id(row: Mapping[str, object], l3_id_col: str | None, l3_label_col: str | None) -> str | None:
    if l3_id_col:
        value = row.get(l3_id_col)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()

    if l3_label_col:
        label = row.get(l3_label_col)
        if pd.notna(label):
            return LABEL_TO_ID.get(str(label).strip())

    return None


def quote_sql(value: str) -> str:
    return value.replace("'", "''")


def get_output_table(category: str, env: str) -> str:
    return f"{BASE_TABLE}.{category.upper()}_SPEC_{ENV_CONFIGS[env]['output_suffix']}"


def set_snowflake_context(sf) -> None:
    """Set database/schema so Snowpark can create temp stages for pandas writes."""
    sf.sql(f"USE DATABASE {SNOWFLAKE_DATABASE}").collect()
    sf.sql(f"USE SCHEMA {SNOWFLAKE_SCHEMA}").collect()


def close_snowflake_session(sf) -> None:
    """Best-effort close so Snowflake auth/HTTP resources do not keep Python alive."""
    try:
        sf.close()
    except Exception as exc:  # pragma: no cover - cleanup should not mask run success.
        print(f"Warning: failed to close Snowflake session cleanly: {exc}")


def get_category_output_dir(output_dir: Path, env: str, mode: str, category: str) -> Path:
    return output_dir / env / mode / category


def get_description(row: Mapping[str, object]) -> str:
    value = row.get("DESCRIPTION")
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def get_unextractable_reason(row: Mapping[str, object]) -> str | None:
    """Return why a row lacks local extractable detail, or None if usable."""
    description = get_description(row)
    normalized = description.strip()

    if not normalized:
        return "blank_description"

    if normalized.rstrip(":").strip().upper() == "INSERT":
        return "insert_placeholder"

    if URL_PATTERN.search(normalized):
        return "url_description"

    return None


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
    """Return final value columns only for full-mode Snowflake output."""
    values = {}
    for spec in specs:
        column = normalize_field_name(spec.field_name).upper()
        values[column] = spec.value if spec.status == "matched" else None
    return values


def update_coverage(
    coverage: dict[tuple[str, str], dict[str, int]],
    l3_id: str,
    specs: list[ExtractedSpec],
) -> None:
    for spec in specs:
        key = (l3_id, spec.field_name)
        counts = coverage.setdefault(
            key,
            {"total_rows": 0, "matched": 0, "missing": 0, "ambiguous": 0, "invalid": 0},
        )
        counts["total_rows"] += 1
        counts[spec.status] = counts.get(spec.status, 0) + 1


def build_coverage(coverage: dict[tuple[str, str], dict[str, int]]) -> pd.DataFrame:
    summaries = []
    for (l3_id, field_name), counts in sorted(coverage.items()):
        total = counts["total_rows"]
        matched = counts.get("matched", 0)
        summaries.append(
            {
                "l3_id": l3_id,
                "field": field_name,
                "total_rows": total,
                "matched_rows": matched,
                "matched_pct": round((matched / total * 100), 2) if total else 0.0,
                "missing_rows": counts.get("missing", 0),
                "ambiguous_rows": counts.get("ambiguous", 0),
                "invalid_rows": counts.get("invalid", 0),
            }
        )
    return pd.DataFrame(summaries)


def print_report(coverage: dict[tuple[str, str], dict[str, int]], filter_counts: dict[str, int]) -> None:
    print("\n=== Filter Summary ===")
    for metric, count in sorted(filter_counts.items()):
        print(f"{metric}: {count:,}")

    print("\n=== Extraction Coverage ===")
    coverage_df = build_coverage(coverage)
    if coverage_df.empty:
        print("No extractable rows found.")
    else:
        print(coverage_df.to_string(index=False))


def extract_chunk(
    df: pd.DataFrame,
    category: str,
    coverage: dict[tuple[str, str], dict[str, int]],
    filter_counts: dict[str, int],
    apply_filters: bool,
    full_mode: bool = False,
) -> list[dict[str, object]]:
    output_rows = []
    extractor = EXTRACTORS[category]
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        filter_counts["pilot_rows_seen"] = filter_counts.get("pilot_rows_seen", 0) + 1
        if apply_filters:
            reason = get_unextractable_reason(row_dict)
            if reason is not None:
                filter_counts[f"excluded_{reason}"] = filter_counts.get(f"excluded_{reason}", 0) + 1
                continue

        filter_counts["extractable_rows"] = filter_counts.get("extractable_rows", 0) + 1
        specs = extractor(row_dict)
        update_coverage(coverage, category, specs)
        product_values = {column: row_dict.get(column) for column in PRODUCT_COLUMNS + OPTIONAL_SOURCE_COLUMNS}
        if full_mode:
            output_rows.append({**product_values, **flatten_values(specs)})
        else:
            output_rows.append({**product_values, **flatten_specs(specs)})
    return output_rows


def build_category_query(env: str, category: str, mode: str, sample_size: int) -> str:
    table = ENV_CONFIGS[env]["input_table"]
    sample_clause = "ORDER BY RANDOM()" if mode == "sample" else ""
    limit_clause = f"\nLIMIT {int(sample_size)}" if mode == "sample" and sample_size > 0 else ""
    return f"""
        SELECT *
        FROM {table}
        WHERE ASSIGNED_NEW_L3_ID = '{quote_sql(category)}'
        {sample_clause}
        {limit_clause}
    """


def load_category_data(env: str, category: str, mode: str, sample_size: int) -> pd.DataFrame:
    table = ENV_CONFIGS[env]["input_table"]
    query = build_category_query(env=env, category=category, mode=mode, sample_size=sample_size)

    print(f"Loading {category} from {table} ({mode})...")
    sf = get_snowflake_session()
    try:
        df = sf.sql(query).to_pandas()
        print(f"Loaded {len(df):,} classified rows.")
        return df
    finally:
        close_snowflake_session(sf)


def write_sample_outputs(
    output_dir: Path,
    rows: Iterable[dict[str, object]],
    coverage: dict[tuple[str, str], dict[str, int]],
    filter_counts: dict[str, int],
    category: str,
    review_size: int,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "spec_extraction_pilot_details.csv"
    coverage_path = output_dir / "spec_extraction_pilot_coverage.csv"
    filter_summary_path = output_dir / "spec_extraction_filter_summary.csv"
    review_path = output_dir / "manual_review_sample.csv"

    for path in (detail_path, coverage_path, filter_summary_path, review_path):
        path.unlink(missing_ok=True)

    detail_columns = get_detail_columns(category)
    rows_df = pd.DataFrame(rows).reindex(columns=detail_columns)
    rows_df.to_csv(detail_path, index=False)
    build_coverage(coverage).to_csv(coverage_path, index=False)
    pd.DataFrame(
        [{"metric": metric, "count": count} for metric, count in sorted(filter_counts.items())]
    ).to_csv(filter_summary_path, index=False)
    build_manual_review_sample(rows_df, category, review_size).to_csv(review_path, index=False)
    return detail_path, coverage_path, filter_summary_path, review_path


def build_manual_review_sample(rows_df: pd.DataFrame, category: str, review_size: int) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df

    samples = []
    fields = CATEGORY_FIELDS[category]
    per_field = max(1, review_size // max(1, len(fields)))
    for field_name in fields:
        prefix = normalize_field_name(field_name)
        status_col = f"{prefix}_status"
        if status_col not in rows_df.columns:
            continue

        subset = rows_df[rows_df[status_col].eq("matched")].copy()
        if subset.empty:
            continue

        sample = subset.sample(n=min(per_field, len(subset)), random_state=42).copy()
        sample.insert(0, "REVIEW_BUCKET", f"{category}_{prefix}_matched")
        samples.append(sample)

    if not samples:
        review = rows_df.sample(n=min(review_size, len(rows_df)), random_state=42).copy()
        review.insert(0, "REVIEW_BUCKET", f"{category}_random")
    else:
        review = pd.concat(samples, ignore_index=True)

    for column in ["IS_CORRECT", "CORRECTION", "NOTES"]:
        review[column] = ""

    review_columns = ["REVIEW_BUCKET", *get_detail_columns(category), "IS_CORRECT", "CORRECTION", "NOTES"]
    return review.reindex(columns=review_columns)


def get_full_output_columns(category: str) -> list[str]:
    value_columns = [normalize_field_name(field).upper() for field in CATEGORY_FIELDS[category]]
    return [*PRODUCT_COLUMNS, *OPTIONAL_SOURCE_COLUMNS, *value_columns]


def write_full_batch(sf, rows: list[dict[str, object]], category: str, table: str, mode: str) -> int:
    if not rows:
        return 0

    columns = get_full_output_columns(category)
    output_df = pd.DataFrame(rows).reindex(columns=columns)
    output_df = output_df.astype("string").where(pd.notna(output_df), None)
    sf.create_dataframe(output_df).write.mode(mode).save_as_table(table)
    return len(output_df)


def run_sample(
    env: str,
    category: str,
    output_dir: Path,
    sample_size: int,
    review_size: int,
) -> tuple[Path, Path, Path, Path]:
    df = load_category_data(env=env, category=category, mode="sample", sample_size=sample_size)
    coverage: dict[tuple[str, str], dict[str, int]] = {}
    filter_counts: dict[str, int] = {}
    rows = extract_chunk(df, category, coverage, filter_counts, apply_filters=True)
    print_report(coverage, filter_counts)
    return write_sample_outputs(
        output_dir=get_category_output_dir(output_dir, env, "sample", category),
        rows=rows,
        coverage=coverage,
        filter_counts=filter_counts,
        category=category,
        review_size=review_size,
    )


def run_full(env: str, category: str) -> str:
    table = get_output_table(category, env)
    input_table = ENV_CONFIGS[env]["input_table"]
    query = build_category_query(env=env, category=category, mode="full", sample_size=0)
    print(f"Loading {category} from {input_table} (full)...")
    sf = get_snowflake_session()
    try:
        set_snowflake_context(sf)
        coverage: dict[tuple[str, str], dict[str, int]] = {}
        filter_counts: dict[str, int] = {}
        total_written = 0
        write_mode = "overwrite"

        for batch_number, batch in enumerate(sf.sql(query).to_pandas_batches(), start=1):
            rows = extract_chunk(batch, category, coverage, filter_counts, apply_filters=False, full_mode=True)
            written = write_full_batch(sf, rows, category=category, table=table, mode=write_mode)
            total_written += written
            if written > 0:
                write_mode = "append"
            print(f"Batch {batch_number}: wrote {written:,} rows")

        print_report(coverage, filter_counts)
        print(f"Done. Wrote {total_written:,} rows to {table}.")
        return table
    finally:
        close_snowflake_session(sf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(ENV_CONFIGS), default="stage")
    parser.add_argument("--mode", choices=("sample", "full"), default="sample")
    parser.add_argument("--category", choices=sorted(EXTRACTORS), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--review-size", type=int, default=DEFAULT_REVIEW_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "sample":
        detail_path, coverage_path, filter_summary_path, review_path = run_sample(
            env=args.env,
            category=args.category,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            review_size=args.review_size,
        )
        print(f"Wrote detail output: {detail_path}")
        print(f"Wrote coverage output: {coverage_path}")
        print(f"Wrote filter summary: {filter_summary_path}")
        print(f"Wrote manual review sample: {review_path}")
    else:
        output_table = run_full(env=args.env, category=args.category)
        print(f"Wrote full output table: {output_table}")


if __name__ == "__main__":
    main()
