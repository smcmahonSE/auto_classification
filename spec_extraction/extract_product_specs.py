"""Run deterministic specification extraction for pilot L3 categories."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spec_extraction.extractors import extract_antibody_specs, extract_chemical_specs
from spec_extraction.extractors.common import ExtractedSpec


DEFAULT_INPUT = PROJECT_ROOT / "artifacts/analysis/master_classification_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "spec_extraction/outputs"

L3_ID_COLUMNS = ("ASSIGNED_NEW_L3_ID", "ASSIGNED_L4_ID")
L3_LABEL_COLUMNS = ("ASSIGNED_NEW_L3_LABEL", "ASSIGNED_L4_LABEL")

EXTRACTORS: dict[str, Callable[[Mapping[str, object]], list[ExtractedSpec]]] = {
    "chemicals_solvents": extract_chemical_specs,
    "antibodies": extract_antibody_specs,
}

LABEL_TO_ID = {
    "Chemicals & Solvents": "chemicals_solvents",
    "Antibodies": "antibodies",
}


def first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def normalize_field_name(field_name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", field_name).strip("_").lower()
    return value


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


def build_coverage(spec_rows: pd.DataFrame) -> pd.DataFrame:
    status_cols = [col for col in spec_rows.columns if col.endswith("_status")]
    summaries = []
    for col in status_cols:
        field_name = col.removesuffix("_status")
        counts = spec_rows[col].value_counts(dropna=False).to_dict()
        total = len(spec_rows)
        matched = counts.get("matched", 0)
        summaries.append(
            {
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


def run_extraction(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    df = pd.read_csv(input_path, dtype=str, low_memory=False)
    l3_id_col = first_existing_column(df, L3_ID_COLUMNS)
    l3_label_col = first_existing_column(df, L3_LABEL_COLUMNS)

    if not l3_id_col and not l3_label_col:
        raise ValueError("Input must contain an assigned L3 id or label column.")

    output_rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        l3_id = get_l3_id(row_dict, l3_id_col, l3_label_col)
        extractor = EXTRACTORS.get(l3_id or "")
        if extractor is None:
            continue

        specs = extractor(row_dict)
        output_rows.append(
            {
                "PRODUCT_ID": row_dict.get("PRODUCT_ID"),
                "SOURCE": row_dict.get("SOURCE"),
                "ASSIGNED_NEW_L3_ID": l3_id,
                "ASSIGNED_NEW_L3_LABEL": row_dict.get(l3_label_col) if l3_label_col else None,
                **flatten_specs(specs),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    spec_rows = pd.DataFrame(output_rows)
    detail_path = output_dir / "spec_extraction_pilot_details.csv"
    coverage_path = output_dir / "spec_extraction_pilot_coverage.csv"

    spec_rows.to_csv(detail_path, index=False)
    build_coverage(spec_rows).to_csv(coverage_path, index=False)
    return detail_path, coverage_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail_path, coverage_path = run_extraction(args.input, args.output_dir)
    print(f"Wrote detail output: {detail_path}")
    print(f"Wrote coverage output: {coverage_path}")


if __name__ == "__main__":
    main()
