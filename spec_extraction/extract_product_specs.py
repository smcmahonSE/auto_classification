"""Run deterministic specification extraction for pilot L3 categories."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spec_extraction.extractors import (
    extract_antibody_specs,
    extract_chemical_specs,
    extract_lab_supplies_specs,
    extract_molecular_biology_specs,
)
from spec_extraction.extractors.common import ExtractedSpec


DEFAULT_INPUT = PROJECT_ROOT / "artifacts/analysis/master_classification_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "spec_extraction/outputs"
DEFAULT_CHUNK_SIZE = 100_000
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)

L3_ID_COLUMNS = ("ASSIGNED_NEW_L3_ID", "ASSIGNED_L4_ID")
L3_LABEL_COLUMNS = ("ASSIGNED_NEW_L3_LABEL", "ASSIGNED_L4_LABEL")

EXTRACTORS: dict[str, Callable[[Mapping[str, object]], list[ExtractedSpec]]] = {
    "chemicals_solvents": extract_chemical_specs,
    "molecular_biology_reagents": extract_molecular_biology_specs,
    "lab_supplies_consumables": extract_lab_supplies_specs,
    "antibodies": extract_antibody_specs,
}

LABEL_TO_ID = {
    "Chemicals & Solvents": "chemicals_solvents",
    "Molecular Biology Reagents": "molecular_biology_reagents",
    "Lab Supplies & Consumables": "lab_supplies_consumables",
    "Antibodies": "antibodies",
}

DETAIL_COLUMNS = [
    "PRODUCT_ID",
    "SOURCE",
    "PRODUCT_NAME",
    "DESCRIPTION",
    "ASSIGNED_NEW_L3_ID",
    "ASSIGNED_NEW_L3_LABEL",
    "cas_number_value",
    "cas_number_status",
    "cas_number_method",
    "cas_number_evidence",
    "cas_number_source_field",
    "cas_number_confidence",
    "purity_value",
    "purity_status",
    "purity_method",
    "purity_evidence",
    "purity_source_field",
    "purity_confidence",
    "sub_type_value",
    "sub_type_status",
    "sub_type_method",
    "sub_type_evidence",
    "sub_type_source_field",
    "sub_type_confidence",
    "target_gene_region_value",
    "target_gene_region_status",
    "target_gene_region_method",
    "target_gene_region_evidence",
    "target_gene_region_source_field",
    "target_gene_region_confidence",
    "material_value",
    "material_status",
    "material_method",
    "material_evidence",
    "material_source_field",
    "material_confidence",
    "sterility_value",
    "sterility_status",
    "sterility_method",
    "sterility_evidence",
    "sterility_source_field",
    "sterility_confidence",
    "capacity_volume_size_value",
    "capacity_volume_size_status",
    "capacity_volume_size_method",
    "capacity_volume_size_evidence",
    "capacity_volume_size_source_field",
    "capacity_volume_size_confidence",
    "pack_size_value",
    "pack_size_status",
    "pack_size_method",
    "pack_size_evidence",
    "pack_size_source_field",
    "pack_size_confidence",
    "color_value",
    "color_status",
    "color_method",
    "color_evidence",
    "color_source_field",
    "color_confidence",
    "target_specificity_value",
    "target_specificity_status",
    "target_specificity_method",
    "target_specificity_evidence",
    "target_specificity_source_field",
    "target_specificity_confidence",
    "host_species_value",
    "host_species_status",
    "host_species_method",
    "host_species_evidence",
    "host_species_source_field",
    "host_species_confidence",
    "clonality_value",
    "clonality_status",
    "clonality_method",
    "clonality_evidence",
    "clonality_source_field",
    "clonality_confidence",
    "reactivity_value",
    "reactivity_status",
    "reactivity_method",
    "reactivity_evidence",
    "reactivity_source_field",
    "reactivity_confidence",
    "application_value",
    "application_status",
    "application_method",
    "application_evidence",
    "application_source_field",
    "application_confidence",
]


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


def extract_chunk(
    df: pd.DataFrame,
    l3_id_col: str | None,
    l3_label_col: str | None,
    coverage: dict[tuple[str, str], dict[str, int]],
    filter_counts: dict[str, int],
) -> list[dict[str, object]]:
    output_rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        l3_id = get_l3_id(row_dict, l3_id_col, l3_label_col)
        extractor = EXTRACTORS.get(l3_id or "")
        if extractor is None or l3_id is None:
            continue

        filter_counts["pilot_rows_seen"] = filter_counts.get("pilot_rows_seen", 0) + 1
        reason = get_unextractable_reason(row_dict)
        if reason is not None:
            filter_counts[f"excluded_{reason}"] = filter_counts.get(f"excluded_{reason}", 0) + 1
            continue

        filter_counts["extractable_rows"] = filter_counts.get("extractable_rows", 0) + 1
        specs = extractor(row_dict)
        update_coverage(coverage, l3_id, specs)
        output_rows.append(
            {
                "PRODUCT_ID": row_dict.get("PRODUCT_ID"),
                "SOURCE": row_dict.get("SOURCE"),
                "PRODUCT_NAME": row_dict.get("PRODUCT_NAME"),
                "DESCRIPTION": row_dict.get("DESCRIPTION"),
                "ASSIGNED_NEW_L3_ID": l3_id,
                "ASSIGNED_NEW_L3_LABEL": row_dict.get(l3_label_col) if l3_label_col else None,
                **flatten_specs(specs),
            }
        )
    return output_rows


def run_extraction(
    input_path: Path,
    output_dir: Path,
    chunk_size: int,
    max_chunks: int | None = None,
) -> tuple[Path, Path]:
    header = pd.read_csv(input_path, nrows=0)
    l3_id_col = first_existing_column(header, L3_ID_COLUMNS)
    l3_label_col = first_existing_column(header, L3_LABEL_COLUMNS)

    if not l3_id_col and not l3_label_col:
        raise ValueError("Input must contain an assigned L3 id or label column.")

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "spec_extraction_pilot_details.csv"
    coverage_path = output_dir / "spec_extraction_pilot_coverage.csv"
    filter_summary_path = output_dir / "spec_extraction_filter_summary.csv"
    detail_path.unlink(missing_ok=True)
    coverage_path.unlink(missing_ok=True)
    filter_summary_path.unlink(missing_ok=True)

    coverage: dict[tuple[str, str], dict[str, int]] = {}
    filter_counts: dict[str, int] = {}
    chunks = pd.read_csv(input_path, dtype=str, low_memory=False, chunksize=chunk_size)
    with detail_path.open("w", newline="") as detail_file:
        writer = csv.DictWriter(detail_file, fieldnames=DETAIL_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for chunk_number, chunk in enumerate(chunks, start=1):
            if max_chunks is not None and chunk_number > max_chunks:
                print(f"Stopping after {max_chunks} chunk(s).")
                break

            output_rows = extract_chunk(chunk, l3_id_col, l3_label_col, coverage, filter_counts)
            if not output_rows:
                print(f"Chunk {chunk_number}: no pilot rows")
                continue

            writer.writerows(output_rows)
            print(f"Chunk {chunk_number}: wrote {len(output_rows):,} pilot rows")

    build_coverage(coverage).to_csv(coverage_path, index=False)
    pd.DataFrame(
        [{"metric": metric, "count": count} for metric, count in sorted(filter_counts.items())]
    ).to_csv(filter_summary_path, index=False)
    return detail_path, coverage_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-chunks", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail_path, coverage_path = run_extraction(
        args.input,
        args.output_dir,
        args.chunk_size,
        args.max_chunks,
    )
    print(f"Wrote detail output: {detail_path}")
    print(f"Wrote coverage output: {coverage_path}")


if __name__ == "__main__":
    main()
