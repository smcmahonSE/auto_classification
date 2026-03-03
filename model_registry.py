import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect model registry records and show best/latest runs."
    )
    parser.add_argument(
        "--registry-path",
        default="artifacts/model/model_registry.jsonl",
        help="Path to model registry JSONL produced by train_model.py",
    )
    parser.add_argument(
        "--sort-by",
        default="scores.macro_f1",
        choices=[
            "scores.macro_f1",
            "scores.weighted_f1",
            "scores.accuracy",
            "scores.balanced_accuracy",
            "registered_at_utc",
        ],
        help="Metric to sort by.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (default: descending).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of rows to print.",
    )
    return parser.parse_args()


def load_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Registry is empty: {path}")
    return pd.json_normalize(records)


def main():
    args = parse_args()
    df = load_registry(Path(args.registry_path))
    df_sorted = df.sort_values(args.sort_by, ascending=args.ascending).reset_index(drop=True)
    df_sorted.insert(0, "rank", range(1, len(df_sorted) + 1))

    display_cols = [
        "rank",
        "run_id",
        "experiment_name",
        "scores.macro_f1",
        "scores.balanced_accuracy",
        "scores.accuracy",
        "scores.weighted_f1",
        "model_artifact_path",
        "metrics_path",
        "registered_at_utc",
    ]
    cols = [c for c in display_cols if c in df_sorted.columns]
    table = df_sorted[cols].head(max(args.top_n, 1))
    print(table.to_string(index=False))

    latest_idx = df["registered_at_utc"].astype(str).sort_values().index[-1]
    latest = df.loc[latest_idx]
    print("\nLatest registered model:")
    print(f"  run_id: {latest.get('run_id')}")
    print(f"  model: {latest.get('model_artifact_path')}")
    print(f"  metrics: {latest.get('metrics_path')}")


if __name__ == "__main__":
    main()
