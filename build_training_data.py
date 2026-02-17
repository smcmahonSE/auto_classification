import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from product_classifier_utils import (
    build_product_text,
    embed_texts_from_cache,
    ensure_parent_dir,
    get_bedrock_client,
    get_snowflake_session,
    load_product_data,
    stable_text_hash,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build cached training features (Titan embeddings + labels)."
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Snowflake table (DATABASE.SCHEMA.TABLE). Defaults to SNOWFLAKE_PRODUCTS_TABLE.",
    )
    parser.add_argument("--min-category-count", type=int, default=100)
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Optional row limit for small test runs.",
    )
    parser.add_argument(
        "--sample-per-category",
        type=int,
        default=None,
        help="Optional cap per label for quick dry-run/testing.",
    )
    parser.add_argument(
        "--stratified-sample-size",
        type=int,
        default=None,
        help="Optional approximate sample size drawn with label stratification.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for stratified sampling.",
    )
    parser.add_argument("--id-column", default="PRODUCT_ID")
    parser.add_argument("--label-column", default="PARENT_3_CATEGORY")
    parser.add_argument("--aws-profile", default="staging.admin")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--model-id", default="amazon.titan-embed-text-v1")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel embedding workers (start with 8-16 for larger runs).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Persist embedding cache every N new embeddings (0 disables periodic checkpointing).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries per embedding request for transient Bedrock failures.",
    )
    parser.add_argument(
        "--cache-path",
        default="artifacts/cache/embedding_cache.pkl",
        help="Pickle file mapping text_hash -> embedding vector.",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/training_data/features_titan.npz",
        help="Compressed feature artifact.",
    )
    parser.add_argument(
        "--metadata-path",
        default="artifacts/training_data/features_metadata.json",
        help="Metadata JSON for feature artifact.",
    )
    parser.add_argument(
        "--save-source-csv",
        default=None,
        help="Optional CSV snapshot path of source rows for reproducibility.",
    )
    return parser.parse_args()


def load_embedding_cache(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict):
        raise ValueError(f"Cache file {path} is not a dict.")
    return cache


def save_embedding_cache(path: str, cache):
    ensure_parent_dir(path)
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def apply_stratified_sample(
    df: pd.DataFrame, label_column: str, sample_size: int, random_state: int
) -> pd.DataFrame:
    """Sample rows with class stratification by label_column."""
    if sample_size <= 0:
        raise ValueError("--stratified-sample-size must be > 0.")
    if sample_size >= len(df):
        print(
            f"Requested stratified sample size {sample_size} >= rows {len(df)}; using full dataset."
        )
        return df.reset_index(drop=True)

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=sample_size, random_state=random_state
    )
    (sample_idx, _), = splitter.split(df, df[label_column].astype(str))
    return df.iloc[sample_idx].reset_index(drop=True)


def main():
    args = parse_args()

    print("Connecting to Snowflake...")
    sf_session = get_snowflake_session()
    df = load_product_data(
        session=sf_session,
        table=args.table,
        min_category_count=args.min_category_count,
        row_limit=args.row_limit,
    )
    print(f"Loaded {len(df)} rows.")

    if args.label_column not in df.columns:
        raise ValueError(
            f"Missing label column '{args.label_column}'. Available: {sorted(df.columns)}"
        )

    if args.sample_per_category and args.stratified_sample_size:
        raise ValueError(
            "Use only one of --sample-per-category or --stratified-sample-size per run."
        )

    if args.stratified_sample_size and args.stratified_sample_size > 0:
        df = apply_stratified_sample(
            df=df,
            label_column=args.label_column,
            sample_size=args.stratified_sample_size,
            random_state=args.random_state,
        )
        print(
            f"Applied stratified sample size={args.stratified_sample_size}. Rows now: {len(df)}"
        )

    if args.sample_per_category and args.sample_per_category > 0:
        df = (
            df.groupby(args.label_column, group_keys=False)
            .head(args.sample_per_category)
            .reset_index(drop=True)
        )
        print(f"Applied sample-per-category={args.sample_per_category}. Rows now: {len(df)}")

    if args.id_column in df.columns:
        ids = df[args.id_column].astype(str).fillna("").to_numpy()
    else:
        ids = np.array([str(i) for i in range(len(df))], dtype=object)

    labels = df[args.label_column].astype(str).to_numpy()
    text_series = build_product_text(df)
    text_values = text_series.tolist()
    text_hashes = [stable_text_hash(t) for t in text_values]

    print("Preparing Bedrock client and embedding cache...")
    bedrock_client = get_bedrock_client(profile_name=args.aws_profile, region=args.aws_region)
    cache = load_embedding_cache(args.cache_path)
    cache_before = len(cache)

    def checkpoint_cb(cache_obj, processed_count):
        save_embedding_cache(args.cache_path, cache_obj)
        print(
            f"Checkpointed cache after {processed_count} new embeddings "
            f"(current size={len(cache_obj)})."
        )

    embeddings = embed_texts_from_cache(
        texts=text_values,
        text_hashes=text_hashes,
        cache=cache,
        client=bedrock_client,
        model_id=args.model_id,
        show_progress=True,
        max_workers=args.max_workers,
        checkpoint_every=args.checkpoint_every if args.checkpoint_every > 0 else None,
        on_checkpoint=checkpoint_cb if args.checkpoint_every > 0 else None,
        max_retries=args.max_retries,
    )
    save_embedding_cache(args.cache_path, cache)
    print(f"Embedding cache size: {cache_before} -> {len(cache)}")

    ensure_parent_dir(args.output_path)
    np.savez_compressed(
        args.output_path,
        embeddings=embeddings,
        labels=np.asarray(labels, dtype=object),
        ids=np.asarray(ids, dtype=object),
        text_hashes=np.asarray(text_hashes, dtype=object),
        model_id=np.asarray([args.model_id], dtype=object),
    )
    print(f"Saved features to {args.output_path}")

    metadata = {
        "rows": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "label_column": args.label_column,
        "id_column": args.id_column if args.id_column in df.columns else None,
        "table": args.table,
        "model_id": args.model_id,
        "cache_path": args.cache_path,
        "max_workers": args.max_workers,
        "checkpoint_every": args.checkpoint_every,
        "max_retries": args.max_retries,
        "stratified_sample_size": args.stratified_sample_size,
        "sample_per_category": args.sample_per_category,
        "random_state": args.random_state,
    }
    ensure_parent_dir(args.metadata_path)
    with open(args.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {args.metadata_path}")

    if args.save_source_csv:
        ensure_parent_dir(args.save_source_csv)
        df.to_csv(args.save_source_csv, index=False)
        print(f"Saved source data snapshot to {args.save_source_csv}")


if __name__ == "__main__":
    main()
