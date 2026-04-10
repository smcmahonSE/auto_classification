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
    stable_text_hash,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build L5 training features from label table + source product table."
    )
    parser.add_argument(
        "--labels-table",
        default="SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PROPOSED_L5",
        help="Table containing PRODUCT_ID + proposed L5 labels.",
    )
    parser.add_argument(
        "--products-table",
        default="SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LCG",
        help="Source product table containing text fields.",
    )
    parser.add_argument("--join-id-column", default="PRODUCT_ID")
    parser.add_argument("--label-column", default="PROPOSED_LABEL")
    parser.add_argument("--min-category-count", type=int, default=20)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--priced-only", action="store_true")
    parser.add_argument(
        "--sample-per-category",
        type=int,
        default=None,
        help="Optional cap per label for quick tests.",
    )
    parser.add_argument(
        "--stratified-sample-size",
        type=int,
        default=None,
        help="Approximate sample size with label stratification.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--aws-profile", default="staging.admin")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--model-id", default="amazon.titan-embed-text-v1")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=2500,
        help="Persist embedding cache every N new embeddings (0 disables).",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--cache-path",
        default="artifacts/cache/embedding_cache.pkl",
        help="Pickle mapping text_hash -> embedding vector.",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/training_data/features_l5.npz",
        help="Compressed feature artifact.",
    )
    parser.add_argument(
        "--metadata-path",
        default="artifacts/training_data/features_l5_metadata.json",
        help="Metadata JSON for feature artifact.",
    )
    parser.add_argument(
        "--save-source-csv",
        default=None,
        help="Optional CSV snapshot path for reproducibility.",
    )
    parser.add_argument(
        "--include-insert-products",
        action="store_true",
        help="Include rows where DESCRIPTION contains INSERT. Default excludes them.",
    )
    return parser.parse_args()


def load_embedding_cache(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("rb") as f:
            cache = pickle.load(f)
    except EOFError:
        print(
            f"Warning: cache file {path} is empty/corrupt (EOFError). "
            "Starting with an empty cache."
        )
        return {}
    except pickle.UnpicklingError:
        print(
            f"Warning: cache file {path} is not a valid pickle. "
            "Starting with an empty cache."
        )
        return {}
    if not isinstance(cache, dict):
        raise ValueError(f"Cache file {path} is not a dict.")
    return cache


def save_embedding_cache(path: str, cache):
    ensure_parent_dir(path)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f)
    Path(tmp_path).replace(path)


def apply_stratified_sample(
    df: pd.DataFrame, label_column: str, sample_size: int, random_state: int
) -> pd.DataFrame:
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


def drop_sparse_labels_for_stratification(
    df: pd.DataFrame, label_column: str, min_count: int = 2
) -> tuple[pd.DataFrame, int, int]:
    counts = df[label_column].astype(str).value_counts(dropna=False)
    keep_labels = counts[counts >= min_count].index
    keep_mask = df[label_column].astype(str).isin(keep_labels)
    dropped_row_count = int((~keep_mask).sum())
    dropped_label_count = int((counts < min_count).sum())
    if dropped_row_count == 0:
        return df, 0, 0
    return df.loc[keep_mask].reset_index(drop=True), dropped_label_count, dropped_row_count


def load_l5_training_source(
    labels_table: str,
    products_table: str,
    join_id_column: str,
    label_column: str,
    row_limit: int | None = None,
    priced_only: bool = False,
    exclude_insert_products: bool = True,
) -> pd.DataFrame:
    session = get_snowflake_session()

    where_parts = [f"l.{label_column} IS NOT NULL"]
    if priced_only:
        where_parts.append("UPPER(COALESCE(p.PRICING_STATUS_C, '')) = 'PRICED'")
    if exclude_insert_products:
        where_parts.append("UPPER(COALESCE(p.DESCRIPTION, '')) NOT LIKE '%INSERT%'")
    where_sql = " AND ".join(where_parts)

    query = f"""
    SELECT
      p.*,
      l.{label_column} AS {label_column},
      l.PROPOSED_CLUSTER
    FROM {labels_table} l
    JOIN {products_table} p
      ON p.{join_id_column} = l.{join_id_column}
    WHERE {where_sql}
    """
    if row_limit is not None and row_limit > 0:
        query = f"{query}\nLIMIT {int(row_limit)}"

    df_snowflake = session.sql(query)
    try:
        return df_snowflake.to_pandas()
    except Exception as exc:
        if "Optional dependency: 'pandas' is not installed" in str(exc):
            rows = df_snowflake.collect()
            return pd.DataFrame([row.as_dict() for row in rows])
        raise


def main():
    args = parse_args()

    print("Loading joined L5 source data from Snowflake...")
    df = load_l5_training_source(
        labels_table=args.labels_table,
        products_table=args.products_table,
        join_id_column=args.join_id_column,
        label_column=args.label_column,
        row_limit=args.row_limit,
        priced_only=args.priced_only,
        exclude_insert_products=not args.include_insert_products,
    )
    print(f"Loaded {len(df)} rows.")

    if args.label_column not in df.columns:
        raise ValueError(
            f"Missing label column '{args.label_column}'. Available: {sorted(df.columns)}"
        )

    # Apply minimum category count for supervised train stability.
    counts = df[args.label_column].astype(str).value_counts(dropna=False)
    keep = counts[counts >= args.min_category_count].index
    before_rows = len(df)
    df = df[df[args.label_column].astype(str).isin(keep)].reset_index(drop=True)
    print(
        f"Applied min-category-count={args.min_category_count}. "
        f"Rows: {before_rows} -> {len(df)}. Labels kept: {len(keep)}"
    )

    # Stratified sampling guardrail requires at least 2 rows per class.
    df, dropped_label_count, dropped_row_count = drop_sparse_labels_for_stratification(
        df, args.label_column, min_count=2
    )
    if dropped_row_count > 0:
        print(
            "Dropped "
            f"{dropped_row_count} rows across {dropped_label_count} labels with <2 rows "
            "to satisfy stratification requirements."
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

    if args.join_id_column in df.columns:
        ids = df[args.join_id_column].astype(str).fillna("").to_numpy()
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
        "id_column": args.join_id_column if args.join_id_column in df.columns else None,
        "labels_table": args.labels_table,
        "products_table": args.products_table,
        "priced_only": args.priced_only,
        "model_id": args.model_id,
        "cache_path": args.cache_path,
        "max_workers": args.max_workers,
        "checkpoint_every": args.checkpoint_every,
        "max_retries": args.max_retries,
        "dropped_sparse_label_count": dropped_label_count,
        "dropped_sparse_row_count": dropped_row_count,
        "stratified_sample_size": args.stratified_sample_size,
        "sample_per_category": args.sample_per_category,
        "min_category_count": args.min_category_count,
        "random_state": args.random_state,
        "exclude_insert_products": not args.include_insert_products,
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
