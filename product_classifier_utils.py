import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import boto3
import numpy as np
import pandas as pd
from snowflake.snowpark import Session

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


DEFAULT_PRODUCTS_TABLE = os.environ.get(
    "SNOWFLAKE_PRODUCTS_TABLE",
    "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_L3_STG",
)


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_snowflake_session(connection_params: Optional[Dict[str, str]] = None) -> Session:
    """Create Snowflake session for local execution."""
    if connection_params is None:
        connection_params = {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT", "NTWRVFU-UEC95409"),
            "user": os.environ.get("SNOWFLAKE_USER", "STEPHANIE.MCMAHON@SCIENCEEXCHANGE.COM"),
            "authenticator": "externalbrowser",
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
            "role": os.environ.get("SNOWFLAKE_ROLE"),
        }
    connection_params = {k: v for k, v in connection_params.items() if v}
    return Session.builder.configs(connection_params).create()


def load_product_data(
    session: Session,
    table: Optional[str] = None,
    min_category_count: int = 100,
    row_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load product training data from Snowflake."""
    table_name = table or DEFAULT_PRODUCTS_TABLE
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE parent_3_category IN (
        SELECT parent_3_category
        FROM {table_name}
        GROUP BY parent_3_category
        HAVING COUNT(*) >= {int(min_category_count)}
    )
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


def build_product_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate feature text fields used by the classifier."""
    required = {"PRODUCT_NAME", "DESCRIPTION", "PRICING_STATUS_C", "LIST_PRICE_C"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for text concat: {missing}")

    return (
        "Name: "
        + df["PRODUCT_NAME"].fillna("Unknown").astype(str)
        + ", Description: "
        + df["DESCRIPTION"].fillna("No description provided").astype(str)
        + ", Pricing Status: "
        + df["PRICING_STATUS_C"].fillna("Unknown").astype(str)
        + ", List Price: "
        + df["LIST_PRICE_C"].fillna("Not available").astype(str)
    )


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_bedrock_client(profile_name: Optional[str], region: str):
    """Create Bedrock runtime client using optional AWS profile."""
    if profile_name:
        boto3.setup_default_session(profile_name=profile_name)
    return boto3.client(service_name="bedrock-runtime", region_name=region)


def invoke_titan_embed(client, text: str, model_id: str, max_retries: int = 5) -> List[float]:
    """Embed a single text using Titan. Retries transient failures."""
    text = str(text).strip() or " "
    payload = {"inputText": text}

    for attempt in range(max_retries):
        try:
            response = client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            result = json.loads(response["body"].read().decode("utf-8"))
            embedding = result.get("embedding", [])
            if not embedding:
                raise RuntimeError("Titan returned empty embedding payload.")
            return embedding
        except Exception:
            if attempt == max_retries - 1:
                raise
            sleep_s = 1.5**attempt
            time.sleep(sleep_s)

    raise RuntimeError("Failed to generate embedding after retries.")


def embed_texts_from_cache(
    texts: Iterable[str],
    text_hashes: Iterable[str],
    cache: Dict[str, np.ndarray],
    client,
    model_id: str,
    show_progress: bool = True,
    max_workers: int = 1,
    checkpoint_every: Optional[int] = None,
    on_checkpoint: Optional[Callable[[Dict[str, np.ndarray], int], None]] = None,
    max_retries: int = 5,
) -> np.ndarray:
    """Embed texts, reusing cache entries by hash."""
    hashes = list(text_hashes)
    text_list = list(texts)
    missing_hashes = [h for h in sorted(set(hashes)) if h not in cache]

    text_by_hash = {}
    for h, t in zip(hashes, text_list):
        if h not in text_by_hash:
            text_by_hash[h] = t

    if missing_hashes:
        processed = 0

        def maybe_checkpoint() -> None:
            if (
                checkpoint_every is not None
                and checkpoint_every > 0
                and on_checkpoint is not None
                and processed > 0
                and processed % checkpoint_every == 0
            ):
                on_checkpoint(cache, processed)

        if max_workers <= 1:
            iterator = missing_hashes
            if show_progress and HAS_TQDM:
                iterator = tqdm(missing_hashes, desc="Embedding missing texts")
            for h in iterator:
                cache[h] = np.asarray(
                    invoke_titan_embed(
                        client=client,
                        text=text_by_hash[h],
                        model_id=model_id,
                        max_retries=max_retries,
                    ),
                    dtype=np.float32,
                )
                processed += 1
                maybe_checkpoint()
        else:
            progress = tqdm(total=len(missing_hashes), desc="Embedding missing texts") if (show_progress and HAS_TQDM) else None

            def _embed_one(h: str):
                emb = invoke_titan_embed(
                    client=client,
                    text=text_by_hash[h],
                    model_id=model_id,
                    max_retries=max_retries,
                )
                return h, np.asarray(emb, dtype=np.float32)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_embed_one, h) for h in missing_hashes]
                for fut in as_completed(futures):
                    h, emb = fut.result()
                    cache[h] = emb
                    processed += 1
                    if progress is not None:
                        progress.update(1)
                    maybe_checkpoint()
            if progress is not None:
                progress.close()

    embeddings = [cache[h] for h in hashes]
    return np.vstack(embeddings).astype(np.float32)
