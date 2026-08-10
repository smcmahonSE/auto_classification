import hashlib
import json
import os
import pickle
import re
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

DEFAULT_L3_ANCHOR_TABLE = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L3_DESCRIPTIONS"
DEFAULT_L4_ANCHOR_TABLE = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L4_DESCRIPTIONS"
DEFAULT_MARGIN_THRESHOLD = 0.05


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


def get_products_session() -> Session:
    """Open a Snowflake session scoped to the SMCMAHON_PRODUCTS schema."""
    sf = get_snowflake_session()
    sf.sql("USE ROLE \"DEPT-ENGINEERING\"").collect()
    sf.sql("USE DATABASE SNOWFLAKE_LEARNING_DB").collect()
    sf.sql("USE SCHEMA SMCMAHON_PRODUCTS").collect()
    return sf


def load_listings(session: Session, table: str) -> pd.DataFrame:
    """Load listing rows (products or services) with usable name/description text."""
    print(f"Loading listings from {table}...")
    df = session.sql(f"""
        SELECT PRODUCT_ID, PRODUCT_VARIANT_ID, PRODUCT_NAME, DESCRIPTION, PRICING_STATUS_C, LIST_PRICE_C, PRODUCT_SEGMENT
        FROM {table}
    """).to_pandas()
    df["PRODUCT_ID"] = df["PRODUCT_ID"].astype(str)
    df = df.rename(columns={"PRODUCT_SEGMENT": "SOURCE"})
    print(f"Loaded: {len(df):,} rows")

    has_text = df["PRODUCT_NAME"].notna() | df["DESCRIPTION"].notna()
    df = df[has_text].copy().reset_index(drop=True)
    print(f"Rows with usable text: {len(df):,}")
    return df


def load_anchors_from_snowflake(
    session: Session,
    l3_table: str = DEFAULT_L3_ANCHOR_TABLE,
    l4_table: str = DEFAULT_L4_ANCHOR_TABLE,
):
    """
    Load pre-embedded L3 and L4 anchor vectors from Snowflake.
    Returns:
        l3_anchors: (ids, labels, normed_matrix)
        l4_by_l3:   dict of l3_id -> (ids, labels, normed_matrix)
    """
    print("Loading L3 anchor vectors from Snowflake...")
    l3_df = session.sql(
        f"SELECT ASSIGNED_NEW_L3_ID, ASSIGNED_NEW_L3_LABEL, L3_EMBED FROM {l3_table}"
    ).to_pandas()
    l3_ids    = l3_df["ASSIGNED_NEW_L3_ID"].tolist()
    l3_labels = l3_df["ASSIGNED_NEW_L3_LABEL"].tolist()
    l3_vecs   = np.array([json.loads(e) for e in l3_df["L3_EMBED"]], dtype=np.float32)
    l3_norms  = np.linalg.norm(l3_vecs, axis=1, keepdims=True)
    l3_normed = l3_vecs / np.clip(l3_norms, 1e-10, None)
    print(f"  L3 anchors: {len(l3_ids)} categories")

    print("Loading L4 anchor vectors from Snowflake...")
    l4_df = session.sql(f"""
        SELECT ASSIGNED_NEW_L3_ID, ASSIGNED_L4_ID, ASSIGNED_L4_LABEL, L4_EMBED
        FROM {l4_table}
        ORDER BY ASSIGNED_NEW_L3_ID, L4_ID
    """).to_pandas()

    l4_by_l3 = {}
    for l3_id, grp in l4_df.groupby("ASSIGNED_NEW_L3_ID"):
        ids    = grp["ASSIGNED_L4_ID"].tolist()
        labels = grp["ASSIGNED_L4_LABEL"].tolist()
        vecs   = np.array([json.loads(e) for e in grp["L4_EMBED"]], dtype=np.float32)
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / np.clip(norms, 1e-10, None)
        l4_by_l3[l3_id] = (ids, labels, normed)
    print(f"  L4 anchors: {sum(len(v[0]) for v in l4_by_l3.values())} subcategories across {len(l4_by_l3)} L3s")

    return (l3_ids, l3_labels, l3_normed), l4_by_l3


def classify_against_anchors(vecs, anchor_ids, anchor_labels, anchor_normed, margin_threshold: float = DEFAULT_MARGIN_THRESHOLD):
    """Cosine similarity classification. Returns (ids, labels, scores, margins, low_conf)."""
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.clip(norms, 1e-10, None)
    sim    = vecs_n @ anchor_normed.T
    top1   = sim.argmax(axis=1)
    top1_s = sim[np.arange(len(sim)), top1]
    sim2   = sim.copy()
    sim2[np.arange(len(sim)), top1] = -1
    margin = top1_s - sim2.max(axis=1)
    return (
        [anchor_ids[i]    for i in top1],
        [anchor_labels[i] for i in top1],
        top1_s.round(4),
        margin.round(4),
        margin < margin_threshold,
    )


def classify_l3_and_l4(vecs, l3_anchors, l4_by_l3):
    """
    Run L3 then L4 classification in one pass.
    Returns all 10 result arrays (L3 + L4 each: ids, labels, scores, margins, low_conf).
    """
    l3_ids, l3_labels, l3_scores, l3_margins, l3_low_conf = classify_against_anchors(vecs, *l3_anchors)

    n = len(vecs)
    l4_ids      = [None] * n
    l4_labels   = [None] * n
    l4_scores   = np.zeros(n, dtype=np.float32)
    l4_margins  = np.zeros(n, dtype=np.float32)
    l4_low_conf = np.ones(n, dtype=bool)

    for unique_l3_id in set(l3_ids):
        if unique_l3_id not in l4_by_l3:
            continue
        idx      = [i for i, lid in enumerate(l3_ids) if lid == unique_l3_id]
        sub_vecs = vecs[np.array(idx)]
        s_ids, s_labels, s_scores, s_margins, s_low_conf = classify_against_anchors(
            sub_vecs, *l4_by_l3[unique_l3_id]
        )
        for pos, i in enumerate(idx):
            l4_ids[i]      = s_ids[pos]
            l4_labels[i]   = s_labels[pos]
            l4_scores[i]   = s_scores[pos]
            l4_margins[i]  = s_margins[pos]
            l4_low_conf[i] = s_low_conf[pos]

    return (
        l3_ids, l3_labels, l3_scores, l3_margins, l3_low_conf,
        l4_ids, l4_labels, l4_scores, l4_margins, l4_low_conf,
    )


def attach_classifications(batch_df: pd.DataFrame, results) -> pd.DataFrame:
    """Attach L3 + L4 classification columns to a DataFrame copy."""
    (l3_ids, l3_labels, l3_scores, l3_margins, l3_low_conf,
     l4_ids, l4_labels, l4_scores, l4_margins, l4_low_conf) = results

    out = batch_df.copy()
    out["ASSIGNED_NEW_L3_ID"]    = l3_ids
    out["ASSIGNED_NEW_L3_LABEL"] = l3_labels
    out["L3_CONFIDENCE"]         = l3_scores
    out["L3_CONFIDENCE_MARGIN"]  = l3_margins
    out["L3_IS_LOW_CONFIDENCE"]  = l3_low_conf
    out["ASSIGNED_L4_ID"]        = l4_ids
    out["ASSIGNED_L4_LABEL"]     = l4_labels
    out["L4_CONFIDENCE"]         = l4_scores
    out["L4_CONFIDENCE_MARGIN"]  = l4_margins
    out["L4_IS_LOW_CONFIDENCE"]  = l4_low_conf
    return out


def load_pickle_cache(path: Path) -> dict:
    """Load a hash->embedding pickle cache from disk, or return {} if absent."""
    path = Path(path)
    if path.exists():
        print(f"Loading cache ({path.stat().st_size/1e9:.2f} GB)...")
        with open(path, "rb") as f:
            cache = pickle.load(f)
        print(f"  Cache: {len(cache):,} entries")
        return cache
    print(f"Cache not found at {path} — starting fresh.")
    return {}


def save_pickle_cache(cache: dict, path: Path) -> None:
    """Atomic write: write to temp file then rename."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    tmp.rename(path)


def load_product_data(
    session: Session,
    table: Optional[str] = None,
    label_column: str = "PARENT_3_CATEGORY",
    min_category_count: int = 100,
    row_limit: Optional[int] = None,
    exclude_insert_products: bool = True,
) -> pd.DataFrame:
    """Load product training data from Snowflake."""
    table_name = table or DEFAULT_PRODUCTS_TABLE
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", label_column):
        raise ValueError(
            "Invalid label column name. Use a simple Snowflake identifier, "
            "for example PARENT_3_CATEGORY or PARENT_4_CATEGORY."
        )
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE {label_column} IN (
        SELECT {label_column}
        FROM {table_name}
        GROUP BY {label_column}
        HAVING COUNT(*) >= {int(min_category_count)}
    )
    """
    if exclude_insert_products:
        query = (
            f"{query}\n"
            "AND UPPER(COALESCE(DESCRIPTION, '')) NOT LIKE '%INSERT%'"
        )
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
            import random
            sleep_s = (2.0 ** attempt) + random.uniform(0, 1)
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
