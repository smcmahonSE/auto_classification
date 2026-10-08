"""
Seed Snowflake with pre-embedded L2 and L3 anchor descriptions.

Creates two tables:
  SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L2_DESCRIPTIONS
  SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L3_DESCRIPTIONS

Embeddings are stored as JSON arrays (VARIANT) for easy loading in downstream services.
Each run overwrites both tables — safe to re-run if taxonomy definitions change.

Run:
    python seed_anchor_tables.py
"""

import json
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "classification_pipeline"))

from product_classifier_utils import get_snowflake_session

# ── Config ────────────────────────────────────────────────────────────────────
AWS_PROFILE  = "bedrock"  # staging.admin lacks Bedrock access (ForbiddenException); use the
                          # same profile as the main classify_listings.py pipeline.
AWS_REGION   = "us-east-1"
MODEL_ID     = "amazon.titan-embed-text-v1"

L2_ANCHORS   = PROJECT_ROOT / "classification_pipeline/taxonomy/l2_taxonomy_anchors.json"
L3_ANCHORS   = PROJECT_ROOT / "classification_pipeline/taxonomy/l3_taxonomy_anchors.json"

L2_TABLE     = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L2_DESCRIPTIONS"
L3_TABLE     = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L3_DESCRIPTIONS"


# ── Bedrock embedding ─────────────────────────────────────────────────────────

def get_bedrock():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client("bedrock-runtime")


def embed_text(client, text: str) -> list[float]:
    """Call Titan and return the 1,536-dim embedding as a plain Python list."""
    body = json.dumps({"inputText": text})
    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    return result["embedding"]


def embed_batch(client, texts: list[str], label: str) -> list[list[float]]:
    """Embed a list of texts with a small delay between calls to avoid throttling."""
    vectors = []
    for i, text in enumerate(texts):
        vec = embed_text(client, text)
        vectors.append(vec)
        print(f"  [{i+1}/{len(texts)}] {label[i]}")
        if i < len(texts) - 1:
            time.sleep(0.2)
    return vectors


# ── Build L2 DataFrame ────────────────────────────────────────────────────────

def build_l2_df(client) -> pd.DataFrame:
    print("\n=== Embedding L2 anchors ===")
    with open(L2_ANCHORS) as f:
        data = json.load(f)

    anchors = data["anchors"]
    labels  = [a["label"] for a in anchors]
    vectors = embed_batch(client, [a["description"] for a in anchors], labels)

    rows = []
    for i, (anchor, vec) in enumerate(zip(anchors, vectors)):
        rows.append({
            "L2_ID":              i + 1,
            "ASSIGNED_L2_LABEL":  anchor["label"],
            "ASSIGNED_L2_ID":     anchor["id"],
            "L2_DESCRIPTION":     anchor["description"],
            "L2_EMBED":           json.dumps(vec),   # stored as JSON string → VARIANT
        })

    df = pd.DataFrame(rows)
    print(f"\nL2 table: {len(df)} rows")
    return df


# ── Build L3 DataFrame ────────────────────────────────────────────────────────

def build_l3_df(client) -> pd.DataFrame:
    print("\n=== Embedding L3 anchors ===")
    with open(L3_ANCHORS) as f:
        data = json.load(f)

    rows   = []
    texts  = []
    meta   = []   # (l2_id, subcat)

    for l2_id, subcats in data["l2_subcategories"].items():
        for subcat in subcats:
            texts.append(subcat["description"])
            meta.append((l2_id, subcat))

    labels  = [m[1]["label"] for m in meta]
    vectors = embed_batch(client, texts, labels)

    for i, ((l2_id, subcat), vec) in enumerate(zip(meta, vectors)):
        rows.append({
            "L3_ID":              i + 1,
            "ASSIGNED_L3_LABEL":  subcat["label"],
            "ASSIGNED_L3_ID":     subcat["id"],
            "ASSIGNED_L2_ID":     l2_id,
            "L3_DESCRIPTION":     subcat["description"],
            "L3_EMBED":           json.dumps(vec),   # stored as JSON string → VARIANT
        })

    df = pd.DataFrame(rows)
    print(f"\nL3 table: {len(df)} rows")
    return df


# ── Publish to Snowflake ──────────────────────────────────────────────────────

def publish(sf, df: pd.DataFrame, table: str):
    print(f"\nWriting {len(df)} rows to {table}...")
    sp_df = sf.create_dataframe(df)
    sp_df.write.mode("overwrite").save_as_table(table)
    print(f"Done. {table} created/updated.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Connecting to AWS Bedrock...")
    bedrock = get_bedrock()

    l2_df = build_l2_df(bedrock)
    l3_df = build_l3_df(bedrock)

    print("\nConnecting to Snowflake...")
    sf = get_snowflake_session()
    sf.sql("USE ROLE \"DEPT-ENGINEERING\"").collect()
    sf.sql("USE DATABASE SNOWFLAKE_LEARNING_DB").collect()
    sf.sql("USE SCHEMA SMCMAHON_PRODUCTS").collect()

    publish(sf, l2_df, L2_TABLE)
    publish(sf, l3_df, L3_TABLE)

    print("\n=== Done ===")
    print(f"  {L2_TABLE}: {len(l2_df)} L2 categories")
    print(f"  {L3_TABLE}: {len(l3_df)} L3 subcategories")
    print("\nL2 categories written:")
    for _, row in l2_df.iterrows():
        print(f"  [{row['L2_ID']:2d}] {row['ASSIGNED_L2_ID']:<35} {row['ASSIGNED_L2_LABEL']}")
    print("\nL3 subcategories written:")
    for _, row in l3_df.iterrows():
        print(f"  [{row['L3_ID']:2d}] {row['ASSIGNED_L2_ID']:<35} {row['ASSIGNED_L3_LABEL']}")
