"""
Seed Snowflake with pre-embedded L3 and L4 anchor descriptions.

Creates two tables:
  SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L3_DESCRIPTIONS
  SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L4_DESCRIPTIONS

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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from product_classifier_utils import get_snowflake_session

# ── Config ────────────────────────────────────────────────────────────────────
AWS_PROFILE  = "staging.admin"
AWS_REGION   = "us-east-1"
MODEL_ID     = "amazon.titan-embed-text-v1"

L3_ANCHORS   = PROJECT_ROOT / "analysis/data/l4_taxonomy_anchors.json"
L4_ANCHORS   = PROJECT_ROOT / "analysis/data/l4_subcategory_anchors.json"

L3_TABLE     = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L3_DESCRIPTIONS"
L4_TABLE     = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.EMBEDDED_L4_DESCRIPTIONS"


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


# ── Build L3 DataFrame ────────────────────────────────────────────────────────

def build_l3_df(client) -> pd.DataFrame:
    print("\n=== Embedding L3 anchors ===")
    with open(L3_ANCHORS) as f:
        data = json.load(f)

    anchors = data["anchors"]
    labels  = [a["label"] for a in anchors]
    vectors = embed_batch(client, [a["description"] for a in anchors], labels)

    rows = []
    for i, (anchor, vec) in enumerate(zip(anchors, vectors)):
        rows.append({
            "L3_ID":                  i + 1,
            "ASSIGNED_NEW_L3_LABEL":  anchor["label"],
            "ASSIGNED_NEW_L3_ID":     anchor["id"],
            "L3_DESCRIPTION":         anchor["description"],
            "L3_EMBED":               json.dumps(vec),   # stored as JSON string → VARIANT
        })

    df = pd.DataFrame(rows)
    print(f"\nL3 table: {len(df)} rows")
    return df


# ── Build L4 DataFrame ────────────────────────────────────────────────────────

def build_l4_df(client) -> pd.DataFrame:
    print("\n=== Embedding L4 anchors ===")
    with open(L4_ANCHORS) as f:
        data = json.load(f)

    rows   = []
    texts  = []
    meta   = []   # (l3_id, subcat)

    for l3_id, subcats in data["l3_subcategories"].items():
        for subcat in subcats:
            texts.append(subcat["description"])
            meta.append((l3_id, subcat))

    labels  = [m[1]["label"] for m in meta]
    vectors = embed_batch(client, texts, labels)

    for i, ((l3_id, subcat), vec) in enumerate(zip(meta, vectors)):
        rows.append({
            "L4_ID":                  i + 1,
            "ASSIGNED_L4_LABEL":      subcat["label"],
            "ASSIGNED_L4_ID":         subcat["id"],
            "ASSIGNED_NEW_L3_ID":     l3_id,
            "L4_DESCRIPTION":         subcat["description"],
            "L4_EMBED":               json.dumps(vec),   # stored as JSON string → VARIANT
        })

    df = pd.DataFrame(rows)
    print(f"\nL4 table: {len(df)} rows")
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

    l3_df = build_l3_df(bedrock)
    l4_df = build_l4_df(bedrock)

    print("\nConnecting to Snowflake...")
    sf = get_snowflake_session()
    sf.sql("USE ROLE \"DEPT-ENGINEERING\"").collect()
    sf.sql("USE DATABASE SNOWFLAKE_LEARNING_DB").collect()
    sf.sql("USE SCHEMA SMCMAHON_PRODUCTS").collect()

    publish(sf, l3_df, L3_TABLE)
    publish(sf, l4_df, L4_TABLE)

    print("\n=== Done ===")
    print(f"  {L3_TABLE}: {len(l3_df)} L3 categories")
    print(f"  {L4_TABLE}: {len(l4_df)} L4 subcategories")
    print("\nL3 categories written:")
    for _, row in l3_df.iterrows():
        print(f"  [{row['L3_ID']:2d}] {row['ASSIGNED_NEW_L3_ID']:<35} {row['ASSIGNED_NEW_L3_LABEL']}")
    print("\nL4 subcategories written:")
    for _, row in l4_df.iterrows():
        print(f"  [{row['L4_ID']:2d}] {row['ASSIGNED_NEW_L3_ID']:<35} {row['ASSIGNED_L4_LABEL']}")
