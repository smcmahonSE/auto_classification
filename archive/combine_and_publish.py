"""
Combine LEI + Services + LCG classification results into a master CSV
and write to Snowflake.

Run after all three datasets are classified:
    python combine_and_publish.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from product_classifier_utils import get_snowflake_session

# ── Config ─────────────────────────────────────────────────────────────────────
MARGIN_THRESHOLD  = 0.05
OUTPUT_TABLE      = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_L3_CLASSIFICATIONS"
MASTER_CSV        = PROJECT_ROOT / "artifacts/analysis/master_classification_results.csv"
MASTER_RESIDUAL   = PROJECT_ROOT / "artifacts/analysis/master_residual_for_clustering.csv"

SOURCES = [
    ("LEI",      PROJECT_ROOT / "artifacts/analysis/l4_classification_validate_products/classification_results.csv"),
    ("SERVICES", PROJECT_ROOT / "artifacts/analysis/l4_classification_classify_services/classification_results.csv"),
    ("LCG",      PROJECT_ROOT / "artifacts/analysis/l4_classification_classify_lcg/classification_results.csv"),
]

# ── Load and combine ───────────────────────────────────────────────────────────
print("Loading classification results...")
parts = []
for source, path in SOURCES:
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping {source}")
        continue
    df = pd.read_csv(path)
    df["SOURCE"] = source
    # Reapply margin threshold consistently (in case any were saved with old threshold)
    df["IS_LOW_CONFIDENCE"] = df["CONFIDENCE_MARGIN"] < MARGIN_THRESHOLD
    parts.append(df)
    high = (~df["IS_LOW_CONFIDENCE"]).sum()
    print(f"  {source}: {len(df):,} rows | high-conf: {high:,} ({high/len(df)*100:.1f}%)")

master = pd.concat(parts, ignore_index=True)
print(f"\nMaster total: {len(master):,} rows")

high = (~master["IS_LOW_CONFIDENCE"]).sum()
low  =   master["IS_LOW_CONFIDENCE"].sum()
print(f"High-confidence: {high:,} ({high/len(master)*100:.1f}%)")
print(f"Residual:        {low:,} ({low/len(master)*100:.1f}%)")

print("\nL4 Distribution (master):")
dist = master.groupby("ASSIGNED_L4_LABEL")["PRODUCT_ID"].count().sort_values(ascending=False)
for label, count in dist.items():
    print(f"  {label:<40} {count:>10,} ({count/len(master)*100:.1f}%)")

# ── Save master CSV ────────────────────────────────────────────────────────────
master.to_csv(MASTER_CSV, index=False)
print(f"\nMaster CSV saved: {MASTER_CSV} ({MASTER_CSV.stat().st_size/1e6:.0f} MB)")

# ── Save combined residual ─────────────────────────────────────────────────────
residual = master[master["IS_LOW_CONFIDENCE"]].copy()
residual.to_csv(MASTER_RESIDUAL, index=False)
print(f"Master residual saved: {MASTER_RESIDUAL} ({len(residual):,} rows)")

# ── Save summary JSON ──────────────────────────────────────────────────────────
summary = {
    "total_listings": len(master),
    "margin_threshold": MARGIN_THRESHOLD,
    "high_confidence_count": int(high),
    "low_confidence_count": int(low),
    "by_source": {
        source: {
            "total": int((master["SOURCE"] == source).sum()),
            "high_confidence": int((master[master["SOURCE"] == source]["IS_LOW_CONFIDENCE"] == False).sum()),
        }
        for source in master["SOURCE"].unique()
    },
    "l4_distribution": dist.to_dict(),
}
summary_path = PROJECT_ROOT / "artifacts/analysis/master_run_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {summary_path}")

# ── Rename L4 → NEW_L3 for publishing ─────────────────────────────────────────
master = master.rename(columns={
    "ASSIGNED_L4_ID":    "ASSIGNED_NEW_L3_ID",
    "ASSIGNED_L4_LABEL": "ASSIGNED_NEW_L3_LABEL",
})

# ── Re-join source columns from Snowflake ──────────────────────────────────────
print("\nReloading source columns from Snowflake for enrichment...")
sf = get_snowflake_session()

SOURCE_TABLES = {
    "LEI":      "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LEI",
    "SERVICES": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES",
    "LCG":      "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LCG",
}
SOURCE_COLS = ["PRODUCT_ID", "PRODUCT_NAME", "DESCRIPTION", "PRICING_STATUS_C", "LIST_PRICE_C"]

source_frames = []
for source, table in SOURCE_TABLES.items():
    if source not in master["SOURCE"].values:
        continue
    print(f"  Loading {source} source columns...")
    src = sf.sql(f"""
        SELECT PRODUCT_ID, PRODUCT_NAME, DESCRIPTION, PRICING_STATUS_C, LIST_PRICE_C
        FROM {table}
    """).to_pandas()
    src["SOURCE"] = source
    source_frames.append(src)

source_df = pd.concat(source_frames, ignore_index=True)
master = master.merge(source_df, on=["PRODUCT_ID", "SOURCE"], how="left")
print(f"Source columns joined. Master shape: {master.shape}")

# ── Write to Snowflake ─────────────────────────────────────────────────────────
print(f"\nWriting {len(master):,} rows to Snowflake table {OUTPUT_TABLE}...")
sf.sql("USE DATABASE SNOWFLAKE_LEARNING_DB").collect()
sf.sql("USE SCHEMA SMCMAHON_PRODUCTS").collect()

publish_cols = [
    "PRODUCT_ID", "SOURCE", "PRODUCT_NAME", "DESCRIPTION",
    "PRICING_STATUS_C", "LIST_PRICE_C",
    "ASSIGNED_NEW_L3_ID", "ASSIGNED_NEW_L3_LABEL",
    "CONFIDENCE", "CONFIDENCE_MARGIN", "IS_LOW_CONFIDENCE",
]
for col in ["CURRENT_L3", "CURRENT_L4", "CURRENT_L5"]:
    if col in master.columns:
        publish_cols.append(col)

publish_df = master[[c for c in publish_cols if c in master.columns]].copy()
publish_df.columns = [c.upper() for c in publish_df.columns]

sp_df = sf.create_dataframe(publish_df)
sp_df.write.mode("overwrite").save_as_table(OUTPUT_TABLE)
print(f"Written to {OUTPUT_TABLE} successfully.")
print("\nAll done!")
