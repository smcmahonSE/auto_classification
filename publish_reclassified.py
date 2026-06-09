"""
Publish the reclassified master CSV to Snowflake.

Reads the already-updated master_classification_results.csv, reloads product
text from Snowflake source tables, and overwrites NEW_L3_CLASSIFICATIONS.

Run after: python reclassify_residual.py --phase combine
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from product_classifier_utils import get_snowflake_session

OUTPUT_TABLE = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_L3_CLASSIFICATIONS"
MASTER_CSV   = PROJECT_ROOT / "artifacts/analysis/master_classification_results.csv"

SOURCE_TABLES = {
    "LEI":      "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LEI",
    "SERVICES": "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES",
    "LCG":      "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_LCG",
}

print(f"Loading master CSV ({MASTER_CSV.stat().st_size/1e6:.0f} MB)...")
master = pd.read_csv(MASTER_CSV, dtype={"PRODUCT_ID": str}, low_memory=False)
print(f"Master: {len(master):,} rows")

high = (~master["IS_LOW_CONFIDENCE"]).sum()
low  =   master["IS_LOW_CONFIDENCE"].sum()
print(f"High-confidence: {high:,} ({high/len(master)*100:.1f}%)")
print(f"Residual:        {low:,} ({low/len(master)*100:.1f}%)")

# Rename internal L4 columns to published NEW_L3 names
master = master.rename(columns={
    "ASSIGNED_L4_ID":    "ASSIGNED_NEW_L3_ID",
    "ASSIGNED_L4_LABEL": "ASSIGNED_NEW_L3_LABEL",
})

# Re-join product text from Snowflake source tables
print("\nReloading product text from Snowflake by SOURCE...")
sf = get_snowflake_session()

text_frames = []
for source, table in SOURCE_TABLES.items():
    if source not in master["SOURCE"].values:
        continue
    print(f"  {source}: loading from {table}...")
    src = sf.sql(f"""
        SELECT PRODUCT_ID, PRODUCT_NAME, DESCRIPTION, PRICING_STATUS_C, LIST_PRICE_C
        FROM {table}
    """).to_pandas()
    src["PRODUCT_ID"] = src["PRODUCT_ID"].astype(str)
    src["SOURCE"] = source
    text_frames.append(src)

text_df = pd.concat(text_frames, ignore_index=True)
master = master.merge(text_df, on=["PRODUCT_ID", "SOURCE"], how="left")
print(f"Product text joined. Shape: {master.shape}")

# Publish
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

print(f"\nWriting {len(publish_df):,} rows to {OUTPUT_TABLE}...")
sf.sql("USE DATABASE SNOWFLAKE_LEARNING_DB").collect()
sf.sql("USE SCHEMA SMCMAHON_PRODUCTS").collect()
sp_df = sf.create_dataframe(publish_df)
sp_df.write.mode("overwrite").save_as_table(OUTPUT_TABLE)
print(f"Done. {OUTPUT_TABLE} updated successfully.")
