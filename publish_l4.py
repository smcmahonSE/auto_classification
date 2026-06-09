"""
Add L4 subcategory columns to the Snowflake NEW_L3_CLASSIFICATIONS table.

Reads l4_results.csv produced by classify_l4.py --phase combine, merges the
five L4 columns onto the existing table by PRODUCT_ID, and overwrites the table.

Run after: python classify_l4.py --phase combine
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from product_classifier_utils import get_snowflake_session

OUTPUT_TABLE = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_L3_CLASSIFICATIONS"
L4_RESULTS   = PROJECT_ROOT / "artifacts/analysis/l4_classification/l4_results.csv"

L4_COLS = [
    "PRODUCT_ID",
    "ASSIGNED_L4_ID",
    "ASSIGNED_L4_LABEL",
    "L4_CONFIDENCE",
    "L4_CONFIDENCE_MARGIN",
    "L4_IS_LOW_CONFIDENCE",
]

print(f"Loading L4 results ({L4_RESULTS.stat().st_size/1e6:.0f} MB)...")
l4 = pd.read_csv(L4_RESULTS, dtype={"PRODUCT_ID": str}, low_memory=False)
print(f"L4 results: {len(l4):,} rows")

# Keep only the L4 assignment columns
l4_slim = l4[[c for c in L4_COLS if c in l4.columns]].drop_duplicates("PRODUCT_ID")
print(f"Unique PRODUCT_IDs in L4 results: {len(l4_slim):,}")

# Load existing Snowflake table
print(f"\nLoading existing Snowflake table {OUTPUT_TABLE}...")
sf      = get_snowflake_session()
sf.sql("USE DATABASE SNOWFLAKE_LEARNING_DB").collect()
sf.sql("USE SCHEMA SMCMAHON_PRODUCTS").collect()

existing = sf.sql(f"SELECT * FROM {OUTPUT_TABLE}").to_pandas()
existing["PRODUCT_ID"] = existing["PRODUCT_ID"].astype(str)
print(f"Existing table: {len(existing):,} rows, {len(existing.columns)} columns")

# Drop any pre-existing L4 columns to avoid duplicates on re-run
drop_cols = [c for c in L4_COLS if c != "PRODUCT_ID" and c in existing.columns]
if drop_cols:
    existing = existing.drop(columns=drop_cols)
    print(f"Dropped existing L4 columns: {drop_cols}")

# Merge L4 results onto existing table
merged = existing.merge(l4_slim, on="PRODUCT_ID", how="left")
print(f"After merge: {len(merged):,} rows, {len(merged.columns)} columns")

# Summary of coverage
total = len(merged)
has_l4  = merged["ASSIGNED_L4_LABEL"].notna().sum()
high_l4 = (merged["L4_IS_LOW_CONFIDENCE"] == False).sum()
print(f"\nL4 coverage:")
print(f"  Has L4 assignment:     {has_l4:,} ({has_l4/total*100:.1f}%)")
print(f"  High-confidence L4:    {high_l4:,} ({high_l4/total*100:.1f}%)")
print(f"  Deferred (no L4 yet):  {total-has_l4:,} ({(total-has_l4)/total*100:.1f}%)")

# Uppercase columns for Snowflake
merged.columns = [c.upper() for c in merged.columns]

print(f"\nWriting {len(merged):,} rows to {OUTPUT_TABLE}...")
sp_df = sf.create_dataframe(merged)
sp_df.write.mode("overwrite").save_as_table(OUTPUT_TABLE)
print(f"Done. {OUTPUT_TABLE} updated with L4 columns.")
print("\nNew columns added:")
for col in L4_COLS[1:]:
    print(f"  {col.upper()}")
