"""
Publish the full re-classified L3 + L4 results to Snowflake.

Reads:
  - artifacts/analysis/master_classification_results.csv  → updated L3 assignments
  - artifacts/analysis/l4_classification/l4_results.csv   → L4 subcategory assignments

Merges them, renames columns for clarity, and overwrites NEW_L3_CLASSIFICATIONS.

Final table schema:
  PRODUCT_ID, SOURCE, PRODUCT_NAME, DESCRIPTION, PRICING_STATUS_C, LIST_PRICE_C
  ASSIGNED_NEW_L3_ID, ASSIGNED_NEW_L3_LABEL
  L3_CONFIDENCE, L3_CONFIDENCE_MARGIN, L3_IS_LOW_CONFIDENCE
  ASSIGNED_L4_ID, ASSIGNED_L4_LABEL
  L4_CONFIDENCE, L4_CONFIDENCE_MARGIN, L4_IS_LOW_CONFIDENCE

Run after: classify_l4.py --phase combine
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from product_classifier_utils import get_snowflake_session

MASTER_CSV   = PROJECT_ROOT / "artifacts/analysis/master_classification_results.csv"
L4_CSV       = PROJECT_ROOT / "artifacts/analysis/l4_classification/l4_results.csv"
OUTPUT_TABLE = "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.NEW_L3_CLASSIFICATIONS"

# ── Load master (L3 assignments) ──────────────────────────────────────────────
print(f"Loading master CSV ({MASTER_CSV.stat().st_size/1e9:.2f} GB)...")
master = pd.read_csv(MASTER_CSV, dtype={"PRODUCT_ID": str}, low_memory=False)
print(f"Master: {len(master):,} rows")

master = master.rename(columns={
    "ASSIGNED_L4_ID":    "ASSIGNED_NEW_L3_ID",
    "ASSIGNED_L4_LABEL": "ASSIGNED_NEW_L3_LABEL",
    "CONFIDENCE":        "L3_CONFIDENCE",
    "CONFIDENCE_MARGIN": "L3_CONFIDENCE_MARGIN",
    "IS_LOW_CONFIDENCE": "L3_IS_LOW_CONFIDENCE",
})

# ── Load L4 results ────────────────────────────────────────────────────────────
print(f"\nLoading L4 results ({L4_CSV.stat().st_size/1e9:.2f} GB)...")
l4 = pd.read_csv(L4_CSV, dtype={"PRODUCT_ID": str}, low_memory=False)
print(f"L4 results: {len(l4):,} rows")

L4_KEEP = ["PRODUCT_ID", "ASSIGNED_L4_ID", "ASSIGNED_L4_LABEL",
           "L4_CONFIDENCE", "L4_CONFIDENCE_MARGIN", "L4_IS_LOW_CONFIDENCE"]
l4_slim = l4[[c for c in L4_KEEP if c in l4.columns]].drop_duplicates("PRODUCT_ID")

# ── Merge ──────────────────────────────────────────────────────────────────────
print("\nMerging L3 + L4 results...")
combined = master.merge(l4_slim, on="PRODUCT_ID", how="left")
print(f"Combined: {len(combined):,} rows, {len(combined.columns)} columns")

# ── Summary ────────────────────────────────────────────────────────────────────
total  = len(combined)
hi_l3  = (~combined["L3_IS_LOW_CONFIDENCE"]).sum()
hi_l4  = (combined["L4_IS_LOW_CONFIDENCE"] == False).sum()
no_l4  = combined["ASSIGNED_L4_LABEL"].isna().sum()

print(f"\nL3 high-confidence:  {hi_l3:,} ({hi_l3/total*100:.1f}%)")
print(f"L3 low-confidence:   {total-hi_l3:,} ({(total-hi_l3)/total*100:.1f}%)")
print(f"L4 assigned:         {total-no_l4:,} ({(total-no_l4)/total*100:.1f}%)")
print(f"L4 high-confidence:  {hi_l4:,} ({hi_l4/total*100:.1f}%)")
print(f"No L4 (General Office Supplies): {no_l4:,} ({no_l4/total*100:.1f}%)")

print("\nL3 distribution:")
print(combined["ASSIGNED_NEW_L3_LABEL"].value_counts().to_string())
print("\nL4 distribution (top 20):")
print(combined["ASSIGNED_L4_LABEL"].value_counts().head(20).to_string())

# ── Publish ────────────────────────────────────────────────────────────────────
combined.columns = [c.upper() for c in combined.columns]

print(f"\nConnecting to Snowflake...")
sf = get_snowflake_session()
sf.sql("USE ROLE \"DEPT-ENGINEERING\"").collect()
sf.sql("USE DATABASE SNOWFLAKE_LEARNING_DB").collect()
sf.sql("USE SCHEMA SMCMAHON_PRODUCTS").collect()

print(f"Writing {len(combined):,} rows to {OUTPUT_TABLE}...")
sp_df = sf.create_dataframe(combined)
sp_df.write.mode("overwrite").save_as_table(OUTPUT_TABLE)
print(f"\nDone. {OUTPUT_TABLE} updated.")
print("\nFinal columns:")
for col in combined.columns:
    print(f"  {col}")
