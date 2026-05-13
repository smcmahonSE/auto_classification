-- Services Product Bucket Query
-- Extracts the ~9.1M listings from the SERVICES table that are almost certainly
-- products rather than genuine services, based on current L3/L4 categorization.
--
-- Bucket breakdown (from data analysis, May 2026):
--   Bucket 1: Chemistry and Materials, no L4/L5  →  4.84M  (46%)  uncategorized chemicals
--   Bucket 2: Biology > Biochemistry & Molecular Biology  →  2.93M  (28%)  ambiguous (kits + services)
--   Bucket 3: Biology > Cells and Tissues  →  570K  (5.5%)  clearly products
--   Bucket 4: Compounds  →  747K  (7%)  raw chemicals, clearly products
--
-- Excludes:
--   Bucket 5: All other service categories  →  1.35M  (13%)  genuine services
--             Pending PM decision on whether these are in scope.
--
-- Usage: This WHERE clause is embedded in l4_taxonomy_classification.ipynb
--        (MODE = "classify_services"). Provided here as a standalone reference.

SELECT
    PRODUCT_ID,
    PRODUCT_NAME,
    DESCRIPTION,
    PRICING_STATUS_C,
    LIST_PRICE_C,
    PARENT_3_CATEGORY AS CURRENT_L3,
    PARENT_4_CATEGORY AS CURRENT_L4,
    PARENT_5_CATEGORY AS CURRENT_L5
FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES
WHERE (
    -- Bucket 1: Chemistry and Materials with no L4 (uncategorized chemicals)
    (PARENT_3_CATEGORY = 'Chemistry and Materials'
        AND (PARENT_4_CATEGORY IS NULL OR TRIM(PARENT_4_CATEGORY) = ''))

    -- Bucket 2: Biology > Biochemistry & Molecular Biology (ambiguous: kits + services)
    OR (PARENT_3_CATEGORY = 'Biology'
        AND PARENT_4_CATEGORY = 'Biochemistry & Molecular Biology')

    -- Bucket 3: Biology > Cells and Tissues (clearly products, misclassified)
    OR (PARENT_3_CATEGORY = 'Biology'
        AND PARENT_4_CATEGORY = 'Cells and Tissues')

    -- Bucket 4: Compounds (raw chemicals, clearly products)
    OR PARENT_3_CATEGORY = 'Compounds'
);

-- ── Row counts by bucket (from pre-analysis, for reference) ──────────────────
-- SELECT
--     CASE
--         WHEN PARENT_3_CATEGORY = 'Chemistry and Materials'
--              AND (PARENT_4_CATEGORY IS NULL OR TRIM(PARENT_4_CATEGORY) = '')
--             THEN 'Bucket 1: Chemistry blank'
--         WHEN PARENT_3_CATEGORY = 'Biology'
--              AND PARENT_4_CATEGORY = 'Biochemistry & Molecular Biology'
--             THEN 'Bucket 2: Biology/Biochemistry'
--         WHEN PARENT_3_CATEGORY = 'Biology'
--              AND PARENT_4_CATEGORY = 'Cells and Tissues'
--             THEN 'Bucket 3: Cells & Tissues'
--         WHEN PARENT_3_CATEGORY = 'Compounds'
--             THEN 'Bucket 4: Compounds'
--         ELSE 'Genuine Services (excluded)'
--     END AS BUCKET,
--     COUNT(*) AS N
-- FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SERVICES
-- GROUP BY 1
-- ORDER BY N DESC;
