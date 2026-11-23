-- GENERATED FILE — do not hand-edit.
-- Regenerate with: python3 sql/generate_coverage_sql.py
-- Source of truth: spec_extraction/schemas/spec_model_v2/l2/*.json via
-- spec_extraction/config/loader.py, enumerated through
-- spec_extraction/config/registry.py's MIGRATED_L2_IDS.
--
-- 181 L3 table(s) covered across 25 (L2, env) shard(s)
-- (0 skipped — see stderr from the generator run for why).
-- 25 shard dynamic table(s) + 1 summary view. Sharded by
-- (L2, env), not just L2 — see this file's module docstring for why.
--
-- WAREHOUSE is a placeholder token; deploy.py substitutes its configured
-- default warehouse (or --warehouse override) before executing each shard
-- statement (the summary view doesn't need one — views aren't materialized).
--
-- Statements are separated by a sentinel comment line, not bare ';', since a
-- quoted field label could in principle contain one; deploy.py splits on
-- that sentinel and executes each statement in order.
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_ANALYTICAL_REFERENCE_STANDARDS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("STANDARD_TYPE") AS "STANDARD_TYPE",
    COUNT("ANALYTE_S_PARAMETER") AS "ANALYTE_S_PARAMETER",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("REGULATORY_STATUS_INTENDED_USE") AS "REGULATORY_STATUS_INTENDED_USE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("EXPIRATION_VALIDITY") AS "EXPIRATION_VALIDITY",
    COUNT("TECHNIQUE") AS "TECHNIQUE",
    COUNT("SINGLE_OR_MULTI_COMPONENT") AS "SINGLE_OR_MULTI_COMPONENT",
    COUNT("CAS_NUMBER") AS "CAS_NUMBER",
    COUNT("MATRIX_SOLVENT") AS "MATRIX_SOLVENT",
    COUNT("CONCENTRATION") AS "CONCENTRATION",
    COUNT("ACCREDITATION") AS "ACCREDITATION",
    COUNT("CERTIFIED_ASSIGNED_VALUE") AS "CERTIFIED_ASSIGNED_VALUE",
    COUNT("HAZARD_CLASSIFICATION") AS "HAZARD_CLASSIFICATION",
    COUNT("CONTROLLED_SUBSTANCE_STATUS") AS "CONTROLLED_SUBSTANCE_STATUS",
    COUNT("TRACEABILITY") AS "TRACEABILITY"
    FROM SNOWFLAKE_LEARNING_DB."ANALYTICAL_REFERENCE_STANDARDS".ANALYTICAL_STANDARDS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('STANDARD_TYPE', 'Standard type', TRUE),
    ('ANALYTE_S_PARAMETER', 'Analyte(s) / parameter', TRUE),
    ('FORMAT', 'Format', FALSE),
    ('REGULATORY_STATUS_INTENDED_USE', 'Regulatory status / intended use', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('EXPIRATION_VALIDITY', 'Expiration / validity', FALSE),
    ('TECHNIQUE', 'Technique', FALSE),
    ('SINGLE_OR_MULTI_COMPONENT', 'Single or multi-component', FALSE),
    ('CAS_NUMBER', 'CAS number', FALSE),
    ('MATRIX_SOLVENT', 'Matrix / solvent', FALSE),
    ('CONCENTRATION', 'Concentration', FALSE),
    ('ACCREDITATION', 'Accreditation', FALSE),
    ('CERTIFIED_ASSIGNED_VALUE', 'Certified / assigned value', FALSE),
    ('HAZARD_CLASSIFICATION', 'Hazard classification', FALSE),
    ('CONTROLLED_SUBSTANCE_STATUS', 'Controlled substance status', FALSE),
    ('TRACEABILITY', 'Traceability', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'analytical_reference_standards' AS ASSIGNED_L2_ID,
    'Analytical & Reference Standards' AS ASSIGNED_L2_LABEL,
    'analytical_standards' AS ASSIGNED_L3_ID,
    'Analytical Standards' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Analytical & Reference Standards > Analytical Standards
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("STANDARD_TYPE") AS "STANDARD_TYPE",
    COUNT("ANALYTE_S_PARAMETER") AS "ANALYTE_S_PARAMETER",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("REGULATORY_STATUS_INTENDED_USE") AS "REGULATORY_STATUS_INTENDED_USE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("EXPIRATION_VALIDITY") AS "EXPIRATION_VALIDITY",
    COUNT("ISSUING_PROGRAM") AS "ISSUING_PROGRAM",
    COUNT("MATRIX_SOLVENT") AS "MATRIX_SOLVENT",
    COUNT("CONCENTRATION") AS "CONCENTRATION",
    COUNT("TRACEABILITY") AS "TRACEABILITY",
    COUNT("ACCREDITATION") AS "ACCREDITATION",
    COUNT("CERTIFIED_ASSIGNED_VALUE") AS "CERTIFIED_ASSIGNED_VALUE",
    COUNT("VALUE_TYPE") AS "VALUE_TYPE",
    COUNT("MINIMUM_SAMPLE_SIZE") AS "MINIMUM_SAMPLE_SIZE",
    COUNT("CURRENT_LOT_VALID_USE_DATE") AS "CURRENT_LOT_VALID_USE_DATE",
    COUNT("HOMOGENEITY") AS "HOMOGENEITY"
    FROM SNOWFLAKE_LEARNING_DB."ANALYTICAL_REFERENCE_STANDARDS".REFERENCE_MATERIALS_CRMS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('STANDARD_TYPE', 'Standard type', TRUE),
    ('ANALYTE_S_PARAMETER', 'Analyte(s) / parameter', TRUE),
    ('FORMAT', 'Format', FALSE),
    ('REGULATORY_STATUS_INTENDED_USE', 'Regulatory status / intended use', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('EXPIRATION_VALIDITY', 'Expiration / validity', FALSE),
    ('ISSUING_PROGRAM', 'Issuing program', TRUE),
    ('MATRIX_SOLVENT', 'Matrix / solvent', FALSE),
    ('CONCENTRATION', 'Concentration', FALSE),
    ('TRACEABILITY', 'Traceability', FALSE),
    ('ACCREDITATION', 'Accreditation', FALSE),
    ('CERTIFIED_ASSIGNED_VALUE', 'Certified / assigned value', FALSE),
    ('VALUE_TYPE', 'Value type', FALSE),
    ('MINIMUM_SAMPLE_SIZE', 'Minimum sample size', FALSE),
    ('CURRENT_LOT_VALID_USE_DATE', 'Current lot / valid use date', FALSE),
    ('HOMOGENEITY', 'Homogeneity', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'analytical_reference_standards' AS ASSIGNED_L2_ID,
    'Analytical & Reference Standards' AS ASSIGNED_L2_LABEL,
    'reference_materials_crms' AS ASSIGNED_L3_ID,
    'Reference Materials & CRMs' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Analytical & Reference Standards > Reference Materials & CRMs
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("STANDARD_TYPE") AS "STANDARD_TYPE",
    COUNT("ANALYTE_S_PARAMETER") AS "ANALYTE_S_PARAMETER",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("REGULATORY_STATUS_INTENDED_USE") AS "REGULATORY_STATUS_INTENDED_USE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("EXPIRATION_VALIDITY") AS "EXPIRATION_VALIDITY",
    COUNT("ASSAYED_OR_UNASSAYED") AS "ASSAYED_OR_UNASSAYED",
    COUNT("LEVELS") AS "LEVELS",
    COUNT("CERTIFIED_ASSIGNED_VALUE") AS "CERTIFIED_ASSIGNED_VALUE",
    COUNT("NUMBER_OF_ANALYTES") AS "NUMBER_OF_ANALYTES",
    COUNT("MATRIX_SOLVENT") AS "MATRIX_SOLVENT",
    COUNT("COMPATIBLE_INSTRUMENTS_ASSAYS") AS "COMPATIBLE_INSTRUMENTS_ASSAYS",
    COUNT("OPEN_VIAL_STABILITY") AS "OPEN_VIAL_STABILITY",
    COUNT("SHELF_LIFE") AS "SHELF_LIFE",
    COUNT("VARIANTS_ALLELE_FREQUENCY") AS "VARIANTS_ALLELE_FREQUENCY",
    COUNT("QUANTIFICATION_METHOD") AS "QUANTIFICATION_METHOD",
    COUNT("PEER_DATA_PROGRAM") AS "PEER_DATA_PROGRAM",
    COUNT("ACCREDITATION") AS "ACCREDITATION"
    FROM SNOWFLAKE_LEARNING_DB."ANALYTICAL_REFERENCE_STANDARDS".QUALITY_CONTROLS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('STANDARD_TYPE', 'Standard type', TRUE),
    ('ANALYTE_S_PARAMETER', 'Analyte(s) / parameter', TRUE),
    ('FORMAT', 'Format', FALSE),
    ('REGULATORY_STATUS_INTENDED_USE', 'Regulatory status / intended use', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('EXPIRATION_VALIDITY', 'Expiration / validity', FALSE),
    ('ASSAYED_OR_UNASSAYED', 'Assayed or unassayed', TRUE),
    ('LEVELS', 'Levels', TRUE),
    ('CERTIFIED_ASSIGNED_VALUE', 'Certified / assigned value', FALSE),
    ('NUMBER_OF_ANALYTES', 'Number of analytes', FALSE),
    ('MATRIX_SOLVENT', 'Matrix / solvent', FALSE),
    ('COMPATIBLE_INSTRUMENTS_ASSAYS', 'Compatible instruments / assays', FALSE),
    ('OPEN_VIAL_STABILITY', 'Open-vial stability', FALSE),
    ('SHELF_LIFE', 'Shelf life', FALSE),
    ('VARIANTS_ALLELE_FREQUENCY', 'Variants / allele frequency', FALSE),
    ('QUANTIFICATION_METHOD', 'Quantification method', FALSE),
    ('PEER_DATA_PROGRAM', 'Peer data program', FALSE),
    ('ACCREDITATION', 'Accreditation', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'analytical_reference_standards' AS ASSIGNED_L2_ID,
    'Analytical & Reference Standards' AS ASSIGNED_L2_LABEL,
    'quality_controls' AS ASSIGNED_L3_ID,
    'Quality Controls' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Analytical & Reference Standards > Quality Controls
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("STANDARD_TYPE") AS "STANDARD_TYPE",
    COUNT("ANALYTE_S_PARAMETER") AS "ANALYTE_S_PARAMETER",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("REGULATORY_STATUS_INTENDED_USE") AS "REGULATORY_STATUS_INTENDED_USE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("EXPIRATION_VALIDITY") AS "EXPIRATION_VALIDITY",
    COUNT("PROGRAM_MATRIX") AS "PROGRAM_MATRIX",
    COUNT("STUDY_SCHEDULE") AS "STUDY_SCHEDULE",
    COUNT("ACCREDITATION") AS "ACCREDITATION",
    COUNT("REGULATORY_ACCEPTANCE") AS "REGULATORY_ACCEPTANCE"
    FROM SNOWFLAKE_LEARNING_DB."ANALYTICAL_REFERENCE_STANDARDS".PROFICIENCY_TESTING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('STANDARD_TYPE', 'Standard type', TRUE),
    ('ANALYTE_S_PARAMETER', 'Analyte(s) / parameter', TRUE),
    ('FORMAT', 'Format', FALSE),
    ('REGULATORY_STATUS_INTENDED_USE', 'Regulatory status / intended use', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('EXPIRATION_VALIDITY', 'Expiration / validity', FALSE),
    ('PROGRAM_MATRIX', 'Program / matrix', TRUE),
    ('STUDY_SCHEDULE', 'Study schedule', FALSE),
    ('ACCREDITATION', 'Accreditation', FALSE),
    ('REGULATORY_ACCEPTANCE', 'Regulatory acceptance', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'analytical_reference_standards' AS ASSIGNED_L2_ID,
    'Analytical & Reference Standards' AS ASSIGNED_L2_LABEL,
    'proficiency_testing' AS ASSIGNED_L3_ID,
    'Proficiency Testing' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Analytical & Reference Standards > Proficiency Testing
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_ANIMAL_BIOSPECIMENS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("IACUC_ANIMAL_WELFARE_CONSENT") AS "IACUC_ANIMAL_WELFARE_CONSENT",
    COUNT("ANATOMICAL_SITE_TISSUE_TYPE") AS "ANATOMICAL_SITE_TISSUE_TYPE",
    COUNT("PRESERVATION") AS "PRESERVATION",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("DERIVATIVE_EXTRACT_TYPE") AS "DERIVATIVE_EXTRACT_TYPE",
    COUNT("SAMPLE_SIZE_AMOUNT") AS "SAMPLE_SIZE_AMOUNT",
    COUNT("STRAIN") AS "STRAIN",
    COUNT("DISEASE_MODEL_STATE") AS "DISEASE_MODEL_STATE",
    COUNT("SEX") AS "SEX",
    COUNT("AGE") AS "AGE",
    COUNT("INDIVIDUAL_VS_POOLED") AS "INDIVIDUAL_VS_POOLED"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_BIOSPECIMENS".ANIMAL_TISSUE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('IACUC_ANIMAL_WELFARE_CONSENT', 'IACUC / animal-welfare consent', FALSE),
    ('ANATOMICAL_SITE_TISSUE_TYPE', 'Anatomical site / tissue type', TRUE),
    ('PRESERVATION', 'Preservation', TRUE),
    ('SHIPPING_TEMPERATURE', 'Shipping Temperature', TRUE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', TRUE),
    ('DERIVATIVE_EXTRACT_TYPE', 'Derivative / extract type', FALSE),
    ('SAMPLE_SIZE_AMOUNT', 'Sample size / amount', FALSE),
    ('STRAIN', 'Strain', FALSE),
    ('DISEASE_MODEL_STATE', 'Disease / model state', FALSE),
    ('SEX', 'Sex', FALSE),
    ('AGE', 'Age', FALSE),
    ('INDIVIDUAL_VS_POOLED', 'Individual vs Pooled', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_biospecimens' AS ASSIGNED_L2_ID,
    'Animal Biospecimens' AS ASSIGNED_L2_LABEL,
    'animal_tissue' AS ASSIGNED_L3_ID,
    'Animal Tissue' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Biospecimens > Animal Tissue
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("IACUC_ANIMAL_WELFARE_CONSENT") AS "IACUC_ANIMAL_WELFARE_CONSENT",
    COUNT("MATRIX") AS "MATRIX",
    COUNT("ANTICOAGULANT_ADDITIVE") AS "ANTICOAGULANT_ADDITIVE",
    COUNT("VOLUME") AS "VOLUME",
    COUNT("PRESERVATION") AS "PRESERVATION",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("PROCESSING_FINISHING") AS "PROCESSING_FINISHING",
    COUNT("STRAIN") AS "STRAIN",
    COUNT("DISEASE_MODEL_STATE") AS "DISEASE_MODEL_STATE",
    COUNT("SEX") AS "SEX",
    COUNT("AGE") AS "AGE",
    COUNT("INDIVIDUAL_VS_POOLED") AS "INDIVIDUAL_VS_POOLED"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_BIOSPECIMENS".ANIMAL_BLOOD_BIOFLUIDS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('IACUC_ANIMAL_WELFARE_CONSENT', 'IACUC / animal-welfare consent', FALSE),
    ('MATRIX', 'Matrix', TRUE),
    ('ANTICOAGULANT_ADDITIVE', 'Anticoagulant / additive', TRUE),
    ('VOLUME', 'Volume', TRUE),
    ('PRESERVATION', 'Preservation', TRUE),
    ('SHIPPING_TEMPERATURE', 'Shipping Temperature', TRUE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', TRUE),
    ('PROCESSING_FINISHING', 'Processing / finishing', FALSE),
    ('STRAIN', 'Strain', FALSE),
    ('DISEASE_MODEL_STATE', 'Disease / model state', FALSE),
    ('SEX', 'Sex', FALSE),
    ('AGE', 'Age', FALSE),
    ('INDIVIDUAL_VS_POOLED', 'Individual vs Pooled', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_biospecimens' AS ASSIGNED_L2_ID,
    'Animal Biospecimens' AS ASSIGNED_L2_LABEL,
    'animal_blood_biofluids' AS ASSIGNED_L3_ID,
    'Animal Blood & Biofluids' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Biospecimens > Animal Blood & Biofluids
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("IACUC_ANIMAL_WELFARE_CONSENT") AS "IACUC_ANIMAL_WELFARE_CONSENT",
    COUNT("CELL_COUNT") AS "CELL_COUNT",
    COUNT("VIABILITY") AS "VIABILITY",
    COUNT("PRESERVATION") AS "PRESERVATION",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("ISOLATION_METHOD") AS "ISOLATION_METHOD",
    COUNT("STRAIN") AS "STRAIN",
    COUNT("DISEASE_MODEL_STATE") AS "DISEASE_MODEL_STATE",
    COUNT("SEX") AS "SEX",
    COUNT("AGE") AS "AGE",
    COUNT("INDIVIDUAL_VS_POOLED") AS "INDIVIDUAL_VS_POOLED"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_BIOSPECIMENS".ANIMAL_CELLS_PBMCS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('IACUC_ANIMAL_WELFARE_CONSENT', 'IACUC / animal-welfare consent', FALSE),
    ('CELL_COUNT', 'Cell count', TRUE),
    ('VIABILITY', 'Viability %', TRUE),
    ('PRESERVATION', 'Preservation', TRUE),
    ('SHIPPING_TEMPERATURE', 'Shipping Temperature', TRUE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', TRUE),
    ('ISOLATION_METHOD', 'Isolation method', FALSE),
    ('STRAIN', 'Strain', FALSE),
    ('DISEASE_MODEL_STATE', 'Disease / model state', FALSE),
    ('SEX', 'Sex', FALSE),
    ('AGE', 'Age', FALSE),
    ('INDIVIDUAL_VS_POOLED', 'Individual vs Pooled', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_biospecimens' AS ASSIGNED_L2_ID,
    'Animal Biospecimens' AS ASSIGNED_L2_LABEL,
    'animal_cells_pbmcs' AS ASSIGNED_L3_ID,
    'Animal Cells & PBMCs' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Biospecimens > Animal Cells & PBMCs
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("IACUC_ANIMAL_WELFARE_CONSENT") AS "IACUC_ANIMAL_WELFARE_CONSENT",
    COUNT("VIABILITY") AS "VIABILITY",
    COUNT("PRESERVATION") AS "PRESERVATION",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("PASSAGE_NUMBER") AS "PASSAGE_NUMBER",
    COUNT("STRAIN") AS "STRAIN",
    COUNT("DISEASE_MODEL_STATE") AS "DISEASE_MODEL_STATE",
    COUNT("SEX") AS "SEX",
    COUNT("AGE") AS "AGE",
    COUNT("INDIVIDUAL_VS_POOLED") AS "INDIVIDUAL_VS_POOLED"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_BIOSPECIMENS".ANIMAL_PRIMARY_CELLS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('IACUC_ANIMAL_WELFARE_CONSENT', 'IACUC / animal-welfare consent', FALSE),
    ('VIABILITY', 'Viability %', TRUE),
    ('PRESERVATION', 'Preservation', TRUE),
    ('SHIPPING_TEMPERATURE', 'Shipping Temperature', TRUE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', TRUE),
    ('PASSAGE_NUMBER', 'Passage number', FALSE),
    ('STRAIN', 'Strain', FALSE),
    ('DISEASE_MODEL_STATE', 'Disease / model state', FALSE),
    ('SEX', 'Sex', FALSE),
    ('AGE', 'Age', FALSE),
    ('INDIVIDUAL_VS_POOLED', 'Individual vs Pooled', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_biospecimens' AS ASSIGNED_L2_ID,
    'Animal Biospecimens' AS ASSIGNED_L2_LABEL,
    'animal_primary_cells' AS ASSIGNED_L3_ID,
    'Animal Primary Cells' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Biospecimens > Animal Primary Cells
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("IACUC_ANIMAL_WELFARE_CONSENT") AS "IACUC_ANIMAL_WELFARE_CONSENT"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_BIOSPECIMENS".ANIMAL_STEM_CELLS_IPSCS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('IACUC_ANIMAL_WELFARE_CONSENT', 'IACUC / animal-welfare consent', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_biospecimens' AS ASSIGNED_L2_ID,
    'Animal Biospecimens' AS ASSIGNED_L2_LABEL,
    'animal_stem_cells_ipscs' AS ASSIGNED_L3_ID,
    'Animal Stem Cells / iPSCs' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Biospecimens > Animal Stem Cells / iPSCs
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("IACUC_ANIMAL_WELFARE_CONSENT") AS "IACUC_ANIMAL_WELFARE_CONSENT",
    COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_BIOSPECIMENS".ANIMAL_CUSTOM_PROSPECTIVE_COLLECTION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('IACUC_ANIMAL_WELFARE_CONSENT', 'IACUC / animal-welfare consent', FALSE),
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_biospecimens' AS ASSIGNED_L2_ID,
    'Animal Biospecimens' AS ASSIGNED_L2_LABEL,
    'animal_custom_prospective_collection' AS ASSIGNED_L3_ID,
    'Animal Custom / prospective collection' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Biospecimens > Animal Custom / prospective collection
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("IACUC_ANIMAL_WELFARE_CONSENT") AS "IACUC_ANIMAL_WELFARE_CONSENT",
    COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_BIOSPECIMENS".ANIMAL_BIOBANKING_STORAGE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('IACUC_ANIMAL_WELFARE_CONSENT', 'IACUC / animal-welfare consent', FALSE),
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_biospecimens' AS ASSIGNED_L2_ID,
    'Animal Biospecimens' AS ASSIGNED_L2_LABEL,
    'animal_biobanking_storage' AS ASSIGNED_L3_ID,
    'Animal Biobanking & storage' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Biospecimens > Animal Biobanking & storage
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("IACUC_ANIMAL_WELFARE_CONSENT") AS "IACUC_ANIMAL_WELFARE_CONSENT",
    COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_BIOSPECIMENS".ANIMAL_SAMPLE_PROCESSING_HISTOLOGY_FFPE_DISSOCIATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('IACUC_ANIMAL_WELFARE_CONSENT', 'IACUC / animal-welfare consent', FALSE),
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_biospecimens' AS ASSIGNED_L2_ID,
    'Animal Biospecimens' AS ASSIGNED_L2_LABEL,
    'animal_sample_processing_histology_ffpe_dissociation' AS ASSIGNED_L3_ID,
    'Animal Sample processing (histology / FFPE / dissociation)' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Biospecimens > Animal Sample processing (histology / FFPE / dissociation)
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_ANIMAL_MODELS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("RESEARCH_THERAPEUTIC_AREA") AS "RESEARCH_THERAPEUTIC_AREA",
    COUNT("GENOTYPE_ALLELES") AS "GENOTYPE_ALLELES",
    COUNT("GENETIC_BACKGROUND") AS "GENETIC_BACKGROUND",
    COUNT("SUPPLIER_STOCK_STRAIN_CODE") AS "SUPPLIER_STOCK_STRAIN_CODE",
    COUNT("HEALTH_STATUS_MICROBIOLOGICAL_STANDARD") AS "HEALTH_STATUS_MICROBIOLOGICAL_STANDARD",
    COUNT("STRAIN_OFFICIAL_NOMENCLATURE") AS "STRAIN_OFFICIAL_NOMENCLATURE",
    COUNT("MODEL_TYPE_GENETIC_STATUS") AS "MODEL_TYPE_GENETIC_STATUS",
    COUNT("DELIVERY_FORM") AS "DELIVERY_FORM",
    COUNT("ZYGOSITY") AS "ZYGOSITY",
    COUNT("MHC_HAPLOTYPE") AS "MHC_HAPLOTYPE",
    COUNT("COAT_COLOR") AS "COAT_COLOR",
    COUNT("HUMANIZATION_ENGRAFTMENT") AS "HUMANIZATION_ENGRAFTMENT",
    COUNT("PRECONDITIONING") AS "PRECONDITIONING",
    COUNT("KNOWN_BACKGROUND_TRAITS") AS "KNOWN_BACKGROUND_TRAITS",
    COUNT("TUMOR_MODEL_TYPE") AS "TUMOR_MODEL_TYPE",
    COUNT("CANCER_TYPE_INDICATION") AS "CANCER_TYPE_INDICATION",
    COUNT("SEX") AS "SEX",
    COUNT("AGE_WEIGHT_RANGE") AS "AGE_WEIGHT_RANGE",
    COUNT("PRODUCTION_LOCATION_ORIGIN") AS "PRODUCTION_LOCATION_ORIGIN",
    COUNT("AVAILABILITY_STATUS") AS "AVAILABILITY_STATUS",
    COUNT("LICENSING_CONDITIONS_OF_USE") AS "LICENSING_CONDITIONS_OF_USE"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_MODELS".RODENT_MODELS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('RESEARCH_THERAPEUTIC_AREA', 'Research / therapeutic area', FALSE),
    ('GENOTYPE_ALLELES', 'Genotype / alleles', TRUE),
    ('GENETIC_BACKGROUND', 'Genetic background', TRUE),
    ('SUPPLIER_STOCK_STRAIN_CODE', 'Supplier stock / strain code', TRUE),
    ('HEALTH_STATUS_MICROBIOLOGICAL_STANDARD', 'Health status / microbiological standard', TRUE),
    ('STRAIN_OFFICIAL_NOMENCLATURE', 'Strain / official nomenclature', TRUE),
    ('MODEL_TYPE_GENETIC_STATUS', 'Model type / genetic status', TRUE),
    ('DELIVERY_FORM', 'Delivery form', TRUE),
    ('ZYGOSITY', 'Zygosity', FALSE),
    ('MHC_HAPLOTYPE', 'MHC haplotype', FALSE),
    ('COAT_COLOR', 'Coat color', FALSE),
    ('HUMANIZATION_ENGRAFTMENT', 'Humanization / engraftment', FALSE),
    ('PRECONDITIONING', 'Preconditioning', FALSE),
    ('KNOWN_BACKGROUND_TRAITS', 'Known background traits', FALSE),
    ('TUMOR_MODEL_TYPE', 'Tumor model type', FALSE),
    ('CANCER_TYPE_INDICATION', 'Cancer type / indication', FALSE),
    ('SEX', 'Sex', FALSE),
    ('AGE_WEIGHT_RANGE', 'Age / weight range', FALSE),
    ('PRODUCTION_LOCATION_ORIGIN', 'Production location / origin', FALSE),
    ('AVAILABILITY_STATUS', 'Availability status', FALSE),
    ('LICENSING_CONDITIONS_OF_USE', 'Licensing / conditions of use', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_models' AS ASSIGNED_L2_ID,
    'Animal Models' AS ASSIGNED_L2_LABEL,
    'rodent_models' AS ASSIGNED_L3_ID,
    'Rodent Models' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Models > Rodent Models
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("RESEARCH_THERAPEUTIC_AREA") AS "RESEARCH_THERAPEUTIC_AREA",
    COUNT("LINE_STRAIN_NAME") AS "LINE_STRAIN_NAME",
    COUNT("MUTANT_ALLELE_DESIGNATION") AS "MUTANT_ALLELE_DESIGNATION",
    COUNT("GENE_SYMBOL_LOCUS") AS "GENE_SYMBOL_LOCUS",
    COUNT("SHIPPING_FORM") AS "SHIPPING_FORM",
    COUNT("MODEL_TYPE_GENETIC_STATUS") AS "MODEL_TYPE_GENETIC_STATUS",
    COUNT("ZYGOSITY") AS "ZYGOSITY",
    COUNT("ADULT_AGE") AS "ADULT_AGE",
    COUNT("SEX") AS "SEX",
    COUNT("PRODUCTION_LOCATION_ORIGIN") AS "PRODUCTION_LOCATION_ORIGIN",
    COUNT("AVAILABILITY_STATUS") AS "AVAILABILITY_STATUS",
    COUNT("LICENSING_CONDITIONS_OF_USE") AS "LICENSING_CONDITIONS_OF_USE"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_MODELS".ZEBRAFISH_MODELS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('RESEARCH_THERAPEUTIC_AREA', 'Research / therapeutic area', FALSE),
    ('LINE_STRAIN_NAME', 'Line / strain name', TRUE),
    ('MUTANT_ALLELE_DESIGNATION', 'Mutant allele designation', TRUE),
    ('GENE_SYMBOL_LOCUS', 'Gene symbol / locus', TRUE),
    ('SHIPPING_FORM', 'Shipping form', TRUE),
    ('MODEL_TYPE_GENETIC_STATUS', 'Model type / genetic status', TRUE),
    ('ZYGOSITY', 'Zygosity', FALSE),
    ('ADULT_AGE', 'Adult age', FALSE),
    ('SEX', 'Sex', FALSE),
    ('PRODUCTION_LOCATION_ORIGIN', 'Production location / origin', FALSE),
    ('AVAILABILITY_STATUS', 'Availability status', FALSE),
    ('LICENSING_CONDITIONS_OF_USE', 'Licensing / conditions of use', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_models' AS ASSIGNED_L2_ID,
    'Animal Models' AS ASSIGNED_L2_LABEL,
    'zebrafish_models' AS ASSIGNED_L3_ID,
    'Zebrafish Models' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Models > Zebrafish Models
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("RESEARCH_THERAPEUTIC_AREA") AS "RESEARCH_THERAPEUTIC_AREA",
    COUNT("GEOGRAPHIC_ORIGIN") AS "GEOGRAPHIC_ORIGIN",
    COUNT("SPF_STATUS") AS "SPF_STATUS",
    COUNT("DELIVERY_FORM") AS "DELIVERY_FORM",
    COUNT("SEX_SEXUAL_MATURITY") AS "SEX_SEXUAL_MATURITY",
    COUNT("HOUSING") AS "HOUSING",
    COUNT("AGE_WEIGHT") AS "AGE_WEIGHT",
    COUNT("IMPORT_CITES_DOCUMENTATION") AS "IMPORT_CITES_DOCUMENTATION",
    COUNT("ADD_ON_PROCEDURES") AS "ADD_ON_PROCEDURES",
    COUNT("AVAILABILITY_STATUS") AS "AVAILABILITY_STATUS",
    COUNT("LICENSING_CONDITIONS_OF_USE") AS "LICENSING_CONDITIONS_OF_USE"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_MODELS".NHP_MODELS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('RESEARCH_THERAPEUTIC_AREA', 'Research / therapeutic area', FALSE),
    ('GEOGRAPHIC_ORIGIN', 'Geographic origin', TRUE),
    ('SPF_STATUS', 'SPF status', TRUE),
    ('DELIVERY_FORM', 'Delivery form', TRUE),
    ('SEX_SEXUAL_MATURITY', 'Sex / sexual maturity', FALSE),
    ('HOUSING', 'Housing', FALSE),
    ('AGE_WEIGHT', 'Age / weight', FALSE),
    ('IMPORT_CITES_DOCUMENTATION', 'Import / CITES documentation', FALSE),
    ('ADD_ON_PROCEDURES', 'Add-on procedures', FALSE),
    ('AVAILABILITY_STATUS', 'Availability status', FALSE),
    ('LICENSING_CONDITIONS_OF_USE', 'Licensing / conditions of use', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_models' AS ASSIGNED_L2_ID,
    'Animal Models' AS ASSIGNED_L2_LABEL,
    'nhp_models' AS ASSIGNED_L3_ID,
    'Non-Human Primate Models' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Models > Non-Human Primate Models
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("RESEARCH_THERAPEUTIC_AREA") AS "RESEARCH_THERAPEUTIC_AREA",
    COUNT("STOCK_STRAIN_NUMBER") AS "STOCK_STRAIN_NUMBER",
    COUNT("GENOTYPE") AS "GENOTYPE",
    COUNT("MODEL_TYPE_GENETIC_STATUS") AS "MODEL_TYPE_GENETIC_STATUS",
    COUNT("DELIVERY_FORM") AS "DELIVERY_FORM",
    COUNT("CHROMOSOMES_AFFECTED") AS "CHROMOSOMES_AFFECTED",
    COUNT("BALANCER") AS "BALANCER",
    COUNT("MUTAGEN") AS "MUTAGEN",
    COUNT("TIMES_OUTCROSSED") AS "TIMES_OUTCROSSED",
    COUNT("DONOR_MADE_BY") AS "DONOR_MADE_BY",
    COUNT("PRODUCTION_LOCATION_ORIGIN") AS "PRODUCTION_LOCATION_ORIGIN",
    COUNT("AVAILABILITY_STATUS") AS "AVAILABILITY_STATUS",
    COUNT("LICENSING_CONDITIONS_OF_USE") AS "LICENSING_CONDITIONS_OF_USE"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_MODELS".INVERTEBRATE_LOWER_ORGANISM_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('RESEARCH_THERAPEUTIC_AREA', 'Research / therapeutic area', FALSE),
    ('STOCK_STRAIN_NUMBER', 'Stock / strain number', TRUE),
    ('GENOTYPE', 'Genotype', TRUE),
    ('MODEL_TYPE_GENETIC_STATUS', 'Model type / genetic status', TRUE),
    ('DELIVERY_FORM', 'Delivery form', TRUE),
    ('CHROMOSOMES_AFFECTED', 'Chromosomes affected', FALSE),
    ('BALANCER', 'Balancer', FALSE),
    ('MUTAGEN', 'Mutagen', FALSE),
    ('TIMES_OUTCROSSED', 'Times outcrossed', FALSE),
    ('DONOR_MADE_BY', 'Donor / made by', FALSE),
    ('PRODUCTION_LOCATION_ORIGIN', 'Production location / origin', FALSE),
    ('AVAILABILITY_STATUS', 'Availability status', FALSE),
    ('LICENSING_CONDITIONS_OF_USE', 'Licensing / conditions of use', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_models' AS ASSIGNED_L2_ID,
    'Animal Models' AS ASSIGNED_L2_LABEL,
    'invertebrate_lower_organism' AS ASSIGNED_L3_ID,
    'Invertebrate & Lower Organism Models' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Models > Invertebrate & Lower Organism Models
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("RESEARCH_THERAPEUTIC_AREA") AS "RESEARCH_THERAPEUTIC_AREA",
    COUNT("SUPPLIER_STOCK_STRAIN_CODE") AS "SUPPLIER_STOCK_STRAIN_CODE",
    COUNT("HEALTH_STATUS_MICROBIOLOGICAL_STANDARD") AS "HEALTH_STATUS_MICROBIOLOGICAL_STANDARD",
    COUNT("STRAIN_OFFICIAL_NOMENCLATURE") AS "STRAIN_OFFICIAL_NOMENCLATURE",
    COUNT("MODEL_TYPE_GENETIC_STATUS") AS "MODEL_TYPE_GENETIC_STATUS",
    COUNT("DELIVERY_FORM") AS "DELIVERY_FORM",
    COUNT("SEX") AS "SEX",
    COUNT("AGE_WEIGHT_RANGE") AS "AGE_WEIGHT_RANGE",
    COUNT("PRODUCTION_LOCATION_ORIGIN") AS "PRODUCTION_LOCATION_ORIGIN",
    COUNT("AVAILABILITY_STATUS") AS "AVAILABILITY_STATUS",
    COUNT("LICENSING_CONDITIONS_OF_USE") AS "LICENSING_CONDITIONS_OF_USE"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_MODELS".OTHER_MAMMALIAN_MODELS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('RESEARCH_THERAPEUTIC_AREA', 'Research / therapeutic area', FALSE),
    ('SUPPLIER_STOCK_STRAIN_CODE', 'Supplier stock / strain code', TRUE),
    ('HEALTH_STATUS_MICROBIOLOGICAL_STANDARD', 'Health status / microbiological standard', TRUE),
    ('STRAIN_OFFICIAL_NOMENCLATURE', 'Strain / official nomenclature', TRUE),
    ('MODEL_TYPE_GENETIC_STATUS', 'Model type / genetic status', TRUE),
    ('DELIVERY_FORM', 'Delivery form', TRUE),
    ('SEX', 'Sex', FALSE),
    ('AGE_WEIGHT_RANGE', 'Age / weight range', FALSE),
    ('PRODUCTION_LOCATION_ORIGIN', 'Production location / origin', FALSE),
    ('AVAILABILITY_STATUS', 'Availability status', FALSE),
    ('LICENSING_CONDITIONS_OF_USE', 'Licensing / conditions of use', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_models' AS ASSIGNED_L2_ID,
    'Animal Models' AS ASSIGNED_L2_LABEL,
    'other_mammalian_models' AS ASSIGNED_L3_ID,
    'Other Mammalian Models' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Models > Other Mammalian Models
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("RESEARCH_THERAPEUTIC_AREA") AS "RESEARCH_THERAPEUTIC_AREA",
    COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_MODELS".COLONY_BREEDING_MANAGEMENT_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('RESEARCH_THERAPEUTIC_AREA', 'Research / therapeutic area', FALSE),
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_models' AS ASSIGNED_L2_ID,
    'Animal Models' AS ASSIGNED_L2_LABEL,
    'colony_breeding_management' AS ASSIGNED_L3_ID,
    'Colony & breeding management' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Models > Colony & breeding management
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("RESEARCH_THERAPEUTIC_AREA") AS "RESEARCH_THERAPEUTIC_AREA",
    COUNT("STUDY_TYPE") AS "STUDY_TYPE",
    COUNT("TUMOR_MODEL_TYPE_EFFICACY") AS "TUMOR_MODEL_TYPE_EFFICACY",
    COUNT("TUMOR_IMPLANTATION_SITE") AS "TUMOR_IMPLANTATION_SITE",
    COUNT("CANCER_TYPE_EFFICACY") AS "CANCER_TYPE_EFFICACY",
    COUNT("NUMBER_OF_MODELS_AVAILABLE") AS "NUMBER_OF_MODELS_AVAILABLE",
    COUNT("TREATMENT_RESISTANT_OR_PRETREATED_MODELS") AS "TREATMENT_RESISTANT_OR_PRETREATED_MODELS",
    COUNT("DISEASE_MODEL_NAME") AS "DISEASE_MODEL_NAME",
    COUNT("EFFICACY_ENDPOINTS") AS "EFFICACY_ENDPOINTS",
    COUNT("IN_LIFE_IMAGING_MODALITY") AS "IN_LIFE_IMAGING_MODALITY",
    COUNT("EX_VIVO_READOUTS") AS "EX_VIVO_READOUTS",
    COUNT("PK_SAMPLING_AND_BIOANALYSIS_AVAILABLE") AS "PK_SAMPLING_AND_BIOANALYSIS_AVAILABLE",
    COUNT("PHARMACODYNAMIC_READOUT") AS "PHARMACODYNAMIC_READOUT",
    COUNT("THERAPEUTIC_MODALITY_SUPPORTED") AS "THERAPEUTIC_MODALITY_SUPPORTED",
    COUNT("MODEL_CHARACTERIZATION_DATA_PROVIDED") AS "MODEL_CHARACTERIZATION_DATA_PROVIDED",
    COUNT("DOSING_ROUTE") AS "DOSING_ROUTE",
    COUNT("HOST_STRAIN") AS "HOST_STRAIN",
    COUNT("HUMANIZED_IMMUNE_SYSTEM_SOURCE") AS "HUMANIZED_IMMUNE_SYSTEM_SOURCE",
    COUNT("NON_TUMOR_FUNCTIONAL_ENDPOINTS") AS "NON_TUMOR_FUNCTIONAL_ENDPOINTS",
    COUNT("IN_LIFE_DURATION") AS "IN_LIFE_DURATION",
    COUNT("GLP_STATUS_EFFICACY") AS "GLP_STATUS_EFFICACY"
    FROM SNOWFLAKE_LEARNING_DB."ANIMAL_MODELS".IN_VIVO_PHARMACOLOGY_EFFICACY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('RESEARCH_THERAPEUTIC_AREA', 'Research / therapeutic area', FALSE),
    ('STUDY_TYPE', 'Study type', TRUE),
    ('TUMOR_MODEL_TYPE_EFFICACY', 'Tumor model type (efficacy)', FALSE),
    ('TUMOR_IMPLANTATION_SITE', 'Tumor implantation site', FALSE),
    ('CANCER_TYPE_EFFICACY', 'Cancer type (efficacy)', FALSE),
    ('NUMBER_OF_MODELS_AVAILABLE', 'Number of models available', FALSE),
    ('TREATMENT_RESISTANT_OR_PRETREATED_MODELS', 'Treatment-resistant or pretreated models', FALSE),
    ('DISEASE_MODEL_NAME', 'Disease model name', FALSE),
    ('EFFICACY_ENDPOINTS', 'Efficacy endpoints', FALSE),
    ('IN_LIFE_IMAGING_MODALITY', 'In-life imaging modality', FALSE),
    ('EX_VIVO_READOUTS', 'Ex vivo readouts', FALSE),
    ('PK_SAMPLING_AND_BIOANALYSIS_AVAILABLE', 'PK sampling and bioanalysis available', FALSE),
    ('PHARMACODYNAMIC_READOUT', 'Pharmacodynamic readout', FALSE),
    ('THERAPEUTIC_MODALITY_SUPPORTED', 'Therapeutic modality supported', FALSE),
    ('MODEL_CHARACTERIZATION_DATA_PROVIDED', 'Model characterization data provided', FALSE),
    ('DOSING_ROUTE', 'Dosing route', FALSE),
    ('HOST_STRAIN', 'Host strain', FALSE),
    ('HUMANIZED_IMMUNE_SYSTEM_SOURCE', 'Humanized immune system source', FALSE),
    ('NON_TUMOR_FUNCTIONAL_ENDPOINTS', 'Non-tumor functional endpoints', FALSE),
    ('IN_LIFE_DURATION', 'In-life duration', FALSE),
    ('GLP_STATUS_EFFICACY', 'GLP status (efficacy)', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'animal_models' AS ASSIGNED_L2_ID,
    'Animal Models' AS ASSIGNED_L2_LABEL,
    'in_vivo_pharmacology_efficacy' AS ASSIGNED_L3_ID,
    'In vivo pharmacology / efficacy' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Animal Models > In vivo pharmacology / efficacy
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_ANTIBODIES_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("TARGET_ANTIGEN") AS "TARGET_ANTIGEN",
    COUNT("GENE_ID") AS "GENE_ID",
    COUNT("UNIPROT_ACCESSION") AS "UNIPROT_ACCESSION",
    COUNT("TARGET_GENE_SYMBOL") AS "TARGET_GENE_SYMBOL",
    COUNT("TARGET_ALTERNATE_NAMES") AS "TARGET_ALTERNATE_NAMES",
    COUNT("TESTED_REACTIVITY") AS "TESTED_REACTIVITY",
    COUNT("HOST_SPECIES") AS "HOST_SPECIES",
    COUNT("CLONALITY") AS "CLONALITY",
    COUNT("APPLICATION") AS "APPLICATION",
    COUNT("CITED_REACTIVITY") AS "CITED_REACTIVITY",
    COUNT("PTM_MODIFICATION_SPECIFICITY") AS "PTM_MODIFICATION_SPECIFICITY",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("PURIFICATION") AS "PURIFICATION",
    COUNT("CLONE_ID") AS "CLONE_ID",
    COUNT("ISOTYPE") AS "ISOTYPE",
    COUNT("CONJUGATE_LABEL") AS "CONJUGATE_LABEL",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("RRID") AS "RRID",
    COUNT("RECOMMENDED_DILUTION") AS "RECOMMENDED_DILUTION"
    FROM SNOWFLAKE_LEARNING_DB."ANTIBODIES".PRIMARY_ANTIBODIES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('TARGET_ANTIGEN', 'Target / Antigen', TRUE),
    ('GENE_ID', 'Gene ID', FALSE),
    ('UNIPROT_ACCESSION', 'UniProt accession', FALSE),
    ('TARGET_GENE_SYMBOL', 'Target gene symbol', FALSE),
    ('TARGET_ALTERNATE_NAMES', 'Target alternate names', FALSE),
    ('TESTED_REACTIVITY', 'Tested reactivity', TRUE),
    ('HOST_SPECIES', 'Host Species', TRUE),
    ('CLONALITY', 'Clonality', TRUE),
    ('APPLICATION', 'Application', TRUE),
    ('CITED_REACTIVITY', 'Cited reactivity', FALSE),
    ('PTM_MODIFICATION_SPECIFICITY', 'PTM / modification specificity', FALSE),
    ('FORMAT', 'Format', FALSE),
    ('PURIFICATION', 'Purification', FALSE),
    ('CLONE_ID', 'Clone ID', FALSE),
    ('ISOTYPE', 'Isotype', FALSE),
    ('CONJUGATE_LABEL', 'Conjugate / Label', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('RRID', 'RRID', FALSE),
    ('RECOMMENDED_DILUTION', 'Recommended dilution', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'antibodies' AS ASSIGNED_L2_ID,
    'Antibodies' AS ASSIGNED_L2_LABEL,
    'primary_antibodies' AS ASSIGNED_L3_ID,
    'Primary Antibodies' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Antibodies > Primary Antibodies
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("TARGET_SPECIES") AS "TARGET_SPECIES",
    COUNT("SPECIFICITY") AS "SPECIFICITY",
    COUNT("CONJUGATE") AS "CONJUGATE",
    COUNT("HOST_SPECIES") AS "HOST_SPECIES",
    COUNT("CLONALITY") AS "CLONALITY",
    COUNT("APPLICATION") AS "APPLICATION",
    COUNT("CROSS_ADSORBED") AS "CROSS_ADSORBED",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("PURIFICATION") AS "PURIFICATION",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("RRID") AS "RRID",
    COUNT("RECOMMENDED_DILUTION") AS "RECOMMENDED_DILUTION"
    FROM SNOWFLAKE_LEARNING_DB."ANTIBODIES".SECONDARY_ANTIBODIES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('TARGET_SPECIES', 'Target species', TRUE),
    ('SPECIFICITY', 'Specificity', TRUE),
    ('CONJUGATE', 'Conjugate', TRUE),
    ('HOST_SPECIES', 'Host Species', TRUE),
    ('CLONALITY', 'Clonality', TRUE),
    ('APPLICATION', 'Application', TRUE),
    ('CROSS_ADSORBED', 'Cross-adsorbed', FALSE),
    ('FORMAT', 'Format', FALSE),
    ('PURIFICATION', 'Purification', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('RRID', 'RRID', FALSE),
    ('RECOMMENDED_DILUTION', 'Recommended dilution', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'antibodies' AS ASSIGNED_L2_ID,
    'Antibodies' AS ASSIGNED_L2_LABEL,
    'secondary_antibodies' AS ASSIGNED_L3_ID,
    'Secondary Antibodies' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Antibodies > Secondary Antibodies
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("TARGET_ANTIGEN") AS "TARGET_ANTIGEN",
    COUNT("GENE_ID") AS "GENE_ID",
    COUNT("UNIPROT_ACCESSION") AS "UNIPROT_ACCESSION",
    COUNT("TARGET_GENE_SYMBOL") AS "TARGET_GENE_SYMBOL",
    COUNT("TARGET_ALTERNATE_NAMES") AS "TARGET_ALTERNATE_NAMES",
    COUNT("TESTED_REACTIVITY") AS "TESTED_REACTIVITY",
    COUNT("HOST_SPECIES") AS "HOST_SPECIES",
    COUNT("CLONALITY") AS "CLONALITY",
    COUNT("APPLICATION") AS "APPLICATION",
    COUNT("CONJUGATE_LABEL") AS "CONJUGATE_LABEL",
    COUNT("CITED_REACTIVITY") AS "CITED_REACTIVITY",
    COUNT("PTM_MODIFICATION_SPECIFICITY") AS "PTM_MODIFICATION_SPECIFICITY",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("PURIFICATION") AS "PURIFICATION",
    COUNT("CLONE_ID") AS "CLONE_ID",
    COUNT("ISOTYPE") AS "ISOTYPE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("RRID") AS "RRID",
    COUNT("RECOMMENDED_DILUTION") AS "RECOMMENDED_DILUTION"
    FROM SNOWFLAKE_LEARNING_DB."ANTIBODIES".CONJUGATED_ANTIBODIES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('TARGET_ANTIGEN', 'Target / Antigen', TRUE),
    ('GENE_ID', 'Gene ID', FALSE),
    ('UNIPROT_ACCESSION', 'UniProt accession', FALSE),
    ('TARGET_GENE_SYMBOL', 'Target gene symbol', FALSE),
    ('TARGET_ALTERNATE_NAMES', 'Target alternate names', FALSE),
    ('TESTED_REACTIVITY', 'Tested reactivity', TRUE),
    ('HOST_SPECIES', 'Host Species', TRUE),
    ('CLONALITY', 'Clonality', TRUE),
    ('APPLICATION', 'Application', TRUE),
    ('CONJUGATE_LABEL', 'Conjugate / Label', FALSE),
    ('CITED_REACTIVITY', 'Cited reactivity', FALSE),
    ('PTM_MODIFICATION_SPECIFICITY', 'PTM / modification specificity', FALSE),
    ('FORMAT', 'Format', FALSE),
    ('PURIFICATION', 'Purification', FALSE),
    ('CLONE_ID', 'Clone ID', FALSE),
    ('ISOTYPE', 'Isotype', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('RRID', 'RRID', FALSE),
    ('RECOMMENDED_DILUTION', 'Recommended dilution', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'antibodies' AS ASSIGNED_L2_ID,
    'Antibodies' AS ASSIGNED_L2_LABEL,
    'conjugated_antibodies' AS ASSIGNED_L3_ID,
    'Conjugated Antibodies' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Antibodies > Conjugated Antibodies
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("ISOTYPE") AS "ISOTYPE",
    COUNT("HOST_SPECIES") AS "HOST_SPECIES",
    COUNT("CLONALITY") AS "CLONALITY",
    COUNT("APPLICATION") AS "APPLICATION",
    COUNT("CONJUGATE") AS "CONJUGATE",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("PURIFICATION") AS "PURIFICATION",
    COUNT("CLONE_ID") AS "CLONE_ID",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("RRID") AS "RRID",
    COUNT("RECOMMENDED_DILUTION") AS "RECOMMENDED_DILUTION"
    FROM SNOWFLAKE_LEARNING_DB."ANTIBODIES".ISOTYPE_CONTROLS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('ISOTYPE', 'Isotype', TRUE),
    ('HOST_SPECIES', 'Host Species', TRUE),
    ('CLONALITY', 'Clonality', TRUE),
    ('APPLICATION', 'Application', TRUE),
    ('CONJUGATE', 'Conjugate', FALSE),
    ('FORMAT', 'Format', FALSE),
    ('PURIFICATION', 'Purification', FALSE),
    ('CLONE_ID', 'Clone ID', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('RRID', 'RRID', FALSE),
    ('RECOMMENDED_DILUTION', 'Recommended dilution', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'antibodies' AS ASSIGNED_L2_ID,
    'Antibodies' AS ASSIGNED_L2_LABEL,
    'isotype_controls' AS ASSIGNED_L3_ID,
    'Isotype Controls' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Antibodies > Isotype Controls
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("FRAGMENT_TYPE") AS "FRAGMENT_TYPE",
    COUNT("TARGET_ANTIGEN") AS "TARGET_ANTIGEN",
    COUNT("GENE_ID") AS "GENE_ID",
    COUNT("UNIPROT_ACCESSION") AS "UNIPROT_ACCESSION",
    COUNT("TARGET_GENE_SYMBOL") AS "TARGET_GENE_SYMBOL",
    COUNT("TARGET_ALTERNATE_NAMES") AS "TARGET_ALTERNATE_NAMES",
    COUNT("TESTED_REACTIVITY") AS "TESTED_REACTIVITY",
    COUNT("HOST_SPECIES") AS "HOST_SPECIES",
    COUNT("CLONALITY") AS "CLONALITY",
    COUNT("APPLICATION") AS "APPLICATION",
    COUNT("CITED_REACTIVITY") AS "CITED_REACTIVITY",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("PURIFICATION") AS "PURIFICATION",
    COUNT("ISOTYPE") AS "ISOTYPE",
    COUNT("CONJUGATE_LABEL") AS "CONJUGATE_LABEL",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("RRID") AS "RRID",
    COUNT("RECOMMENDED_DILUTION") AS "RECOMMENDED_DILUTION"
    FROM SNOWFLAKE_LEARNING_DB."ANTIBODIES".ANTIBODY_FRAGMENTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('FRAGMENT_TYPE', 'Fragment type', TRUE),
    ('TARGET_ANTIGEN', 'Target / Antigen', TRUE),
    ('GENE_ID', 'Gene ID', FALSE),
    ('UNIPROT_ACCESSION', 'UniProt accession', FALSE),
    ('TARGET_GENE_SYMBOL', 'Target gene symbol', FALSE),
    ('TARGET_ALTERNATE_NAMES', 'Target alternate names', FALSE),
    ('TESTED_REACTIVITY', 'Tested reactivity', TRUE),
    ('HOST_SPECIES', 'Host Species', TRUE),
    ('CLONALITY', 'Clonality', TRUE),
    ('APPLICATION', 'Application', TRUE),
    ('CITED_REACTIVITY', 'Cited reactivity', FALSE),
    ('FORMAT', 'Format', FALSE),
    ('PURIFICATION', 'Purification', FALSE),
    ('ISOTYPE', 'Isotype', FALSE),
    ('CONJUGATE_LABEL', 'Conjugate / Label', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('RRID', 'RRID', FALSE),
    ('RECOMMENDED_DILUTION', 'Recommended dilution', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'antibodies' AS ASSIGNED_L2_ID,
    'Antibodies' AS ASSIGNED_L2_LABEL,
    'antibody_fragments' AS ASSIGNED_L3_ID,
    'Antibody Fragments' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Antibodies > Antibody Fragments
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."ANTIBODIES".ANTIBODY_VALIDATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'antibodies' AS ASSIGNED_L2_ID,
    'Antibodies' AS ASSIGNED_L2_LABEL,
    'antibody_validation' AS ASSIGNED_L3_ID,
    'Antibody validation' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Antibodies > Antibody validation
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_BIOANALYSIS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("REGULATORY_COMPLIANCE_LEVEL") AS "REGULATORY_COMPLIANCE_LEVEL",
    COUNT("MOLECULE_MODALITY") AS "MOLECULE_MODALITY",
    COUNT("BIOANALYTICAL_GUIDANCE_FOLLOWED") AS "BIOANALYTICAL_GUIDANCE_FOLLOWED",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("SAMPLE_MATRIX") AS "SAMPLE_MATRIX",
    COUNT("SERVICE_SCOPE") AS "SERVICE_SCOPE",
    COUNT("AUTOMATED_SAMPLE_PREPARATION") AS "AUTOMATED_SAMPLE_PREPARATION",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("VALIDATION_TYPE") AS "VALIDATION_TYPE",
    COUNT("VALIDATION_PARAMETERS") AS "VALIDATION_PARAMETERS",
    COUNT("METHOD_ORIGIN") AS "METHOD_ORIGIN",
    COUNT("VALIDATION_DELIVERABLES") AS "VALIDATION_DELIVERABLES",
    COUNT("REGULATORY_SUBMISSION_SUPPORTED") AS "REGULATORY_SUBMISSION_SUPPORTED",
    COUNT("THERAPEUTIC_AREA_EXPERIENCE") AS "THERAPEUTIC_AREA_EXPERIENCE",
    COUNT("REAL_TIME_DATA_PORTAL") AS "REAL_TIME_DATA_PORTAL",
    COUNT("VALIDATED_ASSAY_LIBRARY_SIZE") AS "VALIDATED_ASSAY_LIBRARY_SIZE",
    COUNT("FEASIBILITY_ASSESSMENT") AS "FEASIBILITY_ASSESSMENT",
    COUNT("METHOD_DEVELOPMENT_TIME") AS "METHOD_DEVELOPMENT_TIME",
    COUNT("MATRIX_LOTS_FOR_MATRIX_EFFECT") AS "MATRIX_LOTS_FOR_MATRIX_EFFECT",
    COUNT("CALIBRATION_STANDARD_LEVELS") AS "CALIBRATION_STANDARD_LEVELS",
    COUNT("ANALYTICAL_PLATFORM") AS "ANALYTICAL_PLATFORM",
    COUNT("MASS_SPECTROMETER_MODEL") AS "MASS_SPECTROMETER_MODEL",
    COUNT("ANALYTE_CLASS") AS "ANALYTE_CLASS",
    COUNT("STUDY_TYPE") AS "STUDY_TYPE",
    COUNT("MICROSAMPLING_FORMAT") AS "MICROSAMPLING_FORMAT",
    COUNT("IONIZATION_MODE") AS "IONIZATION_MODE",
    COUNT("SAMPLE_PREPARATION_TECHNIQUE") AS "SAMPLE_PREPARATION_TECHNIQUE",
    COUNT("MINIMUM_SAMPLE_VOLUME") AS "MINIMUM_SAMPLE_VOLUME",
    COUNT("DEFAULT_ASSAY_RANGE") AS "DEFAULT_ASSAY_RANGE",
    COUNT("NON_PROPRIETARY_METHOD_AVAILABLE") AS "NON_PROPRIETARY_METHOD_AVAILABLE",
    COUNT("RADIOLABELED_14C_ANALYSIS") AS "RADIOLABELED_14C_ANALYSIS",
    COUNT("NON_GLP_SAMPLE_ANALYSIS_TURNAROUND") AS "NON_GLP_SAMPLE_ANALYSIS_TURNAROUND",
    COUNT("GLP_SAMPLE_ANALYSIS_TURNAROUND") AS "GLP_SAMPLE_ANALYSIS_TURNAROUND",
    COUNT("EXPEDITED_ANALYSIS_AVAILABLE") AS "EXPEDITED_ANALYSIS_AVAILABLE",
    COUNT("PK_MODELING_AND_REPORTING") AS "PK_MODELING_AND_REPORTING"
    FROM SNOWFLAKE_LEARNING_DB."BIOANALYSIS".SMALL_MOLECULE_PK_BIOANALYSIS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('REGULATORY_COMPLIANCE_LEVEL', 'Regulatory compliance level', TRUE),
    ('MOLECULE_MODALITY', 'Molecule modality', TRUE),
    ('BIOANALYTICAL_GUIDANCE_FOLLOWED', 'Bioanalytical guidance followed', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('SAMPLE_MATRIX', 'Sample matrix', FALSE),
    ('SERVICE_SCOPE', 'Service scope', FALSE),
    ('AUTOMATED_SAMPLE_PREPARATION', 'Automated sample preparation', FALSE),
    ('SPECIES', 'Species', FALSE),
    ('VALIDATION_TYPE', 'Validation type', FALSE),
    ('VALIDATION_PARAMETERS', 'Validation parameters', FALSE),
    ('METHOD_ORIGIN', 'Method origin', FALSE),
    ('VALIDATION_DELIVERABLES', 'Validation deliverables', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORTED', 'Regulatory submission supported', FALSE),
    ('THERAPEUTIC_AREA_EXPERIENCE', 'Therapeutic area experience', FALSE),
    ('REAL_TIME_DATA_PORTAL', 'Real-time data portal', FALSE),
    ('VALIDATED_ASSAY_LIBRARY_SIZE', 'Validated assay library size', FALSE),
    ('FEASIBILITY_ASSESSMENT', 'Feasibility assessment', FALSE),
    ('METHOD_DEVELOPMENT_TIME', 'Method development time', FALSE),
    ('MATRIX_LOTS_FOR_MATRIX_EFFECT', 'Matrix lots for matrix effect', FALSE),
    ('CALIBRATION_STANDARD_LEVELS', 'Calibration standard levels', FALSE),
    ('ANALYTICAL_PLATFORM', 'Analytical platform', TRUE),
    ('MASS_SPECTROMETER_MODEL', 'Mass spectrometer model', FALSE),
    ('ANALYTE_CLASS', 'Analyte class', FALSE),
    ('STUDY_TYPE', 'Study type', FALSE),
    ('MICROSAMPLING_FORMAT', 'Microsampling format', FALSE),
    ('IONIZATION_MODE', 'Ionization mode', FALSE),
    ('SAMPLE_PREPARATION_TECHNIQUE', 'Sample preparation technique', FALSE),
    ('MINIMUM_SAMPLE_VOLUME', 'Minimum sample volume', FALSE),
    ('DEFAULT_ASSAY_RANGE', 'Default assay range', FALSE),
    ('NON_PROPRIETARY_METHOD_AVAILABLE', 'Non-proprietary method available', FALSE),
    ('RADIOLABELED_14C_ANALYSIS', 'Radiolabeled (14C) analysis', FALSE),
    ('NON_GLP_SAMPLE_ANALYSIS_TURNAROUND', 'Non-GLP sample analysis turnaround', FALSE),
    ('GLP_SAMPLE_ANALYSIS_TURNAROUND', 'GLP sample analysis turnaround', FALSE),
    ('EXPEDITED_ANALYSIS_AVAILABLE', 'Expedited analysis available', FALSE),
    ('PK_MODELING_AND_REPORTING', 'PK modeling and reporting', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'bioanalysis' AS ASSIGNED_L2_ID,
    'Bioanalysis' AS ASSIGNED_L2_LABEL,
    'small_molecule_pk_bioanalysis' AS ASSIGNED_L3_ID,
    'Small-Molecule PK Bioanalysis' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Bioanalysis > Small-Molecule PK Bioanalysis
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("REGULATORY_COMPLIANCE_LEVEL") AS "REGULATORY_COMPLIANCE_LEVEL",
    COUNT("MOLECULE_MODALITY") AS "MOLECULE_MODALITY",
    COUNT("BIOANALYTICAL_GUIDANCE_FOLLOWED") AS "BIOANALYTICAL_GUIDANCE_FOLLOWED",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("SAMPLE_MATRIX") AS "SAMPLE_MATRIX",
    COUNT("SERVICE_SCOPE") AS "SERVICE_SCOPE",
    COUNT("AUTOMATED_SAMPLE_PREPARATION") AS "AUTOMATED_SAMPLE_PREPARATION",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("VALIDATION_TYPE") AS "VALIDATION_TYPE",
    COUNT("VALIDATION_PARAMETERS") AS "VALIDATION_PARAMETERS",
    COUNT("METHOD_ORIGIN") AS "METHOD_ORIGIN",
    COUNT("VALIDATION_DELIVERABLES") AS "VALIDATION_DELIVERABLES",
    COUNT("REGULATORY_SUBMISSION_SUPPORTED") AS "REGULATORY_SUBMISSION_SUPPORTED",
    COUNT("THERAPEUTIC_AREA_EXPERIENCE") AS "THERAPEUTIC_AREA_EXPERIENCE",
    COUNT("REAL_TIME_DATA_PORTAL") AS "REAL_TIME_DATA_PORTAL",
    COUNT("VALIDATED_ASSAY_LIBRARY_SIZE") AS "VALIDATED_ASSAY_LIBRARY_SIZE",
    COUNT("FEASIBILITY_ASSESSMENT") AS "FEASIBILITY_ASSESSMENT",
    COUNT("METHOD_DEVELOPMENT_TIME") AS "METHOD_DEVELOPMENT_TIME",
    COUNT("MATRIX_LOTS_FOR_MATRIX_EFFECT") AS "MATRIX_LOTS_FOR_MATRIX_EFFECT",
    COUNT("CALIBRATION_STANDARD_LEVELS") AS "CALIBRATION_STANDARD_LEVELS",
    COUNT("ANALYTICAL_PLATFORM") AS "ANALYTICAL_PLATFORM",
    COUNT("BIOLOGIC_MODALITY") AS "BIOLOGIC_MODALITY",
    COUNT("STUDY_TYPE") AS "STUDY_TYPE",
    COUNT("CUSTOM_REAGENT_GENERATION") AS "CUSTOM_REAGENT_GENERATION",
    COUNT("CRITICAL_REAGENT_MANAGEMENT") AS "CRITICAL_REAGENT_MANAGEMENT",
    COUNT("ANTIBODY_ORIGIN") AS "ANTIBODY_ORIGIN",
    COUNT("ADC_DRUG_TO_ANTIBODY_RATIO_ANALYSIS") AS "ADC_DRUG_TO_ANTIBODY_RATIO_ANALYSIS",
    COUNT("ULTRASENSITIVE_PLATFORM_AVAILABLE") AS "ULTRASENSITIVE_PLATFORM_AVAILABLE",
    COUNT("GENE_THERAPY_DETECTION_LIMIT") AS "GENE_THERAPY_DETECTION_LIMIT"
    FROM SNOWFLAKE_LEARNING_DB."BIOANALYSIS".LARGE_MOLECULE_BIOANALYSIS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('REGULATORY_COMPLIANCE_LEVEL', 'Regulatory compliance level', TRUE),
    ('MOLECULE_MODALITY', 'Molecule modality', TRUE),
    ('BIOANALYTICAL_GUIDANCE_FOLLOWED', 'Bioanalytical guidance followed', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('SAMPLE_MATRIX', 'Sample matrix', FALSE),
    ('SERVICE_SCOPE', 'Service scope', FALSE),
    ('AUTOMATED_SAMPLE_PREPARATION', 'Automated sample preparation', FALSE),
    ('SPECIES', 'Species', FALSE),
    ('VALIDATION_TYPE', 'Validation type', FALSE),
    ('VALIDATION_PARAMETERS', 'Validation parameters', FALSE),
    ('METHOD_ORIGIN', 'Method origin', FALSE),
    ('VALIDATION_DELIVERABLES', 'Validation deliverables', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORTED', 'Regulatory submission supported', FALSE),
    ('THERAPEUTIC_AREA_EXPERIENCE', 'Therapeutic area experience', FALSE),
    ('REAL_TIME_DATA_PORTAL', 'Real-time data portal', FALSE),
    ('VALIDATED_ASSAY_LIBRARY_SIZE', 'Validated assay library size', FALSE),
    ('FEASIBILITY_ASSESSMENT', 'Feasibility assessment', FALSE),
    ('METHOD_DEVELOPMENT_TIME', 'Method development time', FALSE),
    ('MATRIX_LOTS_FOR_MATRIX_EFFECT', 'Matrix lots for matrix effect', FALSE),
    ('CALIBRATION_STANDARD_LEVELS', 'Calibration standard levels', FALSE),
    ('ANALYTICAL_PLATFORM', 'Analytical platform', TRUE),
    ('BIOLOGIC_MODALITY', 'Biologic modality', FALSE),
    ('STUDY_TYPE', 'Study type', FALSE),
    ('CUSTOM_REAGENT_GENERATION', 'Custom reagent generation', FALSE),
    ('CRITICAL_REAGENT_MANAGEMENT', 'Critical reagent management', FALSE),
    ('ANTIBODY_ORIGIN', 'Antibody origin', FALSE),
    ('ADC_DRUG_TO_ANTIBODY_RATIO_ANALYSIS', 'ADC drug-to-antibody ratio analysis', FALSE),
    ('ULTRASENSITIVE_PLATFORM_AVAILABLE', 'Ultrasensitive platform available', FALSE),
    ('GENE_THERAPY_DETECTION_LIMIT', 'Gene therapy detection limit', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'bioanalysis' AS ASSIGNED_L2_ID,
    'Bioanalysis' AS ASSIGNED_L2_LABEL,
    'large_molecule_bioanalysis' AS ASSIGNED_L3_ID,
    'Large-Molecule Bioanalysis' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Bioanalysis > Large-Molecule Bioanalysis
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("REGULATORY_COMPLIANCE_LEVEL") AS "REGULATORY_COMPLIANCE_LEVEL",
    COUNT("MOLECULE_MODALITY") AS "MOLECULE_MODALITY",
    COUNT("BIOANALYTICAL_GUIDANCE_FOLLOWED") AS "BIOANALYTICAL_GUIDANCE_FOLLOWED",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("SAMPLE_MATRIX") AS "SAMPLE_MATRIX",
    COUNT("SERVICE_SCOPE") AS "SERVICE_SCOPE",
    COUNT("AUTOMATED_SAMPLE_PREPARATION") AS "AUTOMATED_SAMPLE_PREPARATION",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("VALIDATION_TYPE") AS "VALIDATION_TYPE",
    COUNT("VALIDATION_PARAMETERS") AS "VALIDATION_PARAMETERS",
    COUNT("METHOD_ORIGIN") AS "METHOD_ORIGIN",
    COUNT("VALIDATION_DELIVERABLES") AS "VALIDATION_DELIVERABLES",
    COUNT("REGULATORY_SUBMISSION_SUPPORTED") AS "REGULATORY_SUBMISSION_SUPPORTED",
    COUNT("THERAPEUTIC_AREA_EXPERIENCE") AS "THERAPEUTIC_AREA_EXPERIENCE",
    COUNT("REAL_TIME_DATA_PORTAL") AS "REAL_TIME_DATA_PORTAL",
    COUNT("VALIDATED_ASSAY_LIBRARY_SIZE") AS "VALIDATED_ASSAY_LIBRARY_SIZE",
    COUNT("FEASIBILITY_ASSESSMENT") AS "FEASIBILITY_ASSESSMENT",
    COUNT("METHOD_DEVELOPMENT_TIME") AS "METHOD_DEVELOPMENT_TIME",
    COUNT("MATRIX_LOTS_FOR_MATRIX_EFFECT") AS "MATRIX_LOTS_FOR_MATRIX_EFFECT",
    COUNT("CALIBRATION_STANDARD_LEVELS") AS "CALIBRATION_STANDARD_LEVELS",
    COUNT("ADA_TIERS_OFFERED") AS "ADA_TIERS_OFFERED",
    COUNT("NAB_ASSAY_FORMAT") AS "NAB_ASSAY_FORMAT",
    COUNT("DRUG_TOLERANCE_METHOD") AS "DRUG_TOLERANCE_METHOD",
    COUNT("ASSAY_PLATFORM") AS "ASSAY_PLATFORM",
    COUNT("CUT_POINT_STATISTICAL_ANALYSIS") AS "CUT_POINT_STATISTICAL_ANALYSIS",
    COUNT("ADA_CHARACTERIZATION") AS "ADA_CHARACTERIZATION",
    COUNT("CELLULAR_IMMUNOGENICITY") AS "CELLULAR_IMMUNOGENICITY",
    COUNT("IMMUNOGENICITY_GUIDANCE_FOLLOWED") AS "IMMUNOGENICITY_GUIDANCE_FOLLOWED",
    COUNT("CRITICAL_REAGENT_MANAGEMENT") AS "CRITICAL_REAGENT_MANAGEMENT",
    COUNT("DRUG_TOLERANCE_TARGET") AS "DRUG_TOLERANCE_TARGET",
    COUNT("ADA_ASSAY_SENSITIVITY") AS "ADA_ASSAY_SENSITIVITY",
    COUNT("SCREENING_FALSE_POSITIVE_RATE") AS "SCREENING_FALSE_POSITIVE_RATE",
    COUNT("PRE_EXISTING_IMMUNITY_HANDLING") AS "PRE_EXISTING_IMMUNITY_HANDLING",
    COUNT("POSITIVE_CONTROL_GENERATION") AS "POSITIVE_CONTROL_GENERATION",
    COUNT("TDAR_ASSAY") AS "TDAR_ASSAY",
    COUNT("VALIDATION_LEVEL") AS "VALIDATION_LEVEL",
    COUNT("ADA_ASSAY_DEVELOPMENT_TIME") AS "ADA_ASSAY_DEVELOPMENT_TIME"
    FROM SNOWFLAKE_LEARNING_DB."BIOANALYSIS".IMMUNOGENICITY_ADA_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('REGULATORY_COMPLIANCE_LEVEL', 'Regulatory compliance level', TRUE),
    ('MOLECULE_MODALITY', 'Molecule modality', TRUE),
    ('BIOANALYTICAL_GUIDANCE_FOLLOWED', 'Bioanalytical guidance followed', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('SAMPLE_MATRIX', 'Sample matrix', FALSE),
    ('SERVICE_SCOPE', 'Service scope', FALSE),
    ('AUTOMATED_SAMPLE_PREPARATION', 'Automated sample preparation', FALSE),
    ('SPECIES', 'Species', FALSE),
    ('VALIDATION_TYPE', 'Validation type', FALSE),
    ('VALIDATION_PARAMETERS', 'Validation parameters', FALSE),
    ('METHOD_ORIGIN', 'Method origin', FALSE),
    ('VALIDATION_DELIVERABLES', 'Validation deliverables', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORTED', 'Regulatory submission supported', FALSE),
    ('THERAPEUTIC_AREA_EXPERIENCE', 'Therapeutic area experience', FALSE),
    ('REAL_TIME_DATA_PORTAL', 'Real-time data portal', FALSE),
    ('VALIDATED_ASSAY_LIBRARY_SIZE', 'Validated assay library size', FALSE),
    ('FEASIBILITY_ASSESSMENT', 'Feasibility assessment', FALSE),
    ('METHOD_DEVELOPMENT_TIME', 'Method development time', FALSE),
    ('MATRIX_LOTS_FOR_MATRIX_EFFECT', 'Matrix lots for matrix effect', FALSE),
    ('CALIBRATION_STANDARD_LEVELS', 'Calibration standard levels', FALSE),
    ('ADA_TIERS_OFFERED', 'ADA tiers offered', FALSE),
    ('NAB_ASSAY_FORMAT', 'NAb assay format', FALSE),
    ('DRUG_TOLERANCE_METHOD', 'Drug tolerance method', FALSE),
    ('ASSAY_PLATFORM', 'Assay platform', FALSE),
    ('CUT_POINT_STATISTICAL_ANALYSIS', 'Cut point statistical analysis', FALSE),
    ('ADA_CHARACTERIZATION', 'ADA characterization', FALSE),
    ('CELLULAR_IMMUNOGENICITY', 'Cellular immunogenicity', FALSE),
    ('IMMUNOGENICITY_GUIDANCE_FOLLOWED', 'Immunogenicity guidance followed', FALSE),
    ('CRITICAL_REAGENT_MANAGEMENT', 'Critical reagent management', FALSE),
    ('DRUG_TOLERANCE_TARGET', 'Drug tolerance target', FALSE),
    ('ADA_ASSAY_SENSITIVITY', 'ADA assay sensitivity', FALSE),
    ('SCREENING_FALSE_POSITIVE_RATE', 'Screening false positive rate', FALSE),
    ('PRE_EXISTING_IMMUNITY_HANDLING', 'Pre-existing immunity handling', FALSE),
    ('POSITIVE_CONTROL_GENERATION', 'Positive control generation', FALSE),
    ('TDAR_ASSAY', 'TDAR assay', FALSE),
    ('VALIDATION_LEVEL', 'Validation level', FALSE),
    ('ADA_ASSAY_DEVELOPMENT_TIME', 'ADA assay development time', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'bioanalysis' AS ASSIGNED_L2_ID,
    'Bioanalysis' AS ASSIGNED_L2_LABEL,
    'immunogenicity_ada' AS ASSIGNED_L3_ID,
    'Immunogenicity / ADA' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Bioanalysis > Immunogenicity / ADA
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("REGULATORY_COMPLIANCE_LEVEL") AS "REGULATORY_COMPLIANCE_LEVEL",
    COUNT("MOLECULE_MODALITY") AS "MOLECULE_MODALITY",
    COUNT("BIOANALYTICAL_GUIDANCE_FOLLOWED") AS "BIOANALYTICAL_GUIDANCE_FOLLOWED",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("SAMPLE_MATRIX") AS "SAMPLE_MATRIX",
    COUNT("SERVICE_SCOPE") AS "SERVICE_SCOPE",
    COUNT("AUTOMATED_SAMPLE_PREPARATION") AS "AUTOMATED_SAMPLE_PREPARATION",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("VALIDATION_TYPE") AS "VALIDATION_TYPE",
    COUNT("VALIDATION_PARAMETERS") AS "VALIDATION_PARAMETERS",
    COUNT("METHOD_ORIGIN") AS "METHOD_ORIGIN",
    COUNT("VALIDATION_DELIVERABLES") AS "VALIDATION_DELIVERABLES",
    COUNT("REGULATORY_SUBMISSION_SUPPORTED") AS "REGULATORY_SUBMISSION_SUPPORTED",
    COUNT("THERAPEUTIC_AREA_EXPERIENCE") AS "THERAPEUTIC_AREA_EXPERIENCE",
    COUNT("REAL_TIME_DATA_PORTAL") AS "REAL_TIME_DATA_PORTAL",
    COUNT("VALIDATED_ASSAY_LIBRARY_SIZE") AS "VALIDATED_ASSAY_LIBRARY_SIZE",
    COUNT("FEASIBILITY_ASSESSMENT") AS "FEASIBILITY_ASSESSMENT",
    COUNT("METHOD_DEVELOPMENT_TIME") AS "METHOD_DEVELOPMENT_TIME",
    COUNT("MATRIX_LOTS_FOR_MATRIX_EFFECT") AS "MATRIX_LOTS_FOR_MATRIX_EFFECT",
    COUNT("CALIBRATION_STANDARD_LEVELS") AS "CALIBRATION_STANDARD_LEVELS"
    FROM SNOWFLAKE_LEARNING_DB."BIOANALYSIS".METHOD_DEVELOPMENT_VALIDATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('REGULATORY_COMPLIANCE_LEVEL', 'Regulatory compliance level', TRUE),
    ('MOLECULE_MODALITY', 'Molecule modality', TRUE),
    ('BIOANALYTICAL_GUIDANCE_FOLLOWED', 'Bioanalytical guidance followed', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('SAMPLE_MATRIX', 'Sample matrix', FALSE),
    ('SERVICE_SCOPE', 'Service scope', FALSE),
    ('AUTOMATED_SAMPLE_PREPARATION', 'Automated sample preparation', FALSE),
    ('SPECIES', 'Species', FALSE),
    ('VALIDATION_TYPE', 'Validation type', FALSE),
    ('VALIDATION_PARAMETERS', 'Validation parameters', FALSE),
    ('METHOD_ORIGIN', 'Method origin', FALSE),
    ('VALIDATION_DELIVERABLES', 'Validation deliverables', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORTED', 'Regulatory submission supported', FALSE),
    ('THERAPEUTIC_AREA_EXPERIENCE', 'Therapeutic area experience', FALSE),
    ('REAL_TIME_DATA_PORTAL', 'Real-time data portal', FALSE),
    ('VALIDATED_ASSAY_LIBRARY_SIZE', 'Validated assay library size', FALSE),
    ('FEASIBILITY_ASSESSMENT', 'Feasibility assessment', FALSE),
    ('METHOD_DEVELOPMENT_TIME', 'Method development time', FALSE),
    ('MATRIX_LOTS_FOR_MATRIX_EFFECT', 'Matrix lots for matrix effect', FALSE),
    ('CALIBRATION_STANDARD_LEVELS', 'Calibration standard levels', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'bioanalysis' AS ASSIGNED_L2_ID,
    'Bioanalysis' AS ASSIGNED_L2_LABEL,
    'method_development_validation_bioanalysis' AS ASSIGNED_L3_ID,
    'Method Development & Validation' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Bioanalysis > Method Development & Validation
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_BIOMARKER_COMPANION_DIAGNOSTICS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("ASSAY_TECHNOLOGY") AS "ASSAY_TECHNOLOGY",
    COUNT("ASSAY_DEVELOPMENT_STAGE") AS "ASSAY_DEVELOPMENT_STAGE",
    COUNT("SAMPLE_TYPE_BIOMARKER") AS "SAMPLE_TYPE_BIOMARKER",
    COUNT("THERAPEUTIC_AREA_BIOMARKER") AS "THERAPEUTIC_AREA_BIOMARKER",
    COUNT("LAB_REGION_BIOMARKER") AS "LAB_REGION_BIOMARKER",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("VALIDATION_FRAMEWORK") AS "VALIDATION_FRAMEWORK",
    COUNT("BIOSPECIMEN_ACCESS_FOR_VALIDATION") AS "BIOSPECIMEN_ACCESS_FOR_VALIDATION",
    COUNT("PROSPECTIVE_TRIAL_PATIENT_TESTING") AS "PROSPECTIVE_TRIAL_PATIENT_TESTING",
    COUNT("ANALYTICAL_PERFORMANCE_PARAMETER") AS "ANALYTICAL_PERFORMANCE_PARAMETER",
    COUNT("CLINICAL_PERFORMANCE_STUDY_TYPE") AS "CLINICAL_PERFORMANCE_STUDY_TYPE",
    COUNT("MULTI_SITE_REPRODUCIBILITY") AS "MULTI_SITE_REPRODUCIBILITY",
    COUNT("VALIDATION_DELIVERABLE") AS "VALIDATION_DELIVERABLE",
    COUNT("POST_MARKET_PERFORMANCE_MONITORING") AS "POST_MARKET_PERFORMANCE_MONITORING",
    COUNT("BIOMARKER_TYPE") AS "BIOMARKER_TYPE",
    COUNT("ANALYTICAL_PLATFORM") AS "ANALYTICAL_PLATFORM",
    COUNT("MULTIPLEX_CAPABILITY") AS "MULTIPLEX_CAPABILITY",
    COUNT("VALIDATION_APPROACH") AS "VALIDATION_APPROACH",
    COUNT("ASSAY_CONTEXT_OF_USE") AS "ASSAY_CONTEXT_OF_USE",
    COUNT("CLIA_TESTING_AVAILABLE") AS "CLIA_TESTING_AVAILABLE",
    COUNT("ASSAY_SOURCE") AS "ASSAY_SOURCE",
    COUNT("BEST_BIOMARKER_CATEGORY") AS "BEST_BIOMARKER_CATEGORY",
    COUNT("MAX_ANALYTES_PER_SAMPLE") AS "MAX_ANALYTES_PER_SAMPLE",
    COUNT("LOWEST_QUANTITATION_RANGE") AS "LOWEST_QUANTITATION_RANGE",
    COUNT("FLOW_CYTOMETRY_PANEL_SIZE") AS "FLOW_CYTOMETRY_PANEL_SIZE",
    COUNT("MULTIPLEX_IF_MARKER_COUNT") AS "MULTIPLEX_IF_MARKER_COUNT",
    COUNT("TECHNOLOGY_TRANSFER_OFFERED") AS "TECHNOLOGY_TRANSFER_OFFERED",
    COUNT("ASSAY_DELIVERABLE") AS "ASSAY_DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."BIOMARKER_COMPANION_DIAGNOSTICS".BIOMARKER_ASSAYS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('ASSAY_TECHNOLOGY', 'Assay technology', TRUE),
    ('ASSAY_DEVELOPMENT_STAGE', 'Assay development stage', TRUE),
    ('SAMPLE_TYPE_BIOMARKER', 'Sample type (biomarker)', FALSE),
    ('THERAPEUTIC_AREA_BIOMARKER', 'Therapeutic area (biomarker)', FALSE),
    ('LAB_REGION_BIOMARKER', 'Lab region (biomarker)', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('VALIDATION_FRAMEWORK', 'Validation framework', FALSE),
    ('BIOSPECIMEN_ACCESS_FOR_VALIDATION', 'Biospecimen access for validation', FALSE),
    ('PROSPECTIVE_TRIAL_PATIENT_TESTING', 'Prospective trial patient testing', FALSE),
    ('ANALYTICAL_PERFORMANCE_PARAMETER', 'Analytical performance parameter', FALSE),
    ('CLINICAL_PERFORMANCE_STUDY_TYPE', 'Clinical performance study type', FALSE),
    ('MULTI_SITE_REPRODUCIBILITY', 'Multi-site reproducibility', FALSE),
    ('VALIDATION_DELIVERABLE', 'Validation deliverable', FALSE),
    ('POST_MARKET_PERFORMANCE_MONITORING', 'Post-market performance monitoring', FALSE),
    ('BIOMARKER_TYPE', 'Biomarker type', TRUE),
    ('ANALYTICAL_PLATFORM', 'Analytical platform', FALSE),
    ('MULTIPLEX_CAPABILITY', 'Multiplex capability', FALSE),
    ('VALIDATION_APPROACH', 'Validation approach', FALSE),
    ('ASSAY_CONTEXT_OF_USE', 'Assay context of use', FALSE),
    ('CLIA_TESTING_AVAILABLE', 'CLIA testing available', FALSE),
    ('ASSAY_SOURCE', 'Assay source', FALSE),
    ('BEST_BIOMARKER_CATEGORY', 'BEST biomarker category', FALSE),
    ('MAX_ANALYTES_PER_SAMPLE', 'Max analytes per sample', FALSE),
    ('LOWEST_QUANTITATION_RANGE', 'Lowest quantitation range', FALSE),
    ('FLOW_CYTOMETRY_PANEL_SIZE', 'Flow cytometry panel size', FALSE),
    ('MULTIPLEX_IF_MARKER_COUNT', 'Multiplex IF marker count', FALSE),
    ('TECHNOLOGY_TRANSFER_OFFERED', 'Technology transfer offered', FALSE),
    ('ASSAY_DELIVERABLE', 'Assay deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'biomarker_companion_diagnostics' AS ASSIGNED_L2_ID,
    'Biomarker & Companion Diagnostics' AS ASSIGNED_L2_LABEL,
    'biomarker_assays' AS ASSIGNED_L3_ID,
    'Biomarker Assays' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Biomarker & Companion Diagnostics > Biomarker Assays
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("ASSAY_TECHNOLOGY") AS "ASSAY_TECHNOLOGY",
    COUNT("ASSAY_DEVELOPMENT_STAGE") AS "ASSAY_DEVELOPMENT_STAGE",
    COUNT("SAMPLE_TYPE_BIOMARKER") AS "SAMPLE_TYPE_BIOMARKER",
    COUNT("THERAPEUTIC_AREA_BIOMARKER") AS "THERAPEUTIC_AREA_BIOMARKER",
    COUNT("LAB_REGION_BIOMARKER") AS "LAB_REGION_BIOMARKER",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("VALIDATION_FRAMEWORK") AS "VALIDATION_FRAMEWORK",
    COUNT("BIOSPECIMEN_ACCESS_FOR_VALIDATION") AS "BIOSPECIMEN_ACCESS_FOR_VALIDATION",
    COUNT("PROSPECTIVE_TRIAL_PATIENT_TESTING") AS "PROSPECTIVE_TRIAL_PATIENT_TESTING",
    COUNT("ANALYTICAL_PERFORMANCE_PARAMETER") AS "ANALYTICAL_PERFORMANCE_PARAMETER",
    COUNT("CLINICAL_PERFORMANCE_STUDY_TYPE") AS "CLINICAL_PERFORMANCE_STUDY_TYPE",
    COUNT("MULTI_SITE_REPRODUCIBILITY") AS "MULTI_SITE_REPRODUCIBILITY",
    COUNT("VALIDATION_DELIVERABLE") AS "VALIDATION_DELIVERABLE",
    COUNT("POST_MARKET_PERFORMANCE_MONITORING") AS "POST_MARKET_PERFORMANCE_MONITORING",
    COUNT("TEST_FORMAT") AS "TEST_FORMAT",
    COUNT("REGULATORY_PATHWAY") AS "REGULATORY_PATHWAY",
    COUNT("PRIOR_CDX_BIOMARKERS") AS "PRIOR_CDX_BIOMARKERS",
    COUNT("COMMERCIAL_LAUNCH_SUPPORT") AS "COMMERCIAL_LAUNCH_SUPPORT",
    COUNT("COLLABORATION_MODEL") AS "COLLABORATION_MODEL",
    COUNT("CONCURRENT_DRUG_AND_DIAGNOSTIC_APPROVAL_SUPPORT") AS "CONCURRENT_DRUG_AND_DIAGNOSTIC_APPROVAL_SUPPORT",
    COUNT("INVESTIGATIONAL_SUBMISSION_SUPPORT") AS "INVESTIGATIONAL_SUBMISSION_SUPPORT",
    COUNT("DIAGNOSTIC_CLAIM_TYPE") AS "DIAGNOSTIC_CLAIM_TYPE",
    COUNT("GMP_KIT_MANUFACTURING") AS "GMP_KIT_MANUFACTURING",
    COUNT("COUNTRIES_WITH_REGISTRATION_SUPPORT") AS "COUNTRIES_WITH_REGISTRATION_SUPPORT"
    FROM SNOWFLAKE_LEARNING_DB."BIOMARKER_COMPANION_DIAGNOSTICS".COMPANION_DIAGNOSTIC_CDX_DEVELOPMENT_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('ASSAY_TECHNOLOGY', 'Assay technology', TRUE),
    ('ASSAY_DEVELOPMENT_STAGE', 'Assay development stage', TRUE),
    ('SAMPLE_TYPE_BIOMARKER', 'Sample type (biomarker)', FALSE),
    ('THERAPEUTIC_AREA_BIOMARKER', 'Therapeutic area (biomarker)', FALSE),
    ('LAB_REGION_BIOMARKER', 'Lab region (biomarker)', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('VALIDATION_FRAMEWORK', 'Validation framework', FALSE),
    ('BIOSPECIMEN_ACCESS_FOR_VALIDATION', 'Biospecimen access for validation', FALSE),
    ('PROSPECTIVE_TRIAL_PATIENT_TESTING', 'Prospective trial patient testing', FALSE),
    ('ANALYTICAL_PERFORMANCE_PARAMETER', 'Analytical performance parameter', FALSE),
    ('CLINICAL_PERFORMANCE_STUDY_TYPE', 'Clinical performance study type', FALSE),
    ('MULTI_SITE_REPRODUCIBILITY', 'Multi-site reproducibility', FALSE),
    ('VALIDATION_DELIVERABLE', 'Validation deliverable', FALSE),
    ('POST_MARKET_PERFORMANCE_MONITORING', 'Post-market performance monitoring', FALSE),
    ('TEST_FORMAT', 'Test format', TRUE),
    ('REGULATORY_PATHWAY', 'Regulatory pathway', FALSE),
    ('PRIOR_CDX_BIOMARKERS', 'Prior CDx biomarkers', FALSE),
    ('COMMERCIAL_LAUNCH_SUPPORT', 'Commercial launch support', FALSE),
    ('COLLABORATION_MODEL', 'Collaboration model', FALSE),
    ('CONCURRENT_DRUG_AND_DIAGNOSTIC_APPROVAL_SUPPORT', 'Concurrent drug and diagnostic approval support', FALSE),
    ('INVESTIGATIONAL_SUBMISSION_SUPPORT', 'Investigational submission support', FALSE),
    ('DIAGNOSTIC_CLAIM_TYPE', 'Diagnostic claim type', FALSE),
    ('GMP_KIT_MANUFACTURING', 'GMP kit manufacturing', FALSE),
    ('COUNTRIES_WITH_REGISTRATION_SUPPORT', 'Countries with registration support', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'biomarker_companion_diagnostics' AS ASSIGNED_L2_ID,
    'Biomarker & Companion Diagnostics' AS ASSIGNED_L2_LABEL,
    'companion_diagnostic_cdx_development' AS ASSIGNED_L3_ID,
    'Companion Diagnostic (CDx) Development' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Biomarker & Companion Diagnostics > Companion Diagnostic (CDx) Development
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("ASSAY_TECHNOLOGY") AS "ASSAY_TECHNOLOGY",
    COUNT("ASSAY_DEVELOPMENT_STAGE") AS "ASSAY_DEVELOPMENT_STAGE",
    COUNT("SAMPLE_TYPE_BIOMARKER") AS "SAMPLE_TYPE_BIOMARKER",
    COUNT("THERAPEUTIC_AREA_BIOMARKER") AS "THERAPEUTIC_AREA_BIOMARKER",
    COUNT("LAB_REGION_BIOMARKER") AS "LAB_REGION_BIOMARKER",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("VALIDATION_FRAMEWORK") AS "VALIDATION_FRAMEWORK",
    COUNT("BIOSPECIMEN_ACCESS_FOR_VALIDATION") AS "BIOSPECIMEN_ACCESS_FOR_VALIDATION",
    COUNT("PROSPECTIVE_TRIAL_PATIENT_TESTING") AS "PROSPECTIVE_TRIAL_PATIENT_TESTING",
    COUNT("ANALYTICAL_PERFORMANCE_PARAMETER") AS "ANALYTICAL_PERFORMANCE_PARAMETER",
    COUNT("CLINICAL_PERFORMANCE_STUDY_TYPE") AS "CLINICAL_PERFORMANCE_STUDY_TYPE",
    COUNT("MULTI_SITE_REPRODUCIBILITY") AS "MULTI_SITE_REPRODUCIBILITY",
    COUNT("VALIDATION_DELIVERABLE") AS "VALIDATION_DELIVERABLE",
    COUNT("POST_MARKET_PERFORMANCE_MONITORING") AS "POST_MARKET_PERFORMANCE_MONITORING",
    COUNT("SEQUENCING_SCOPE") AS "SEQUENCING_SCOPE",
    COUNT("TEST_REGULATORY_STATUS") AS "TEST_REGULATORY_STATUS",
    COUNT("DATA_DELIVERABLE") AS "DATA_DELIVERABLE",
    COUNT("PANEL_SIZE") AS "PANEL_SIZE",
    COUNT("NUCLEIC_ACID_ANALYZED") AS "NUCLEIC_ACID_ANALYZED",
    COUNT("VARIANT_TYPE") AS "VARIANT_TYPE",
    COUNT("GENOMIC_SIGNATURE") AS "GENOMIC_SIGNATURE",
    COUNT("TURNAROUND_TIME_GENOMIC") AS "TURNAROUND_TIME_GENOMIC",
    COUNT("PROFILING_APPLICATION") AS "PROFILING_APPLICATION",
    COUNT("IHC_ADD_ON") AS "IHC_ADD_ON",
    COUNT("MINIMUM_DNA_INPUT") AS "MINIMUM_DNA_INPUT",
    COUNT("BLOOD_DRAW_VOLUME") AS "BLOOD_DRAW_VOLUME",
    COUNT("MATCHED_NORMAL_ANALYSIS") AS "MATCHED_NORMAL_ANALYSIS",
    COUNT("KITTED_IN_HOUSE_OPTION") AS "KITTED_IN_HOUSE_OPTION"
    FROM SNOWFLAKE_LEARNING_DB."BIOMARKER_COMPANION_DIAGNOSTICS".GENOMIC_MOLECULAR_PROFILING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('ASSAY_TECHNOLOGY', 'Assay technology', TRUE),
    ('ASSAY_DEVELOPMENT_STAGE', 'Assay development stage', TRUE),
    ('SAMPLE_TYPE_BIOMARKER', 'Sample type (biomarker)', FALSE),
    ('THERAPEUTIC_AREA_BIOMARKER', 'Therapeutic area (biomarker)', FALSE),
    ('LAB_REGION_BIOMARKER', 'Lab region (biomarker)', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('VALIDATION_FRAMEWORK', 'Validation framework', FALSE),
    ('BIOSPECIMEN_ACCESS_FOR_VALIDATION', 'Biospecimen access for validation', FALSE),
    ('PROSPECTIVE_TRIAL_PATIENT_TESTING', 'Prospective trial patient testing', FALSE),
    ('ANALYTICAL_PERFORMANCE_PARAMETER', 'Analytical performance parameter', FALSE),
    ('CLINICAL_PERFORMANCE_STUDY_TYPE', 'Clinical performance study type', FALSE),
    ('MULTI_SITE_REPRODUCIBILITY', 'Multi-site reproducibility', FALSE),
    ('VALIDATION_DELIVERABLE', 'Validation deliverable', FALSE),
    ('POST_MARKET_PERFORMANCE_MONITORING', 'Post-market performance monitoring', FALSE),
    ('SEQUENCING_SCOPE', 'Sequencing scope', TRUE),
    ('TEST_REGULATORY_STATUS', 'Test regulatory status', FALSE),
    ('DATA_DELIVERABLE', 'Data deliverable', FALSE),
    ('PANEL_SIZE', 'Panel size', FALSE),
    ('NUCLEIC_ACID_ANALYZED', 'Nucleic acid analyzed', FALSE),
    ('VARIANT_TYPE', 'Variant type', FALSE),
    ('GENOMIC_SIGNATURE', 'Genomic signature', FALSE),
    ('TURNAROUND_TIME_GENOMIC', 'Turnaround time (genomic)', FALSE),
    ('PROFILING_APPLICATION', 'Profiling application', FALSE),
    ('IHC_ADD_ON', 'IHC add-on', FALSE),
    ('MINIMUM_DNA_INPUT', 'Minimum DNA input', FALSE),
    ('BLOOD_DRAW_VOLUME', 'Blood draw volume', FALSE),
    ('MATCHED_NORMAL_ANALYSIS', 'Matched normal analysis', FALSE),
    ('KITTED_IN_HOUSE_OPTION', 'Kitted in-house option', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'biomarker_companion_diagnostics' AS ASSIGNED_L2_ID,
    'Biomarker & Companion Diagnostics' AS ASSIGNED_L2_LABEL,
    'genomic_molecular_profiling' AS ASSIGNED_L3_ID,
    'Genomic / Molecular Profiling' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Biomarker & Companion Diagnostics > Genomic / Molecular Profiling
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_CELL_TISSUE_CULTURE_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BASAL_FORMULATION") AS "BASAL_FORMULATION",
    COUNT("FORMAT_PHYSICAL_FORM") AS "FORMAT_PHYSICAL_FORM",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("SPECIES_CELL_SOURCE_OR_INTENDED_USE") AS "SPECIES_CELL_SOURCE_OR_INTENDED_USE",
    COUNT("GLUCOSE") AS "GLUCOSE",
    COUNT("L_GLUTAMINE") AS "L_GLUTAMINE",
    COUNT("PHENOL_RED") AS "PHENOL_RED",
    COUNT("HEPES_SODIUM_PYRUVATE") AS "HEPES_SODIUM_PYRUVATE",
    COUNT("SPECIALIZED_SYSTEM") AS "SPECIALIZED_SYSTEM",
    COUNT("KIT_COMPONENTS") AS "KIT_COMPONENTS",
    COUNT("SHELF_LIFE") AS "SHELF_LIFE",
    COUNT("FORMULATION_ANIMAL_ORIGIN") AS "FORMULATION_ANIMAL_ORIGIN",
    COUNT("INTENDED_CELL_TYPE_APPLICATION") AS "INTENDED_CELL_TYPE_APPLICATION",
    COUNT("GRADE_REGULATORY") AS "GRADE_REGULATORY",
    COUNT("STERILITY_CONTAMINATION_QC") AS "STERILITY_CONTAMINATION_QC",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE"
    FROM SNOWFLAKE_LEARNING_DB."CELLS_TISSUE_CULTURE".CULTURE_MEDIA_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BASAL_FORMULATION', 'Basal formulation', TRUE),
    ('FORMAT_PHYSICAL_FORM', 'Format / physical form', TRUE),
    ('STORAGE_CONDITIONS', 'Storage conditions', TRUE),
    ('SPECIES_CELL_SOURCE_OR_INTENDED_USE', 'Species (cell source or intended use)', TRUE),
    ('GLUCOSE', 'Glucose', FALSE),
    ('L_GLUTAMINE', 'L-Glutamine', FALSE),
    ('PHENOL_RED', 'Phenol red', FALSE),
    ('HEPES_SODIUM_PYRUVATE', 'HEPES / Sodium pyruvate', FALSE),
    ('SPECIALIZED_SYSTEM', 'Specialized system', FALSE),
    ('KIT_COMPONENTS', 'Kit components', FALSE),
    ('SHELF_LIFE', 'Shelf life', FALSE),
    ('FORMULATION_ANIMAL_ORIGIN', 'Formulation / animal origin', FALSE),
    ('INTENDED_CELL_TYPE_APPLICATION', 'Intended cell type / application', FALSE),
    ('GRADE_REGULATORY', 'Grade / regulatory', FALSE),
    ('STERILITY_CONTAMINATION_QC', 'Sterility / contamination QC', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cell_tissue_culture' AS ASSIGNED_L2_ID,
    'Cells and Tissue Culture' AS ASSIGNED_L2_LABEL,
    'culture_media' AS ASSIGNED_L3_ID,
    'Culture Media' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Cells and Tissue Culture > Culture Media
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SERUM_TYPE_SOURCE") AS "SERUM_TYPE_SOURCE",
    COUNT("COUNTRY_OF_ORIGIN") AS "COUNTRY_OF_ORIGIN",
    COUNT("FORMAT_PHYSICAL_FORM") AS "FORMAT_PHYSICAL_FORM",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("SPECIES_CELL_SOURCE_OR_INTENDED_USE") AS "SPECIES_CELL_SOURCE_OR_INTENDED_USE",
    COUNT("TREATMENT") AS "TREATMENT",
    COUNT("QUALITY_TIER") AS "QUALITY_TIER",
    COUNT("ENDOTOXIN") AS "ENDOTOXIN",
    COUNT("HEMOGLOBIN") AS "HEMOGLOBIN",
    COUNT("ORIGIN_TRACEABILITY") AS "ORIGIN_TRACEABILITY",
    COUNT("FORMULATION_ANIMAL_ORIGIN") AS "FORMULATION_ANIMAL_ORIGIN",
    COUNT("INTENDED_CELL_TYPE_APPLICATION") AS "INTENDED_CELL_TYPE_APPLICATION",
    COUNT("GRADE_REGULATORY") AS "GRADE_REGULATORY",
    COUNT("STERILITY_CONTAMINATION_QC") AS "STERILITY_CONTAMINATION_QC",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE"
    FROM SNOWFLAKE_LEARNING_DB."CELLS_TISSUE_CULTURE".SERA_SUPPLEMENTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SERUM_TYPE_SOURCE', 'Serum type / source', TRUE),
    ('COUNTRY_OF_ORIGIN', 'Country of origin', TRUE),
    ('FORMAT_PHYSICAL_FORM', 'Format / physical form', TRUE),
    ('STORAGE_CONDITIONS', 'Storage conditions', TRUE),
    ('SPECIES_CELL_SOURCE_OR_INTENDED_USE', 'Species (cell source or intended use)', TRUE),
    ('TREATMENT', 'Treatment', FALSE),
    ('QUALITY_TIER', 'Quality tier', FALSE),
    ('ENDOTOXIN', 'Endotoxin', FALSE),
    ('HEMOGLOBIN', 'Hemoglobin', FALSE),
    ('ORIGIN_TRACEABILITY', 'Origin traceability', FALSE),
    ('FORMULATION_ANIMAL_ORIGIN', 'Formulation / animal origin', FALSE),
    ('INTENDED_CELL_TYPE_APPLICATION', 'Intended cell type / application', FALSE),
    ('GRADE_REGULATORY', 'Grade / regulatory', FALSE),
    ('STERILITY_CONTAMINATION_QC', 'Sterility / contamination QC', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cell_tissue_culture' AS ASSIGNED_L2_ID,
    'Cells and Tissue Culture' AS ASSIGNED_L2_LABEL,
    'sera_supplements' AS ASSIGNED_L3_ID,
    'Sera & Supplements' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Cells and Tissue Culture > Sera & Supplements
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("MATRIX_TYPE") AS "MATRIX_TYPE",
    COUNT("GROWTH_FACTOR_CONTENT") AS "GROWTH_FACTOR_CONTENT",
    COUNT("FORMAT_PHYSICAL_FORM") AS "FORMAT_PHYSICAL_FORM",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("SPECIES_CELL_SOURCE_OR_INTENDED_USE") AS "SPECIES_CELL_SOURCE_OR_INTENDED_USE",
    COUNT("QUALIFICATION") AS "QUALIFICATION",
    COUNT("PROTEIN_CONCENTRATION") AS "PROTEIN_CONCENTRATION",
    COUNT("PHENOL_RED") AS "PHENOL_RED",
    COUNT("DELIVERY_FORMAT") AS "DELIVERY_FORMAT",
    COUNT("FORMULATION_ANIMAL_ORIGIN") AS "FORMULATION_ANIMAL_ORIGIN",
    COUNT("INTENDED_CELL_TYPE_APPLICATION") AS "INTENDED_CELL_TYPE_APPLICATION",
    COUNT("GRADE_REGULATORY") AS "GRADE_REGULATORY",
    COUNT("STERILITY_CONTAMINATION_QC") AS "STERILITY_CONTAMINATION_QC",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE"
    FROM SNOWFLAKE_LEARNING_DB."CELLS_TISSUE_CULTURE".MATRICES_COATINGS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('MATRIX_TYPE', 'Matrix type', TRUE),
    ('GROWTH_FACTOR_CONTENT', 'Growth factor content', TRUE),
    ('FORMAT_PHYSICAL_FORM', 'Format / physical form', TRUE),
    ('STORAGE_CONDITIONS', 'Storage conditions', TRUE),
    ('SPECIES_CELL_SOURCE_OR_INTENDED_USE', 'Species (cell source or intended use)', TRUE),
    ('QUALIFICATION', 'Qualification', FALSE),
    ('PROTEIN_CONCENTRATION', 'Protein concentration', FALSE),
    ('PHENOL_RED', 'Phenol red', FALSE),
    ('DELIVERY_FORMAT', 'Delivery format', FALSE),
    ('FORMULATION_ANIMAL_ORIGIN', 'Formulation / animal origin', FALSE),
    ('INTENDED_CELL_TYPE_APPLICATION', 'Intended cell type / application', FALSE),
    ('GRADE_REGULATORY', 'Grade / regulatory', FALSE),
    ('STERILITY_CONTAMINATION_QC', 'Sterility / contamination QC', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cell_tissue_culture' AS ASSIGNED_L2_ID,
    'Cells and Tissue Culture' AS ASSIGNED_L2_LABEL,
    'matrices_coatings' AS ASSIGNED_L3_ID,
    'Matrices & Coatings' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Cells and Tissue Culture > Matrices & Coatings
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DMSO_CONCENTRATION") AS "DMSO_CONCENTRATION",
    COUNT("FORMAT_PHYSICAL_FORM") AS "FORMAT_PHYSICAL_FORM",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("SPECIES_CELL_SOURCE_OR_INTENDED_USE") AS "SPECIES_CELL_SOURCE_OR_INTENDED_USE",
    COUNT("FORMULATION") AS "FORMULATION",
    COUNT("READY_TO_USE") AS "READY_TO_USE",
    COUNT("VALIDATED_CELL_TYPES") AS "VALIDATED_CELL_TYPES",
    COUNT("SHELF_LIFE") AS "SHELF_LIFE",
    COUNT("FORMULATION_ANIMAL_ORIGIN") AS "FORMULATION_ANIMAL_ORIGIN",
    COUNT("INTENDED_CELL_TYPE_APPLICATION") AS "INTENDED_CELL_TYPE_APPLICATION",
    COUNT("GRADE_REGULATORY") AS "GRADE_REGULATORY",
    COUNT("STERILITY_CONTAMINATION_QC") AS "STERILITY_CONTAMINATION_QC",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE"
    FROM SNOWFLAKE_LEARNING_DB."CELLS_TISSUE_CULTURE".CRYOPRESERVATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DMSO_CONCENTRATION', 'DMSO concentration', TRUE),
    ('FORMAT_PHYSICAL_FORM', 'Format / physical form', TRUE),
    ('STORAGE_CONDITIONS', 'Storage conditions', TRUE),
    ('SPECIES_CELL_SOURCE_OR_INTENDED_USE', 'Species (cell source or intended use)', TRUE),
    ('FORMULATION', 'Formulation', FALSE),
    ('READY_TO_USE', 'Ready-to-use', FALSE),
    ('VALIDATED_CELL_TYPES', 'Validated cell types', FALSE),
    ('SHELF_LIFE', 'Shelf life', FALSE),
    ('FORMULATION_ANIMAL_ORIGIN', 'Formulation / animal origin', FALSE),
    ('INTENDED_CELL_TYPE_APPLICATION', 'Intended cell type / application', FALSE),
    ('GRADE_REGULATORY', 'Grade / regulatory', FALSE),
    ('STERILITY_CONTAMINATION_QC', 'Sterility / contamination QC', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cell_tissue_culture' AS ASSIGNED_L2_ID,
    'Cells and Tissue Culture' AS ASSIGNED_L2_LABEL,
    'cryopreservation' AS ASSIGNED_L3_ID,
    'Cryopreservation' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Cells and Tissue Culture > Cryopreservation
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("ENZYME_AGENT") AS "ENZYME_AGENT",
    COUNT("CONCENTRATION") AS "CONCENTRATION",
    COUNT("FORMAT_PHYSICAL_FORM") AS "FORMAT_PHYSICAL_FORM",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("SPECIES_CELL_SOURCE_OR_INTENDED_USE") AS "SPECIES_CELL_SOURCE_OR_INTENDED_USE",
    COUNT("CHELATOR") AS "CHELATOR",
    COUNT("PHENOL_RED") AS "PHENOL_RED",
    COUNT("ANIMAL_ORIGIN") AS "ANIMAL_ORIGIN",
    COUNT("FORMULATION_ANIMAL_ORIGIN") AS "FORMULATION_ANIMAL_ORIGIN",
    COUNT("INTENDED_CELL_TYPE_APPLICATION") AS "INTENDED_CELL_TYPE_APPLICATION",
    COUNT("GRADE_REGULATORY") AS "GRADE_REGULATORY",
    COUNT("STERILITY_CONTAMINATION_QC") AS "STERILITY_CONTAMINATION_QC",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE"
    FROM SNOWFLAKE_LEARNING_DB."CELLS_TISSUE_CULTURE".DISSOCIATION_REAGENTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('ENZYME_AGENT', 'Enzyme / agent', TRUE),
    ('CONCENTRATION', 'Concentration', TRUE),
    ('FORMAT_PHYSICAL_FORM', 'Format / physical form', TRUE),
    ('STORAGE_CONDITIONS', 'Storage conditions', TRUE),
    ('SPECIES_CELL_SOURCE_OR_INTENDED_USE', 'Species (cell source or intended use)', TRUE),
    ('CHELATOR', 'Chelator', FALSE),
    ('PHENOL_RED', 'Phenol red', FALSE),
    ('ANIMAL_ORIGIN', 'Animal origin', FALSE),
    ('FORMULATION_ANIMAL_ORIGIN', 'Formulation / animal origin', FALSE),
    ('INTENDED_CELL_TYPE_APPLICATION', 'Intended cell type / application', FALSE),
    ('GRADE_REGULATORY', 'Grade / regulatory', FALSE),
    ('STERILITY_CONTAMINATION_QC', 'Sterility / contamination QC', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cell_tissue_culture' AS ASSIGNED_L2_ID,
    'Cells and Tissue Culture' AS ASSIGNED_L2_LABEL,
    'dissociation_reagents' AS ASSIGNED_L3_ID,
    'Dissociation Reagents' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Cells and Tissue Culture > Dissociation Reagents
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("CELL_LINE_NAME") AS "CELL_LINE_NAME",
    COUNT("REPOSITORY_CATALOG_NUMBER") AS "REPOSITORY_CATALOG_NUMBER",
    COUNT("CELL_LINE_SYNONYMS") AS "CELL_LINE_SYNONYMS",
    COUNT("ORGANISM") AS "ORGANISM",
    COUNT("TISSUE_ORGAN") AS "TISSUE_ORGAN",
    COUNT("GROWTH_PROPERTIES") AS "GROWTH_PROPERTIES",
    COUNT("BIOSAFETY_LEVEL") AS "BIOSAFETY_LEVEL",
    COUNT("AUTHENTICATION_STR_PROFILE") AS "AUTHENTICATION_STR_PROFILE",
    COUNT("FORMAT_PHYSICAL_FORM") AS "FORMAT_PHYSICAL_FORM",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("CELL_TYPE_MORPHOLOGY") AS "CELL_TYPE_MORPHOLOGY",
    COUNT("DISEASE") AS "DISEASE",
    COUNT("CONTAMINATION_VIRUS_STATUS") AS "CONTAMINATION_VIRUS_STATUS",
    COUNT("LICENSING_PERMITTED_USE") AS "LICENSING_PERMITTED_USE",
    COUNT("RECOMMENDED_MEDIUM_CONDITIONS") AS "RECOMMENDED_MEDIUM_CONDITIONS",
    COUNT("SUBCULTURING_RATIO") AS "SUBCULTURING_RATIO",
    COUNT("DONOR_AGE_SEX") AS "DONOR_AGE_SEX",
    COUNT("KARYOTYPE") AS "KARYOTYPE",
    COUNT("GRADE_REGULATORY") AS "GRADE_REGULATORY",
    COUNT("STERILITY_CONTAMINATION_QC") AS "STERILITY_CONTAMINATION_QC",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE"
    FROM SNOWFLAKE_LEARNING_DB."CELLS_TISSUE_CULTURE".CELL_LINES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('CELL_LINE_NAME', 'Cell line name', FALSE),
    ('REPOSITORY_CATALOG_NUMBER', 'Repository catalog number', FALSE),
    ('CELL_LINE_SYNONYMS', 'Cell line synonyms', FALSE),
    ('ORGANISM', 'Organism', TRUE),
    ('TISSUE_ORGAN', 'Tissue / organ', TRUE),
    ('GROWTH_PROPERTIES', 'Growth properties', TRUE),
    ('BIOSAFETY_LEVEL', 'Biosafety level', TRUE),
    ('AUTHENTICATION_STR_PROFILE', 'Authentication (STR profile)', TRUE),
    ('FORMAT_PHYSICAL_FORM', 'Format / physical form', TRUE),
    ('STORAGE_CONDITIONS', 'Storage conditions', TRUE),
    ('CELL_TYPE_MORPHOLOGY', 'Cell type / morphology', FALSE),
    ('DISEASE', 'Disease', FALSE),
    ('CONTAMINATION_VIRUS_STATUS', 'Contamination / virus status', FALSE),
    ('LICENSING_PERMITTED_USE', 'Licensing / permitted use', FALSE),
    ('RECOMMENDED_MEDIUM_CONDITIONS', 'Recommended medium & conditions', FALSE),
    ('SUBCULTURING_RATIO', 'Subculturing ratio', FALSE),
    ('DONOR_AGE_SEX', 'Donor age / sex', FALSE),
    ('KARYOTYPE', 'Karyotype', FALSE),
    ('GRADE_REGULATORY', 'Grade / regulatory', FALSE),
    ('STERILITY_CONTAMINATION_QC', 'Sterility / contamination QC', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cell_tissue_culture' AS ASSIGNED_L2_ID,
    'Cells and Tissue Culture' AS ASSIGNED_L2_LABEL,
    'cell_lines' AS ASSIGNED_L3_ID,
    'Cell Lines' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Cells and Tissue Culture > Cell Lines
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("TISSUE_SOURCE") AS "TISSUE_SOURCE",
    COUNT("CELL_TYPE") AS "CELL_TYPE",
    COUNT("DONOR_TYPE") AS "DONOR_TYPE",
    COUNT("PASSAGE_SUPPLIED") AS "PASSAGE_SUPPLIED",
    COUNT("CELL_NUMBER_PER_VIAL") AS "CELL_NUMBER_PER_VIAL",
    COUNT("FORMAT_PHYSICAL_FORM") AS "FORMAT_PHYSICAL_FORM",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("SPECIES_CELL_SOURCE_OR_INTENDED_USE") AS "SPECIES_CELL_SOURCE_OR_INTENDED_USE",
    COUNT("GUARANTEED_POPULATION_DOUBLINGS") AS "GUARANTEED_POPULATION_DOUBLINGS",
    COUNT("CHARACTERIZATION_MARKERS") AS "CHARACTERIZATION_MARKERS",
    COUNT("RECOMMENDED_MEDIUM") AS "RECOMMENDED_MEDIUM",
    COUNT("VIABILITY") AS "VIABILITY",
    COUNT("DONOR_INFECTIOUS_DISEASE_SCREENING") AS "DONOR_INFECTIOUS_DISEASE_SCREENING",
    COUNT("HLA_TYPED_DONOR") AS "HLA_TYPED_DONOR",
    COUNT("FORMULATION_ANIMAL_ORIGIN") AS "FORMULATION_ANIMAL_ORIGIN",
    COUNT("INTENDED_CELL_TYPE_APPLICATION") AS "INTENDED_CELL_TYPE_APPLICATION",
    COUNT("GRADE_REGULATORY") AS "GRADE_REGULATORY",
    COUNT("STERILITY_CONTAMINATION_QC") AS "STERILITY_CONTAMINATION_QC",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE"
    FROM SNOWFLAKE_LEARNING_DB."CELLS_TISSUE_CULTURE".PRIMARY_CELLS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('TISSUE_SOURCE', 'Tissue / source', TRUE),
    ('CELL_TYPE', 'Cell type', TRUE),
    ('DONOR_TYPE', 'Donor type', TRUE),
    ('PASSAGE_SUPPLIED', 'Passage supplied', TRUE),
    ('CELL_NUMBER_PER_VIAL', 'Cell number per vial', TRUE),
    ('FORMAT_PHYSICAL_FORM', 'Format / physical form', TRUE),
    ('STORAGE_CONDITIONS', 'Storage conditions', TRUE),
    ('SPECIES_CELL_SOURCE_OR_INTENDED_USE', 'Species (cell source or intended use)', TRUE),
    ('GUARANTEED_POPULATION_DOUBLINGS', 'Guaranteed population doublings', FALSE),
    ('CHARACTERIZATION_MARKERS', 'Characterization markers', FALSE),
    ('RECOMMENDED_MEDIUM', 'Recommended medium', FALSE),
    ('VIABILITY', 'Viability %', FALSE),
    ('DONOR_INFECTIOUS_DISEASE_SCREENING', 'Donor infectious-disease screening', FALSE),
    ('HLA_TYPED_DONOR', 'HLA-typed donor', FALSE),
    ('FORMULATION_ANIMAL_ORIGIN', 'Formulation / animal origin', FALSE),
    ('INTENDED_CELL_TYPE_APPLICATION', 'Intended cell type / application', FALSE),
    ('GRADE_REGULATORY', 'Grade / regulatory', FALSE),
    ('STERILITY_CONTAMINATION_QC', 'Sterility / contamination QC', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cell_tissue_culture' AS ASSIGNED_L2_ID,
    'Cells and Tissue Culture' AS ASSIGNED_L2_LABEL,
    'primary_cells' AS ASSIGNED_L3_ID,
    'Primary Cells' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Cells and Tissue Culture > Primary Cells
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("STEM_CELL_TYPE") AS "STEM_CELL_TYPE",
    COUNT("LICENSING_PERMITTED_USE") AS "LICENSING_PERMITTED_USE",
    COUNT("REPROGRAMMING_METHOD") AS "REPROGRAMMING_METHOD",
    COUNT("FORMAT_PHYSICAL_FORM") AS "FORMAT_PHYSICAL_FORM",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("SPECIES_CELL_SOURCE_OR_INTENDED_USE") AS "SPECIES_CELL_SOURCE_OR_INTENDED_USE",
    COUNT("PARENTAL_CELL_TISSUE_ORIGIN") AS "PARENTAL_CELL_TISSUE_ORIGIN",
    COUNT("DONOR_AGE_SEX_DISEASE") AS "DONOR_AGE_SEX_DISEASE",
    COUNT("CULTURE_SYSTEM") AS "CULTURE_SYSTEM",
    COUNT("KARYOTYPE") AS "KARYOTYPE",
    COUNT("PLURIPOTENCY_MARKERS") AS "PLURIPOTENCY_MARKERS",
    COUNT("DIFFERENTIATION_POTENTIAL") AS "DIFFERENTIATION_POTENTIAL",
    COUNT("DERIVED_CELL_TYPE_PURITY") AS "DERIVED_CELL_TYPE_PURITY",
    COUNT("GRADE_REGULATORY") AS "GRADE_REGULATORY",
    COUNT("STERILITY_CONTAMINATION_QC") AS "STERILITY_CONTAMINATION_QC",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE"
    FROM SNOWFLAKE_LEARNING_DB."CELLS_TISSUE_CULTURE".STEM_CELLS_IPSCS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('STEM_CELL_TYPE', 'Stem cell type', TRUE),
    ('LICENSING_PERMITTED_USE', 'Licensing / permitted use', TRUE),
    ('REPROGRAMMING_METHOD', 'Reprogramming method', TRUE),
    ('FORMAT_PHYSICAL_FORM', 'Format / physical form', TRUE),
    ('STORAGE_CONDITIONS', 'Storage conditions', TRUE),
    ('SPECIES_CELL_SOURCE_OR_INTENDED_USE', 'Species (cell source or intended use)', TRUE),
    ('PARENTAL_CELL_TISSUE_ORIGIN', 'Parental cell / tissue origin', FALSE),
    ('DONOR_AGE_SEX_DISEASE', 'Donor age / sex / disease', FALSE),
    ('CULTURE_SYSTEM', 'Culture system', FALSE),
    ('KARYOTYPE', 'Karyotype', FALSE),
    ('PLURIPOTENCY_MARKERS', 'Pluripotency markers', FALSE),
    ('DIFFERENTIATION_POTENTIAL', 'Differentiation potential', FALSE),
    ('DERIVED_CELL_TYPE_PURITY', 'Derived cell type & purity', FALSE),
    ('GRADE_REGULATORY', 'Grade / regulatory', FALSE),
    ('STERILITY_CONTAMINATION_QC', 'Sterility / contamination QC', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cell_tissue_culture' AS ASSIGNED_L2_ID,
    'Cells and Tissue Culture' AS ASSIGNED_L2_LABEL,
    'stem_cells_ipscs' AS ASSIGNED_L3_ID,
    'Stem Cells / iPSCs' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Cells and Tissue Culture > Stem Cells / iPSCs
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."CELLS_TISSUE_CULTURE".ORGANOID_GENERATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cell_tissue_culture' AS ASSIGNED_L2_ID,
    'Cells and Tissue Culture' AS ASSIGNED_L2_LABEL,
    'organoid_generation' AS ASSIGNED_L3_ID,
    'Organoid generation' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Cells and Tissue Culture > Organoid generation
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."CELLS_TISSUE_CULTURE".CUSTOM_CELL_CULTURE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cell_tissue_culture' AS ASSIGNED_L2_ID,
    'Cells and Tissue Culture' AS ASSIGNED_L2_LABEL,
    'custom_cell_culture' AS ASSIGNED_L3_ID,
    'Custom cell culture' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Cells and Tissue Culture > Custom cell culture
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_CENTRAL_LAB_SERVICES_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("LAB_REGION") AS "LAB_REGION",
    COUNT("THERAPEUTIC_AREA_CENTRAL_LAB") AS "THERAPEUTIC_AREA_CENTRAL_LAB",
    COUNT("DATA_DELIVERY_FORMAT") AS "DATA_DELIVERY_FORMAT",
    COUNT("DATA_TRANSFER_METHOD") AS "DATA_TRANSFER_METHOD",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DATA_DELIVERY_FREQUENCY") AS "DATA_DELIVERY_FREQUENCY",
    COUNT("SAFETY_TEST_DISCIPLINE") AS "SAFETY_TEST_DISCIPLINE",
    COUNT("HARMONIZED_PLATFORMS_ACROSS_SITES") AS "HARMONIZED_PLATFORMS_ACROSS_SITES",
    COUNT("HBA1C_CERTIFICATION") AS "HBA1C_CERTIFICATION",
    COUNT("ANALYZER_PLATFORM") AS "ANALYZER_PLATFORM",
    COUNT("RESULT_TURNAROUND") AS "RESULT_TURNAROUND",
    COUNT("CRITICAL_VALUE_FLAGGING") AS "CRITICAL_VALUE_FLAGGING",
    COUNT("GLOBAL_REFERENCE_RANGES") AS "GLOBAL_REFERENCE_RANGES",
    COUNT("SPONSOR_SPECIFIC_REFERENCE_RANGES") AS "SPONSOR_SPECIFIC_REFERENCE_RANGES",
    COUNT("TOXICITY_GRADING_SCALE") AS "TOXICITY_GRADING_SCALE",
    COUNT("RESULT_BLINDING") AS "RESULT_BLINDING",
    COUNT("INTER_LAB_COMPARISON_PROGRAM") AS "INTER_LAB_COMPARISON_PROGRAM"
    FROM SNOWFLAKE_LEARNING_DB."CENTRAL_LAB_SERVICES".ROUTINE_SAFETY_LABS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('LAB_REGION', 'Lab region', FALSE),
    ('THERAPEUTIC_AREA_CENTRAL_LAB', 'Therapeutic area (central lab)', FALSE),
    ('DATA_DELIVERY_FORMAT', 'Data delivery format', FALSE),
    ('DATA_TRANSFER_METHOD', 'Data transfer method', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('DATA_DELIVERY_FREQUENCY', 'Data delivery frequency', FALSE),
    ('SAFETY_TEST_DISCIPLINE', 'Safety test discipline', TRUE),
    ('HARMONIZED_PLATFORMS_ACROSS_SITES', 'Harmonized platforms across sites', FALSE),
    ('HBA1C_CERTIFICATION', 'HbA1c certification', FALSE),
    ('ANALYZER_PLATFORM', 'Analyzer platform', FALSE),
    ('RESULT_TURNAROUND', 'Result turnaround', FALSE),
    ('CRITICAL_VALUE_FLAGGING', 'Critical value flagging', FALSE),
    ('GLOBAL_REFERENCE_RANGES', 'Global reference ranges', FALSE),
    ('SPONSOR_SPECIFIC_REFERENCE_RANGES', 'Sponsor-specific reference ranges', FALSE),
    ('TOXICITY_GRADING_SCALE', 'Toxicity grading scale', FALSE),
    ('RESULT_BLINDING', 'Result blinding', FALSE),
    ('INTER_LAB_COMPARISON_PROGRAM', 'Inter-lab comparison program', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'central_lab_services' AS ASSIGNED_L2_ID,
    'Central Lab Services' AS ASSIGNED_L2_LABEL,
    'routine_safety_labs' AS ASSIGNED_L3_ID,
    'Routine Safety Labs' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Central Lab Services > Routine Safety Labs
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("LAB_REGION") AS "LAB_REGION",
    COUNT("THERAPEUTIC_AREA_CENTRAL_LAB") AS "THERAPEUTIC_AREA_CENTRAL_LAB",
    COUNT("DATA_DELIVERY_FORMAT") AS "DATA_DELIVERY_FORMAT",
    COUNT("DATA_TRANSFER_METHOD") AS "DATA_TRANSFER_METHOD",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DATA_DELIVERY_FREQUENCY") AS "DATA_DELIVERY_FREQUENCY",
    COUNT("SPECIALTY_DISCIPLINE") AS "SPECIALTY_DISCIPLINE",
    COUNT("DIGITAL_PATHOLOGY_SCANNER") AS "DIGITAL_PATHOLOGY_SCANNER",
    COUNT("PATHOLOGY_SERVICE") AS "PATHOLOGY_SERVICE",
    COUNT("TISSUE_SAMPLE_TYPE") AS "TISSUE_SAMPLE_TYPE",
    COUNT("COMPANION_DIAGNOSTIC_SUPPORT") AS "COMPANION_DIAGNOSTIC_SUPPORT",
    COUNT("MASS_SPEC_PLATFORM") AS "MASS_SPEC_PLATFORM",
    COUNT("VIRAL_TARGET") AS "VIRAL_TARGET",
    COUNT("TARGET_MARKER") AS "TARGET_MARKER",
    COUNT("MICROBIOLOGY_METHOD") AS "MICROBIOLOGY_METHOD",
    COUNT("AST_BREAKPOINT_STANDARD") AS "AST_BREAKPOINT_STANDARD"
    FROM SNOWFLAKE_LEARNING_DB."CENTRAL_LAB_SERVICES".SPECIALTY_ESOTERIC_TESTING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('LAB_REGION', 'Lab region', FALSE),
    ('THERAPEUTIC_AREA_CENTRAL_LAB', 'Therapeutic area (central lab)', FALSE),
    ('DATA_DELIVERY_FORMAT', 'Data delivery format', FALSE),
    ('DATA_TRANSFER_METHOD', 'Data transfer method', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('DATA_DELIVERY_FREQUENCY', 'Data delivery frequency', FALSE),
    ('SPECIALTY_DISCIPLINE', 'Specialty discipline', TRUE),
    ('DIGITAL_PATHOLOGY_SCANNER', 'Digital pathology scanner', FALSE),
    ('PATHOLOGY_SERVICE', 'Pathology service', FALSE),
    ('TISSUE_SAMPLE_TYPE', 'Tissue sample type', FALSE),
    ('COMPANION_DIAGNOSTIC_SUPPORT', 'Companion diagnostic support', FALSE),
    ('MASS_SPEC_PLATFORM', 'Mass spec platform', FALSE),
    ('VIRAL_TARGET', 'Viral target', FALSE),
    ('TARGET_MARKER', 'Target / marker', FALSE),
    ('MICROBIOLOGY_METHOD', 'Microbiology method', FALSE),
    ('AST_BREAKPOINT_STANDARD', 'AST breakpoint standard', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'central_lab_services' AS ASSIGNED_L2_ID,
    'Central Lab Services' AS ASSIGNED_L2_LABEL,
    'specialty_esoteric_testing' AS ASSIGNED_L3_ID,
    'Specialty / Esoteric Testing' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Central Lab Services > Specialty / Esoteric Testing
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("LAB_REGION") AS "LAB_REGION",
    COUNT("THERAPEUTIC_AREA_CENTRAL_LAB") AS "THERAPEUTIC_AREA_CENTRAL_LAB",
    COUNT("DATA_DELIVERY_FORMAT") AS "DATA_DELIVERY_FORMAT",
    COUNT("DATA_TRANSFER_METHOD") AS "DATA_TRANSFER_METHOD",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DATA_DELIVERY_FREQUENCY") AS "DATA_DELIVERY_FREQUENCY",
    COUNT("ASSAY_APPLICATION_CENTRAL_LAB_FLOW") AS "ASSAY_APPLICATION_CENTRAL_LAB_FLOW",
    COUNT("CUSTOM_PANEL_DEVELOPMENT") AS "CUSTOM_PANEL_DEVELOPMENT",
    COUNT("MAX_COLORS_PER_PANEL") AS "MAX_COLORS_PER_PANEL",
    COUNT("CYTOMETER_PLATFORM") AS "CYTOMETER_PLATFORM",
    COUNT("SPECTRAL_FLOW") AS "SPECTRAL_FLOW",
    COUNT("PBMC_ISOLATION") AS "PBMC_ISOLATION",
    COUNT("FLOW_SAMPLE_TYPE") AS "FLOW_SAMPLE_TYPE",
    COUNT("VALIDATION_GUIDELINE") AS "VALIDATION_GUIDELINE",
    COUNT("READY_MADE_PANELS") AS "READY_MADE_PANELS",
    COUNT("SAMPLE_STABILITY") AS "SAMPLE_STABILITY",
    COUNT("ANALYSIS_SOFTWARE") AS "ANALYSIS_SOFTWARE",
    COUNT("MASS_CYTOMETRY") AS "MASS_CYTOMETRY",
    COUNT("STABILIZATION_TUBES") AS "STABILIZATION_TUBES",
    COUNT("ASSAY_REGULATORY_STATUS") AS "ASSAY_REGULATORY_STATUS",
    COUNT("RAW_DATA_FILE_ACCESS") AS "RAW_DATA_FILE_ACCESS"
    FROM SNOWFLAKE_LEARNING_DB."CENTRAL_LAB_SERVICES".FLOW_CYTOMETRY_CENTRAL_LAB_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('LAB_REGION', 'Lab region', FALSE),
    ('THERAPEUTIC_AREA_CENTRAL_LAB', 'Therapeutic area (central lab)', FALSE),
    ('DATA_DELIVERY_FORMAT', 'Data delivery format', FALSE),
    ('DATA_TRANSFER_METHOD', 'Data transfer method', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('DATA_DELIVERY_FREQUENCY', 'Data delivery frequency', FALSE),
    ('ASSAY_APPLICATION_CENTRAL_LAB_FLOW', 'Assay application (central lab flow)', TRUE),
    ('CUSTOM_PANEL_DEVELOPMENT', 'Custom panel development', FALSE),
    ('MAX_COLORS_PER_PANEL', 'Max colors per panel', FALSE),
    ('CYTOMETER_PLATFORM', 'Cytometer platform', FALSE),
    ('SPECTRAL_FLOW', 'Spectral flow', FALSE),
    ('PBMC_ISOLATION', 'PBMC isolation', FALSE),
    ('FLOW_SAMPLE_TYPE', 'Flow sample type', FALSE),
    ('VALIDATION_GUIDELINE', 'Validation guideline', FALSE),
    ('READY_MADE_PANELS', 'Ready-made panels', FALSE),
    ('SAMPLE_STABILITY', 'Sample stability', FALSE),
    ('ANALYSIS_SOFTWARE', 'Analysis software', FALSE),
    ('MASS_CYTOMETRY', 'Mass cytometry', FALSE),
    ('STABILIZATION_TUBES', 'Stabilization tubes', FALSE),
    ('ASSAY_REGULATORY_STATUS', 'Assay regulatory status', FALSE),
    ('RAW_DATA_FILE_ACCESS', 'Raw data file access', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'central_lab_services' AS ASSIGNED_L2_ID,
    'Central Lab Services' AS ASSIGNED_L2_LABEL,
    'flow_cytometry_central_lab' AS ASSIGNED_L3_ID,
    'Flow Cytometry' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Central Lab Services > Flow Cytometry
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("LAB_REGION") AS "LAB_REGION",
    COUNT("THERAPEUTIC_AREA_CENTRAL_LAB") AS "THERAPEUTIC_AREA_CENTRAL_LAB",
    COUNT("DATA_DELIVERY_FORMAT") AS "DATA_DELIVERY_FORMAT",
    COUNT("DATA_TRANSFER_METHOD") AS "DATA_TRANSFER_METHOD",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DATA_DELIVERY_FREQUENCY") AS "DATA_DELIVERY_FREQUENCY",
    COUNT("KIT_BUILD_LOCATION") AS "KIT_BUILD_LOCATION",
    COUNT("CUSTOM_KIT_DESIGN") AS "CUSTOM_KIT_DESIGN",
    COUNT("SHIPPING_TEMPERATURE_KIT") AS "SHIPPING_TEMPERATURE_KIT",
    COUNT("SHIPMENT_TRACKING") AS "SHIPMENT_TRACKING",
    COUNT("REQUISITION_FORMS_PROVIDED") AS "REQUISITION_FORMS_PROVIDED",
    COUNT("KIT_LABELING") AS "KIT_LABELING",
    COUNT("KIT_ORDERING_PORTAL") AS "KIT_ORDERING_PORTAL",
    COUNT("START_UP_KIT_LEAD_TIME") AS "START_UP_KIT_LEAD_TIME",
    COUNT("RESUPPLY_TIME") AS "RESUPPLY_TIME",
    COUNT("AUTOMATIC_RESUPPLY") AS "AUTOMATIC_RESUPPLY",
    COUNT("KIT_EXPIRY_MANAGEMENT") AS "KIT_EXPIRY_MANAGEMENT",
    COUNT("COURIER") AS "COURIER",
    COUNT("VISIT_SPECIFIC_KITS") AS "VISIT_SPECIFIC_KITS",
    COUNT("ELECTRONIC_REQUISITION") AS "ELECTRONIC_REQUISITION",
    COUNT("HOME_COLLECTION_KITS") AS "HOME_COLLECTION_KITS",
    COUNT("LAB_ABNORMALITY_FOLLOW_UP_KITS") AS "LAB_ABNORMALITY_FOLLOW_UP_KITS"
    FROM SNOWFLAKE_LEARNING_DB."CENTRAL_LAB_SERVICES".KIT_BUILD_LOGISTICS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('LAB_REGION', 'Lab region', FALSE),
    ('THERAPEUTIC_AREA_CENTRAL_LAB', 'Therapeutic area (central lab)', FALSE),
    ('DATA_DELIVERY_FORMAT', 'Data delivery format', FALSE),
    ('DATA_TRANSFER_METHOD', 'Data transfer method', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('DATA_DELIVERY_FREQUENCY', 'Data delivery frequency', FALSE),
    ('KIT_BUILD_LOCATION', 'Kit build location', FALSE),
    ('CUSTOM_KIT_DESIGN', 'Custom kit design', FALSE),
    ('SHIPPING_TEMPERATURE_KIT', 'Shipping temperature (kit)', FALSE),
    ('SHIPMENT_TRACKING', 'Shipment tracking', FALSE),
    ('REQUISITION_FORMS_PROVIDED', 'Requisition forms provided', FALSE),
    ('KIT_LABELING', 'Kit labeling', FALSE),
    ('KIT_ORDERING_PORTAL', 'Kit ordering portal', FALSE),
    ('START_UP_KIT_LEAD_TIME', 'Start-up kit lead time', FALSE),
    ('RESUPPLY_TIME', 'Resupply time', FALSE),
    ('AUTOMATIC_RESUPPLY', 'Automatic resupply', FALSE),
    ('KIT_EXPIRY_MANAGEMENT', 'Kit expiry management', FALSE),
    ('COURIER', 'Courier', FALSE),
    ('VISIT_SPECIFIC_KITS', 'Visit-specific kits', FALSE),
    ('ELECTRONIC_REQUISITION', 'Electronic requisition', FALSE),
    ('HOME_COLLECTION_KITS', 'Home collection kits', FALSE),
    ('LAB_ABNORMALITY_FOLLOW_UP_KITS', 'Lab abnormality follow-up kits', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'central_lab_services' AS ASSIGNED_L2_ID,
    'Central Lab Services' AS ASSIGNED_L2_LABEL,
    'kit_build_logistics' AS ASSIGNED_L3_ID,
    'Kit Build & Logistics' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Central Lab Services > Kit Build & Logistics
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("LAB_REGION") AS "LAB_REGION",
    COUNT("THERAPEUTIC_AREA_CENTRAL_LAB") AS "THERAPEUTIC_AREA_CENTRAL_LAB",
    COUNT("DATA_DELIVERY_FORMAT") AS "DATA_DELIVERY_FORMAT",
    COUNT("DATA_TRANSFER_METHOD") AS "DATA_TRANSFER_METHOD",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DATA_DELIVERY_FREQUENCY") AS "DATA_DELIVERY_FREQUENCY",
    COUNT("SAMPLE_TRACKING_SYSTEM") AS "SAMPLE_TRACKING_SYSTEM",
    COUNT("STORAGE_TEMPERATURE_CENTRAL_LAB") AS "STORAGE_TEMPERATURE_CENTRAL_LAB",
    COUNT("LONG_TERM_STORAGE") AS "LONG_TERM_STORAGE",
    COUNT("BIOREPOSITORY_LOCATION") AS "BIOREPOSITORY_LOCATION",
    COUNT("TEMPERATURE_MONITORING") AS "TEMPERATURE_MONITORING",
    COUNT("PROCESSING_SERVICE") AS "PROCESSING_SERVICE",
    COUNT("DEFAULT_RETENTION_PERIOD") AS "DEFAULT_RETENTION_PERIOD",
    COUNT("CHAIN_OF_CUSTODY") AS "CHAIN_OF_CUSTODY",
    COUNT("SAMPLE_DESTRUCTION") AS "SAMPLE_DESTRUCTION",
    COUNT("BACKUP_POWER") AS "BACKUP_POWER",
    COUNT("BIOREPOSITORY_STANDARD") AS "BIOREPOSITORY_STANDARD",
    COUNT("STORED_SAMPLE_TYPE") AS "STORED_SAMPLE_TYPE",
    COUNT("THIRD_PARTY_SAMPLE_INTAKE") AS "THIRD_PARTY_SAMPLE_INTAKE",
    COUNT("CONSENT_TRACKING") AS "CONSENT_TRACKING",
    COUNT("RETROSPECTIVE_BIOSPECIMEN_SOURCING") AS "RETROSPECTIVE_BIOSPECIMEN_SOURCING"
    FROM SNOWFLAKE_LEARNING_DB."CENTRAL_LAB_SERVICES".SAMPLE_MANAGEMENT_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('LAB_REGION', 'Lab region', FALSE),
    ('THERAPEUTIC_AREA_CENTRAL_LAB', 'Therapeutic area (central lab)', FALSE),
    ('DATA_DELIVERY_FORMAT', 'Data delivery format', FALSE),
    ('DATA_TRANSFER_METHOD', 'Data transfer method', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('DATA_DELIVERY_FREQUENCY', 'Data delivery frequency', FALSE),
    ('SAMPLE_TRACKING_SYSTEM', 'Sample tracking system', FALSE),
    ('STORAGE_TEMPERATURE_CENTRAL_LAB', 'Storage temperature (central lab)', FALSE),
    ('LONG_TERM_STORAGE', 'Long-term storage', FALSE),
    ('BIOREPOSITORY_LOCATION', 'Biorepository location', FALSE),
    ('TEMPERATURE_MONITORING', 'Temperature monitoring', FALSE),
    ('PROCESSING_SERVICE', 'Processing service', FALSE),
    ('DEFAULT_RETENTION_PERIOD', 'Default retention period', FALSE),
    ('CHAIN_OF_CUSTODY', 'Chain of custody', FALSE),
    ('SAMPLE_DESTRUCTION', 'Sample destruction', FALSE),
    ('BACKUP_POWER', 'Backup power', FALSE),
    ('BIOREPOSITORY_STANDARD', 'Biorepository standard', FALSE),
    ('STORED_SAMPLE_TYPE', 'Stored sample type', FALSE),
    ('THIRD_PARTY_SAMPLE_INTAKE', 'Third-party sample intake', FALSE),
    ('CONSENT_TRACKING', 'Consent tracking', FALSE),
    ('RETROSPECTIVE_BIOSPECIMEN_SOURCING', 'Retrospective biospecimen sourcing', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'central_lab_services' AS ASSIGNED_L2_ID,
    'Central Lab Services' AS ASSIGNED_L2_LABEL,
    'sample_management' AS ASSIGNED_L3_ID,
    'Sample Management' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Central Lab Services > Sample Management
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_CHEMICALS_SOLVENTS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".ALCOHOLS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'alcohols' AS ASSIGNED_L3_ID,
    'Alcohols' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Alcohols
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".ORGANIC_ACIDS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'organic_acids' AS ASSIGNED_L3_ID,
    'Organic Acids' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Organic Acids
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".POLYOLS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'polyols' AS ASSIGNED_L3_ID,
    'Polyols' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Polyols
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".SODIUM_SALTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'sodium_salts' AS ASSIGNED_L3_ID,
    'Sodium Salts' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Sodium Salts
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".ESTERS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'esters' AS ASSIGNED_L3_ID,
    'Esters' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Esters
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".INORGANIC_COMPOUNDS_SALTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'inorganic_compounds_salts' AS ASSIGNED_L3_ID,
    'Inorganic Compounds & Salts' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Inorganic Compounds & Salts
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("CHEMISTRY") AS "CHEMISTRY",
    COUNT("NUMBER_OF_COMPONENTS") AS "NUMBER_OF_COMPONENTS",
    COUNT("MIX_RATIO") AS "MIX_RATIO",
    COUNT("CURE_METHOD") AS "CURE_METHOD",
    COUNT("FULL_CURE_TIME") AS "FULL_CURE_TIME",
    COUNT("WORK_LIFE") AS "WORK_LIFE",
    COUNT("VISCOSITY") AS "VISCOSITY",
    COUNT("COLOR") AS "COLOR",
    COUNT("OPERATING_TEMPERATURE") AS "OPERATING_TEMPERATURE",
    COUNT("SUBSTRATES") AS "SUBSTRATES",
    COUNT("SHEAR_STRENGTH") AS "SHEAR_STRENGTH",
    COUNT("CERTIFICATIONS") AS "CERTIFICATIONS",
    COUNT("PACKAGE_TYPE") AS "PACKAGE_TYPE",
    COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".COATINGS_SEALANTS_ADHESIVES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('CHEMISTRY', 'Chemistry', TRUE),
    ('NUMBER_OF_COMPONENTS', 'Number of components', FALSE),
    ('MIX_RATIO', 'Mix ratio', FALSE),
    ('CURE_METHOD', 'Cure method', FALSE),
    ('FULL_CURE_TIME', 'Full cure time', FALSE),
    ('WORK_LIFE', 'Work life', FALSE),
    ('VISCOSITY', 'Viscosity', FALSE),
    ('COLOR', 'Color', FALSE),
    ('OPERATING_TEMPERATURE', 'Operating temperature', FALSE),
    ('SUBSTRATES', 'Substrates', FALSE),
    ('SHEAR_STRENGTH', 'Shear strength', FALSE),
    ('CERTIFICATIONS', 'Certifications', FALSE),
    ('PACKAGE_TYPE', 'Package type', FALSE),
    ('CAS_NUMBER_S', 'CAS Number(s)', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'coatings_sealants_adhesives' AS ASSIGNED_L3_ID,
    'Coatings, Sealants & Adhesives' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Coatings, Sealants & Adhesives
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("RESIN_TYPE") AS "RESIN_TYPE",
    COUNT("APPLICATION") AS "APPLICATION",
    COUNT("KIT_CONTENTS") AS "KIT_CONTENTS",
    COUNT("MATRIX") AS "MATRIX",
    COUNT("FUNCTIONAL_GROUP") AS "FUNCTIONAL_GROUP",
    COUNT("IONIC_FORM") AS "IONIC_FORM",
    COUNT("CROSS_LINKAGE") AS "CROSS_LINKAGE",
    COUNT("PARTICLE_MESH_SIZE") AS "PARTICLE_MESH_SIZE",
    COUNT("LOADING_CAPACITY") AS "LOADING_CAPACITY",
    COUNT("MOISTURE_CONTENT") AS "MOISTURE_CONTENT",
    COUNT("VISCOSITY") AS "VISCOSITY",
    COUNT("POLYMERIZATION_TIME") AS "POLYMERIZATION_TIME",
    COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".RESINS_MODIFIERS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('RESIN_TYPE', 'Resin type', TRUE),
    ('APPLICATION', 'Application', FALSE),
    ('KIT_CONTENTS', 'Kit contents', FALSE),
    ('MATRIX', 'Matrix', FALSE),
    ('FUNCTIONAL_GROUP', 'Functional group', FALSE),
    ('IONIC_FORM', 'Ionic form', FALSE),
    ('CROSS_LINKAGE', 'Cross-linkage', FALSE),
    ('PARTICLE_MESH_SIZE', 'Particle / mesh size', FALSE),
    ('LOADING_CAPACITY', 'Loading / capacity', FALSE),
    ('MOISTURE_CONTENT', 'Moisture content', FALSE),
    ('VISCOSITY', 'Viscosity', FALSE),
    ('POLYMERIZATION_TIME', 'Polymerization time', FALSE),
    ('CAS_NUMBER_S', 'CAS Number(s)', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'resins_modifiers' AS ASSIGNED_L3_ID,
    'Resins & Modifiers' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Resins & Modifiers
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("TARGET_PATHWAY") AS "TARGET_PATHWAY",
    COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("MECHANISM_ACTIVITY") AS "MECHANISM_ACTIVITY",
    COUNT("SOLUBILITY") AS "SOLUBILITY",
    COUNT("SOLUTION_CONCENTRATION") AS "SOLUTION_CONCENTRATION",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".BIOACTIVE_SMALL_MOLECULES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('TARGET_PATHWAY', 'Target / Pathway', TRUE),
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('MECHANISM_ACTIVITY', 'Mechanism / activity', FALSE),
    ('SOLUBILITY', 'Solubility', FALSE),
    ('SOLUTION_CONCENTRATION', 'Solution concentration', FALSE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'bioactive_small_molecules' AS ASSIGNED_L3_ID,
    'Bioactive Small Molecules' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Bioactive Small Molecules
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("FORMAT") AS "FORMAT",
    COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("PHYSICOCHEMICAL_DESCRIPTORS") AS "PHYSICOCHEMICAL_DESCRIPTORS",
    COUNT("FUNCTIONAL_GROUP_WARHEAD") AS "FUNCTIONAL_GROUP_WARHEAD",
    COUNT("LIBRARY_COLLECTION_SIZE") AS "LIBRARY_COLLECTION_SIZE",
    COUNT("LIBRARY_TYPE") AS "LIBRARY_TYPE",
    COUNT("PURITY_TIER") AS "PURITY_TIER",
    COUNT("SOLUTION_CONCENTRATION") AS "SOLUTION_CONCENTRATION",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".BUILDING_BLOCKS_SCREENING_LIBRARIES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('FORMAT', 'Format', TRUE),
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('PHYSICOCHEMICAL_DESCRIPTORS', 'Physicochemical descriptors', FALSE),
    ('FUNCTIONAL_GROUP_WARHEAD', 'Functional group / warhead', FALSE),
    ('LIBRARY_COLLECTION_SIZE', 'Library / collection size', FALSE),
    ('LIBRARY_TYPE', 'Library type', FALSE),
    ('PURITY_TIER', 'Purity tier', FALSE),
    ('SOLUTION_CONCENTRATION', 'Solution concentration', FALSE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'building_blocks_screening_libraries' AS ASSIGNED_L3_ID,
    'Building Blocks & Screening Libraries' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Building Blocks & Screening Libraries
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SOLVENT_GRADE") AS "SOLVENT_GRADE",
    COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("WATER_CONTENT") AS "WATER_CONTENT",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".SOLVENTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SOLVENT_GRADE', 'Solvent grade', TRUE),
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('WATER_CONTENT', 'Water content', FALSE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'solvents' AS ASSIGNED_L3_ID,
    'Solvents' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Solvents
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PH") AS "PH",
    COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("CONCENTRATION_MOLARITY") AS "CONCENTRATION_MOLARITY",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".BUFFERS_PH_REAGENTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PH', 'pH', TRUE),
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('CONCENTRATION_MOLARITY', 'Concentration / molarity', FALSE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'buffers_ph_reagents' AS ASSIGNED_L3_ID,
    'Buffers & pH Reagents' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Buffers & pH Reagents
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("ISOTOPE") AS "ISOTOPE",
    COUNT("CAS_NUMBER_S") AS "CAS_NUMBER_S",
    COUNT("PURITY") AS "PURITY",
    COUNT("SPECIFIC_ACTIVITY") AS "SPECIFIC_ACTIVITY",
    COUNT("MOLECULAR_FORMULA") AS "MOLECULAR_FORMULA",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("PHYSICAL_FORM_STATE") AS "PHYSICAL_FORM_STATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("HAZARD_GHS_CLASSIFICATION") AS "HAZARD_GHS_CLASSIFICATION",
    COUNT("APPEARANCE_COLOR") AS "APPEARANCE_COLOR",
    COUNT("PURITY_ANALYSIS_METHOD") AS "PURITY_ANALYSIS_METHOD"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".RADIOCHEMISTRY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('ISOTOPE', 'Isotope', TRUE),
    ('CAS_NUMBER_S', 'CAS Number(s)', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('SPECIFIC_ACTIVITY', 'Specific activity', FALSE),
    ('MOLECULAR_FORMULA', 'Molecular Formula', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular Weight', FALSE),
    ('PHYSICAL_FORM_STATE', 'Physical Form / State', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('STORAGE_CONDITIONS', 'Storage Conditions', FALSE),
    ('HAZARD_GHS_CLASSIFICATION', 'Hazard / GHS classification', FALSE),
    ('APPEARANCE_COLOR', 'Appearance / Color', FALSE),
    ('PURITY_ANALYSIS_METHOD', 'Purity analysis method', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'radiochemistry' AS ASSIGNED_L3_ID,
    'Radiochemistry' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Radiochemistry
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SERVICE_TYPE") AS "SERVICE_TYPE",
    COUNT("ENGAGEMENT_MODEL") AS "ENGAGEMENT_MODEL",
    COUNT("COMPOUND_CLASS") AS "COMPOUND_CLASS",
    COUNT("ISOTOPE_LABELING") AS "ISOTOPE_LABELING",
    COUNT("MINIMUM_SCALE") AS "MINIMUM_SCALE",
    COUNT("MAXIMUM_SCALE") AS "MAXIMUM_SCALE",
    COUNT("ANALYTICAL_DATA_DELIVERED") AS "ANALYTICAL_DATA_DELIVERED",
    COUNT("CHIRAL_SEPARATION") AS "CHIRAL_SEPARATION",
    COUNT("ENABLING_CHEMISTRY_TECHNOLOGIES") AS "ENABLING_CHEMISTRY_TECHNOLOGIES",
    COUNT("DRUG_DISCOVERY_STAGE_SUPPORTED") AS "DRUG_DISCOVERY_STAGE_SUPPORTED",
    COUNT("COMPOUND_DESIGN_RESPONSIBILITY") AS "COMPOUND_DESIGN_RESPONSIBILITY",
    COUNT("GUARANTEED_MINIMUM_PURITY") AS "GUARANTEED_MINIMUM_PURITY",
    COUNT("ASYMMETRIC_SYNTHESIS") AS "ASYMMETRIC_SYNTHESIS",
    COUNT("SPECIAL_HANDLING_CAPABILITIES") AS "SPECIAL_HANDLING_CAPABILITIES",
    COUNT("GMP_SYNTHESIS_AVAILABLE") AS "GMP_SYNTHESIS_AVAILABLE",
    COUNT("DELIVERY_FORMAT") AS "DELIVERY_FORMAT"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".CUSTOM_MED_CHEM_SYNTHESIS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SERVICE_TYPE', 'Service type', TRUE),
    ('ENGAGEMENT_MODEL', 'Engagement model', FALSE),
    ('COMPOUND_CLASS', 'Compound class', FALSE),
    ('ISOTOPE_LABELING', 'Isotope labeling', FALSE),
    ('MINIMUM_SCALE', 'Minimum scale', FALSE),
    ('MAXIMUM_SCALE', 'Maximum scale', FALSE),
    ('ANALYTICAL_DATA_DELIVERED', 'Analytical data delivered', FALSE),
    ('CHIRAL_SEPARATION', 'Chiral separation', FALSE),
    ('ENABLING_CHEMISTRY_TECHNOLOGIES', 'Enabling chemistry technologies', FALSE),
    ('DRUG_DISCOVERY_STAGE_SUPPORTED', 'Drug discovery stage supported', FALSE),
    ('COMPOUND_DESIGN_RESPONSIBILITY', 'Compound design responsibility', FALSE),
    ('GUARANTEED_MINIMUM_PURITY', 'Guaranteed minimum purity (%)', FALSE),
    ('ASYMMETRIC_SYNTHESIS', 'Asymmetric synthesis', FALSE),
    ('SPECIAL_HANDLING_CAPABILITIES', 'Special handling capabilities', FALSE),
    ('GMP_SYNTHESIS_AVAILABLE', 'GMP synthesis available', FALSE),
    ('DELIVERY_FORMAT', 'Delivery format', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'custom_med_chem_synthesis' AS ASSIGNED_L3_ID,
    'Custom / med-chem synthesis' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Custom / med-chem synthesis
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("ANALYSIS_TYPE") AS "ANALYSIS_TYPE",
    COUNT("ANALYTICAL_TECHNIQUE") AS "ANALYTICAL_TECHNIQUE",
    COUNT("SAMPLE_TYPE") AS "SAMPLE_TYPE",
    COUNT("QUALITY_STANDARD") AS "QUALITY_STANDARD",
    COUNT("REFERENCE_METHOD") AS "REFERENCE_METHOD",
    COUNT("METHOD_STATUS") AS "METHOD_STATUS",
    COUNT("DETECTION_LEVEL") AS "DETECTION_LEVEL",
    COUNT("ELEMENTAL_METHOD") AS "ELEMENTAL_METHOD",
    COUNT("MASS_SPEC_ANALYZER") AS "MASS_SPEC_ANALYZER",
    COUNT("NMR_FIELD_STRENGTH_MHZ") AS "NMR_FIELD_STRENGTH_MHZ",
    COUNT("NMR_EXPERIMENTS") AS "NMR_EXPERIMENTS",
    COUNT("STEREOCHEMISTRY_DETERMINATION") AS "STEREOCHEMISTRY_DETERMINATION",
    COUNT("PREPARATIVE_ISOLATION_AVAILABLE") AS "PREPARATIVE_ISOLATION_AVAILABLE",
    COUNT("SAMPLE_HANDLING_CONSTRAINTS") AS "SAMPLE_HANDLING_CONSTRAINTS",
    COUNT("RUSH_TURNAROUND_AVAILABLE") AS "RUSH_TURNAROUND_AVAILABLE",
    COUNT("REPORT_FORMAT") AS "REPORT_FORMAT"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".ANALYTICAL_CHEMISTRY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('ANALYSIS_TYPE', 'Analysis type', TRUE),
    ('ANALYTICAL_TECHNIQUE', 'Analytical technique', FALSE),
    ('SAMPLE_TYPE', 'Sample type', FALSE),
    ('QUALITY_STANDARD', 'Quality standard', FALSE),
    ('REFERENCE_METHOD', 'Reference method', FALSE),
    ('METHOD_STATUS', 'Method status', FALSE),
    ('DETECTION_LEVEL', 'Detection level', FALSE),
    ('ELEMENTAL_METHOD', 'Elemental method', FALSE),
    ('MASS_SPEC_ANALYZER', 'Mass spec analyzer', FALSE),
    ('NMR_FIELD_STRENGTH_MHZ', 'NMR field strength (MHz)', FALSE),
    ('NMR_EXPERIMENTS', 'NMR experiments', FALSE),
    ('STEREOCHEMISTRY_DETERMINATION', 'Stereochemistry determination', FALSE),
    ('PREPARATIVE_ISOLATION_AVAILABLE', 'Preparative isolation available', FALSE),
    ('SAMPLE_HANDLING_CONSTRAINTS', 'Sample handling constraints', FALSE),
    ('RUSH_TURNAROUND_AVAILABLE', 'Rush turnaround available', FALSE),
    ('REPORT_FORMAT', 'Report format', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'analytical_chemistry' AS ASSIGNED_L3_ID,
    'Analytical chemistry' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Analytical chemistry
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".CHROMATOGRAPHY_PURIFICATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'chromatography_purification' AS ASSIGNED_L3_ID,
    'Chromatography & purification' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Chromatography & purification
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."CHEMICALS_SOLVENTS".PHYSICAL_STRUCTURAL_ANALYSIS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'chemicals_solvents' AS ASSIGNED_L2_ID,
    'Chemicals and Solvents' AS ASSIGNED_L2_LABEL,
    'physical_structural_analysis' AS ASSIGNED_L3_ID,
    'Physical / structural analysis' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Chemicals and Solvents > Physical / structural analysis
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_CLINICAL_TRIAL_SERVICES_CRO_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DELIVERY_MODEL") AS "DELIVERY_MODEL",
    COUNT("THERAPEUTIC_AREA_CLINICAL_TRIAL") AS "THERAPEUTIC_AREA_CLINICAL_TRIAL",
    COUNT("FSP_ENGAGEMENT_TYPE") AS "FSP_ENGAGEMENT_TYPE",
    COUNT("MONITORING_APPROACH") AS "MONITORING_APPROACH",
    COUNT("TRIAL_MODEL") AS "TRIAL_MODEL",
    COUNT("CENTRALIZED_MONITORING") AS "CENTRALIZED_MONITORING",
    COUNT("DECENTRALIZED_COMPONENTS") AS "DECENTRALIZED_COMPONENTS",
    COUNT("OVERSIGHT_PLATFORM") AS "OVERSIGHT_PLATFORM",
    COUNT("PM_DELIVERABLES") AS "PM_DELIVERABLES",
    COUNT("REMOTE_MONITORING_VISITS") AS "REMOTE_MONITORING_VISITS",
    COUNT("SOURCE_DATA_VERIFICATION_LEVEL") AS "SOURCE_DATA_VERIFICATION_LEVEL",
    COUNT("MONITORING_SETTING") AS "MONITORING_SETTING",
    COUNT("SPECIALTY_MONITORING") AS "SPECIALTY_MONITORING"
    FROM SNOWFLAKE_LEARNING_DB."CLINICAL_TRIAL_SERVICES".TRIAL_MANAGEMENT_MONITORING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', TRUE),
    ('DELIVERY_MODEL', 'Delivery model', FALSE),
    ('THERAPEUTIC_AREA_CLINICAL_TRIAL', 'Therapeutic area (clinical trial)', FALSE),
    ('FSP_ENGAGEMENT_TYPE', 'FSP engagement type', FALSE),
    ('MONITORING_APPROACH', 'Monitoring approach', FALSE),
    ('TRIAL_MODEL', 'Trial model', FALSE),
    ('CENTRALIZED_MONITORING', 'Centralized monitoring', FALSE),
    ('DECENTRALIZED_COMPONENTS', 'Decentralized components', FALSE),
    ('OVERSIGHT_PLATFORM', 'Oversight platform', FALSE),
    ('PM_DELIVERABLES', 'PM deliverables', FALSE),
    ('REMOTE_MONITORING_VISITS', 'Remote monitoring visits', FALSE),
    ('SOURCE_DATA_VERIFICATION_LEVEL', 'Source data verification level', FALSE),
    ('MONITORING_SETTING', 'Monitoring setting', FALSE),
    ('SPECIALTY_MONITORING', 'Specialty monitoring', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'clinical_trial_services_cro' AS ASSIGNED_L2_ID,
    'Clinical Trial Services' AS ASSIGNED_L2_LABEL,
    'trial_management_monitoring' AS ASSIGNED_L3_ID,
    'Trial Management & Monitoring' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Clinical Trial Services > Trial Management & Monitoring
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DELIVERY_MODEL") AS "DELIVERY_MODEL",
    COUNT("THERAPEUTIC_AREA_CLINICAL_TRIAL") AS "THERAPEUTIC_AREA_CLINICAL_TRIAL",
    COUNT("FSP_ENGAGEMENT_TYPE") AS "FSP_ENGAGEMENT_TYPE",
    COUNT("START_UP_ACTIVITIES") AS "START_UP_ACTIVITIES",
    COUNT("SITE_SELECTION_METHOD") AS "SITE_SELECTION_METHOD",
    COUNT("SITE_AUGMENTATION_STAFF") AS "SITE_AUGMENTATION_STAFF",
    COUNT("SITE_PORTAL") AS "SITE_PORTAL",
    COUNT("FEASIBILITY_DATA_SOURCES") AS "FEASIBILITY_DATA_SOURCES",
    COUNT("SITE_TYPES") AS "SITE_TYPES",
    COUNT("INVESTIGATOR_PAYMENTS") AS "INVESTIGATOR_PAYMENTS",
    COUNT("NETWORK_THERAPEUTIC_FOCUS") AS "NETWORK_THERAPEUTIC_FOCUS"
    FROM SNOWFLAKE_LEARNING_DB."CLINICAL_TRIAL_SERVICES".SITE_MANAGEMENT_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', TRUE),
    ('DELIVERY_MODEL', 'Delivery model', FALSE),
    ('THERAPEUTIC_AREA_CLINICAL_TRIAL', 'Therapeutic area (clinical trial)', FALSE),
    ('FSP_ENGAGEMENT_TYPE', 'FSP engagement type', FALSE),
    ('START_UP_ACTIVITIES', 'Start-up activities', FALSE),
    ('SITE_SELECTION_METHOD', 'Site selection method', FALSE),
    ('SITE_AUGMENTATION_STAFF', 'Site augmentation staff', FALSE),
    ('SITE_PORTAL', 'Site portal', FALSE),
    ('FEASIBILITY_DATA_SOURCES', 'Feasibility data sources', FALSE),
    ('SITE_TYPES', 'Site types', FALSE),
    ('INVESTIGATOR_PAYMENTS', 'Investigator payments', FALSE),
    ('NETWORK_THERAPEUTIC_FOCUS', 'Network therapeutic focus', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'clinical_trial_services_cro' AS ASSIGNED_L2_ID,
    'Clinical Trial Services' AS ASSIGNED_L2_LABEL,
    'site_management' AS ASSIGNED_L3_ID,
    'Site Management' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Clinical Trial Services > Site Management
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DELIVERY_MODEL") AS "DELIVERY_MODEL",
    COUNT("THERAPEUTIC_AREA_CLINICAL_TRIAL") AS "THERAPEUTIC_AREA_CLINICAL_TRIAL",
    COUNT("FSP_ENGAGEMENT_TYPE") AS "FSP_ENGAGEMENT_TYPE",
    COUNT("RECRUITMENT_CHANNELS") AS "RECRUITMENT_CHANNELS",
    COUNT("TARGET_POPULATIONS") AS "TARGET_POPULATIONS",
    COUNT("RETENTION_SERVICES") AS "RETENTION_SERVICES",
    COUNT("DIVERSITY_PLAN_SUPPORT") AS "DIVERSITY_PLAN_SUPPORT",
    COUNT("RECRUITMENT_TECHNOLOGY") AS "RECRUITMENT_TECHNOLOGY",
    COUNT("PRE_SCREENING") AS "PRE_SCREENING",
    COUNT("LANGUAGES_SUPPORTED_RECRUITMENT") AS "LANGUAGES_SUPPORTED_RECRUITMENT",
    COUNT("HOME_OR_REMOTE_ENROLLMENT") AS "HOME_OR_REMOTE_ENROLLMENT",
    COUNT("PATIENT_PAYMENTS") AS "PATIENT_PAYMENTS",
    COUNT("RECRUITMENT_COUNTRIES") AS "RECRUITMENT_COUNTRIES"
    FROM SNOWFLAKE_LEARNING_DB."CLINICAL_TRIAL_SERVICES".PATIENT_RECRUITMENT_RETENTION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', TRUE),
    ('DELIVERY_MODEL', 'Delivery model', FALSE),
    ('THERAPEUTIC_AREA_CLINICAL_TRIAL', 'Therapeutic area (clinical trial)', FALSE),
    ('FSP_ENGAGEMENT_TYPE', 'FSP engagement type', FALSE),
    ('RECRUITMENT_CHANNELS', 'Recruitment channels', FALSE),
    ('TARGET_POPULATIONS', 'Target populations', FALSE),
    ('RETENTION_SERVICES', 'Retention services', FALSE),
    ('DIVERSITY_PLAN_SUPPORT', 'Diversity plan support', FALSE),
    ('RECRUITMENT_TECHNOLOGY', 'Recruitment technology', FALSE),
    ('PRE_SCREENING', 'Pre-screening', FALSE),
    ('LANGUAGES_SUPPORTED_RECRUITMENT', 'Languages supported (recruitment)', FALSE),
    ('HOME_OR_REMOTE_ENROLLMENT', 'Home or remote enrollment', FALSE),
    ('PATIENT_PAYMENTS', 'Patient payments', FALSE),
    ('RECRUITMENT_COUNTRIES', 'Recruitment countries', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'clinical_trial_services_cro' AS ASSIGNED_L2_ID,
    'Clinical Trial Services' AS ASSIGNED_L2_LABEL,
    'patient_recruitment_retention' AS ASSIGNED_L3_ID,
    'Patient Recruitment & Retention' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Clinical Trial Services > Patient Recruitment & Retention
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DELIVERY_MODEL") AS "DELIVERY_MODEL",
    COUNT("THERAPEUTIC_AREA_CLINICAL_TRIAL") AS "THERAPEUTIC_AREA_CLINICAL_TRIAL",
    COUNT("FSP_ENGAGEMENT_TYPE") AS "FSP_ENGAGEMENT_TYPE",
    COUNT("CDM_DELIVERABLES") AS "CDM_DELIVERABLES",
    COUNT("EDC_PLATFORM") AS "EDC_PLATFORM",
    COUNT("DATA_STANDARDS_CDM") AS "DATA_STANDARDS_CDM",
    COUNT("EXTERNAL_DATA_SOURCES") AS "EXTERNAL_DATA_SOURCES",
    COUNT("SYSTEM_OWNERSHIP") AS "SYSTEM_OWNERSHIP",
    COUNT("DELIVERY_LOCATION") AS "DELIVERY_LOCATION",
    COUNT("GCP_GUIDELINE") AS "GCP_GUIDELINE",
    COUNT("24_HOUR_COVERAGE") AS "24_HOUR_COVERAGE"
    FROM SNOWFLAKE_LEARNING_DB."CLINICAL_TRIAL_SERVICES".CLINICAL_DATA_MANAGEMENT_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', TRUE),
    ('DELIVERY_MODEL', 'Delivery model', FALSE),
    ('THERAPEUTIC_AREA_CLINICAL_TRIAL', 'Therapeutic area (clinical trial)', FALSE),
    ('FSP_ENGAGEMENT_TYPE', 'FSP engagement type', FALSE),
    ('CDM_DELIVERABLES', 'CDM deliverables', FALSE),
    ('EDC_PLATFORM', 'EDC platform', FALSE),
    ('DATA_STANDARDS_CDM', 'Data standards (cdm)', FALSE),
    ('EXTERNAL_DATA_SOURCES', 'External data sources', FALSE),
    ('SYSTEM_OWNERSHIP', 'System ownership', FALSE),
    ('DELIVERY_LOCATION', 'Delivery location', FALSE),
    ('GCP_GUIDELINE', 'GCP guideline', FALSE),
    ('24_HOUR_COVERAGE', '24-hour coverage', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'clinical_trial_services_cro' AS ASSIGNED_L2_ID,
    'Clinical Trial Services' AS ASSIGNED_L2_LABEL,
    'clinical_data_management' AS ASSIGNED_L3_ID,
    'Clinical Data Management' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Clinical Trial Services > Clinical Data Management
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DELIVERY_MODEL") AS "DELIVERY_MODEL",
    COUNT("THERAPEUTIC_AREA_CLINICAL_TRIAL") AS "THERAPEUTIC_AREA_CLINICAL_TRIAL",
    COUNT("FSP_ENGAGEMENT_TYPE") AS "FSP_ENGAGEMENT_TYPE",
    COUNT("STATS_DELIVERABLES") AS "STATS_DELIVERABLES",
    COUNT("DMC_OR_DSMB_SUPPORT") AS "DMC_OR_DSMB_SUPPORT",
    COUNT("DATA_STANDARDS_BIOSTATS") AS "DATA_STANDARDS_BIOSTATS",
    COUNT("TRIAL_DESIGNS_SUPPORTED") AS "TRIAL_DESIGNS_SUPPORTED",
    COUNT("STATISTICAL_SOFTWARE") AS "STATISTICAL_SOFTWARE",
    COUNT("STATISTICAL_GUIDELINE") AS "STATISTICAL_GUIDELINE",
    COUNT("SUBMISSION_AGENCIES") AS "SUBMISSION_AGENCIES",
    COUNT("STUDY_TYPES") AS "STUDY_TYPES",
    COUNT("BLINDING_OF_OUTPUTS") AS "BLINDING_OF_OUTPUTS"
    FROM SNOWFLAKE_LEARNING_DB."CLINICAL_TRIAL_SERVICES".BIOSTATISTICS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', TRUE),
    ('DELIVERY_MODEL', 'Delivery model', FALSE),
    ('THERAPEUTIC_AREA_CLINICAL_TRIAL', 'Therapeutic area (clinical trial)', FALSE),
    ('FSP_ENGAGEMENT_TYPE', 'FSP engagement type', FALSE),
    ('STATS_DELIVERABLES', 'Stats deliverables', FALSE),
    ('DMC_OR_DSMB_SUPPORT', 'DMC or DSMB support', FALSE),
    ('DATA_STANDARDS_BIOSTATS', 'Data standards (biostats)', FALSE),
    ('TRIAL_DESIGNS_SUPPORTED', 'Trial designs supported', FALSE),
    ('STATISTICAL_SOFTWARE', 'Statistical software', FALSE),
    ('STATISTICAL_GUIDELINE', 'Statistical guideline', FALSE),
    ('SUBMISSION_AGENCIES', 'Submission agencies', FALSE),
    ('STUDY_TYPES', 'Study types', FALSE),
    ('BLINDING_OF_OUTPUTS', 'Blinding of outputs', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'clinical_trial_services_cro' AS ASSIGNED_L2_ID,
    'Clinical Trial Services' AS ASSIGNED_L2_LABEL,
    'biostatistics' AS ASSIGNED_L3_ID,
    'Biostatistics' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Clinical Trial Services > Biostatistics
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DELIVERY_MODEL") AS "DELIVERY_MODEL",
    COUNT("THERAPEUTIC_AREA_CLINICAL_TRIAL") AS "THERAPEUTIC_AREA_CLINICAL_TRIAL",
    COUNT("FSP_ENGAGEMENT_TYPE") AS "FSP_ENGAGEMENT_TYPE",
    COUNT("CLINICAL_DOCUMENT_TYPES") AS "CLINICAL_DOCUMENT_TYPES",
    COUNT("DISCLOSURE_DOCUMENTS") AS "DISCLOSURE_DOCUMENTS",
    COUNT("CSR_FORMAT") AS "CSR_FORMAT",
    COUNT("TARGET_AGENCIES_MEDWRITING_CRO") AS "TARGET_AGENCIES_MEDWRITING_CRO",
    COUNT("WRITING_GUIDELINE") AS "WRITING_GUIDELINE",
    COUNT("PRICING_BASIS") AS "PRICING_BASIS",
    COUNT("STANDALONE_QC_REVIEW") AS "STANDALONE_QC_REVIEW"
    FROM SNOWFLAKE_LEARNING_DB."CLINICAL_TRIAL_SERVICES".MEDICAL_WRITING_CRO_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', TRUE),
    ('DELIVERY_MODEL', 'Delivery model', FALSE),
    ('THERAPEUTIC_AREA_CLINICAL_TRIAL', 'Therapeutic area (clinical trial)', FALSE),
    ('FSP_ENGAGEMENT_TYPE', 'FSP engagement type', FALSE),
    ('CLINICAL_DOCUMENT_TYPES', 'Clinical document types', TRUE),
    ('DISCLOSURE_DOCUMENTS', 'Disclosure documents', FALSE),
    ('CSR_FORMAT', 'CSR format', FALSE),
    ('TARGET_AGENCIES_MEDWRITING_CRO', 'Target agencies (medwriting cro)', FALSE),
    ('WRITING_GUIDELINE', 'Writing guideline', FALSE),
    ('PRICING_BASIS', 'Pricing basis', FALSE),
    ('STANDALONE_QC_REVIEW', 'Standalone QC review', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'clinical_trial_services_cro' AS ASSIGNED_L2_ID,
    'Clinical Trial Services' AS ASSIGNED_L2_LABEL,
    'medical_writing_cro' AS ASSIGNED_L3_ID,
    'Medical Writing' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Clinical Trial Services > Medical Writing
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DELIVERY_MODEL") AS "DELIVERY_MODEL",
    COUNT("THERAPEUTIC_AREA_CLINICAL_TRIAL") AS "THERAPEUTIC_AREA_CLINICAL_TRIAL",
    COUNT("FSP_ENGAGEMENT_TYPE") AS "FSP_ENGAGEMENT_TYPE",
    COUNT("DISTRIBUTION_MODEL") AS "DISTRIBUTION_MODEL",
    COUNT("RETURNS_AND_DESTRUCTION") AS "RETURNS_AND_DESTRUCTION",
    COUNT("DEPOT_NETWORK") AS "DEPOT_NETWORK",
    COUNT("PACKAGING_FORMAT") AS "PACKAGING_FORMAT",
    COUNT("PACKAGING_LEVEL") AS "PACKAGING_LEVEL",
    COUNT("COMPARATOR_SOURCING") AS "COMPARATOR_SOURCING",
    COUNT("IRT_SYSTEM") AS "IRT_SYSTEM",
    COUNT("IMPORTER_OF_RECORD") AS "IMPORTER_OF_RECORD",
    COUNT("SUPPLY_FORECASTING") AS "SUPPLY_FORECASTING",
    COUNT("BLINDING_METHOD") AS "BLINDING_METHOD",
    COUNT("LABEL_TYPE") AS "LABEL_TYPE",
    COUNT("TEMPERATURE_RANGE_SUPPLY") AS "TEMPERATURE_RANGE_SUPPLY",
    COUNT("SOURCING_MODEL") AS "SOURCING_MODEL",
    COUNT("IRT_FUNCTIONS") AS "IRT_FUNCTIONS",
    COUNT("QP_RELEASE") AS "QP_RELEASE",
    COUNT("PRODUCT_MODALITY_SUPPLY") AS "PRODUCT_MODALITY_SUPPLY",
    COUNT("COMPLIANCE_STANDARD") AS "COMPLIANCE_STANDARD",
    COUNT("COMPARATOR_USE") AS "COMPARATOR_USE",
    COUNT("CONTROLLED_SUBSTANCES") AS "CONTROLLED_SUBSTANCES",
    COUNT("DIRECT_TO_PATIENT_COUNTRIES") AS "DIRECT_TO_PATIENT_COUNTRIES"
    FROM SNOWFLAKE_LEARNING_DB."CLINICAL_TRIAL_SERVICES".CLINICAL_SUPPLY_LOGISTICS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', TRUE),
    ('DELIVERY_MODEL', 'Delivery model', FALSE),
    ('THERAPEUTIC_AREA_CLINICAL_TRIAL', 'Therapeutic area (clinical trial)', FALSE),
    ('FSP_ENGAGEMENT_TYPE', 'FSP engagement type', FALSE),
    ('DISTRIBUTION_MODEL', 'Distribution model', FALSE),
    ('RETURNS_AND_DESTRUCTION', 'Returns and destruction', FALSE),
    ('DEPOT_NETWORK', 'Depot network', FALSE),
    ('PACKAGING_FORMAT', 'Packaging format', FALSE),
    ('PACKAGING_LEVEL', 'Packaging level', FALSE),
    ('COMPARATOR_SOURCING', 'Comparator sourcing', FALSE),
    ('IRT_SYSTEM', 'IRT system', FALSE),
    ('IMPORTER_OF_RECORD', 'Importer of record', FALSE),
    ('SUPPLY_FORECASTING', 'Supply forecasting', FALSE),
    ('BLINDING_METHOD', 'Blinding method', FALSE),
    ('LABEL_TYPE', 'Label type', FALSE),
    ('TEMPERATURE_RANGE_SUPPLY', 'Temperature range (supply)', FALSE),
    ('SOURCING_MODEL', 'Sourcing model', FALSE),
    ('IRT_FUNCTIONS', 'IRT functions', FALSE),
    ('QP_RELEASE', 'QP release', FALSE),
    ('PRODUCT_MODALITY_SUPPLY', 'Product modality (supply)', FALSE),
    ('COMPLIANCE_STANDARD', 'Compliance standard', FALSE),
    ('COMPARATOR_USE', 'Comparator use', FALSE),
    ('CONTROLLED_SUBSTANCES', 'Controlled substances', FALSE),
    ('DIRECT_TO_PATIENT_COUNTRIES', 'Direct-to-patient countries', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'clinical_trial_services_cro' AS ASSIGNED_L2_ID,
    'Clinical Trial Services' AS ASSIGNED_L2_LABEL,
    'clinical_supply_logistics' AS ASSIGNED_L3_ID,
    'Clinical Supply & Logistics' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Clinical Trial Services > Clinical Supply & Logistics
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_CMC_ANALYTICAL_QC_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_MODALITY") AS "PRODUCT_MODALITY",
    COUNT("GMP_STATUS") AS "GMP_STATUS",
    COUNT("MATERIAL_TESTED") AS "MATERIAL_TESTED",
    COUNT("DOSAGE_FORM") AS "DOSAGE_FORM",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PHARMACOPEIA") AS "PHARMACOPEIA",
    COUNT("TARGET_REGULATORY_AGENCY") AS "TARGET_REGULATORY_AGENCY",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("METHOD_SOURCE") AS "METHOD_SOURCE",
    COUNT("STANDARD_TURNAROUND") AS "STANDARD_TURNAROUND",
    COUNT("EXPEDITED_TURNAROUND_AVAILABLE") AS "EXPEDITED_TURNAROUND_AVAILABLE",
    COUNT("CONTROLLED_SUBSTANCES_ACCEPTED") AS "CONTROLLED_SUBSTANCES_ACCEPTED",
    COUNT("HIGHLY_POTENT_COMPOUNDS_ACCEPTED") AS "HIGHLY_POTENT_COMPOUNDS_ACCEPTED",
    COUNT("METHOD_LIFECYCLE_ACTIVITY") AS "METHOD_LIFECYCLE_ACTIVITY",
    COUNT("VALIDATION_GUIDELINE") AS "VALIDATION_GUIDELINE",
    COUNT("METHOD_PURPOSE") AS "METHOD_PURPOSE",
    COUNT("ANALYTICAL_TECHNIQUE") AS "ANALYTICAL_TECHNIQUE",
    COUNT("FORCED_DEGRADATION_STUDIES") AS "FORCED_DEGRADATION_STUDIES",
    COUNT("VALIDATION_APPROACH") AS "VALIDATION_APPROACH",
    COUNT("VALIDATION_PARAMETERS") AS "VALIDATION_PARAMETERS",
    COUNT("TRANSFER_LAB_ROLE") AS "TRANSFER_LAB_ROLE",
    COUNT("ANALYTICAL_QUALITY_BY_DESIGN") AS "ANALYTICAL_QUALITY_BY_DESIGN"
    FROM SNOWFLAKE_LEARNING_DB."CMC_ANALYTICAL_QC".METHOD_DEVELOPMENT_VALIDATION_CMC_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_MODALITY', 'Product modality', TRUE),
    ('GMP_STATUS', 'GMP status', FALSE),
    ('MATERIAL_TESTED', 'Material tested', FALSE),
    ('DOSAGE_FORM', 'Dosage form', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PHARMACOPEIA', 'Pharmacopeia', FALSE),
    ('TARGET_REGULATORY_AGENCY', 'Target regulatory agency', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('METHOD_SOURCE', 'Method source', FALSE),
    ('STANDARD_TURNAROUND', 'Standard turnaround', FALSE),
    ('EXPEDITED_TURNAROUND_AVAILABLE', 'Expedited turnaround available', FALSE),
    ('CONTROLLED_SUBSTANCES_ACCEPTED', 'Controlled substances accepted', FALSE),
    ('HIGHLY_POTENT_COMPOUNDS_ACCEPTED', 'Highly potent compounds accepted', FALSE),
    ('METHOD_LIFECYCLE_ACTIVITY', 'Method lifecycle activity', TRUE),
    ('VALIDATION_GUIDELINE', 'Validation guideline', FALSE),
    ('METHOD_PURPOSE', 'Method purpose', FALSE),
    ('ANALYTICAL_TECHNIQUE', 'Analytical technique', FALSE),
    ('FORCED_DEGRADATION_STUDIES', 'Forced degradation studies', FALSE),
    ('VALIDATION_APPROACH', 'Validation approach', FALSE),
    ('VALIDATION_PARAMETERS', 'Validation parameters', FALSE),
    ('TRANSFER_LAB_ROLE', 'Transfer lab role', FALSE),
    ('ANALYTICAL_QUALITY_BY_DESIGN', 'Analytical quality by design', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cmc_analytical_qc' AS ASSIGNED_L2_ID,
    'CMC, Analytical & QC' AS ASSIGNED_L2_LABEL,
    'method_development_validation_cmc' AS ASSIGNED_L3_ID,
    'Method Development & Validation' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- CMC, Analytical & QC > Method Development & Validation
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_MODALITY") AS "PRODUCT_MODALITY",
    COUNT("GMP_STATUS") AS "GMP_STATUS",
    COUNT("MATERIAL_TESTED") AS "MATERIAL_TESTED",
    COUNT("DOSAGE_FORM") AS "DOSAGE_FORM",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PHARMACOPEIA") AS "PHARMACOPEIA",
    COUNT("TARGET_REGULATORY_AGENCY") AS "TARGET_REGULATORY_AGENCY",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("METHOD_SOURCE") AS "METHOD_SOURCE",
    COUNT("STANDARD_TURNAROUND") AS "STANDARD_TURNAROUND",
    COUNT("EXPEDITED_TURNAROUND_AVAILABLE") AS "EXPEDITED_TURNAROUND_AVAILABLE",
    COUNT("CONTROLLED_SUBSTANCES_ACCEPTED") AS "CONTROLLED_SUBSTANCES_ACCEPTED",
    COUNT("HIGHLY_POTENT_COMPOUNDS_ACCEPTED") AS "HIGHLY_POTENT_COMPOUNDS_ACCEPTED",
    COUNT("STUDY_TYPE") AS "STUDY_TYPE",
    COUNT("PHOTOSTABILITY_TESTING") AS "PHOTOSTABILITY_TESTING",
    COUNT("STORAGE_CONDITION") AS "STORAGE_CONDITION",
    COUNT("STABILITY_GUIDELINE") AS "STABILITY_GUIDELINE",
    COUNT("PHOTOSTABILITY_OPTION") AS "PHOTOSTABILITY_OPTION",
    COUNT("STORAGE_ONLY_SERVICE") AS "STORAGE_ONLY_SERVICE",
    COUNT("PROGRAM_MANAGEMENT_SERVICES") AS "PROGRAM_MANAGEMENT_SERVICES",
    COUNT("STABILITY_OUTCOME") AS "STABILITY_OUTCOME"
    FROM SNOWFLAKE_LEARNING_DB."CMC_ANALYTICAL_QC".STABILITY_STUDIES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_MODALITY', 'Product modality', TRUE),
    ('GMP_STATUS', 'GMP status', FALSE),
    ('MATERIAL_TESTED', 'Material tested', FALSE),
    ('DOSAGE_FORM', 'Dosage form', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PHARMACOPEIA', 'Pharmacopeia', FALSE),
    ('TARGET_REGULATORY_AGENCY', 'Target regulatory agency', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('METHOD_SOURCE', 'Method source', FALSE),
    ('STANDARD_TURNAROUND', 'Standard turnaround', FALSE),
    ('EXPEDITED_TURNAROUND_AVAILABLE', 'Expedited turnaround available', FALSE),
    ('CONTROLLED_SUBSTANCES_ACCEPTED', 'Controlled substances accepted', FALSE),
    ('HIGHLY_POTENT_COMPOUNDS_ACCEPTED', 'Highly potent compounds accepted', FALSE),
    ('STUDY_TYPE', 'Study type', TRUE),
    ('PHOTOSTABILITY_TESTING', 'Photostability testing', FALSE),
    ('STORAGE_CONDITION', 'Storage condition', FALSE),
    ('STABILITY_GUIDELINE', 'Stability guideline', FALSE),
    ('PHOTOSTABILITY_OPTION', 'Photostability option', FALSE),
    ('STORAGE_ONLY_SERVICE', 'Storage-only service', FALSE),
    ('PROGRAM_MANAGEMENT_SERVICES', 'Program management services', FALSE),
    ('STABILITY_OUTCOME', 'Stability outcome', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cmc_analytical_qc' AS ASSIGNED_L2_ID,
    'CMC, Analytical & QC' AS ASSIGNED_L2_LABEL,
    'stability_studies' AS ASSIGNED_L3_ID,
    'Stability Studies' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- CMC, Analytical & QC > Stability Studies
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_MODALITY") AS "PRODUCT_MODALITY",
    COUNT("GMP_STATUS") AS "GMP_STATUS",
    COUNT("MATERIAL_TESTED") AS "MATERIAL_TESTED",
    COUNT("DOSAGE_FORM") AS "DOSAGE_FORM",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PHARMACOPEIA") AS "PHARMACOPEIA",
    COUNT("TARGET_REGULATORY_AGENCY") AS "TARGET_REGULATORY_AGENCY",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("METHOD_SOURCE") AS "METHOD_SOURCE",
    COUNT("STANDARD_TURNAROUND") AS "STANDARD_TURNAROUND",
    COUNT("EXPEDITED_TURNAROUND_AVAILABLE") AS "EXPEDITED_TURNAROUND_AVAILABLE",
    COUNT("CONTROLLED_SUBSTANCES_ACCEPTED") AS "CONTROLLED_SUBSTANCES_ACCEPTED",
    COUNT("HIGHLY_POTENT_COMPOUNDS_ACCEPTED") AS "HIGHLY_POTENT_COMPOUNDS_ACCEPTED",
    COUNT("RELEASE_TEST") AS "RELEASE_TEST",
    COUNT("RELEASE_SCOPE") AS "RELEASE_SCOPE",
    COUNT("CONTAINER_CLOSURE_INTEGRITY_METHOD") AS "CONTAINER_CLOSURE_INTEGRITY_METHOD",
    COUNT("EU_QP_CERTIFICATION") AS "EU_QP_CERTIFICATION",
    COUNT("EU_IMPORT_TESTING") AS "EU_IMPORT_TESTING",
    COUNT("SPECIFICATION_DEVELOPMENT_SUPPORT") AS "SPECIFICATION_DEVELOPMENT_SUPPORT",
    COUNT("OOS_INVESTIGATION_SUPPORT") AS "OOS_INVESTIGATION_SUPPORT"
    FROM SNOWFLAKE_LEARNING_DB."CMC_ANALYTICAL_QC".RELEASE_TESTING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_MODALITY', 'Product modality', TRUE),
    ('GMP_STATUS', 'GMP status', FALSE),
    ('MATERIAL_TESTED', 'Material tested', FALSE),
    ('DOSAGE_FORM', 'Dosage form', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PHARMACOPEIA', 'Pharmacopeia', FALSE),
    ('TARGET_REGULATORY_AGENCY', 'Target regulatory agency', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('METHOD_SOURCE', 'Method source', FALSE),
    ('STANDARD_TURNAROUND', 'Standard turnaround', FALSE),
    ('EXPEDITED_TURNAROUND_AVAILABLE', 'Expedited turnaround available', FALSE),
    ('CONTROLLED_SUBSTANCES_ACCEPTED', 'Controlled substances accepted', FALSE),
    ('HIGHLY_POTENT_COMPOUNDS_ACCEPTED', 'Highly potent compounds accepted', FALSE),
    ('RELEASE_TEST', 'Release test', TRUE),
    ('RELEASE_SCOPE', 'Release scope', FALSE),
    ('CONTAINER_CLOSURE_INTEGRITY_METHOD', 'Container closure integrity method', FALSE),
    ('EU_QP_CERTIFICATION', 'EU QP certification', FALSE),
    ('EU_IMPORT_TESTING', 'EU import testing', FALSE),
    ('SPECIFICATION_DEVELOPMENT_SUPPORT', 'Specification development support', FALSE),
    ('OOS_INVESTIGATION_SUPPORT', 'OOS investigation support', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cmc_analytical_qc' AS ASSIGNED_L2_ID,
    'CMC, Analytical & QC' AS ASSIGNED_L2_LABEL,
    'release_testing' AS ASSIGNED_L3_ID,
    'Release Testing' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- CMC, Analytical & QC > Release Testing
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_MODALITY") AS "PRODUCT_MODALITY",
    COUNT("GMP_STATUS") AS "GMP_STATUS",
    COUNT("MATERIAL_TESTED") AS "MATERIAL_TESTED",
    COUNT("DOSAGE_FORM") AS "DOSAGE_FORM",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PHARMACOPEIA") AS "PHARMACOPEIA",
    COUNT("TARGET_REGULATORY_AGENCY") AS "TARGET_REGULATORY_AGENCY",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("METHOD_SOURCE") AS "METHOD_SOURCE",
    COUNT("STANDARD_TURNAROUND") AS "STANDARD_TURNAROUND",
    COUNT("EXPEDITED_TURNAROUND_AVAILABLE") AS "EXPEDITED_TURNAROUND_AVAILABLE",
    COUNT("CONTROLLED_SUBSTANCES_ACCEPTED") AS "CONTROLLED_SUBSTANCES_ACCEPTED",
    COUNT("HIGHLY_POTENT_COMPOUNDS_ACCEPTED") AS "HIGHLY_POTENT_COMPOUNDS_ACCEPTED",
    COUNT("IMPURITY_CLASS") AS "IMPURITY_CLASS",
    COUNT("ELEMENTAL_IMPURITY_CHAPTER") AS "ELEMENTAL_IMPURITY_CHAPTER",
    COUNT("E_L_STANDARD") AS "E_L_STANDARD",
    COUNT("E_L_STUDY_TYPE") AS "E_L_STUDY_TYPE",
    COUNT("E_L_COMPONENT_TYPE") AS "E_L_COMPONENT_TYPE",
    COUNT("IMPURITY_GUIDELINE") AS "IMPURITY_GUIDELINE",
    COUNT("TOXICOLOGICAL_RISK_ASSESSMENT") AS "TOXICOLOGICAL_RISK_ASSESSMENT",
    COUNT("SAFETY_THRESHOLD_APPLIED") AS "SAFETY_THRESHOLD_APPLIED",
    COUNT("NITROSAMINE_LOQ") AS "NITROSAMINE_LOQ",
    COUNT("E_L_ANALYTICAL_TECHNIQUE") AS "E_L_ANALYTICAL_TECHNIQUE",
    COUNT("EXTRACTION_TECHNIQUE") AS "EXTRACTION_TECHNIQUE",
    COUNT("UNKNOWN_STRUCTURE_ELUCIDATION") AS "UNKNOWN_STRUCTURE_ELUCIDATION"
    FROM SNOWFLAKE_LEARNING_DB."CMC_ANALYTICAL_QC".IMPURITY_EL_TESTING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_MODALITY', 'Product modality', TRUE),
    ('GMP_STATUS', 'GMP status', FALSE),
    ('MATERIAL_TESTED', 'Material tested', FALSE),
    ('DOSAGE_FORM', 'Dosage form', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PHARMACOPEIA', 'Pharmacopeia', FALSE),
    ('TARGET_REGULATORY_AGENCY', 'Target regulatory agency', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('METHOD_SOURCE', 'Method source', FALSE),
    ('STANDARD_TURNAROUND', 'Standard turnaround', FALSE),
    ('EXPEDITED_TURNAROUND_AVAILABLE', 'Expedited turnaround available', FALSE),
    ('CONTROLLED_SUBSTANCES_ACCEPTED', 'Controlled substances accepted', FALSE),
    ('HIGHLY_POTENT_COMPOUNDS_ACCEPTED', 'Highly potent compounds accepted', FALSE),
    ('IMPURITY_CLASS', 'Impurity class', TRUE),
    ('ELEMENTAL_IMPURITY_CHAPTER', 'Elemental impurity chapter', FALSE),
    ('E_L_STANDARD', 'E&L standard', FALSE),
    ('E_L_STUDY_TYPE', 'E&L study type', FALSE),
    ('E_L_COMPONENT_TYPE', 'E&L component type', FALSE),
    ('IMPURITY_GUIDELINE', 'Impurity guideline', FALSE),
    ('TOXICOLOGICAL_RISK_ASSESSMENT', 'Toxicological risk assessment', FALSE),
    ('SAFETY_THRESHOLD_APPLIED', 'Safety threshold applied', FALSE),
    ('NITROSAMINE_LOQ', 'Nitrosamine LOQ', FALSE),
    ('E_L_ANALYTICAL_TECHNIQUE', 'E&L analytical technique', FALSE),
    ('EXTRACTION_TECHNIQUE', 'Extraction technique', FALSE),
    ('UNKNOWN_STRUCTURE_ELUCIDATION', 'Unknown structure elucidation', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cmc_analytical_qc' AS ASSIGNED_L2_ID,
    'CMC, Analytical & QC' AS ASSIGNED_L2_LABEL,
    'impurity_el_testing' AS ASSIGNED_L3_ID,
    'Impurity / E&L Testing' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- CMC, Analytical & QC > Impurity / E&L Testing
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_MODALITY") AS "PRODUCT_MODALITY",
    COUNT("GMP_STATUS") AS "GMP_STATUS",
    COUNT("MATERIAL_TESTED") AS "MATERIAL_TESTED",
    COUNT("DOSAGE_FORM") AS "DOSAGE_FORM",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PHARMACOPEIA") AS "PHARMACOPEIA",
    COUNT("TARGET_REGULATORY_AGENCY") AS "TARGET_REGULATORY_AGENCY",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("METHOD_SOURCE") AS "METHOD_SOURCE",
    COUNT("STANDARD_TURNAROUND") AS "STANDARD_TURNAROUND",
    COUNT("EXPEDITED_TURNAROUND_AVAILABLE") AS "EXPEDITED_TURNAROUND_AVAILABLE",
    COUNT("CONTROLLED_SUBSTANCES_ACCEPTED") AS "CONTROLLED_SUBSTANCES_ACCEPTED",
    COUNT("HIGHLY_POTENT_COMPOUNDS_ACCEPTED") AS "HIGHLY_POTENT_COMPOUNDS_ACCEPTED",
    COUNT("MICROBIOLOGY_TEST") AS "MICROBIOLOGY_TEST",
    COUNT("STERILITY_TEST_ENVIRONMENT") AS "STERILITY_TEST_ENVIRONMENT",
    COUNT("STERILITY_CHAPTER") AS "STERILITY_CHAPTER",
    COUNT("MICROBIAL_IDENTIFICATION_METHOD") AS "MICROBIAL_IDENTIFICATION_METHOD",
    COUNT("RAPID_MICROBIAL_METHOD_PLATFORM") AS "RAPID_MICROBIAL_METHOD_PLATFORM",
    COUNT("RAPID_METHOD_VALIDATION_BASIS") AS "RAPID_METHOD_VALIDATION_BASIS",
    COUNT("ENDOTOXIN_METHOD") AS "ENDOTOXIN_METHOD",
    COUNT("MYCOPLASMA_METHOD") AS "MYCOPLASMA_METHOD",
    COUNT("MICROBIAL_LIMITS_CHAPTER") AS "MICROBIAL_LIMITS_CHAPTER",
    COUNT("ENVIRONMENTAL_MONITORING_SAMPLE_TYPE") AS "ENVIRONMENTAL_MONITORING_SAMPLE_TYPE",
    COUNT("METHOD_SUITABILITY_TESTING") AS "METHOD_SUITABILITY_TESTING",
    COUNT("STERILITY_INCUBATION_PERIOD") AS "STERILITY_INCUBATION_PERIOD",
    COUNT("STERILITY_METHOD") AS "STERILITY_METHOD",
    COUNT("ENDOTOXIN_CHAPTER") AS "ENDOTOXIN_CHAPTER",
    COUNT("LOW_ENDOTOXIN_RECOVERY_STUDY") AS "LOW_ENDOTOXIN_RECOVERY_STUDY",
    COUNT("MYCOPLASMA_CHAPTER") AS "MYCOPLASMA_CHAPTER"
    FROM SNOWFLAKE_LEARNING_DB."CMC_ANALYTICAL_QC".MICROBIOLOGY_STERILITY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_MODALITY', 'Product modality', TRUE),
    ('GMP_STATUS', 'GMP status', FALSE),
    ('MATERIAL_TESTED', 'Material tested', FALSE),
    ('DOSAGE_FORM', 'Dosage form', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PHARMACOPEIA', 'Pharmacopeia', FALSE),
    ('TARGET_REGULATORY_AGENCY', 'Target regulatory agency', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('METHOD_SOURCE', 'Method source', FALSE),
    ('STANDARD_TURNAROUND', 'Standard turnaround', FALSE),
    ('EXPEDITED_TURNAROUND_AVAILABLE', 'Expedited turnaround available', FALSE),
    ('CONTROLLED_SUBSTANCES_ACCEPTED', 'Controlled substances accepted', FALSE),
    ('HIGHLY_POTENT_COMPOUNDS_ACCEPTED', 'Highly potent compounds accepted', FALSE),
    ('MICROBIOLOGY_TEST', 'Microbiology test', TRUE),
    ('STERILITY_TEST_ENVIRONMENT', 'Sterility test environment', FALSE),
    ('STERILITY_CHAPTER', 'Sterility chapter', FALSE),
    ('MICROBIAL_IDENTIFICATION_METHOD', 'Microbial identification method', FALSE),
    ('RAPID_MICROBIAL_METHOD_PLATFORM', 'Rapid microbial method platform', FALSE),
    ('RAPID_METHOD_VALIDATION_BASIS', 'Rapid method validation basis', FALSE),
    ('ENDOTOXIN_METHOD', 'Endotoxin method', FALSE),
    ('MYCOPLASMA_METHOD', 'Mycoplasma method', FALSE),
    ('MICROBIAL_LIMITS_CHAPTER', 'Microbial limits chapter', FALSE),
    ('ENVIRONMENTAL_MONITORING_SAMPLE_TYPE', 'Environmental monitoring sample type', FALSE),
    ('METHOD_SUITABILITY_TESTING', 'Method suitability testing', FALSE),
    ('STERILITY_INCUBATION_PERIOD', 'Sterility incubation period', FALSE),
    ('STERILITY_METHOD', 'Sterility method', FALSE),
    ('ENDOTOXIN_CHAPTER', 'Endotoxin chapter', FALSE),
    ('LOW_ENDOTOXIN_RECOVERY_STUDY', 'Low endotoxin recovery study', FALSE),
    ('MYCOPLASMA_CHAPTER', 'Mycoplasma chapter', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cmc_analytical_qc' AS ASSIGNED_L2_ID,
    'CMC, Analytical & QC' AS ASSIGNED_L2_LABEL,
    'microbiology_sterility' AS ASSIGNED_L3_ID,
    'Microbiology / Sterility' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- CMC, Analytical & QC > Microbiology / Sterility
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_MODALITY") AS "PRODUCT_MODALITY",
    COUNT("GMP_STATUS") AS "GMP_STATUS",
    COUNT("MATERIAL_TESTED") AS "MATERIAL_TESTED",
    COUNT("DOSAGE_FORM") AS "DOSAGE_FORM",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PHARMACOPEIA") AS "PHARMACOPEIA",
    COUNT("TARGET_REGULATORY_AGENCY") AS "TARGET_REGULATORY_AGENCY",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("METHOD_SOURCE") AS "METHOD_SOURCE",
    COUNT("STANDARD_TURNAROUND") AS "STANDARD_TURNAROUND",
    COUNT("EXPEDITED_TURNAROUND_AVAILABLE") AS "EXPEDITED_TURNAROUND_AVAILABLE",
    COUNT("CONTROLLED_SUBSTANCES_ACCEPTED") AS "CONTROLLED_SUBSTANCES_ACCEPTED",
    COUNT("HIGHLY_POTENT_COMPOUNDS_ACCEPTED") AS "HIGHLY_POTENT_COMPOUNDS_ACCEPTED",
    COUNT("MATERIAL_CATEGORY") AS "MATERIAL_CATEGORY",
    COUNT("TESTING_BASIS") AS "TESTING_BASIS",
    COUNT("RAW_MATERIAL_TEST") AS "RAW_MATERIAL_TEST",
    COUNT("COMPENDIUM") AS "COMPENDIUM",
    COUNT("FULL_MONOGRAPH_TESTING") AS "FULL_MONOGRAPH_TESTING",
    COUNT("IDENTITY_TECHNIQUE") AS "IDENTITY_TECHNIQUE",
    COUNT("SPECIFICATION_DEVELOPMENT_SUPPORT") AS "SPECIFICATION_DEVELOPMENT_SUPPORT"
    FROM SNOWFLAKE_LEARNING_DB."CMC_ANALYTICAL_QC".RAW_MATERIAL_TESTING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_MODALITY', 'Product modality', TRUE),
    ('GMP_STATUS', 'GMP status', FALSE),
    ('MATERIAL_TESTED', 'Material tested', FALSE),
    ('DOSAGE_FORM', 'Dosage form', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PHARMACOPEIA', 'Pharmacopeia', FALSE),
    ('TARGET_REGULATORY_AGENCY', 'Target regulatory agency', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('METHOD_SOURCE', 'Method source', FALSE),
    ('STANDARD_TURNAROUND', 'Standard turnaround', FALSE),
    ('EXPEDITED_TURNAROUND_AVAILABLE', 'Expedited turnaround available', FALSE),
    ('CONTROLLED_SUBSTANCES_ACCEPTED', 'Controlled substances accepted', FALSE),
    ('HIGHLY_POTENT_COMPOUNDS_ACCEPTED', 'Highly potent compounds accepted', FALSE),
    ('MATERIAL_CATEGORY', 'Material category', TRUE),
    ('TESTING_BASIS', 'Testing basis', FALSE),
    ('RAW_MATERIAL_TEST', 'Raw material test', FALSE),
    ('COMPENDIUM', 'Compendium', FALSE),
    ('FULL_MONOGRAPH_TESTING', 'Full monograph testing', FALSE),
    ('IDENTITY_TECHNIQUE', 'Identity technique', FALSE),
    ('SPECIFICATION_DEVELOPMENT_SUPPORT', 'Specification development support', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'cmc_analytical_qc' AS ASSIGNED_L2_ID,
    'CMC, Analytical & QC' AS ASSIGNED_L2_LABEL,
    'raw_material_testing' AS ASSIGNED_L3_ID,
    'Raw Material Testing' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- CMC, Analytical & QC > Raw Material Testing
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_DMPK_ADME_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("ANALYTICAL_PLATFORM") AS "ANALYTICAL_PLATFORM",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("ORDERING_FORMAT") AS "ORDERING_FORMAT",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("COMPOUND_AMOUNT_REQUIRED") AS "COMPOUND_AMOUNT_REQUIRED",
    COUNT("COMPOUND_FORMAT") AS "COMPOUND_FORMAT",
    COUNT("DELIVERABLE_FORMAT") AS "DELIVERABLE_FORMAT",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("ASSAY_TYPE") AS "ASSAY_TYPE",
    COUNT("TEST_SYSTEM") AS "TEST_SYSTEM",
    COUNT("READOUT_PARAMETERS") AS "READOUT_PARAMETERS",
    COUNT("SOLUBILITY_TYPE") AS "SOLUBILITY_TYPE",
    COUNT("STABILITY_MATRIX") AS "STABILITY_MATRIX",
    COUNT("TISSUE_SOURCE") AS "TISSUE_SOURCE",
    COUNT("DONOR_POOL_SIZE") AS "DONOR_POOL_SIZE",
    COUNT("DONOR_SEX") AS "DONOR_SEX",
    COUNT("METABOLIC_PHASE_COVERED") AS "METABOLIC_PHASE_COVERED",
    COUNT("TIME_POINTS") AS "TIME_POINTS",
    COUNT("LOW_CLEARANCE_METHOD") AS "LOW_CLEARANCE_METHOD",
    COUNT("IN_VIVO_CLEARANCE_PREDICTION") AS "IN_VIVO_CLEARANCE_PREDICTION",
    COUNT("METABOLITE_DETECTION_ADD_ON") AS "METABOLITE_DETECTION_ADD_ON",
    COUNT("LIPOPHILICITY_MEASURE") AS "LIPOPHILICITY_MEASURE",
    COUNT("TEST_CONCENTRATION") AS "TEST_CONCENTRATION",
    COUNT("REPLICATES") AS "REPLICATES"
    FROM SNOWFLAKE_LEARNING_DB."DMPK_ADME".IN_VITRO_ADME_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('SPECIES', 'Species', FALSE),
    ('ANALYTICAL_PLATFORM', 'Analytical platform', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('ORDERING_FORMAT', 'Ordering format', FALSE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('COMPOUND_AMOUNT_REQUIRED', 'Compound amount required', FALSE),
    ('COMPOUND_FORMAT', 'Compound format', FALSE),
    ('DELIVERABLE_FORMAT', 'Deliverable format', FALSE),
    ('GLP_STATUS', 'GLP status', FALSE),
    ('ASSAY_TYPE', 'Assay type', TRUE),
    ('TEST_SYSTEM', 'Test system', FALSE),
    ('READOUT_PARAMETERS', 'Readout parameters', FALSE),
    ('SOLUBILITY_TYPE', 'Solubility type', FALSE),
    ('STABILITY_MATRIX', 'Stability matrix', FALSE),
    ('TISSUE_SOURCE', 'Tissue source', FALSE),
    ('DONOR_POOL_SIZE', 'Donor pool size', FALSE),
    ('DONOR_SEX', 'Donor sex', FALSE),
    ('METABOLIC_PHASE_COVERED', 'Metabolic phase covered', FALSE),
    ('TIME_POINTS', 'Time points', FALSE),
    ('LOW_CLEARANCE_METHOD', 'Low-clearance method', FALSE),
    ('IN_VIVO_CLEARANCE_PREDICTION', 'In vivo clearance prediction', FALSE),
    ('METABOLITE_DETECTION_ADD_ON', 'Metabolite detection add-on', FALSE),
    ('LIPOPHILICITY_MEASURE', 'Lipophilicity measure', FALSE),
    ('TEST_CONCENTRATION', 'Test concentration', FALSE),
    ('REPLICATES', 'Replicates', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'dmpk_adme' AS ASSIGNED_L2_ID,
    'DMPK / ADME' AS ASSIGNED_L2_LABEL,
    'in_vitro_adme' AS ASSIGNED_L3_ID,
    'In Vitro ADME' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- DMPK / ADME > In Vitro ADME
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("ANALYTICAL_PLATFORM") AS "ANALYTICAL_PLATFORM",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("ORDERING_FORMAT") AS "ORDERING_FORMAT",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("COMPOUND_AMOUNT_REQUIRED") AS "COMPOUND_AMOUNT_REQUIRED",
    COUNT("COMPOUND_FORMAT") AS "COMPOUND_FORMAT",
    COUNT("DELIVERABLE_FORMAT") AS "DELIVERABLE_FORMAT",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("ROUTE_OF_ADMINISTRATION") AS "ROUTE_OF_ADMINISTRATION",
    COUNT("DOSING_DESIGN") AS "DOSING_DESIGN",
    COUNT("SAMPLING_MATRIX") AS "SAMPLING_MATRIX",
    COUNT("BLOOD_SAMPLING_METHOD") AS "BLOOD_SAMPLING_METHOD",
    COUNT("SURGICAL_OR_CANNULATED_MODEL") AS "SURGICAL_OR_CANNULATED_MODEL",
    COUNT("DOSE_REGIMEN") AS "DOSE_REGIMEN",
    COUNT("TISSUE_DISTRIBUTION") AS "TISSUE_DISTRIBUTION",
    COUNT("EXCRETION_STUDY") AS "EXCRETION_STUDY",
    COUNT("FORMULATION_SUPPORT") AS "FORMULATION_SUPPORT",
    COUNT("ANIMAL_MODEL_TYPE") AS "ANIMAL_MODEL_TYPE",
    COUNT("PK_ANALYSIS_APPROACH") AS "PK_ANALYSIS_APPROACH",
    COUNT("PK_SOFTWARE") AS "PK_SOFTWARE",
    COUNT("TK_ALONGSIDE_TOX") AS "TK_ALONGSIDE_TOX",
    COUNT("STRAIN") AS "STRAIN",
    COUNT("MAX_COMPOUNDS_PER_CASSETTE") AS "MAX_COMPOUNDS_PER_CASSETTE",
    COUNT("ANIMALS_PER_GROUP") AS "ANIMALS_PER_GROUP",
    COUNT("PK_PARAMETERS") AS "PK_PARAMETERS",
    COUNT("MAX_INFUSION_DURATION") AS "MAX_INFUSION_DURATION"
    FROM SNOWFLAKE_LEARNING_DB."DMPK_ADME".IN_VIVO_PK_TK_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('SPECIES', 'Species', FALSE),
    ('ANALYTICAL_PLATFORM', 'Analytical platform', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('ORDERING_FORMAT', 'Ordering format', FALSE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('COMPOUND_AMOUNT_REQUIRED', 'Compound amount required', FALSE),
    ('COMPOUND_FORMAT', 'Compound format', FALSE),
    ('DELIVERABLE_FORMAT', 'Deliverable format', FALSE),
    ('GLP_STATUS', 'GLP status', FALSE),
    ('ROUTE_OF_ADMINISTRATION', 'Route of administration', FALSE),
    ('DOSING_DESIGN', 'Dosing design', FALSE),
    ('SAMPLING_MATRIX', 'Sampling matrix', FALSE),
    ('BLOOD_SAMPLING_METHOD', 'Blood sampling method', FALSE),
    ('SURGICAL_OR_CANNULATED_MODEL', 'Surgical or cannulated model', FALSE),
    ('DOSE_REGIMEN', 'Dose regimen', FALSE),
    ('TISSUE_DISTRIBUTION', 'Tissue distribution', FALSE),
    ('EXCRETION_STUDY', 'Excretion study', FALSE),
    ('FORMULATION_SUPPORT', 'Formulation support', FALSE),
    ('ANIMAL_MODEL_TYPE', 'Animal model type', FALSE),
    ('PK_ANALYSIS_APPROACH', 'PK analysis approach', FALSE),
    ('PK_SOFTWARE', 'PK software', FALSE),
    ('TK_ALONGSIDE_TOX', 'TK alongside tox', FALSE),
    ('STRAIN', 'Strain', FALSE),
    ('MAX_COMPOUNDS_PER_CASSETTE', 'Max compounds per cassette', FALSE),
    ('ANIMALS_PER_GROUP', 'Animals per group', FALSE),
    ('PK_PARAMETERS', 'PK parameters', FALSE),
    ('MAX_INFUSION_DURATION', 'Max infusion duration', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'dmpk_adme' AS ASSIGNED_L2_ID,
    'DMPK / ADME' AS ASSIGNED_L2_LABEL,
    'in_vivo_pk_tk' AS ASSIGNED_L3_ID,
    'In Vivo PK / TK' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- DMPK / ADME > In Vivo PK / TK
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("ANALYTICAL_PLATFORM") AS "ANALYTICAL_PLATFORM",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("ORDERING_FORMAT") AS "ORDERING_FORMAT",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("COMPOUND_AMOUNT_REQUIRED") AS "COMPOUND_AMOUNT_REQUIRED",
    COUNT("COMPOUND_FORMAT") AS "COMPOUND_FORMAT",
    COUNT("DELIVERABLE_FORMAT") AS "DELIVERABLE_FORMAT",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("STUDY_TYPE") AS "STUDY_TYPE",
    COUNT("RADIOLABELED_TEST_ARTICLE") AS "RADIOLABELED_TEST_ARTICLE",
    COUNT("TEST_SYSTEM") AS "TEST_SYSTEM",
    COUNT("IN_VIVO_SAMPLE_MATRIX") AS "IN_VIVO_SAMPLE_MATRIX",
    COUNT("CROSS_SPECIES_COMPARISON") AS "CROSS_SPECIES_COMPARISON",
    COUNT("MS_PLATFORM") AS "MS_PLATFORM",
    COUNT("ISOTOPE") AS "ISOTOPE",
    COUNT("RADIOSYNTHESIS_IN_HOUSE") AS "RADIOSYNTHESIS_IN_HOUSE",
    COUNT("RADIO_DETECTION") AS "RADIO_DETECTION",
    COUNT("SERVICE_LEVEL") AS "SERVICE_LEVEL",
    COUNT("MASS_BALANCE") AS "MASS_BALANCE",
    COUNT("QWBA") AS "QWBA",
    COUNT("MIST_COVERAGE") AS "MIST_COVERAGE",
    COUNT("TRAPPING_AGENT") AS "TRAPPING_AGENT",
    COUNT("NMR_STRUCTURE_CONFIRMATION") AS "NMR_STRUCTURE_CONFIRMATION",
    COUNT("METABOLITE_BIOSYNTHESIS") AS "METABOLITE_BIOSYNTHESIS",
    COUNT("TEST_CONCENTRATION") AS "TEST_CONCENTRATION"
    FROM SNOWFLAKE_LEARNING_DB."DMPK_ADME".METABOLITE_IDENTIFICATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('SPECIES', 'Species', FALSE),
    ('ANALYTICAL_PLATFORM', 'Analytical platform', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('ORDERING_FORMAT', 'Ordering format', FALSE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('COMPOUND_AMOUNT_REQUIRED', 'Compound amount required', FALSE),
    ('COMPOUND_FORMAT', 'Compound format', FALSE),
    ('DELIVERABLE_FORMAT', 'Deliverable format', FALSE),
    ('GLP_STATUS', 'GLP status', FALSE),
    ('STUDY_TYPE', 'Study type', TRUE),
    ('RADIOLABELED_TEST_ARTICLE', 'Radiolabeled test article', FALSE),
    ('TEST_SYSTEM', 'Test system', FALSE),
    ('IN_VIVO_SAMPLE_MATRIX', 'In vivo sample matrix', FALSE),
    ('CROSS_SPECIES_COMPARISON', 'Cross-species comparison', FALSE),
    ('MS_PLATFORM', 'MS platform', FALSE),
    ('ISOTOPE', 'Isotope', FALSE),
    ('RADIOSYNTHESIS_IN_HOUSE', 'Radiosynthesis in-house', FALSE),
    ('RADIO_DETECTION', 'Radio detection', FALSE),
    ('SERVICE_LEVEL', 'Service level', FALSE),
    ('MASS_BALANCE', 'Mass balance', FALSE),
    ('QWBA', 'QWBA', FALSE),
    ('MIST_COVERAGE', 'MIST coverage', FALSE),
    ('TRAPPING_AGENT', 'Trapping agent', FALSE),
    ('NMR_STRUCTURE_CONFIRMATION', 'NMR structure confirmation', FALSE),
    ('METABOLITE_BIOSYNTHESIS', 'Metabolite biosynthesis', FALSE),
    ('TEST_CONCENTRATION', 'Test concentration', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'dmpk_adme' AS ASSIGNED_L2_ID,
    'DMPK / ADME' AS ASSIGNED_L2_LABEL,
    'metabolite_identification' AS ASSIGNED_L3_ID,
    'Metabolite Identification' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- DMPK / ADME > Metabolite Identification
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("ANALYTICAL_PLATFORM") AS "ANALYTICAL_PLATFORM",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("ORDERING_FORMAT") AS "ORDERING_FORMAT",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("COMPOUND_AMOUNT_REQUIRED") AS "COMPOUND_AMOUNT_REQUIRED",
    COUNT("COMPOUND_FORMAT") AS "COMPOUND_FORMAT",
    COUNT("DELIVERABLE_FORMAT") AS "DELIVERABLE_FORMAT",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("ASSAY_TYPE") AS "ASSAY_TYPE",
    COUNT("NON_CYP_ENZYMES") AS "NON_CYP_ENZYMES",
    COUNT("CYP_ISOFORMS") AS "CYP_ISOFORMS",
    COUNT("INHIBITION_PARAMETER") AS "INHIBITION_PARAMETER",
    COUNT("INDUCTION_TEST_SYSTEM") AS "INDUCTION_TEST_SYSTEM",
    COUNT("INDUCTION_ENDPOINT") AS "INDUCTION_ENDPOINT",
    COUNT("PHENOTYPING_METHOD") AS "PHENOTYPING_METHOD",
    COUNT("REGULATORY_GUIDANCE") AS "REGULATORY_GUIDANCE",
    COUNT("NUCLEAR_RECEPTOR_ASSAY") AS "NUCLEAR_RECEPTOR_ASSAY",
    COUNT("PROBE_FORMAT") AS "PROBE_FORMAT",
    COUNT("INHIBITOR_CONCENTRATION_RANGE") AS "INHIBITOR_CONCENTRATION_RANGE",
    COUNT("INDUCTION_PARAMETERS") AS "INDUCTION_PARAMETERS",
    COUNT("RISK_ASSESSMENT_OR_MODELING") AS "RISK_ASSESSMENT_OR_MODELING",
    COUNT("HEPATOCYTE_DONORS") AS "HEPATOCYTE_DONORS",
    COUNT("UGT_ISOFORMS") AS "UGT_ISOFORMS"
    FROM SNOWFLAKE_LEARNING_DB."DMPK_ADME".DRUG_DRUG_INTERACTION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('SPECIES', 'Species', FALSE),
    ('ANALYTICAL_PLATFORM', 'Analytical platform', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('ORDERING_FORMAT', 'Ordering format', FALSE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('COMPOUND_AMOUNT_REQUIRED', 'Compound amount required', FALSE),
    ('COMPOUND_FORMAT', 'Compound format', FALSE),
    ('DELIVERABLE_FORMAT', 'Deliverable format', FALSE),
    ('GLP_STATUS', 'GLP status', FALSE),
    ('ASSAY_TYPE', 'Assay type', TRUE),
    ('NON_CYP_ENZYMES', 'Non-CYP enzymes', FALSE),
    ('CYP_ISOFORMS', 'CYP isoforms', FALSE),
    ('INHIBITION_PARAMETER', 'Inhibition parameter', FALSE),
    ('INDUCTION_TEST_SYSTEM', 'Induction test system', FALSE),
    ('INDUCTION_ENDPOINT', 'Induction endpoint', FALSE),
    ('PHENOTYPING_METHOD', 'Phenotyping method', FALSE),
    ('REGULATORY_GUIDANCE', 'Regulatory guidance', FALSE),
    ('NUCLEAR_RECEPTOR_ASSAY', 'Nuclear receptor assay', FALSE),
    ('PROBE_FORMAT', 'Probe format', FALSE),
    ('INHIBITOR_CONCENTRATION_RANGE', 'Inhibitor concentration range', FALSE),
    ('INDUCTION_PARAMETERS', 'Induction parameters', FALSE),
    ('RISK_ASSESSMENT_OR_MODELING', 'Risk assessment or modeling', FALSE),
    ('HEPATOCYTE_DONORS', 'Hepatocyte donors', FALSE),
    ('UGT_ISOFORMS', 'UGT isoforms', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'dmpk_adme' AS ASSIGNED_L2_ID,
    'DMPK / ADME' AS ASSIGNED_L2_LABEL,
    'drug_drug_interaction' AS ASSIGNED_L3_ID,
    'Drug-Drug Interaction' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- DMPK / ADME > Drug-Drug Interaction
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("ANALYTICAL_PLATFORM") AS "ANALYTICAL_PLATFORM",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("ORDERING_FORMAT") AS "ORDERING_FORMAT",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("COMPOUND_AMOUNT_REQUIRED") AS "COMPOUND_AMOUNT_REQUIRED",
    COUNT("COMPOUND_FORMAT") AS "COMPOUND_FORMAT",
    COUNT("DELIVERABLE_FORMAT") AS "DELIVERABLE_FORMAT",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("METHOD") AS "METHOD",
    COUNT("MATRIX") AS "MATRIX",
    COUNT("READOUT") AS "READOUT",
    COUNT("BLOOD_TO_PLASMA_RATIO") AS "BLOOD_TO_PLASMA_RATIO",
    COUNT("TEST_CONCENTRATION") AS "TEST_CONCENTRATION",
    COUNT("DERIVED_UNBOUND_PARAMETERS") AS "DERIVED_UNBOUND_PARAMETERS",
    COUNT("ANTICOAGULANT") AS "ANTICOAGULANT",
    COUNT("INCUBATION_TIME") AS "INCUBATION_TIME",
    COUNT("REPLICATES") AS "REPLICATES",
    COUNT("TISSUE_DILUTION_FACTOR") AS "TISSUE_DILUTION_FACTOR",
    COUNT("DYNAMIC_RANGE") AS "DYNAMIC_RANGE"
    FROM SNOWFLAKE_LEARNING_DB."DMPK_ADME".PROTEIN_BINDING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('SPECIES', 'Species', FALSE),
    ('ANALYTICAL_PLATFORM', 'Analytical platform', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('ORDERING_FORMAT', 'Ordering format', FALSE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('COMPOUND_AMOUNT_REQUIRED', 'Compound amount required', FALSE),
    ('COMPOUND_FORMAT', 'Compound format', FALSE),
    ('DELIVERABLE_FORMAT', 'Deliverable format', FALSE),
    ('GLP_STATUS', 'GLP status', FALSE),
    ('METHOD', 'Method', FALSE),
    ('MATRIX', 'Matrix', FALSE),
    ('READOUT', 'Readout', FALSE),
    ('BLOOD_TO_PLASMA_RATIO', 'Blood-to-plasma ratio', FALSE),
    ('TEST_CONCENTRATION', 'Test concentration', FALSE),
    ('DERIVED_UNBOUND_PARAMETERS', 'Derived unbound parameters', FALSE),
    ('ANTICOAGULANT', 'Anticoagulant', FALSE),
    ('INCUBATION_TIME', 'Incubation time', FALSE),
    ('REPLICATES', 'Replicates', FALSE),
    ('TISSUE_DILUTION_FACTOR', 'Tissue dilution factor', FALSE),
    ('DYNAMIC_RANGE', 'Dynamic range', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'dmpk_adme' AS ASSIGNED_L2_ID,
    'DMPK / ADME' AS ASSIGNED_L2_LABEL,
    'protein_binding' AS ASSIGNED_L3_ID,
    'Protein Binding' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- DMPK / ADME > Protein Binding
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("ANALYTICAL_PLATFORM") AS "ANALYTICAL_PLATFORM",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("ORDERING_FORMAT") AS "ORDERING_FORMAT",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("COMPOUND_AMOUNT_REQUIRED") AS "COMPOUND_AMOUNT_REQUIRED",
    COUNT("COMPOUND_FORMAT") AS "COMPOUND_FORMAT",
    COUNT("DELIVERABLE_FORMAT") AS "DELIVERABLE_FORMAT",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("PERMEABILITY_MODEL") AS "PERMEABILITY_MODEL",
    COUNT("TRANSPORTERS") AS "TRANSPORTERS",
    COUNT("STUDY_DIRECTION") AS "STUDY_DIRECTION",
    COUNT("DIRECTION") AS "DIRECTION",
    COUNT("PERMEABILITY_READOUTS") AS "PERMEABILITY_READOUTS",
    COUNT("TRANSPORTER_TEST_SYSTEM") AS "TRANSPORTER_TEST_SYSTEM",
    COUNT("HEPATIC_UPTAKE_ASSAY") AS "HEPATIC_UPTAKE_ASSAY",
    COUNT("REGULATORY_GUIDANCE") AS "REGULATORY_GUIDANCE",
    COUNT("TRANSPORTER_READOUTS") AS "TRANSPORTER_READOUTS",
    COUNT("INCUBATION_TIME") AS "INCUBATION_TIME",
    COUNT("MONOLAYER_AGE") AS "MONOLAYER_AGE",
    COUNT("INTEGRITY_MARKER") AS "INTEGRITY_MARKER",
    COUNT("EFFLUX_INHIBITOR") AS "EFFLUX_INHIBITOR",
    COUNT("CLASSIFICATION_OUTPUT") AS "CLASSIFICATION_OUTPUT",
    COUNT("PAMPA_TYPE") AS "PAMPA_TYPE",
    COUNT("BUFFER_PH") AS "BUFFER_PH",
    COUNT("TEST_CONCENTRATION") AS "TEST_CONCENTRATION",
    COUNT("INHIBITION_FORMAT") AS "INHIBITION_FORMAT",
    COUNT("PROBE_SUBSTRATE") AS "PROBE_SUBSTRATE",
    COUNT("NON_HUMAN_TRANSPORTER_SPECIES") AS "NON_HUMAN_TRANSPORTER_SPECIES"
    FROM SNOWFLAKE_LEARNING_DB."DMPK_ADME".PERMEABILITY_TRANSPORTER_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('SPECIES', 'Species', FALSE),
    ('ANALYTICAL_PLATFORM', 'Analytical platform', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('ORDERING_FORMAT', 'Ordering format', FALSE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('COMPOUND_AMOUNT_REQUIRED', 'Compound amount required', FALSE),
    ('COMPOUND_FORMAT', 'Compound format', FALSE),
    ('DELIVERABLE_FORMAT', 'Deliverable format', FALSE),
    ('GLP_STATUS', 'GLP status', FALSE),
    ('PERMEABILITY_MODEL', 'Permeability model', TRUE),
    ('TRANSPORTERS', 'Transporters', FALSE),
    ('STUDY_DIRECTION', 'Study direction', FALSE),
    ('DIRECTION', 'Direction', FALSE),
    ('PERMEABILITY_READOUTS', 'Permeability readouts', FALSE),
    ('TRANSPORTER_TEST_SYSTEM', 'Transporter test system', FALSE),
    ('HEPATIC_UPTAKE_ASSAY', 'Hepatic uptake assay', FALSE),
    ('REGULATORY_GUIDANCE', 'Regulatory guidance', FALSE),
    ('TRANSPORTER_READOUTS', 'Transporter readouts', FALSE),
    ('INCUBATION_TIME', 'Incubation time', FALSE),
    ('MONOLAYER_AGE', 'Monolayer age', FALSE),
    ('INTEGRITY_MARKER', 'Integrity marker', FALSE),
    ('EFFLUX_INHIBITOR', 'Efflux inhibitor', FALSE),
    ('CLASSIFICATION_OUTPUT', 'Classification output', FALSE),
    ('PAMPA_TYPE', 'PAMPA type', FALSE),
    ('BUFFER_PH', 'Buffer pH', FALSE),
    ('TEST_CONCENTRATION', 'Test concentration', FALSE),
    ('INHIBITION_FORMAT', 'Inhibition format', FALSE),
    ('PROBE_SUBSTRATE', 'Probe substrate', FALSE),
    ('NON_HUMAN_TRANSPORTER_SPECIES', 'Non-human transporter species', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'dmpk_adme' AS ASSIGNED_L2_ID,
    'DMPK / ADME' AS ASSIGNED_L2_LABEL,
    'permeability_transporter' AS ASSIGNED_L3_ID,
    'Permeability & Transporter' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- DMPK / ADME > Permeability & Transporter
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_EQUIPMENT_INSTRUMENTS_PARTS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("TECHNIQUE") AS "TECHNIQUE",
    COUNT("FORM_FACTOR") AS "FORM_FACTOR",
    COUNT("MODEL_CONFIGURATION") AS "MODEL_CONFIGURATION",
    COUNT("PRODUCT_ROLE") AS "PRODUCT_ROLE",
    COUNT("DETECTOR") AS "DETECTOR",
    COUNT("MAX_PRESSURE") AS "MAX_PRESSURE",
    COUNT("FLOW_RATE_RANGE") AS "FLOW_RATE_RANGE",
    COUNT("SAMPLE_CAPACITY") AS "SAMPLE_CAPACITY",
    COUNT("MASS_RANGE") AS "MASS_RANGE",
    COUNT("IONIZATION_SOURCES") AS "IONIZATION_SOURCES",
    COUNT("SCAN_SPEED") AS "SCAN_SPEED",
    COUNT("COMPATIBLE_LC_FRONT_END") AS "COMPATIBLE_LC_FRONT_END",
    COUNT("ACQUISITION_MODES") AS "ACQUISITION_MODES",
    COUNT("SOFTWARE") AS "SOFTWARE",
    COUNT("COMPLIANCE_DATA_INTEGRITY") AS "COMPLIANCE_DATA_INTEGRITY",
    COUNT("CERTIFICATIONS") AS "CERTIFICATIONS",
    COUNT("DIMENSIONS_W_D_H") AS "DIMENSIONS_W_D_H",
    COUNT("WEIGHT") AS "WEIGHT",
    COUNT("POWER_REQUIREMENTS") AS "POWER_REQUIREMENTS",
    COUNT("WARRANTY_SERVICE_PLAN") AS "WARRANTY_SERVICE_PLAN",
    COUNT("CONDITION") AS "CONDITION"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".MASS_SPEC_CHROMATOGRAPHY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('TECHNIQUE', 'Technique', TRUE),
    ('FORM_FACTOR', 'Form factor', TRUE),
    ('MODEL_CONFIGURATION', 'Model / configuration', TRUE),
    ('PRODUCT_ROLE', 'Product role', TRUE),
    ('DETECTOR', 'Detector', FALSE),
    ('MAX_PRESSURE', 'Max pressure', FALSE),
    ('FLOW_RATE_RANGE', 'Flow rate range', FALSE),
    ('SAMPLE_CAPACITY', 'Sample capacity', FALSE),
    ('MASS_RANGE', 'Mass range', FALSE),
    ('IONIZATION_SOURCES', 'Ionization sources', FALSE),
    ('SCAN_SPEED', 'Scan speed', FALSE),
    ('COMPATIBLE_LC_FRONT_END', 'Compatible LC front end', FALSE),
    ('ACQUISITION_MODES', 'Acquisition modes', FALSE),
    ('SOFTWARE', 'Software', FALSE),
    ('COMPLIANCE_DATA_INTEGRITY', 'Compliance / data integrity', FALSE),
    ('CERTIFICATIONS', 'Certifications', FALSE),
    ('DIMENSIONS_W_D_H', 'Dimensions (W × D × H)', FALSE),
    ('WEIGHT', 'Weight', FALSE),
    ('POWER_REQUIREMENTS', 'Power requirements', FALSE),
    ('WARRANTY_SERVICE_PLAN', 'Warranty / service plan', FALSE),
    ('CONDITION', 'Condition', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'mass_spec_chromatography' AS ASSIGNED_L3_ID,
    'Mass Spec & Chromatography' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Mass Spec & Chromatography
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("MAX_READ_LENGTH") AS "MAX_READ_LENGTH",
    COUNT("OUTPUT_PER_FLOW_CELL") AS "OUTPUT_PER_FLOW_CELL",
    COUNT("FORM_FACTOR") AS "FORM_FACTOR",
    COUNT("MODEL_CONFIGURATION") AS "MODEL_CONFIGURATION",
    COUNT("PRODUCT_ROLE") AS "PRODUCT_ROLE",
    COUNT("FLOW_CELL_TYPES") AS "FLOW_CELL_TYPES",
    COUNT("READS_PER_RUN") AS "READS_PER_RUN",
    COUNT("RUN_TIME") AS "RUN_TIME",
    COUNT("QUALITY_SCORE") AS "QUALITY_SCORE",
    COUNT("SAMPLES_PER_FLOW_CELL") AS "SAMPLES_PER_FLOW_CELL",
    COUNT("SOFTWARE") AS "SOFTWARE",
    COUNT("COMPLIANCE_DATA_INTEGRITY") AS "COMPLIANCE_DATA_INTEGRITY",
    COUNT("CERTIFICATIONS") AS "CERTIFICATIONS",
    COUNT("DIMENSIONS_W_D_H") AS "DIMENSIONS_W_D_H",
    COUNT("WEIGHT") AS "WEIGHT",
    COUNT("POWER_REQUIREMENTS") AS "POWER_REQUIREMENTS",
    COUNT("WARRANTY_SERVICE_PLAN") AS "WARRANTY_SERVICE_PLAN",
    COUNT("CONDITION") AS "CONDITION"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".SEQUENCING_GENOMICS_PLATFORMS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('MAX_READ_LENGTH', 'Max read length', TRUE),
    ('OUTPUT_PER_FLOW_CELL', 'Output per flow cell', TRUE),
    ('FORM_FACTOR', 'Form factor', TRUE),
    ('MODEL_CONFIGURATION', 'Model / configuration', TRUE),
    ('PRODUCT_ROLE', 'Product role', TRUE),
    ('FLOW_CELL_TYPES', 'Flow cell types', FALSE),
    ('READS_PER_RUN', 'Reads per run', FALSE),
    ('RUN_TIME', 'Run time', FALSE),
    ('QUALITY_SCORE', 'Quality score', FALSE),
    ('SAMPLES_PER_FLOW_CELL', 'Samples per flow cell', FALSE),
    ('SOFTWARE', 'Software', FALSE),
    ('COMPLIANCE_DATA_INTEGRITY', 'Compliance / data integrity', FALSE),
    ('CERTIFICATIONS', 'Certifications', FALSE),
    ('DIMENSIONS_W_D_H', 'Dimensions (W × D × H)', FALSE),
    ('WEIGHT', 'Weight', FALSE),
    ('POWER_REQUIREMENTS', 'Power requirements', FALSE),
    ('WARRANTY_SERVICE_PLAN', 'Warranty / service plan', FALSE),
    ('CONDITION', 'Condition', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'sequencing_genomics_platforms' AS ASSIGNED_L3_ID,
    'Sequencing & Genomics Platforms' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Sequencing & Genomics Platforms
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("LASERS") AS "LASERS",
    COUNT("PARAMETERS_DETECTORS") AS "PARAMETERS_DETECTORS",
    COUNT("FORM_FACTOR") AS "FORM_FACTOR",
    COUNT("MODEL_CONFIGURATION") AS "MODEL_CONFIGURATION",
    COUNT("PRODUCT_ROLE") AS "PRODUCT_ROLE",
    COUNT("ACQUISITION_RATE") AS "ACQUISITION_RATE",
    COUNT("SAMPLE_FORMATS") AS "SAMPLE_FORMATS",
    COUNT("SAMPLE_LOADER_AUTOMATION") AS "SAMPLE_LOADER_AUTOMATION",
    COUNT("SMALL_PARTICLE_DETECTION") AS "SMALL_PARTICLE_DETECTION",
    COUNT("UPGRADE_PATH") AS "UPGRADE_PATH",
    COUNT("SOFTWARE") AS "SOFTWARE",
    COUNT("COMPLIANCE_DATA_INTEGRITY") AS "COMPLIANCE_DATA_INTEGRITY",
    COUNT("CERTIFICATIONS") AS "CERTIFICATIONS",
    COUNT("DIMENSIONS_W_D_H") AS "DIMENSIONS_W_D_H",
    COUNT("WEIGHT") AS "WEIGHT",
    COUNT("POWER_REQUIREMENTS") AS "POWER_REQUIREMENTS",
    COUNT("WARRANTY_SERVICE_PLAN") AS "WARRANTY_SERVICE_PLAN",
    COUNT("CONDITION") AS "CONDITION"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".FLOW_CYTOMETRY_CELL_ANALYSIS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('LASERS', 'Lasers', TRUE),
    ('PARAMETERS_DETECTORS', 'Parameters / detectors', TRUE),
    ('FORM_FACTOR', 'Form factor', TRUE),
    ('MODEL_CONFIGURATION', 'Model / configuration', TRUE),
    ('PRODUCT_ROLE', 'Product role', TRUE),
    ('ACQUISITION_RATE', 'Acquisition rate', FALSE),
    ('SAMPLE_FORMATS', 'Sample formats', FALSE),
    ('SAMPLE_LOADER_AUTOMATION', 'Sample loader / automation', FALSE),
    ('SMALL_PARTICLE_DETECTION', 'Small particle detection', FALSE),
    ('UPGRADE_PATH', 'Upgrade path', FALSE),
    ('SOFTWARE', 'Software', FALSE),
    ('COMPLIANCE_DATA_INTEGRITY', 'Compliance / data integrity', FALSE),
    ('CERTIFICATIONS', 'Certifications', FALSE),
    ('DIMENSIONS_W_D_H', 'Dimensions (W × D × H)', FALSE),
    ('WEIGHT', 'Weight', FALSE),
    ('POWER_REQUIREMENTS', 'Power requirements', FALSE),
    ('WARRANTY_SERVICE_PLAN', 'Warranty / service plan', FALSE),
    ('CONDITION', 'Condition', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'flow_cytometry_cell_analysis' AS ASSIGNED_L3_ID,
    'Flow Cytometry & Cell Analysis' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Flow Cytometry & Cell Analysis
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("IMAGING_MODES") AS "IMAGING_MODES",
    COUNT("MAGNIFICATION_OBJECTIVES") AS "MAGNIFICATION_OBJECTIVES",
    COUNT("FORM_FACTOR") AS "FORM_FACTOR",
    COUNT("MODEL_CONFIGURATION") AS "MODEL_CONFIGURATION",
    COUNT("PRODUCT_ROLE") AS "PRODUCT_ROLE",
    COUNT("FLUORESCENCE_CHANNELS") AS "FLUORESCENCE_CHANNELS",
    COUNT("LIGHT_SOURCE") AS "LIGHT_SOURCE",
    COUNT("CAMERA_SENSOR") AS "CAMERA_SENSOR",
    COUNT("AUTOMATION") AS "AUTOMATION",
    COUNT("VESSEL_FORMATS") AS "VESSEL_FORMATS",
    COUNT("ENVIRONMENTAL_CONTROL") AS "ENVIRONMENTAL_CONTROL",
    COUNT("SOFTWARE") AS "SOFTWARE",
    COUNT("COMPLIANCE_DATA_INTEGRITY") AS "COMPLIANCE_DATA_INTEGRITY",
    COUNT("CERTIFICATIONS") AS "CERTIFICATIONS",
    COUNT("DIMENSIONS_W_D_H") AS "DIMENSIONS_W_D_H",
    COUNT("WEIGHT") AS "WEIGHT",
    COUNT("POWER_REQUIREMENTS") AS "POWER_REQUIREMENTS",
    COUNT("WARRANTY_SERVICE_PLAN") AS "WARRANTY_SERVICE_PLAN",
    COUNT("CONDITION") AS "CONDITION"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".IMAGING_MICROSCOPY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('IMAGING_MODES', 'Imaging modes', TRUE),
    ('MAGNIFICATION_OBJECTIVES', 'Magnification / objectives', TRUE),
    ('FORM_FACTOR', 'Form factor', TRUE),
    ('MODEL_CONFIGURATION', 'Model / configuration', TRUE),
    ('PRODUCT_ROLE', 'Product role', TRUE),
    ('FLUORESCENCE_CHANNELS', 'Fluorescence channels', FALSE),
    ('LIGHT_SOURCE', 'Light source', FALSE),
    ('CAMERA_SENSOR', 'Camera / sensor', FALSE),
    ('AUTOMATION', 'Automation', FALSE),
    ('VESSEL_FORMATS', 'Vessel formats', FALSE),
    ('ENVIRONMENTAL_CONTROL', 'Environmental control', FALSE),
    ('SOFTWARE', 'Software', FALSE),
    ('COMPLIANCE_DATA_INTEGRITY', 'Compliance / data integrity', FALSE),
    ('CERTIFICATIONS', 'Certifications', FALSE),
    ('DIMENSIONS_W_D_H', 'Dimensions (W × D × H)', FALSE),
    ('WEIGHT', 'Weight', FALSE),
    ('POWER_REQUIREMENTS', 'Power requirements', FALSE),
    ('WARRANTY_SERVICE_PLAN', 'Warranty / service plan', FALSE),
    ('CONDITION', 'Condition', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'imaging_microscopy' AS ASSIGNED_L3_ID,
    'Imaging & Microscopy' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Imaging & Microscopy
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("DECK_WORKTABLE_SIZE") AS "DECK_WORKTABLE_SIZE",
    COUNT("PIPETTING_ARMS_CHANNELS") AS "PIPETTING_ARMS_CHANNELS",
    COUNT("VOLUME_RANGE") AS "VOLUME_RANGE",
    COUNT("FORM_FACTOR") AS "FORM_FACTOR",
    COUNT("MODEL_CONFIGURATION") AS "MODEL_CONFIGURATION",
    COUNT("PRODUCT_ROLE") AS "PRODUCT_ROLE",
    COUNT("TIP_TYPE") AS "TIP_TYPE",
    COUNT("PLATE_HANDLING") AS "PLATE_HANDLING",
    COUNT("INTEGRATION") AS "INTEGRATION",
    COUNT("SOFTWARE") AS "SOFTWARE",
    COUNT("COMPLIANCE_DATA_INTEGRITY") AS "COMPLIANCE_DATA_INTEGRITY",
    COUNT("CERTIFICATIONS") AS "CERTIFICATIONS",
    COUNT("DIMENSIONS_W_D_H") AS "DIMENSIONS_W_D_H",
    COUNT("WEIGHT") AS "WEIGHT",
    COUNT("POWER_REQUIREMENTS") AS "POWER_REQUIREMENTS",
    COUNT("WARRANTY_SERVICE_PLAN") AS "WARRANTY_SERVICE_PLAN",
    COUNT("CONDITION") AS "CONDITION"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".LIQUID_HANDLING_AUTOMATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('DECK_WORKTABLE_SIZE', 'Deck / worktable size', TRUE),
    ('PIPETTING_ARMS_CHANNELS', 'Pipetting arms / channels', TRUE),
    ('VOLUME_RANGE', 'Volume range', TRUE),
    ('FORM_FACTOR', 'Form factor', TRUE),
    ('MODEL_CONFIGURATION', 'Model / configuration', TRUE),
    ('PRODUCT_ROLE', 'Product role', TRUE),
    ('TIP_TYPE', 'Tip type', FALSE),
    ('PLATE_HANDLING', 'Plate handling', FALSE),
    ('INTEGRATION', 'Integration', FALSE),
    ('SOFTWARE', 'Software', FALSE),
    ('COMPLIANCE_DATA_INTEGRITY', 'Compliance / data integrity', FALSE),
    ('CERTIFICATIONS', 'Certifications', FALSE),
    ('DIMENSIONS_W_D_H', 'Dimensions (W × D × H)', FALSE),
    ('WEIGHT', 'Weight', FALSE),
    ('POWER_REQUIREMENTS', 'Power requirements', FALSE),
    ('WARRANTY_SERVICE_PLAN', 'Warranty / service plan', FALSE),
    ('CONDITION', 'Condition', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'liquid_handling_automation' AS ASSIGNED_L3_ID,
    'Liquid Handling & Automation' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Liquid Handling & Automation
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("SYSTEM_TYPE") AS "SYSTEM_TYPE",
    COUNT("VESSEL_SIZE") AS "VESSEL_SIZE",
    COUNT("WORKING_VOLUME") AS "WORKING_VOLUME",
    COUNT("APPLICATION") AS "APPLICATION",
    COUNT("FORM_FACTOR") AS "FORM_FACTOR",
    COUNT("MODEL_CONFIGURATION") AS "MODEL_CONFIGURATION",
    COUNT("PRODUCT_ROLE") AS "PRODUCT_ROLE",
    COUNT("BAG_CONSUMABLE_TYPE") AS "BAG_CONSUMABLE_TYPE",
    COUNT("CONTROL_SYSTEM") AS "CONTROL_SYSTEM",
    COUNT("SENSORS") AS "SENSORS",
    COUNT("GMP_READINESS") AS "GMP_READINESS",
    COUNT("SCALABILITY") AS "SCALABILITY",
    COUNT("SOFTWARE") AS "SOFTWARE",
    COUNT("COMPLIANCE_DATA_INTEGRITY") AS "COMPLIANCE_DATA_INTEGRITY",
    COUNT("CERTIFICATIONS") AS "CERTIFICATIONS",
    COUNT("DIMENSIONS_W_D_H") AS "DIMENSIONS_W_D_H",
    COUNT("WEIGHT") AS "WEIGHT",
    COUNT("POWER_REQUIREMENTS") AS "POWER_REQUIREMENTS",
    COUNT("WARRANTY_SERVICE_PLAN") AS "WARRANTY_SERVICE_PLAN",
    COUNT("CONDITION") AS "CONDITION"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".BIOPROCESS_SINGLE_USE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('SYSTEM_TYPE', 'System type', TRUE),
    ('VESSEL_SIZE', 'Vessel size', TRUE),
    ('WORKING_VOLUME', 'Working volume', TRUE),
    ('APPLICATION', 'Application', TRUE),
    ('FORM_FACTOR', 'Form factor', TRUE),
    ('MODEL_CONFIGURATION', 'Model / configuration', TRUE),
    ('PRODUCT_ROLE', 'Product role', TRUE),
    ('BAG_CONSUMABLE_TYPE', 'Bag / consumable type', FALSE),
    ('CONTROL_SYSTEM', 'Control system', FALSE),
    ('SENSORS', 'Sensors', FALSE),
    ('GMP_READINESS', 'GMP readiness', FALSE),
    ('SCALABILITY', 'Scalability', FALSE),
    ('SOFTWARE', 'Software', FALSE),
    ('COMPLIANCE_DATA_INTEGRITY', 'Compliance / data integrity', FALSE),
    ('CERTIFICATIONS', 'Certifications', FALSE),
    ('DIMENSIONS_W_D_H', 'Dimensions (W × D × H)', FALSE),
    ('WEIGHT', 'Weight', FALSE),
    ('POWER_REQUIREMENTS', 'Power requirements', FALSE),
    ('WARRANTY_SERVICE_PLAN', 'Warranty / service plan', FALSE),
    ('CONDITION', 'Condition', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'bioprocess_single_use' AS ASSIGNED_L3_ID,
    'Bioprocess & Single-Use' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Bioprocess & Single-Use
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("FORM_FACTOR") AS "FORM_FACTOR",
    COUNT("MODEL_CONFIGURATION") AS "MODEL_CONFIGURATION",
    COUNT("PRODUCT_ROLE") AS "PRODUCT_ROLE",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("TEMPERATURE_RANGE") AS "TEMPERATURE_RANGE",
    COUNT("MAX_SPEED_RCF") AS "MAX_SPEED_RCF",
    COUNT("REFRIGERATED") AS "REFRIGERATED",
    COUNT("NOISE_LEVEL") AS "NOISE_LEVEL",
    COUNT("SOFTWARE") AS "SOFTWARE",
    COUNT("COMPLIANCE_DATA_INTEGRITY") AS "COMPLIANCE_DATA_INTEGRITY",
    COUNT("CERTIFICATIONS") AS "CERTIFICATIONS",
    COUNT("DIMENSIONS_W_D_H") AS "DIMENSIONS_W_D_H",
    COUNT("WEIGHT") AS "WEIGHT",
    COUNT("POWER_REQUIREMENTS") AS "POWER_REQUIREMENTS",
    COUNT("WARRANTY_SERVICE_PLAN") AS "WARRANTY_SERVICE_PLAN",
    COUNT("CONDITION") AS "CONDITION"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".GENERAL_LAB_EQUIPMENT_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('FORM_FACTOR', 'Form factor', TRUE),
    ('MODEL_CONFIGURATION', 'Model / configuration', TRUE),
    ('PRODUCT_ROLE', 'Product role', TRUE),
    ('CAPACITY', 'Capacity', FALSE),
    ('TEMPERATURE_RANGE', 'Temperature range', FALSE),
    ('MAX_SPEED_RCF', 'Max speed / RCF', FALSE),
    ('REFRIGERATED', 'Refrigerated', FALSE),
    ('NOISE_LEVEL', 'Noise level', FALSE),
    ('SOFTWARE', 'Software', FALSE),
    ('COMPLIANCE_DATA_INTEGRITY', 'Compliance / data integrity', FALSE),
    ('CERTIFICATIONS', 'Certifications', FALSE),
    ('DIMENSIONS_W_D_H', 'Dimensions (W × D × H)', FALSE),
    ('WEIGHT', 'Weight', FALSE),
    ('POWER_REQUIREMENTS', 'Power requirements', FALSE),
    ('WARRANTY_SERVICE_PLAN', 'Warranty / service plan', FALSE),
    ('CONDITION', 'Condition', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'general_lab_equipment' AS ASSIGNED_L3_ID,
    'General Lab Equipment' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > General Lab Equipment
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("PART_NUMBER") AS "PART_NUMBER",
    COUNT("COMPATIBLE_INSTRUMENTS_MODULES") AS "COMPATIBLE_INSTRUMENTS_MODULES",
    COUNT("OEM_VS_COMPATIBLE") AS "OEM_VS_COMPATIBLE",
    COUNT("MATERIAL") AS "MATERIAL",
    COUNT("PERFORMANCE_RATING") AS "PERFORMANCE_RATING",
    COUNT("OEM_PART_CROSS_REFERENCE") AS "OEM_PART_CROSS_REFERENCE",
    COUNT("MATING_PART") AS "MATING_PART",
    COUNT("CONDITION") AS "CONDITION"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".PARTS_SPARES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('PART_NUMBER', 'Part number', TRUE),
    ('COMPATIBLE_INSTRUMENTS_MODULES', 'Compatible instruments / modules', TRUE),
    ('OEM_VS_COMPATIBLE', 'OEM vs compatible', TRUE),
    ('MATERIAL', 'Material', FALSE),
    ('PERFORMANCE_RATING', 'Performance rating', FALSE),
    ('OEM_PART_CROSS_REFERENCE', 'OEM part cross-reference', FALSE),
    ('MATING_PART', 'Mating part', FALSE),
    ('CONDITION', 'Condition', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'parts_spares' AS ASSIGNED_L3_ID,
    'Parts & Spares' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Parts & Spares
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("COMPATIBLE_PLATFORM") AS "COMPATIBLE_PLATFORM",
    COUNT("SIZE_CAPACITY") AS "SIZE_CAPACITY",
    COUNT("FILTERED") AS "FILTERED"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".INSTRUMENT_CONSUMABLES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('COMPATIBLE_PLATFORM', 'Compatible platform', TRUE),
    ('SIZE_CAPACITY', 'Size / capacity', FALSE),
    ('FILTERED', 'Filtered', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'instrument_consumables' AS ASSIGNED_L3_ID,
    'Instrument Consumables' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Instrument Consumables
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".INSTALLATION_QUALIFICATION_IQ_OQ_PQ_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'installation_qualification_iq_oq_pq' AS ASSIGNED_L3_ID,
    'Installation & Qualification (IQ/OQ/PQ)' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Installation & Qualification (IQ/OQ/PQ)
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".CALIBRATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'calibration' AS ASSIGNED_L3_ID,
    'Calibration' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Calibration
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".PREVENTIVE_MAINTENANCE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'preventive_maintenance' AS ASSIGNED_L3_ID,
    'Preventive Maintenance' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Preventive Maintenance
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".REPAIR_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'repair' AS ASSIGNED_L3_ID,
    'Repair' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Repair
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".SERVICE_CONTRACTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'service_contracts' AS ASSIGNED_L3_ID,
    'Service Contracts' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Service Contracts
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EQUIPMENT_TYPE") AS "EQUIPMENT_TYPE",
    COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("PROTOCOL_SCOPE") AS "PROTOCOL_SCOPE",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("DELIVERABLE") AS "DELIVERABLE"
    FROM SNOWFLAKE_LEARNING_DB."EQUIPMENT_INSTRUMENTS_PARTS".RELOCATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EQUIPMENT_TYPE', 'Equipment type', TRUE),
    ('BRAND_MANUFACTURER', 'Brand / manufacturer', FALSE),
    ('PROTOCOL_SCOPE', 'Protocol / scope', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('DELIVERABLE', 'Deliverable', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'equipment_instruments_parts' AS ASSIGNED_L2_ID,
    'Equipment, Instruments and Parts' AS ASSIGNED_L2_LABEL,
    'relocation' AS ASSIGNED_L3_ID,
    'Relocation' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Equipment, Instruments and Parts > Relocation
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_HEOR_RWE_MARKET_ACCESS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREAS") AS "THERAPEUTIC_AREAS",
    COUNT("GEOGRAPHIC_MARKETS_COVERED") AS "GEOGRAPHIC_MARKETS_COVERED",
    COUNT("PRODUCT_TYPE_SUPPORTED") AS "PRODUCT_TYPE_SUPPORTED",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("TARGET_DECISION_MAKER") AS "TARGET_DECISION_MAKER",
    COUNT("STRATEGIC_CONSULTING_INCLUDED") AS "STRATEGIC_CONSULTING_INCLUDED",
    COUNT("PEER_REVIEWED_PUBLICATION_SUPPORT") AS "PEER_REVIEWED_PUBLICATION_SUPPORT",
    COUNT("AI_ASSISTED_METHODS") AS "AI_ASSISTED_METHODS",
    COUNT("ECONOMIC_MODEL_TYPE") AS "ECONOMIC_MODEL_TYPE",
    COUNT("MODEL_BUILD_SCOPE") AS "MODEL_BUILD_SCOPE",
    COUNT("INTERACTIVE_MODEL_TOOL") AS "INTERACTIVE_MODEL_TOOL",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("TARGET_HTA_BODY") AS "TARGET_HTA_BODY",
    COUNT("UTILITY_MEASUREMENT") AS "UTILITY_MEASUREMENT",
    COUNT("PROPRIETARY_MODELING_PLATFORM") AS "PROPRIETARY_MODELING_PLATFORM",
    COUNT("MODEL_STRUCTURE") AS "MODEL_STRUCTURE",
    COUNT("SURVIVAL_ANALYSIS_INCLUDED") AS "SURVIVAL_ANALYSIS_INCLUDED",
    COUNT("MODEL_VALIDATION") AS "MODEL_VALIDATION"
    FROM SNOWFLAKE_LEARNING_DB."HEOR_RWE_MARKET_ACCESS".HEOR_MODELING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREAS', 'Therapeutic areas', FALSE),
    ('GEOGRAPHIC_MARKETS_COVERED', 'Geographic markets covered', FALSE),
    ('PRODUCT_TYPE_SUPPORTED', 'Product type supported', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('TARGET_DECISION_MAKER', 'Target decision maker', FALSE),
    ('STRATEGIC_CONSULTING_INCLUDED', 'Strategic consulting included', FALSE),
    ('PEER_REVIEWED_PUBLICATION_SUPPORT', 'Peer-reviewed publication support', FALSE),
    ('AI_ASSISTED_METHODS', 'AI-assisted methods', FALSE),
    ('ECONOMIC_MODEL_TYPE', 'Economic model type', TRUE),
    ('MODEL_BUILD_SCOPE', 'Model build scope', FALSE),
    ('INTERACTIVE_MODEL_TOOL', 'Interactive model tool', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('TARGET_HTA_BODY', 'Target HTA body', FALSE),
    ('UTILITY_MEASUREMENT', 'Utility measurement', FALSE),
    ('PROPRIETARY_MODELING_PLATFORM', 'Proprietary modeling platform', FALSE),
    ('MODEL_STRUCTURE', 'Model structure', FALSE),
    ('SURVIVAL_ANALYSIS_INCLUDED', 'Survival analysis included', FALSE),
    ('MODEL_VALIDATION', 'Model validation', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'heor_rwe_market_access' AS ASSIGNED_L2_ID,
    'HEOR, RWE & Market Access' AS ASSIGNED_L2_LABEL,
    'heor_modeling' AS ASSIGNED_L3_ID,
    'HEOR Modeling' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- HEOR, RWE & Market Access > HEOR Modeling
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREAS") AS "THERAPEUTIC_AREAS",
    COUNT("GEOGRAPHIC_MARKETS_COVERED") AS "GEOGRAPHIC_MARKETS_COVERED",
    COUNT("PRODUCT_TYPE_SUPPORTED") AS "PRODUCT_TYPE_SUPPORTED",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("TARGET_DECISION_MAKER") AS "TARGET_DECISION_MAKER",
    COUNT("STRATEGIC_CONSULTING_INCLUDED") AS "STRATEGIC_CONSULTING_INCLUDED",
    COUNT("PEER_REVIEWED_PUBLICATION_SUPPORT") AS "PEER_REVIEWED_PUBLICATION_SUPPORT",
    COUNT("AI_ASSISTED_METHODS") AS "AI_ASSISTED_METHODS",
    COUNT("STUDY_DESIGN") AS "STUDY_DESIGN",
    COUNT("STUDY_TIMING") AS "STUDY_TIMING",
    COUNT("DATA_SOURCE_TYPE") AS "DATA_SOURCE_TYPE",
    COUNT("INTENDED_USE_OF_EVIDENCE") AS "INTENDED_USE_OF_EVIDENCE",
    COUNT("PROPRIETARY_DATA_ASSET_USED") AS "PROPRIETARY_DATA_ASSET_USED",
    COUNT("DATA_COUNTRY_OF_ORIGIN") AS "DATA_COUNTRY_OF_ORIGIN",
    COUNT("DATA_LINKAGE") AS "DATA_LINKAGE",
    COUNT("DELIVERABLES") AS "DELIVERABLES"
    FROM SNOWFLAKE_LEARNING_DB."HEOR_RWE_MARKET_ACCESS".REAL_WORLD_EVIDENCE_STUDIES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREAS', 'Therapeutic areas', FALSE),
    ('GEOGRAPHIC_MARKETS_COVERED', 'Geographic markets covered', FALSE),
    ('PRODUCT_TYPE_SUPPORTED', 'Product type supported', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('TARGET_DECISION_MAKER', 'Target decision maker', FALSE),
    ('STRATEGIC_CONSULTING_INCLUDED', 'Strategic consulting included', FALSE),
    ('PEER_REVIEWED_PUBLICATION_SUPPORT', 'Peer-reviewed publication support', FALSE),
    ('AI_ASSISTED_METHODS', 'AI-assisted methods', FALSE),
    ('STUDY_DESIGN', 'Study design', TRUE),
    ('STUDY_TIMING', 'Study timing', FALSE),
    ('DATA_SOURCE_TYPE', 'Data source type', FALSE),
    ('INTENDED_USE_OF_EVIDENCE', 'Intended use of evidence', FALSE),
    ('PROPRIETARY_DATA_ASSET_USED', 'Proprietary data asset used', FALSE),
    ('DATA_COUNTRY_OF_ORIGIN', 'Data country of origin', FALSE),
    ('DATA_LINKAGE', 'Data linkage', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'heor_rwe_market_access' AS ASSIGNED_L2_ID,
    'HEOR, RWE & Market Access' AS ASSIGNED_L2_LABEL,
    'real_world_evidence_studies' AS ASSIGNED_L3_ID,
    'Real-World Evidence Studies' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- HEOR, RWE & Market Access > Real-World Evidence Studies
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREAS") AS "THERAPEUTIC_AREAS",
    COUNT("GEOGRAPHIC_MARKETS_COVERED") AS "GEOGRAPHIC_MARKETS_COVERED",
    COUNT("PRODUCT_TYPE_SUPPORTED") AS "PRODUCT_TYPE_SUPPORTED",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("TARGET_DECISION_MAKER") AS "TARGET_DECISION_MAKER",
    COUNT("STRATEGIC_CONSULTING_INCLUDED") AS "STRATEGIC_CONSULTING_INCLUDED",
    COUNT("PEER_REVIEWED_PUBLICATION_SUPPORT") AS "PEER_REVIEWED_PUBLICATION_SUPPORT",
    COUNT("AI_ASSISTED_METHODS") AS "AI_ASSISTED_METHODS",
    COUNT("REVIEW_TYPE") AS "REVIEW_TYPE",
    COUNT("INDIRECT_TREATMENT_COMPARISON_METHOD") AS "INDIRECT_TREATMENT_COMPARISON_METHOD",
    COUNT("EVIDENCE_TOPIC") AS "EVIDENCE_TOPIC",
    COUNT("META_ANALYSIS_INCLUDED") AS "META_ANALYSIS_INCLUDED",
    COUNT("GREY_LITERATURE_INCLUDED") AS "GREY_LITERATURE_INCLUDED",
    COUNT("HTA_COMPLIANT_METHODS") AS "HTA_COMPLIANT_METHODS",
    COUNT("SLR_PLATFORM_OR_TOOLING") AS "SLR_PLATFORM_OR_TOOLING"
    FROM SNOWFLAKE_LEARNING_DB."HEOR_RWE_MARKET_ACCESS".SYSTEMATIC_LITERATURE_REVIEW_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREAS', 'Therapeutic areas', FALSE),
    ('GEOGRAPHIC_MARKETS_COVERED', 'Geographic markets covered', FALSE),
    ('PRODUCT_TYPE_SUPPORTED', 'Product type supported', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('TARGET_DECISION_MAKER', 'Target decision maker', FALSE),
    ('STRATEGIC_CONSULTING_INCLUDED', 'Strategic consulting included', FALSE),
    ('PEER_REVIEWED_PUBLICATION_SUPPORT', 'Peer-reviewed publication support', FALSE),
    ('AI_ASSISTED_METHODS', 'AI-assisted methods', FALSE),
    ('REVIEW_TYPE', 'Review type', TRUE),
    ('INDIRECT_TREATMENT_COMPARISON_METHOD', 'Indirect treatment comparison method', FALSE),
    ('EVIDENCE_TOPIC', 'Evidence topic', FALSE),
    ('META_ANALYSIS_INCLUDED', 'Meta-analysis included', FALSE),
    ('GREY_LITERATURE_INCLUDED', 'Grey literature included', FALSE),
    ('HTA_COMPLIANT_METHODS', 'HTA-compliant methods', FALSE),
    ('SLR_PLATFORM_OR_TOOLING', 'SLR platform or tooling', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'heor_rwe_market_access' AS ASSIGNED_L2_ID,
    'HEOR, RWE & Market Access' AS ASSIGNED_L2_LABEL,
    'systematic_literature_review' AS ASSIGNED_L3_ID,
    'Systematic Literature Review' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- HEOR, RWE & Market Access > Systematic Literature Review
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREAS") AS "THERAPEUTIC_AREAS",
    COUNT("GEOGRAPHIC_MARKETS_COVERED") AS "GEOGRAPHIC_MARKETS_COVERED",
    COUNT("PRODUCT_TYPE_SUPPORTED") AS "PRODUCT_TYPE_SUPPORTED",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("TARGET_DECISION_MAKER") AS "TARGET_DECISION_MAKER",
    COUNT("STRATEGIC_CONSULTING_INCLUDED") AS "STRATEGIC_CONSULTING_INCLUDED",
    COUNT("PEER_REVIEWED_PUBLICATION_SUPPORT") AS "PEER_REVIEWED_PUBLICATION_SUPPORT",
    COUNT("AI_ASSISTED_METHODS") AS "AI_ASSISTED_METHODS",
    COUNT("DOSSIER_TYPE") AS "DOSSIER_TYPE",
    COUNT("VALUE_COMMUNICATION_DELIVERABLES") AS "VALUE_COMMUNICATION_DELIVERABLES",
    COUNT("TARGET_HTA_BODY") AS "TARGET_HTA_BODY",
    COUNT("SUBMISSION_SUPPORT_STAGE") AS "SUBMISSION_SUPPORT_STAGE",
    COUNT("INTERACTIVE_DIGITAL_DOSSIER") AS "INTERACTIVE_DIGITAL_DOSSIER",
    COUNT("MESSAGE_TESTING_WITH_PAYERS") AS "MESSAGE_TESTING_WITH_PAYERS"
    FROM SNOWFLAKE_LEARNING_DB."HEOR_RWE_MARKET_ACCESS".PAYER_VALUE_DOSSIERS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREAS', 'Therapeutic areas', FALSE),
    ('GEOGRAPHIC_MARKETS_COVERED', 'Geographic markets covered', FALSE),
    ('PRODUCT_TYPE_SUPPORTED', 'Product type supported', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('TARGET_DECISION_MAKER', 'Target decision maker', FALSE),
    ('STRATEGIC_CONSULTING_INCLUDED', 'Strategic consulting included', FALSE),
    ('PEER_REVIEWED_PUBLICATION_SUPPORT', 'Peer-reviewed publication support', FALSE),
    ('AI_ASSISTED_METHODS', 'AI-assisted methods', FALSE),
    ('DOSSIER_TYPE', 'Dossier type', TRUE),
    ('VALUE_COMMUNICATION_DELIVERABLES', 'Value communication deliverables', FALSE),
    ('TARGET_HTA_BODY', 'Target HTA body', FALSE),
    ('SUBMISSION_SUPPORT_STAGE', 'Submission support stage', FALSE),
    ('INTERACTIVE_DIGITAL_DOSSIER', 'Interactive digital dossier', FALSE),
    ('MESSAGE_TESTING_WITH_PAYERS', 'Message testing with payers', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'heor_rwe_market_access' AS ASSIGNED_L2_ID,
    'HEOR, RWE & Market Access' AS ASSIGNED_L2_LABEL,
    'payer_value_dossiers' AS ASSIGNED_L3_ID,
    'Payer / Value Dossiers' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- HEOR, RWE & Market Access > Payer / Value Dossiers
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREAS") AS "THERAPEUTIC_AREAS",
    COUNT("GEOGRAPHIC_MARKETS_COVERED") AS "GEOGRAPHIC_MARKETS_COVERED",
    COUNT("PRODUCT_TYPE_SUPPORTED") AS "PRODUCT_TYPE_SUPPORTED",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("TARGET_DECISION_MAKER") AS "TARGET_DECISION_MAKER",
    COUNT("STRATEGIC_CONSULTING_INCLUDED") AS "STRATEGIC_CONSULTING_INCLUDED",
    COUNT("PEER_REVIEWED_PUBLICATION_SUPPORT") AS "PEER_REVIEWED_PUBLICATION_SUPPORT",
    COUNT("AI_ASSISTED_METHODS") AS "AI_ASSISTED_METHODS",
    COUNT("PRICING_SERVICE") AS "PRICING_SERVICE",
    COUNT("PAYER_RESEARCH_METHOD") AS "PAYER_RESEARCH_METHOD",
    COUNT("PATIENT_SERVICES_STRATEGY") AS "PATIENT_SERVICES_STRATEGY",
    COUNT("ACCESS_SCOPE") AS "ACCESS_SCOPE",
    COUNT("POLICY_AND_HTA_CONSULTING") AS "POLICY_AND_HTA_CONSULTING",
    COUNT("LAUNCH_READINESS_SUPPORT") AS "LAUNCH_READINESS_SUPPORT",
    COUNT("PULL_THROUGH_AND_FIELD_TRAINING") AS "PULL_THROUGH_AND_FIELD_TRAINING"
    FROM SNOWFLAKE_LEARNING_DB."HEOR_RWE_MARKET_ACCESS".PRICING_MARKET_ACCESS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREAS', 'Therapeutic areas', FALSE),
    ('GEOGRAPHIC_MARKETS_COVERED', 'Geographic markets covered', FALSE),
    ('PRODUCT_TYPE_SUPPORTED', 'Product type supported', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('TARGET_DECISION_MAKER', 'Target decision maker', FALSE),
    ('STRATEGIC_CONSULTING_INCLUDED', 'Strategic consulting included', FALSE),
    ('PEER_REVIEWED_PUBLICATION_SUPPORT', 'Peer-reviewed publication support', FALSE),
    ('AI_ASSISTED_METHODS', 'AI-assisted methods', FALSE),
    ('PRICING_SERVICE', 'Pricing service', TRUE),
    ('PAYER_RESEARCH_METHOD', 'Payer research method', FALSE),
    ('PATIENT_SERVICES_STRATEGY', 'Patient services strategy', FALSE),
    ('ACCESS_SCOPE', 'Access scope', FALSE),
    ('POLICY_AND_HTA_CONSULTING', 'Policy and HTA consulting', FALSE),
    ('LAUNCH_READINESS_SUPPORT', 'Launch readiness support', FALSE),
    ('PULL_THROUGH_AND_FIELD_TRAINING', 'Pull-through and field training', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'heor_rwe_market_access' AS ASSIGNED_L2_ID,
    'HEOR, RWE & Market Access' AS ASSIGNED_L2_LABEL,
    'pricing_market_access' AS ASSIGNED_L3_ID,
    'Pricing & Market Access' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- HEOR, RWE & Market Access > Pricing & Market Access
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_HISTOLOGY_PATHOLOGY_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("SAMPLE_TYPE") AS "SAMPLE_TYPE",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("PRICING_UNIT") AS "PRICING_UNIT",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("TARGET_MARKER") AS "TARGET_MARKER",
    COUNT("DETECTION") AS "DETECTION",
    COUNT("MULTIPLEX") AS "MULTIPLEX",
    COUNT("ANTIBODY_SOURCE") AS "ANTIBODY_SOURCE",
    COUNT("QUANTIFICATION") AS "QUANTIFICATION",
    COUNT("TISSUE_CROSS_REACTIVITY_TCR") AS "TISSUE_CROSS_REACTIVITY_TCR",
    COUNT("PATHOLOGIST_CREDENTIALS") AS "PATHOLOGIST_CREDENTIALS"
    FROM SNOWFLAKE_LEARNING_DB."HISTOLOGY_PATHOLOGY".IHC_IMMUNOHISTOCHEMISTRY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('SAMPLE_TYPE', 'Sample type', TRUE),
    ('GLP_STATUS', 'GLP status', TRUE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('PRICING_UNIT', 'Pricing unit', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('TARGET_MARKER', 'Target / marker', FALSE),
    ('DETECTION', 'Detection', TRUE),
    ('MULTIPLEX', 'Multiplex', FALSE),
    ('ANTIBODY_SOURCE', 'Antibody source', FALSE),
    ('QUANTIFICATION', 'Quantification', FALSE),
    ('TISSUE_CROSS_REACTIVITY_TCR', 'Tissue cross-reactivity (TCR)', FALSE),
    ('PATHOLOGIST_CREDENTIALS', 'Pathologist credentials', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'histology_pathology' AS ASSIGNED_L2_ID,
    'Histology & Pathology' AS ASSIGNED_L2_LABEL,
    'ihc_immunohistochemistry' AS ASSIGNED_L3_ID,
    'IHC (Immunohistochemistry)' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Histology & Pathology > IHC (Immunohistochemistry)
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("SAMPLE_TYPE") AS "SAMPLE_TYPE",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("PRICING_UNIT") AS "PRICING_UNIT",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("STAIN") AS "STAIN"
    FROM SNOWFLAKE_LEARNING_DB."HISTOLOGY_PATHOLOGY".SPECIAL_HISTOCHEMICAL_STAINS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('SAMPLE_TYPE', 'Sample type', TRUE),
    ('GLP_STATUS', 'GLP status', TRUE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('PRICING_UNIT', 'Pricing unit', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('STAIN', 'Stain', TRUE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'histology_pathology' AS ASSIGNED_L2_ID,
    'Histology & Pathology' AS ASSIGNED_L2_LABEL,
    'special_histochemical_stains' AS ASSIGNED_L3_ID,
    'Special / Histochemical Stains' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Histology & Pathology > Special / Histochemical Stains
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("SAMPLE_TYPE") AS "SAMPLE_TYPE",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("PRICING_UNIT") AS "PRICING_UNIT",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("TARGET_GENE") AS "TARGET_GENE",
    COUNT("PLATFORM_CHEMISTRY") AS "PLATFORM_CHEMISTRY",
    COUNT("PROBE_SOURCE") AS "PROBE_SOURCE",
    COUNT("DETECTION") AS "DETECTION",
    COUNT("COMBINED_WITH_IF") AS "COMBINED_WITH_IF",
    COUNT("PATHOLOGIST_CREDENTIALS") AS "PATHOLOGIST_CREDENTIALS"
    FROM SNOWFLAKE_LEARNING_DB."HISTOLOGY_PATHOLOGY".IN_SITU_HYBRIDIZATION_ISH_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('SAMPLE_TYPE', 'Sample type', TRUE),
    ('GLP_STATUS', 'GLP status', TRUE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('PRICING_UNIT', 'Pricing unit', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('TARGET_GENE', 'Target gene', FALSE),
    ('PLATFORM_CHEMISTRY', 'Platform / chemistry', TRUE),
    ('PROBE_SOURCE', 'Probe source', FALSE),
    ('DETECTION', 'Detection', FALSE),
    ('COMBINED_WITH_IF', 'Combined with IF', FALSE),
    ('PATHOLOGIST_CREDENTIALS', 'Pathologist credentials', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'histology_pathology' AS ASSIGNED_L2_ID,
    'Histology & Pathology' AS ASSIGNED_L2_LABEL,
    'in_situ_hybridization_ish' AS ASSIGNED_L3_ID,
    'In Situ Hybridization (ISH)' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Histology & Pathology > In Situ Hybridization (ISH)
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("SAMPLE_TYPE") AS "SAMPLE_TYPE",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("PRICING_UNIT") AS "PRICING_UNIT",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("IMAGING_MODE") AS "IMAGING_MODE",
    COUNT("SCAN_MAGNIFICATION") AS "SCAN_MAGNIFICATION",
    COUNT("IMAGE_ANALYSIS_PLATFORM") AS "IMAGE_ANALYSIS_PLATFORM",
    COUNT("SCANNER") AS "SCANNER",
    COUNT("IMAGE_FORMAT") AS "IMAGE_FORMAT",
    COUNT("21_CFR_PART_11_VALIDATED") AS "21_CFR_PART_11_VALIDATED",
    COUNT("PATHOLOGIST_CREDENTIALS") AS "PATHOLOGIST_CREDENTIALS"
    FROM SNOWFLAKE_LEARNING_DB."HISTOLOGY_PATHOLOGY".DIGITAL_PATHOLOGY_IMAGE_ANALYSIS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('SAMPLE_TYPE', 'Sample type', TRUE),
    ('GLP_STATUS', 'GLP status', TRUE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('PRICING_UNIT', 'Pricing unit', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('IMAGING_MODE', 'Imaging mode', TRUE),
    ('SCAN_MAGNIFICATION', 'Scan magnification', FALSE),
    ('IMAGE_ANALYSIS_PLATFORM', 'Image analysis platform', FALSE),
    ('SCANNER', 'Scanner', FALSE),
    ('IMAGE_FORMAT', 'Image format', FALSE),
    ('21_CFR_PART_11_VALIDATED', '21 CFR Part 11 validated', FALSE),
    ('PATHOLOGIST_CREDENTIALS', 'Pathologist credentials', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'histology_pathology' AS ASSIGNED_L2_ID,
    'Histology & Pathology' AS ASSIGNED_L2_LABEL,
    'digital_pathology_image_analysis' AS ASSIGNED_L3_ID,
    'Digital Pathology & Image Analysis' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Histology & Pathology > Digital Pathology & Image Analysis
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("SAMPLE_TYPE") AS "SAMPLE_TYPE",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("PRICING_UNIT") AS "PRICING_UNIT",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("TECHNIQUE") AS "TECHNIQUE",
    COUNT("FIXATION_REQUIREMENT") AS "FIXATION_REQUIREMENT",
    COUNT("SECTION_THICKNESS") AS "SECTION_THICKNESS",
    COUNT("IMAGE_DELIVERABLES") AS "IMAGE_DELIVERABLES"
    FROM SNOWFLAKE_LEARNING_DB."HISTOLOGY_PATHOLOGY".ELECTRON_MICROSCOPY_TEM_SEM_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('SAMPLE_TYPE', 'Sample type', TRUE),
    ('GLP_STATUS', 'GLP status', TRUE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('PRICING_UNIT', 'Pricing unit', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('TECHNIQUE', 'Technique', FALSE),
    ('FIXATION_REQUIREMENT', 'Fixation requirement', FALSE),
    ('SECTION_THICKNESS', 'Section thickness', FALSE),
    ('IMAGE_DELIVERABLES', 'Image deliverables', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'histology_pathology' AS ASSIGNED_L2_ID,
    'Histology & Pathology' AS ASSIGNED_L2_LABEL,
    'electron_microscopy_tem_sem' AS ASSIGNED_L3_ID,
    'Electron Microscopy (TEM/SEM)' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Histology & Pathology > Electron Microscopy (TEM/SEM)
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("SAMPLE_TYPE") AS "SAMPLE_TYPE",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("PRICING_UNIT") AS "PRICING_UNIT",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("EMBEDDING_MEDIUM") AS "EMBEDDING_MEDIUM",
    COUNT("SECTIONS_PER_SLIDE") AS "SECTIONS_PER_SLIDE",
    COUNT("DECALCIFICATION") AS "DECALCIFICATION",
    COUNT("TISSUE_SIZE") AS "TISSUE_SIZE"
    FROM SNOWFLAKE_LEARNING_DB."HISTOLOGY_PATHOLOGY".TISSUE_PROCESSING_SECTIONING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('SAMPLE_TYPE', 'Sample type', TRUE),
    ('GLP_STATUS', 'GLP status', TRUE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('PRICING_UNIT', 'Pricing unit', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('EMBEDDING_MEDIUM', 'Embedding medium', TRUE),
    ('SECTIONS_PER_SLIDE', 'Sections per slide', FALSE),
    ('DECALCIFICATION', 'Decalcification', FALSE),
    ('TISSUE_SIZE', 'Tissue size', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'histology_pathology' AS ASSIGNED_L2_ID,
    'Histology & Pathology' AS ASSIGNED_L2_LABEL,
    'tissue_processing_sectioning' AS ASSIGNED_L3_ID,
    'Tissue Processing & Sectioning' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Histology & Pathology > Tissue Processing & Sectioning
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("SAMPLE_TYPE") AS "SAMPLE_TYPE",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("PRICING_UNIT") AS "PRICING_UNIT",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("STUDY_TYPE") AS "STUDY_TYPE",
    COUNT("SEND_READY") AS "SEND_READY",
    COUNT("DATA_SYSTEM") AS "DATA_SYSTEM",
    COUNT("PROSECTOR_SUPERVISION") AS "PROSECTOR_SUPERVISION",
    COUNT("PATHOLOGIST_CREDENTIALS") AS "PATHOLOGIST_CREDENTIALS",
    COUNT("REGULATORY_SUBMISSION_SUPPORT") AS "REGULATORY_SUBMISSION_SUPPORT"
    FROM SNOWFLAKE_LEARNING_DB."HISTOLOGY_PATHOLOGY".NECROPSY_TOXICOLOGIC_PATHOLOGY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('SAMPLE_TYPE', 'Sample type', TRUE),
    ('GLP_STATUS', 'GLP status', TRUE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('PRICING_UNIT', 'Pricing unit', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('STUDY_TYPE', 'Study type', TRUE),
    ('SEND_READY', 'SEND-ready', FALSE),
    ('DATA_SYSTEM', 'Data system', FALSE),
    ('PROSECTOR_SUPERVISION', 'Prosector supervision', FALSE),
    ('PATHOLOGIST_CREDENTIALS', 'Pathologist credentials', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORT', 'Regulatory submission support', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'histology_pathology' AS ASSIGNED_L2_ID,
    'Histology & Pathology' AS ASSIGNED_L2_LABEL,
    'necropsy_toxicologic_pathology' AS ASSIGNED_L3_ID,
    'Necropsy & Toxicologic Pathology' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Histology & Pathology > Necropsy & Toxicologic Pathology
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIES") AS "SPECIES",
    COUNT("SAMPLE_TYPE") AS "SAMPLE_TYPE",
    COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("PRICING_UNIT") AS "PRICING_UNIT",
    COUNT("CAPACITY") AS "CAPACITY",
    COUNT("REVIEW_FORMAT") AS "REVIEW_FORMAT",
    COUNT("REVIEW_SCOPE") AS "REVIEW_SCOPE",
    COUNT("TIMING") AS "TIMING",
    COUNT("PATHOLOGY_WORKING_GROUP_PWG") AS "PATHOLOGY_WORKING_GROUP_PWG",
    COUNT("PATHOLOGIST_CREDENTIALS") AS "PATHOLOGIST_CREDENTIALS",
    COUNT("REGULATORY_SUBMISSION_SUPPORT") AS "REGULATORY_SUBMISSION_SUPPORT"
    FROM SNOWFLAKE_LEARNING_DB."HISTOLOGY_PATHOLOGY".PATHOLOGY_PEER_REVIEW_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIES', 'Species', TRUE),
    ('SAMPLE_TYPE', 'Sample type', TRUE),
    ('GLP_STATUS', 'GLP status', TRUE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE),
    ('PRICING_UNIT', 'Pricing unit', FALSE),
    ('CAPACITY', 'Capacity', FALSE),
    ('REVIEW_FORMAT', 'Review format', FALSE),
    ('REVIEW_SCOPE', 'Review scope', FALSE),
    ('TIMING', 'Timing', FALSE),
    ('PATHOLOGY_WORKING_GROUP_PWG', 'Pathology Working Group (PWG)', FALSE),
    ('PATHOLOGIST_CREDENTIALS', 'Pathologist credentials', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORT', 'Regulatory submission support', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'histology_pathology' AS ASSIGNED_L2_ID,
    'Histology & Pathology' AS ASSIGNED_L2_LABEL,
    'pathology_peer_review' AS ASSIGNED_L3_ID,
    'Pathology Peer Review' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Histology & Pathology > Pathology Peer Review
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_HUMAN_BIOSPECIMENS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DIAGNOSIS") AS "DIAGNOSIS",
    COUNT("CONSENT_LEVEL") AS "CONSENT_LEVEL",
    COUNT("ANATOMICAL_SITE_TISSUE_TYPE") AS "ANATOMICAL_SITE_TISSUE_TYPE",
    COUNT("SPECIMEN_FORMAT") AS "SPECIMEN_FORMAT",
    COUNT("DISEASE_AREAS") AS "DISEASE_AREAS",
    COUNT("DISEASE_STAGE") AS "DISEASE_STAGE",
    COUNT("RIN_RNA_INTEGRITY") AS "RIN_RNA_INTEGRITY",
    COUNT("TUMOR_CONTENT") AS "TUMOR_CONTENT",
    COUNT("NECROSIS") AS "NECROSIS",
    COUNT("SAMPLE_SIZE_AMOUNT") AS "SAMPLE_SIZE_AMOUNT",
    COUNT("DONOR_AGE") AS "DONOR_AGE",
    COUNT("SEX") AS "SEX",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("RACE_ETHNICITY") AS "RACE_ETHNICITY",
    COUNT("TREATMENT_HISTORY") AS "TREATMENT_HISTORY",
    COUNT("INDIVIDUAL_VS_POOLED") AS "INDIVIDUAL_VS_POOLED",
    COUNT("MATCHED_SAMPLE_SET") AS "MATCHED_SAMPLE_SET",
    COUNT("TIMEPOINT") AS "TIMEPOINT"
    FROM SNOWFLAKE_LEARNING_DB."HUMAN_BIOSPECIMENS".HUMAN_TISSUE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DIAGNOSIS', 'Diagnosis', TRUE),
    ('CONSENT_LEVEL', 'Consent level', TRUE),
    ('ANATOMICAL_SITE_TISSUE_TYPE', 'Anatomical site / tissue type', TRUE),
    ('SPECIMEN_FORMAT', 'Specimen format', TRUE),
    ('DISEASE_AREAS', 'Disease areas', FALSE),
    ('DISEASE_STAGE', 'Disease stage', FALSE),
    ('RIN_RNA_INTEGRITY', 'RIN (RNA integrity)', FALSE),
    ('TUMOR_CONTENT', 'Tumor content %', FALSE),
    ('NECROSIS', 'Necrosis %', FALSE),
    ('SAMPLE_SIZE_AMOUNT', 'Sample size / amount', FALSE),
    ('DONOR_AGE', 'Donor Age', FALSE),
    ('SEX', 'Sex', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('RACE_ETHNICITY', 'Race / Ethnicity', FALSE),
    ('TREATMENT_HISTORY', 'Treatment history', FALSE),
    ('INDIVIDUAL_VS_POOLED', 'Individual vs Pooled', FALSE),
    ('MATCHED_SAMPLE_SET', 'Matched sample set', FALSE),
    ('TIMEPOINT', 'Timepoint', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'human_biospecimens' AS ASSIGNED_L2_ID,
    'Human Biospecimens' AS ASSIGNED_L2_LABEL,
    'human_tissue' AS ASSIGNED_L3_ID,
    'Human Tissue' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Human Biospecimens > Human Tissue
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DIAGNOSIS") AS "DIAGNOSIS",
    COUNT("CONSENT_LEVEL") AS "CONSENT_LEVEL",
    COUNT("MATRIX") AS "MATRIX",
    COUNT("ANTICOAGULANT_ADDITIVE") AS "ANTICOAGULANT_ADDITIVE",
    COUNT("VOLUME") AS "VOLUME",
    COUNT("DISEASE_AREAS") AS "DISEASE_AREAS",
    COUNT("DISEASE_STAGE") AS "DISEASE_STAGE",
    COUNT("PROCESSING") AS "PROCESSING",
    COUNT("BLOOD_TYPE") AS "BLOOD_TYPE",
    COUNT("CLINICAL_DATA") AS "CLINICAL_DATA",
    COUNT("MEDICATIONS") AS "MEDICATIONS",
    COUNT("SPECIMEN_FORMAT") AS "SPECIMEN_FORMAT",
    COUNT("DONOR_AGE") AS "DONOR_AGE",
    COUNT("SEX") AS "SEX",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("RACE_ETHNICITY") AS "RACE_ETHNICITY",
    COUNT("TREATMENT_HISTORY") AS "TREATMENT_HISTORY",
    COUNT("MATCHED_SAMPLE_SET") AS "MATCHED_SAMPLE_SET",
    COUNT("TIMEPOINT") AS "TIMEPOINT"
    FROM SNOWFLAKE_LEARNING_DB."HUMAN_BIOSPECIMENS".HUMAN_BLOOD_BIOFLUIDS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DIAGNOSIS', 'Diagnosis', TRUE),
    ('CONSENT_LEVEL', 'Consent level', TRUE),
    ('MATRIX', 'Matrix', TRUE),
    ('ANTICOAGULANT_ADDITIVE', 'Anticoagulant / additive', TRUE),
    ('VOLUME', 'Volume', TRUE),
    ('DISEASE_AREAS', 'Disease areas', FALSE),
    ('DISEASE_STAGE', 'Disease stage', FALSE),
    ('PROCESSING', 'Processing', FALSE),
    ('BLOOD_TYPE', 'Blood type', FALSE),
    ('CLINICAL_DATA', 'Clinical data', FALSE),
    ('MEDICATIONS', 'Medications', FALSE),
    ('SPECIMEN_FORMAT', 'Specimen format', FALSE),
    ('DONOR_AGE', 'Donor Age', FALSE),
    ('SEX', 'Sex', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('RACE_ETHNICITY', 'Race / Ethnicity', FALSE),
    ('TREATMENT_HISTORY', 'Treatment history', FALSE),
    ('MATCHED_SAMPLE_SET', 'Matched sample set', FALSE),
    ('TIMEPOINT', 'Timepoint', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'human_biospecimens' AS ASSIGNED_L2_ID,
    'Human Biospecimens' AS ASSIGNED_L2_LABEL,
    'human_blood_biofluids' AS ASSIGNED_L3_ID,
    'Human Blood & Biofluids' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Human Biospecimens > Human Blood & Biofluids
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DIAGNOSIS") AS "DIAGNOSIS",
    COUNT("CONSENT_LEVEL") AS "CONSENT_LEVEL",
    COUNT("CELL_COUNT") AS "CELL_COUNT",
    COUNT("SPECIMEN_FORMAT") AS "SPECIMEN_FORMAT",
    COUNT("DISEASE_AREAS") AS "DISEASE_AREAS",
    COUNT("DISEASE_STAGE") AS "DISEASE_STAGE",
    COUNT("VIABILITY") AS "VIABILITY",
    COUNT("ISOLATION_METHOD") AS "ISOLATION_METHOD",
    COUNT("HLA_TYPE") AS "HLA_TYPE",
    COUNT("CLINICAL_DATA") AS "CLINICAL_DATA",
    COUNT("DONOR_AGE") AS "DONOR_AGE",
    COUNT("SEX") AS "SEX",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("RACE_ETHNICITY") AS "RACE_ETHNICITY",
    COUNT("TREATMENT_HISTORY") AS "TREATMENT_HISTORY",
    COUNT("INDIVIDUAL_VS_POOLED") AS "INDIVIDUAL_VS_POOLED",
    COUNT("MATCHED_SAMPLE_SET") AS "MATCHED_SAMPLE_SET",
    COUNT("TIMEPOINT") AS "TIMEPOINT"
    FROM SNOWFLAKE_LEARNING_DB."HUMAN_BIOSPECIMENS".HUMAN_CELLS_PBMCS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DIAGNOSIS', 'Diagnosis', TRUE),
    ('CONSENT_LEVEL', 'Consent level', TRUE),
    ('CELL_COUNT', 'Cell count', TRUE),
    ('SPECIMEN_FORMAT', 'Specimen format', TRUE),
    ('DISEASE_AREAS', 'Disease areas', FALSE),
    ('DISEASE_STAGE', 'Disease stage', FALSE),
    ('VIABILITY', 'Viability %', FALSE),
    ('ISOLATION_METHOD', 'Isolation method', FALSE),
    ('HLA_TYPE', 'HLA type', FALSE),
    ('CLINICAL_DATA', 'Clinical data', FALSE),
    ('DONOR_AGE', 'Donor Age', FALSE),
    ('SEX', 'Sex', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('RACE_ETHNICITY', 'Race / Ethnicity', FALSE),
    ('TREATMENT_HISTORY', 'Treatment history', FALSE),
    ('INDIVIDUAL_VS_POOLED', 'Individual vs Pooled', FALSE),
    ('MATCHED_SAMPLE_SET', 'Matched sample set', FALSE),
    ('TIMEPOINT', 'Timepoint', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'human_biospecimens' AS ASSIGNED_L2_ID,
    'Human Biospecimens' AS ASSIGNED_L2_LABEL,
    'human_cells_pbmcs' AS ASSIGNED_L3_ID,
    'Human Cells & PBMCs' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Human Biospecimens > Human Cells & PBMCs
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DIAGNOSIS") AS "DIAGNOSIS",
    COUNT("SPECIMEN_FORMAT") AS "SPECIMEN_FORMAT",
    COUNT("DISEASE_AREAS") AS "DISEASE_AREAS",
    COUNT("VIABILITY") AS "VIABILITY",
    COUNT("PASSAGE_NUMBER") AS "PASSAGE_NUMBER",
    COUNT("DONOR_AGE") AS "DONOR_AGE",
    COUNT("SEX") AS "SEX",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("RACE_ETHNICITY") AS "RACE_ETHNICITY",
    COUNT("TREATMENT_HISTORY") AS "TREATMENT_HISTORY",
    COUNT("INDIVIDUAL_VS_POOLED") AS "INDIVIDUAL_VS_POOLED",
    COUNT("MATCHED_SAMPLE_SET") AS "MATCHED_SAMPLE_SET"
    FROM SNOWFLAKE_LEARNING_DB."HUMAN_BIOSPECIMENS".HUMAN_PRIMARY_CELLS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DIAGNOSIS', 'Diagnosis', TRUE),
    ('SPECIMEN_FORMAT', 'Specimen format', TRUE),
    ('DISEASE_AREAS', 'Disease areas', FALSE),
    ('VIABILITY', 'Viability %', FALSE),
    ('PASSAGE_NUMBER', 'Passage number', FALSE),
    ('DONOR_AGE', 'Donor Age', FALSE),
    ('SEX', 'Sex', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('RACE_ETHNICITY', 'Race / Ethnicity', FALSE),
    ('TREATMENT_HISTORY', 'Treatment history', FALSE),
    ('INDIVIDUAL_VS_POOLED', 'Individual vs Pooled', FALSE),
    ('MATCHED_SAMPLE_SET', 'Matched sample set', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'human_biospecimens' AS ASSIGNED_L2_ID,
    'Human Biospecimens' AS ASSIGNED_L2_LABEL,
    'human_primary_cells' AS ASSIGNED_L3_ID,
    'Human Primary Cells' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Human Biospecimens > Human Primary Cells
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DIAGNOSIS") AS "DIAGNOSIS",
    COUNT("DISEASE_AREAS") AS "DISEASE_AREAS",
    COUNT("VIABILITY") AS "VIABILITY",
    COUNT("KARYOTYPE") AS "KARYOTYPE",
    COUNT("CHARACTERIZATION_MARKERS") AS "CHARACTERIZATION_MARKERS",
    COUNT("REPROGRAMMING_METHOD") AS "REPROGRAMMING_METHOD",
    COUNT("SPECIMEN_FORMAT") AS "SPECIMEN_FORMAT",
    COUNT("DONOR_AGE") AS "DONOR_AGE",
    COUNT("SEX") AS "SEX",
    COUNT("STORAGE_CONDITIONS") AS "STORAGE_CONDITIONS",
    COUNT("RACE_ETHNICITY") AS "RACE_ETHNICITY"
    FROM SNOWFLAKE_LEARNING_DB."HUMAN_BIOSPECIMENS".HUMAN_STEM_CELLS_IPSCS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DIAGNOSIS', 'Diagnosis', TRUE),
    ('DISEASE_AREAS', 'Disease areas', FALSE),
    ('VIABILITY', 'Viability %', FALSE),
    ('KARYOTYPE', 'Karyotype', FALSE),
    ('CHARACTERIZATION_MARKERS', 'Characterization markers', FALSE),
    ('REPROGRAMMING_METHOD', 'Reprogramming method', FALSE),
    ('SPECIMEN_FORMAT', 'Specimen format', FALSE),
    ('DONOR_AGE', 'Donor Age', FALSE),
    ('SEX', 'Sex', FALSE),
    ('STORAGE_CONDITIONS', 'Storage conditions', FALSE),
    ('RACE_ETHNICITY', 'Race / Ethnicity', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'human_biospecimens' AS ASSIGNED_L2_ID,
    'Human Biospecimens' AS ASSIGNED_L2_LABEL,
    'human_stem_cells_ipscs' AS ASSIGNED_L3_ID,
    'Human Stem Cells / iPSCs' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Human Biospecimens > Human Stem Cells / iPSCs
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SAMPLE_TYPES_COLLECTED") AS "SAMPLE_TYPES_COLLECTED",
    COUNT("DONOR_HEALTH_STATUS") AS "DONOR_HEALTH_STATUS",
    COUNT("COLLECTION_GEOGRAPHY") AS "COLLECTION_GEOGRAPHY",
    COUNT("ACCEPTS_CLIENT_PROTOCOL") AS "ACCEPTS_CLIENT_PROTOCOL",
    COUNT("IRB_APPROVAL_HANDLING") AS "IRB_APPROVAL_HANDLING",
    COUNT("PRIVACY_COMPLIANCE") AS "PRIVACY_COMPLIANCE",
    COUNT("LONGITUDINAL_TIMEPOINTS_AVAILABLE") AS "LONGITUDINAL_TIMEPOINTS_AVAILABLE",
    COUNT("MATCHED_SAMPLE_SETS") AS "MATCHED_SAMPLE_SETS",
    COUNT("CLINICAL_DATA_PROVIDED") AS "CLINICAL_DATA_PROVIDED",
    COUNT("COLLECTION_KITS_PROVIDED") AS "COLLECTION_KITS_PROVIDED",
    COUNT("PRESERVATION_AT_COLLECTION") AS "PRESERVATION_AT_COLLECTION",
    COUNT("DISEASE_AREAS") AS "DISEASE_AREAS",
    COUNT("COLLECTION_SITE_TYPES") AS "COLLECTION_SITE_TYPES",
    COUNT("DONOR_CHARACTERIZATION_TESTS") AS "DONOR_CHARACTERIZATION_TESTS",
    COUNT("FRESH_DELIVERY_TIME_AFTER_COLLECTION") AS "FRESH_DELIVERY_TIME_AFTER_COLLECTION",
    COUNT("SHIPPING_TEMPERATURE") AS "SHIPPING_TEMPERATURE",
    COUNT("ACCREDITATIONS") AS "ACCREDITATIONS",
    COUNT("PATIENT_OR_SITE_RECRUITMENT") AS "PATIENT_OR_SITE_RECRUITMENT",
    COUNT("PROJECT_MANAGEMENT_INCLUDED") AS "PROJECT_MANAGEMENT_INCLUDED",
    COUNT("TIME_TO_START_COLLECTION") AS "TIME_TO_START_COLLECTION",
    COUNT("COLLECTION_TO_PROCESSING_TIME") AS "COLLECTION_TO_PROCESSING_TIME",
    COUNT("RECALLABLE_DONORS") AS "RECALLABLE_DONORS",
    COUNT("CUSTOM_CONSENT_FORM") AS "CUSTOM_CONSENT_FORM",
    COUNT("CLINICAL_DATA_DELIVERY_METHOD") AS "CLINICAL_DATA_DELIVERY_METHOD",
    COUNT("FEASIBILITY_ASSESSMENT") AS "FEASIBILITY_ASSESSMENT"
    FROM SNOWFLAKE_LEARNING_DB."HUMAN_BIOSPECIMENS".HUMAN_CUSTOM_PROSPECTIVE_COLLECTION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SAMPLE_TYPES_COLLECTED', 'Sample types collected', TRUE),
    ('DONOR_HEALTH_STATUS', 'Donor health status', TRUE),
    ('COLLECTION_GEOGRAPHY', 'Collection geography', TRUE),
    ('ACCEPTS_CLIENT_PROTOCOL', 'Accepts client protocol', TRUE),
    ('IRB_APPROVAL_HANDLING', 'IRB approval handling', TRUE),
    ('PRIVACY_COMPLIANCE', 'Privacy compliance', TRUE),
    ('LONGITUDINAL_TIMEPOINTS_AVAILABLE', 'Longitudinal timepoints available', TRUE),
    ('MATCHED_SAMPLE_SETS', 'Matched sample sets', TRUE),
    ('CLINICAL_DATA_PROVIDED', 'Clinical data provided', TRUE),
    ('COLLECTION_KITS_PROVIDED', 'Collection kits provided', TRUE),
    ('PRESERVATION_AT_COLLECTION', 'Preservation at collection', TRUE),
    ('DISEASE_AREAS', 'Disease areas', FALSE),
    ('COLLECTION_SITE_TYPES', 'Collection site types', FALSE),
    ('DONOR_CHARACTERIZATION_TESTS', 'Donor characterization tests', FALSE),
    ('FRESH_DELIVERY_TIME_AFTER_COLLECTION', 'Fresh delivery time after collection', FALSE),
    ('SHIPPING_TEMPERATURE', 'Shipping temperature', FALSE),
    ('ACCREDITATIONS', 'Accreditations', FALSE),
    ('PATIENT_OR_SITE_RECRUITMENT', 'Patient or site recruitment', FALSE),
    ('PROJECT_MANAGEMENT_INCLUDED', 'Project management included', FALSE),
    ('TIME_TO_START_COLLECTION', 'Time to start collection', FALSE),
    ('COLLECTION_TO_PROCESSING_TIME', 'Collection-to-processing time', FALSE),
    ('RECALLABLE_DONORS', 'Recallable donors', FALSE),
    ('CUSTOM_CONSENT_FORM', 'Custom consent form', FALSE),
    ('CLINICAL_DATA_DELIVERY_METHOD', 'Clinical data delivery method', FALSE),
    ('FEASIBILITY_ASSESSMENT', 'Feasibility assessment', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'human_biospecimens' AS ASSIGNED_L2_ID,
    'Human Biospecimens' AS ASSIGNED_L2_LABEL,
    'human_custom_prospective_collection' AS ASSIGNED_L3_ID,
    'Human Custom / prospective collection' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Human Biospecimens > Human Custom / prospective collection
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("STORAGE_TEMPERATURES") AS "STORAGE_TEMPERATURES",
    COUNT("SAMPLE_TYPES_STORED") AS "SAMPLE_TYPES_STORED",
    COUNT("FACILITY_LOCATIONS") AS "FACILITY_LOCATIONS",
    COUNT("BACKUP_SYSTEMS") AS "BACKUP_SYSTEMS",
    COUNT("LIMS_OR_CLIENT_PORTAL") AS "LIMS_OR_CLIENT_PORTAL",
    COUNT("INVENTORY_REPORTING") AS "INVENTORY_REPORTING",
    COUNT("CHAIN_OF_CUSTODY") AS "CHAIN_OF_CUSTODY",
    COUNT("ACCREDITATIONS") AS "ACCREDITATIONS",
    COUNT("GXP_COMPLIANCE") AS "GXP_COMPLIANCE",
    COUNT("21_CFR_PART_11_COMPLIANT") AS "21_CFR_PART_11_COMPLIANT",
    COUNT("TEMPERATURE_MONITORING") AS "TEMPERATURE_MONITORING",
    COUNT("RETRIEVAL_TURNAROUND") AS "RETRIEVAL_TURNAROUND",
    COUNT("ACCESSIONING_TURNAROUND") AS "ACCESSIONING_TURNAROUND",
    COUNT("TRANSPORT_TEMPERATURE_RANGE") AS "TRANSPORT_TEMPERATURE_RANGE",
    COUNT("DANGEROUS_GOODS_CERTIFICATION") AS "DANGEROUS_GOODS_CERTIFICATION",
    COUNT("COURIER_PARTNERS") AS "COURIER_PARTNERS",
    COUNT("IMPORT_AND_EXPORT_PERMIT_SUPPORT") AS "IMPORT_AND_EXPORT_PERMIT_SUPPORT",
    COUNT("SAMPLE_RELOCATION_SERVICE") AS "SAMPLE_RELOCATION_SERVICE",
    COUNT("BIOSAFETY_LEVEL") AS "BIOSAFETY_LEVEL"
    FROM SNOWFLAKE_LEARNING_DB."HUMAN_BIOSPECIMENS".HUMAN_BIOBANKING_STORAGE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('STORAGE_TEMPERATURES', 'Storage temperatures', FALSE),
    ('SAMPLE_TYPES_STORED', 'Sample types stored', FALSE),
    ('FACILITY_LOCATIONS', 'Facility locations', FALSE),
    ('BACKUP_SYSTEMS', 'Backup systems', FALSE),
    ('LIMS_OR_CLIENT_PORTAL', 'LIMS or client portal', FALSE),
    ('INVENTORY_REPORTING', 'Inventory reporting', FALSE),
    ('CHAIN_OF_CUSTODY', 'Chain of custody', FALSE),
    ('ACCREDITATIONS', 'Accreditations', FALSE),
    ('GXP_COMPLIANCE', 'GxP compliance', FALSE),
    ('21_CFR_PART_11_COMPLIANT', '21 CFR Part 11 compliant', FALSE),
    ('TEMPERATURE_MONITORING', 'Temperature monitoring', FALSE),
    ('RETRIEVAL_TURNAROUND', 'Retrieval turnaround', FALSE),
    ('ACCESSIONING_TURNAROUND', 'Accessioning turnaround', FALSE),
    ('TRANSPORT_TEMPERATURE_RANGE', 'Transport temperature range', FALSE),
    ('DANGEROUS_GOODS_CERTIFICATION', 'Dangerous goods certification', FALSE),
    ('COURIER_PARTNERS', 'Courier partners', FALSE),
    ('IMPORT_AND_EXPORT_PERMIT_SUPPORT', 'Import and export permit support', FALSE),
    ('SAMPLE_RELOCATION_SERVICE', 'Sample relocation service', FALSE),
    ('BIOSAFETY_LEVEL', 'Biosafety level', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'human_biospecimens' AS ASSIGNED_L2_ID,
    'Human Biospecimens' AS ASSIGNED_L2_LABEL,
    'human_biobanking_storage' AS ASSIGNED_L3_ID,
    'Human Biobanking & storage' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Human Biospecimens > Human Biobanking & storage
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SERVICES_OFFERED") AS "SERVICES_OFFERED",
    COUNT("EMBEDDING_TYPES") AS "EMBEDDING_TYPES",
    COUNT("INPUT_SAMPLE_TYPES") AS "INPUT_SAMPLE_TYPES",
    COUNT("SPECIAL_STAINS") AS "SPECIAL_STAINS",
    COUNT("TURNAROUND") AS "TURNAROUND",
    COUNT("ACCREDITATIONS") AS "ACCREDITATIONS",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("CUSTOM_ANTIBODY_OPTIMIZATION") AS "CUSTOM_ANTIBODY_OPTIMIZATION",
    COUNT("IMAGE_ANALYSIS_SOFTWARE") AS "IMAGE_ANALYSIS_SOFTWARE",
    COUNT("FIXATION_REQUIREMENTS") AS "FIXATION_REQUIREMENTS",
    COUNT("INBOUND_SHIPPING_CONDITIONS") AS "INBOUND_SHIPPING_CONDITIONS",
    COUNT("SLIDE_ARCHIVING") AS "SLIDE_ARCHIVING",
    COUNT("TISSUE_DISSOCIATION") AS "TISSUE_DISSOCIATION",
    COUNT("POST_THAW_VIABILITY") AS "POST_THAW_VIABILITY",
    COUNT("NUCLEIC_ACID_EXTRACTION") AS "NUCLEIC_ACID_EXTRACTION"
    FROM SNOWFLAKE_LEARNING_DB."HUMAN_BIOSPECIMENS".SAMPLE_PROCESSING_HISTOLOGY_FFPE_DISSOCIATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SERVICES_OFFERED', 'Services offered', TRUE),
    ('EMBEDDING_TYPES', 'Embedding types', TRUE),
    ('INPUT_SAMPLE_TYPES', 'Input sample types', TRUE),
    ('SPECIAL_STAINS', 'Special stains', TRUE),
    ('TURNAROUND', 'Turnaround', TRUE),
    ('ACCREDITATIONS', 'Accreditations', TRUE),
    ('DELIVERABLES', 'Deliverables', TRUE),
    ('CUSTOM_ANTIBODY_OPTIMIZATION', 'Custom antibody optimization', FALSE),
    ('IMAGE_ANALYSIS_SOFTWARE', 'Image analysis software', FALSE),
    ('FIXATION_REQUIREMENTS', 'Fixation requirements', FALSE),
    ('INBOUND_SHIPPING_CONDITIONS', 'Inbound shipping conditions', FALSE),
    ('SLIDE_ARCHIVING', 'Slide archiving', FALSE),
    ('TISSUE_DISSOCIATION', 'Tissue dissociation', FALSE),
    ('POST_THAW_VIABILITY', 'Post-thaw viability', FALSE),
    ('NUCLEIC_ACID_EXTRACTION', 'Nucleic acid extraction', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'human_biospecimens' AS ASSIGNED_L2_ID,
    'Human Biospecimens' AS ASSIGNED_L2_LABEL,
    'sample_processing_histology_ffpe_dissociation' AS ASSIGNED_L3_ID,
    'Sample processing (histology / FFPE / dissociation)' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Human Biospecimens > Sample processing (histology / FFPE / dissociation)
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_KITS_ASSAYS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("ANALYTE") AS "ANALYTE",
    COUNT("ASSAY_FORMAT") AS "ASSAY_FORMAT",
    COUNT("SAMPLE_TYPE_ELISA") AS "SAMPLE_TYPE_ELISA",
    COUNT("SENSITIVITY_ELISA") AS "SENSITIVITY_ELISA",
    COUNT("RANGE_ELISA") AS "RANGE_ELISA",
    COUNT("SPECIES_ELISA") AS "SPECIES_ELISA",
    COUNT("DETECTION_METHOD_ELISA") AS "DETECTION_METHOD_ELISA",
    COUNT("STORAGE_ELISA") AS "STORAGE_ELISA",
    COUNT("KIT_FORMAT") AS "KIT_FORMAT",
    COUNT("TESTS_PER_KIT_ELISA") AS "TESTS_PER_KIT_ELISA",
    COUNT("APPLICATION_ELISA") AS "APPLICATION_ELISA",
    COUNT("ASSAY_TIME_ELISA") AS "ASSAY_TIME_ELISA",
    COUNT("SAMPLE_VOLUME") AS "SAMPLE_VOLUME",
    COUNT("PLATE_FORMAT") AS "PLATE_FORMAT",
    COUNT("QUANTITATION_TYPE") AS "QUANTITATION_TYPE",
    COUNT("CROSS_REACTIVITY") AS "CROSS_REACTIVITY",
    COUNT("INTRA_ASSAY_CV") AS "INTRA_ASSAY_CV",
    COUNT("INTER_ASSAY_CV") AS "INTER_ASSAY_CV",
    COUNT("SPIKE_RECOVERY") AS "SPIKE_RECOVERY",
    COUNT("LINEARITY_OF_DILUTION") AS "LINEARITY_OF_DILUTION",
    COUNT("GENE_ID") AS "GENE_ID",
    COUNT("UNIPROT_ACCESSION") AS "UNIPROT_ACCESSION",
    COUNT("KIT_COMPONENTS_ELISA") AS "KIT_COMPONENTS_ELISA",
    COUNT("SHIPPING_CONDITION_ELISA") AS "SHIPPING_CONDITION_ELISA"
    FROM SNOWFLAKE_LEARNING_DB."KITS_ASSAYS".ELISA_KITS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('ANALYTE', 'Analyte', TRUE),
    ('ASSAY_FORMAT', 'Assay format', TRUE),
    ('SAMPLE_TYPE_ELISA', 'Sample type (elisa)', TRUE),
    ('SENSITIVITY_ELISA', 'Sensitivity (elisa)', FALSE),
    ('RANGE_ELISA', 'Range (elisa)', FALSE),
    ('SPECIES_ELISA', 'Species (elisa)', FALSE),
    ('DETECTION_METHOD_ELISA', 'Detection method (elisa)', FALSE),
    ('STORAGE_ELISA', 'Storage (elisa)', FALSE),
    ('KIT_FORMAT', 'Kit format', FALSE),
    ('TESTS_PER_KIT_ELISA', 'Tests per kit (elisa)', FALSE),
    ('APPLICATION_ELISA', 'Application (elisa)', FALSE),
    ('ASSAY_TIME_ELISA', 'Assay time (elisa)', FALSE),
    ('SAMPLE_VOLUME', 'Sample volume', FALSE),
    ('PLATE_FORMAT', 'Plate format', FALSE),
    ('QUANTITATION_TYPE', 'Quantitation type', FALSE),
    ('CROSS_REACTIVITY', 'Cross-reactivity', FALSE),
    ('INTRA_ASSAY_CV', 'Intra-assay CV', FALSE),
    ('INTER_ASSAY_CV', 'Inter-assay CV', FALSE),
    ('SPIKE_RECOVERY', 'Spike recovery', FALSE),
    ('LINEARITY_OF_DILUTION', 'Linearity of dilution', FALSE),
    ('GENE_ID', 'Gene ID', FALSE),
    ('UNIPROT_ACCESSION', 'UniProt accession', FALSE),
    ('KIT_COMPONENTS_ELISA', 'Kit components (elisa)', FALSE),
    ('SHIPPING_CONDITION_ELISA', 'Shipping condition (elisa)', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'kits_assays' AS ASSIGNED_L2_ID,
    'Kits and Assays' AS ASSIGNED_L2_LABEL,
    'elisa_kits' AS ASSIGNED_L3_ID,
    'ELISA Kits' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Kits and Assays > ELISA Kits
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("READOUT") AS "READOUT",
    COUNT("APPLICATION_CELL_BASED") AS "APPLICATION_CELL_BASED",
    COUNT("WELL_FORMAT") AS "WELL_FORMAT",
    COUNT("STORAGE_CELL_BASED") AS "STORAGE_CELL_BASED",
    COUNT("REAGENT_FORM_CELL_BASED") AS "REAGENT_FORM_CELL_BASED",
    COUNT("TESTS_PER_KIT_CELL_BASED") AS "TESTS_PER_KIT_CELL_BASED",
    COUNT("SAMPLE_TYPE_CELL_BASED") AS "SAMPLE_TYPE_CELL_BASED",
    COUNT("ASSAY_CHEMISTRY") AS "ASSAY_CHEMISTRY",
    COUNT("ASSAY_TIME_CELL_BASED") AS "ASSAY_TIME_CELL_BASED",
    COUNT("DETECTION_INSTRUMENT") AS "DETECTION_INSTRUMENT",
    COUNT("WAVELENGTH_CELL_BASED") AS "WAVELENGTH_CELL_BASED",
    COUNT("SENSITIVITY_CELL_BASED") AS "SENSITIVITY_CELL_BASED",
    COUNT("TARGET_CELL_BASED") AS "TARGET_CELL_BASED",
    COUNT("KIT_COMPONENTS_CELL_BASED") AS "KIT_COMPONENTS_CELL_BASED",
    COUNT("SHIPPING_CONDITION_CELL_BASED") AS "SHIPPING_CONDITION_CELL_BASED",
    COUNT("MULTIPLEX") AS "MULTIPLEX",
    COUNT("MULTIPLEX_COMPATIBLE") AS "MULTIPLEX_COMPATIBLE"
    FROM SNOWFLAKE_LEARNING_DB."KITS_ASSAYS".CELL_BASED_ASSAYS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('READOUT', 'Readout', TRUE),
    ('APPLICATION_CELL_BASED', 'Application (cell-based)', TRUE),
    ('WELL_FORMAT', 'Well format', FALSE),
    ('STORAGE_CELL_BASED', 'Storage (cell-based)', FALSE),
    ('REAGENT_FORM_CELL_BASED', 'Reagent form (cell-based)', FALSE),
    ('TESTS_PER_KIT_CELL_BASED', 'Tests per kit (cell-based)', FALSE),
    ('SAMPLE_TYPE_CELL_BASED', 'Sample type (cell-based)', FALSE),
    ('ASSAY_CHEMISTRY', 'Assay chemistry', FALSE),
    ('ASSAY_TIME_CELL_BASED', 'Assay time (cell-based)', FALSE),
    ('DETECTION_INSTRUMENT', 'Detection instrument', FALSE),
    ('WAVELENGTH_CELL_BASED', 'Wavelength (cell-based)', FALSE),
    ('SENSITIVITY_CELL_BASED', 'Sensitivity (cell-based)', FALSE),
    ('TARGET_CELL_BASED', 'Target (cell-based)', FALSE),
    ('KIT_COMPONENTS_CELL_BASED', 'Kit components (cell-based)', FALSE),
    ('SHIPPING_CONDITION_CELL_BASED', 'Shipping condition (cell-based)', FALSE),
    ('MULTIPLEX', 'Multiplex', FALSE),
    ('MULTIPLEX_COMPATIBLE', 'Multiplex-compatible', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'kits_assays' AS ASSIGNED_L2_ID,
    'Kits and Assays' AS ASSIGNED_L2_LABEL,
    'cell_based_assays' AS ASSIGNED_L3_ID,
    'Cell-Based Assays' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Kits and Assays > Cell-Based Assays
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("TARGET_NUCLEIC_ACID") AS "TARGET_NUCLEIC_ACID",
    COUNT("INPUT_SAMPLE_TYPE") AS "INPUT_SAMPLE_TYPE",
    COUNT("METHOD_NUCLEIC_ACID") AS "METHOD_NUCLEIC_ACID",
    COUNT("YIELD") AS "YIELD",
    COUNT("ELUTION_VOLUME") AS "ELUTION_VOLUME",
    COUNT("TESTS_PER_KIT_NUCLEIC_ACID") AS "TESTS_PER_KIT_NUCLEIC_ACID",
    COUNT("PROCESSING_TIME") AS "PROCESSING_TIME",
    COUNT("SAMPLE_INPUT_AMOUNT") AS "SAMPLE_INPUT_AMOUNT",
    COUNT("FORMAT_NUCLEIC_ACID") AS "FORMAT_NUCLEIC_ACID",
    COUNT("AUTOMATION_COMPATIBLE") AS "AUTOMATION_COMPATIBLE",
    COUNT("DOWNSTREAM_APPLICATION") AS "DOWNSTREAM_APPLICATION",
    COUNT("PURITY_A260_A280") AS "PURITY_A260_A280",
    COUNT("BLOOD_COLLECTION_TUBE") AS "BLOOD_COLLECTION_TUBE",
    COUNT("STORAGE_NUCLEIC_ACID") AS "STORAGE_NUCLEIC_ACID"
    FROM SNOWFLAKE_LEARNING_DB."KITS_ASSAYS".NUCLEIC_ACID_EXTRACTION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('TARGET_NUCLEIC_ACID', 'Target (nucleic acid)', TRUE),
    ('INPUT_SAMPLE_TYPE', 'Input sample type', TRUE),
    ('METHOD_NUCLEIC_ACID', 'Method (nucleic acid)', FALSE),
    ('YIELD', 'Yield', FALSE),
    ('ELUTION_VOLUME', 'Elution volume', FALSE),
    ('TESTS_PER_KIT_NUCLEIC_ACID', 'Tests per kit (nucleic acid)', FALSE),
    ('PROCESSING_TIME', 'Processing time', FALSE),
    ('SAMPLE_INPUT_AMOUNT', 'Sample input amount', FALSE),
    ('FORMAT_NUCLEIC_ACID', 'Format (nucleic acid)', FALSE),
    ('AUTOMATION_COMPATIBLE', 'Automation compatible', FALSE),
    ('DOWNSTREAM_APPLICATION', 'Downstream application', FALSE),
    ('PURITY_A260_A280', 'Purity (A260/A280)', FALSE),
    ('BLOOD_COLLECTION_TUBE', 'Blood collection tube', FALSE),
    ('STORAGE_NUCLEIC_ACID', 'Storage (nucleic acid)', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'kits_assays' AS ASSIGNED_L2_ID,
    'Kits and Assays' AS ASSIGNED_L2_LABEL,
    'nucleic_acid_extraction' AS ASSIGNED_L3_ID,
    'Nucleic Acid Extraction Kits' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Kits and Assays > Nucleic Acid Extraction Kits
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("METHOD_PROTEIN_QUANT") AS "METHOD_PROTEIN_QUANT",
    COUNT("RANGE_PROTEIN_QUANT") AS "RANGE_PROTEIN_QUANT",
    COUNT("STORAGE_PROTEIN_QUANT") AS "STORAGE_PROTEIN_QUANT",
    COUNT("REAGENT_FORM_PROTEIN_QUANT") AS "REAGENT_FORM_PROTEIN_QUANT",
    COUNT("TESTS_PER_KIT_PROTEIN_QUANT") AS "TESTS_PER_KIT_PROTEIN_QUANT",
    COUNT("DETECTION_METHOD_PROTEIN_QUANT") AS "DETECTION_METHOD_PROTEIN_QUANT",
    COUNT("FORMAT_PROTEIN_QUANT") AS "FORMAT_PROTEIN_QUANT",
    COUNT("STANDARD_INCLUDED") AS "STANDARD_INCLUDED",
    COUNT("DETERGENT_COMPATIBILITY") AS "DETERGENT_COMPATIBILITY",
    COUNT("REDUCING_AGENT_COMPATIBILITY") AS "REDUCING_AGENT_COMPATIBILITY",
    COUNT("WAVELENGTH_PROTEIN_QUANT") AS "WAVELENGTH_PROTEIN_QUANT",
    COUNT("ASSAY_TIME_PROTEIN_QUANT") AS "ASSAY_TIME_PROTEIN_QUANT",
    COUNT("SENSITIVITY_PROTEIN_QUANT") AS "SENSITIVITY_PROTEIN_QUANT",
    COUNT("SAMPLE_TYPE_PROTEIN_QUANT") AS "SAMPLE_TYPE_PROTEIN_QUANT"
    FROM SNOWFLAKE_LEARNING_DB."KITS_ASSAYS".PROTEIN_QUANTIFICATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('METHOD_PROTEIN_QUANT', 'Method (protein quant)', TRUE),
    ('RANGE_PROTEIN_QUANT', 'Range (protein quant)', FALSE),
    ('STORAGE_PROTEIN_QUANT', 'Storage (protein quant)', FALSE),
    ('REAGENT_FORM_PROTEIN_QUANT', 'Reagent form (protein quant)', FALSE),
    ('TESTS_PER_KIT_PROTEIN_QUANT', 'Tests per kit (protein quant)', FALSE),
    ('DETECTION_METHOD_PROTEIN_QUANT', 'Detection method (protein quant)', FALSE),
    ('FORMAT_PROTEIN_QUANT', 'Format (protein quant)', FALSE),
    ('STANDARD_INCLUDED', 'Standard included', FALSE),
    ('DETERGENT_COMPATIBILITY', 'Detergent compatibility', FALSE),
    ('REDUCING_AGENT_COMPATIBILITY', 'Reducing agent compatibility', FALSE),
    ('WAVELENGTH_PROTEIN_QUANT', 'Wavelength (protein quant)', FALSE),
    ('ASSAY_TIME_PROTEIN_QUANT', 'Assay time (protein quant)', FALSE),
    ('SENSITIVITY_PROTEIN_QUANT', 'Sensitivity (protein quant)', FALSE),
    ('SAMPLE_TYPE_PROTEIN_QUANT', 'Sample type (protein quant)', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'kits_assays' AS ASSIGNED_L2_ID,
    'Kits and Assays' AS ASSIGNED_L2_LABEL,
    'protein_quantification' AS ASSIGNED_L3_ID,
    'Protein Quantification Kits' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Kits and Assays > Protein Quantification Kits
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DETECTION_METHOD_DETECTION_KITS") AS "DETECTION_METHOD_DETECTION_KITS",
    COUNT("ENZYME") AS "ENZYME",
    COUNT("SUBSTRATE") AS "SUBSTRATE",
    COUNT("APPLICATION_DETECTION_KITS") AS "APPLICATION_DETECTION_KITS",
    COUNT("STORAGE_DETECTION_KITS") AS "STORAGE_DETECTION_KITS",
    COUNT("REAGENT_FORM_DETECTION_KITS") AS "REAGENT_FORM_DETECTION_KITS",
    COUNT("TESTS_PER_KIT_DETECTION_KITS") AS "TESTS_PER_KIT_DETECTION_KITS",
    COUNT("ANTIBODY_HOST_COMPATIBILITY") AS "ANTIBODY_HOST_COMPATIBILITY",
    COUNT("DETECTION_SYSTEM") AS "DETECTION_SYSTEM",
    COUNT("SAMPLE_TYPE_DETECTION_KITS") AS "SAMPLE_TYPE_DETECTION_KITS",
    COUNT("CHROMOGEN_COLOR") AS "CHROMOGEN_COLOR",
    COUNT("SENSITIVITY_DETECTION_KITS") AS "SENSITIVITY_DETECTION_KITS",
    COUNT("SIGNAL_DURATION") AS "SIGNAL_DURATION",
    COUNT("SHELF_LIFE_DETECTION_KITS") AS "SHELF_LIFE_DETECTION_KITS",
    COUNT("RECOMMENDED_ANTIBODY_DILUTION") AS "RECOMMENDED_ANTIBODY_DILUTION",
    COUNT("KIT_COMPONENTS_DETECTION_KITS") AS "KIT_COMPONENTS_DETECTION_KITS",
    COUNT("MEMBRANE_COMPATIBILITY") AS "MEMBRANE_COMPATIBILITY",
    COUNT("IMAGING_COMPATIBILITY") AS "IMAGING_COMPATIBILITY"
    FROM SNOWFLAKE_LEARNING_DB."KITS_ASSAYS".DETECTION_IMAGING_KITS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DETECTION_METHOD_DETECTION_KITS', 'Detection method (detection kits)', TRUE),
    ('ENZYME', 'Enzyme', FALSE),
    ('SUBSTRATE', 'Substrate', FALSE),
    ('APPLICATION_DETECTION_KITS', 'Application (detection kits)', FALSE),
    ('STORAGE_DETECTION_KITS', 'Storage (detection kits)', FALSE),
    ('REAGENT_FORM_DETECTION_KITS', 'Reagent form (detection kits)', FALSE),
    ('TESTS_PER_KIT_DETECTION_KITS', 'Tests per kit (detection kits)', FALSE),
    ('ANTIBODY_HOST_COMPATIBILITY', 'Antibody host compatibility', FALSE),
    ('DETECTION_SYSTEM', 'Detection system', FALSE),
    ('SAMPLE_TYPE_DETECTION_KITS', 'Sample type (detection kits)', FALSE),
    ('CHROMOGEN_COLOR', 'Chromogen color', FALSE),
    ('SENSITIVITY_DETECTION_KITS', 'Sensitivity (detection kits)', FALSE),
    ('SIGNAL_DURATION', 'Signal duration', FALSE),
    ('SHELF_LIFE_DETECTION_KITS', 'Shelf life (detection kits)', FALSE),
    ('RECOMMENDED_ANTIBODY_DILUTION', 'Recommended antibody dilution', FALSE),
    ('KIT_COMPONENTS_DETECTION_KITS', 'Kit components (detection kits)', FALSE),
    ('MEMBRANE_COMPATIBILITY', 'Membrane compatibility', FALSE),
    ('IMAGING_COMPATIBILITY', 'Imaging compatibility', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'kits_assays' AS ASSIGNED_L2_ID,
    'Kits and Assays' AS ASSIGNED_L2_LABEL,
    'detection_imaging_kits' AS ASSIGNED_L3_ID,
    'Detection & Imaging Kits' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Kits and Assays > Detection & Imaging Kits
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("APPLICATION_FLOW_ASSAYS") AS "APPLICATION_FLOW_ASSAYS",
    COUNT("MARKERS") AS "MARKERS",
    COUNT("FLUOROPHORES") AS "FLUOROPHORES",
    COUNT("SPECIES_REACTIVITY_FLOW_ASSAYS") AS "SPECIES_REACTIVITY_FLOW_ASSAYS",
    COUNT("STORAGE_FLOW_ASSAYS") AS "STORAGE_FLOW_ASSAYS",
    COUNT("REAGENT_FORM_FLOW_ASSAYS") AS "REAGENT_FORM_FLOW_ASSAYS",
    COUNT("TESTS_PER_KIT_FLOW_ASSAYS") AS "TESTS_PER_KIT_FLOW_ASSAYS",
    COUNT("EXCITATION_LASER") AS "EXCITATION_LASER",
    COUNT("FIXABLE") AS "FIXABLE",
    COUNT("DETECTION_PLATFORM_FLOW_ASSAYS") AS "DETECTION_PLATFORM_FLOW_ASSAYS",
    COUNT("SAMPLE_TYPE_FLOW_ASSAYS") AS "SAMPLE_TYPE_FLOW_ASSAYS",
    COUNT("NUMBER_OF_COLORS") AS "NUMBER_OF_COLORS",
    COUNT("REGULATORY_STATUS_FLOW_ASSAYS") AS "REGULATORY_STATUS_FLOW_ASSAYS",
    COUNT("KIT_COMPONENTS_FLOW_ASSAYS") AS "KIT_COMPONENTS_FLOW_ASSAYS",
    COUNT("SHIPPING_CONDITION_FLOW_ASSAYS") AS "SHIPPING_CONDITION_FLOW_ASSAYS"
    FROM SNOWFLAKE_LEARNING_DB."KITS_ASSAYS".FLOW_CYTOMETRY_ASSAYS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('APPLICATION_FLOW_ASSAYS', 'Application (flow assays)', TRUE),
    ('MARKERS', 'Markers', FALSE),
    ('FLUOROPHORES', 'Fluorophores', FALSE),
    ('SPECIES_REACTIVITY_FLOW_ASSAYS', 'Species reactivity (flow assays)', FALSE),
    ('STORAGE_FLOW_ASSAYS', 'Storage (flow assays)', FALSE),
    ('REAGENT_FORM_FLOW_ASSAYS', 'Reagent form (flow assays)', FALSE),
    ('TESTS_PER_KIT_FLOW_ASSAYS', 'Tests per kit (flow assays)', FALSE),
    ('EXCITATION_LASER', 'Excitation laser', FALSE),
    ('FIXABLE', 'Fixable', FALSE),
    ('DETECTION_PLATFORM_FLOW_ASSAYS', 'Detection platform (flow assays)', FALSE),
    ('SAMPLE_TYPE_FLOW_ASSAYS', 'Sample type (flow assays)', FALSE),
    ('NUMBER_OF_COLORS', 'Number of colors', FALSE),
    ('REGULATORY_STATUS_FLOW_ASSAYS', 'Regulatory status (flow assays)', FALSE),
    ('KIT_COMPONENTS_FLOW_ASSAYS', 'Kit components (flow assays)', FALSE),
    ('SHIPPING_CONDITION_FLOW_ASSAYS', 'Shipping condition (flow assays)', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'kits_assays' AS ASSIGNED_L2_ID,
    'Kits and Assays' AS ASSIGNED_L2_LABEL,
    'flow_cytometry_assays' AS ASSIGNED_L3_ID,
    'Flow Cytometry Assays' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Kits and Assays > Flow Cytometry Assays
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("ASSAY_FORMAT_CUSTOM_DEV") AS "ASSAY_FORMAT_CUSTOM_DEV",
    COUNT("TARGET_CLASS_CUSTOM_DEV") AS "TARGET_CLASS_CUSTOM_DEV",
    COUNT("DETECTION_TECHNOLOGY_CUSTOM_DEV") AS "DETECTION_TECHNOLOGY_CUSTOM_DEV",
    COUNT("HTS_READY_OUTPUT") AS "HTS_READY_OUTPUT",
    COUNT("CELL_LINE_ENGINEERING_INCLUDED") AS "CELL_LINE_ENGINEERING_INCLUDED",
    COUNT("CLIENT_MATERIALS_ACCEPTED") AS "CLIENT_MATERIALS_ACCEPTED",
    COUNT("CELL_MODEL_TYPE") AS "CELL_MODEL_TYPE",
    COUNT("SPECIES_CUSTOM_DEV") AS "SPECIES_CUSTOM_DEV",
    COUNT("VALIDATION_PARAMETERS_CUSTOM_DEV") AS "VALIDATION_PARAMETERS_CUSTOM_DEV",
    COUNT("POTENCY_READOUT_CUSTOM_DEV") AS "POTENCY_READOUT_CUSTOM_DEV",
    COUNT("DELIVERABLES_CUSTOM_DEV") AS "DELIVERABLES_CUSTOM_DEV",
    COUNT("TURNAROUND_TIME_CUSTOM_DEV") AS "TURNAROUND_TIME_CUSTOM_DEV",
    COUNT("QUALITY_SYSTEM_CUSTOM_DEV") AS "QUALITY_SYSTEM_CUSTOM_DEV",
    COUNT("REGULATORY_GUIDELINE_CUSTOM_DEV") AS "REGULATORY_GUIDELINE_CUSTOM_DEV",
    COUNT("PROTEIN_PRODUCTION_INCLUDED") AS "PROTEIN_PRODUCTION_INCLUDED"
    FROM SNOWFLAKE_LEARNING_DB."KITS_ASSAYS".CUSTOM_ASSAY_DEVELOPMENT_VALIDATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('ASSAY_FORMAT_CUSTOM_DEV', 'Assay format (custom dev)', TRUE),
    ('TARGET_CLASS_CUSTOM_DEV', 'Target class (custom dev)', TRUE),
    ('DETECTION_TECHNOLOGY_CUSTOM_DEV', 'Detection technology (custom dev)', FALSE),
    ('HTS_READY_OUTPUT', 'HTS-ready output', FALSE),
    ('CELL_LINE_ENGINEERING_INCLUDED', 'Cell line engineering included', FALSE),
    ('CLIENT_MATERIALS_ACCEPTED', 'Client materials accepted', FALSE),
    ('CELL_MODEL_TYPE', 'Cell model type', FALSE),
    ('SPECIES_CUSTOM_DEV', 'Species (custom dev)', FALSE),
    ('VALIDATION_PARAMETERS_CUSTOM_DEV', 'Validation parameters (custom dev)', FALSE),
    ('POTENCY_READOUT_CUSTOM_DEV', 'Potency readout (custom dev)', FALSE),
    ('DELIVERABLES_CUSTOM_DEV', 'Deliverables (custom dev)', FALSE),
    ('TURNAROUND_TIME_CUSTOM_DEV', 'Turnaround time (custom dev)', FALSE),
    ('QUALITY_SYSTEM_CUSTOM_DEV', 'Quality system (custom dev)', FALSE),
    ('REGULATORY_GUIDELINE_CUSTOM_DEV', 'Regulatory guideline (custom dev)', FALSE),
    ('PROTEIN_PRODUCTION_INCLUDED', 'Protein production included', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'kits_assays' AS ASSIGNED_L2_ID,
    'Kits and Assays' AS ASSIGNED_L2_LABEL,
    'custom_assay_development_validation' AS ASSIGNED_L3_ID,
    'Custom assay development & validation' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Kits and Assays > Custom assay development & validation
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SCREENING_APPROACH") AS "SCREENING_APPROACH",
    COUNT("LIBRARY_SOURCE") AS "LIBRARY_SOURCE",
    COUNT("TARGET_CLASS_HTS") AS "TARGET_CLASS_HTS",
    COUNT("LIBRARY_SIZE") AS "LIBRARY_SIZE",
    COUNT("LIBRARY_TYPE") AS "LIBRARY_TYPE",
    COUNT("PLATE_FORMAT_HTS") AS "PLATE_FORMAT_HTS",
    COUNT("DETECTION_TECHNOLOGY_HTS") AS "DETECTION_TECHNOLOGY_HTS",
    COUNT("ASSAY_DEVELOPMENT_INCLUDED") AS "ASSAY_DEVELOPMENT_INCLUDED",
    COUNT("HIT_CONFIRMATION_STEPS") AS "HIT_CONFIRMATION_STEPS",
    COUNT("COMPOUND_FORMAT") AS "COMPOUND_FORMAT",
    COUNT("AUTOMATION_HTS") AS "AUTOMATION_HTS",
    COUNT("COMPUTATIONAL_HIT_EXPANSION") AS "COMPUTATIONAL_HIT_EXPANSION",
    COUNT("DELIVERABLES_HTS") AS "DELIVERABLES_HTS",
    COUNT("TURNAROUND_TIME_HTS") AS "TURNAROUND_TIME_HTS",
    COUNT("BIOSAFETY_LEVEL_HTS") AS "BIOSAFETY_LEVEL_HTS",
    COUNT("HTS_PRICING_MODEL") AS "HTS_PRICING_MODEL",
    COUNT("SCREEN_QC_METRICS") AS "SCREEN_QC_METRICS",
    COUNT("MAX_COMPOUNDS_PER_CAMPAIGN") AS "MAX_COMPOUNDS_PER_CAMPAIGN",
    COUNT("MAX_HITS_DELIVERED") AS "MAX_HITS_DELIVERED",
    COUNT("CLIENT_SUPPLIED_PROTEIN_REQUIRED") AS "CLIENT_SUPPLIED_PROTEIN_REQUIRED"
    FROM SNOWFLAKE_LEARNING_DB."KITS_ASSAYS".HIGH_THROUGHPUT_SCREENING_HTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SCREENING_APPROACH', 'Screening approach', TRUE),
    ('LIBRARY_SOURCE', 'Library source', TRUE),
    ('TARGET_CLASS_HTS', 'Target class (hts)', FALSE),
    ('LIBRARY_SIZE', 'Library size', FALSE),
    ('LIBRARY_TYPE', 'Library type', FALSE),
    ('PLATE_FORMAT_HTS', 'Plate format (hts)', FALSE),
    ('DETECTION_TECHNOLOGY_HTS', 'Detection technology (hts)', FALSE),
    ('ASSAY_DEVELOPMENT_INCLUDED', 'Assay development included', FALSE),
    ('HIT_CONFIRMATION_STEPS', 'Hit confirmation steps', FALSE),
    ('COMPOUND_FORMAT', 'Compound format', FALSE),
    ('AUTOMATION_HTS', 'Automation (hts)', FALSE),
    ('COMPUTATIONAL_HIT_EXPANSION', 'Computational hit expansion', FALSE),
    ('DELIVERABLES_HTS', 'Deliverables (hts)', FALSE),
    ('TURNAROUND_TIME_HTS', 'Turnaround time (hts)', FALSE),
    ('BIOSAFETY_LEVEL_HTS', 'Biosafety level (hts)', FALSE),
    ('HTS_PRICING_MODEL', 'HTS pricing model', FALSE),
    ('SCREEN_QC_METRICS', 'Screen QC metrics', FALSE),
    ('MAX_COMPOUNDS_PER_CAMPAIGN', 'Max compounds per campaign', FALSE),
    ('MAX_HITS_DELIVERED', 'Max hits delivered', FALSE),
    ('CLIENT_SUPPLIED_PROTEIN_REQUIRED', 'Client-supplied protein required', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'kits_assays' AS ASSIGNED_L2_ID,
    'Kits and Assays' AS ASSIGNED_L2_LABEL,
    'high_throughput_screening_hts' AS ASSIGNED_L3_ID,
    'High-throughput screening (HTS)' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Kits and Assays > High-throughput screening (HTS)
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_LAB_SUPPLIES_CONSUMABLES_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("STERILITY") AS "STERILITY",
    COUNT("STERILIZATION_METHOD") AS "STERILIZATION_METHOD",
    COUNT("CLOSURE_TYPE") AS "CLOSURE_TYPE",
    COUNT("OPACITY") AS "OPACITY",
    COUNT("NECK_FINISH") AS "NECK_FINISH",
    COUNT("CAPACITY_VOLUME_SIZE") AS "CAPACITY_VOLUME_SIZE",
    COUNT("PURITY_CERTS") AS "PURITY_CERTS",
    COUNT("MATERIAL") AS "MATERIAL",
    COUNT("COLOR") AS "COLOR",
    COUNT("TEMPERATURE_RANGE") AS "TEMPERATURE_RANGE"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".BOTTLES_CONTAINERS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('STERILITY', 'Sterility', TRUE),
    ('STERILIZATION_METHOD', 'Sterilization method', FALSE),
    ('CLOSURE_TYPE', 'Closure type', FALSE),
    ('OPACITY', 'Opacity', FALSE),
    ('NECK_FINISH', 'Neck finish', FALSE),
    ('CAPACITY_VOLUME_SIZE', 'Capacity / Volume / Size', FALSE),
    ('PURITY_CERTS', 'Purity certs', FALSE),
    ('MATERIAL', 'Material', FALSE),
    ('COLOR', 'Color', FALSE),
    ('TEMPERATURE_RANGE', 'Temperature range', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'bottles_containers' AS ASSIGNED_L3_ID,
    'Bottles & Containers' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Bottles & Containers
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("STERILITY") AS "STERILITY",
    COUNT("STERILIZATION_METHOD") AS "STERILIZATION_METHOD",
    COUNT("GLASS_TYPE") AS "GLASS_TYPE",
    COUNT("GRADUATION_CLASS") AS "GRADUATION_CLASS",
    COUNT("CAPACITY_VOLUME_SIZE") AS "CAPACITY_VOLUME_SIZE",
    COUNT("TEMPERATURE_RANGE") AS "TEMPERATURE_RANGE"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".LABORATORY_GLASSWARE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('STERILITY', 'Sterility', FALSE),
    ('STERILIZATION_METHOD', 'Sterilization method', FALSE),
    ('GLASS_TYPE', 'Glass type', FALSE),
    ('GRADUATION_CLASS', 'Graduation class', FALSE),
    ('CAPACITY_VOLUME_SIZE', 'Capacity / Volume / Size', FALSE),
    ('TEMPERATURE_RANGE', 'Temperature range', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'laboratory_glassware' AS ASSIGNED_L3_ID,
    'Laboratory Glassware' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Laboratory Glassware
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("STERILITY") AS "STERILITY",
    COUNT("CAP_TYPE") AS "CAP_TYPE",
    COUNT("STERILIZATION_METHOD") AS "STERILIZATION_METHOD",
    COUNT("BOTTOM_SHAPE") AS "BOTTOM_SHAPE",
    COUNT("GRADUATIONS") AS "GRADUATIONS",
    COUNT("MAX_RCF_G_FORCE") AS "MAX_RCF_G_FORCE",
    COUNT("LOW_BINDING") AS "LOW_BINDING",
    COUNT("CAPACITY_VOLUME_SIZE") AS "CAPACITY_VOLUME_SIZE",
    COUNT("PURITY_CERTS") AS "PURITY_CERTS",
    COUNT("MATERIAL") AS "MATERIAL",
    COUNT("COLOR") AS "COLOR",
    COUNT("COMPATIBILITY") AS "COMPATIBILITY",
    COUNT("TEMPERATURE_RANGE") AS "TEMPERATURE_RANGE"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".TUBES_VIALS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('STERILITY', 'Sterility', TRUE),
    ('CAP_TYPE', 'Cap type', TRUE),
    ('STERILIZATION_METHOD', 'Sterilization method', FALSE),
    ('BOTTOM_SHAPE', 'Bottom shape', FALSE),
    ('GRADUATIONS', 'Graduations', FALSE),
    ('MAX_RCF_G_FORCE', 'Max RCF (g-force)', FALSE),
    ('LOW_BINDING', 'Low-binding', FALSE),
    ('CAPACITY_VOLUME_SIZE', 'Capacity / Volume / Size', FALSE),
    ('PURITY_CERTS', 'Purity certs', FALSE),
    ('MATERIAL', 'Material', FALSE),
    ('COLOR', 'Color', FALSE),
    ('COMPATIBILITY', 'Compatibility', FALSE),
    ('TEMPERATURE_RANGE', 'Temperature range', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'tubes_vials' AS ASSIGNED_L3_ID,
    'Tubes & Vials' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Tubes & Vials
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("STERILITY") AS "STERILITY",
    COUNT("WELL_NUMBER") AS "WELL_NUMBER",
    COUNT("WELL_BOTTOM") AS "WELL_BOTTOM",
    COUNT("SURFACE_TREATMENT") AS "SURFACE_TREATMENT",
    COUNT("STERILIZATION_METHOD") AS "STERILIZATION_METHOD",
    COUNT("COLOR") AS "COLOR",
    COUNT("LID") AS "LID",
    COUNT("OPTICAL_ASSAY_READOUT") AS "OPTICAL_ASSAY_READOUT",
    COUNT("PURITY_CERTS") AS "PURITY_CERTS",
    COUNT("MATERIAL") AS "MATERIAL",
    COUNT("COMPATIBILITY") AS "COMPATIBILITY",
    COUNT("TEMPERATURE_RANGE") AS "TEMPERATURE_RANGE"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".MICROPLATES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('STERILITY', 'Sterility', TRUE),
    ('WELL_NUMBER', 'Well number', TRUE),
    ('WELL_BOTTOM', 'Well bottom', TRUE),
    ('SURFACE_TREATMENT', 'Surface treatment', TRUE),
    ('STERILIZATION_METHOD', 'Sterilization method', FALSE),
    ('COLOR', 'Color', FALSE),
    ('LID', 'Lid', FALSE),
    ('OPTICAL_ASSAY_READOUT', 'Optical / assay readout', FALSE),
    ('PURITY_CERTS', 'Purity certs', FALSE),
    ('MATERIAL', 'Material', FALSE),
    ('COMPATIBILITY', 'Compatibility', FALSE),
    ('TEMPERATURE_RANGE', 'Temperature range', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'microplates' AS ASSIGNED_L3_ID,
    'Microplates' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Microplates
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("STERILITY") AS "STERILITY",
    COUNT("VOLUME") AS "VOLUME",
    COUNT("FILTERED_VS_NON_FILTERED") AS "FILTERED_VS_NON_FILTERED",
    COUNT("STERILIZATION_METHOD") AS "STERILIZATION_METHOD",
    COUNT("RACK_FORMAT") AS "RACK_FORMAT",
    COUNT("LOW_RETENTION") AS "LOW_RETENTION",
    COUNT("UNIVERSAL_PIPETTE_FIT") AS "UNIVERSAL_PIPETTE_FIT",
    COUNT("PURITY_CERTS") AS "PURITY_CERTS",
    COUNT("MATERIAL") AS "MATERIAL"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".PIPETTES_TIPS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('STERILITY', 'Sterility', TRUE),
    ('VOLUME', 'Volume', TRUE),
    ('FILTERED_VS_NON_FILTERED', 'Filtered vs non-filtered', TRUE),
    ('STERILIZATION_METHOD', 'Sterilization method', FALSE),
    ('RACK_FORMAT', 'Rack format', FALSE),
    ('LOW_RETENTION', 'Low-retention', FALSE),
    ('UNIVERSAL_PIPETTE_FIT', 'Universal / pipette fit', FALSE),
    ('PURITY_CERTS', 'Purity certs', FALSE),
    ('MATERIAL', 'Material', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'pipettes_tips' AS ASSIGNED_L3_ID,
    'Pipettes & Tips' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Pipettes & Tips
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("STERILITY") AS "STERILITY",
    COUNT("PORE_SIZE") AS "PORE_SIZE",
    COUNT("MEMBRANE_MATERIAL") AS "MEMBRANE_MATERIAL",
    COUNT("DIAMETER") AS "DIAMETER",
    COUNT("STERILIZATION_METHOD") AS "STERILIZATION_METHOD",
    COUNT("HOUSING_MATERIAL") AS "HOUSING_MATERIAL",
    COUNT("LUER_CONNECTION") AS "LUER_CONNECTION",
    COUNT("COMPATIBILITY") AS "COMPATIBILITY"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".FILTRATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('STERILITY', 'Sterility', TRUE),
    ('PORE_SIZE', 'Pore size', TRUE),
    ('MEMBRANE_MATERIAL', 'Membrane material', TRUE),
    ('DIAMETER', 'Diameter', TRUE),
    ('STERILIZATION_METHOD', 'Sterilization method', FALSE),
    ('HOUSING_MATERIAL', 'Housing material', FALSE),
    ('LUER_CONNECTION', 'Luer connection', FALSE),
    ('COMPATIBILITY', 'Compatibility', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'filtration' AS ASSIGNED_L3_ID,
    'Filtration' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Filtration
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("STERILITY") AS "STERILITY",
    COUNT("LUER_TYPE") AS "LUER_TYPE",
    COUNT("STERILIZATION_METHOD") AS "STERILIZATION_METHOD",
    COUNT("NEEDLE_GAUGE_LENGTH") AS "NEEDLE_GAUGE_LENGTH",
    COUNT("CAPACITY_VOLUME_SIZE") AS "CAPACITY_VOLUME_SIZE",
    COUNT("MATERIAL") AS "MATERIAL",
    COUNT("COMPATIBILITY") AS "COMPATIBILITY"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".SYRINGES_NEEDLES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('STERILITY', 'Sterility', TRUE),
    ('LUER_TYPE', 'Luer type', TRUE),
    ('STERILIZATION_METHOD', 'Sterilization method', FALSE),
    ('NEEDLE_GAUGE_LENGTH', 'Needle gauge / length', FALSE),
    ('CAPACITY_VOLUME_SIZE', 'Capacity / Volume / Size', FALSE),
    ('MATERIAL', 'Material', FALSE),
    ('COMPATIBILITY', 'Compatibility', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'syringes_needles' AS ASSIGNED_L3_ID,
    'Syringes & Needles' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Syringes & Needles
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("TYPE") AS "TYPE",
    COUNT("MATERIAL") AS "MATERIAL",
    COUNT("SIZE") AS "SIZE",
    COUNT("STERILITY") AS "STERILITY",
    COUNT("STERILIZATION_METHOD") AS "STERILIZATION_METHOD",
    COUNT("POWDER_FREE") AS "POWDER_FREE",
    COUNT("COLOR") AS "COLOR"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".SAFETY_PPE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('TYPE', 'Type', TRUE),
    ('MATERIAL', 'Material', TRUE),
    ('SIZE', 'Size', TRUE),
    ('STERILITY', 'Sterility', FALSE),
    ('STERILIZATION_METHOD', 'Sterilization method', FALSE),
    ('POWDER_FREE', 'Powder-free', FALSE),
    ('COLOR', 'Color', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'safety_ppe' AS ASSIGNED_L3_ID,
    'Safety & PPE' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Safety & PPE
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("INNER_DIAMETER") AS "INNER_DIAMETER",
    COUNT("TUBING_MATERIAL") AS "TUBING_MATERIAL",
    COUNT("STERILITY") AS "STERILITY",
    COUNT("STERILIZATION_METHOD") AS "STERILIZATION_METHOD",
    COUNT("OUTER_DIAMETER") AS "OUTER_DIAMETER",
    COUNT("WALL_THICKNESS") AS "WALL_THICKNESS",
    COUNT("LENGTH") AS "LENGTH",
    COUNT("DUROMETER") AS "DUROMETER",
    COUNT("MAX_PRESSURE") AS "MAX_PRESSURE",
    COUNT("TUBING_TYPE") AS "TUBING_TYPE",
    COUNT("COMPATIBILITY") AS "COMPATIBILITY",
    COUNT("TEMPERATURE_RANGE") AS "TEMPERATURE_RANGE"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".TUBING_HOSES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('INNER_DIAMETER', 'Inner diameter', TRUE),
    ('TUBING_MATERIAL', 'Tubing material', TRUE),
    ('STERILITY', 'Sterility', FALSE),
    ('STERILIZATION_METHOD', 'Sterilization method', FALSE),
    ('OUTER_DIAMETER', 'Outer diameter', FALSE),
    ('WALL_THICKNESS', 'Wall thickness', FALSE),
    ('LENGTH', 'Length', FALSE),
    ('DUROMETER', 'Durometer', FALSE),
    ('MAX_PRESSURE', 'Max pressure', FALSE),
    ('TUBING_TYPE', 'Tubing type', FALSE),
    ('COMPATIBILITY', 'Compatibility', FALSE),
    ('TEMPERATURE_RANGE', 'Temperature range', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'tubing_hoses' AS ASSIGNED_L3_ID,
    'Tubing & Hoses' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Tubing & Hoses
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("ACTIVE_INGREDIENT") AS "ACTIVE_INGREDIENT",
    COUNT("EFFICACY") AS "EFFICACY",
    COUNT("EPA_REGISTERED") AS "EPA_REGISTERED",
    COUNT("USE_ENVIRONMENT") AS "USE_ENVIRONMENT"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".CLEANING_STERILIZATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('PRODUCT_TYPE', 'Product type', TRUE),
    ('FORMAT', 'Format', TRUE),
    ('ACTIVE_INGREDIENT', 'Active ingredient', FALSE),
    ('EFFICACY', 'Efficacy', FALSE),
    ('EPA_REGISTERED', 'EPA registered', FALSE),
    ('USE_ENVIRONMENT', 'Use environment', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'cleaning_sterilization' AS ASSIGNED_L3_ID,
    'Cleaning & Sterilization' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Cleaning & Sterilization
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("WIDTH") AS "WIDTH",
    COUNT("ROLL_LENGTH") AS "ROLL_LENGTH",
    COUNT("USE") AS "USE",
    COUNT("WRITABLE") AS "WRITABLE",
    COUNT("MATERIAL") AS "MATERIAL",
    COUNT("COLOR") AS "COLOR",
    COUNT("TEMPERATURE_RANGE") AS "TEMPERATURE_RANGE"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".LABELING_TAPE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('WIDTH', 'Width', FALSE),
    ('ROLL_LENGTH', 'Roll length', FALSE),
    ('USE', 'Use', FALSE),
    ('WRITABLE', 'Writable', FALSE),
    ('MATERIAL', 'Material', FALSE),
    ('COLOR', 'Color', FALSE),
    ('TEMPERATURE_RANGE', 'Temperature range', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'labeling_tape' AS ASSIGNED_L3_ID,
    'Labeling & Tape' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Labeling & Tape
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("DIET_FORM") AS "DIET_FORM",
    COUNT("DIET_STERILIZATION") AS "DIET_STERILIZATION",
    COUNT("TARGET_SPECIES") AS "TARGET_SPECIES",
    COUNT("LIFE_STAGE") AS "LIFE_STAGE",
    COUNT("CRUDE_PROTEIN") AS "CRUDE_PROTEIN",
    COUNT("FAT") AS "FAT",
    COUNT("CRUDE_FIBER") AS "CRUDE_FIBER",
    COUNT("ENERGY_DENSITY") AS "ENERGY_DENSITY",
    COUNT("SHELF_LIFE") AS "SHELF_LIFE",
    COUNT("CERTIFIED_DIET") AS "CERTIFIED_DIET"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".RESEARCH_ANIMAL_DIETS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('DIET_FORM', 'Diet form', TRUE),
    ('DIET_STERILIZATION', 'Diet sterilization', FALSE),
    ('TARGET_SPECIES', 'Target species', FALSE),
    ('LIFE_STAGE', 'Life stage', FALSE),
    ('CRUDE_PROTEIN', 'Crude protein', FALSE),
    ('FAT', 'Fat', FALSE),
    ('CRUDE_FIBER', 'Crude fiber', FALSE),
    ('ENERGY_DENSITY', 'Energy density', FALSE),
    ('SHELF_LIFE', 'Shelf life', FALSE),
    ('CERTIFIED_DIET', 'Certified diet', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'research_animal_diets' AS ASSIGNED_L3_ID,
    'Research Animal Diets & Husbandry Supplies' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Research Animal Diets & Husbandry Supplies
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BRAND_MANUFACTURER") AS "BRAND_MANUFACTURER",
    COUNT("ITEM_TYPE") AS "ITEM_TYPE",
    COUNT("DIMENSIONS") AS "DIMENSIONS",
    COUNT("CORNER_STYLE") AS "CORNER_STYLE",
    COUNT("FROSTED") AS "FROSTED",
    COUNT("SURFACE_TREATMENT") AS "SURFACE_TREATMENT"
    FROM SNOWFLAKE_LEARNING_DB."LAB_SUPPLIES_CONSUMABLES".MICROSCOPY_SUPPLIES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BRAND_MANUFACTURER', 'Brand / Manufacturer', FALSE),
    ('ITEM_TYPE', 'Item type', TRUE),
    ('DIMENSIONS', 'Dimensions', FALSE),
    ('CORNER_STYLE', 'Corner style', FALSE),
    ('FROSTED', 'Frosted', FALSE),
    ('SURFACE_TREATMENT', 'Surface treatment', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'lab_supplies_consumables' AS ASSIGNED_L2_ID,
    'Lab Supplies and Consumables' AS ASSIGNED_L2_LABEL,
    'microscopy_supplies' AS ASSIGNED_L3_ID,
    'Microscopy Supplies' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Lab Supplies and Consumables > Microscopy Supplies
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_MEDICAL_AFFAIRS_MEDICAL_COMMUNICATIONS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREAS_MEDAFFAIRS") AS "THERAPEUTIC_AREAS_MEDAFFAIRS",
    COUNT("GEOGRAPHIC_COVERAGE_MEDAFFAIRS") AS "GEOGRAPHIC_COVERAGE_MEDAFFAIRS",
    COUNT("AI_ASSISTED_DELIVERY") AS "AI_ASSISTED_DELIVERY",
    COUNT("PROGRAM_SCOPE") AS "PROGRAM_SCOPE",
    COUNT("PROMOTIONAL_STATUS") AS "PROMOTIONAL_STATUS",
    COUNT("COMPLIANCE_WITH_CODES_OF_PRACTICE") AS "COMPLIANCE_WITH_CODES_OF_PRACTICE",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("ENGAGEMENT_MODEL_MEDAFFAIRS") AS "ENGAGEMENT_MODEL_MEDAFFAIRS",
    COUNT("PUBLICATION_TYPES") AS "PUBLICATION_TYPES",
    COUNT("PUBLICATION_PLANNING") AS "PUBLICATION_PLANNING",
    COUNT("DIGITAL_PUBLICATION_ENHANCEMENTS") AS "DIGITAL_PUBLICATION_ENHANCEMENTS",
    COUNT("PUBLICATION_GAP_ANALYSIS") AS "PUBLICATION_GAP_ANALYSIS",
    COUNT("STEERING_COMMITTEE_MANAGEMENT") AS "STEERING_COMMITTEE_MANAGEMENT",
    COUNT("GPP_COMPLIANCE") AS "GPP_COMPLIANCE",
    COUNT("PUBLICATION_AMPLIFICATION") AS "PUBLICATION_AMPLIFICATION",
    COUNT("LITERATURE_MONITORING_MEDWRITING") AS "LITERATURE_MONITORING_MEDWRITING",
    COUNT("CREDENTIALED_PUBLICATION_STAFF") AS "CREDENTIALED_PUBLICATION_STAFF"
    FROM SNOWFLAKE_LEARNING_DB."MEDICAL_AFFAIRS_MEDICAL_COMMUNICATIONS".MEDICAL_WRITING_PUBLICATIONS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREAS_MEDAFFAIRS', 'Therapeutic areas (medaffairs)', FALSE),
    ('GEOGRAPHIC_COVERAGE_MEDAFFAIRS', 'Geographic coverage (medaffairs)', FALSE),
    ('AI_ASSISTED_DELIVERY', 'AI-assisted delivery', FALSE),
    ('PROGRAM_SCOPE', 'Program scope', FALSE),
    ('PROMOTIONAL_STATUS', 'Promotional status', FALSE),
    ('COMPLIANCE_WITH_CODES_OF_PRACTICE', 'Compliance with codes of practice', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('ENGAGEMENT_MODEL_MEDAFFAIRS', 'Engagement model (medaffairs)', FALSE),
    ('PUBLICATION_TYPES', 'Publication types', TRUE),
    ('PUBLICATION_PLANNING', 'Publication planning', FALSE),
    ('DIGITAL_PUBLICATION_ENHANCEMENTS', 'Digital publication enhancements', FALSE),
    ('PUBLICATION_GAP_ANALYSIS', 'Publication gap analysis', FALSE),
    ('STEERING_COMMITTEE_MANAGEMENT', 'Steering committee management', FALSE),
    ('GPP_COMPLIANCE', 'GPP compliance', FALSE),
    ('PUBLICATION_AMPLIFICATION', 'Publication amplification', FALSE),
    ('LITERATURE_MONITORING_MEDWRITING', 'Literature monitoring (medwriting)', FALSE),
    ('CREDENTIALED_PUBLICATION_STAFF', 'Credentialed publication staff', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'medical_affairs_medical_communications' AS ASSIGNED_L2_ID,
    'Medical Affairs & Medical Communications' AS ASSIGNED_L2_LABEL,
    'medical_writing_publications' AS ASSIGNED_L3_ID,
    'Medical Writing & Publications' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Medical Affairs & Medical Communications > Medical Writing & Publications
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREAS_MEDAFFAIRS") AS "THERAPEUTIC_AREAS_MEDAFFAIRS",
    COUNT("GEOGRAPHIC_COVERAGE_MEDAFFAIRS") AS "GEOGRAPHIC_COVERAGE_MEDAFFAIRS",
    COUNT("AI_ASSISTED_DELIVERY") AS "AI_ASSISTED_DELIVERY",
    COUNT("PROGRAM_SCOPE") AS "PROGRAM_SCOPE",
    COUNT("PROMOTIONAL_STATUS") AS "PROMOTIONAL_STATUS",
    COUNT("COMPLIANCE_WITH_CODES_OF_PRACTICE") AS "COMPLIANCE_WITH_CODES_OF_PRACTICE",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("ENGAGEMENT_MODEL_MEDAFFAIRS") AS "ENGAGEMENT_MODEL_MEDAFFAIRS",
    COUNT("DELIVERABLE_TYPES") AS "DELIVERABLE_TYPES",
    COUNT("SCIENTIFIC_PLATFORM_AND_NARRATIVE") AS "SCIENTIFIC_PLATFORM_AND_NARRATIVE",
    COUNT("MEDICAL_INFORMATION_CONTACT_CENTER") AS "MEDICAL_INFORMATION_CONTACT_CENTER",
    COUNT("ADVERSE_EVENT_INTAKE") AS "ADVERSE_EVENT_INTAKE",
    COUNT("PRODUCT_QUALITY_COMPLAINT_INTAKE") AS "PRODUCT_QUALITY_COMPLAINT_INTAKE",
    COUNT("CONGRESS_SUPPORT") AS "CONGRESS_SUPPORT",
    COUNT("MEETING_FORMAT") AS "MEETING_FORMAT",
    COUNT("MEDICAL_TRAINING_AUDIENCE") AS "MEDICAL_TRAINING_AUDIENCE",
    COUNT("ACCREDITED_MEDICAL_EDUCATION") AS "ACCREDITED_MEDICAL_EDUCATION",
    COUNT("MEDICAL_INFORMATION_LANGUAGES") AS "MEDICAL_INFORMATION_LANGUAGES",
    COUNT("MEDICAL_INFORMATION_HOURS") AS "MEDICAL_INFORMATION_HOURS",
    COUNT("MEDICAL_INFORMATION_CHANNELS") AS "MEDICAL_INFORMATION_CHANNELS",
    COUNT("STANDARD_RESPONSE_DOCUMENTS") AS "STANDARD_RESPONSE_DOCUMENTS",
    COUNT("MEDICAL_CONTENT_TRANSLATION") AS "MEDICAL_CONTENT_TRANSLATION",
    COUNT("SOCIAL_MEDIA_MONITORING") AS "SOCIAL_MEDIA_MONITORING",
    COUNT("PROMOTIONAL_REVIEW") AS "PROMOTIONAL_REVIEW"
    FROM SNOWFLAKE_LEARNING_DB."MEDICAL_AFFAIRS_MEDICAL_COMMUNICATIONS".MEDICAL_COMMUNICATIONS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREAS_MEDAFFAIRS', 'Therapeutic areas (medaffairs)', FALSE),
    ('GEOGRAPHIC_COVERAGE_MEDAFFAIRS', 'Geographic coverage (medaffairs)', FALSE),
    ('AI_ASSISTED_DELIVERY', 'AI-assisted delivery', FALSE),
    ('PROGRAM_SCOPE', 'Program scope', FALSE),
    ('PROMOTIONAL_STATUS', 'Promotional status', FALSE),
    ('COMPLIANCE_WITH_CODES_OF_PRACTICE', 'Compliance with codes of practice', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('ENGAGEMENT_MODEL_MEDAFFAIRS', 'Engagement model (medaffairs)', FALSE),
    ('DELIVERABLE_TYPES', 'Deliverable types', TRUE),
    ('SCIENTIFIC_PLATFORM_AND_NARRATIVE', 'Scientific platform and narrative', FALSE),
    ('MEDICAL_INFORMATION_CONTACT_CENTER', 'Medical information contact center', FALSE),
    ('ADVERSE_EVENT_INTAKE', 'Adverse event intake', FALSE),
    ('PRODUCT_QUALITY_COMPLAINT_INTAKE', 'Product quality complaint intake', FALSE),
    ('CONGRESS_SUPPORT', 'Congress support', FALSE),
    ('MEETING_FORMAT', 'Meeting format', FALSE),
    ('MEDICAL_TRAINING_AUDIENCE', 'Medical training audience', FALSE),
    ('ACCREDITED_MEDICAL_EDUCATION', 'Accredited medical education', FALSE),
    ('MEDICAL_INFORMATION_LANGUAGES', 'Medical information languages', FALSE),
    ('MEDICAL_INFORMATION_HOURS', 'Medical information hours', FALSE),
    ('MEDICAL_INFORMATION_CHANNELS', 'Medical information channels', FALSE),
    ('STANDARD_RESPONSE_DOCUMENTS', 'Standard response documents', FALSE),
    ('MEDICAL_CONTENT_TRANSLATION', 'Medical content translation', FALSE),
    ('SOCIAL_MEDIA_MONITORING', 'Social media monitoring', FALSE),
    ('PROMOTIONAL_REVIEW', 'Promotional review', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'medical_affairs_medical_communications' AS ASSIGNED_L2_ID,
    'Medical Affairs & Medical Communications' AS ASSIGNED_L2_LABEL,
    'medical_communications' AS ASSIGNED_L3_ID,
    'Medical Communications' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Medical Affairs & Medical Communications > Medical Communications
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREAS_MEDAFFAIRS") AS "THERAPEUTIC_AREAS_MEDAFFAIRS",
    COUNT("GEOGRAPHIC_COVERAGE_MEDAFFAIRS") AS "GEOGRAPHIC_COVERAGE_MEDAFFAIRS",
    COUNT("AI_ASSISTED_DELIVERY") AS "AI_ASSISTED_DELIVERY",
    COUNT("PROGRAM_SCOPE") AS "PROGRAM_SCOPE",
    COUNT("PROMOTIONAL_STATUS") AS "PROMOTIONAL_STATUS",
    COUNT("COMPLIANCE_WITH_CODES_OF_PRACTICE") AS "COMPLIANCE_WITH_CODES_OF_PRACTICE",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("ENGAGEMENT_MODEL_MEDAFFAIRS") AS "ENGAGEMENT_MODEL_MEDAFFAIRS",
    COUNT("MSL_TRAINING") AS "MSL_TRAINING",
    COUNT("MSL_DEPLOYMENT_MODEL") AS "MSL_DEPLOYMENT_MODEL",
    COUNT("FIELD_MEDICAL_ROLES") AS "FIELD_MEDICAL_ROLES",
    COUNT("ENGAGEMENT_CHANNEL") AS "ENGAGEMENT_CHANNEL",
    COUNT("RECRUITMENT_AND_TEAM_BUILD") AS "RECRUITMENT_AND_TEAM_BUILD",
    COUNT("FIELD_INSIGHTS_REPORTING") AS "FIELD_INSIGHTS_REPORTING",
    COUNT("PERFORMANCE_KPIS") AS "PERFORMANCE_KPIS",
    COUNT("CRM_INTEGRATION") AS "CRM_INTEGRATION",
    COUNT("COUNTRIES_COVERED_MSL") AS "COUNTRIES_COVERED_MSL"
    FROM SNOWFLAKE_LEARNING_DB."MEDICAL_AFFAIRS_MEDICAL_COMMUNICATIONS".MSL_SUPPORT_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREAS_MEDAFFAIRS', 'Therapeutic areas (medaffairs)', FALSE),
    ('GEOGRAPHIC_COVERAGE_MEDAFFAIRS', 'Geographic coverage (medaffairs)', FALSE),
    ('AI_ASSISTED_DELIVERY', 'AI-assisted delivery', FALSE),
    ('PROGRAM_SCOPE', 'Program scope', FALSE),
    ('PROMOTIONAL_STATUS', 'Promotional status', FALSE),
    ('COMPLIANCE_WITH_CODES_OF_PRACTICE', 'Compliance with codes of practice', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('ENGAGEMENT_MODEL_MEDAFFAIRS', 'Engagement model (medaffairs)', FALSE),
    ('MSL_TRAINING', 'MSL training', FALSE),
    ('MSL_DEPLOYMENT_MODEL', 'MSL deployment model', FALSE),
    ('FIELD_MEDICAL_ROLES', 'Field medical roles', FALSE),
    ('ENGAGEMENT_CHANNEL', 'Engagement channel', FALSE),
    ('RECRUITMENT_AND_TEAM_BUILD', 'Recruitment and team build', FALSE),
    ('FIELD_INSIGHTS_REPORTING', 'Field insights reporting', FALSE),
    ('PERFORMANCE_KPIS', 'Performance KPIs', FALSE),
    ('CRM_INTEGRATION', 'CRM integration', FALSE),
    ('COUNTRIES_COVERED_MSL', 'Countries covered (msl)', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'medical_affairs_medical_communications' AS ASSIGNED_L2_ID,
    'Medical Affairs & Medical Communications' AS ASSIGNED_L2_LABEL,
    'msl_support' AS ASSIGNED_L3_ID,
    'MSL Support' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Medical Affairs & Medical Communications > MSL Support
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREAS_MEDAFFAIRS") AS "THERAPEUTIC_AREAS_MEDAFFAIRS",
    COUNT("GEOGRAPHIC_COVERAGE_MEDAFFAIRS") AS "GEOGRAPHIC_COVERAGE_MEDAFFAIRS",
    COUNT("AI_ASSISTED_DELIVERY") AS "AI_ASSISTED_DELIVERY",
    COUNT("PROGRAM_SCOPE") AS "PROGRAM_SCOPE",
    COUNT("PROMOTIONAL_STATUS") AS "PROMOTIONAL_STATUS",
    COUNT("COMPLIANCE_WITH_CODES_OF_PRACTICE") AS "COMPLIANCE_WITH_CODES_OF_PRACTICE",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("ENGAGEMENT_MODEL_MEDAFFAIRS") AS "ENGAGEMENT_MODEL_MEDAFFAIRS",
    COUNT("KOL_IDENTIFICATION") AS "KOL_IDENTIFICATION",
    COUNT("KOL_MAPPING") AS "KOL_MAPPING",
    COUNT("INSIGHTS_REPORTING") AS "INSIGHTS_REPORTING",
    COUNT("EXPERT_SEGMENTS_COVERED") AS "EXPERT_SEGMENTS_COVERED",
    COUNT("KOL_ENGAGEMENT_PLAN") AS "KOL_ENGAGEMENT_PLAN",
    COUNT("ADVISORY_BOARD_FORMAT") AS "ADVISORY_BOARD_FORMAT",
    COUNT("ADVISORY_BOARD_MANAGEMENT") AS "ADVISORY_BOARD_MANAGEMENT",
    COUNT("SPEAKER_PROGRAMS") AS "SPEAKER_PROGRAMS",
    COUNT("IMPACT_MEASUREMENT") AS "IMPACT_MEASUREMENT"
    FROM SNOWFLAKE_LEARNING_DB."MEDICAL_AFFAIRS_MEDICAL_COMMUNICATIONS".SCIENTIFIC_ADVISORY_KOL_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREAS_MEDAFFAIRS', 'Therapeutic areas (medaffairs)', FALSE),
    ('GEOGRAPHIC_COVERAGE_MEDAFFAIRS', 'Geographic coverage (medaffairs)', FALSE),
    ('AI_ASSISTED_DELIVERY', 'AI-assisted delivery', FALSE),
    ('PROGRAM_SCOPE', 'Program scope', FALSE),
    ('PROMOTIONAL_STATUS', 'Promotional status', FALSE),
    ('COMPLIANCE_WITH_CODES_OF_PRACTICE', 'Compliance with codes of practice', FALSE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('ENGAGEMENT_MODEL_MEDAFFAIRS', 'Engagement model (medaffairs)', FALSE),
    ('KOL_IDENTIFICATION', 'KOL identification', FALSE),
    ('KOL_MAPPING', 'KOL mapping', FALSE),
    ('INSIGHTS_REPORTING', 'Insights reporting', FALSE),
    ('EXPERT_SEGMENTS_COVERED', 'Expert segments covered', FALSE),
    ('KOL_ENGAGEMENT_PLAN', 'KOL engagement plan', FALSE),
    ('ADVISORY_BOARD_FORMAT', 'Advisory board format', FALSE),
    ('ADVISORY_BOARD_MANAGEMENT', 'Advisory board management', FALSE),
    ('SPEAKER_PROGRAMS', 'Speaker programs', FALSE),
    ('IMPACT_MEASUREMENT', 'Impact measurement', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'medical_affairs_medical_communications' AS ASSIGNED_L2_ID,
    'Medical Affairs & Medical Communications' AS ASSIGNED_L2_LABEL,
    'scientific_advisory_kol' AS ASSIGNED_L3_ID,
    'Scientific Advisory / KOL' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Medical Affairs & Medical Communications > Scientific Advisory / KOL
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_MOLECULAR_BIOLOGY_REAGENTS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("REAGENT_TYPE") AS "REAGENT_TYPE",
    COUNT("TARGET_GENE_REGION") AS "TARGET_GENE_REGION",
    COUNT("TARGET_SPECIES") AS "TARGET_SPECIES",
    COUNT("PROBE_CHEMISTRY") AS "PROBE_CHEMISTRY",
    COUNT("REPORTER_DYE") AS "REPORTER_DYE",
    COUNT("QUENCHER") AS "QUENCHER",
    COUNT("CONFIGURATION") AS "CONFIGURATION",
    COUNT("MULTIPLEX") AS "MULTIPLEX"
    FROM SNOWFLAKE_LEARNING_DB."MOLECULAR_BIOLOGY_GENOMICS".PCR_QPCR_REAGENTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('REAGENT_TYPE', 'Reagent type', TRUE),
    ('TARGET_GENE_REGION', 'Target Gene / Region', FALSE),
    ('TARGET_SPECIES', 'Target Species', FALSE),
    ('PROBE_CHEMISTRY', 'Probe chemistry', FALSE),
    ('REPORTER_DYE', 'Reporter dye', FALSE),
    ('QUENCHER', 'Quencher', FALSE),
    ('CONFIGURATION', 'Configuration', FALSE),
    ('MULTIPLEX', 'Multiplex', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'molecular_biology_reagents' AS ASSIGNED_L2_ID,
    'Molecular Biology & Genomics' AS ASSIGNED_L2_LABEL,
    'pcr_qpcr' AS ASSIGNED_L3_ID,
    'PCR & qPCR Reagents' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Molecular Biology & Genomics > PCR & qPCR Reagents
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("VECTOR_SYSTEM") AS "VECTOR_SYSTEM",
    COUNT("HOST_EXPRESSION_SYSTEM") AS "HOST_EXPRESSION_SYSTEM",
    COUNT("TAGS_SELECTION") AS "TAGS_SELECTION"
    FROM SNOWFLAKE_LEARNING_DB."MOLECULAR_BIOLOGY_GENOMICS".CLONING_EXPRESSION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('VECTOR_SYSTEM', 'Vector / system', TRUE),
    ('HOST_EXPRESSION_SYSTEM', 'Host / expression system', FALSE),
    ('TAGS_SELECTION', 'Tags / selection', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'molecular_biology_reagents' AS ASSIGNED_L2_ID,
    'Molecular Biology & Genomics' AS ASSIGNED_L2_LABEL,
    'cloning_expression' AS ASSIGNED_L3_ID,
    'Cloning & Expression' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Molecular Biology & Genomics > Cloning & Expression
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("TARGET_GENE_REGION") AS "TARGET_GENE_REGION",
    COUNT("NUCLEASE_TYPE") AS "NUCLEASE_TYPE",
    COUNT("EDIT_TYPE") AS "EDIT_TYPE",
    COUNT("GRNA_FORMAT") AS "GRNA_FORMAT",
    COUNT("TARGET_SPECIES") AS "TARGET_SPECIES",
    COUNT("DELIVERY_FORMAT") AS "DELIVERY_FORMAT",
    COUNT("GRNA_MODIFICATIONS") AS "GRNA_MODIFICATIONS",
    COUNT("PURIFICATION") AS "PURIFICATION"
    FROM SNOWFLAKE_LEARNING_DB."MOLECULAR_BIOLOGY_GENOMICS".GENE_EDITING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('TARGET_GENE_REGION', 'Target Gene / Region', TRUE),
    ('NUCLEASE_TYPE', 'Nuclease type', TRUE),
    ('EDIT_TYPE', 'Edit type', TRUE),
    ('GRNA_FORMAT', 'gRNA format', TRUE),
    ('TARGET_SPECIES', 'Target Species', FALSE),
    ('DELIVERY_FORMAT', 'Delivery format', FALSE),
    ('GRNA_MODIFICATIONS', 'gRNA modifications', FALSE),
    ('PURIFICATION', 'Purification', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'molecular_biology_reagents' AS ASSIGNED_L2_ID,
    'Molecular Biology & Genomics' AS ASSIGNED_L2_LABEL,
    'gene_editing' AS ASSIGNED_L3_ID,
    'Gene Editing' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Molecular Biology & Genomics > Gene Editing
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("TARGET_GENE_REGION") AS "TARGET_GENE_REGION",
    COUNT("TYPE") AS "TYPE",
    COUNT("TARGET_SPECIES") AS "TARGET_SPECIES",
    COUNT("POOLING") AS "POOLING",
    COUNT("CHEMISTRY_MODIFICATIONS") AS "CHEMISTRY_MODIFICATIONS"
    FROM SNOWFLAKE_LEARNING_DB."MOLECULAR_BIOLOGY_GENOMICS".RNA_INTERFERENCE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('TARGET_GENE_REGION', 'Target Gene / Region', TRUE),
    ('TYPE', 'Type', TRUE),
    ('TARGET_SPECIES', 'Target Species', FALSE),
    ('POOLING', 'Pooling', FALSE),
    ('CHEMISTRY_MODIFICATIONS', 'Chemistry / modifications', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'molecular_biology_reagents' AS ASSIGNED_L2_ID,
    'Molecular Biology & Genomics' AS ASSIGNED_L2_LABEL,
    'rna_interference' AS ASSIGNED_L3_ID,
    'RNA Interference' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Molecular Biology & Genomics > RNA Interference
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("METHOD") AS "METHOD",
    COUNT("SAMPLE_TYPE") AS "SAMPLE_TYPE",
    COUNT("READ_CONFIGURATION") AS "READ_CONFIGURATION",
    COUNT("READ_LENGTH") AS "READ_LENGTH",
    COUNT("READ_DEPTH_READS_ORDERED") AS "READ_DEPTH_READS_ORDERED",
    COUNT("REGULATORY") AS "REGULATORY",
    COUNT("TURNAROUND") AS "TURNAROUND"
    FROM SNOWFLAKE_LEARNING_DB."MOLECULAR_BIOLOGY_GENOMICS".SEQUENCING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('METHOD', 'Method', TRUE),
    ('SAMPLE_TYPE', 'Sample type', TRUE),
    ('READ_CONFIGURATION', 'Read configuration', FALSE),
    ('READ_LENGTH', 'Read length', FALSE),
    ('READ_DEPTH_READS_ORDERED', 'Read depth / # reads ordered', FALSE),
    ('REGULATORY', 'Regulatory', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'molecular_biology_reagents' AS ASSIGNED_L2_ID,
    'Molecular Biology & Genomics' AS ASSIGNED_L2_LABEL,
    'sequencing' AS ASSIGNED_L3_ID,
    'Sequencing' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Molecular Biology & Genomics > Sequencing
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("LENGTH_NT") AS "LENGTH_NT",
    COUNT("SYNTHESIS_SCALE") AS "SYNTHESIS_SCALE",
    COUNT("PURIFICATION") AS "PURIFICATION",
    COUNT("TARGET_GENE_REGION") AS "TARGET_GENE_REGION",
    COUNT("TARGET_SPECIES") AS "TARGET_SPECIES",
    COUNT("MODIFICATIONS") AS "MODIFICATIONS",
    COUNT("QC") AS "QC",
    COUNT("FORMAT") AS "FORMAT"
    FROM SNOWFLAKE_LEARNING_DB."MOLECULAR_BIOLOGY_GENOMICS".OLIGOS_PRIMERS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('LENGTH_NT', 'Length (nt)', TRUE),
    ('SYNTHESIS_SCALE', 'Synthesis scale', TRUE),
    ('PURIFICATION', 'Purification', TRUE),
    ('TARGET_GENE_REGION', 'Target Gene / Region', FALSE),
    ('TARGET_SPECIES', 'Target Species', FALSE),
    ('MODIFICATIONS', 'Modifications', FALSE),
    ('QC', 'QC', FALSE),
    ('FORMAT', 'Format', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'molecular_biology_reagents' AS ASSIGNED_L2_ID,
    'Molecular Biology & Genomics' AS ASSIGNED_L2_LABEL,
    'oligos_primers' AS ASSIGNED_L3_ID,
    'Oligos & Primers' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Molecular Biology & Genomics > Oligos & Primers
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("LENGTH_BP") AS "LENGTH_BP",
    COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("SEQUENCE_VERIFICATION") AS "SEQUENCE_VERIFICATION",
    COUNT("VECTOR_CLONING") AS "VECTOR_CLONING",
    COUNT("CODON_OPTIMIZATION") AS "CODON_OPTIMIZATION",
    COUNT("FORMAT") AS "FORMAT",
    COUNT("TURNAROUND") AS "TURNAROUND"
    FROM SNOWFLAKE_LEARNING_DB."MOLECULAR_BIOLOGY_GENOMICS".GENE_SYNTHESIS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('LENGTH_BP', 'Length (bp)', TRUE),
    ('PRODUCT_TYPE', 'Product type', TRUE),
    ('SEQUENCE_VERIFICATION', 'Sequence verification', FALSE),
    ('VECTOR_CLONING', 'Vector / cloning', FALSE),
    ('CODON_OPTIMIZATION', 'Codon optimization', FALSE),
    ('FORMAT', 'Format', FALSE),
    ('TURNAROUND', 'Turnaround', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'molecular_biology_reagents' AS ASSIGNED_L2_ID,
    'Molecular Biology & Genomics' AS ASSIGNED_L2_LABEL,
    'gene_synthesis' AS ASSIGNED_L3_ID,
    'Gene Synthesis' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Molecular Biology & Genomics > Gene Synthesis
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PREP_SCALE") AS "PREP_SCALE",
    COUNT("VECTOR_TYPE_BACKBONE") AS "VECTOR_TYPE_BACKBONE",
    COUNT("GRADE") AS "GRADE",
    COUNT("VIRAL_SEROTYPE") AS "VIRAL_SEROTYPE",
    COUNT("TITER") AS "TITER",
    COUNT("SELECTION_MARKER") AS "SELECTION_MARKER",
    COUNT("YIELD") AS "YIELD"
    FROM SNOWFLAKE_LEARNING_DB."MOLECULAR_BIOLOGY_GENOMICS".PLASMIDS_VECTORS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PREP_SCALE', 'Prep scale', TRUE),
    ('VECTOR_TYPE_BACKBONE', 'Vector type / backbone', TRUE),
    ('GRADE', 'Grade', TRUE),
    ('VIRAL_SEROTYPE', 'Viral serotype', FALSE),
    ('TITER', 'Titer', FALSE),
    ('SELECTION_MARKER', 'Selection marker', FALSE),
    ('YIELD', 'Yield', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'molecular_biology_reagents' AS ASSIGNED_L2_ID,
    'Molecular Biology & Genomics' AS ASSIGNED_L2_LABEL,
    'plasmids_vectors' AS ASSIGNED_L3_ID,
    'Plasmids & Vectors' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Molecular Biology & Genomics > Plasmids & Vectors
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_PHARMACOVIGILANCE_DRUG_SAFETY_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PRODUCT_TYPES_COVERED") AS "PRODUCT_TYPES_COVERED",
    COUNT("REGIONS_SERVED") AS "REGIONS_SERVED",
    COUNT("MULTILINGUAL_INTAKE_AND_TRANSLATION") AS "MULTILINGUAL_INTAKE_AND_TRANSLATION",
    COUNT("AI_OR_AUTOMATION_IN_DELIVERY") AS "AI_OR_AUTOMATION_IN_DELIVERY",
    COUNT("24_7_AVAILABILITY") AS "24_7_AVAILABILITY",
    COUNT("LANGUAGES_SUPPORTED") AS "LANGUAGES_SUPPORTED",
    COUNT("EXPANDED_ACCESS_SUPPORTED") AS "EXPANDED_ACCESS_SUPPORTED",
    COUNT("PROCESSING_STEPS_INCLUDED") AS "PROCESSING_STEPS_INCLUDED",
    COUNT("CASE_SOURCES") AS "CASE_SOURCES",
    COUNT("INTAKE_CHANNELS") AS "INTAKE_CHANNELS",
    COUNT("VENDOR_HOSTED_SAFETY_DATABASE") AS "VENDOR_HOSTED_SAFETY_DATABASE",
    COUNT("SUBMISSION_FORMATS") AS "SUBMISSION_FORMATS",
    COUNT("REPORTING_DESTINATIONS") AS "REPORTING_DESTINATIONS",
    COUNT("ON_TIME_SUBMISSION_COMPLIANCE") AS "ON_TIME_SUBMISSION_COMPLIANCE",
    COUNT("SAFETY_DATABASE") AS "SAFETY_DATABASE",
    COUNT("CLINICAL_TRIAL_CASE_TYPES") AS "CLINICAL_TRIAL_CASE_TYPES",
    COUNT("CODING_DICTIONARIES") AS "CODING_DICTIONARIES"
    FROM SNOWFLAKE_LEARNING_DB."PHARMACOVIGILANCE_DRUG_SAFETY".CASE_PROCESSING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PRODUCT_TYPES_COVERED', 'Product types covered', FALSE),
    ('REGIONS_SERVED', 'Regions served', FALSE),
    ('MULTILINGUAL_INTAKE_AND_TRANSLATION', 'Multilingual intake and translation', FALSE),
    ('AI_OR_AUTOMATION_IN_DELIVERY', 'AI or automation in delivery', FALSE),
    ('24_7_AVAILABILITY', '24/7 availability', FALSE),
    ('LANGUAGES_SUPPORTED', 'Languages supported', FALSE),
    ('EXPANDED_ACCESS_SUPPORTED', 'Expanded access supported', FALSE),
    ('PROCESSING_STEPS_INCLUDED', 'Processing steps included', FALSE),
    ('CASE_SOURCES', 'Case sources', FALSE),
    ('INTAKE_CHANNELS', 'Intake channels', FALSE),
    ('VENDOR_HOSTED_SAFETY_DATABASE', 'Vendor-hosted safety database', FALSE),
    ('SUBMISSION_FORMATS', 'Submission formats', FALSE),
    ('REPORTING_DESTINATIONS', 'Reporting destinations', FALSE),
    ('ON_TIME_SUBMISSION_COMPLIANCE', 'On-time submission compliance', FALSE),
    ('SAFETY_DATABASE', 'Safety database', FALSE),
    ('CLINICAL_TRIAL_CASE_TYPES', 'Clinical trial case types', FALSE),
    ('CODING_DICTIONARIES', 'Coding dictionaries', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'pharmacovigilance_drug_safety' AS ASSIGNED_L2_ID,
    'Pharmacovigilance & Drug Safety' AS ASSIGNED_L2_LABEL,
    'case_processing' AS ASSIGNED_L3_ID,
    'Case Processing' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Pharmacovigilance & Drug Safety > Case Processing
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PRODUCT_TYPES_COVERED") AS "PRODUCT_TYPES_COVERED",
    COUNT("REGIONS_SERVED") AS "REGIONS_SERVED",
    COUNT("MULTILINGUAL_INTAKE_AND_TRANSLATION") AS "MULTILINGUAL_INTAKE_AND_TRANSLATION",
    COUNT("AI_OR_AUTOMATION_IN_DELIVERY") AS "AI_OR_AUTOMATION_IN_DELIVERY",
    COUNT("24_7_AVAILABILITY") AS "24_7_AVAILABILITY",
    COUNT("LANGUAGES_SUPPORTED") AS "LANGUAGES_SUPPORTED",
    COUNT("EXPANDED_ACCESS_SUPPORTED") AS "EXPANDED_ACCESS_SUPPORTED",
    COUNT("SIGNAL_MANAGEMENT_STAGES") AS "SIGNAL_MANAGEMENT_STAGES",
    COUNT("DETECTION_METHOD") AS "DETECTION_METHOD",
    COUNT("SIGNAL_TOOLS") AS "SIGNAL_TOOLS",
    COUNT("DATA_SOURCES_SEARCHED") AS "DATA_SOURCES_SEARCHED",
    COUNT("SIGNAL_DELIVERABLES") AS "SIGNAL_DELIVERABLES",
    COUNT("REVIEW_FREQUENCY") AS "REVIEW_FREQUENCY",
    COUNT("SIGNAL_SCOPE") AS "SIGNAL_SCOPE",
    COUNT("BENEFIT_RISK_ASSESSMENT_INCLUDED") AS "BENEFIT_RISK_ASSESSMENT_INCLUDED"
    FROM SNOWFLAKE_LEARNING_DB."PHARMACOVIGILANCE_DRUG_SAFETY".SIGNAL_DETECTION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PRODUCT_TYPES_COVERED', 'Product types covered', FALSE),
    ('REGIONS_SERVED', 'Regions served', FALSE),
    ('MULTILINGUAL_INTAKE_AND_TRANSLATION', 'Multilingual intake and translation', FALSE),
    ('AI_OR_AUTOMATION_IN_DELIVERY', 'AI or automation in delivery', FALSE),
    ('24_7_AVAILABILITY', '24/7 availability', FALSE),
    ('LANGUAGES_SUPPORTED', 'Languages supported', FALSE),
    ('EXPANDED_ACCESS_SUPPORTED', 'Expanded access supported', FALSE),
    ('SIGNAL_MANAGEMENT_STAGES', 'Signal management stages', FALSE),
    ('DETECTION_METHOD', 'Detection method', FALSE),
    ('SIGNAL_TOOLS', 'Signal tools', FALSE),
    ('DATA_SOURCES_SEARCHED', 'Data sources searched', FALSE),
    ('SIGNAL_DELIVERABLES', 'Signal deliverables', FALSE),
    ('REVIEW_FREQUENCY', 'Review frequency', FALSE),
    ('SIGNAL_SCOPE', 'Signal scope', FALSE),
    ('BENEFIT_RISK_ASSESSMENT_INCLUDED', 'Benefit-risk assessment included', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'pharmacovigilance_drug_safety' AS ASSIGNED_L2_ID,
    'Pharmacovigilance & Drug Safety' AS ASSIGNED_L2_LABEL,
    'signal_detection' AS ASSIGNED_L3_ID,
    'Signal Detection' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Pharmacovigilance & Drug Safety > Signal Detection
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PRODUCT_TYPES_COVERED") AS "PRODUCT_TYPES_COVERED",
    COUNT("REGIONS_SERVED") AS "REGIONS_SERVED",
    COUNT("MULTILINGUAL_INTAKE_AND_TRANSLATION") AS "MULTILINGUAL_INTAKE_AND_TRANSLATION",
    COUNT("AI_OR_AUTOMATION_IN_DELIVERY") AS "AI_OR_AUTOMATION_IN_DELIVERY",
    COUNT("24_7_AVAILABILITY") AS "24_7_AVAILABILITY",
    COUNT("LANGUAGES_SUPPORTED") AS "LANGUAGES_SUPPORTED",
    COUNT("EXPANDED_ACCESS_SUPPORTED") AS "EXPANDED_ACCESS_SUPPORTED",
    COUNT("REPORT_TYPES") AS "REPORT_TYPES",
    COUNT("AGGREGATE_REPORT_SCOPE") AS "AGGREGATE_REPORT_SCOPE",
    COUNT("REFERENCE_SAFETY_DOCUMENTS") AS "REFERENCE_SAFETY_DOCUMENTS"
    FROM SNOWFLAKE_LEARNING_DB."PHARMACOVIGILANCE_DRUG_SAFETY".AGGREGATE_REPORTS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PRODUCT_TYPES_COVERED', 'Product types covered', FALSE),
    ('REGIONS_SERVED', 'Regions served', FALSE),
    ('MULTILINGUAL_INTAKE_AND_TRANSLATION', 'Multilingual intake and translation', FALSE),
    ('AI_OR_AUTOMATION_IN_DELIVERY', 'AI or automation in delivery', FALSE),
    ('24_7_AVAILABILITY', '24/7 availability', FALSE),
    ('LANGUAGES_SUPPORTED', 'Languages supported', FALSE),
    ('EXPANDED_ACCESS_SUPPORTED', 'Expanded access supported', FALSE),
    ('REPORT_TYPES', 'Report types', TRUE),
    ('AGGREGATE_REPORT_SCOPE', 'Aggregate report scope', FALSE),
    ('REFERENCE_SAFETY_DOCUMENTS', 'Reference safety documents', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'pharmacovigilance_drug_safety' AS ASSIGNED_L2_ID,
    'Pharmacovigilance & Drug Safety' AS ASSIGNED_L2_LABEL,
    'aggregate_reports' AS ASSIGNED_L3_ID,
    'Aggregate Reports' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Pharmacovigilance & Drug Safety > Aggregate Reports
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PRODUCT_TYPES_COVERED") AS "PRODUCT_TYPES_COVERED",
    COUNT("REGIONS_SERVED") AS "REGIONS_SERVED",
    COUNT("MULTILINGUAL_INTAKE_AND_TRANSLATION") AS "MULTILINGUAL_INTAKE_AND_TRANSLATION",
    COUNT("AI_OR_AUTOMATION_IN_DELIVERY") AS "AI_OR_AUTOMATION_IN_DELIVERY",
    COUNT("24_7_AVAILABILITY") AS "24_7_AVAILABILITY",
    COUNT("LANGUAGES_SUPPORTED") AS "LANGUAGES_SUPPORTED",
    COUNT("EXPANDED_ACCESS_SUPPORTED") AS "EXPANDED_ACCESS_SUPPORTED",
    COUNT("SEARCH_SCOPE") AS "SEARCH_SCOPE",
    COUNT("LITERATURE_DATABASES") AS "LITERATURE_DATABASES",
    COUNT("EMA_MLM_HANDLING") AS "EMA_MLM_HANDLING",
    COUNT("ARTICLE_TRANSLATION") AS "ARTICLE_TRANSLATION",
    COUNT("FULL_TEXT_ARTICLE_RETRIEVAL") AS "FULL_TEXT_ARTICLE_RETRIEVAL",
    COUNT("SEARCH_STRATEGY_DEVELOPMENT") AS "SEARCH_STRATEGY_DEVELOPMENT",
    COUNT("LITERATURE_TOOL") AS "LITERATURE_TOOL"
    FROM SNOWFLAKE_LEARNING_DB."PHARMACOVIGILANCE_DRUG_SAFETY".LITERATURE_MONITORING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PRODUCT_TYPES_COVERED', 'Product types covered', FALSE),
    ('REGIONS_SERVED', 'Regions served', FALSE),
    ('MULTILINGUAL_INTAKE_AND_TRANSLATION', 'Multilingual intake and translation', FALSE),
    ('AI_OR_AUTOMATION_IN_DELIVERY', 'AI or automation in delivery', FALSE),
    ('24_7_AVAILABILITY', '24/7 availability', FALSE),
    ('LANGUAGES_SUPPORTED', 'Languages supported', FALSE),
    ('EXPANDED_ACCESS_SUPPORTED', 'Expanded access supported', FALSE),
    ('SEARCH_SCOPE', 'Search scope', FALSE),
    ('LITERATURE_DATABASES', 'Literature databases', FALSE),
    ('EMA_MLM_HANDLING', 'EMA MLM handling', FALSE),
    ('ARTICLE_TRANSLATION', 'Article translation', FALSE),
    ('FULL_TEXT_ARTICLE_RETRIEVAL', 'Full-text article retrieval', FALSE),
    ('SEARCH_STRATEGY_DEVELOPMENT', 'Search strategy development', FALSE),
    ('LITERATURE_TOOL', 'Literature tool', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'pharmacovigilance_drug_safety' AS ASSIGNED_L2_ID,
    'Pharmacovigilance & Drug Safety' AS ASSIGNED_L2_LABEL,
    'literature_monitoring' AS ASSIGNED_L3_ID,
    'Literature Monitoring' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Pharmacovigilance & Drug Safety > Literature Monitoring
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PRODUCT_TYPES_COVERED") AS "PRODUCT_TYPES_COVERED",
    COUNT("REGIONS_SERVED") AS "REGIONS_SERVED",
    COUNT("MULTILINGUAL_INTAKE_AND_TRANSLATION") AS "MULTILINGUAL_INTAKE_AND_TRANSLATION",
    COUNT("AI_OR_AUTOMATION_IN_DELIVERY") AS "AI_OR_AUTOMATION_IN_DELIVERY",
    COUNT("24_7_AVAILABILITY") AS "24_7_AVAILABILITY",
    COUNT("LANGUAGES_SUPPORTED") AS "LANGUAGES_SUPPORTED",
    COUNT("EXPANDED_ACCESS_SUPPORTED") AS "EXPANDED_ACCESS_SUPPORTED",
    COUNT("PLAN_TYPES") AS "PLAN_TYPES",
    COUNT("RISK_MANAGEMENT_SCOPE") AS "RISK_MANAGEMENT_SCOPE",
    COUNT("REMS_PROGRAM_OPERATIONS") AS "REMS_PROGRAM_OPERATIONS",
    COUNT("PROGRAM_LANGUAGES") AS "PROGRAM_LANGUAGES"
    FROM SNOWFLAKE_LEARNING_DB."PHARMACOVIGILANCE_DRUG_SAFETY".RISK_MANAGEMENT_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PRODUCT_TYPES_COVERED', 'Product types covered', FALSE),
    ('REGIONS_SERVED', 'Regions served', FALSE),
    ('MULTILINGUAL_INTAKE_AND_TRANSLATION', 'Multilingual intake and translation', FALSE),
    ('AI_OR_AUTOMATION_IN_DELIVERY', 'AI or automation in delivery', FALSE),
    ('24_7_AVAILABILITY', '24/7 availability', FALSE),
    ('LANGUAGES_SUPPORTED', 'Languages supported', FALSE),
    ('EXPANDED_ACCESS_SUPPORTED', 'Expanded access supported', FALSE),
    ('PLAN_TYPES', 'Plan types', TRUE),
    ('RISK_MANAGEMENT_SCOPE', 'Risk management scope', FALSE),
    ('REMS_PROGRAM_OPERATIONS', 'REMS program operations', FALSE),
    ('PROGRAM_LANGUAGES', 'Program languages', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'pharmacovigilance_drug_safety' AS ASSIGNED_L2_ID,
    'Pharmacovigilance & Drug Safety' AS ASSIGNED_L2_LABEL,
    'risk_management' AS ASSIGNED_L3_ID,
    'Risk Management' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Pharmacovigilance & Drug Safety > Risk Management
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("PRODUCT_TYPES_COVERED") AS "PRODUCT_TYPES_COVERED",
    COUNT("REGIONS_SERVED") AS "REGIONS_SERVED",
    COUNT("MULTILINGUAL_INTAKE_AND_TRANSLATION") AS "MULTILINGUAL_INTAKE_AND_TRANSLATION",
    COUNT("AI_OR_AUTOMATION_IN_DELIVERY") AS "AI_OR_AUTOMATION_IN_DELIVERY",
    COUNT("24_7_AVAILABILITY") AS "24_7_AVAILABILITY",
    COUNT("LANGUAGES_SUPPORTED") AS "LANGUAGES_SUPPORTED",
    COUNT("EXPANDED_ACCESS_SUPPORTED") AS "EXPANDED_ACCESS_SUPPORTED",
    COUNT("SERVICE_TYPE") AS "SERVICE_TYPE"
    FROM SNOWFLAKE_LEARNING_DB."PHARMACOVIGILANCE_DRUG_SAFETY".QPPV_PV_SYSTEM_SERVICES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('PRODUCT_TYPES_COVERED', 'Product types covered', FALSE),
    ('REGIONS_SERVED', 'Regions served', FALSE),
    ('MULTILINGUAL_INTAKE_AND_TRANSLATION', 'Multilingual intake and translation', FALSE),
    ('AI_OR_AUTOMATION_IN_DELIVERY', 'AI or automation in delivery', FALSE),
    ('24_7_AVAILABILITY', '24/7 availability', FALSE),
    ('LANGUAGES_SUPPORTED', 'Languages supported', FALSE),
    ('EXPANDED_ACCESS_SUPPORTED', 'Expanded access supported', FALSE),
    ('SERVICE_TYPE', 'Service type', TRUE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'pharmacovigilance_drug_safety' AS ASSIGNED_L2_ID,
    'Pharmacovigilance & Drug Safety' AS ASSIGNED_L2_LABEL,
    'qppv_pv_system_services' AS ASSIGNED_L3_ID,
    'QPPV & PV System Services' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Pharmacovigilance & Drug Safety > QPPV & PV System Services
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_PHASE_IV_POST_MARKETING_STUDIES_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREA") AS "THERAPEUTIC_AREA",
    COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("GEOGRAPHIC_COVERAGE") AS "GEOGRAPHIC_COVERAGE",
    COUNT("DATA_COLLECTION_APPROACH") AS "DATA_COLLECTION_APPROACH",
    COUNT("DATA_SOURCES") AS "DATA_SOURCES",
    COUNT("REGULATORY_PURPOSE") AS "REGULATORY_PURPOSE",
    COUNT("TARGET_REGULATOR") AS "TARGET_REGULATOR",
    COUNT("ENDPOINTS_CAPTURED") AS "ENDPOINTS_CAPTURED",
    COUNT("PATIENT_REPORTED_OUTCOMES_COLLECTED") AS "PATIENT_REPORTED_OUTCOMES_COLLECTED",
    COUNT("SCOPE_OF_SERVICE") AS "SCOPE_OF_SERVICE",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("NUMBER_OF_COUNTRIES") AS "NUMBER_OF_COUNTRIES",
    COUNT("DECENTRALIZED_OR_REMOTE_STUDY_MODEL") AS "DECENTRALIZED_OR_REMOTE_STUDY_MODEL",
    COUNT("PATIENT_POPULATION") AS "PATIENT_POPULATION",
    COUNT("DIRECT_TO_PATIENT_DATA_COLLECTION") AS "DIRECT_TO_PATIENT_DATA_COLLECTION",
    COUNT("TARGET_SAMPLE_SIZE") AS "TARGET_SAMPLE_SIZE",
    COUNT("OBSERVATIONAL_STUDY_TYPE") AS "OBSERVATIONAL_STUDY_TYPE",
    COUNT("DATA_LINKAGE_ACROSS_SOURCES") AS "DATA_LINKAGE_ACROSS_SOURCES",
    COUNT("COMPARATOR_APPROACH") AS "COMPARATOR_APPROACH",
    COUNT("EPIDEMIOLOGIC_DESIGN") AS "EPIDEMIOLOGIC_DESIGN",
    COUNT("NUMBER_OF_DATA_SOURCES_LINKED") AS "NUMBER_OF_DATA_SOURCES_LINKED",
    COUNT("CAUSAL_INFERENCE_METHODS") AS "CAUSAL_INFERENCE_METHODS",
    COUNT("SITE_TYPE") AS "SITE_TYPE"
    FROM SNOWFLAKE_LEARNING_DB."PHASE_IV_POST_MARKETING_STUDIES".OBSERVATIONAL_STUDIES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREA', 'Therapeutic area', FALSE),
    ('PRODUCT_TYPE', 'Product type', FALSE),
    ('GEOGRAPHIC_COVERAGE', 'Geographic coverage', FALSE),
    ('DATA_COLLECTION_APPROACH', 'Data collection approach', FALSE),
    ('DATA_SOURCES', 'Data sources', FALSE),
    ('REGULATORY_PURPOSE', 'Regulatory purpose', FALSE),
    ('TARGET_REGULATOR', 'Target regulator', FALSE),
    ('ENDPOINTS_CAPTURED', 'Endpoints captured', FALSE),
    ('PATIENT_REPORTED_OUTCOMES_COLLECTED', 'Patient-reported outcomes collected', FALSE),
    ('SCOPE_OF_SERVICE', 'Scope of service', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('NUMBER_OF_COUNTRIES', 'Number of countries', FALSE),
    ('DECENTRALIZED_OR_REMOTE_STUDY_MODEL', 'Decentralized or remote study model', FALSE),
    ('PATIENT_POPULATION', 'Patient population', FALSE),
    ('DIRECT_TO_PATIENT_DATA_COLLECTION', 'Direct-to-patient data collection', FALSE),
    ('TARGET_SAMPLE_SIZE', 'Target sample size', FALSE),
    ('OBSERVATIONAL_STUDY_TYPE', 'Observational study type', TRUE),
    ('DATA_LINKAGE_ACROSS_SOURCES', 'Data linkage across sources', FALSE),
    ('COMPARATOR_APPROACH', 'Comparator approach', FALSE),
    ('EPIDEMIOLOGIC_DESIGN', 'Epidemiologic design', FALSE),
    ('NUMBER_OF_DATA_SOURCES_LINKED', 'Number of data sources linked', FALSE),
    ('CAUSAL_INFERENCE_METHODS', 'Causal inference methods', FALSE),
    ('SITE_TYPE', 'Site type', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'phase_iv_post_marketing_studies' AS ASSIGNED_L2_ID,
    'Phase IV / Post-Marketing Studies' AS ASSIGNED_L2_LABEL,
    'observational_studies' AS ASSIGNED_L3_ID,
    'Observational Studies' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Phase IV / Post-Marketing Studies > Observational Studies
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREA") AS "THERAPEUTIC_AREA",
    COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("GEOGRAPHIC_COVERAGE") AS "GEOGRAPHIC_COVERAGE",
    COUNT("DATA_COLLECTION_APPROACH") AS "DATA_COLLECTION_APPROACH",
    COUNT("DATA_SOURCES") AS "DATA_SOURCES",
    COUNT("REGULATORY_PURPOSE") AS "REGULATORY_PURPOSE",
    COUNT("TARGET_REGULATOR") AS "TARGET_REGULATOR",
    COUNT("ENDPOINTS_CAPTURED") AS "ENDPOINTS_CAPTURED",
    COUNT("PATIENT_REPORTED_OUTCOMES_COLLECTED") AS "PATIENT_REPORTED_OUTCOMES_COLLECTED",
    COUNT("SCOPE_OF_SERVICE") AS "SCOPE_OF_SERVICE",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("NUMBER_OF_COUNTRIES") AS "NUMBER_OF_COUNTRIES",
    COUNT("DECENTRALIZED_OR_REMOTE_STUDY_MODEL") AS "DECENTRALIZED_OR_REMOTE_STUDY_MODEL",
    COUNT("PATIENT_POPULATION") AS "PATIENT_POPULATION",
    COUNT("DIRECT_TO_PATIENT_DATA_COLLECTION") AS "DIRECT_TO_PATIENT_DATA_COLLECTION",
    COUNT("TARGET_SAMPLE_SIZE") AS "TARGET_SAMPLE_SIZE",
    COUNT("REGISTRY_TYPE") AS "REGISTRY_TYPE",
    COUNT("SPONSOR_MODEL") AS "SPONSOR_MODEL",
    COUNT("REGISTRY_DURATION") AS "REGISTRY_DURATION",
    COUNT("INFANT_FOLLOW_UP_PERIOD") AS "INFANT_FOLLOW_UP_PERIOD",
    COUNT("EXPOSED_VERSUS_UNEXPOSED_COMPARISON") AS "EXPOSED_VERSUS_UNEXPOSED_COMPARISON",
    COUNT("ENROLLMENT_CHANNEL") AS "ENROLLMENT_CHANNEL",
    COUNT("REGISTRY_AWARENESS_CAMPAIGN") AS "REGISTRY_AWARENESS_CAMPAIGN",
    COUNT("GLOBAL_COORDINATING_CENTER") AS "GLOBAL_COORDINATING_CENTER",
    COUNT("DATA_REPORTERS") AS "DATA_REPORTERS",
    COUNT("PREGNANCY_OUTCOMES_CAPTURED") AS "PREGNANCY_OUTCOMES_CAPTURED",
    COUNT("LACTATION_STUDY_TYPE") AS "LACTATION_STUDY_TYPE",
    COUNT("BIOSPECIMEN_COLLECTION") AS "BIOSPECIMEN_COLLECTION"
    FROM SNOWFLAKE_LEARNING_DB."PHASE_IV_POST_MARKETING_STUDIES".PATIENT_REGISTRIES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREA', 'Therapeutic area', FALSE),
    ('PRODUCT_TYPE', 'Product type', FALSE),
    ('GEOGRAPHIC_COVERAGE', 'Geographic coverage', FALSE),
    ('DATA_COLLECTION_APPROACH', 'Data collection approach', FALSE),
    ('DATA_SOURCES', 'Data sources', FALSE),
    ('REGULATORY_PURPOSE', 'Regulatory purpose', FALSE),
    ('TARGET_REGULATOR', 'Target regulator', FALSE),
    ('ENDPOINTS_CAPTURED', 'Endpoints captured', FALSE),
    ('PATIENT_REPORTED_OUTCOMES_COLLECTED', 'Patient-reported outcomes collected', FALSE),
    ('SCOPE_OF_SERVICE', 'Scope of service', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('NUMBER_OF_COUNTRIES', 'Number of countries', FALSE),
    ('DECENTRALIZED_OR_REMOTE_STUDY_MODEL', 'Decentralized or remote study model', FALSE),
    ('PATIENT_POPULATION', 'Patient population', FALSE),
    ('DIRECT_TO_PATIENT_DATA_COLLECTION', 'Direct-to-patient data collection', FALSE),
    ('TARGET_SAMPLE_SIZE', 'Target sample size', FALSE),
    ('REGISTRY_TYPE', 'Registry type', TRUE),
    ('SPONSOR_MODEL', 'Sponsor model', FALSE),
    ('REGISTRY_DURATION', 'Registry duration', FALSE),
    ('INFANT_FOLLOW_UP_PERIOD', 'Infant follow-up period', FALSE),
    ('EXPOSED_VERSUS_UNEXPOSED_COMPARISON', 'Exposed versus unexposed comparison', FALSE),
    ('ENROLLMENT_CHANNEL', 'Enrollment channel', FALSE),
    ('REGISTRY_AWARENESS_CAMPAIGN', 'Registry awareness campaign', FALSE),
    ('GLOBAL_COORDINATING_CENTER', 'Global coordinating center', FALSE),
    ('DATA_REPORTERS', 'Data reporters', FALSE),
    ('PREGNANCY_OUTCOMES_CAPTURED', 'Pregnancy outcomes captured', FALSE),
    ('LACTATION_STUDY_TYPE', 'Lactation study type', FALSE),
    ('BIOSPECIMEN_COLLECTION', 'Biospecimen collection', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'phase_iv_post_marketing_studies' AS ASSIGNED_L2_ID,
    'Phase IV / Post-Marketing Studies' AS ASSIGNED_L2_LABEL,
    'patient_registries' AS ASSIGNED_L3_ID,
    'Patient Registries' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Phase IV / Post-Marketing Studies > Patient Registries
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREA") AS "THERAPEUTIC_AREA",
    COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("GEOGRAPHIC_COVERAGE") AS "GEOGRAPHIC_COVERAGE",
    COUNT("DATA_COLLECTION_APPROACH") AS "DATA_COLLECTION_APPROACH",
    COUNT("DATA_SOURCES") AS "DATA_SOURCES",
    COUNT("REGULATORY_PURPOSE") AS "REGULATORY_PURPOSE",
    COUNT("TARGET_REGULATOR") AS "TARGET_REGULATOR",
    COUNT("ENDPOINTS_CAPTURED") AS "ENDPOINTS_CAPTURED",
    COUNT("PATIENT_REPORTED_OUTCOMES_COLLECTED") AS "PATIENT_REPORTED_OUTCOMES_COLLECTED",
    COUNT("SCOPE_OF_SERVICE") AS "SCOPE_OF_SERVICE",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("NUMBER_OF_COUNTRIES") AS "NUMBER_OF_COUNTRIES",
    COUNT("DECENTRALIZED_OR_REMOTE_STUDY_MODEL") AS "DECENTRALIZED_OR_REMOTE_STUDY_MODEL",
    COUNT("PATIENT_POPULATION") AS "PATIENT_POPULATION",
    COUNT("DIRECT_TO_PATIENT_DATA_COLLECTION") AS "DIRECT_TO_PATIENT_DATA_COLLECTION",
    COUNT("TARGET_SAMPLE_SIZE") AS "TARGET_SAMPLE_SIZE",
    COUNT("LATE_PHASE_TRIAL_TYPE") AS "LATE_PHASE_TRIAL_TYPE",
    COUNT("PRAGMATIC_DESIGN") AS "PRAGMATIC_DESIGN",
    COUNT("ACCESS_PROGRAM_TYPE") AS "ACCESS_PROGRAM_TYPE",
    COUNT("LONG_TERM_FOLLOW_UP_FOR_CELL_OR_GENE_THERAPY") AS "LONG_TERM_FOLLOW_UP_FOR_CELL_OR_GENE_THERAPY",
    COUNT("FOLLOW_UP_DURATION") AS "FOLLOW_UP_DURATION",
    COUNT("RANDOMIZED") AS "RANDOMIZED",
    COUNT("VECTOR_TYPE_FOLLOWED") AS "VECTOR_TYPE_FOLLOWED",
    COUNT("SEPARATE_LTFU_PROTOCOL") AS "SEPARATE_LTFU_PROTOCOL",
    COUNT("NON_DRUG_INTERVENTION_STUDIED") AS "NON_DRUG_INTERVENTION_STUDIED",
    COUNT("DIGITAL_ENDPOINTS") AS "DIGITAL_ENDPOINTS"
    FROM SNOWFLAKE_LEARNING_DB."PHASE_IV_POST_MARKETING_STUDIES".LATE_PHASE_TRIALS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREA', 'Therapeutic area', FALSE),
    ('PRODUCT_TYPE', 'Product type', FALSE),
    ('GEOGRAPHIC_COVERAGE', 'Geographic coverage', FALSE),
    ('DATA_COLLECTION_APPROACH', 'Data collection approach', FALSE),
    ('DATA_SOURCES', 'Data sources', FALSE),
    ('REGULATORY_PURPOSE', 'Regulatory purpose', FALSE),
    ('TARGET_REGULATOR', 'Target regulator', FALSE),
    ('ENDPOINTS_CAPTURED', 'Endpoints captured', FALSE),
    ('PATIENT_REPORTED_OUTCOMES_COLLECTED', 'Patient-reported outcomes collected', FALSE),
    ('SCOPE_OF_SERVICE', 'Scope of service', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('NUMBER_OF_COUNTRIES', 'Number of countries', FALSE),
    ('DECENTRALIZED_OR_REMOTE_STUDY_MODEL', 'Decentralized or remote study model', FALSE),
    ('PATIENT_POPULATION', 'Patient population', FALSE),
    ('DIRECT_TO_PATIENT_DATA_COLLECTION', 'Direct-to-patient data collection', FALSE),
    ('TARGET_SAMPLE_SIZE', 'Target sample size', FALSE),
    ('LATE_PHASE_TRIAL_TYPE', 'Late-phase trial type', TRUE),
    ('PRAGMATIC_DESIGN', 'Pragmatic design', FALSE),
    ('ACCESS_PROGRAM_TYPE', 'Access program type', FALSE),
    ('LONG_TERM_FOLLOW_UP_FOR_CELL_OR_GENE_THERAPY', 'Long-term follow-up for cell or gene therapy', FALSE),
    ('FOLLOW_UP_DURATION', 'Follow-up duration', FALSE),
    ('RANDOMIZED', 'Randomized', FALSE),
    ('VECTOR_TYPE_FOLLOWED', 'Vector type followed', FALSE),
    ('SEPARATE_LTFU_PROTOCOL', 'Separate LTFU protocol', FALSE),
    ('NON_DRUG_INTERVENTION_STUDIED', 'Non-drug intervention studied', FALSE),
    ('DIGITAL_ENDPOINTS', 'Digital endpoints', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'phase_iv_post_marketing_studies' AS ASSIGNED_L2_ID,
    'Phase IV / Post-Marketing Studies' AS ASSIGNED_L2_LABEL,
    'late_phase_trials' AS ASSIGNED_L3_ID,
    'Late-Phase Trials' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Phase IV / Post-Marketing Studies > Late-Phase Trials
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("THERAPEUTIC_AREA") AS "THERAPEUTIC_AREA",
    COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("GEOGRAPHIC_COVERAGE") AS "GEOGRAPHIC_COVERAGE",
    COUNT("DATA_COLLECTION_APPROACH") AS "DATA_COLLECTION_APPROACH",
    COUNT("DATA_SOURCES") AS "DATA_SOURCES",
    COUNT("REGULATORY_PURPOSE") AS "REGULATORY_PURPOSE",
    COUNT("TARGET_REGULATOR") AS "TARGET_REGULATOR",
    COUNT("ENDPOINTS_CAPTURED") AS "ENDPOINTS_CAPTURED",
    COUNT("PATIENT_REPORTED_OUTCOMES_COLLECTED") AS "PATIENT_REPORTED_OUTCOMES_COLLECTED",
    COUNT("SCOPE_OF_SERVICE") AS "SCOPE_OF_SERVICE",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("NUMBER_OF_COUNTRIES") AS "NUMBER_OF_COUNTRIES",
    COUNT("DECENTRALIZED_OR_REMOTE_STUDY_MODEL") AS "DECENTRALIZED_OR_REMOTE_STUDY_MODEL",
    COUNT("PATIENT_POPULATION") AS "PATIENT_POPULATION",
    COUNT("DIRECT_TO_PATIENT_DATA_COLLECTION") AS "DIRECT_TO_PATIENT_DATA_COLLECTION",
    COUNT("TARGET_SAMPLE_SIZE") AS "TARGET_SAMPLE_SIZE",
    COUNT("SAFETY_STUDY_TYPE") AS "SAFETY_STUDY_TYPE",
    COUNT("RISK_PROGRAM_SUPPORTED") AS "RISK_PROGRAM_SUPPORTED",
    COUNT("SAFETY_OUTCOMES_OF_INTEREST") AS "SAFETY_OUTCOMES_OF_INTEREST",
    COUNT("INTERVENTIONAL_STATUS") AS "INTERVENTIONAL_STATUS",
    COUNT("FOLLOWS_GVP_MODULE_VIII") AS "FOLLOWS_GVP_MODULE_VIII",
    COUNT("FOLLOWS_ENCEPP_METHODS_GUIDE") AS "FOLLOWS_ENCEPP_METHODS_GUIDE",
    COUNT("SAFETY_DESIGN") AS "SAFETY_DESIGN",
    COUNT("SIGNAL_DETECTION_SUPPORT") AS "SIGNAL_DETECTION_SUPPORT"
    FROM SNOWFLAKE_LEARNING_DB."PHASE_IV_POST_MARKETING_STUDIES".POST_AUTH_SAFETY_STUDIES_PASS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('THERAPEUTIC_AREA', 'Therapeutic area', FALSE),
    ('PRODUCT_TYPE', 'Product type', FALSE),
    ('GEOGRAPHIC_COVERAGE', 'Geographic coverage', FALSE),
    ('DATA_COLLECTION_APPROACH', 'Data collection approach', FALSE),
    ('DATA_SOURCES', 'Data sources', FALSE),
    ('REGULATORY_PURPOSE', 'Regulatory purpose', FALSE),
    ('TARGET_REGULATOR', 'Target regulator', FALSE),
    ('ENDPOINTS_CAPTURED', 'Endpoints captured', FALSE),
    ('PATIENT_REPORTED_OUTCOMES_COLLECTED', 'Patient-reported outcomes collected', FALSE),
    ('SCOPE_OF_SERVICE', 'Scope of service', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('NUMBER_OF_COUNTRIES', 'Number of countries', FALSE),
    ('DECENTRALIZED_OR_REMOTE_STUDY_MODEL', 'Decentralized or remote study model', FALSE),
    ('PATIENT_POPULATION', 'Patient population', FALSE),
    ('DIRECT_TO_PATIENT_DATA_COLLECTION', 'Direct-to-patient data collection', FALSE),
    ('TARGET_SAMPLE_SIZE', 'Target sample size', FALSE),
    ('SAFETY_STUDY_TYPE', 'Safety study type', TRUE),
    ('RISK_PROGRAM_SUPPORTED', 'Risk program supported', FALSE),
    ('SAFETY_OUTCOMES_OF_INTEREST', 'Safety outcomes of interest', FALSE),
    ('INTERVENTIONAL_STATUS', 'Interventional status', FALSE),
    ('FOLLOWS_GVP_MODULE_VIII', 'Follows GVP Module VIII', FALSE),
    ('FOLLOWS_ENCEPP_METHODS_GUIDE', 'Follows ENCePP methods guide', FALSE),
    ('SAFETY_DESIGN', 'Safety design', FALSE),
    ('SIGNAL_DETECTION_SUPPORT', 'Signal detection support', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'phase_iv_post_marketing_studies' AS ASSIGNED_L2_ID,
    'Phase IV / Post-Marketing Studies' AS ASSIGNED_L2_LABEL,
    'post_auth_safety_studies_pass' AS ASSIGNED_L3_ID,
    'Post-Auth Safety Studies (PASS)' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Phase IV / Post-Marketing Studies > Post-Auth Safety Studies (PASS)
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_PROTEINS_PEPTIDES_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EXPRESSION_SYSTEM") AS "EXPRESSION_SYSTEM",
    COUNT("FORM") AS "FORM",
    COUNT("PURITY") AS "PURITY",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("SEQUENCE_REGION") AS "SEQUENCE_REGION",
    COUNT("TAG") AS "TAG",
    COUNT("BIOACTIVITY") AS "BIOACTIVITY",
    COUNT("GRADE") AS "GRADE",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("STORAGE_RECONSTITUTION") AS "STORAGE_RECONSTITUTION",
    COUNT("ENDOTOXIN_LEVEL") AS "ENDOTOXIN_LEVEL",
    COUNT("GENE_ID") AS "GENE_ID",
    COUNT("UNIPROT_ACCESSION") AS "UNIPROT_ACCESSION",
    COUNT("GENE_SYMBOL") AS "GENE_SYMBOL",
    COUNT("ALTERNATE_NAMES") AS "ALTERNATE_NAMES"
    FROM SNOWFLAKE_LEARNING_DB."PROTEINS_PEPTIDES".RECOMBINANT_PROTEINS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EXPRESSION_SYSTEM', 'Expression system', TRUE),
    ('FORM', 'Form', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('SPECIES', 'Species', FALSE),
    ('SEQUENCE_REGION', 'Sequence / region', FALSE),
    ('TAG', 'Tag', FALSE),
    ('BIOACTIVITY', 'Bioactivity', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular weight', FALSE),
    ('STORAGE_RECONSTITUTION', 'Storage / reconstitution', FALSE),
    ('ENDOTOXIN_LEVEL', 'Endotoxin level', FALSE),
    ('GENE_ID', 'Gene ID', FALSE),
    ('UNIPROT_ACCESSION', 'UniProt accession', FALSE),
    ('GENE_SYMBOL', 'Gene symbol', FALSE),
    ('ALTERNATE_NAMES', 'Alternate names', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'proteins_peptides' AS ASSIGNED_L2_ID,
    'Proteins and Peptides' AS ASSIGNED_L2_LABEL,
    'recombinant_proteins' AS ASSIGNED_L3_ID,
    'Recombinant Proteins' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Proteins and Peptides > Recombinant Proteins
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("BIOACTIVITY_ED50") AS "BIOACTIVITY_ED50",
    COUNT("FORM") AS "FORM",
    COUNT("PURITY") AS "PURITY",
    COUNT("SOURCE_EXPRESSION_SYSTEM") AS "SOURCE_EXPRESSION_SYSTEM",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("SEQUENCE_REGION") AS "SEQUENCE_REGION",
    COUNT("CARRIER") AS "CARRIER",
    COUNT("ENDOTOXIN_LEVEL") AS "ENDOTOXIN_LEVEL",
    COUNT("GRADE") AS "GRADE",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("STORAGE_RECONSTITUTION") AS "STORAGE_RECONSTITUTION",
    COUNT("GENE_ID") AS "GENE_ID",
    COUNT("UNIPROT_ACCESSION") AS "UNIPROT_ACCESSION",
    COUNT("GENE_SYMBOL") AS "GENE_SYMBOL",
    COUNT("ALTERNATE_NAMES") AS "ALTERNATE_NAMES"
    FROM SNOWFLAKE_LEARNING_DB."PROTEINS_PEPTIDES".CYTOKINES_GROWTH_FACTORS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('BIOACTIVITY_ED50', 'Bioactivity / ED50', TRUE),
    ('FORM', 'Form', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('SOURCE_EXPRESSION_SYSTEM', 'Source / Expression system', TRUE),
    ('SPECIES', 'Species', FALSE),
    ('SEQUENCE_REGION', 'Sequence / region', FALSE),
    ('CARRIER', 'Carrier', FALSE),
    ('ENDOTOXIN_LEVEL', 'Endotoxin level', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular weight', FALSE),
    ('STORAGE_RECONSTITUTION', 'Storage / reconstitution', FALSE),
    ('GENE_ID', 'Gene ID', FALSE),
    ('UNIPROT_ACCESSION', 'UniProt accession', FALSE),
    ('GENE_SYMBOL', 'Gene symbol', FALSE),
    ('ALTERNATE_NAMES', 'Alternate names', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'proteins_peptides' AS ASSIGNED_L2_ID,
    'Proteins and Peptides' AS ASSIGNED_L2_LABEL,
    'cytokines_growth_factors' AS ASSIGNED_L3_ID,
    'Cytokines & Growth Factors' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Proteins and Peptides > Cytokines & Growth Factors
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SPECIFIC_ACTIVITY") AS "SPECIFIC_ACTIVITY",
    COUNT("FORM") AS "FORM",
    COUNT("PURITY") AS "PURITY",
    COUNT("SOURCE_EXPRESSION_SYSTEM") AS "SOURCE_EXPRESSION_SYSTEM",
    COUNT("COFACTORS_SUBSTRATE") AS "COFACTORS_SUBSTRATE",
    COUNT("GRADE") AS "GRADE",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("STORAGE_RECONSTITUTION") AS "STORAGE_RECONSTITUTION",
    COUNT("ENDOTOXIN_LEVEL") AS "ENDOTOXIN_LEVEL",
    COUNT("GENE_ID") AS "GENE_ID",
    COUNT("UNIPROT_ACCESSION") AS "UNIPROT_ACCESSION",
    COUNT("GENE_SYMBOL") AS "GENE_SYMBOL",
    COUNT("ALTERNATE_NAMES") AS "ALTERNATE_NAMES"
    FROM SNOWFLAKE_LEARNING_DB."PROTEINS_PEPTIDES".ENZYMES_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SPECIFIC_ACTIVITY', 'Specific activity', TRUE),
    ('FORM', 'Form', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('SOURCE_EXPRESSION_SYSTEM', 'Source / Expression system', TRUE),
    ('COFACTORS_SUBSTRATE', 'Cofactors / substrate', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular weight', FALSE),
    ('STORAGE_RECONSTITUTION', 'Storage / reconstitution', FALSE),
    ('ENDOTOXIN_LEVEL', 'Endotoxin level', FALSE),
    ('GENE_ID', 'Gene ID', FALSE),
    ('UNIPROT_ACCESSION', 'UniProt accession', FALSE),
    ('GENE_SYMBOL', 'Gene symbol', FALSE),
    ('ALTERNATE_NAMES', 'Alternate names', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'proteins_peptides' AS ASSIGNED_L2_ID,
    'Proteins and Peptides' AS ASSIGNED_L2_LABEL,
    'enzymes' AS ASSIGNED_L3_ID,
    'Enzymes' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Proteins and Peptides > Enzymes
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SOURCE") AS "SOURCE",
    COUNT("FORM") AS "FORM",
    COUNT("PURITY") AS "PURITY",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("FATTY_ACID_FREE_GLOBULIN_FREE") AS "FATTY_ACID_FREE_GLOBULIN_FREE",
    COUNT("FRACTION_PURITY") AS "FRACTION_PURITY",
    COUNT("GRADE") AS "GRADE",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("STORAGE_RECONSTITUTION") AS "STORAGE_RECONSTITUTION",
    COUNT("ENDOTOXIN_LEVEL") AS "ENDOTOXIN_LEVEL"
    FROM SNOWFLAKE_LEARNING_DB."PROTEINS_PEPTIDES".ALBUMIN_CARRIER_PROTEINS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SOURCE', 'Source', TRUE),
    ('FORM', 'Form', TRUE),
    ('PURITY', 'Purity', TRUE),
    ('SPECIES', 'Species', FALSE),
    ('FATTY_ACID_FREE_GLOBULIN_FREE', 'Fatty-acid-free / globulin-free', FALSE),
    ('FRACTION_PURITY', 'Fraction / purity', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular weight', FALSE),
    ('STORAGE_RECONSTITUTION', 'Storage / reconstitution', FALSE),
    ('ENDOTOXIN_LEVEL', 'Endotoxin level', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'proteins_peptides' AS ASSIGNED_L2_ID,
    'Proteins and Peptides' AS ASSIGNED_L2_LABEL,
    'albumin_carrier_proteins' AS ASSIGNED_L3_ID,
    'Albumin & Carrier Proteins' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Proteins and Peptides > Albumin & Carrier Proteins
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("SEQUENCE") AS "SEQUENCE",
    COUNT("LENGTH_AA") AS "LENGTH_AA",
    COUNT("PURITY_HPLC") AS "PURITY_HPLC",
    COUNT("FORM") AS "FORM",
    COUNT("COUNTERION_SALT") AS "COUNTERION_SALT",
    COUNT("MODIFICATIONS") AS "MODIFICATIONS",
    COUNT("SCALE") AS "SCALE",
    COUNT("GRADE") AS "GRADE",
    COUNT("MOLECULAR_WEIGHT") AS "MOLECULAR_WEIGHT",
    COUNT("STORAGE_RECONSTITUTION") AS "STORAGE_RECONSTITUTION",
    COUNT("ENDOTOXIN_LEVEL") AS "ENDOTOXIN_LEVEL"
    FROM SNOWFLAKE_LEARNING_DB."PROTEINS_PEPTIDES".PEPTIDES_AMINO_ACIDS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('SEQUENCE', 'Sequence', TRUE),
    ('LENGTH_AA', 'Length (aa)', TRUE),
    ('PURITY_HPLC', 'Purity (HPLC)', TRUE),
    ('FORM', 'Form', TRUE),
    ('COUNTERION_SALT', 'Counterion / salt', FALSE),
    ('MODIFICATIONS', 'Modifications', FALSE),
    ('SCALE', 'Scale', FALSE),
    ('GRADE', 'Grade', FALSE),
    ('MOLECULAR_WEIGHT', 'Molecular weight', FALSE),
    ('STORAGE_RECONSTITUTION', 'Storage / reconstitution', FALSE),
    ('ENDOTOXIN_LEVEL', 'Endotoxin level', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'proteins_peptides' AS ASSIGNED_L2_ID,
    'Proteins and Peptides' AS ASSIGNED_L2_LABEL,
    'peptides_amino_acids' AS ASSIGNED_L3_ID,
    'Peptides & Amino Acids' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Proteins and Peptides > Peptides & Amino Acids
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("EXPRESSION_SYSTEM") AS "EXPRESSION_SYSTEM",
    COUNT("PROTEIN_TYPE") AS "PROTEIN_TYPE",
    COUNT("MAXIMUM_PRODUCTION_SCALE") AS "MAXIMUM_PRODUCTION_SCALE",
    COUNT("PURITY_SPECIFICATION") AS "PURITY_SPECIFICATION",
    COUNT("AFFINITY_TAG") AS "AFFINITY_TAG",
    COUNT("PURIFICATION_METHOD") AS "PURIFICATION_METHOD",
    COUNT("ENDOTOXIN_LEVEL_EU_MG") AS "ENDOTOXIN_LEVEL_EU_MG",
    COUNT("QC_METHODS_INCLUDED") AS "QC_METHODS_INCLUDED",
    COUNT("GENE_SYNTHESIS_INCLUDED") AS "GENE_SYNTHESIS_INCLUDED",
    COUNT("ACCEPTED_STARTING_MATERIAL") AS "ACCEPTED_STARTING_MATERIAL",
    COUNT("DELIVERED_FORM") AS "DELIVERED_FORM",
    COUNT("QUALITY_GRADE") AS "QUALITY_GRADE",
    COUNT("EXPRESSION_MODE") AS "EXPRESSION_MODE",
    COUNT("TYPICAL_EXPRESSION_YIELD_MG_L") AS "TYPICAL_EXPRESSION_YIELD_MG_L",
    COUNT("PURITY_ASSAY_BASIS") AS "PURITY_ASSAY_BASIS",
    COUNT("TAG_REMOVAL_OFFERED") AS "TAG_REMOVAL_OFFERED",
    COUNT("INCLUSION_BODY_REFOLDING") AS "INCLUSION_BODY_REFOLDING",
    COUNT("MEMBRANE_PROTEIN_FORMAT") AS "MEMBRANE_PROTEIN_FORMAT"
    FROM SNOWFLAKE_LEARNING_DB."PROTEINS_PEPTIDES".PROTEIN_PURIFICATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('EXPRESSION_SYSTEM', 'Expression system', TRUE),
    ('PROTEIN_TYPE', 'Protein type', FALSE),
    ('MAXIMUM_PRODUCTION_SCALE', 'Maximum production scale', FALSE),
    ('PURITY_SPECIFICATION', 'Purity specification (%)', FALSE),
    ('AFFINITY_TAG', 'Affinity tag', FALSE),
    ('PURIFICATION_METHOD', 'Purification method', FALSE),
    ('ENDOTOXIN_LEVEL_EU_MG', 'Endotoxin level (EU/mg)', FALSE),
    ('QC_METHODS_INCLUDED', 'QC methods included', FALSE),
    ('GENE_SYNTHESIS_INCLUDED', 'Gene synthesis included', FALSE),
    ('ACCEPTED_STARTING_MATERIAL', 'Accepted starting material', FALSE),
    ('DELIVERED_FORM', 'Delivered form', FALSE),
    ('QUALITY_GRADE', 'Quality grade', FALSE),
    ('EXPRESSION_MODE', 'Expression mode', FALSE),
    ('TYPICAL_EXPRESSION_YIELD_MG_L', 'Typical expression yield (mg/L)', FALSE),
    ('PURITY_ASSAY_BASIS', 'Purity assay basis', FALSE),
    ('TAG_REMOVAL_OFFERED', 'Tag removal offered', FALSE),
    ('INCLUSION_BODY_REFOLDING', 'Inclusion body refolding', FALSE),
    ('MEMBRANE_PROTEIN_FORMAT', 'Membrane protein format', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'proteins_peptides' AS ASSIGNED_L2_ID,
    'Proteins and Peptides' AS ASSIGNED_L2_LABEL,
    'protein_purification' AS ASSIGNED_L3_ID,
    'Protein purification' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Proteins and Peptides > Protein purification
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("MOLECULE_TYPE") AS "MOLECULE_TYPE",
    COUNT("PURITY_METHOD") AS "PURITY_METHOD",
    COUNT("IDENTITY_CONFIRMATION_METHOD") AS "IDENTITY_CONFIRMATION_METHOD",
    COUNT("INTACT_MASS_METHOD") AS "INTACT_MASS_METHOD",
    COUNT("GLYCAN_ANALYSIS") AS "GLYCAN_ANALYSIS",
    COUNT("AGGREGATION_METHOD") AS "AGGREGATION_METHOD",
    COUNT("THERMAL_STABILITY_METHOD") AS "THERMAL_STABILITY_METHOD",
    COUNT("BINDING_KINETICS_METHOD") AS "BINDING_KINETICS_METHOD",
    COUNT("PTM_ANALYSIS") AS "PTM_ANALYSIS",
    COUNT("POTENCY_ASSAYS") AS "POTENCY_ASSAYS",
    COUNT("REGULATORY_STANDARD") AS "REGULATORY_STANDARD",
    COUNT("CHARGE_VARIANT_METHOD") AS "CHARGE_VARIANT_METHOD",
    COUNT("HIGHER_ORDER_STRUCTURE_METHOD") AS "HIGHER_ORDER_STRUCTURE_METHOD",
    COUNT("PROCESS_IMPURITY_ASSAYS") AS "PROCESS_IMPURITY_ASSAYS",
    COUNT("DEVELOPABILITY_ASSAYS") AS "DEVELOPABILITY_ASSAYS",
    COUNT("EPITOPE_MAPPING") AS "EPITOPE_MAPPING"
    FROM SNOWFLAKE_LEARNING_DB."PROTEINS_PEPTIDES".PROTEIN_CHARACTERIZATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('MOLECULE_TYPE', 'Molecule type', FALSE),
    ('PURITY_METHOD', 'Purity method', FALSE),
    ('IDENTITY_CONFIRMATION_METHOD', 'Identity confirmation method', FALSE),
    ('INTACT_MASS_METHOD', 'Intact mass method', FALSE),
    ('GLYCAN_ANALYSIS', 'Glycan analysis', FALSE),
    ('AGGREGATION_METHOD', 'Aggregation method', FALSE),
    ('THERMAL_STABILITY_METHOD', 'Thermal stability method', FALSE),
    ('BINDING_KINETICS_METHOD', 'Binding kinetics method', FALSE),
    ('PTM_ANALYSIS', 'PTM analysis', FALSE),
    ('POTENCY_ASSAYS', 'Potency assays', FALSE),
    ('REGULATORY_STANDARD', 'Regulatory standard', FALSE),
    ('CHARGE_VARIANT_METHOD', 'Charge variant method', FALSE),
    ('HIGHER_ORDER_STRUCTURE_METHOD', 'Higher-order structure method', FALSE),
    ('PROCESS_IMPURITY_ASSAYS', 'Process impurity assays', FALSE),
    ('DEVELOPABILITY_ASSAYS', 'Developability assays', FALSE),
    ('EPITOPE_MAPPING', 'Epitope mapping', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'proteins_peptides' AS ASSIGNED_L2_ID,
    'Proteins and Peptides' AS ASSIGNED_L2_LABEL,
    'protein_characterization' AS ASSIGNED_L3_ID,
    'Protein characterization' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Proteins and Peptides > Protein characterization
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("ENGINEERING_TYPE") AS "ENGINEERING_TYPE",
    COUNT("HUMANIZATION_METHOD") AS "HUMANIZATION_METHOD",
    COUNT("SOURCE_SPECIES") AS "SOURCE_SPECIES",
    COUNT("AFFINITY_MATURATION_APPROACH") AS "AFFINITY_MATURATION_APPROACH",
    COUNT("DISPLAY_PLATFORM") AS "DISPLAY_PLATFORM",
    COUNT("TARGET_AFFINITY_IMPROVEMENT_FOLD") AS "TARGET_AFFINITY_IMPROVEMENT_FOLD",
    COUNT("ANTIBODY_FORMAT") AS "ANTIBODY_FORMAT",
    COUNT("FC_ENGINEERING_OPTION") AS "FC_ENGINEERING_OPTION",
    COUNT("DELIVERABLES") AS "DELIVERABLES",
    COUNT("LIBRARY_SIZE_CLONES") AS "LIBRARY_SIZE_CLONES",
    COUNT("HUMANNESS_TARGET") AS "HUMANNESS_TARGET",
    COUNT("AFFINITY_MEASUREMENT_METHOD") AS "AFFINITY_MEASUREMENT_METHOD",
    COUNT("IMMUNOGENICITY_ASSESSMENT") AS "IMMUNOGENICITY_ASSESSMENT",
    COUNT("DEVELOPABILITY_LIABILITY_SCREENING") AS "DEVELOPABILITY_LIABILITY_SCREENING",
    COUNT("NUMBER_OF_LEAD_CANDIDATES_DELIVERED") AS "NUMBER_OF_LEAD_CANDIDATES_DELIVERED",
    COUNT("IP_ROYALTY_FREE") AS "IP_ROYALTY_FREE"
    FROM SNOWFLAKE_LEARNING_DB."PROTEINS_PEPTIDES".PROTEIN_ENGINEERING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('ENGINEERING_TYPE', 'Engineering type', TRUE),
    ('HUMANIZATION_METHOD', 'Humanization method', FALSE),
    ('SOURCE_SPECIES', 'Source species', FALSE),
    ('AFFINITY_MATURATION_APPROACH', 'Affinity maturation approach', FALSE),
    ('DISPLAY_PLATFORM', 'Display platform', FALSE),
    ('TARGET_AFFINITY_IMPROVEMENT_FOLD', 'Target affinity improvement (fold)', FALSE),
    ('ANTIBODY_FORMAT', 'Antibody format', FALSE),
    ('FC_ENGINEERING_OPTION', 'Fc engineering option', FALSE),
    ('DELIVERABLES', 'Deliverables', FALSE),
    ('LIBRARY_SIZE_CLONES', 'Library size (clones)', FALSE),
    ('HUMANNESS_TARGET', 'Humanness target (%)', FALSE),
    ('AFFINITY_MEASUREMENT_METHOD', 'Affinity measurement method', FALSE),
    ('IMMUNOGENICITY_ASSESSMENT', 'Immunogenicity assessment', FALSE),
    ('DEVELOPABILITY_LIABILITY_SCREENING', 'Developability liability screening', FALSE),
    ('NUMBER_OF_LEAD_CANDIDATES_DELIVERED', 'Number of lead candidates delivered', FALSE),
    ('IP_ROYALTY_FREE', 'IP / royalty-free', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'proteins_peptides' AS ASSIGNED_L2_ID,
    'Proteins and Peptides' AS ASSIGNED_L2_LABEL,
    'protein_engineering' AS ASSIGNED_L3_ID,
    'Protein engineering' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Proteins and Peptides > Protein engineering
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("CONJUGATE_TYPE") AS "CONJUGATE_TYPE",
    COUNT("LABEL_TYPE") AS "LABEL_TYPE",
    COUNT("CONJUGATION_CHEMISTRY") AS "CONJUGATION_CHEMISTRY",
    COUNT("CARRIER_MOLECULE") AS "CARRIER_MOLECULE",
    COUNT("TARGET_LOADING_RATIO") AS "TARGET_LOADING_RATIO",
    COUNT("LOADING_CHARACTERIZATION_METHOD") AS "LOADING_CHARACTERIZATION_METHOD",
    COUNT("FREE_DRUG_ANALYSIS") AS "FREE_DRUG_ANALYSIS",
    COUNT("DRUG_PAYLOAD_CLASS") AS "DRUG_PAYLOAD_CLASS",
    COUNT("LINKER_TYPE") AS "LINKER_TYPE",
    COUNT("CONJUGATION_SITE_IDENTIFICATION") AS "CONJUGATION_SITE_IDENTIFICATION",
    COUNT("ACTIVITY_RETENTION_CONFIRMED") AS "ACTIVITY_RETENTION_CONFIRMED",
    COUNT("BATCH_SIZE_MG") AS "BATCH_SIZE_MG",
    COUNT("MINIMUM_INPUT_ANTIBODY_PURITY") AS "MINIMUM_INPUT_ANTIBODY_PURITY",
    COUNT("GMP_MANUFACTURING_AVAILABLE") AS "GMP_MANUFACTURING_AVAILABLE"
    FROM SNOWFLAKE_LEARNING_DB."PROTEINS_PEPTIDES".PROTEIN_CONJUGATION_LABELING_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('CONJUGATE_TYPE', 'Conjugate type', TRUE),
    ('LABEL_TYPE', 'Label type', FALSE),
    ('CONJUGATION_CHEMISTRY', 'Conjugation chemistry', FALSE),
    ('CARRIER_MOLECULE', 'Carrier molecule', FALSE),
    ('TARGET_LOADING_RATIO', 'Target loading ratio', FALSE),
    ('LOADING_CHARACTERIZATION_METHOD', 'Loading characterization method', FALSE),
    ('FREE_DRUG_ANALYSIS', 'Free drug analysis', FALSE),
    ('DRUG_PAYLOAD_CLASS', 'Drug payload class', FALSE),
    ('LINKER_TYPE', 'Linker type', FALSE),
    ('CONJUGATION_SITE_IDENTIFICATION', 'Conjugation site identification', FALSE),
    ('ACTIVITY_RETENTION_CONFIRMED', 'Activity retention confirmed', FALSE),
    ('BATCH_SIZE_MG', 'Batch size (mg)', FALSE),
    ('MINIMUM_INPUT_ANTIBODY_PURITY', 'Minimum input antibody purity (%)', FALSE),
    ('GMP_MANUFACTURING_AVAILABLE', 'GMP manufacturing available', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'proteins_peptides' AS ASSIGNED_L2_ID,
    'Proteins and Peptides' AS ASSIGNED_L2_LABEL,
    'protein_conjugation_labeling' AS ASSIGNED_L3_ID,
    'Protein conjugation & labeling' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Proteins and Peptides > Protein conjugation & labeling
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_REGULATORY_AFFAIRS_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("REGULATORY_AGENCY") AS "REGULATORY_AGENCY",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DEVELOPMENT_DISCIPLINE_COVERED") AS "DEVELOPMENT_DISCIPLINE_COVERED",
    COUNT("TARGET_MARKET_REGION") AS "TARGET_MARKET_REGION",
    COUNT("ENGAGEMENT_MODEL") AS "ENGAGEMENT_MODEL",
    COUNT("REGULATORY_INFORMATION_MANAGEMENT_SYSTEM") AS "REGULATORY_INFORMATION_MANAGEMENT_SYSTEM",
    COUNT("PRICING_STRUCTURE") AS "PRICING_STRUCTURE",
    COUNT("STRATEGY_DELIVERABLE") AS "STRATEGY_DELIVERABLE",
    COUNT("EXPEDITED_PATHWAY_OR_DESIGNATION") AS "EXPEDITED_PATHWAY_OR_DESIGNATION",
    COUNT("PEDIATRIC_PLAN") AS "PEDIATRIC_PLAN",
    COUNT("REGULATORY_INTELLIGENCE_INCLUDED") AS "REGULATORY_INTELLIGENCE_INCLUDED",
    COUNT("REGULATORY_INTELLIGENCE_DELIVERY") AS "REGULATORY_INTELLIGENCE_DELIVERY",
    COUNT("HTA_ALIGNMENT") AS "HTA_ALIGNMENT",
    COUNT("DUE_DILIGENCE_SUPPORT") AS "DUE_DILIGENCE_SUPPORT",
    COUNT("EXPERT_WITNESS_SUPPORT") AS "EXPERT_WITNESS_SUPPORT"
    FROM SNOWFLAKE_LEARNING_DB."REGULATORY_AFFAIRS".REGULATORY_STRATEGY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_TYPE', 'Product type', TRUE),
    ('REGULATORY_AGENCY', 'Regulatory agency', TRUE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('DEVELOPMENT_DISCIPLINE_COVERED', 'Development discipline covered', FALSE),
    ('TARGET_MARKET_REGION', 'Target market region', FALSE),
    ('ENGAGEMENT_MODEL', 'Engagement model', FALSE),
    ('REGULATORY_INFORMATION_MANAGEMENT_SYSTEM', 'Regulatory information management system', FALSE),
    ('PRICING_STRUCTURE', 'Pricing structure', FALSE),
    ('STRATEGY_DELIVERABLE', 'Strategy deliverable', TRUE),
    ('EXPEDITED_PATHWAY_OR_DESIGNATION', 'Expedited pathway or designation', FALSE),
    ('PEDIATRIC_PLAN', 'Pediatric plan', FALSE),
    ('REGULATORY_INTELLIGENCE_INCLUDED', 'Regulatory intelligence included', FALSE),
    ('REGULATORY_INTELLIGENCE_DELIVERY', 'Regulatory intelligence delivery', FALSE),
    ('HTA_ALIGNMENT', 'HTA alignment', FALSE),
    ('DUE_DILIGENCE_SUPPORT', 'Due diligence support', FALSE),
    ('EXPERT_WITNESS_SUPPORT', 'Expert witness support', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'regulatory_affairs' AS ASSIGNED_L2_ID,
    'Regulatory Affairs' AS ASSIGNED_L2_LABEL,
    'regulatory_strategy' AS ASSIGNED_L3_ID,
    'Regulatory Strategy' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Regulatory Affairs > Regulatory Strategy
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("REGULATORY_AGENCY") AS "REGULATORY_AGENCY",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DEVELOPMENT_DISCIPLINE_COVERED") AS "DEVELOPMENT_DISCIPLINE_COVERED",
    COUNT("TARGET_MARKET_REGION") AS "TARGET_MARKET_REGION",
    COUNT("ENGAGEMENT_MODEL") AS "ENGAGEMENT_MODEL",
    COUNT("REGULATORY_INFORMATION_MANAGEMENT_SYSTEM") AS "REGULATORY_INFORMATION_MANAGEMENT_SYSTEM",
    COUNT("PRICING_STRUCTURE") AS "PRICING_STRUCTURE",
    COUNT("DOCUMENT_TYPE_AUTHORED") AS "DOCUMENT_TYPE_AUTHORED",
    COUNT("SUBMISSION_FORMAT") AS "SUBMISSION_FORMAT",
    COUNT("PUBLISHING_TOOL") AS "PUBLISHING_TOOL",
    COUNT("PUBLISHING_QC_AND_VALIDATION") AS "PUBLISHING_QC_AND_VALIDATION",
    COUNT("CTD_MODULE_SCOPE") AS "CTD_MODULE_SCOPE",
    COUNT("MASTER_FILE_TYPE") AS "MASTER_FILE_TYPE",
    COUNT("STRUCTURED_DATA_SUBMISSION") AS "STRUCTURED_DATA_SUBMISSION",
    COUNT("LEGACY_CONVERSION") AS "LEGACY_CONVERSION",
    COUNT("TRANSLATION_AND_LOCALIZATION") AS "TRANSLATION_AND_LOCALIZATION",
    COUNT("AI_ASSISTED_AUTHORING") AS "AI_ASSISTED_AUTHORING"
    FROM SNOWFLAKE_LEARNING_DB."REGULATORY_AFFAIRS".DOSSIER_ECTD_PREPARATION_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_TYPE', 'Product type', TRUE),
    ('REGULATORY_AGENCY', 'Regulatory agency', TRUE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('DEVELOPMENT_DISCIPLINE_COVERED', 'Development discipline covered', FALSE),
    ('TARGET_MARKET_REGION', 'Target market region', FALSE),
    ('ENGAGEMENT_MODEL', 'Engagement model', FALSE),
    ('REGULATORY_INFORMATION_MANAGEMENT_SYSTEM', 'Regulatory information management system', FALSE),
    ('PRICING_STRUCTURE', 'Pricing structure', FALSE),
    ('DOCUMENT_TYPE_AUTHORED', 'Document type authored', TRUE),
    ('SUBMISSION_FORMAT', 'Submission format', FALSE),
    ('PUBLISHING_TOOL', 'Publishing tool', FALSE),
    ('PUBLISHING_QC_AND_VALIDATION', 'Publishing QC and validation', FALSE),
    ('CTD_MODULE_SCOPE', 'CTD module scope', FALSE),
    ('MASTER_FILE_TYPE', 'Master file type', FALSE),
    ('STRUCTURED_DATA_SUBMISSION', 'Structured data submission', FALSE),
    ('LEGACY_CONVERSION', 'Legacy conversion', FALSE),
    ('TRANSLATION_AND_LOCALIZATION', 'Translation and localization', FALSE),
    ('AI_ASSISTED_AUTHORING', 'AI-assisted authoring', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'regulatory_affairs' AS ASSIGNED_L2_ID,
    'Regulatory Affairs' AS ASSIGNED_L2_LABEL,
    'dossier_ectd_preparation' AS ASSIGNED_L3_ID,
    'Dossier / eCTD Preparation' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Regulatory Affairs > Dossier / eCTD Preparation
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("REGULATORY_AGENCY") AS "REGULATORY_AGENCY",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DEVELOPMENT_DISCIPLINE_COVERED") AS "DEVELOPMENT_DISCIPLINE_COVERED",
    COUNT("TARGET_MARKET_REGION") AS "TARGET_MARKET_REGION",
    COUNT("ENGAGEMENT_MODEL") AS "ENGAGEMENT_MODEL",
    COUNT("REGULATORY_INFORMATION_MANAGEMENT_SYSTEM") AS "REGULATORY_INFORMATION_MANAGEMENT_SYSTEM",
    COUNT("PRICING_STRUCTURE") AS "PRICING_STRUCTURE",
    COUNT("SUBMISSION_TYPE") AS "SUBMISSION_TYPE",
    COUNT("MEETING_TYPE") AS "MEETING_TYPE",
    COUNT("AGENCY_QUERY_RESPONSE") AS "AGENCY_QUERY_RESPONSE",
    COUNT("BRIEFING_PACKAGE_PREPARATION") AS "BRIEFING_PACKAGE_PREPARATION",
    COUNT("EU_PROCEDURE") AS "EU_PROCEDURE",
    COUNT("ELECTRONIC_GATEWAY") AS "ELECTRONIC_GATEWAY",
    COUNT("MARKETS_SUPPORTED_FOR_DISPATCH") AS "MARKETS_SUPPORTED_FOR_DISPATCH",
    COUNT("US_AGENT_SERVICE") AS "US_AGENT_SERVICE",
    COUNT("LOCAL_AFFILIATE_SERVICE") AS "LOCAL_AFFILIATE_SERVICE",
    COUNT("CLINICAL_TRIAL_TRANSPARENCY") AS "CLINICAL_TRIAL_TRANSPARENCY",
    COUNT("ESTABLISHMENT_REGISTRATION_AND_LISTING") AS "ESTABLISHMENT_REGISTRATION_AND_LISTING"
    FROM SNOWFLAKE_LEARNING_DB."REGULATORY_AFFAIRS".SUBMISSIONS_AGENCY_LIAISON_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_TYPE', 'Product type', TRUE),
    ('REGULATORY_AGENCY', 'Regulatory agency', TRUE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('DEVELOPMENT_DISCIPLINE_COVERED', 'Development discipline covered', FALSE),
    ('TARGET_MARKET_REGION', 'Target market region', FALSE),
    ('ENGAGEMENT_MODEL', 'Engagement model', FALSE),
    ('REGULATORY_INFORMATION_MANAGEMENT_SYSTEM', 'Regulatory information management system', FALSE),
    ('PRICING_STRUCTURE', 'Pricing structure', FALSE),
    ('SUBMISSION_TYPE', 'Submission type', TRUE),
    ('MEETING_TYPE', 'Meeting type', FALSE),
    ('AGENCY_QUERY_RESPONSE', 'Agency query response', FALSE),
    ('BRIEFING_PACKAGE_PREPARATION', 'Briefing package preparation', FALSE),
    ('EU_PROCEDURE', 'EU procedure', FALSE),
    ('ELECTRONIC_GATEWAY', 'Electronic gateway', FALSE),
    ('MARKETS_SUPPORTED_FOR_DISPATCH', 'Markets supported for dispatch', FALSE),
    ('US_AGENT_SERVICE', 'US agent service', FALSE),
    ('LOCAL_AFFILIATE_SERVICE', 'Local affiliate service', FALSE),
    ('CLINICAL_TRIAL_TRANSPARENCY', 'Clinical trial transparency', FALSE),
    ('ESTABLISHMENT_REGISTRATION_AND_LISTING', 'Establishment registration and listing', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'regulatory_affairs' AS ASSIGNED_L2_ID,
    'Regulatory Affairs' AS ASSIGNED_L2_LABEL,
    'submissions_agency_liaison' AS ASSIGNED_L3_ID,
    'Submissions & Agency Liaison' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Regulatory Affairs > Submissions & Agency Liaison
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("REGULATORY_AGENCY") AS "REGULATORY_AGENCY",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DEVELOPMENT_DISCIPLINE_COVERED") AS "DEVELOPMENT_DISCIPLINE_COVERED",
    COUNT("TARGET_MARKET_REGION") AS "TARGET_MARKET_REGION",
    COUNT("ENGAGEMENT_MODEL") AS "ENGAGEMENT_MODEL",
    COUNT("REGULATORY_INFORMATION_MANAGEMENT_SYSTEM") AS "REGULATORY_INFORMATION_MANAGEMENT_SYSTEM",
    COUNT("PRICING_STRUCTURE") AS "PRICING_STRUCTURE",
    COUNT("LABELING_ACTIVITY") AS "LABELING_ACTIVITY",
    COUNT("LABELING_DOCUMENT_TYPE") AS "LABELING_DOCUMENT_TYPE",
    COUNT("STRUCTURED_PRODUCT_LABELING") AS "STRUCTURED_PRODUCT_LABELING",
    COUNT("ARTWORK_SERVICE") AS "ARTWORK_SERVICE",
    COUNT("LABEL_TRANSLATION") AS "LABEL_TRANSLATION",
    COUNT("READABILITY_USER_TESTING") AS "READABILITY_USER_TESTING",
    COUNT("PROMOTIONAL_MATERIAL_REVIEW") AS "PROMOTIONAL_MATERIAL_REVIEW",
    COUNT("SPL_SUBMISSION_TYPE") AS "SPL_SUBMISSION_TYPE",
    COUNT("PACKAGING_COMPONENT") AS "PACKAGING_COMPONENT"
    FROM SNOWFLAKE_LEARNING_DB."REGULATORY_AFFAIRS".LABELING_REGULATORY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_TYPE', 'Product type', TRUE),
    ('REGULATORY_AGENCY', 'Regulatory agency', TRUE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('DEVELOPMENT_DISCIPLINE_COVERED', 'Development discipline covered', FALSE),
    ('TARGET_MARKET_REGION', 'Target market region', FALSE),
    ('ENGAGEMENT_MODEL', 'Engagement model', FALSE),
    ('REGULATORY_INFORMATION_MANAGEMENT_SYSTEM', 'Regulatory information management system', FALSE),
    ('PRICING_STRUCTURE', 'Pricing structure', FALSE),
    ('LABELING_ACTIVITY', 'Labeling activity', TRUE),
    ('LABELING_DOCUMENT_TYPE', 'Labeling document type', FALSE),
    ('STRUCTURED_PRODUCT_LABELING', 'Structured Product Labeling', FALSE),
    ('ARTWORK_SERVICE', 'Artwork service', FALSE),
    ('LABEL_TRANSLATION', 'Label translation', FALSE),
    ('READABILITY_USER_TESTING', 'Readability user testing', FALSE),
    ('PROMOTIONAL_MATERIAL_REVIEW', 'Promotional material review', FALSE),
    ('SPL_SUBMISSION_TYPE', 'SPL submission type', FALSE),
    ('PACKAGING_COMPONENT', 'Packaging component', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'regulatory_affairs' AS ASSIGNED_L2_ID,
    'Regulatory Affairs' AS ASSIGNED_L2_LABEL,
    'labeling_regulatory' AS ASSIGNED_L3_ID,
    'Labeling' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Regulatory Affairs > Labeling
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("PRODUCT_TYPE") AS "PRODUCT_TYPE",
    COUNT("REGULATORY_AGENCY") AS "REGULATORY_AGENCY",
    COUNT("DEVELOPMENT_STAGE") AS "DEVELOPMENT_STAGE",
    COUNT("DEVELOPMENT_DISCIPLINE_COVERED") AS "DEVELOPMENT_DISCIPLINE_COVERED",
    COUNT("TARGET_MARKET_REGION") AS "TARGET_MARKET_REGION",
    COUNT("ENGAGEMENT_MODEL") AS "ENGAGEMENT_MODEL",
    COUNT("REGULATORY_INFORMATION_MANAGEMENT_SYSTEM") AS "REGULATORY_INFORMATION_MANAGEMENT_SYSTEM",
    COUNT("PRICING_STRUCTURE") AS "PRICING_STRUCTURE",
    COUNT("AUDIT_TYPE") AS "AUDIT_TYPE",
    COUNT("INSPECTION_READINESS_SERVICE") AS "INSPECTION_READINESS_SERVICE",
    COUNT("ENFORCEMENT_REMEDIATION") AS "ENFORCEMENT_REMEDIATION",
    COUNT("SOP_AND_QMS_DEVELOPMENT") AS "SOP_AND_QMS_DEVELOPMENT",
    COUNT("AUDITED_ENTITY") AS "AUDITED_ENTITY",
    COUNT("AUDIT_DELIVERABLE") AS "AUDIT_DELIVERABLE",
    COUNT("STANDARD_AUDITED_AGAINST") AS "STANDARD_AUDITED_AGAINST"
    FROM SNOWFLAKE_LEARNING_DB."REGULATORY_AFFAIRS".AUDITING_COMPLIANCE_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('PRODUCT_TYPE', 'Product type', TRUE),
    ('REGULATORY_AGENCY', 'Regulatory agency', TRUE),
    ('DEVELOPMENT_STAGE', 'Development stage', FALSE),
    ('DEVELOPMENT_DISCIPLINE_COVERED', 'Development discipline covered', FALSE),
    ('TARGET_MARKET_REGION', 'Target market region', FALSE),
    ('ENGAGEMENT_MODEL', 'Engagement model', FALSE),
    ('REGULATORY_INFORMATION_MANAGEMENT_SYSTEM', 'Regulatory information management system', FALSE),
    ('PRICING_STRUCTURE', 'Pricing structure', FALSE),
    ('AUDIT_TYPE', 'Audit type', TRUE),
    ('INSPECTION_READINESS_SERVICE', 'Inspection readiness service', FALSE),
    ('ENFORCEMENT_REMEDIATION', 'Enforcement remediation', FALSE),
    ('SOP_AND_QMS_DEVELOPMENT', 'SOP and QMS development', FALSE),
    ('AUDITED_ENTITY', 'Audited entity', FALSE),
    ('AUDIT_DELIVERABLE', 'Audit deliverable', FALSE),
    ('STANDARD_AUDITED_AGAINST', 'Standard audited against', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'regulatory_affairs' AS ASSIGNED_L2_ID,
    'Regulatory Affairs' AS ASSIGNED_L2_LABEL,
    'auditing_compliance' AS ASSIGNED_L3_ID,
    'Auditing & Compliance' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Regulatory Affairs > Auditing & Compliance
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE DYNAMIC TABLE SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_TOXICOLOGY_SAFETY_ASSESSMENT_STAGE
  TARGET_LAG = '7 days'
  WAREHOUSE = {{WAREHOUSE}}
AS
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("REGULATORY_SUBMISSION_SUPPORTED") AS "REGULATORY_SUBMISSION_SUPPORTED",
    COUNT("SEND_DATASETS") AS "SEND_DATASETS",
    COUNT("STUDY_START_LEAD_TIME") AS "STUDY_START_LEAD_TIME",
    COUNT("TIME_TO_FINAL_REPORT") AS "TIME_TO_FINAL_REPORT",
    COUNT("STUDY_TYPE") AS "STUDY_TYPE",
    COUNT("ROUTE_OF_ADMINISTRATION") AS "ROUTE_OF_ADMINISTRATION",
    COUNT("ENDPOINTS") AS "ENDPOINTS",
    COUNT("DOSING_DURATION") AS "DOSING_DURATION",
    COUNT("DOSE_FORMULATION_ANALYSIS") AS "DOSE_FORMULATION_ANALYSIS",
    COUNT("NHP_SPECIES") AS "NHP_SPECIES",
    COUNT("INFUSION_MODE") AS "INFUSION_MODE",
    COUNT("BOARD_CERTIFIED_PATHOLOGISTS") AS "BOARD_CERTIFIED_PATHOLOGISTS",
    COUNT("GUIDELINES_FOLLOWED") AS "GUIDELINES_FOLLOWED",
    COUNT("SEND_VERSION") AS "SEND_VERSION",
    COUNT("REPORT_FORMAT") AS "REPORT_FORMAT",
    COUNT("CARCINOGENICITY_MODEL") AS "CARCINOGENICITY_MODEL",
    COUNT("TK_INCLUDED") AS "TK_INCLUDED",
    COUNT("RODENT_STRAIN") AS "RODENT_STRAIN",
    COUNT("REGIONS_WHERE_REPORTS_ARE_ACCEPTED") AS "REGIONS_WHERE_REPORTS_ARE_ACCEPTED"
    FROM SNOWFLAKE_LEARNING_DB."TOXICOLOGY_SAFETY_ASSESSMENT".GENERAL_TOXICOLOGY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('GLP_STATUS', 'GLP status', TRUE),
    ('SPECIES', 'Species', TRUE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORTED', 'Regulatory submission supported', FALSE),
    ('SEND_DATASETS', 'SEND datasets', FALSE),
    ('STUDY_START_LEAD_TIME', 'Study start lead time', FALSE),
    ('TIME_TO_FINAL_REPORT', 'Time to final report', FALSE),
    ('STUDY_TYPE', 'Study type', TRUE),
    ('ROUTE_OF_ADMINISTRATION', 'Route of administration', FALSE),
    ('ENDPOINTS', 'Endpoints', FALSE),
    ('DOSING_DURATION', 'Dosing duration', FALSE),
    ('DOSE_FORMULATION_ANALYSIS', 'Dose formulation analysis', FALSE),
    ('NHP_SPECIES', 'NHP species', FALSE),
    ('INFUSION_MODE', 'Infusion mode', FALSE),
    ('BOARD_CERTIFIED_PATHOLOGISTS', 'Board-certified pathologists', FALSE),
    ('GUIDELINES_FOLLOWED', 'Guidelines followed', FALSE),
    ('SEND_VERSION', 'SEND version', FALSE),
    ('REPORT_FORMAT', 'Report format', FALSE),
    ('CARCINOGENICITY_MODEL', 'Carcinogenicity model', FALSE),
    ('TK_INCLUDED', 'TK included', FALSE),
    ('RODENT_STRAIN', 'Rodent strain', FALSE),
    ('REGIONS_WHERE_REPORTS_ARE_ACCEPTED', 'Regions where reports are accepted', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'toxicology_safety_assessment' AS ASSIGNED_L2_ID,
    'Toxicology & Safety Assessment' AS ASSIGNED_L2_LABEL,
    'general_toxicology' AS ASSIGNED_L3_ID,
    'General Toxicology' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Toxicology & Safety Assessment > General Toxicology
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("REGULATORY_SUBMISSION_SUPPORTED") AS "REGULATORY_SUBMISSION_SUPPORTED",
    COUNT("SEND_DATASETS") AS "SEND_DATASETS",
    COUNT("STUDY_START_LEAD_TIME") AS "STUDY_START_LEAD_TIME",
    COUNT("TIME_TO_FINAL_REPORT") AS "TIME_TO_FINAL_REPORT",
    COUNT("CORE_BATTERY_SYSTEM") AS "CORE_BATTERY_SYSTEM",
    COUNT("TELEMETRY_TYPE") AS "TELEMETRY_TYPE",
    COUNT("CNS_METHOD") AS "CNS_METHOD",
    COUNT("RESPIRATORY_METHOD") AS "RESPIRATORY_METHOD",
    COUNT("SUPPLEMENTAL_SYSTEM") AS "SUPPLEMENTAL_SYSTEM",
    COUNT("CV_MEASUREMENT") AS "CV_MEASUREMENT",
    COUNT("GUIDELINES_FOLLOWED") AS "GUIDELINES_FOLLOWED",
    COUNT("IN_VITRO_HERG_ION_CHANNEL") AS "IN_VITRO_HERG_ION_CHANNEL",
    COUNT("STUDY_DESIGN") AS "STUDY_DESIGN",
    COUNT("ION_CHANNEL_PANEL") AS "ION_CHANNEL_PANEL",
    COUNT("CIPA_ASSESSMENT") AS "CIPA_ASSESSMENT",
    COUNT("DATA_ACQUISITION_SYSTEM") AS "DATA_ACQUISITION_SYSTEM",
    COUNT("PATCH_CLAMP_MODE") AS "PATCH_CLAMP_MODE"
    FROM SNOWFLAKE_LEARNING_DB."TOXICOLOGY_SAFETY_ASSESSMENT".SAFETY_PHARMACOLOGY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('GLP_STATUS', 'GLP status', TRUE),
    ('SPECIES', 'Species', TRUE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORTED', 'Regulatory submission supported', FALSE),
    ('SEND_DATASETS', 'SEND datasets', FALSE),
    ('STUDY_START_LEAD_TIME', 'Study start lead time', FALSE),
    ('TIME_TO_FINAL_REPORT', 'Time to final report', FALSE),
    ('CORE_BATTERY_SYSTEM', 'Core battery system', TRUE),
    ('TELEMETRY_TYPE', 'Telemetry type', FALSE),
    ('CNS_METHOD', 'CNS method', FALSE),
    ('RESPIRATORY_METHOD', 'Respiratory method', FALSE),
    ('SUPPLEMENTAL_SYSTEM', 'Supplemental system', FALSE),
    ('CV_MEASUREMENT', 'CV measurement', FALSE),
    ('GUIDELINES_FOLLOWED', 'Guidelines followed', FALSE),
    ('IN_VITRO_HERG_ION_CHANNEL', 'In vitro hERG / ion channel', FALSE),
    ('STUDY_DESIGN', 'Study design', FALSE),
    ('ION_CHANNEL_PANEL', 'Ion channel panel', FALSE),
    ('CIPA_ASSESSMENT', 'CiPA assessment', FALSE),
    ('DATA_ACQUISITION_SYSTEM', 'Data acquisition system', FALSE),
    ('PATCH_CLAMP_MODE', 'Patch clamp mode', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'toxicology_safety_assessment' AS ASSIGNED_L2_ID,
    'Toxicology & Safety Assessment' AS ASSIGNED_L2_LABEL,
    'safety_pharmacology' AS ASSIGNED_L3_ID,
    'Safety Pharmacology' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Toxicology & Safety Assessment > Safety Pharmacology
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("REGULATORY_SUBMISSION_SUPPORTED") AS "REGULATORY_SUBMISSION_SUPPORTED",
    COUNT("SEND_DATASETS") AS "SEND_DATASETS",
    COUNT("STUDY_START_LEAD_TIME") AS "STUDY_START_LEAD_TIME",
    COUNT("TIME_TO_FINAL_REPORT") AS "TIME_TO_FINAL_REPORT",
    COUNT("ASSAY") AS "ASSAY",
    COUNT("TEST_SYSTEM_TYPE") AS "TEST_SYSTEM_TYPE",
    COUNT("STUDY_TIER") AS "STUDY_TIER",
    COUNT("IN_VIVO_SPECIES") AS "IN_VIVO_SPECIES",
    COUNT("FOLLOW_UP_ASSAY") AS "FOLLOW_UP_ASSAY",
    COUNT("ICH_GUIDANCE") AS "ICH_GUIDANCE",
    COUNT("ICH_S2_BATTERY_OPTION") AS "ICH_S2_BATTERY_OPTION",
    COUNT("MAMMALIAN_CELL_LINE") AS "MAMMALIAN_CELL_LINE",
    COUNT("AMES_FORMAT") AS "AMES_FORMAT",
    COUNT("INTEGRATED_INTO_TOX_STUDY") AS "INTEGRATED_INTO_TOX_STUDY",
    COUNT("OECD_TEST_GUIDELINE") AS "OECD_TEST_GUIDELINE",
    COUNT("ENHANCED_AMES_FOR_NITROSAMINES") AS "ENHANCED_AMES_FOR_NITROSAMINES",
    COUNT("MICRONUCLEUS_SCORING_METHOD") AS "MICRONUCLEUS_SCORING_METHOD",
    COUNT("IN_SILICO_Q_SAR") AS "IN_SILICO_Q_SAR"
    FROM SNOWFLAKE_LEARNING_DB."TOXICOLOGY_SAFETY_ASSESSMENT".GENETIC_TOXICOLOGY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('GLP_STATUS', 'GLP status', TRUE),
    ('SPECIES', 'Species', TRUE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORTED', 'Regulatory submission supported', FALSE),
    ('SEND_DATASETS', 'SEND datasets', FALSE),
    ('STUDY_START_LEAD_TIME', 'Study start lead time', FALSE),
    ('TIME_TO_FINAL_REPORT', 'Time to final report', FALSE),
    ('ASSAY', 'Assay', TRUE),
    ('TEST_SYSTEM_TYPE', 'Test system type', FALSE),
    ('STUDY_TIER', 'Study tier', FALSE),
    ('IN_VIVO_SPECIES', 'In vivo species', FALSE),
    ('FOLLOW_UP_ASSAY', 'Follow-up assay', FALSE),
    ('ICH_GUIDANCE', 'ICH guidance', FALSE),
    ('ICH_S2_BATTERY_OPTION', 'ICH S2 battery option', FALSE),
    ('MAMMALIAN_CELL_LINE', 'Mammalian cell line', FALSE),
    ('AMES_FORMAT', 'Ames format', FALSE),
    ('INTEGRATED_INTO_TOX_STUDY', 'Integrated into tox study', FALSE),
    ('OECD_TEST_GUIDELINE', 'OECD test guideline', FALSE),
    ('ENHANCED_AMES_FOR_NITROSAMINES', 'Enhanced Ames for nitrosamines', FALSE),
    ('MICRONUCLEUS_SCORING_METHOD', 'Micronucleus scoring method', FALSE),
    ('IN_SILICO_Q_SAR', 'In silico (Q)SAR', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'toxicology_safety_assessment' AS ASSIGNED_L2_ID,
    'Toxicology & Safety Assessment' AS ASSIGNED_L2_LABEL,
    'genetic_toxicology' AS ASSIGNED_L3_ID,
    'Genetic Toxicology' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Toxicology & Safety Assessment > Genetic Toxicology
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("REGULATORY_SUBMISSION_SUPPORTED") AS "REGULATORY_SUBMISSION_SUPPORTED",
    COUNT("SEND_DATASETS") AS "SEND_DATASETS",
    COUNT("STUDY_START_LEAD_TIME") AS "STUDY_START_LEAD_TIME",
    COUNT("TIME_TO_FINAL_REPORT") AS "TIME_TO_FINAL_REPORT",
    COUNT("BIOANALYSIS_PLATFORM") AS "BIOANALYSIS_PLATFORM",
    COUNT("SAMPLE_MATRIX") AS "SAMPLE_MATRIX",
    COUNT("SAMPLING_TECHNIQUE") AS "SAMPLING_TECHNIQUE",
    COUNT("ADA_IMMUNOGENICITY") AS "ADA_IMMUNOGENICITY",
    COUNT("RUN_ALONGSIDE_TOX_STUDY") AS "RUN_ALONGSIDE_TOX_STUDY",
    COUNT("VALIDATION_GUIDANCE") AS "VALIDATION_GUIDANCE",
    COUNT("PK_ANALYSIS_SOFTWARE") AS "PK_ANALYSIS_SOFTWARE",
    COUNT("21_CFR_PART_11_SYSTEMS") AS "21_CFR_PART_11_SYSTEMS",
    COUNT("METHOD_VALIDATION_LEAD_TIME") AS "METHOD_VALIDATION_LEAD_TIME"
    FROM SNOWFLAKE_LEARNING_DB."TOXICOLOGY_SAFETY_ASSESSMENT".TOXICOKINETICS_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('GLP_STATUS', 'GLP status', TRUE),
    ('SPECIES', 'Species', TRUE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORTED', 'Regulatory submission supported', FALSE),
    ('SEND_DATASETS', 'SEND datasets', FALSE),
    ('STUDY_START_LEAD_TIME', 'Study start lead time', FALSE),
    ('TIME_TO_FINAL_REPORT', 'Time to final report', FALSE),
    ('BIOANALYSIS_PLATFORM', 'Bioanalysis platform', TRUE),
    ('SAMPLE_MATRIX', 'Sample matrix', FALSE),
    ('SAMPLING_TECHNIQUE', 'Sampling technique', FALSE),
    ('ADA_IMMUNOGENICITY', 'ADA / immunogenicity', FALSE),
    ('RUN_ALONGSIDE_TOX_STUDY', 'Run alongside tox study', FALSE),
    ('VALIDATION_GUIDANCE', 'Validation guidance', FALSE),
    ('PK_ANALYSIS_SOFTWARE', 'PK analysis software', FALSE),
    ('21_CFR_PART_11_SYSTEMS', '21 CFR Part 11 systems', FALSE),
    ('METHOD_VALIDATION_LEAD_TIME', 'Method validation lead time', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'toxicology_safety_assessment' AS ASSIGNED_L2_ID,
    'Toxicology & Safety Assessment' AS ASSIGNED_L2_LABEL,
    'toxicokinetics' AS ASSIGNED_L3_ID,
    'Toxicokinetics' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Toxicology & Safety Assessment > Toxicokinetics
)
UNION ALL
(
  WITH agg AS (
    SELECT
      COUNT(*) AS total_rows,
      COUNT("GLP_STATUS") AS "GLP_STATUS",
    COUNT("SPECIES") AS "SPECIES",
    COUNT("MODALITY_SUPPORTED") AS "MODALITY_SUPPORTED",
    COUNT("REGULATORY_SUBMISSION_SUPPORTED") AS "REGULATORY_SUBMISSION_SUPPORTED",
    COUNT("SEND_DATASETS") AS "SEND_DATASETS",
    COUNT("STUDY_START_LEAD_TIME") AS "STUDY_START_LEAD_TIME",
    COUNT("TIME_TO_FINAL_REPORT") AS "TIME_TO_FINAL_REPORT",
    COUNT("SPECIALTY_TYPE") AS "SPECIALTY_TYPE",
    COUNT("DART_SPECIES") AS "DART_SPECIES",
    COUNT("DART_GUIDELINE") AS "DART_GUIDELINE",
    COUNT("JUVENILE_SPECIES") AS "JUVENILE_SPECIES",
    COUNT("OCULAR_ROUTE") AS "OCULAR_ROUTE",
    COUNT("DART_STUDY_DESIGN") AS "DART_STUDY_DESIGN",
    COUNT("DART_ENDPOINTS") AS "DART_ENDPOINTS",
    COUNT("JUVENILE_BEHAVIORAL_TESTS") AS "JUVENILE_BEHAVIORAL_TESTS",
    COUNT("PHOTOTOX_ASSAY") AS "PHOTOTOX_ASSAY",
    COUNT("IN_VITRO_SKIN_OR_EYE_MODEL") AS "IN_VITRO_SKIN_OR_EYE_MODEL",
    COUNT("BIOSAFETY_CONTAINMENT") AS "BIOSAFETY_CONTAINMENT",
    COUNT("INHALATION_EXPOSURE_SYSTEM") AS "INHALATION_EXPOSURE_SYSTEM",
    COUNT("INHALATION_ATMOSPHERE") AS "INHALATION_ATMOSPHERE",
    COUNT("IMMUNOTOX_ASSAY") AS "IMMUNOTOX_ASSAY",
    COUNT("HISTORICAL_CONTROL_DATABASE") AS "HISTORICAL_CONTROL_DATABASE"
    FROM SNOWFLAKE_LEARNING_DB."TOXICOLOGY_SAFETY_ASSESSMENT".SPECIALTY_TOXICOLOGY_SPEC_STAGE
  ),
  agg_obj AS (
    -- OBJECT_CONSTRUCT(*) is only allowed directly in a SELECT clause, not as a
    -- table-function argument expression, so it has to be materialized here
    -- first and flattened via its own column below, not inline. (UNPIVOT would
    -- be the more obvious tool for this, but Snowflake explicitly disallows
    -- UNPIVOT inside a dynamic table definition — "not supported in Dynamic
    -- Table definition" — so this FLATTEN-based pivot is the one that's
    -- actually usable here, not just a stylistic choice.)
    SELECT total_rows, OBJECT_CONSTRUCT(*) AS obj
    FROM agg
  ),
  req AS (
    SELECT * FROM VALUES
    ('GLP_STATUS', 'GLP status', TRUE),
    ('SPECIES', 'Species', TRUE),
    ('MODALITY_SUPPORTED', 'Modality supported', FALSE),
    ('REGULATORY_SUBMISSION_SUPPORTED', 'Regulatory submission supported', FALSE),
    ('SEND_DATASETS', 'SEND datasets', FALSE),
    ('STUDY_START_LEAD_TIME', 'Study start lead time', FALSE),
    ('TIME_TO_FINAL_REPORT', 'Time to final report', FALSE),
    ('SPECIALTY_TYPE', 'Specialty type', TRUE),
    ('DART_SPECIES', 'DART species', FALSE),
    ('DART_GUIDELINE', 'DART guideline', FALSE),
    ('JUVENILE_SPECIES', 'Juvenile species', FALSE),
    ('OCULAR_ROUTE', 'Ocular route', FALSE),
    ('DART_STUDY_DESIGN', 'DART study design', FALSE),
    ('DART_ENDPOINTS', 'DART endpoints', FALSE),
    ('JUVENILE_BEHAVIORAL_TESTS', 'Juvenile behavioral tests', FALSE),
    ('PHOTOTOX_ASSAY', 'Phototox assay', FALSE),
    ('IN_VITRO_SKIN_OR_EYE_MODEL', 'In vitro skin or eye model', FALSE),
    ('BIOSAFETY_CONTAINMENT', 'Biosafety containment', FALSE),
    ('INHALATION_EXPOSURE_SYSTEM', 'Inhalation exposure system', FALSE),
    ('INHALATION_ATMOSPHERE', 'Inhalation atmosphere', FALSE),
    ('IMMUNOTOX_ASSAY', 'Immunotox assay', FALSE),
    ('HISTORICAL_CONTROL_DATABASE', 'Historical control database', FALSE)
    AS t(col_name, spec_label, required)
  ),
  flat AS (
    SELECT agg_obj.total_rows AS total_rows, f.key AS col_name, f.value::NUMBER AS nonnull_count
    FROM agg_obj, LATERAL FLATTEN(input => agg_obj.obj) f
  )
  SELECT
    'toxicology_safety_assessment' AS ASSIGNED_L2_ID,
    'Toxicology & Safety Assessment' AS ASSIGNED_L2_LABEL,
    'specialty_toxicology' AS ASSIGNED_L3_ID,
    'Specialty Toxicology' AS ASSIGNED_L3_LABEL,
    req.spec_label AS SPECIFICATION,
    IFF(flat.total_rows = 0, 0, flat.nonnull_count / flat.total_rows * 100) AS COVERAGE_PCT,
    req.required AS REQUIRED,
    'STAGE' AS SOURCE
  FROM flat
  JOIN req ON req.col_name = flat.col_name
  -- Toxicology & Safety Assessment > Specialty Toxicology
)
;
-- @@STATEMENT_BREAK@@
CREATE OR REPLACE VIEW SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SUMMARY AS
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_ANALYTICAL_REFERENCE_STANDARDS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_ANIMAL_BIOSPECIMENS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_ANIMAL_MODELS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_ANTIBODIES_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_BIOANALYSIS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_BIOMARKER_COMPANION_DIAGNOSTICS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_CELL_TISSUE_CULTURE_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_CENTRAL_LAB_SERVICES_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_CHEMICALS_SOLVENTS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_CLINICAL_TRIAL_SERVICES_CRO_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_CMC_ANALYTICAL_QC_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_DMPK_ADME_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_EQUIPMENT_INSTRUMENTS_PARTS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_HEOR_RWE_MARKET_ACCESS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_HISTOLOGY_PATHOLOGY_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_HUMAN_BIOSPECIMENS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_KITS_ASSAYS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_LAB_SUPPLIES_CONSUMABLES_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_MEDICAL_AFFAIRS_MEDICAL_COMMUNICATIONS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_MOLECULAR_BIOLOGY_REAGENTS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_PHARMACOVIGILANCE_DRUG_SAFETY_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_PHASE_IV_POST_MARKETING_STUDIES_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_PROTEINS_PEPTIDES_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_REGULATORY_AFFAIRS_STAGE
UNION ALL
SELECT * FROM SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.SPECIFICATION_COVERAGE_SHARD_TOXICOLOGY_SAFETY_ASSESSMENT_STAGE
;
