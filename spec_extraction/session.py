"""Schema-aware Snowflake session helper for the v2 (per-L2-schema) spec model.

The old flat system (extract_product_specs.py) builds its own session inline via
get_snowflake_session() + set_snowflake_context(), always targeting SMCMAHON_PRODUCTS
— untouched, still works exactly as before. This module adds the one new thing the
v2 model needs: targeting (and creating, if needed) an arbitrary per-L2 schema.
It does not touch or call classification_pipeline.product_classifier_utils.get_products_session(),
which stays hardcoded to SMCMAHON_PRODUCTS for the classification pipeline's own use.
"""

from __future__ import annotations

from snowflake.snowpark import Session

from classification_pipeline.product_classifier_utils import get_snowflake_session

DATABASE = "SNOWFLAKE_LEARNING_DB"
DEFAULT_SCHEMA = "SMCMAHON_PRODUCTS"


def get_spec_session(schema: str | None = None, create_schema_if_missing: bool = False) -> Session:
    """Open a Snowflake session scoped to `schema` (default SMCMAHON_PRODUCTS,
    preserving old-system behavior when schema is omitted). Pass
    create_schema_if_missing=True for a new per-L2 schema that may not exist yet —
    CREATE SCHEMA IF NOT EXISTS is idempotent, safe to run on every invocation."""
    sf = get_snowflake_session()
    target_schema = schema or DEFAULT_SCHEMA
    sf.sql(f"USE DATABASE {DATABASE}").collect()
    if create_schema_if_missing and schema:
        sf.sql(f'CREATE SCHEMA IF NOT EXISTS "{target_schema}"').collect()
    sf.sql(f'USE SCHEMA "{target_schema}"').collect()
    return sf


def close_spec_session(sf: Session) -> None:
    try:
        sf.close()
    except Exception as exc:  # pragma: no cover - cleanup should not mask run success.
        print(f"Warning: failed to close Snowflake session cleanly: {exc}")
