"""Which L2 categories have been migrated to the new hierarchical (v2) spec model.

Every other L2/category id still goes through the old flat system in
extract_product_specs.py, unchanged. Add an id here only once its L2 config file
exists under schemas/spec_model_v2/l2/ and (if real extraction logic is wanted
beyond the config shape) its overrides module exists under v2/overrides/.
"""

from __future__ import annotations

MIGRATED_L2_IDS: set[str] = {
    "lab_supplies_consumables",
}
