"""Ported logic for Lab Supplies and Consumables fields that need more than pure
declarative config. Reuses the proven amount+unit parsing engine from the old flat
extractors/lab_supplies.py (find_capacity_volume_size / size_spec) rather than
re-implementing it — one engine serves "Capacity / Volume / Size" (Bottles &
Containers, Tubes & Vials, Microplates, Syringes & Needles, Laboratory Glassware),
"Volume" (Pipettes & Tips), and "Diameter" (Filtration), since they're all the same
underlying amount+unit extraction problem just applied to different field labels.
"""

from __future__ import annotations

from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec
from spec_extraction.extractors.lab_supplies import find_capacity_volume_size, size_spec


def extract_capacity_volume_size(row: Mapping[str, object], field_name: str) -> ExtractedSpec:
    size_match = find_capacity_volume_size(row)
    return size_spec(size_match, field_name, "value")


# Registry the v2 dispatcher looks up by override_key (see schemas/spec_model_v2/l2/*.json)
FIELD_OVERRIDES = {
    "capacity_volume_size": extract_capacity_volume_size,
}
