"""Loads the hierarchical L1/L2/L3 spec config and computes the effective field set
for a given L3 node. This is the actual runtime source of truth for the v2 model —
unlike the old schemas/l3_required_fields.json, these files are loaded and used
directly, not just documentation that has to be hand-synced with Python code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_MODEL_DIR = PROJECT_ROOT / "spec_extraction/schemas/spec_model_v2"
L1_PATH = SPEC_MODEL_DIR / "l1_universal.json"
L2_DIR = SPEC_MODEL_DIR / "l2"

# Same taxonomy file classification_pipeline/seed_anchor_tables.py reads — the
# single source of truth for which L3 node ids are real, under which L2.
TAXONOMY_L3_PATH = PROJECT_ROOT / "classification_pipeline/taxonomy/l3_taxonomy_anchors.json"


@dataclass(frozen=True)
class FieldDef:
    name: str
    type: str
    extractor: str
    required: bool = False
    product_only: bool = False
    allowed_values: Any = None
    vocabulary_ref: str | None = None
    pattern: str | None = None
    override_key: str | None = None
    source_column: str | None = None
    multi_select: bool = False
    tier: str = ""  # "l1", "l2", or "l3" — set by get_effective_field_set, not read from JSON

    @staticmethod
    def from_dict(d: dict[str, Any], tier: str) -> "FieldDef":
        return FieldDef(
            name=d["name"],
            type=d["type"],
            extractor=d["extractor"],
            required=d.get("required", False),
            product_only=d.get("product_only", False),
            allowed_values=d.get("allowed_values"),
            vocabulary_ref=d.get("vocabulary_ref"),
            pattern=d.get("pattern"),
            override_key=d.get("override_key"),
            source_column=d.get("source_column"),
            multi_select=d.get("multi_select", False),
            tier=tier,
        )


@dataclass(frozen=True)
class L3Node:
    id: str
    label: str
    form: str
    table_name: str
    fields: tuple[FieldDef, ...]


@dataclass(frozen=True)
class L2Config:
    l2_id: str
    l2_label: str
    schema_name: str
    l2_fields: tuple[FieldDef, ...]
    l3_nodes: dict[str, L3Node]


@dataclass(frozen=True)
class L1Config:
    all_listing_fields: tuple[FieldDef, ...]
    product_only_fields: tuple[FieldDef, ...]
    requester_mapping_fields: tuple[FieldDef, ...]


@lru_cache(maxsize=1)
def _load_taxonomy_l2_subcategories() -> dict[str, list[str]]:
    with open(TAXONOMY_L3_PATH) as f:
        data = json.load(f)
    return {
        l2_id: [subcat["id"] for subcat in subcats]
        for l2_id, subcats in data["l2_subcategories"].items()
    }


@lru_cache(maxsize=1)
def load_l1() -> L1Config:
    with open(L1_PATH) as f:
        data = json.load(f)
    return L1Config(
        all_listing_fields=tuple(FieldDef.from_dict(d, "l1") for d in data["all_listing_fields"]),
        product_only_fields=tuple(FieldDef.from_dict(d, "l1") for d in data["product_only_fields"]),
        requester_mapping_fields=tuple(FieldDef.from_dict(d, "l1") for d in data["requester_mapping_fields"]),
    )


@lru_cache(maxsize=None)
def load_l2(l2_id: str) -> L2Config:
    path = L2_DIR / f"{l2_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No v2 config for L2 id '{l2_id}' at {path}")

    with open(path) as f:
        data = json.load(f)

    if data["l2_id"] != l2_id:
        raise ValueError(f"Config file {path} has l2_id '{data['l2_id']}', expected '{l2_id}'")

    known_l3_ids = set(_load_taxonomy_l2_subcategories().get(l2_id, []))
    l3_nodes: dict[str, L3Node] = {}
    seen_field_names: dict[str, set[str]] = {}

    for node_id, node_data in data["l3_nodes"].items():
        if node_id not in known_l3_ids:
            raise ValueError(
                f"L3 node id '{node_id}' in {path} does not match any subcategory of "
                f"'{l2_id}' in {TAXONOMY_L3_PATH} — check for a typo or a stale taxonomy slug."
            )

        fields = tuple(FieldDef.from_dict(d, "l3") for d in node_data["fields"])
        names = [f.name for f in fields]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate field name(s) {dupes} within L3 node '{node_id}' in {path}")
        seen_field_names[node_id] = set(names)

        l3_nodes[node_id] = L3Node(
            id=node_id,
            label=node_data["label"],
            form=node_data.get("form", "product"),
            table_name=node_data["table_name"],
            fields=fields,
        )

    _validate_vocabulary_refs(data, path)

    return L2Config(
        l2_id=data["l2_id"],
        l2_label=data["l2_label"],
        schema_name=data["schema_name"],
        l2_fields=tuple(FieldDef.from_dict(d, "l2") for d in data["l2_fields"]),
        l3_nodes=l3_nodes,
    )


def _validate_vocabulary_refs(raw_l2_data: dict[str, Any], path: Path) -> None:
    from spec_extraction.v2.vocabularies import VOCABULARIES

    refs: set[str] = set()
    for field_data in raw_l2_data.get("l2_fields", []):
        if field_data.get("vocabulary_ref"):
            refs.add(field_data["vocabulary_ref"])
    for node_data in raw_l2_data.get("l3_nodes", {}).values():
        for field_data in node_data.get("fields", []):
            if field_data.get("vocabulary_ref"):
                refs.add(field_data["vocabulary_ref"])

    missing = refs - set(VOCABULARIES.keys())
    if missing:
        raise ValueError(f"Config {path} references unknown vocabulary_ref(s): {sorted(missing)}")


def get_effective_field_set(l2_id: str, l3_id: str) -> list[FieldDef]:
    """L1 all-listing + L1 product-only + L1 requester-mapping + this L2's fields +
    this L3 node's own fields, in that fixed order (stable output column order)."""
    l1 = load_l1()
    l2 = load_l2(l2_id)
    if l3_id not in l2.l3_nodes:
        raise KeyError(f"L3 node '{l3_id}' not found under L2 '{l2_id}'")
    node = l2.l3_nodes[l3_id]

    return [
        *l1.all_listing_fields,
        *l1.product_only_fields,
        *l1.requester_mapping_fields,
        *(FieldDef(**{**f.__dict__, "tier": "l2"}) for f in l2.l2_fields),
        *(FieldDef(**{**f.__dict__, "tier": "l3"}) for f in node.fields),
    ]
