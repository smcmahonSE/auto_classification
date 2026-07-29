"""Deterministic extraction rules for Equipment, Instruments and Parts."""

from __future__ import annotations

from typing import Mapping

from spec_extraction.extractors.common import ExtractedSpec, vocabulary_matches
from spec_extraction.extractors.standards import first_vocabulary_match


EQUIPMENT_TYPES = {
    "Centrifuge": ("centrifuge", "microcentrifuge", "ultracentrifuge"),
    "Pipettor": ("pipettor", "pipette", "multichannel pipette", "electronic pipette"),
    "Liquid Handler": ("liquid handler", "liquid handling", "robotic pipetting", "plate washer", "reagent dispenser"),
    "Microscope": ("microscope", "confocal", "fluorescence microscope", "inverted microscope"),
    "Spectrophotometer": ("spectrophotometer", "uv-vis", "nanodrop"),
    "Plate Reader": ("plate reader", "microplate reader"),
    "Flow Cytometer": ("flow cytometer", "facs", "cell sorter"),
    "PCR Machine": ("pcr machine", "thermal cycler", "qpcr", "real-time pcr", "digital pcr"),
    "Electrophoresis System": ("electrophoresis", "gel system", "western blot transfer"),
    "Incubator": ("incubator", "co2 incubator", "hypoxia incubator"),
    "Shaker": ("shaker", "orbital shaker", "rocking platform", "tube rotator"),
    "Vortexer": ("vortex", "vortexer", "vortex mixer"),
    "Homogenizer": ("homogenizer", "homogeniser", "bead mill"),
    "Sonicator": ("sonicator", "ultrasonic"),
    "Freeze Dryer": ("freeze dryer", "lyophilizer", "lyophiliser"),
    "Water Bath": ("water bath", "circulator"),
    "Heating Block": ("heating block", "heat block", "dry bath"),
    "Stirrer": ("magnetic stirrer", "stirrer", "hot plate"),
    "Pump": ("pump",),
    "Cell Counter": ("cell counter", "hemocytometer", "vi-cell", "countess"),
    "Autoclave": ("autoclave", "sterilizer", "steriliser"),
    "Biosafety Cabinet": ("biosafety cabinet", "bsc", "laminar flow hood"),
    "Fume Hood": ("fume hood",),
    "Chromatography System": ("chromatography", "hplc", "uhplc", "fplc"),
    "Mass Spectrometer": ("mass spectrometer", "mass spec", "lc-ms", "gc-ms"),
    "Balance": ("balance", "analytical balance", "precision balance"),
    "pH Meter": ("ph meter",),
}

SUB_TYPES = {
    "Benchtop": ("benchtop", "bench top", "tabletop"),
    "Floor": ("floor", "floor-standing", "floor standing"),
    "Handheld": ("handheld", "hand-held"),
    "Portable": ("portable",),
    "Software": ("software", "license", "licence"),
}

PRODUCT_ROLES = {
    "Instrument": ("instrument", "system", "analyzer", "analyser", "reader", "machine"),
    "Accessory": ("accessory", "adapter", "adaptor", "rotor", "rack", "holder", "module", "attachment"),
    "Replacement Part": ("replacement part", "spare part", "part", "component", "kit"),
    "Consumable": ("consumable", "cartridge", "plate", "tube", "tip", "filter"),
}

def extract_equipment_type(row: Mapping[str, object]) -> ExtractedSpec:
    return first_vocabulary_match(row, "Equipment Type", EQUIPMENT_TYPES, "equipment_type_dictionary")


def extract_equipment_instruments_specs(row: Mapping[str, object]) -> list[ExtractedSpec]:
    """Extract SME-proposed fields for Equipment, Instruments and Parts."""
    return [
        extract_equipment_type(row),
        vocabulary_matches(row, "Sub-Type", SUB_TYPES, "equipment_subtype_dictionary"),
        vocabulary_matches(row, "Product Role", PRODUCT_ROLES, "product_role_dictionary"),
    ]
