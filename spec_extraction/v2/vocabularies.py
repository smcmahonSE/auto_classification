"""Controlled vocabularies for the v2 (L1/L2/L3) spec model, referenced by
`vocabulary_ref` in schemas/spec_model_v2/**.json. Reuses the shared vocabularies
in spec_extraction.extractors.standards wherever an existing one already matches
(material, color, sterility) rather than duplicating them, and defines new ones
here only for fields the old flat model never needed.

Each vocabulary is a dict of normalized_value -> tuple of lowercase aliases to
match in listing text, in the same shape vocabulary_matches()/first_vocabulary_match()
in spec_extraction.extractors.common already expect.
"""

from __future__ import annotations

from spec_extraction.extractors.standards import COLORS, MATERIALS, STERILITY

VOCABULARIES: dict[str, dict[str, tuple[str, ...]]] = {
    "sterility": STERILITY,
    "material": MATERIALS,
    "color": COLORS,

    "sterilization_method": {
        "Gamma irradiated": ("gamma irradiated", "gamma-irradiated", "gamma sterilized", "gamma sterilised"),
        "EO gas": ("eo gas", "ethylene oxide", "eto sterilized", "eto-sterilized"),
        "E-beam": ("e-beam", "e beam", "electron beam"),
        "Autoclavable": ("autoclavable", "autoclaveable"),
        "Not sterilized": ("not sterilized", "not sterilised", "non-sterilized"),
    },
    "closure_type": {
        "Screw cap": ("screw cap", "screw-cap", "screw top"),
        "Snap cap": ("snap cap", "snap-cap"),
        "Dropper": ("dropper", "dropper cap", "dropper top"),
        "Pump / spray": ("pump spray", "pump", "spray top", "spray pump"),
        "Dispensing": ("dispensing cap", "dispensing closure"),
        "None": ("no closure", "no cap"),
    },
    "opacity": {
        "Clear": ("clear",),
        "Amber": ("amber",),
        "Opaque": ("opaque",),
    },
    "purity_certs": {
        "RNase-free": ("rnase-free", "rnase free"),
        "DNase-free": ("dnase-free", "dnase free"),
        "DNA-free": ("dna-free", "dna free"),
        "Endotoxin-free": ("endotoxin-free", "endotoxin free", "pyrogen-free dna-free"),
        "Pyrogen-free": ("pyrogen-free", "pyrogen free"),
        "Human DNA-free": ("human dna-free", "human dna free"),
        "None": ("not certified",),
    },
    "glass_type": {
        "Borosilicate 3.3": ("borosilicate 3.3", "boro 3.3", "pyrex", "kimax"),
        "Borosilicate 5.0": ("borosilicate 5.0", "boro 5.0"),
        "Soda-lime": ("soda-lime", "soda lime"),
        "Quartz": ("quartz",),
    },
    "graduation_class": {
        "Class A": ("class a",),
        "Class B": ("class b",),
        "Ungraduated": ("ungraduated", "non-graduated"),
    },
    "cap_type": {
        "Screw": ("screw cap", "screw-cap", "screw top"),
        "Snap": ("snap cap", "snap-cap"),
        "Plug-seal": ("plug-seal", "plug seal"),
        "Flip-top": ("flip-top", "flip top"),
        "None": ("no cap",),
    },
    "bottom_shape": {
        "Conical": ("conical bottom", "conical"),
        "Round": ("round bottom", "round-bottom"),
        "Flat": ("flat bottom", "flat-bottom"),
        "Skirted": ("skirted",),
    },
    "graduations": {
        "Graduated": ("graduated",),
        "Non-graduated": ("non-graduated", "ungraduated", "not graduated"),
    },
    "low_binding": {
        "DNA low-binding": ("dna low-binding", "dna low binding", "dna lobind"),
        "Protein low-binding": ("protein low-binding", "protein low binding", "protein lobind"),
        "Standard": ("standard binding",),
    },
    "well_bottom": {
        "Flat": ("flat bottom", "flat-bottom"),
        "Round (U)": ("round bottom", "u-bottom", "u bottom", "round (u)"),
        "V": ("v-bottom", "v bottom"),
        "Conical": ("conical bottom", "conical"),
    },
    "surface_treatment_microplate": {
        "TC-treated": ("tc-treated", "tc treated", "tissue culture treated", "tissue-culture-treated"),
        "Non-treated": ("non-treated", "non treated", "untreated"),
        "Poly-D-lysine": ("poly-d-lysine", "poly d lysine", "pdl-coated", "pdl coated"),
        "Collagen-coated": ("collagen-coated", "collagen coated"),
        "Ultra-low attachment": ("ultra-low attachment", "ultra low attachment", "ula"),
        "High-binding": ("high-binding", "high binding"),
    },
    "color_microplate": {
        "Clear": ("clear",),
        "Black": ("black",),
        "White": ("white",),
        "Black / clear bottom": ("black / clear bottom", "black with clear bottom", "black/clear bottom"),
        "White / clear bottom": ("white / clear bottom", "white with clear bottom", "white/clear bottom"),
    },
    "lid": {
        "With lid": ("with lid", "lidded"),
        "No lid": ("no lid", "lidless"),
    },
    "optical_assay_readout": {
        "UV-transparent": ("uv-transparent", "uv transparent"),
        "Low-autofluorescence": ("low-autofluorescence", "low autofluorescence"),
        "Opaque": ("opaque",),
        "Standard": ("standard optical",),
    },
    "filtered_nonfiltered": {
        "Filtered": ("filtered",),
        "Non-filtered": ("non-filtered", "non filtered", "unfiltered"),
    },
    "rack_format": {
        "Racked": ("racked", "rack format", "pre-racked"),
        "Bulk / bag": ("bulk", "bulk/bag", "bulk bag", "in bag"),
        "Refill / reload": ("refill", "reload", "refill/reload"),
        "Stacked": ("stacked",),
    },
    "low_retention": {
        "Low-retention": ("low-retention", "low retention"),
        "Standard": ("standard retention",),
    },
    "pipette_fit": {
        "Universal": ("universal fit", "universal"),
        "Rainin LTS": ("rainin lts", "lts"),
        "Eppendorf": ("eppendorf fit", "eppendorf"),
        "Gilson": ("gilson fit", "gilson"),
    },
    "pore_size": {
        "0.1 µm": ("0.1 um", "0.1µm", "0.1 micron"),
        "0.2 µm": ("0.2 um", "0.2µm", "0.2 micron"),
        "0.22 µm": ("0.22 um", "0.22µm", "0.22 micron"),
        "0.45 µm": ("0.45 um", "0.45µm", "0.45 micron"),
        "0.8 µm": ("0.8 um", "0.8µm", "0.8 micron"),
        "1.0 µm": ("1.0 um", "1 um", "1.0µm", "1 micron"),
        "5 µm": ("5 um", "5µm", "5 micron"),
    },
    "membrane_material": {
        "PES": ("pes membrane", "pes"),
        "PTFE": ("ptfe membrane", "ptfe", "teflon"),
        "Nylon": ("nylon membrane", "nylon"),
        "Cellulose acetate": ("cellulose acetate",),
        "PVDF": ("pvdf membrane", "pvdf"),
        "Glass fiber": ("glass fiber", "glass-fiber"),
        "MCE": ("mce membrane", "mixed cellulose ester", "mce"),
        "Regenerated cellulose": ("regenerated cellulose",),
    },
    "housing_material": {
        "Polypropylene": ("polypropylene",),
        "Acrylic": ("acrylic",),
        "Polycarbonate": ("polycarbonate",),
    },
    "luer_connection": {
        "Female luer-lock in / male slip out": ("female luer-lock in", "female luer lock in / male slip out"),
        "Luer-lock both ends": ("luer-lock both ends", "luer lock both ends"),
        "Luer-slip both ends": ("luer-slip both ends", "luer slip both ends"),
    },
    "luer_type": {
        "Luer-lock": ("luer-lock", "luer lock"),
        "Luer-slip": ("luer-slip", "luer slip"),
    },
    "ppe_type": {
        "Gloves": ("gloves",),
        "Lab coat": ("lab coat", "labcoat"),
        "Goggles": ("goggles",),
        "Safety glasses": ("safety glasses",),
        "Face shield": ("face shield",),
        "Respirator": ("respirator",),
        "Sleeves": ("sleeves", "arm sleeves"),
        "Shoe covers": ("shoe covers", "boot covers"),
    },
    "ppe_material": {
        "Nitrile": ("nitrile",),
        "Latex": ("latex",),
        "Vinyl": ("vinyl",),
        "Neoprene": ("neoprene",),
        "Polypropylene": ("polypropylene",),
    },
    "apparel_size": {
        "XS": ("xs", "extra small"),
        "S": ("small",),
        "M": ("medium",),
        "L": ("large",),
        "XL": ("xl", "extra large"),
        "XXL": ("xxl", "2xl", "extra extra large"),
    },
    "powder_free": {
        "Powder-free": ("powder-free", "powder free"),
        "Powdered": ("powdered",),
    },
    "tubing_material": {
        "PVC": ("pvc",),
        "Silicone": ("silicone",),
        "Natural rubber": ("natural rubber",),
        "Latex": ("latex",),
        "TPE": ("tpe", "thermoplastic elastomer"),
        "FKM": ("fkm", "viton"),
        "PTFE": ("ptfe", "teflon"),
        "Polyolefin elastomer": ("polyolefin elastomer",),
    },
    "tubing_type": {
        "Laboratory": ("laboratory tubing", "lab tubing"),
        "Pressure & vacuum": ("pressure & vacuum", "pressure and vacuum", "vacuum tubing"),
        "Peristaltic pump": ("peristaltic pump", "peristaltic"),
    },
    "cleaning_product_type": {
        "Sporicide": ("sporicide", "sporicidal"),
        "Sterilant": ("sterilant",),
        "Disinfectant": ("disinfectant",),
        "Sanitizer": ("sanitizer", "sanitiser"),
        "Detergent": ("detergent",),
    },
    "cleaning_format": {
        "Concentrate": ("concentrate", "concentrated"),
        "Ready-to-use": ("ready-to-use", "ready to use", "rtu"),
    },
    "efficacy": {
        "Bacteria": ("bacteria", "bactericidal"),
        "Fungi": ("fungi", "fungicidal"),
        "Viruses": ("viruses", "virucidal"),
        "Spores": ("spores", "sporicidal"),
    },
    "epa_registered": {
        "Yes": ("epa registered", "epa-registered"),
        "No": ("not epa registered", "non-epa registered"),
    },
    "use_environment": {
        "Cleanroom / critical area": ("cleanroom", "critical area"),
        "General lab": ("general lab", "general laboratory"),
    },
    "labeling_use": {
        "General purpose": ("general purpose",),
        "Cryogenic": ("cryogenic", "cryo"),
        "Freezer": ("freezer",),
        "Autoclave / oven": ("autoclave", "oven"),
        "Incubator": ("incubator",),
    },
    "writable": {
        "Yes": ("writable", "write-on"),
        "No": ("non-writable", "not writable"),
    },
    "diet_form": {
        "Pellet": ("pellet", "pelleted"),
        "Powder / meal": ("powder", "meal"),
        "Gel": ("gel diet", "gel"),
        "Liquid": ("liquid diet", "liquid"),
    },
    "diet_sterilization": {
        "Autoclavable": ("autoclavable", "autoclaved"),
        "Irradiated": ("irradiated",),
        "None": ("non-sterilized", "not sterilized"),
    },
    "target_species_diet": {
        "Mouse": ("mouse", "mice"),
        "Rat": ("rat", "rats"),
        "Rodents (general)": ("rodent", "rodents"),
        "Rabbit": ("rabbit", "rabbits"),
        "Guinea pig": ("guinea pig",),
    },
    "life_stage": {
        "Maintenance": ("maintenance diet", "maintenance"),
        "Growth & reproduction": ("growth & reproduction", "growth and reproduction", "gestation", "lactation", "breeding"),
        "All life stages": ("all life stages",),
    },
    "certified_diet": {
        "Yes": ("certified diet", "certified"),
        "No": ("non-certified", "not certified"),
    },
    "microscopy_item_type": {
        "Slides": ("microscope slides", "slides", "slide"),
        "Coverslips": ("coverslips", "cover glass", "cover slip"),
        "Immersion oil": ("immersion oil",),
        "Mounting media": ("mounting media", "mounting medium"),
        "Slide storage": ("slide storage", "slide box", "slide mailer"),
    },
    "corner_style": {
        "Clipped": ("clipped corners", "clipped"),
        "Square": ("square corners", "square"),
    },
    "frosted": {
        "Frosted end": ("frosted end", "frosted"),
        "Plain": ("plain end", "plain"),
    },
    "surface_treatment_microscopy": {
        "Charged / adhesion": ("charged", "adhesion slide", "poly-l-lysine coated", "adhesion"),
        "Untreated": ("untreated",),
    },
    "currency": {
        "USD": ("usd", "$", "us dollar", "us dollars"),
        "EUR": ("eur", "€", "euro", "euros"),
        "GBP": ("gbp", "£", "british pound"),
        "CHF": ("chf", "swiss franc"),
        "JPY": ("jpy", "¥", "japanese yen"),
        "CNY": ("cny", "rmb", "chinese yuan"),
        "CAD": ("cad", "canadian dollar"),
        "AUD": ("aud", "australian dollar"),
    },
    "unit_of_measure": {
        "Each": ("each", "ea"),
        "Pack": ("pack", "pk"),
        "Kit": ("kit",),
        "Bottle": ("bottle",),
        "Tube": ("tube",),
        "Case": ("case", "cs"),
        "Bag": ("bag",),
        "Plate": ("plate",),
        "Set": ("set",),
        "Pair": ("pair",),
        "Bundle": ("bundle",),
        "Slide": ("slide",),
        "Gallon": ("gallon", "gal"),
        "Liter": ("liter", "litre"),
        "Grams (g)": ("grams", "gram"),
        "Milligrams (mg)": ("milligrams", "milligram"),
        "Kilograms (kg)": ("kilograms", "kilogram"),
        "Milliliters": ("milliliters", "milliliter", "millilitres"),
        "Reaction": ("reaction", "reactions", "rxn"),
        "Assay": ("assay", "assays"),
        "Test": ("test", "tests"),
        "Sample": ("sample", "samples"),
        "Prep": ("prep", "preps"),
        "Lane": ("lane", "lanes"),
        "Array": ("array", "arrays"),
    },
}
