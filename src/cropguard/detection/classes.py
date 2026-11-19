"""
CropGuard AI - IP102 Insect Pest Taxonomy & Class Mapping.
Contains official class definitions for the 102 insect pest categories
established in the IP102 benchmark (Wu et al., CVPR 2019).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CLASSES_TXT = PROJECT_ROOT / "data" / "raw" / "ip102" / "classes.txt"

# Official 102 IP102 benchmark classes (1-indexed in upstream benchmark)
IP102_CLASSES_CANONICAL: Dict[int, str] = {
    1: "rice leaf roller",
    2: "rice leaf caterpillar",
    3: "paddy stem maggot",
    4: "asiatic rice borer",
    5: "yellow rice borer",
    6: "rice gall midge",
    7: "Rice Stemfly",
    8: "brown plant hopper",
    9: "white backed plant hopper",
    10: "small brown plant hopper",
    11: "rice water weevil",
    12: "rice leafhopper",
    13: "grain spreader thrips",
    14: "rice shell pest",
    15: "grub",
    16: "mole cricket",
    17: "wireworm",
    18: "white margined moth",
    19: "black cutworm",
    20: "large cutworm",
    21: "yellow cutworm",
    22: "red spider",
    23: "corn borer",
    24: "army worm",
    25: "aphids",
    26: "Potosiabre vitarsis",
    27: "peach borer",
    28: "english grain aphid",
    29: "green bug",
    30: "bird cherry-oataphid",
    31: "wheat blossom midge",
    32: "penthaleus major",
    33: "longlegged spider mite",
    34: "wheat phloeothrips",
    35: "wheat sawfly",
    36: "cerodonta denticornis",
    37: "beet fly",
    38: "flea beetle",
    39: "cabbage army worm",
    40: "beet army worm",
    41: "Beet spot flies",
    42: "meadow moth",
    43: "beet weevil",
    44: "sericaorient alismots chulsky",
    45: "alfalfa weevil",
    46: "alfalfa plant bug",
    47: "tarnished plant bug",
    48: "Locustoidea",
    49: "lytta polita",
    50: "legume blister beetle",
    51: "blister beetle",
    52: "therioaphis maculata Buckton",
    53: "odontothrips loti",
    54: "Thrips",
    55: "alfalfa seed ch後ろalcid",
    56: "Pieris canidia",
    57: "Apolygus lucorum",
    58: "Limacodidae",
    59: "Viteus vitifoliae",
    60: "Colomerus vitis",
    61: "Brevipalpus lewisi McGregor",
    62: "oides decempunctata",
    63: "Polyphagotarsonemus latus",
    64: "Pseudococcus comstocki Kuwana",
    65: "paratrioza cockerelli",
    66: "minuscule pest",
    67: "green peach aphid",
    68: "drosophila melanogaster",
    69: "whitefly",
    70: "trialeurodes vaporariorum",
    71: "aphididae",
    72: "thrips palmi Karny",
    73: "panonychus citri",
    74: "phyllocnistis citrella Stainton",
    75: "tetranychus cinnabarinus",
    76: "paratetranychus pratensis",
    77: "alnetoidia alneti",
    78: "apate subopaca",
    79: "parthenolecanium corni",
    80: "unaspis yanonensis",
    81: "alceris",
    82: "pantomorus cervinus",
    83: "ceroplastes ceriferus",
    84: "cryptoblabes gnidiella",
    85: "quadraspidiotus perniciosus",
    86: "ceratitis capitata",
    87: "bactrocera cucurbitae",
    88: "bactrocera dorsalis",
    89: "tetranychus urticae",
    90: "spodoptera frugiperda",
    91: "helicoverpa armigera",
    92: "spodoptera exigua",
    93: "plutella xylostella",
    94: "bemisia tabaci",
    95: "tetranychus viennensis",
    96: "arhopala bazalus",
    97: "anarsia lineatella",
    98: "Chlumetia transversa",
    99: "Mango flat beak leafhopper",
    100: "Rhytidodera bowrinii white",
    101: "Sternochetus frigidus",
    102: "Cicadellidae",
}


def load_ip102_classes(
    classes_path: Optional[Union[str, Path]] = None,
) -> Dict[int, str]:
    """
    Loads IP102 class mapping from classes.txt if present,
    otherwise returns the canonical 102-class dictionary.
    """
    path = Path(classes_path) if classes_path else DEFAULT_CLASSES_TXT
    if path.exists():
        classes: Dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    idx = int(parts[0])
                    name = parts[1].strip()
                    classes[idx] = name
        if len(classes) == 102:
            return classes

    return dict(IP102_CLASSES_CANONICAL)


def get_ip102_class_list() -> List[str]:
    """Returns sorted list of 102 class names."""
    mapping = load_ip102_classes()
    return [mapping[i] for i in sorted(mapping.keys())]


def format_pest_name(raw_name: str) -> str:
    """Formats a raw pest category name into title case for UI presentation."""
    # Clean up whitespace and capitalize words
    words = raw_name.strip().split()
    return " ".join(word.capitalize() for word in words)
