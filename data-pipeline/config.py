from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RAW_ROBOFLOW = DATA / "raw" / "roboflow"
BACKGROUNDS = DATA / "backgrounds"
MERGED = DATA / "merged"                     # roboflow data, unified classes
SYNTH = DATA / "synth"                       # generated bill scenes
DATASET = DATA / "dataset"                   # final YOLO dataset

# Order = YOLO class id. Never reorder after training.
CLASSES = [
    "coin_10", "coin_50", "coin_100", "coin_500",
    "bill_1000", "bill_5000", "bill_10000", "bill_50000",
]
VALUES = {c: int(c.split("_")[1]) for c in CLASSES}
CLASS_BY_VALUE = {v: i for i, v in enumerate(VALUES.values())}

# Public Roboflow Universe datasets (workspace, project). Latest version is used.
ROBOFLOW_DATASETS = [
    ("yewon-v80ql", "korean-money-lkeay"),   # coins + bills, ~5.2k images, CC BY 4.0
    ("jhkimishere", "korean-coins"),          # coins, CC BY 4.0
]
# Datasets whose class names don't say the value (checked against filename prefixes).
ROBOFLOW_CLASS_OVERRIDES = {
    "korean-money-lkeay": {str(i): c for i, c in enumerate(CLASSES)},
}

# DTD textures used as synthetic backgrounds (table/cloth/paper-like surfaces).
DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_CATEGORIES = [
    "grooved", "stratified", "veined", "marbled", "woven", "fibrous", "lined",
    "matted", "porous", "wrinkled", "blotchy", "stained", "flecked", "striped",
    "chequered", "knitted", "pleated", "crosshatched", "grid", "sprinkled",
]

IMG_SIZE = 640
SEED = 42
