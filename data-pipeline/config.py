from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RAW_ROBOFLOW = DATA / "raw" / "roboflow"
RAW_TEMPLATES = DATA / "raw" / "templates"   # downloaded bill scans
TEMPLATES = DATA / "templates"               # cropped bill templates (png)
BACKGROUNDS = DATA / "backgrounds"
REAL_TEST = DATA / "real_test"               # hand-labeled real photos (images/ + labels/)
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

# Current-series bill scans on Wikimedia Commons ({{South Korean currency}} tag).
COMMONS_TEMPLATES = {
    "bill_1000":  ["1000 won serieIII obverse.jpeg", "1000 won serieIII reverse.jpeg"],
    "bill_5000":  ["5000 won serieV obverse.jpeg", "5000 won serieV reverse.jpeg"],
    "bill_10000": ["10000 won serieVI obverse.jpeg", "10000 won serieVI reverse.jpeg"],
    "bill_50000": ["50000 KRW 2009 ob.jpg", "50000 KRW 2009 rev.jpg"],
}

# Real photos for the honest test set (labeled by hand, never trained on).
COMMONS_REAL_TEST = [
    "Currency South Korea.jpg",
    "50000 won banknote.jpg",
    "10000 won banknote.jpg",
    "오만원.jpg",
    "Won surcoreano.jpg",
    "Southkoreancoins.jpg",
    "500 yen next to 500 South Korean won coin.jpg",
    "Tiền Xu 500 won Hàn Quốc.jpg",
    "Tiền Xu 500 won Hàn Quốc (47595).jpg",
    "CORÉE DU SUD, 100 Won à l'effigie de l'Amiral Yi Sun-Shin.jpg",
    "20150214-20150219최광모C37.JPG",
]

# DTD textures used as synthetic backgrounds (table/cloth/paper-like surfaces).
DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_CATEGORIES = [
    "grooved", "stratified", "veined", "marbled", "woven", "fibrous", "lined",
    "matted", "porous", "wrinkled", "blotchy", "stained", "flecked", "striped",
    "chequered", "knitted", "pleated", "crosshatched", "grid", "sprinkled",
]

IMG_SIZE = 640
SEED = 42
