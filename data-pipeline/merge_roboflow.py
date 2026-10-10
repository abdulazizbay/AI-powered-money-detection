"""Merge Roboflow datasets into one YOLO dataset with our 8 unified classes.

Class names differ per dataset ("100won", "10000", "오만원", ...), so each is mapped by value.
Images containing a box we can't map (e.g. 1 won, 5 won) are dropped, so no object is left unlabeled.
Near-duplicate images across datasets are skipped with an average hash.
"""
import json
import re
import shutil
from collections import Counter

import cv2
import yaml

from config import CLASS_BY_VALUE, CLASSES, MERGED, RAW_ROBOFLOW, ROBOFLOW_CLASS_OVERRIDES

SPLITS = {"train": "train", "valid": "val", "test": "test"}
KOREAN_VALUES = {
    "십원": 10, "오십원": 50, "백원": 100, "오백원": 500,
    "천원": 1000, "오천원": 5000, "만원": 10000, "일만원": 10000, "오만원": 50000,
}
WORD_VALUES = {"ten": 10, "fifty": 50, "hundred": 100, "fivehundred": 500,
               "thousand": 1000, "fivethousand": 5000, "tenthousand": 10000, "fiftythousand": 50000}


def map_name(name):
    """Return unified class id for a dataset class name, or None."""
    n = name.strip().lower().replace(",", "").replace(" ", "").replace("_", "").replace("-", "")
    for k, v in sorted(KOREAN_VALUES.items(), key=lambda kv: -len(kv[0])):
        if k in n:
            return CLASS_BY_VALUE[v]
    digits = re.findall(r"\d+", n)
    if digits:
        return CLASS_BY_VALUE.get(int(digits[0]))
    for k, v in sorted(WORD_VALUES.items(), key=lambda kv: -len(kv[0])):
        if k in n:
            return CLASS_BY_VALUE[v]
    return None


def ahash(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).tobytes()


def main():
    if MERGED.exists():
        shutil.rmtree(MERGED)
    seen = set()
    stats = {s: Counter() for s in SPLITS.values()}
    report = {}

    for ds_idx, ds_dir in enumerate(sorted(p for p in RAW_ROBOFLOW.iterdir() if p.is_dir())):
        names = yaml.safe_load((ds_dir / "data.yaml").read_text("utf-8"))["names"]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        override = ROBOFLOW_CLASS_OVERRIDES.get(ds_dir.name, {})
        mapping = {i: CLASSES.index(override[str(n)]) if str(n) in override else map_name(n)
                   for i, n in enumerate(names)}
        report[ds_dir.name] = {n: (CLASSES[mapping[i]] if mapping[i] is not None else None)
                               for i, n in enumerate(names)}
        kept = dropped = dupes = 0

        for src_split, dst_split in SPLITS.items():
            img_dir = ds_dir / src_split / "images"
            if not img_dir.exists():
                continue
            out_img = MERGED / dst_split / "images"
            out_lbl = MERGED / dst_split / "labels"
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            for img in sorted(img_dir.iterdir()):
                lbl = ds_dir / src_split / "labels" / f"{img.stem}.txt"
                lines = lbl.read_text().split("\n") if lbl.exists() else []
                rows = [l.split() for l in lines if l.strip()]
                # polygon labels (seg exports) -> skip image, we only use boxes
                if any(len(r) != 5 for r in rows) or any(mapping.get(int(r[0])) is None for r in rows):
                    dropped += 1
                    continue
                h = ahash(img)
                if h is None or h in seen:
                    dupes += 1
                    continue
                seen.add(h)
                name = f"rf{ds_idx}_{img.stem}"
                shutil.copy2(img, out_img / f"{name}{img.suffix.lower()}")
                out = [f"{mapping[int(r[0])]} {' '.join(r[1:])}" for r in rows]
                (out_lbl / f"{name}.txt").write_text("\n".join(out))
                for r in rows:
                    stats[dst_split][CLASSES[mapping[int(r[0])]]] += 1
                kept += 1
        print(f"{ds_dir.name}: kept {kept}, dropped {dropped} (unmapped/polygon), duplicates {dupes}")

    summary = {"class_mapping": report, "boxes_per_split": {s: dict(c) for s, c in stats.items()}}
    (MERGED / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
