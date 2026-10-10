"""Assemble the final YOLO dataset and zip it for Kaggle.

  train = real train + synthetic train
  val   = half of real val + synthetic val
  test  = other half of real val + real test   (real photos only, never trained on)
Images are downscaled to max 960 px (labels are normalized, so they don't change).
"""
import random
import shutil

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from config import CLASSES, DATA, DATASET, MERGED, SEED, SYNTH

MAX_SIDE = 960


def copy_pair(img_path, split):
    out_img = DATASET / split / "images" / f"{img_path.stem}.jpg"
    out_lbl = DATASET / split / "labels" / f"{img_path.stem}.txt"
    img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    shutil.copy2(img_path.parent.parent / "labels" / f"{img_path.stem}.txt", out_lbl)


def images(d):
    return sorted((d / "images").glob("*")) if (d / "images").exists() else []


def main():
    random.seed(SEED)
    if DATASET.exists():
        shutil.rmtree(DATASET)

    real_val = images(MERGED / "val")
    random.shuffle(real_val)
    half = len(real_val) // 2
    plan = {
        "train": images(MERGED / "train") + images(SYNTH / "train"),
        "val": real_val[:half] + images(SYNTH / "val"),
        "test": real_val[half:] + images(MERGED / "test"),
    }
    for split, paths in plan.items():
        (DATASET / split / "images").mkdir(parents=True)
        (DATASET / split / "labels").mkdir(parents=True)
        for p in tqdm(paths, desc=split):
            copy_pair(p, split)

    (DATASET / "data.yaml").write_text(yaml.safe_dump({
        "path": ".", "train": "train/images", "val": "val/images", "test": "test/images",
        "names": dict(enumerate(CLASSES)),
    }, sort_keys=False))
    zip_path = shutil.make_archive(str(DATA / "krw-dataset"), "zip", DATASET)
    print({s: len(p) for s, p in plan.items()}, zip_path)


if __name__ == "__main__":
    main()
