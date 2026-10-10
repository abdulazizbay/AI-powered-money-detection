"""Draw YOLO labels on images and save a grid, to check labels by eye.

Usage: python preview.py <images_dir> [--n 16] [--match TEXT] [--out preview.jpg]
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from config import CLASSES, DATA

COLORS = [(0, 140, 255), (0, 200, 0), (255, 120, 0), (200, 0, 200),
          (0, 220, 220), (255, 0, 120), (120, 255, 120), (0, 0, 255)]


def draw(img_path, size=480):
    img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    lbl = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    for line in (lbl.read_text().splitlines() if lbl.exists() else []):
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()[:5]
        c, cx, cy, bw, bh = int(c), float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        p1, p2 = (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2))
        t = max(2, w // 200)
        cv2.rectangle(img, p1, p2, COLORS[c], t)
        cv2.putText(img, CLASSES[c], (p1[0] + 4, p1[1] + 12 * t), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * t, COLORS[c], t)
    scale = size / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--match", default="")
    ap.add_argument("--out", default=str(DATA / "preview.jpg"))
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.images).iterdir() if args.match in p.name)
    random.seed(0)
    paths = random.sample(paths, min(args.n, len(paths)))
    tiles = [draw(p) for p in paths]
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(args.out, np.vstack(rows))
    print(args.out)


if __name__ == "__main__":
    main()
