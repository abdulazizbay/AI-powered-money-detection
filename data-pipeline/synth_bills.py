"""Generate labeled synthetic scenes: bills (and real coin cutouts) on random surfaces.

Each scene:
  background  = DTD texture or a real coin photo (its coin labels are kept)
  bills       = template -> color jitter -> crease shading -> perspective warp -> drop shadow
  coins       = elliptical cutouts from real labeled coin photos, sized relative to the bills
  camera      = lighting gradient, blur, noise, JPEG artifacts
Labels are boxes around the *visible* part of each object; mostly hidden objects are dropped.

Usage: python synth_bills.py --train 3000 --val 300
"""
import argparse
import random

import cv2
import numpy as np
from tqdm import tqdm

from config import BACKGROUNDS, CLASSES, IMG_SIZE, MERGED, SEED, SYNTH, TEMPLATES

S = IMG_SIZE
BILL_WIDTH_MM = {"bill_1000": 136, "bill_5000": 142, "bill_10000": 148, "bill_50000": 154}
COIN_DIAM_MM = {"coin_10": 18.0, "coin_50": 21.6, "coin_100": 24.0, "coin_500": 26.5}
MIN_VISIBLE = {"bill": 0.35, "coin": 0.6}


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def read_labels(img_path):
    lbl = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
    return [(int(r[0]), *map(float, r[1:5])) for r in rows]


class Assets:
    def __init__(self, split):
        self.templates = {}
        for p in sorted(TEMPLATES.glob("bill_*.png")):
            cls = "_".join(p.stem.split("_")[:2])
            self.templates.setdefault(cls, []).append(imread(p))
        if len(self.templates) != 4:
            raise SystemExit(f"need templates for 4 bills, found {sorted(self.templates)}")

        self.textures = sorted(BACKGROUNDS.glob("*.jpg"))
        # real coin photos: used as backgrounds and as a source of coin cutouts (same split only)
        coin_dir = MERGED / split / "images"
        self.coin_scenes, self.coin_cutouts = [], {c: [] for c in COIN_DIAM_MM}
        for img_path in sorted(coin_dir.glob("*")) if coin_dir.exists() else []:
            labels = read_labels(img_path)
            if labels and all(CLASSES[l[0]].startswith("coin") for l in labels):
                self.coin_scenes.append(img_path)
        for img_path in self.coin_scenes[:1500]:
            img = imread(img_path)
            h, w = img.shape[:2]
            for cls_id, cx, cy, bw, bh in read_labels(img_path):
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                if min(x2 - x1, y2 - y1) >= 40 and 0.8 < (x2 - x1) / (y2 - y1) < 1.25:
                    self.coin_cutouts[CLASSES[cls_id]].append(img[max(y1, 0):y2, max(x1, 0):x2].copy())
        print(f"[{split}] textures {len(self.textures)}, coin scenes {len(self.coin_scenes)}, "
              f"coin cutouts {[len(v) for v in self.coin_cutouts.values()]}")


def random_crop_resize(img):
    h, w = img.shape[:2]
    scale = S / min(h, w) * random.uniform(1.0, 1.4)
    img = cv2.resize(img, (max(S, int(w * scale)), max(S, int(h * scale))))
    y = random.randint(0, img.shape[0] - S)
    x = random.randint(0, img.shape[1] - S)
    return img[y:y + S, x:x + S]


def coin_scene_background(path):
    """Square-pad a real coin photo to SxS and return it with its coin objects (as masks)."""
    img = imread(path)
    h, w = img.shape[:2]
    scale = S / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = cv2.resize(img, (S, S))  # stretched copy as filler behind the letterbox
    canvas = cv2.GaussianBlur(canvas, (51, 51), 0)
    oy, ox = (S - img.shape[0]) // 2, (S - img.shape[1]) // 2
    canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    objects = []
    for cls_id, cx, cy, bw, bh in read_labels(path):
        m = np.zeros((S, S), np.float32)
        center = (int(ox + cx * img.shape[1]), int(oy + cy * img.shape[0]))
        axes = (max(1, int(bw * img.shape[1] / 2)), max(1, int(bh * img.shape[0] / 2)))
        cv2.ellipse(m, center, axes, 0, 0, 360, 1.0, -1)
        objects.append({"cls": cls_id, "mask": m, "area": m.sum()})
    return canvas, objects


def jitter_color(img):
    img = img.astype(np.float32)
    img = img * random.uniform(0.7, 1.25) + random.uniform(-25, 25)          # contrast / brightness
    img = img * np.array([random.uniform(0.9, 1.1) for _ in range(3)])        # white balance
    return np.clip(img, 0, 255).astype(np.uint8)


def add_crease(img):
    """Fake a fold: darken one side of a random line and draw a soft crease highlight."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    if random.random() < 0.5:
        x = int(w * random.uniform(0.3, 0.7))
        ramp = np.linspace(1.0, random.uniform(0.75, 0.95), w - x)
        out[:, x:] *= ramp[None, :, None]
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), max(1, w // 300))
    if random.random() < 0.3:
        y = int(h * random.uniform(0.3, 0.7))
        out[y:] *= random.uniform(0.85, 0.97)
    return np.clip(out, 0, 255).astype(np.uint8)


def paste(canvas, objects, patch, mask, cls_id, shadow=True):
    """Alpha-blend a warped patch, cast a soft shadow, and occlude earlier objects."""
    if shadow:
        dx, dy = random.randint(-12, 12), random.randint(-12, 12)
        sh = np.roll(np.roll(mask, dy, 0), dx, 1)
        sh = cv2.GaussianBlur(sh, (31, 31), 0) * random.uniform(0.25, 0.5)
        canvas[:] = (canvas * (1 - sh[..., None])).astype(np.uint8)
    soft = cv2.GaussianBlur(mask, (3, 3), 0)[..., None]
    canvas[:] = (canvas * (1 - soft) + patch * soft).astype(np.uint8)
    for o in objects:
        o["mask"] = o["mask"] * (1 - mask)
    objects.append({"cls": cls_id, "mask": mask.copy(), "area": None})


def warp_to_canvas(img, quad):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad))
    patch = cv2.warpPerspective(img, M, (S, S), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (S, S))
    return patch, mask


def bill_quad(width, aspect, center, angle, persp):
    bw, bh = width, width / aspect
    pts = np.array([[-bw / 2, -bh / 2], [bw / 2, -bh / 2], [bw / 2, bh / 2], [-bw / 2, bh / 2]])
    pts += np.random.uniform(-persp, persp, pts.shape) * bw
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return pts @ R.T + center


def camera_effects(img):
    img = img.astype(np.float32)
    # lighting gradient
    gx, gy = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S))
    a = np.random.uniform(-1, 1, 2)
    light = 1 + random.uniform(0, 0.35) * ((gx * a[0] + gy * a[1]) - (a.sum() / 2))
    img *= light[..., None]
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.4:
        img += np.random.normal(0, random.uniform(2, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = random.randint(45, 95)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def make_scene(assets):
    if assets.coin_scenes and random.random() < 0.4:
        canvas, objects = coin_scene_background(random.choice(assets.coin_scenes))
    else:
        canvas, objects = random_crop_resize(imread(random.choice(assets.textures))), []
    for o in objects:
        o["area"] = o["mask"].sum()

    n_bills = random.choices([1, 2, 3, 4, 5], weights=[30, 30, 20, 12, 8])[0]
    base = S * random.uniform(0.35, 0.8) / (1 + 0.25 * (n_bills - 1))  # width of a 50,000 won bill
    spread = random.uniform(0.15, 0.35) * S
    anchor = np.array([random.uniform(0.3, 0.7) * S, random.uniform(0.3, 0.7) * S])
    fan_angle = random.uniform(0, 360)

    for i in range(n_bills):
        cls = random.choice(list(BILL_WIDTH_MM))
        tpl = random.choice(assets.templates[cls])
        tpl = add_crease(jitter_color(tpl))
        width = base * BILL_WIDTH_MM[cls] / 154
        center = anchor + np.random.uniform(-spread, spread, 2)
        angle = fan_angle + random.uniform(-40, 40) if random.random() < 0.6 else random.uniform(0, 360)
        quad = bill_quad(width, tpl.shape[1] / tpl.shape[0], center, angle, random.uniform(0, 0.08))
        patch, mask = warp_to_canvas(tpl, quad)
        area = mask.sum()
        paste(canvas, objects, patch, mask, CLASSES.index(cls))
        objects[-1]["area"] = area

    # real coins on top of the bills, sized relative to them
    if any(assets.coin_cutouts.values()) and random.random() < 0.6:
        for _ in range(random.randint(1, 6)):
            cls = random.choice([c for c, v in assets.coin_cutouts.items() if v])
            crop = jitter_color(random.choice(assets.coin_cutouts[cls]))
            d = base / 154 * COIN_DIAM_MM[cls] * random.uniform(0.9, 1.1)
            crop = cv2.resize(crop, (max(8, int(d)), max(8, int(d))))
            m = np.zeros(crop.shape[:2], np.float32)
            cv2.ellipse(m, (crop.shape[1] // 2, crop.shape[0] // 2),
                        (crop.shape[1] // 2 - 1, crop.shape[0] // 2 - 1), 0, 0, 360, 1.0, -1)
            x, y = random.randint(0, S - crop.shape[1]), random.randint(0, S - crop.shape[0])
            patch = np.zeros_like(canvas)
            mask = np.zeros((S, S), np.float32)
            patch[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
            mask[y:y + crop.shape[0], x:x + crop.shape[1]] = m
            area = mask.sum()
            paste(canvas, objects, patch, mask, CLASSES.index(cls), shadow=random.random() < 0.5)
            objects[-1]["area"] = area

    labels = []
    for o in objects:
        vis = o["mask"] > 0.5
        kind = CLASSES[o["cls"]].split("_")[0]
        if o["area"] <= 0 or vis.sum() / o["area"] < MIN_VISIBLE[kind]:
            continue
        ys, xs = np.where(vis)
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if (x2 - x1) < 6 or (y2 - y1) < 6:
            continue
        labels.append(f"{o['cls']} {(x1 + x2) / 2 / S:.6f} {(y1 + y2) / 2 / S:.6f} "
                      f"{(x2 - x1) / S:.6f} {(y2 - y1) / S:.6f}")
    return camera_effects(canvas), labels


def generate(split, count):
    assets = Assets(split)
    img_dir, lbl_dir = SYNTH / split / "images", SYNTH / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(count), desc=split):
        img, labels = make_scene(assets)
        cv2.imwrite(str(img_dir / f"synth_{split}_{i:05d}.jpg"), img)
        (lbl_dir / f"synth_{split}_{i:05d}.txt").write_text("\n".join(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=300)
    args = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    generate("train", args.train)
    generate("val", args.val)


if __name__ == "__main__":
    main()
