"""Generate labeled multi-object scenes from real cutouts: several bills + coins together.

The Roboflow photos show one denomination at a time; the app must count mixed piles.
Each scene:
  background = a real labeled photo (its labels are kept)
  bills      = real bill crops -> color jitter -> crease shading -> perspective warp -> drop shadow
  coins      = elliptical crops of real coins, sized relative to the bills (real mm ratios)
  camera     = lighting gradient, blur, noise, JPEG artifacts
Cutouts come from the same split only (train scenes never use val photos).
Labels are boxes around the *visible* part of each object; mostly hidden objects are dropped.

Usage: python synth_bills.py --train 3000 --val 300 [--workers 12]
"""
import argparse
import pickle
import random
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm

from config import CLASSES, IMG_SIZE, MERGED, SEED, SYNTH

S = IMG_SIZE
BILL_WIDTH_MM = {"bill_1000": 136, "bill_5000": 142, "bill_10000": 148, "bill_50000": 154}
COIN_DIAM_MM = {"coin_10": 18.0, "coin_50": 21.6, "coin_100": 24.0, "coin_500": 26.5}
MIN_VISIBLE = {"bill": 0.35, "coin": 0.6}
MAX_CUTOUTS = 150  # per class


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def read_labels(img_path):
    lbl = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
    return [(int(r[0]), *map(float, r[1:5])) for r in rows]


def box_px(label, w, h):
    _, cx, cy, bw, bh = label
    return (max(int((cx - bw / 2) * w), 0), max(int((cy - bh / 2) * h), 0),
            min(int((cx + bw / 2) * w), w), min(int((cy + bh / 2) * h), h))


class Assets:
    def __init__(self, split):
        self.scenes = sorted((MERGED / split / "images").glob("*"))
        self.cutouts = {c: [] for c in CLASSES}
        for img_path in random.sample(self.scenes, len(self.scenes)):
            if all(len(v) >= MAX_CUTOUTS for v in self.cutouts.values()):
                break
            labels = read_labels(img_path)
            wanted = [l for l in labels if len(self.cutouts[CLASSES[l[0]]]) < MAX_CUTOUTS]
            if not wanted:
                continue
            img = imread(img_path)
            h, w = img.shape[:2]
            for l in wanted:
                x1, y1, x2, y2 = box_px(l, w, h)
                bw, bh = x2 - x1, y2 - y1
                if min(bw, bh) < 40:
                    continue
                crop = img[y1:y2, x1:x2]
                if CLASSES[l[0]].startswith("coin"):
                    ok = 0.8 < bw / bh < 1.25            # round, seen from above
                else:
                    if bh > bw:
                        crop = cv2.rotate(crop, random.choice([cv2.ROTATE_90_CLOCKWISE,
                                                               cv2.ROTATE_90_COUNTERCLOCKWISE]))
                    ok = 1.7 < max(bw, bh) / min(bw, bh) < 2.6  # flat, roughly axis-aligned bill
                if ok:
                    # store at the largest size a scene can use, to keep the cache small
                    limit = 160 if CLASSES[l[0]].startswith("coin") else 320
                    r = min(1.0, limit / max(crop.shape[:2]))
                    self.cutouts[CLASSES[l[0]]].append(cv2.resize(crop, None, fx=r, fy=r,
                                                                  interpolation=cv2.INTER_AREA))
        print(f"[{split}] scenes {len(self.scenes)}, cutouts "
              + ", ".join(f"{c}={len(v)}" for c, v in self.cutouts.items()))
        missing = [c for c in BILL_WIDTH_MM if not self.cutouts[c]]
        if missing:
            raise SystemExit(f"no bill cutouts for {missing}; run merge_roboflow.py first")


def scene_background(path):
    """Letterbox a real labeled photo into SxS; its objects become (ellipse/rect) masks."""
    img = imread(path)
    h, w = img.shape[:2]
    scale = S / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = cv2.GaussianBlur(cv2.resize(img, (S, S)), (51, 51), 0)
    oy, ox = (S - img.shape[0]) // 2, (S - img.shape[1]) // 2
    canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    objects = []
    for l in read_labels(path):
        x1, y1, x2, y2 = box_px(l, img.shape[1], img.shape[0])
        m = np.zeros((S, S), np.float32)
        if CLASSES[l[0]].startswith("coin"):
            cv2.ellipse(m, ((x1 + x2) // 2 + ox, (y1 + y2) // 2 + oy),
                        (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)), 0, 0, 360, 1.0, -1)
        else:
            m[y1 + oy:y2 + oy, x1 + ox:x2 + ox] = 1.0
        objects.append({"cls": l[0], "mask": m, "area": m.sum()})
    return canvas, objects


def jitter_color(img):
    img = img.astype(np.float32)
    img = img * random.uniform(0.75, 1.2) + random.uniform(-20, 20)       # contrast / brightness
    img = img * np.array([random.uniform(0.92, 1.08) for _ in range(3)])  # white balance
    return np.clip(img, 0, 255).astype(np.uint8)


def add_crease(img):
    """Fake a fold: darken one side of a random line and draw a soft crease highlight."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    if random.random() < 0.4:
        x = int(w * random.uniform(0.3, 0.7))
        out[:, x:] *= np.linspace(1.0, random.uniform(0.75, 0.95), w - x)[None, :, None]
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), max(1, w // 300))
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
    objects.append({"cls": cls_id, "mask": mask.copy(), "area": mask.sum()})


def warp_quad(img, quad):
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
    gx, gy = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S))
    a = np.random.uniform(-1, 1, 2)
    img *= (1 + random.uniform(0, 0.35) * ((gx * a[0] + gy * a[1]) - a.sum() / 2))[..., None]
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.4:
        img += np.random.normal(0, random.uniform(2, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = random.randint(45, 95)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def make_scene(assets):
    canvas, objects = scene_background(random.choice(assets.scenes))

    n_bills = random.choices([0, 1, 2, 3, 4, 5], weights=[10, 25, 25, 20, 12, 8])[0]
    base = S * random.uniform(0.45, 0.8) / (1 + 0.2 * max(n_bills - 1, 0))  # 50,000 won bill width
    spread = random.uniform(0.15, 0.35) * S
    anchor = np.array([random.uniform(0.3, 0.7) * S, random.uniform(0.3, 0.7) * S])
    fan = random.uniform(0, 360)

    for _ in range(n_bills):
        cls = random.choice(list(BILL_WIDTH_MM))
        crop = add_crease(jitter_color(random.choice(assets.cutouts[cls])))
        width = base * BILL_WIDTH_MM[cls] / 154
        center = anchor + np.random.uniform(-spread, spread, 2)
        angle = fan + random.uniform(-40, 40) if random.random() < 0.6 else random.uniform(0, 360)
        quad = bill_quad(width, crop.shape[1] / crop.shape[0], center, angle, random.uniform(0, 0.08))
        patch, mask = warp_quad(crop, quad)
        paste(canvas, objects, patch, mask, CLASSES.index(cls))

    coins = [c for c in COIN_DIAM_MM if assets.cutouts[c]]
    for _ in range(random.randint(1 if n_bills == 0 else 0, 7)):
        cls = random.choice(coins)
        crop = jitter_color(random.choice(assets.cutouts[cls]))
        d = max(20, int(base / 154 * COIN_DIAM_MM[cls] * random.uniform(0.9, 1.1)))
        crop = cv2.resize(crop, (d, d))
        m = np.zeros((d, d), np.float32)
        cv2.ellipse(m, (d // 2, d // 2), (d // 2 - 1, d // 2 - 1), 0, 0, 360, 1.0, -1)
        x, y = random.randint(0, S - d), random.randint(0, S - d)
        patch, mask = np.zeros_like(canvas), np.zeros((S, S), np.float32)
        patch[y:y + d, x:x + d] = crop
        mask[y:y + d, x:x + d] = m
        paste(canvas, objects, patch, mask, CLASSES.index(cls), shadow=random.random() < 0.5)

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


def load_assets(split):
    """Build cutouts once per split and cache them, so worker processes start fast."""
    cache = SYNTH / f"assets_{split}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    assets = Assets(split)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(assets))
    return assets


def worker(job):
    split, start, end = job
    cv2.setNumThreads(1)
    random.seed(SEED + start)
    np.random.seed(SEED + start)
    assets = load_assets(split)
    for i in range(start, end):
        img, labels = make_scene(assets)
        cv2.imwrite(str(SYNTH / split / "images" / f"synth_{split}_{i:05d}.jpg"), img)
        (SYNTH / split / "labels" / f"synth_{split}_{i:05d}.txt").write_text("\n".join(labels))
    return end - start


def generate(split, count, workers):
    random.seed(SEED)
    load_assets(split)
    (SYNTH / split / "images").mkdir(parents=True, exist_ok=True)
    (SYNTH / split / "labels").mkdir(parents=True, exist_ok=True)
    step = 50
    jobs = [(split, s, min(s + step, count)) for s in range(0, count, step)]
    with Pool(workers) as pool, tqdm(total=count, desc=split) as bar:
        for n in pool.imap_unordered(worker, jobs):
            bar.update(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--workers", type=int, default=min(4, cpu_count()))
    args = ap.parse_args()
    generate("train", args.train, args.workers)
    generate("val", args.val, args.workers)


if __name__ == "__main__":
    main()
