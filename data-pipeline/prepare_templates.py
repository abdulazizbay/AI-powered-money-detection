"""Crop downloaded bill scans to the bill edges and save them as flat templates."""
import cv2
import numpy as np

from config import RAW_TEMPLATES, TEMPLATES


def crop_to_bill(img):
    """Remove the plain margin around a scan: pixels that differ from the corner color."""
    border = np.concatenate([img[:5].reshape(-1, 3), img[-5:].reshape(-1, 3),
                             img[:, :5].reshape(-1, 3), img[:, -5:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    diff = np.abs(img.astype(np.int16) - bg).max(axis=2)
    mask = (diff > 30).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w * h < 0.5 * img.shape[0] * img.shape[1]:
        return img  # detection failed, keep the full scan
    return img[y:y + h, x:x + w]


def main():
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    for path in sorted(RAW_TEMPLATES.glob("bill_*")):
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        out = crop_to_bill(img)
        cv2.imwrite(str(TEMPLATES / f"{path.stem}.png"), out)
        print(f"{path.name}: {img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]}")


if __name__ == "__main__":
    main()
