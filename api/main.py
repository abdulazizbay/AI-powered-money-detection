import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from detector import CLASSES, Detector

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.onnx"
DB_PATH = ROOT / "history.db"
WEB_DIR = ROOT.parent / "web"
RATE_TTL = 3600

app = FastAPI(title="KRW Money Counter")
detector = Detector(MODEL_PATH)
rates_cache = {"at": 0, "data": None}

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute("""CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    total INTEGER NOT NULL,
    counts TEXT NOT NULL,
    items INTEGER NOT NULL,
    avg_confidence REAL NOT NULL
)""")


def summarize(detections):
    counts = Counter(d["label"] for d in detections)
    return {
        "total": sum(d["value"] for d in detections),
        "counts": {c: counts[c] for c in CLASSES if counts[c]},
        "items": len(detections),
        "avg_confidence": round(sum(d["confidence"] for d in detections) / len(detections), 3)
        if detections else 0,
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), source: str = "upload", save: bool = True, conf: float = 0.4):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not an image")
    detections, ms = detector(img, conf=conf)
    summary = summarize(detections)
    if save and detections:
        db.execute("INSERT INTO scans (created_at, source, total, counts, items, avg_confidence) "
                   "VALUES (?, ?, ?, ?, ?, ?)",
                   (time.time(), source, summary["total"], json.dumps(summary["counts"]),
                    summary["items"], summary["avg_confidence"]))
        db.commit()
    return {"detections": detections, **summary, "inference_ms": ms,
            "image": {"width": img.shape[1], "height": img.shape[0]}}


@app.get("/api/history")
def history(limit: int = 20):
    rows = db.execute("SELECT id, created_at, source, total, counts, items, avg_confidence "
                      "FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": r[0], "created_at": r[1], "source": r[2], "total": r[3],
             "counts": json.loads(r[4]), "items": r[5], "avg_confidence": r[6]} for r in rows]


@app.get("/api/stats")
def stats():
    scans, total, items = db.execute("SELECT COUNT(*), COALESCE(SUM(total), 0), COALESCE(SUM(items), 0) "
                                     "FROM scans").fetchone()
    counts = Counter()
    for (c,) in db.execute("SELECT counts FROM scans"):
        counts.update(json.loads(c))
    return {"scans": scans, "total": total, "items": items,
            "counts": {c: counts[c] for c in CLASSES if counts[c]}}


@app.delete("/api/history")
def clear_history():
    db.execute("DELETE FROM scans")
    db.commit()
    return {"ok": True}


@app.get("/api/rates")
async def rates():
    if rates_cache["data"] is None or time.time() - rates_cache["at"] > RATE_TTL:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get("https://api.frankfurter.app/latest",
                                     params={"from": "KRW", "to": "USD,EUR,JPY,CNY"})
                r.raise_for_status()
            rates_cache.update(at=time.time(), data=r.json())
        except httpx.HTTPError:
            if rates_cache["data"] is None:
                raise HTTPException(503, "Exchange rates unavailable")
    return rates_cache["data"]


@app.get("/api/model")
def model_info():
    metrics_path = ROOT / "models" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    return {"name": "YOLOv8n", "format": "ONNX", "input": 640, "classes": CLASSES, "metrics": metrics}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
