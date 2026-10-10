# KRW Money Counter

AI system that detects Korean coins and bills in an image or live camera feed, counts each denomination and shows the total amount in won (plus USD/EUR/JPY/CNY).

**Classes (8):** ₩10 · ₩50 · ₩100 · ₩500 coins, ₩1,000 · ₩5,000 · ₩10,000 · ₩50,000 bills

## How it works

```
Image / camera → YOLOv8n (ONNX Runtime) → FastAPI → counting engine → Frankfurter FX API → dashboard
```

| Part | Tech |
|---|---|
| Model | YOLOv8n fine-tuned from COCO weights, exported to ONNX |
| Training | PyTorch / Ultralytics on Kaggle GPU (`training/train_kaggle.ipynb`) |
| API | FastAPI, onnxruntime (CPU), OpenCV, class-agnostic NMS, SQLite history |
| Web | HTML/JS, Chart.js, webcam via `getUserMedia` |

## Dataset

| Source | Content | License |
|---|---|---|
| [korean money (yewon)](https://universe.roboflow.com/yewon-v80ql/korean-money-lkeay) | real photos of all 8 denominations | CC BY 4.0 |
| [korean-coins (jhkimishere)](https://universe.roboflow.com/jhkimishere/korean-coins) | real coin photos | CC BY 4.0 |
| Synthetic scenes (`data-pipeline/synth_bills.py`) | 3,300 mixed piles built from real cutouts | — |

The real photos show one denomination at a time, so the synthetic generator builds mixed scenes: bills and coins cut from real photos (same split only, no leakage), placed with perspective warp, overlap, creases, shadows, true size ratios and camera noise. Labels cover only the visible part of each object.

Splits: train = real + synthetic, val = half of real val + synthetic val, **test = real photos only** (never seen in training).

## Run locally

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r data-pipeline/requirements.txt -r api/requirements.txt

# data (needs ROBOFLOW_API_KEY in .env)
cd data-pipeline
python download_roboflow.py && python merge_roboflow.py
python synth_bills.py && python build_dataset.py   # -> data/krw-dataset.zip for Kaggle

# app: put best.onnx + metrics.json from Kaggle into api/models/
cd ../api && uvicorn main:app --port 8000          # open http://localhost:8000
```

## Author

Abdulaziz Khalilov
