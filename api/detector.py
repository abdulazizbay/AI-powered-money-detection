import time

import cv2
import numpy as np
import onnxruntime as ort

CLASSES = [
    "coin_10", "coin_50", "coin_100", "coin_500",
    "bill_1000", "bill_5000", "bill_10000", "bill_50000",
]
VALUES = [int(c.split("_")[1]) for c in CLASSES]


class Detector:
    def __init__(self, model_path, size=640):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.size = size

    def letterbox(self, img):
        h, w = img.shape[:2]
        r = self.size / max(h, w)
        nh, nw = round(h * r), round(w * r)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        return canvas, r, left, top

    def __call__(self, img, conf=0.4, iou=0.5):
        start = time.perf_counter()
        canvas, r, dx, dy = self.letterbox(img)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
        out = self.session.run(None, {self.input: x})[0][0].T  # (8400, 4 + classes)

        scores = out[:, 4:]
        cls = scores.argmax(1)
        best = scores.max(1)
        keep = best >= conf
        boxes, cls, best = out[keep, :4], cls[keep], best[keep]
        # cx,cy,w,h in letterbox space -> x1,y1,x2,y2 in original image
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
        xyxy = (xyxy - [dx, dy, dx, dy]) / r
        h, w = img.shape[:2]
        xyxy = xyxy.clip(0, [w, h, w, h])

        # class-agnostic NMS: one object should get one denomination
        idx = cv2.dnn.NMSBoxes([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in xyxy.tolist()],
                               best.tolist(), conf, iou)
        detections = [{
            "label": CLASSES[cls[i]],
            "value": VALUES[cls[i]],
            "confidence": round(float(best[i]), 3),
            "box": [round(v) for v in xyxy[i].tolist()],
        } for i in np.array(idx).flatten()]
        detections.sort(key=lambda d: -d["value"])
        return detections, round((time.perf_counter() - start) * 1000, 1)
