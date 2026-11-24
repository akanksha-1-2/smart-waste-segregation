import os
import ast
import time
from collections import deque, Counter

import cv2
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(os.path.abspath(__file__))
PHONE_URL = 'http://192.168.1.4:4747/video'   # DroidCam URL
IMGSZ = 320
CONFIDENCE = 0.25      # base threshold before per-class filter
IOU = 0.45
SKIP = 3               # run detection every 3rd frame (speed)
RECONNECT_WAIT = 2

# Decision layer (time window vote)
WINDOW_SECONDS = 1.5
MIN_VOTES = 3
MIN_SHARE = 0.5

# Which model is responsible for which class
ROUTE = {
    'cardboard': 'best_ep33',
    'paper':     'best_ep33',
    'plastic':   'best_v8',
    'ewaste':    'best_50',
    'e-waste':   'best_50',
    'metal':     'best_50',
}

# Minimum confidence per class
MIN_CONF = {
    'paper': 0.25,
    'cardboard': 0.30,
    'plastic': 0.20,
    'metal': 0.6,
    'e-waste': 0.5,
    'ewaste': 0.5,
}

# Which bin each class goes to
BIN_MAP = {
    'plastic':   'PLASTIC BIN',
    'paper':     'PAPER BIN',
    'cardboard': 'PAPER BIN',
    'metal':     'METAL BIN',
    'e-waste':   'E-WASTE BIN',
    'ewaste':    'E-WASTE BIN',
}
DEFAULT_BIN = 'GENERAL BIN'

# Class names (from print_names.py output)
NAMES_EP33 = {
    0: 'Book', 1: 'Earphones', 2: 'Glasses', 3: 'Headsets', 4: 'Keyboard',
    5: 'Laptop', 6: 'MobilePhone', 7: 'Monitor', 8: 'Mouse', 9: 'Smartphone',
    10: 'TV', 11: 'Tablet', 12: 'cardboard', 13: 'cloth', 14: 'ewaste',
    15: 'glass', 16: 'medical', 17: 'metal', 18: 'organic', 19: 'paper',
    20: 'paper-plastic-glass-metal-organi', 21: 'plastic', 22: 'shoes', 23: 'trash',
}
NAMES_50 = {0: 'cardboard', 1: 'ewaste', 2: 'metal', 3: 'paper', 4: 'plastic'}

MODEL_FILES = {
    'best_ep33': os.path.join(BASE, 'weights', 'best_ep33.onnx'),
    'best_50':   os.path.join(BASE, 'weights', 'best_50.onnx'),
    'best_v8':   os.path.join(BASE, 'weights', 'best_v8.onnx'),
}
MODEL_TYPE = {'best_ep33': 'v5', 'best_50': 'v5', 'best_v8': 'v8'}
MODEL_NAMES = {'best_ep33': NAMES_EP33, 'best_50': NAMES_50}

# ---- Load ONNX sessions ----
sessions = {}
for name, path in MODEL_FILES.items():
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    sessions[name] = s
    if MODEL_TYPE[name] == 'v8':
        raw = s.get_modelmeta().custom_metadata_map.get('names', '')
        if not raw:
            raise SystemExit('best_v8.onnx mein names metadata nahi mila.')
        MODEL_NAMES[name] = {int(k): v for k, v in ast.literal_eval(raw).items()}
    print(name, 'loaded, classes:', MODEL_NAMES[name])


def preprocess(frame):
    h, w = frame.shape[:2]
    scale = IMGSZ / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(x, 0), scale


def nms(boxes, scores, iou_thr):
    idx = scores.argsort()[::-1]
    keep = []
    while idx.size > 0:
        i = idx[0]
        keep.append(i)
        if idx.size == 1:
            break
        x1 = np.maximum(boxes[i, 0], boxes[idx[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[idx[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[idx[1:], 2] - boxes[idx[1:], 0]) * (boxes[idx[1:], 3] - boxes[idx[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-9)
        idx = idx[1:][iou < iou_thr]
    return keep


def postprocess(xywh, class_conf, scale, names):
    cls_id = class_conf.argmax(1)
    conf = class_conf.max(1)
    mask = conf > CONFIDENCE
    xywh, conf, cls_id = xywh[mask], conf[mask], cls_id[mask]
    if len(conf) == 0:
        return []
    cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / scale
    results = []
    for c in np.unique(cls_id):
        m = cls_id == c
        for i in nms(boxes[m], conf[m], IOU):
            results.append((boxes[m][i], float(conf[m][i]), names.get(int(c), str(c))))
    return results


def run_model(name, x, scale):
    s = sessions[name]
    out = s.run(None, {s.get_inputs()[0].name: x})[0]
    if MODEL_TYPE[name] == 'v8':
        # YOLOv8 output: (1, 4 + nc, N) -> transpose to (N, 4 + nc)
        pred = out[0].T
        xywh, class_conf = pred[:, :4], pred[:, 4:]
    else:
        # YOLOv5 output: (1, N, 5 + nc) -> [cx, cy, w, h, obj, cls...]
        pred = out[0]
        xywh = pred[:, :4]
        class_conf = pred[:, 4:5] * pred[:, 5:]
    return postprocess(xywh, class_conf, scale, MODEL_NAMES[name])


def detect_routed(frame):
    x, scale = preprocess(frame)
    dets = []
    for name in MODEL_FILES:
        for box, conf, cls in run_model(name, x, scale):
            if ROUTE.get(cls) == name:
                dets.append((box, conf, cls, name))
    if not dets:
        return []
    boxes = np.array([d[0] for d in dets], dtype=np.float32)
    scores = np.array([d[1] for d in dets], dtype=np.float32)
    keep = nms(boxes, scores, IOU)
    results = [dets[i] for i in keep]
    return [d for d in results if d[1] >= MIN_CONF.get(d[2], 0.0)]


# ---- Decision layer (time window vote) ----
vote_log = deque()

def decide(dets):
    now = time.time()
    vote_log.append((now, [d[2] for d in dets]))
    while vote_log and now - vote_log[0][0] > WINDOW_SECONDS:
        vote_log.popleft()
    votes = Counter()
    for _, classes in vote_log:
        for c in set(classes):
            votes[c] += 1
    if not votes:
        return None, None
    total = len(vote_log)
    cls, n = votes.most_common(1)[0]
    if n >= MIN_VOTES and n / total >= MIN_SHARE:
        return cls, BIN_MAP.get(cls, DEFAULT_BIN)
    return None, None


def open_camera():
    cap = cv2.VideoCapture(PHONE_URL)
    if cap.isOpened():
        print('Camera connected')
        return cap
    cap.release()
    return None


# ---- Main loop ----
cap = open_camera()
while cap is None:
    print(f'Camera not found, retrying in {RECONNECT_WAIT} seconds...')
    time.sleep(RECONNECT_WAIT)
    cap = open_camera()

print('Running. Click the window and press q to quit.')

frame_n = 0
fail_count = 0
last_out = None
last_decision = None

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        fail_count += 1
        if fail_count >= 10:
            print('Stream lost, reconnecting...')
            cap.release()
            cap = None
            while cap is None:
                time.sleep(RECONNECT_WAIT)
                cap = open_camera()
            fail_count = 0
        continue
    fail_count = 0

    frame_n += 1
    if frame_n % SKIP == 1 or last_out is None:
        dets = detect_routed(frame)
        cls, bin_name = decide(dets)
        if cls is not None and cls != last_decision:
            print(f'DECISION: {cls} -> {bin_name}')
            last_decision = cls
        if cls is None:
            last_decision = None

        out = frame.copy()
        for box, conf, name, src in dets:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f'{name} {conf:.2f} ({src})', (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if cls is not None:
            cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(out, f'{cls} -> BIN: {bin_name}', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        last_out = out

    cv2.imshow('ONNX Live (routed + bin)', last_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()