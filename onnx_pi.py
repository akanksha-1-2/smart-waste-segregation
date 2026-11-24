import os
import ast
import csv
import time
import shutil
import subprocess
from datetime import datetime
from collections import deque, Counter

import cv2
import numpy as np
import onnxruntime as ort

# ================= SETTINGS =================
BASE = os.path.dirname(os.path.abspath(__file__))

CAMERA_SOURCE = 'http://192.168.0.101:4747/video'   # DroidCam URL
SHOW_DISPLAY = True        # True = show camera window on Pi desktop
USE_LCD = True            # True = show result on LCD
USE_VOICE = False          # True = speak the result
SAVE_PHOTOS = True         # Save a photo for each new decision
LCD_ADDRESS = 0x27         # Try 0x3f if LCD does not work

OUTPUT_DIR = os.path.join(BASE, 'output')

IMGSZ = 320
CONFIDENCE = 0.25
IOU = 0.45
SKIP = 3                   # Run detection every 3rd frame
RECONNECT_WAIT = 2

# Decision settings (time window vote)
WINDOW_SECONDS = 1.5
MIN_VOTES = 3
MIN_SHARE = 0.5

# Which model handles which class
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
    'paper': 0.35,
    'cardboard': 0.35,
    'plastic': 0.35,
    'metal': 0.6,
    'e-waste': 0.4,
    'ewaste': 0.4,
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

# Disposal advice for each class
DISPOSAL = {
    'plastic':   'Put dry plastic in recycling.',
    'paper':     'Put dry paper in paper recycling.',
    'cardboard': 'Fold cardboard and give it for recycling.',
    'metal':     'Put metal scrap in the metal bin.',
    'e-waste':   'Do not put in normal waste. Give to an authorised e-waste centre.',
    'ewaste':    'Do not put in normal waste. Give to an authorised e-waste centre.',
}

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

# ================= LCD, VOICE, OUTPUT FOLDER =================
lcd = None
if USE_LCD:
    try:
        from RPLCD.i2c import CharLCD
        lcd = CharLCD('PCF8574', LCD_ADDRESS, port=1, cols=16, rows=2)
    except Exception as e:
        print('LCD not available:', e)
        lcd = None

ESPEAK = shutil.which('espeak-ng') if USE_VOICE else None

os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, 'log.csv')
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(['time', 'class', 'bin', 'confidence', 'photo'])


def show_lcd(line1, line2=''):
    if lcd is None:
        return
    try:
        lcd.clear()
        lcd.write_string(line1[:16])
        lcd.cursor_pos = (1, 0)
        lcd.write_string(line2[:16])
    except Exception as e:
        print('LCD error:', e)


def speak(text):
    if ESPEAK:
        subprocess.Popen([ESPEAK, text])


def make_message(cls, bin_name):
    if cls in ('ewaste', 'e-waste'):
        return 'E waste detected. Give it to an authorised e-waste centre.'
    return f'{cls} detected. Put it in {bin_name.lower()}.'


def log_decision(cls, bin_name, conf, frame_annot):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    photo_name = ''
    if SAVE_PHOTOS:
        photo_name = f'{stamp}_{cls}.jpg'
        cv2.imwrite(os.path.join(OUTPUT_DIR, photo_name), frame_annot)
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([stamp, cls, bin_name, f'{conf:.2f}', photo_name])


# ================= LOAD ONNX MODELS =================
sessions = {}
for name, path in MODEL_FILES.items():
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    sessions[name] = s
    if MODEL_TYPE[name] == 'v8':
        raw = s.get_modelmeta().custom_metadata_map.get('names', '')
        if not raw:
            raise SystemExit('No names metadata found in best_v8.onnx')
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
        pred = out[0].T
        xywh, class_conf = pred[:, :4], pred[:, 4:]
    else:
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


# ================= DECISION =================
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


def draw_boxes(frame, dets, cls, bin_name):
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
    return out


def open_camera():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if cap.isOpened():
        print('Camera connected')
        return cap
    cap.release()
    return None


# ================= MAIN LOOP =================
def main():
    cap = open_camera()
    while cap is None:
        print(f'Camera not found, retrying in {RECONNECT_WAIT} seconds...')
        time.sleep(RECONNECT_WAIT)
        cap = open_camera()

    show_lcd('Waste Segregation', 'Show an item')
    print('Running. Click the window and press q to quit, or Ctrl+C in terminal.')

    frame_n = 0
    fail_count = 0
    last_out = None
    last_decision = None

    try:
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
                annotated = draw_boxes(frame, dets, cls, bin_name)
                last_out = annotated

                if cls is not None:
                    if cls != last_decision:
                        conf = max(d[1] for d in dets if d[2] == cls)
                        msg = make_message(cls, bin_name)
                        print(f'DECISION: {cls} -> {bin_name}')
                        print(f'DISPOSAL: {DISPOSAL.get(cls, "Check local guidelines.")}')
                        print(f'SAY: {msg}')
                        show_lcd(cls.upper(), bin_name)
                        speak(msg)
                        log_decision(cls, bin_name, conf, annotated)
                        last_decision = cls
                else:
                    if last_decision is not None:
                        show_lcd('Waste Segregation', 'Show an item')
                    last_decision = None

            if SHOW_DISPLAY and last_out is not None:
                cv2.imshow('ONNX Live (routed + bin)', last_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        cap.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
        show_lcd('Stopped', '')


if __name__ == '__main__':
    main()