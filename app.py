"""
True Guard — Flask streaming server
Run: python app.py
Then open: http://localhost:5000
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
from flask import Flask, Response, jsonify
from threading import Thread, Lock
import json


FONT_SIZE = 3
FONT_THICKNESS = 9

app = Flask(__name__)

# ── Model & detection config (mirrors your original script) ──────────────────
model = YOLO('yolov8n.pt')
DISABLED_CLASSES = ['cat', 'bear', 'bird']
ALERT_COOLDOWN = 5
ALERT_DURATION = 5

# ── Shared state (written by camera thread, read by Flask routes) ─────────────
lock = Lock()
latest_frame = None          # raw JPEG bytes of annotated frame
detection_state = {
    "detections": [],        # list of {label, conf, x1,y1,x2,y2}
    "person_detected": False,
    "alert_active": False,
    "alert_start": 0,
    "last_alert_time": 0,
    "total_detections": 0,
    "alerts_suppressed": False,  # when True, detections run but no alert fires
}


def camera_loop():
    global latest_frame, detection_state

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.8, verbose=False)

        boxes = []
        person_detected = False

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = model.names[int(box.cls[0])]

            if cls in DISABLED_CLASSES:
                continue
            if cls == 'person':
                person_detected = True

            boxes.append({
                'label': cls,
                'conf': round(conf, 3),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            })

        current_time = time.time()

        # Alert logic (same as your original script)
        with lock:
            state = detection_state
            if (person_detected and not state["alerts_suppressed"]
                    and (current_time - state["last_alert_time"]) > ALERT_COOLDOWN):
                state["alert_active"] = True
                state["alert_start"] = current_time
                state["last_alert_time"] = current_time

            if state["alert_active"] and (current_time - state["alert_start"]) > ALERT_DURATION:
                state["alert_active"] = False

            state["detections"] = boxes
            state["person_detected"] = person_detected
            state["total_detections"] = len(boxes)

        # Draw bounding boxes on frame
        annotated = frame.copy()
        for d in boxes:
            color = (0, 0, 255) if d['label'] == 'person' else (0, 200, 255)
            cv2.rectangle(annotated, (d['x1'], d['y1']), (d['x2'], d['y2']), color, 2)
            label_text = f"{d['label']} {d['conf']:.2f}"
            cv2.putText(annotated, label_text, (d['x1'], d['y1'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, color, FONT_THICKNESS)

        # Alert overlay on frame
        with lock:
            if detection_state["alert_active"]:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 50), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
                cv2.putText(annotated, "PERSON DETECTED — ALERT ACTIVE",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(annotated, f"Detections: {len(boxes)}", (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), FONT_THICKNESS)

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with lock:
            latest_frame = buffer.tobytes()

    cap.release()


def generate_stream():
    """Yields MJPEG frames for the browser."""
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1 / 30)  # ~30 fps cap


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return open('index.html', encoding='utf-8').read()


@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state')
def state():
    with lock:
        s = detection_state.copy()
    # Make JSON-serialisable
    s['alert_active'] = bool(s['alert_active'])
    s['person_detected'] = bool(s['person_detected'])
    return jsonify(s)


@app.route('/dismiss', methods=['POST'])
def dismiss():
    with lock:
        detection_state['alert_active'] = False
    return jsonify({'ok': True})


@app.route('/toggle_suppress', methods=['POST'])
def toggle_suppress():
    with lock:
        detection_state['alerts_suppressed'] = not detection_state['alerts_suppressed']
        if detection_state['alerts_suppressed']:
            detection_state['alert_active'] = False
        state = detection_state['alerts_suppressed']
    return jsonify({'alerts_suppressed': state})


if __name__ == '__main__':
    # Start camera inference in background thread
    t = Thread(target=camera_loop, daemon=True)
    t.start()
    print("True Guard running → http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
