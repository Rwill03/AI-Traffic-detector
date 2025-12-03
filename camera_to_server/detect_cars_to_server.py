import time
import base64
from datetime import datetime, timezone

import cv2
import torch
import requests

from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES


# ===== CONFIG =====
CAMERA_DEVICE = "/dev/video1"          # of /dev/video0 als dat je device is
SAMPLE_INTERVAL_SEC = 10.0

SERVER_URL = "http://100.89.11.82:8001/api/v1/observations"

CONF_THRESHOLD = 0.30
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]


def load_model() -> RFDETRBase:
    """
    RF-DETR model laden met COCO pretrain weights.
    Zorg dat rf-detr-base.pth in dezelfde map staat als dit script.
    """
    print("Loading RF-DETR (RFDETRBase) model on CPU...")
    # Als jouw checkpoint anders heet, pas dit pad aan:
    model = RFDETRBase(pretrain_weights="rf-detr-base.pth")
    # Optioneel kun je dit doen als je performance wil tunen:
    # model.optimize_for_inference()
    print("Model loaded.")
    return model


def count_and_annotate(frame_bgr, model: RFDETRBase):
    """
    Run RF-DETR op het frame, tel voertuigen en teken kaders + labels.

    Returns:
      total_vehicles (int),
      counts (dict),
      annotated_frame_bgr (np.ndarray)
    """
    # RF-DETR werkt prima met een RGB image (np array)
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # model.predict kan direct op numpy array / image
    detections = model.predict(img_rgb, threshold=CONF_THRESHOLD)

    xyxy = detections.xyxy            # [N, 4]
    class_ids = detections.class_id   # [N]
    scores = detections.confidence    # [N]

    counts = {k: 0 for k in VEHICLE_CLASSES}
    total_vehicles = 0

    annotated = frame_bgr.copy()

    for box, cid, score in zip(xyxy, class_ids, scores):
        score = float(score)
        class_idx = int(cid)

        # COCO label-naam ophalen
        if class_idx < 0 or class_idx >= len(COCO_CLASSES):
            continue
        cls_name = COCO_CLASSES[class_idx].lower()

        if cls_name not in VEHICLE_CLASSES:
            continue

        total_vehicles += 1
        counts[cls_name] += 1

        x1, y1, x2, y2 = map(int, box)

        # Rode box
        color = (0, 0, 255)  # BGR: rood
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {score:.2f}"
        text_org = (x1, max(y1 - 5, 10))
        cv2.putText(
            annotated,
            label,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return total_vehicles, counts, annotated


def frame_to_base64_jpeg(frame_bgr):
    """
    Converteer een BGR-frame naar JPEG en dan naar base64 string.
    """
    try:
        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return b64
    except Exception:
        return None


def main():
    # Model op CPU laden
    model = load_model()

    # Camera openen met expliciet V4L2-backend
    print(f"Opening camera {CAMERA_DEVICE} with V4L2 backend...")
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    # Match settings die ffmpeg liet zien: YUY2, 1280x960, 5 fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    cap.set(cv2.CAP_PROP_FPS, 5)

    print("VideoCapture opened:", cap.isOpened())
    if not cap.isOpened():
        print(f"❌ Could not open camera device {CAMERA_DEVICE} (inside Docker)")
        return

    print(
        f"Starting headless vehicle counting on {CAMERA_DEVICE}, "
        f"every {SAMPLE_INTERVAL_SEC} seconds..."
    )

    try:
        while True:
            # Frame lezen
            ret, frame = cap.read()
            print("cap.read() ret:", ret)  # extra debug
            if not ret:
                print("⚠️ Failed to read frame from camera, retrying in 2s...")
                time.sleep(2.0)
                continue

            # Timestamp in UTC
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()

            # Detectie + annotatie
            total_vehicles, counts, annotated_frame = count_and_annotate(frame, model)

            # Annotated frame → JPEG → base64
            frame_b64 = frame_to_base64_jpeg(annotated_frame)

            # JSON-payload richting VM API
            payload = {
                "timestamp": now_iso,
                "total_vehicles": int(total_vehicles),
                "car": int(counts.get("car", 0)),
                "truck": int(counts.get("truck", 0)),
                "bus": int(counts.get("bus", 0)),
                "motorcycle": int(counts.get("motorcycle", 0)),
                "bicycle": int(counts.get("bicycle", 0)),
                "camera_id": "rock5-1",
                "frame_jpeg_b64": frame_b64,
            }

            print(
                f"{now_iso}  total_vehicles={total_vehicles}  breakdown={counts}"
            )

            # POST naar server
            try:
                resp = requests.post(SERVER_URL, json=payload, timeout=2)
                print(f"POST status: {resp.status_code}")
                if resp.status_code >= 400:
                    print(f"Response body: {resp.text}")
            except Exception as e:
                print(f"POST failed: {e}")

            # Wachten tot volgende sample
            time.sleep(SAMPLE_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("Camera released, exiting.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
