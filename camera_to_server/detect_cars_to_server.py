import time
from collections import Counter

import cv2
import requests
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# Welke labels beschouwen we als voertuigen?
VEHICLE_LABELS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}

# Camera index: /dev/video1
CAMERA_INDEX = 1
SAMPLING_PERIOD_SEC = 10.0  # elke 10 seconden één meting

# >>> IP + poort van je school-VM <<<  (werkt, je testte net /docs)
SERVER_URL = "http://100.89.11.82:8001/api/v1/observations"


def main():
    print("Loading RF-DETR model (CPU)...")
    model = RFDETRBase()

    # Camera openen
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera /dev/video{CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    print(
        f"Starting headless vehicle counting on /dev/video{CAMERA_INDEX}, "
        f"every {SAMPLING_PERIOD_SEC} seconds..."
    )

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read frame from camera")
                break

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Detectie
            detections = model.predict(frame_rgb, threshold=0.3)

            # Tellen per categorie
            category_counts = Counter()
            for bbox, cls_id, score in zip(
                detections.xyxy, detections.class_id, detections.confidence
            ):
                try:
                    cls_name = COCO_CLASSES[cls_id]
                except Exception:
                    cls_name = str(cls_id)

                if cls_name not in VEHICLE_LABELS:
                    continue

                category_counts[cls_name] += 1

            total_vehicles = sum(category_counts.values())
            ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

            # Console log
            print(
                f"{ts}  total_vehicles={total_vehicles}  breakdown={dict(category_counts)}"
            )

            # Payload vlak maken voor FastAPI
            payload = {
                "timestamp": ts,
                "total_vehicles": total_vehicles,
                "car": category_counts.get("car", 0),
                "truck": category_counts.get("truck", 0),
                "bus": category_counts.get("bus", 0),
                "motorcycle": category_counts.get("motorcycle", 0),
                "bicycle": category_counts.get("bicycle", 0),
            }

            # POST naar de server
            try:
                resp = requests.post(SERVER_URL, json=payload, timeout=2)
                print("POST status:", resp.status_code)
                if resp.status_code >= 400:
                    print("Response body:", resp.text)
            except Exception as e:
                print("POST failed:", e)

            # Wachten tot volgende meting
            time.sleep(SAMPLING_PERIOD_SEC)

    finally:
        cap.release()
        print("Camera released, exiting.")


if __name__ == "__main__":
    main()
