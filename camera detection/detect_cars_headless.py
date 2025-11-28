import time
from collections import Counter

import cv2
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# Welke labels zien we als “voertuig”?
VEHICLE_LABELS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "train",
    "boat",
}

# Camera index: 1 -> /dev/video1
CAMERA_INDEX = 1
SAMPLING_PERIOD_SEC = 10.0  # elke 10 seconden een meting


def main():
    # --- 1. Model laden ---
    print("Loading RF-DETR model (CPU)...")
    model = RFDETRBase()
    # GEEN model.optimize_for_inference() hier, dat gaf TorchScript errors op deze setup

    # Kleine debug: eerste COCO-classes tonen
    print("Type COCO_CLASSES:", type(COCO_CLASSES))
    try:
        print("First 10 COCO_CLASSES entries:")
        for i in range(10):
            print(i, "->", COCO_CLASSES[i])
    except Exception as e:
        print("Could not preview COCO_CLASSES:", e)

    # --- 2. Camera openen ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera /dev/video{CAMERA_INDEX}")

    # Eventueel resolutie zetten (pas aan als je wil)
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

            # BGR (OpenCV) -> RGB (model)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # RF-DETR normaliseert intern, dus uint8 [0..255] is ok

            # --- 3. Detectie draaien ---
            detections = model.predict(frame_rgb, threshold=0.3)
            # Je kan deze debug laten staan of later weghalen:
            # print("Total raw detections in frame:", len(detections.class_id))

            # --- 4. Tellen per klasse ---
            category_counts = Counter()
            for bbox, cls_id, score in zip(
                detections.xyxy, detections.class_id, detections.confidence
            ):
                # COCO klassenaam ophalen
                try:
                    cls_name = COCO_CLASSES[cls_id]
                except Exception:
                    cls_name = str(cls_id)

                if cls_name not in VEHICLE_LABELS:
                    continue

                category_counts[cls_name] += 1

            total_vehicles = sum(category_counts.values())

            ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            # Mooie logregel die later ook makkelijk te parseren is
            print(f"{ts}  total_vehicles={total_vehicles}  breakdown={dict(category_counts)}")

            # Wachten tot de volgende sample
            time.sleep(SAMPLING_PERIOD_SEC)

    finally:
        cap.release()
        print("Camera released, exiting.")


if __name__ == "__main__":
    main()
