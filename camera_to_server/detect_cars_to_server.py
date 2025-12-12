import os
import time
import json
import uuid
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import requests
from rfdetr import RFDETRBase

# =========================
# CONFIG
# =========================

CAMERA_DEVICE = "/dev/video1"  # device in Docker: /dev/video1
SAMPLE_EVERY_SECONDS = 10.0

VM_API_URL = "http://100.89.11.82:8001/api/v1/observations"  # FastAPI endpoint

CAMERA_ID = "rock5-camera-1"

SNAPSHOT_DIR = Path("/app/snapshots")  # in container
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# MODEL LOADING
# =========================

def load_model(device: str = "cpu") -> RFDETRBase:
    """
    Load RF-DETR model once, on CPU.
    We use default pretrained weights from the library.
    """
    print(f"Loading RF-DETR (RFDETRBase) model on {device}...", flush=True)
    model = RFDETRBase()
    # De lib laadt zelf weights; we log gewoon.
    print("Loading pretrain weights", flush=True)
    # Sommige versies doen init al in __init__, maar dit is veilig.
    # Als dit niets doet, ook goed.
    try:
        model.load_pretrain_weights()
        print("Pretrained weights loaded via load_pretrain_weights().", flush=True)
    except Exception as e:
        # Als deze methode niet bestaat, gewoon verder met default weights
        print(f"load_pretrain_weights() not available or failed ({e}), using default model weights.", flush=True)

    print("Model loaded (RFDETRBase, default weights).", flush=True)
    return model


# =========================
# DETECTION → COUNTS / BOXES
# =========================

def count_vehicles_and_boxes(detections) -> tuple[int, dict, list, list]:
    """
    Neemt RF-DETR 'detections' object en telt voertuigen.
    Returned:
        total_vehicles (int),
        breakdown (dict),
        boxes (list of [x1,y1,x2,y2]),
        labels (list of str)
    """
    breakdown = {
        "car": 0,
        "truck": 0,
        "bus": 0,
        "motorcycle": 0,
        "bicycle": 0,
    }

    # mapping COCO-class-id -> onze labels
    # COCO (standaard):
    # 1: person, 2: bicycle, 3: car, 4: motorcycle, 6: bus, 8: truck, ...
    id2label = {
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        6: "bus",
        8: "truck",
    }

    boxes_out = []
    labels_out = []

    # Als detections geen expected attrs heeft → geen voertuigen
    if detections is None:
        return 0, breakdown, boxes_out, labels_out

    class_ids = getattr(detections, "class_id", None)
    boxes = getattr(detections, "xyxy", None)

    if class_ids is None or boxes is None:
        return 0, breakdown, boxes_out, labels_out

    # class_ids en boxes zijn typisch np.array of lijst
    # Zorg dat we erover kunnen itereren
    if hasattr(class_ids, "tolist"):
        class_ids = class_ids.tolist()
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()

    for i, cid in enumerate(class_ids):
        try:
            cid_int = int(cid)
        except Exception:
            continue

        label = id2label.get(cid_int)
        if label is None:
            continue  # skip classes die ons niet boeien

        breakdown[label] += 1

        if i < len(boxes):
            box = boxes[i]
            if len(box) == 4:
                boxes_out.append(box)
                labels_out.append(label)

    total_vehicles = sum(breakdown.values())
    return total_vehicles, breakdown, boxes_out, labels_out


# =========================
# SNAPSHOT ANNOTATION
# =========================

def draw_boxes_on_frame(frame: np.ndarray, boxes: list, labels: list) -> np.ndarray:
    """
    Tekent simpele bounding boxes + label op het frame.
    """
    annotated = frame.copy()
    for (box, label) in zip(boxes, labels):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def save_snapshot(frame: np.ndarray, boxes: list, labels: list) -> Path:
    """
    Slaat elke 10s een snapshot op (met boxes getekend).
    Returnt het pad in de container.
    """
    annotated = draw_boxes_on_frame(frame, boxes, labels)

    ts_str = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    uid = uuid.uuid4().hex[:6]
    filename = f"snapshot_{ts_str}_{uid}.jpg"
    out_path = SNAPSHOT_DIR / filename

    ok = cv2.imwrite(str(out_path), annotated)
    if not ok:
        print(f"⚠️ Failed to write snapshot to {out_path}", flush=True)
    else:
        print(f"📸 Saved snapshot: {out_path}", flush=True)

    return out_path


# =========================
# API CALL
# =========================

def post_observation(total_vehicles: int, breakdown: dict, snapshot_path: Path | None):
    """
    Stuurt payload + optioneel snapshot naar de VM API.
    Snapshot wordt *niet* in SQL opgeslagen; alleen de path op de VM
    wordt daar bijgehouden (server-side).
    """
    now_iso = dt.datetime.utcnow().isoformat() + "Z"

    payload = {
        "camera_id": CAMERA_ID,
        "total_vehicles": int(total_vehicles),
        "car": int(breakdown.get("car", 0)),
        "truck": int(breakdown.get("truck", 0)),
        "bus": int(breakdown.get("bus", 0)),
        "motorcycle": int(breakdown.get("motorcycle", 0)),
        "bicycle": int(breakdown.get("bicycle", 0)),
        "timestamp": now_iso,
    }

    files = {}
    data = {"payload": json.dumps(payload)}

    if snapshot_path is not None and snapshot_path.exists():
        files["snapshot"] = (
            snapshot_path.name,
            open(snapshot_path, "rb"),
            "image/jpeg",
        )

    try:
        resp = requests.post(VM_API_URL, data=data, files=files, timeout=5)
        print(f"POST status: {resp.status_code}", flush=True)
        if resp.status_code >= 400:
            print(f"Response body: {resp.text}", flush=True)
    except Exception as e:
        print(f"POST failed: {e}", flush=True)
    finally:
        # file handle clean
        if "snapshot" in files:
            files["snapshot"][1].close()


# =========================
# MAIN LOOP
# =========================

def main():
    device = "cpu"
    model = load_model(device=device)

    print(f"Opening camera {CAMERA_DEVICE} with V4L2 backend...", flush=True)
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not cap or not cap.isOpened():
        print(f"❌ Failed to open camera {CAMERA_DEVICE}", flush=True)
        return

    print(
        f"✅ VideoCapture opened: True\nStarting headless vehicle counting on {CAMERA_DEVICE}, every {SAMPLE_EVERY_SECONDS} seconds...",
        flush=True,
    )

    last_sample_ts = 0.0
    sample_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Failed to read frame from camera, retrying in 2s...", flush=True)
                time.sleep(2.0)
                continue

            now = time.time()
            if now - last_sample_ts < SAMPLE_EVERY_SECONDS:
                # Niet elke frame loggen; gewoon een short sleep
                time.sleep(0.05)
                continue

            last_sample_ts = now
            sample_idx += 1

            # Inference
            try:
                # BELANGRIJK: RFDETRBase is niet callable, gebruik predict()
                prediction = model.predict(frame)
            except Exception as e:
                print(f"Model.predict(frame) failed: {e}", flush=True)
                continue

            total_vehicles, breakdown, boxes, labels = count_vehicles_and_boxes(prediction)

            ts_iso = dt.datetime.utcnow().isoformat() + "+00:00"
            print(
                f"{ts_iso}  sample={sample_idx}  total_vehicles={total_vehicles}  breakdown={breakdown}",
                flush=True,
            )

            # Snapshot opslaan (altijd, om de 10s)
            snapshot_path = save_snapshot(frame, boxes, labels)

            # Naar VM sturen
            post_observation(total_vehicles, breakdown, snapshot_path)

    except KeyboardInterrupt:
        print("👋 Interrupted by user, shutting down...", flush=True)
    finally:
        cap.release()
        print("Camera released, exiting.", flush=True)


if __name__ == "__main__":
    main()
