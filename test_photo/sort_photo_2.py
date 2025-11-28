import time
from collections import Counter
import os

import cv2
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# PAS DIT AAN naar jouw fotonaam op de Rock5
# bv. "traffic_test.jpg" of "snapshot.jpg"
IMAGE_PATH = "traffic_test.jpg"

# Welke labels tellen we als “voertuig”?
VEHICLE_LABELS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "train",
    "boat",
}


def main():
    # --- 0. Check of de foto bestaat ---
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    print("Loading RF-DETR model (CPU)...")
    model = RFDETRBase()
    # GEEN optimize_for_inference() -> gaf TorchScript errors op jouw setup

    # --- 1. Kleine debug over COCO_CLASSES ---
    print("Type COCO_CLASSES:", type(COCO_CLASSES))
    # Probeer de eerste paar entries te printen
    try:
        print("First 10 COCO_CLASSES entries:")
        for i in range(10):
            try:
                print(i, "->", COCO_CLASSES[i])
            except Exception:
                break
    except Exception:
        pass

    # --- 2. Afbeelding inladen ---
    frame_bgr = cv2.imread(IMAGE_PATH)
    if frame_bgr is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    # BGR -> RGB (model verwacht RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Belangrijk: RF-DETR normaliseert zelf, dus gewoon uint8 [0..255] doorgeven

    # --- 3. Detectie draaien ---
    detections = model.predict(frame_rgb, threshold=0.3)
    print("Total raw detections:", len(detections.class_id))

    # --- 4. Annoteren: cirkels + tellen ---
    category_counts = Counter()
    total_vehicles = 0

    for bbox, cls_id, score in zip(
        detections.xyxy, detections.class_id, detections.confidence
    ):
        # Haal klassenaam op uit COCO_CLASSES (werkt voor list én dict)
        try:
            cls_name = COCO_CLASSES[cls_id]
        except Exception:
            cls_name = str(cls_id)

        if cls_name not in VEHICLE_LABELS:
            continue

        category_counts[cls_name] += 1
        total_vehicles += 1

        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1

        # center + radius voor cirkel
        cx = x1 + w // 2
        cy = y1 + h // 2
        r = max(5, int(0.4 * min(w, h)))  # 40% van kleinste zijde

        # Rode cirkel (BGR: (0, 0, 255))
        cv2.circle(frame_bgr, (cx, cy), r, (0, 0, 255), 2)

        # Label tekst
        label = f"{cls_name} {score:.2f}"
        text_x = x1
        text_y = max(0, y1 - 10)
        cv2.putText(
            frame_bgr,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    # --- 5. Summary linksboven ---
    summary_lines = [f"Total vehicles: {total_vehicles}"]
    for cat in sorted(category_counts.keys()):
        summary_lines.append(f"{cat}: {category_counts[cat]}")

    y0 = 30
    for i, line in enumerate(summary_lines):
        y = y0 + i * 20
        cv2.putText(
            frame_bgr,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # --- 6. Output-bestandsnaam ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    out_file = f"{base_name}_annotated_{ts}.jpg"

    cv2.imwrite(out_file, frame_bgr)

    # --- 7. Console output ---
    print(f"Input image: {IMAGE_PATH}")
    print(f"Saved annotated image to: {out_file}")
    print(f"Total vehicles detected: {total_vehicles}")
    for cat in sorted(category_counts.keys()):
        print(f"  {cat}: {category_counts[cat]}")


if __name__ == "__main__":
    main()
