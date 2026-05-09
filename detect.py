"""
detect.py  —  Urban-Eye · AI Detection Pipeline
Runs YOLOv8s inference on each configured camera feed in a separate thread.
Writes confirmed alerts to alerts.json with full metadata + severity scores.
"""

import json
import threading
import time

import cv2
from ultralytics import YOLO

from severity import calculate_severity

# ── Constants ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD   = 0.50   # Ignore detections below this
CROWD_THRESHOLD        = 5      # Min persons in frame to trigger crowd alert
ALERT_COOLDOWN_SECONDS = 5      # Minimum gap between same alert type per camera
COLLISION_THRESHOLD_PX = 50     # Max centre-to-centre distance to flag as collision
ALERTS_FILE            = "alerts.json"
MODEL_PATH             = "yolov8s.pt"
DEVICE                 = "mps"  # Change to "cuda" on Nvidia GPU, "cpu" as fallback

# ── Incident class mapping ────────────────────────────────────────────────────
INCIDENT_CLASSES: dict[str, str] = {
    "person":    "potential_incident",
    "car":       "suspicious_vehicle",
    "truck":     "suspicious_vehicle",
    "bus":       "suspicious_vehicle",
    "backpack":  "abandoned_object",
    "suitcase":  "abandoned_object",
    "handbag":   "abandoned_object",
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# ── Camera configurations ─────────────────────────────────────────────────────
# Add / uncomment cameras as you get more video files.
CAMERA_CONFIGS: list[dict] = [
    {
        "id":    "CAM_01",
        "name":  "Shivajinagar Junction",
        "lat":   18.5308,
        "lng":   73.8474,
        "video": "truck accident.mp4",
        "zone":  "high_traffic",
    },
    # {
    #     "id":    "CAM_02",
    #     "name":  "Koregaon Park Entry",
    #     "lat":   18.5362,
    #     "lng":   73.8938,
    #     "video": "crowd_video.mp4",
    #     "zone":  "commercial",
    # },
    # {
    #     "id":    "CAM_03",
    #     "name":  "Hinjewadi IT Park Gate",
    #     "lat":   18.5912,
    #     "lng":   73.7389,
    #     "video": "suspicious_vehicle.mp4",
    #     "zone":  "restricted",
    # },
]

# ── Shared state (protected by lock) ─────────────────────────────────────────
alert_log:       list[dict]        = []
alert_log_lock:  threading.Lock    = threading.Lock()
last_alert_time: dict[str, float]  = {}

# Load model once — shared across all threads (YOLO inference is thread-safe)
model = YOLO(MODEL_PATH)


# ── Collision detection ───────────────────────────────────────────────────────
def detect_collisions(boxes, class_names: dict) -> list[dict]:
    """
    Checks every pair of detected vehicles for overlap or proximity.
    Returns a list of collision dicts (may be empty).
    """
    vehicles = []
    for box in boxes:
        conf       = float(box.conf[0])
        class_name = class_names[int(box.cls[0])]
        if conf < CONFIDENCE_THRESHOLD or class_name not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicles.append({
            "class":  class_name,
            "bbox":   [x1, y1, x2, y2],
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "conf":   round(conf, 2),
        })

    collisions = []
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            v1, v2 = vehicles[i], vehicles[j]

            # Euclidean distance between centres
            dx = v1["center"][0] - v2["center"][0]
            dy = v1["center"][1] - v2["center"][1]
            distance = (dx**2 + dy**2) ** 0.5

            # Intersection area
            b1, b2    = v1["bbox"], v2["bbox"]
            overlap_x = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            overlap_y = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            overlap_area = overlap_x * overlap_y

            if overlap_area <= 0 and distance >= COLLISION_THRESHOLD_PX:
                continue  # Not a collision

            if overlap_area > 5_000:
                level, fscore, reason = "CRITICAL", 10, "Direct impact — large overlap"
            elif overlap_area > 1_000:
                level, fscore, reason = "HIGH",     8,  "Significant contact between vehicles"
            elif overlap_area > 0:
                level, fscore, reason = "HIGH",     7,  "Minor contact detected"
            else:
                level, fscore, reason = "MEDIUM",   5,  f"Vehicles dangerously close ({int(distance)}px)"

            collisions.append({
                "vehicle_1":   v1["class"],
                "vehicle_2":   v2["class"],
                "overlap_area": overlap_area,
                "distance":    round(distance, 1),
                "fatal_level": level,
                "fatal_score": fscore,
                "fatal_reason": reason,
            })

    return collisions


# ── Write alerts safely ───────────────────────────────────────────────────────
def _write_alerts() -> None:
    """Called inside alert_log_lock — writes current log to disk."""
    try:
        with open(ALERTS_FILE, "w", encoding="utf-8") as f:
            json.dump(alert_log, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[ERROR] Could not write {ALERTS_FILE}: {exc}")


# ── Per-camera detection loop ─────────────────────────────────────────────────
def process_camera(camera: dict) -> None:
    cam_id   = camera["id"]
    cam_name = camera["name"]
    cam_lat  = camera["lat"]
    cam_lng  = camera["lng"]
    zone     = camera["zone"]
    video    = camera["video"]

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video for {cam_name}: {video}")
        return

    print(f"[INFO] Started detection → {cam_name} ({cam_id})")
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video when it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_number = 0
            continue

        frame_number += 1
        results    = model(frame, device=DEVICE, verbose=False)
        boxes      = results[0].boxes
        frame_alerts: list[dict] = []
        person_count = 0

        # ── Per-detection analysis ────────────────────────────────────
        for box in boxes:
            conf       = float(box.conf[0])
            class_name = model.names[int(box.cls[0])]

            if conf < CONFIDENCE_THRESHOLD:
                continue

            if class_name == "person":
                person_count += 1

            if class_name in INCIDENT_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                sev = calculate_severity({
                    "type":       INCIDENT_CLASSES[class_name],
                    "confidence": round(conf, 2),
                    "zone":       zone,
                    "bbox":       [x1, y1, x2, y2],
                })
                frame_alerts.append({
                    "type":             INCIDENT_CLASSES[class_name],
                    "class":            class_name,
                    "confidence":       round(conf, 2),
                    "bbox":             [x1, y1, x2, y2],
                    "frame":            frame_number,
                    "timestamp":        round(time.time(), 2),
                    "camera_id":        cam_id,
                    "camera_name":      cam_name,
                    "lat":              cam_lat,
                    "lng":              cam_lng,
                    "zone":             zone,
                    "severity_score":   sev["score"],
                    "severity_label":   sev["label"],
                    "severity_color":   sev["color"],
                    "severity_reasons": sev["reasons"],
                })

        # ── Collision detection ───────────────────────────────────────
        for col in detect_collisions(boxes, model.names):
            sev_color = "#f44336" if col["fatal_score"] >= 8 else "#ff9800"
            frame_alerts.append({
                "type":             "collision_detected",
                "class":            f"{col['vehicle_1']}+{col['vehicle_2']}",
                "confidence":       1.0,
                "bbox":             [0, 0, 100, 100],
                "frame":            frame_number,
                "timestamp":        round(time.time(), 2),
                "camera_id":        cam_id,
                "camera_name":      cam_name,
                "lat":              cam_lat,
                "lng":              cam_lng,
                "zone":             zone,
                "severity_score":   col["fatal_score"],
                "severity_label":   col["fatal_level"],
                "severity_color":   sev_color,
                "severity_reasons": [
                    f"Collision: {col['vehicle_1']} vs {col['vehicle_2']}",
                    col["fatal_reason"],
                    f"Overlap: {col['overlap_area']}px²",
                    f"Centre distance: {col['distance']}px",
                ],
                "collision_details": col,
            })

        # ── Crowd detection ───────────────────────────────────────────
        if person_count >= CROWD_THRESHOLD:
            sev = calculate_severity({
                "type":       "crowd_gathering",
                "confidence": 1.0,
                "zone":       zone,
                "bbox":       [0, 0, 100, 100],
                "count":      person_count,
            })
            frame_alerts.append({
                "type":             "crowd_gathering",
                "class":            "person",
                "confidence":       1.0,
                "count":            person_count,
                "frame":            frame_number,
                "timestamp":        round(time.time(), 2),
                "camera_id":        cam_id,
                "camera_name":      cam_name,
                "lat":              cam_lat,
                "lng":              cam_lng,
                "zone":             zone,
                "severity_score":   sev["score"],
                "severity_label":   sev["label"],
                "severity_color":   sev["color"],
                "severity_reasons": sev["reasons"],
            })

        # ── Cooldown check and persist ────────────────────────────────
        now = time.time()
        for alert in frame_alerts:
            key = f"{cam_id}_{alert['type']}"
            if now - last_alert_time.get(key, 0) < ALERT_COOLDOWN_SECONDS:
                continue

            last_alert_time[key] = now
            print(
                f"[ALERT] {cam_name} | {alert['type']:25s} | "
                f"{alert['class']:15s} | conf:{alert['confidence']:.2f} | "
                f"sev:{alert['severity_score']}/10 {alert['severity_label']}"
            )

            with alert_log_lock:
                alert_log.append(alert)
                _write_alerts()

        # Save latest annotated frame (used by dashboard thumbnail feature)
        annotated = results[0].plot()
        cv2.imwrite(f"frame_{cam_id}.jpg", annotated)

    cap.release()
    print(f"[INFO] Camera thread ended: {cam_name}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    if not CAMERA_CONFIGS:
        print("[ERROR] No cameras configured. Add entries to CAMERA_CONFIGS.")
        return

    threads = []
    for camera in CAMERA_CONFIGS:
        t = threading.Thread(
            target=process_camera,
            args=(camera,),
            name=f"cam-{camera['id']}",
            daemon=True,
        )
        threads.append(t)
        t.start()

    print(f"[INFO] Urban-Eye detection running · {len(threads)} camera(s) active")
    print("[INFO] Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down detection pipeline...")


if __name__ == "__main__":
    main()
