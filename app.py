"""
app.py  —  Urban-Eye · Flask Backend API
Serves drone state, alerts, and dispatch endpoints.
All drone movement is simulated in background threads.
"""

import json
import math
import os
import threading
import time

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ALERTS_FILE = "alerts.json"

# ── Drone fleet ───────────────────────────────────────────────────────────────
# Base positions are real Pune coordinates.
# State is held in memory — resets when Flask restarts (intentional for demo).
drones: list[dict] = [
    {"id": "D1", "name": "Drone 1", "lat": 18.5204, "lng": 73.8567, "status": "idle", "battery": 95},
    {"id": "D2", "name": "Drone 2", "lat": 18.5314, "lng": 73.8446, "status": "idle", "battery": 88},
    {"id": "D3", "name": "Drone 3", "lat": 18.5094, "lng": 73.8678, "status": "idle", "battery": 92},
    {"id": "D4", "name": "Drone 4", "lat": 18.5414, "lng": 73.8767, "status": "idle", "battery": 79},
]

# Drone base positions (for return journey)
DRONE_BASES: dict[str, dict] = {d["id"]: {"lat": d["lat"], "lng": d["lng"]} for d in drones}

# ── Camera registry ───────────────────────────────────────────────────────────
CAMERAS: list[dict] = [
    {"id": "CAM_01", "name": "Shivajinagar Junction",  "lat": 18.5308, "lng": 73.8474, "zone": "high_traffic"},
    {"id": "CAM_02", "name": "Koregaon Park Entry",    "lat": 18.5362, "lng": 73.8938, "zone": "commercial"},
    {"id": "CAM_03", "name": "Hinjewadi IT Park Gate", "lat": 18.5912, "lng": 73.7389, "zone": "restricted"},
    {"id": "CAM_04", "name": "Pune Railway Station",   "lat": 18.5284, "lng": 73.8742, "zone": "public"},
]

# ── Auto-dispatch tracking ────────────────────────────────────────────────────
dispatched_indices: set[int] = set()
dispatch_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Returns great-circle distance in km between two lat/lng points."""
    R = 6_371
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lng / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def find_nearest_drone(inc_lat: float, inc_lng: float) -> tuple[dict | None, float]:
    """Returns (drone, distance_km) for the closest idle drone, or (None, inf)."""
    nearest, min_dist = None, float("inf")
    for drone in drones:
        if drone["status"] != "idle":
            continue
        dist = haversine(drone["lat"], drone["lng"], inc_lat, inc_lng)
        if dist < min_dist:
            min_dist, nearest = dist, drone
    return nearest, min_dist


def load_alerts() -> list[dict]:
    """Reads alerts.json safely. Returns empty list if file missing or corrupt."""
    if not os.path.exists(ALERTS_FILE):
        return []
    try:
        with open(ALERTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def simulate_drone_movement(
    drone: dict, target_lat: float, target_lng: float, mission_id: int
) -> None:
    """
    Moves drone from current position to target in 30 steps (1 s each),
    waits 10 s on-site, then returns to base.
    Runs in a daemon thread — does not block the API.
    """
    steps     = 30
    base      = DRONE_BASES[drone["id"]]
    start_lat = base["lat"]
    start_lng = base["lng"]

    # Outbound journey
    drone["status"] = "dispatched"
    for i in range(1, steps + 1):
        time.sleep(1)
        drone["lat"]     = start_lat + (target_lat - start_lat) * (i / steps)
        drone["lng"]     = start_lng + (target_lng - start_lng) * (i / steps)
        drone["battery"] = max(0, drone["battery"] - 0.3)

    # On-site
    drone["status"] = "on-site"
    time.sleep(10)

    # Return journey
    drone["status"] = "returning"
    for i in range(1, steps + 1):
        time.sleep(1)
        drone["lat"]     = target_lat + (start_lat - target_lat) * (i / steps)
        drone["lng"]     = target_lng + (start_lng - target_lng) * (i / steps)
        drone["battery"] = max(0, drone["battery"] - 0.2)

    # Snap back to exact base position and mark idle
    drone["lat"]    = start_lat
    drone["lng"]    = start_lng
    drone["status"] = "idle"
    print(f"[INFO] {drone['name']} returned to base after mission {mission_id}")


def _do_dispatch(alert: dict, index: int) -> dict:
    """
    Core dispatch logic shared by manual and auto-dispatch routes.
    Returns a result dict with drone info or an error key.
    Must be called with dispatch_lock held externally if needed.
    """
    # Use the alert's real GPS — not random coordinates
    inc_lat = alert.get("lat")
    inc_lng = alert.get("lng")

    if inc_lat is None or inc_lng is None:
        return {"error": "Alert has no geolocation data"}

    drone, distance = find_nearest_drone(inc_lat, inc_lng)
    if drone is None:
        return {"error": "No idle drones available"}

    # ETA: assume drone speed ~50 km/h
    eta_seconds = int((distance / 50) * 3600)

    thread = threading.Thread(
        target=simulate_drone_movement,
        args=(drone, inc_lat, inc_lng, index),
        daemon=True,
    )
    thread.start()

    print(
        f"[DISPATCH] {drone['name']} → {alert.get('type', 'unknown')} "
        f"at {alert.get('camera_name', '?')} | {distance:.2f}km | ETA {eta_seconds}s"
    )

    return {
        "message":      f"{drone['name']} dispatched",
        "drone_id":     drone["id"],
        "incident":     alert.get("type", "unknown"),
        "camera":       alert.get("camera_name", "unknown"),
        "destination":  {"lat": inc_lat, "lng": inc_lng},
        "distance_km":  round(distance, 2),
        "eta_seconds":  eta_seconds,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/status")
def status():
    return jsonify({"status": "online", "timestamp": round(time.time(), 2)})


@app.route("/alerts")
def get_alerts():
    alerts = load_alerts()
    # Sort by severity descending, return top 50
    alerts.sort(key=lambda x: x.get("severity_score", 0), reverse=True)
    return jsonify(alerts[:50])


@app.route("/drones")
def get_drones():
    return jsonify(drones)


@app.route("/cameras")
def get_cameras():
    return jsonify(CAMERAS)


@app.route("/dispatch/<int:alert_index>")
def dispatch_drone(alert_index: int):
    alerts = load_alerts()

    if not alerts:
        return jsonify({"error": "No alerts found"}), 404

    if alert_index < 0 or alert_index >= len(alerts):
        return jsonify({"error": f"Alert index {alert_index} out of range (0–{len(alerts)-1})"}), 400

    with dispatch_lock:
        if alert_index in dispatched_indices:
            return jsonify({"error": "Already dispatched for this alert"}), 409
        dispatched_indices.add(alert_index)

    result = _do_dispatch(alerts[alert_index], alert_index)
    if "error" in result:
        # Roll back so it can be retried
        with dispatch_lock:
            dispatched_indices.discard(alert_index)
        return jsonify(result), 503

    return jsonify(result)


@app.route("/auto_dispatch")
def auto_dispatch():
    """
    Called every few seconds by the dashboard.
    Dispatches one drone per call for any CRITICAL or collision alert
    that hasn't been dispatched yet.
    """
    alerts = load_alerts()
    dispatched_count = 0

    for index, alert in enumerate(alerts):
        sev_label  = alert.get("severity_label", "LOW")
        alert_type = alert.get("type", "")

        should_dispatch = (
            sev_label == "CRITICAL"
            or alert_type == "collision_detected"
            or alert.get("severity_score", 0) >= 8
        )

        if not should_dispatch:
            continue

        with dispatch_lock:
            if index in dispatched_indices:
                continue
            dispatched_indices.add(index)

        result = _do_dispatch(alert, index)
        if "error" not in result:
            dispatched_count += 1
            break  # One dispatch per poll cycle — avoids emptying fleet instantly

    return jsonify({
        "dispatched":            dispatched_count,
        "total_auto_dispatched": len(dispatched_indices),
    })


@app.route("/dispatched_indices")
def get_dispatched_indices():
    return jsonify({"indices": list(dispatched_indices)})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("⬡ Urban-Eye API starting on http://localhost:5050")
    # debug=False in production — set to True only during development
    app.run(host="0.0.0.0", port=5050, debug=False)
