def calculate_severity(alert):
    """
    Scores an alert from 1-10 based on multiple factors.
    Returns score + reasoning for display.
    """
    score = 0
    reasons = []

    incident_type = alert.get("type")
    confidence = alert.get("confidence", 0)
    zone = alert.get("zone", "public")
    bbox = alert.get("bbox", [0, 0, 100, 100])

    # ── Base score by incident type ──────────────────────
    type_scores = {
        "crowd_gathering": 6,
        "suspicious_vehicle": 5,
        "potential_incident": 7,
        "abandoned_object": 4,
        "collision_detected": 9,
    }
    base = type_scores.get(incident_type, 3)
    score += base
    reasons.append(f"Incident type: {incident_type} (+{base})")

    # ── Confidence factor ────────────────────────────────
    # High confidence = more certain = higher priority
    if confidence >= 0.85:
        score += 2
        reasons.append("High confidence detection (+2)")
    elif confidence >= 0.70:
        score += 1
        reasons.append("Medium confidence detection (+1)")

    # ── Zone factor ──────────────────────────────────────
    zone_scores = {
        "restricted": 3,
        "high_traffic": 2,
        "commercial": 1,
        "public": 0
    }
    zone_bonus = zone_scores.get(zone, 0)
    if zone_bonus > 0:
        score += zone_bonus
        reasons.append(f"Zone risk ({zone}) (+{zone_bonus})")

    # ── Object size factor ───────────────────────────────
    # Larger bounding box = closer to camera = more severe
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    if area > 50000:
        score += 2
        reasons.append("Large object/close range (+2)")
    elif area > 20000:
        score += 1
        reasons.append("Medium range (+1)")

    # ── Crowd count factor ───────────────────────────────
    if incident_type == "crowd_gathering":
        count = alert.get("count", 5)
        if count >= 15:
            score += 3
            reasons.append(f"Large crowd ({count} people) (+3)")
        elif count >= 10:
            score += 2
            reasons.append(f"Medium crowd ({count} people) (+2)")
        elif count >= 5:
            score += 1
            reasons.append(f"Small crowd ({count} people) (+1)")

    # ── Cap at 10 ────────────────────────────────────────
    score = min(score, 10)

    # ── Severity label ───────────────────────────────────
    if score >= 8:
        label = "CRITICAL"
        color = "#f44336"
    elif score >= 6:
        label = "HIGH"
        color = "#ff9800"
    elif score >= 4:
        label = "MEDIUM"
        color = "#ffeb3b"
    else:
        label = "LOW"
        color = "#4caf50"

    return {
        "score": score,
        "label": label,
        "color": color,
        "reasons": reasons
    }