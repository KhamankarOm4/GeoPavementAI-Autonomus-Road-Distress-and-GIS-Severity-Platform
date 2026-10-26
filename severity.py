"""Rule-based severity scoring.
score = type_weight * min(1, box_area / image_area / FULL_AREA)
Potholes and alligator cracks are structurally worse than thin cracks, so they weigh more.
"""
KEYS = ((("pothole", "d40"), 1.0), (("alligator", "d20"), 0.8), (("transverse", "d10"), 0.5), (("longitudinal", "d00"), 0.4))


def type_weight(name: str) -> float:
    n = name.lower()
    return next((w for keys, w in KEYS if any(k in n for k in keys)), 0.5)
FULL_AREA = 0.08          # area ratio at which the size term saturates
LOW_MAX, MED_MAX = 0.20, 0.50


def label_for(score: float) -> str:
    return "Low" if score < LOW_MAX else "Medium" if score < MED_MAX else "High"


def box_severity(cls_name, x1, y1, x2, y2, img_w, img_h):
    x1, y1, x2, y2 = (float(v) for v in (x1, y1, x2, y2))   # numpy floats are not JSON/SQLite safe
    ratio = max(0.0, (x2 - x1) * (y2 - y1)) / (float(img_w) * float(img_h))
    score = type_weight(cls_name) * min(1.0, ratio / FULL_AREA)
    return round(float(score), 3), label_for(score), round(float(ratio), 4)


def image_severity(scores):
    """Worst damage dominates; each extra damage adds a small bonus."""
    if not scores:
        return 0.0, "None"
    s = float(min(1.0, max(float(v) for v in scores) + 0.05 * (len(scores) - 1)))
    return round(s, 3), label_for(s)
