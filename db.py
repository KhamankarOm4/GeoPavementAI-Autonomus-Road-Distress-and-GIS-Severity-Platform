"""SQLite storage (built into Python, no server to install). File: data/survey.db"""
import json, sqlite3
from contextlib import closing
from config import ROOT

DB_PATH = ROOT / "data" / "survey.db"
DB_PATH.parent.mkdir(exist_ok=True)


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init():
    with closing(_conn()) as c, c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS reports(
            id TEXT PRIMARY KEY, time TEXT, lat REAL, lon REAL, loc_source TEXT,
            count INTEGER, score REAL, label TEXT, types TEXT, image TEXT,
            location_name TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS detections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT REFERENCES reports(id) ON DELETE CASCADE,
            type TEXT, confidence REAL, severity TEXT, score REAL, area_ratio REAL,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER);
        """)
        # Safe migration for existing DB: add location_name column if missing
        try:
            c.execute("ALTER TABLE reports ADD COLUMN location_name TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # already exists

        # Clean up any legacy class_N names in detections
        c.execute("UPDATE detections SET type = 'longitudinal_crack' WHERE type IN ('class_0', 'd00', 'D00')")
        c.execute("UPDATE detections SET type = 'transverse_crack' WHERE type IN ('class_1', 'd10', 'D10')")
        c.execute("UPDATE detections SET type = 'alligator_crack' WHERE type IN ('class_2', 'd20', 'D20')")
        c.execute("UPDATE detections SET type = 'pothole' WHERE type IN ('class_3', 'd40', 'D40')")

        # Fix any existing reports that have blank location_name
        c.execute("""
            UPDATE reports
            SET location_name = 'Road Corridor (' || ROUND(lat, 4) || ', ' || ROUND(lon, 4) || ')'
            WHERE location_name IS NULL OR TRIM(location_name) = '';
        """)


def resolve_location_name(lat, lon) -> str:
    if lat is None or lon is None:
        return "Designated Road Corridor"
    try:
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError):
        return "Designated Road Corridor"

    POIS = [
        ("Sinhagad Road, Vadgaon Budruk", 18.4682, 73.8241),
        ("Law College Road, Erandwane", 18.5132, 73.8310),
        ("Karve Road, Kothrud", 18.5074, 73.8077),
        ("Senapati Bapat Road, Model Colony", 18.5314, 73.8340),
        ("Magarpatta Bypass, Hadapsar", 18.5089, 73.9259),
        ("Pune-Bangalore Highway NH48, Warje", 18.4795, 73.8012),
        ("Nagar Road, Viman Nagar", 18.5679, 73.9143),
        ("Paud Road, MIT College Corner", 18.5147, 73.8118),
        ("Jangali Maharaj Road, Deccan", 18.5204, 73.8567),
        ("Aundh-Baner Link Road", 18.5590, 73.8070),
        ("Old Mumbai-Pune Highway, Khadki", 18.5635, 73.8402),
        ("Fergusson College Road, Shivajinagar", 18.5236, 73.8418),
        ("Rajgopalachari Chowk, Somwar Peth", 18.5228, 73.8896),
        ("Hinjawadi IT Park Corridor", 18.5913, 73.7389),
        ("Swargate Chowk, Pune", 18.5018, 73.8636),
        ("Pune Station Road Corridor", 18.5284, 73.8744),
        ("Mumbai-Pune Expressway Toll Corridor", 18.7500, 73.4000),
        ("Seminary Hills, Nagpur", 21.1710, 79.0393),
        ("Wardha Road, Nagpur", 21.0900, 79.0600),
    ]

    import math
    closest_name, min_d = None, float("inf")
    for name, p_lat, p_lon in POIS:
        d = math.hypot(p_lat - lat, p_lon - lon)
        if d < min_d:
            min_d = d
            closest_name = name

    if closest_name and min_d < 0.045:
        return closest_name

    return f"Road Corridor ({lat:.4f}, {lon:.4f})"


def normalize_type_name(t: str) -> str:
    CANON = {
        "class_0": "longitudinal_crack", "class_1": "transverse_crack",
        "class_2": "alligator_crack", "class_3": "pothole",
        "d00": "longitudinal_crack", "d10": "transverse_crack",
        "d20": "alligator_crack", "d40": "pothole",
        "D00": "longitudinal_crack", "D10": "transverse_crack",
        "D20": "alligator_crack", "D40": "pothole",
        "0": "longitudinal_crack", "1": "transverse_crack",
        "2": "alligator_crack", "3": "pothole",
        "longitudinal_crack": "longitudinal_crack", "transverse_crack": "transverse_crack",
        "alligator_crack": "alligator_crack", "pothole": "pothole"
    }
    return CANON.get(str(t).strip().lower(), str(t).strip().lower())


def add_report(rec, dets):
    clean_types = [normalize_type_name(t) for t in rec.get("types", [])]
    loc_name = (rec.get("location_name") or "").strip()
    if not loc_name:
        loc_name = resolve_location_name(rec.get("lat"), rec.get("lon"))
        rec["location_name"] = loc_name
    with closing(_conn()) as c, c:
        c.execute("INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (rec["id"], rec["time"], rec["lat"], rec["lon"], rec["loc_source"], rec["count"],
                   rec["overall"]["score"], rec["overall"]["label"], json.dumps(clean_types), rec["image"],
                   loc_name))
        c.executemany("INSERT INTO detections(report_id,type,confidence,severity,score,area_ratio,x1,y1,x2,y2) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?)",
                      [(rec["id"], normalize_type_name(d["type"]), d["confidence"], d["severity"], d["score"], d["area_ratio"], *d["box"])
                       for d in dets])


def delete_report(rid: str):
    with closing(_conn()) as c, c:
        c.execute("DELETE FROM reports WHERE id = ?", (rid,))


def get_report(rid: str):
    with closing(_conn()) as c:
        r = c.execute("SELECT * FROM reports WHERE id = ?", (rid,)).fetchone()
        if not r:
            return None
        dets = c.execute("SELECT * FROM detections WHERE report_id = ?", (rid,)).fetchall()

    raw_score = r["score"]
    if isinstance(raw_score, (bytes, bytearray)):
        import struct
        try:
            score = round(struct.unpack("d", raw_score)[0], 3)
        except Exception:
            score = 0.0
    else:
        score = round(float(raw_score), 3) if raw_score is not None else 0.0

    try:
        raw_types = json.loads(r["types"])
        clean_types = [normalize_type_name(t) for t in raw_types]
    except Exception:
        clean_types = []

    loc_name = r["location_name"] if "location_name" in r.keys() and r["location_name"] else ""

    det_list = []
    for d in dets:
        det_list.append(dict(
            type=normalize_type_name(d["type"]),
            confidence=round(float(d["confidence"]), 3) if d["confidence"] is not None else 0.0,
            severity=d["severity"],
            score=round(float(d["score"]), 3) if d["score"] is not None else 0.0,
            area_ratio=round(float(d["area_ratio"]), 4) if d["area_ratio"] is not None else 0.0,
            box=[d["x1"], d["y1"], d["x2"], d["y2"]]
        ))

    return dict(
        id=r["id"], time=r["time"], lat=r["lat"], lon=r["lon"],
        loc_source=r["loc_source"], count=r["count"],
        overall=dict(score=score, label=r["label"]),
        types=clean_types, image=r["image"],
        location_name=loc_name,
        detections=det_list
    )


def list_reports():
    with closing(_conn()) as c:
        rows = c.execute("SELECT * FROM reports ORDER BY time DESC, rowid DESC").fetchall()
    result = []
    for r in rows:
        # score may be stored as bytes (SQLite REAL quirk with older rows); coerce safely
        raw_score = r["score"]
        if isinstance(raw_score, (bytes, bytearray)):
            import struct
            try:
                score = round(struct.unpack("d", raw_score)[0], 3)
            except Exception:
                score = 0.0
        else:
            score = round(float(raw_score), 3) if raw_score is not None else 0.0

        try:
            raw_types = json.loads(r["types"])
            clean_types = [normalize_type_name(t) for t in raw_types]
        except Exception:
            clean_types = []

        keys = r.keys()
        loc_name = r["location_name"] if "location_name" in keys and r["location_name"] else ""

        result.append(dict(
            id=r["id"], time=r["time"], lat=r["lat"], lon=r["lon"],
            loc_source=r["loc_source"], count=r["count"],
            overall=dict(score=score, label=r["label"]),
            types=clean_types, image=r["image"],
            location_name=loc_name
        ))
    return result


def stats():
    with closing(_conn()) as c:
        by_sev  = c.execute("SELECT severity, COUNT(*) n FROM detections GROUP BY severity").fetchall()
        by_type = c.execute("SELECT type, COUNT(*) n FROM detections GROUP BY type ORDER BY n DESC").fetchall()
        total_reports = c.execute("SELECT COUNT(*) FROM reports").fetchone()[0]

    # Pre-populate all 4 standard classes so charts always have full spectrum
    merged = {
        "pothole": 0,
        "alligator_crack": 0,
        "transverse_crack": 0,
        "longitudinal_crack": 0
    }
    for r in by_type:
        key = normalize_type_name(r["type"])
        merged[key] = merged.get(key, 0) + r["n"]

    total_detections = sum(merged.values())
    sev_map = {r["severity"]: r["n"] for r in by_sev}

    return dict(
        by_severity=sev_map,
        by_type=merged,
        total_reports=total_reports,
        total_detections=total_detections,
        pothole_count=merged.get("pothole", 0),
        high_severity_count=sev_map.get("High", 0)
    )


def clear():
    with closing(_conn()) as c, c:
        c.execute("DELETE FROM reports")   # detections are removed by ON DELETE CASCADE


init()
