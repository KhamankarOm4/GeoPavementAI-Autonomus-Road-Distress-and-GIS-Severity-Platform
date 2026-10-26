import base64, io, json, os, random, threading, time, uuid
from pathlib import Path
from typing import Optional
import cv2, numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO
import db
from config import ROOT, DISPLAY_NAMES
from severity import box_severity, image_severity

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "weights" / "best.pt"))
DEFAULT_LAT, DEFAULT_LON = float(os.getenv("DEFAULT_LAT", 18.5204)), float(os.getenv("DEFAULT_LON", 73.8567))
DATA = ROOT / "data"; (DATA / "images").mkdir(parents=True, exist_ok=True)
BGR = {"Low": (87, 139, 46), "Medium": (0, 161, 224), "High": (40, 40, 198)}
model = None

app = FastAPI(title="Road Damage Survey")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/media", StaticFiles(directory=DATA / "images"), name="media")


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"Weights not found at {MODEL_PATH}. Copy your best.pt there or set MODEL_PATH.")
        model = YOLO(str(MODEL_PATH))
    return model


def exif_gps(raw: bytes):
    try:
        gps = Image.open(io.BytesIO(raw)).getexif().get_ifd(0x8825)
        dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        lat, lon = dms(gps[2]), dms(gps[4])
        return (-lat if gps.get(1) == "S" else lat), (-lon if gps.get(3) == "W" else lon)
    except Exception:
        return None


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), conf: float = Form(0.25),
                 lat: Optional[float] = Form(None), lon: Optional[float] = Form(None),
                 location_name: Optional[str] = Form(None)):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read that file as an image.")
    H, W = img.shape[:2]
    mdl = get_model()
    res = mdl.predict(img, conf=conf, imgsz=640, verbose=False)[0]

    dets, out = [], img.copy()
    for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.cpu().numpy().tolist(), res.boxes.conf.cpu().numpy().tolist(),
                                      res.boxes.cls.cpu().numpy().astype(int).tolist()):
        cls_raw = str(mdl.names.get(k, k))
        name = db.normalize_type_name(DISPLAY_NAMES.get(cls_raw, cls_raw))
        score, label, ratio = box_severity(name, x1, y1, x2, y2, W, H)
        dets.append(dict(type=name, confidence=round(float(c), 3), severity=label, score=score, area_ratio=ratio,
                         box=[int(x1), int(y1), int(x2), int(y2)]))
        col = BGR[label]
        display_label = name.replace("_", " ").title()
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
        cv2.putText(out, f"{display_label} [{label}]", (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    o_score, o_label = image_severity([d["score"] for d in dets]) if dets else (0.0, "Low")
    ok, buf = cv2.imencode(".jpg", out)
    result = dict(count=len(dets), overall=dict(score=o_score, label=o_label), detections=dets,
                  image="data:image/jpeg;base64," + base64.b64encode(buf).decode())

    if lat is not None and lon is not None:
        loc_src = "manual"
    elif (g := exif_gps(raw)):
        lat, lon = g; loc_src = "exif"
    else:   # demo fallback: scatter slightly so markers don't overlap
        lat, lon = DEFAULT_LAT + random.uniform(-.02, .02), DEFAULT_LON + random.uniform(-.02, .02); loc_src = "demo"
    rid = uuid.uuid4().hex[:8]
    cv2.imwrite(str(DATA / "images" / f"{rid}.jpg"), out)
    clean_loc = (location_name or "").strip()
    if not clean_loc:
        clean_loc = db.resolve_location_name(lat, lon)
    rec = dict(id=rid, time=time.strftime("%Y-%m-%d %H:%M"), lat=lat, lon=lon, loc_source=loc_src, count=len(dets),
               overall=result["overall"], types=sorted({d["type"] for d in dets}), image=f"/media/{rid}.jpg",
               location_name=clean_loc)
    db.add_report(rec, dets)
    result["record"] = rec
    return result


@app.get("/api/records")
def records():
    return db.list_reports()


@app.get("/api/records/{rid}")
def get_record(rid: str):
    rep = db.get_report(rid)
    if not rep:
        raise HTTPException(404, "Report not found")
    return rep


@app.get("/api/stats")
def stats():
    return db.stats()


@app.delete("/api/records/{rid}")
def delete_one(rid: str):
    db.delete_report(rid)
    img_file = DATA / "images" / f"{rid}.jpg"
    if img_file.exists():
        try:
            img_file.unlink()
        except Exception:
            pass
    return {"ok": True, "id": rid}


@app.delete("/api/records")
def clear():
    db.clear()
    return {"ok": True}
