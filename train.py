import argparse
from ultralytics import YOLO
from config import DATA_YAML, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolov8n.pt")
ap.add_argument("--name", default="yolov8n")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU (default: auto)")
ap.add_argument("--no-aug", action="store_true", help="Ablation: switch off augmentation")
a = ap.parse_args()

aug = dict(mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
           fliplr=0.0, scale=0.0, translate=0.0) if a.no_aug else {}

YOLO(a.model).train(data=str(DATA_YAML), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                    device=a.device, project=str(ROOT / "runs"), name=a.name,
                    seed=42, patience=15, **aug)
