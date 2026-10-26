"""Evaluate every trained model on the test split + robustness tests (blur / low light / noise)."""
import argparse, shutil
from pathlib import Path
import cv2, numpy as np, pandas as pd, yaml
from ultralytics import YOLO
from config import *


def degrade(img, kind):
    if kind == "motion_blur":
        k = np.zeros((15, 15), np.float32); k[7, :] = 1 / 15
        return cv2.filter2D(img, -1, k)
    if kind == "low_light":
        return cv2.convertScaleAbs(img, alpha=0.35, beta=0)
    if kind == "noise":
        noise = np.random.default_rng(0).normal(0, 25, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def make_variant(kind):
    out = ROOT / "datasets" / "robust" / kind
    if out.exists():
        shutil.rmtree(out)
    (out / "images" / "test").mkdir(parents=True); (out / "labels" / "test").mkdir(parents=True)
    for p in (DATA_DIR / "images" / "test").iterdir():
        cv2.imwrite(str(out / "images" / "test" / p.name), degrade(cv2.imread(str(p)), kind))
        shutil.copy(DATA_DIR / "labels" / "test" / f"{p.stem}.txt", out / "labels" / "test")
    d = yaml.safe_load(DATA_YAML.read_text())
    d.update(path=out.as_posix(), train="images/test", val="images/test", test="images/test")
    y = out / "data.yaml"
    y.write_text(yaml.safe_dump(d))
    return y


def evaluate(weights, yaml, tag):
    m = YOLO(str(weights)).val(data=str(yaml), split="test", plots=False, verbose=False)
    row = dict(model=Path(weights).parents[1].name, condition=tag, precision=m.box.mp, recall=m.box.mr,
               mAP50=m.box.map50, mAP50_95=m.box.map, inference_ms=m.speed["inference"])
    for j, ci in enumerate(m.box.ap_class_index):
        row[f"AP50_{m.names[int(ci)]}"] = m.box.ap50[j]
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-robust", action="store_true")
    a = ap.parse_args()
    weights = sorted((ROOT / "runs").glob("*/weights/best.pt"))
    if not weights:
        raise SystemExit("No trained weights found in runs/. Run train.py first.")

    rows = [evaluate(w, DATA_YAML, "clean") for w in weights]
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results_comparison.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    if not a.no_robust:
        best = max(rows, key=lambda r: r["mAP50"])["model"]
        w = ROOT / "runs" / best / "weights" / "best.pt"
        rob = [evaluate(w, make_variant(k), k) for k in ("motion_blur", "low_light", "noise")]
        rob = [evaluate(w, DATA_YAML, "clean")] + rob
        pd.DataFrame(rob).round(4).to_csv(ROOT / "results_robustness.csv", index=False)
        print(f"\nRobustness of best model ({best}):")
        print(pd.DataFrame(rob)[["condition", "precision", "recall", "mAP50", "mAP50_95"]].round(3).to_string(index=False))
