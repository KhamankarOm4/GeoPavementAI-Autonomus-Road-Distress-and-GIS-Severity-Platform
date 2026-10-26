# Road Damage Detection and Severity Assessment (YOLOv8 + FastAPI + Leaflet map)

## 1. Setup (Colab / Kaggle GPU needed for training: Runtime > Change runtime type > T4 GPU)
    pip install -r requirements.txt

## 2. Data: download RDD2022, convert to YOLO format, split 80/10/10
    python prepare_data.py                  # add --max-images 3000 for a quick first run
Prints per-split class counts (put these in your report) and writes datasets/rdd_india/data.yaml.
The Kaggle copy is already split into train / valid / test, which the script detects. `python prepare_data.py --inspect` prints the folder tree.

## 3. Train (model comparison + augmentation ablation)
    python train.py --model yolov8n.pt --name yolov8n --epochs 50
    python train.py --model yolov8s.pt --name yolov8s --epochs 50
    python train.py --model yolov8n.pt --name yolov8n_noaug --epochs 50 --no-aug

## 4. Evaluate
    python evaluate.py
Creates results_comparison.csv (precision, recall, mAP, speed, per-class AP) and results_robustness.csv
(best model on motion blur / low light / noisy copies of the test set).

## 5. Run the web app
    mkdir weights && cp runs/yolov8s/weights/best.pt weights/best.pt
    uvicorn app:app --reload          # open http://127.0.0.1:8000
Optional env vars: MODEL_PATH, DEFAULT_LAT, DEFAULT_LON (demo location when a photo has no GPS tag).
On Colab, expose the app with a tunnel such as ngrok or cloudflared, or run the app on your laptop after copying best.pt.

## Database
Reports and every detected damage are stored in SQLite (`data/survey.db`, built into Python). Tables: `reports`, `detections`.
Try it: `sqlite3 data/survey.db "SELECT type, severity, COUNT(*) FROM detections GROUP BY 1,2"` or open `/api/stats`.

## Severity formula (explain this in the viva)
score = type_weight x min(1, box_area / image_area / 0.08)
Weights: pothole 1.0, alligator 0.8, transverse 0.5, longitudinal 0.4.
Low < 0.20, Medium < 0.50, High >= 0.50. Image severity = worst box + 0.05 per extra damage.
Limitation: area in pixels depends on camera distance; no real-world depth is used.
