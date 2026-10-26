from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "rdd_india"      # YOLO-format dataset is created here
DATA_YAML = DATA_DIR / "data.yaml"
KAGGLE_SLUG = "sujityp/rdd2022"

COUNTRIES = ["India"]                           # add "Japan" etc. for more data
CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
XML_TO_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}   # RDD2022 labels -> YOLO class ids
SPLIT = (0.8, 0.1, 0.1)                         # train / val / test
SEED = 42
DISPLAY_NAMES = {"class_0": "longitudinal_crack", "class_1": "transverse_crack",
                    "class_2": "alligator_crack", "class_3": "pothole"}
