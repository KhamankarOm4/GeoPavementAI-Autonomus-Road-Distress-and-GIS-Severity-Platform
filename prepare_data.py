"""Prepare RDD2022 (Kaggle: sujityp/rdd2022) for YOLOv8.
Handles (1) already-split YOLO datasets (train / valid / test with images + labels)
and (2) raw Pascal-VOC XML datasets. Builds a (sub)set in datasets/rdd_india with data.yaml."""
import argparse, json, os, random, shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from PIL import Image
from config import *

IMG_EXT = {".jpg", ".jpeg", ".png"}
ALIAS = {"train": "train", "training": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}


def inspect(root: Path, depth=4):
    def walk(d, level):
        files = [f for f in d.iterdir() if f.is_file()]
        ext = dict(Counter(f.suffix.lower() for f in files))
        print("  " * level + f"{d.name}/  {ext if ext else ''}")
        if level < depth:
            for x in sorted(x for x in d.iterdir() if x.is_dir())[:8]:
                walk(x, level + 1)
    walk(root, 0)
    for y in list(root.rglob("*.yaml"))[:2]:
        print("\n", y, "\n", y.read_text()[:600])
    for t in list(root.rglob("*.txt"))[:1]:
        print("\nSample label file", t, "\n", t.read_text()[:200])


def find_yolo_splits(root: Path):
    found = {}
    for dp, dns, _ in os.walk(root):
        for dn in dns:
            if dn.lower() not in ALIAS:
                continue
            d, split = Path(dp) / dn, ALIAS[dn.lower()]
            if (d / "images").is_dir() and (d / "labels").is_dir():
                pair = (d / "images", d / "labels")
            elif d.parent.name.lower() == "images" and (d.parent.parent / "labels" / dn).is_dir():
                pair = (d, d.parent.parent / "labels" / dn)
            else:
                continue
            if split not in found or len(d.parts) < found[split][0]:
                found[split] = (len(d.parts), *pair)
    return {k: (v[1], v[2]) for k, v in found.items()}


def read_names(root: Path):
    import yaml
    for y in sorted(root.rglob("*.yaml"), key=lambda p: len(p.parts)):
        try:
            names = (yaml.safe_load(y.read_text()) or {}).get("names")
        except Exception:
            continue
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names:
            return [str(n) for n in names]
    return None


def xml_to_yolo(xml_path: Path, img_path: Path):
    root = ET.parse(xml_path).getroot()
    w = h = 0
    size = root.find("size")
    if size is not None:
        w, h = int(float(size.findtext("width", "0"))), int(float(size.findtext("height", "0")))
    if w <= 0 or h <= 0:
        with Image.open(img_path) as im:
            w, h = im.size
    lines = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in XML_TO_ID:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{XML_TO_ID[name]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    return lines


def get_lines(img, src):
    if src is None:
        return []
    if src.suffix == ".xml":
        return xml_to_yolo(src, img)
    return [l.strip() for l in src.read_text().splitlines() if l.strip()]


def yolo_pairs(images_dir, labels_dir):
    out = []
    for img in images_dir.rglob("*"):
        if img.suffix.lower() in IMG_EXT:
            lbl = labels_dir / img.relative_to(images_dir).with_suffix(".txt")
            out.append((img, lbl if lbl.exists() else None))
    return out


def country_filter(pairs, countries):
    if not countries:
        return pairs
    keep = [p for p in pairs if any(p[0].name.lower().startswith(c.lower()) for c in countries)]
    if not keep:
        print(f"Note: no file names start with {countries}; using ALL images instead.")
        return pairs
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", help="Use an already downloaded folder instead of kagglehub")
    ap.add_argument("--max-images", type=int, default=0, help="Train images to use (0 = all). val/test get 1/8 each")
    ap.add_argument("--countries", nargs="*", default=COUNTRIES, help="Keep only images whose name starts with these")
    ap.add_argument("--inspect", action="store_true", help="Just print the dataset folder tree")
    args = ap.parse_args()

    if args.dataset_path:
        src = Path(args.dataset_path)
    else:
        import kagglehub
        src = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    print("Dataset path:", src)
    if args.inspect:
        return inspect(src)

    rng = random.Random(SEED)
    yolo = find_yolo_splits(src)
    if "train" in yolo:
        print("Detected a pre-split YOLO dataset:", {k: str(v[0]) for k, v in yolo.items()})
        splits = {s: country_filter(yolo_pairs(*yolo[s]), args.countries) for s in ("train", "val", "test") if s in yolo}
        if "val" not in splits:  # carve val out of train
            tr = splits["train"]; rng.shuffle(tr); k = max(1, len(tr) // 10)
            splits["val"], splits["train"] = tr[:k], tr[k:]
        if "test" not in splits:
            splits["test"] = splits["val"]
        names = read_names(src)
    else:
        pairs = []
        imgs = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT}
        pairs = [(imgs[x.stem], x) for x in src.rglob("*.xml") if x.stem in imgs]
        pairs = country_filter(pairs, args.countries)
        if not pairs:
            print("Could not recognise the dataset layout. Folder tree:\n")
            inspect(src)
            raise SystemExit("\nSend this tree to get the script adjusted.")
        print(f"Using XML annotations: {len(pairs)} images")
        rng.shuffle(pairs)
        n_tr, n_va = int(len(pairs) * SPLIT[0]), int(len(pairs) * SPLIT[1])
        splits = {"train": pairs[:n_tr], "val": pairs[n_tr:n_tr + n_va], "test": pairs[n_tr + n_va:]}
        names = CLASS_NAMES

    if args.max_images:
        want = {"train": args.max_images, "val": max(args.max_images // 8, 50), "test": max(args.max_images // 8, 50)}
        for s in splits:
            rng.shuffle(splits[s]); splits[s] = splits[s][: want[s]]

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    stats, max_id = {}, -1
    for split, items in splits.items():
        (DATA_DIR / "images" / split).mkdir(parents=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True)
        counts = Counter()
        for img, lab in items:
            lines = get_lines(img, lab)
            shutil.copy(img, DATA_DIR / "images" / split / img.name)
            (DATA_DIR / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines))
            counts.update(int(l.split()[0]) for l in lines)
        max_id = max([max_id, *counts])
        stats[split] = {"images": len(items), "boxes_per_class_id": dict(sorted(counts.items()))}

    if not names or len(names) <= max_id:
        print("WARNING: class names not found in the dataset; using generic names.")
        names = [f"class_{i}" for i in range(max_id + 1)]
    stats["class_names"] = names
    yaml_names = "\n".join(f"  {i}: {c}" for i, c in enumerate(names))
    DATA_YAML.write_text(f"path: {DATA_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{yaml_names}\n")
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("Wrote", DATA_YAML)


if __name__ == "__main__":
    main()
