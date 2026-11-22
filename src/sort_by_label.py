import sys
from collections import defaultdict
from pathlib import Path
import cv2
import yaml

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/roboflow_infant")
OUT = Path("data/test_postures")
PAD = 0.10       # extra margin around each box
MIN_SIDE = 32    # skip crops smaller than this (pixels)
SPLIT_NAME = {"train": "train", "valid": "val", "test": "test"}  # Roboflow -> YOLO folder names

with open(SRC / "data.yaml", encoding="utf-8") as f:
    names = yaml.safe_load(f)["names"]

if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
print("Classes:", names)

counts = defaultdict(int)
skipped = 0

for rf_split, out_split in SPLIT_NAME.items():
    img_dir, lbl_dir = SRC / rf_split / "images", SRC / rf_split / "labels"
    if not img_dir.exists():
        continue

    img_paths = (
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.jpeg"))
        + list(img_dir.glob("*.png"))
    )

    for img_path in img_paths:
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            skipped += 1
            continue

        lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]

        for i, line in enumerate(lines):
            p = line.split()
            if len(p) < 5:
                skipped += 1
                continue

            cid = int(p[0])
            if cid < 0 or cid >= len(names):
                skipped += 1
                continue

            cls = names[cid].lower()  # also merges 'Warning' / 'warning'

            if len(p) == 5:  # box: cx cy w h
                cx, cy, bw, bh = map(float, p[1:5])
                x1, x2, y1, y2 = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
            else:  # polygon: x1 y1 x2 y2 ...
                xs, ys = list(map(float, p[1::2])), list(map(float, p[2::2]))
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)

            pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
            X1, X2 = max(0, int((x1 - pw) * w)), min(w, int((x2 + pw) * w))
            Y1, Y2 = max(0, int((y1 - ph) * h)), min(h, int((y2 + ph) * h))

            if X2 - X1 < MIN_SIDE or Y2 - Y1 < MIN_SIDE:
                skipped += 1
                continue

            dest = OUT / out_split / cls
            dest.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest / f"{img_path.stem}_{i:02d}.jpg"), img[Y1:Y2, X1:X2])
            counts[(out_split, cls)] += 1

print(f"\nSkipped (no/empty label, bad class id, or too small): {skipped}")
print(f"{'split':<8}{'class':<12}{'crops':>6}")
for (split, cls), n in sorted(counts.items()):
    print(f"{split:<8}{cls:<12}{n:>6}")