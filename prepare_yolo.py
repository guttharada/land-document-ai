"""Turn data_v2 train + val into one YOLO-OBB zip for Colab.  Test is never included.

    python3 prepare_yolo.py --out colab/land_yolo.zip

Pages are cut into 960 px tiles (small text stays readable).  Dense sheets (P, Q)
give more tiles than single-parcel pages (S, H).  Tiles go straight into the zip,
so nothing extra is left on disk.  Label line: class x1 y1 x2 y2 x3 y3 x4 y4 (0-1).
"""
import argparse, io, json, random, zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

TILE, OVERLAP = 960, 160
CLASSES = ["parcel", "struck", "marker", "length", "area", "text"]


def starts(size):
    """Tile origins covering `size`; the last tile is aligned to the edge."""
    if size <= TILE:
        return [0]
    s = list(range(0, size - TILE, TILE - OVERLAP))
    return s + [size - TILE]


def cls_of(l):
    if l["cls"] == "parcel":
        return 1 if l["void"] else 0
    return {"marker": 2, "length": 3, "area": 4}.get(l["cls"], 5)


def area(q):
    return abs(sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4))) / 2


def tile_labels(labels, x, y):
    """Labels whose centre is inside and that are at least 70% visible; corners clipped to the tile."""
    out = []
    for l in labels:
        q = [(px - x, py - y) for px, py in l["quad"]]
        cx, cy = sum(p[0] for p in q) / 4, sum(p[1] for p in q) / 4
        if not (0 <= cx < TILE and 0 <= cy < TILE):
            continue
        c = [(min(max(px, 0), TILE), min(max(py, 0), TILE)) for px, py in q]
        if area(q) < 1 or area(c) / area(q) < .7:
            continue
        out.append((cls_of(l), c))
    return out


YAML = """path: /content/land_yolo
train: images/train
val: images/val
names:
""" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v2")
    ap.add_argument("--out", default="colab/land_yolo.zip")
    ap.add_argument("--dense-train", type=int, default=3, help="tiles per P/Q page (max 6)")
    ap.add_argument("--sparse-train", type=int, default=1, help="tiles per S/H page")
    ap.add_argument("--dense-val", type=int, default=2)
    ap.add_argument("--val-pages", type=int, default=30, help="whole val pages per tier")
    ap.add_argument("--quality", type=int, default=78)
    ap.add_argument("--limit", type=int, default=0, help="pages per split, for smoke runs")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    stats = {}
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:
        z.writestr("land_yolo/data.yaml", YAML)
        for split in ("train", "val"):
            r, c, tiles, offices = random.Random(21), Counter(), 0, set()
            for p in sorted((Path(a.data) / split).glob("*.json"))[: a.limit or None]:
                rec = json.loads(p.read_text())
                assert rec["split"] == split
                offices.add(rec["office"])
                im = Image.open(Path(a.data) / rec["image"]).convert("RGB")
                cand = [(x, y, tile_labels(rec["labels"], x, y)) for y in starts(im.height) for x in starts(im.width)]
                cand = [t for t in cand if t[2]]
                n = (a.dense_train if split == "train" else a.dense_val) if rec["tier"] in "PQ" else a.sparse_train
                for x, y, labs in r.sample(cand, min(n, len(cand))):
                    name = f"{p.stem}_{x}_{y}"
                    buf = io.BytesIO()
                    im.crop((x, y, x + TILE, y + TILE)).save(buf, "JPEG", quality=a.quality)
                    z.writestr(f"land_yolo/images/{split}/{name}.jpg", buf.getvalue())
                    z.writestr(f"land_yolo/labels/{split}/{name}.txt", "\n".join(
                        f"{k} " + " ".join(f"{v / TILE:.6f}" for pt in q for v in pt) for k, q in labs))
                    c.update(CLASSES[k] for k, _ in labs)
                    tiles += 1
            stats[split] = dict(tiles=tiles, offices=sorted(offices), labels=dict(c))
        # whole val pages (images only) for page-level prediction in the notebook
        pages = sorted((Path(a.data) / "val").glob("*.json"))[: a.limit or None]
        for t in "SHPQ":
            pool = [p for p in pages if p.stem.endswith(t)]
            for p in random.Random(22).sample(pool, min(a.val_pages, len(pool))):
                z.write(p.with_suffix(".jpg"), f"land_yolo/val_pages/{p.stem}.jpg")
    print(json.dumps(stats, indent=1))
    print(f"{Path(a.out).stat().st_size / 1e6:.0f} MB -> {a.out}")
