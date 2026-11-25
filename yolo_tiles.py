"""Run a YOLO-OBB model over whole pages tile by tile (used inside the Colab notebook).

predict_page(detect, image) -> [{"cls": int, "conf": float, "quad": [[x, y] * 4]}]
`detect` takes a list of PIL tiles and returns, per tile, a list of (cls, conf, quad).
"""
import json
import math
from pathlib import Path

from PIL import Image

TILE, OVERLAP = 960, 160


def starts(size):
    if size <= TILE:
        return [0]
    s = list(range(0, size - TILE, TILE - OVERLAP))
    return s + [size - TILE]


def center(q):
    return sum(p[0] for p in q) / 4, sum(p[1] for p in q) / 4


def short_side(q):
    return min(math.dist(q[0], q[1]), math.dist(q[1], q[2]))


def predict_page(detect, im):
    boxes = [(x, y) for y in starts(im.height) for x in starts(im.width)]
    found = []
    for (x, y), dets in zip(boxes, detect([im.crop((x, y, x + TILE, y + TILE)) for x, y in boxes])):
        for cls, conf, q in dets:
            xs, ys = [p[0] for p in q], [p[1] for p in q]
            m = 3  # cut by an inner tile edge: the neighbouring tile has it whole
            if ((x > 0 and min(xs) < m) or (y > 0 and min(ys) < m)
                    or (x + TILE < im.width and max(xs) > TILE - m) or (y + TILE < im.height and max(ys) > TILE - m)):
                continue
            found.append(dict(cls=int(cls), conf=float(conf), quad=[[float(px + x), float(py + y)] for px, py in q]))
    keep = []
    for d in sorted(found, key=lambda d: -d["conf"]):
        c, s = center(d["quad"]), short_side(d["quad"])
        if all(math.dist(c, center(k["quad"])) > .5 * min(s, short_side(k["quad"])) for k in keep):
            keep.append(d)
    return keep


def ultralytics_detector(model, conf=.15, batch=6):
    def detect(tiles):
        out = []
        for i in range(0, len(tiles), batch):
            for r in model.predict(tiles[i:i + batch], imgsz=TILE, conf=conf, max_det=600, verbose=False):
                o = r.obb
                out.append([(int(c), float(p), q) for c, p, q in
                            zip(o.cls.int().tolist(), o.conf.tolist(), o.xyxyxyxy.tolist())])
        return out
    return detect


def run_folder(detect, folder, out):
    res = {}
    for p in sorted(Path(folder).rglob("*.jpg")):
        res[p.name] = predict_page(detect, Image.open(p).convert("RGB"))
    Path(out).write_text(json.dumps(res))
    print(len(res), "pages ->", out)
