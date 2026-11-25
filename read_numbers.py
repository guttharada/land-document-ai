"""YOLO boxes + our digit reader.

Score on val pages (same boxes as score_yolo.py, no registry used):
    .venv/bin/python read_numbers.py --preds preds_val.json --reader ocr_thai.pt
Read numbers on one real image, locally (prints only, saves nothing):
    .venv/bin/python read_numbers.py --image deed.png --scale 1 --reader ocr_thai.pt
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import evaluate as ev
from ocr_data import cut, fit
from train_ocr import load, predict

CLASSES = ["parcel", "struck", "marker", "length", "area", "text"]


def upright(q):
    """Clockwise corners, long side first, reading right or upwards."""
    if sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4)) < 0:
        q = q[::-1]
    if math.dist(q[0], q[1]) < math.dist(q[1], q[2]):
        q = q[1:] + q[:1]
    ang = math.degrees(math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]))
    return q if -100 <= ang < 80 else q[2:] + q[:2]


def read(im, quads, model, dev="cpu"):
    """[(text, confidence)] per quad, in the normal reading direction only.

    Trying both directions and keeping the surer one cost 7 points on val: the reader is
    over-confident on upside-down digits, and the flipped reading was never the right one.
    """
    x = [fit(cut(im, upright([list(p) for p in q]))) for q in quads]
    return predict(model, np.stack(x), dev) if x else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="ocr_thai.pt")
    ap.add_argument("--data", default="data_v2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--preds")
    ap.add_argument("--image")
    ap.add_argument("--yolo", default="best.pt")
    ap.add_argument("--scale", type=float, default=1)
    ap.add_argument("--conf", type=float, default=.25)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model = load(a.reader)

    if a.image:
        from ultralytics import YOLO
        import yolo_tiles as T
        im0 = Image.open(a.image).convert("RGB")
        im = im0.resize((int(im0.width * a.scale), int(im0.height * a.scale)), Image.LANCZOS)
        cv = Image.new("RGB", (max(im.width, T.TILE), max(im.height, T.TILE)), im.resize((1, 1)).getpixel((0, 0)))
        cv.paste(im, (0, 0))
        P = [p for p in T.predict_page(T.ultralytics_detector(YOLO(a.yolo)), cv) if p["conf"] >= a.conf and p["cls"] < 2]
        for p, (t, c) in zip(P, read(im0, [[[x / a.scale, y / a.scale] for x, y in p["quad"]] for p in P], model)):
            x, y = (round(v / a.scale) for v in ev.center(p["quad"]))
            print(f"{CLASSES[p['cls']]:7s} at ({x},{y})  box {p['conf']:.2f}  read {t or '-'}  conf {c:.2f}")
        raise SystemExit

    preds = json.loads(Path(a.preds).read_text())
    c, conf, wrong = defaultdict(Counter), [], Counter()
    for r in ev.load(a.data, a.split):
        name = Path(r["image"]).name
        if name not in preds:
            continue
        live = [l for l in r["labels"] if l["cls"] == "parcel" and not l["void"]]
        keys = lambda l: ("all", r["tier"], "rotated" if l["rotated"] else "upright", "thai" if l["thai"] else "arabic")
        for l in live:
            for k in keys(l):
                c[k]["gt"] += 1
        Q = [p for p in preds[name] if p["cls"] == 0 and p["conf"] >= a.conf]
        im = Image.open(Path(a.data) / r["image"]).convert("RGB")
        for p, (t, s) in zip(Q, read(im, [p["quad"] for p in Q], model)):
            p["read"], p["sure"] = ev.norm(t), s
        m = ev.match(Q, live)
        hit = {id(p): i for i, p in m.items()}
        for p in Q:
            l = live[hit[id(p)]] if id(p) in hit else None
            ok = l is not None and p["read"] == l["text"]
            conf.append((p["sure"], ok))
            if l is not None:
                for k in keys(l):
                    c[k].update(found=1, read=int(ok))
                if not ok and len(p["read"]) == len(l["text"]):
                    wrong.update((g, x) for g, x in zip(l["text"], p["read"]) if g != x)
    pct = lambda x, y: round(100 * x / y, 2) if y else None
    out = dict(parcel={k: dict(gt=v["gt"], found=pct(v["found"], v["gt"]), read=pct(v["read"], v["gt"])) for k, v in sorted(c.items())},
               confusions=[[*k, v] for k, v in wrong.most_common(10)],
               auto_pass={str(t): dict(coverage=pct(sum(o for s, o in conf if s >= t), c["all"]["gt"]),
                                       wrong=pct(sum(not o for s, o in conf if s >= t), sum(s >= t for s, _ in conf)),
                                       passed=sum(s >= t for s, _ in conf)) for t in (0, .8, .9, .95, .99)})
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
