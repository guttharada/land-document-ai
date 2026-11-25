"""One deed image -> suggested parcel number, for a person to confirm.  Runs locally."""
import json, math, os, subprocess, tempfile
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "true")
import numpy as np
from PIL import Image

import evaluate as ev
import yolo_tiles as T
from ocr_data import fit
from read_numbers import CLASSES, read
from train_ocr import load, predict

HERE = Path(__file__).parent
_models = {}


def models():
    if not _models:
        from ultralytics import YOLO
        _models["det"] = T.ultralytics_detector(YOLO(str(HERE / "best.pt")))
        _models["read"] = load(HERE / "ocr_thai.pt")
    return _models


def vision_lines(im):
    """Printed words found by Apple Vision, used only to locate form fields."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        im.save(p / "i.png")
        (p / "l.txt").write_text(str(p / "i.png"))
        subprocess.run([str(HERE / "vision_ocr"), str(p / "l.txt"), str(p / "o.jsonl"), "0", "0", "1"], check=True)
        return json.loads((p / "o.jsonl").read_text().splitlines()[0])["obs"]


def find_regions(im):
    """(map box, header number box), either None when the form words were not found."""
    W, H = im.size
    top = bottom = head = None
    for o in vision_lines(im):
        t = ev.norm(o["text"])
        xs, ys = [p[0] for p in o["quad"]], [p[1] for p in o["quad"]]
        if "มาตราส่วน" in t or "รูปแผนที่" in t:
            top = max(top or 0, max(ys))
        if ("ออกณ" in t or "พุทธศักราช" in t) and min(ys) > H * .5:
            bottom = min(bottom or H, min(ys))
        if t.startswith("เลขที่ดิน") and min(xs) < W * .5 and max(ys) < H * .45 and head is None:
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            cut = min(1, len("เลขที่ดิน") / max(len(t.rstrip(".")), 1))  # label and number may come as one line
            x0 = min(xs) + w * cut
            head = [x0, min(ys) - h * .25, min(W, x0 + max(w * cut * 1.6, h * 5)), max(ys) + h * .05]
    box = None
    if top is not None:  # inside the printed frame
        bottom = bottom if bottom and bottom > top + H * .1 else min(H * .85, top + H * .5)
        box = [W * .11, top + H * .01, W * .89, bottom - H * .005]
    return box, head


def trim_ink(crop):
    """Keep the written number: drop the dotted form line, light fibres and blank margins."""
    a = np.asarray(crop.convert("L"), np.float32)
    h, w = a.shape
    ink = a < np.median(a) - max(45, (np.median(a) - a.min()) * .45)
    dotted = (ink[:, 1:] != ink[:, :-1]).sum(1) > max(14, w * .12)  # form lines: many short runs in a row
    ink[dotted | np.roll(dotted, 1) | np.roll(dotted, -1)] = False
    cols = np.nonzero(ink.sum(0) >= max(1, h * .05))[0]
    if len(cols) == 0:
        return crop
    groups = np.split(cols, np.nonzero(np.diff(cols) > h * .7)[0] + 1)
    g = max(groups, key=lambda g: ink[:, g[0]:g[-1] + 1].sum())
    rows = np.nonzero(ink[:, g[0]:g[-1] + 1].sum(1))[0]
    p = h * .12
    return crop.crop((max(0, int(g[0] - p)), max(0, int(rows[0] - p)), min(w, int(g[-1] + p) + 1), min(h, int(rows[-1] + p) + 1)))


def detect(im, box):
    """YOLO on the map crop at several sizes; keep the size whose boxes look surest."""
    crop = im.crop([int(v) for v in box])
    best = (-1, 1, [])
    for s in (.75, 1, 1.5, 2):
        w, h = int(crop.width * s), int(crop.height * s)
        if max(w, h) > 3200 or (max(w, h) < 500 and s < 2):
            continue
        c = crop.resize((w, h), Image.LANCZOS)
        cv = Image.new("RGB", (max(w, T.TILE), max(h, T.TILE)), c.resize((1, 1)).getpixel((0, 0)))
        cv.paste(c, (0, 0))
        P = [p for p in T.predict_page(models()["det"], cv) if p["conf"] >= .25]
        score = sum(p["conf"] for p in P if p["cls"] in (0, 2) and p["conf"] >= .5)
        if score > best[0]:
            best = (score, s, P)
    _, s, P = best
    for p in P:
        p["quad"] = [[x / s + box[0], y / s + box[1]] for x, y in p["quad"]]
    return s, P


def hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def half(seq):
        h = []
        for p in seq:
            while len(h) > 1 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) <= (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]):
                h.pop()
            h.append(p)
        return h[:-1]
    return half(pts) + half(pts[::-1])


def pick_target(parcels, markers):
    """The parcel number nearest the middle of the boundary markers.  A guess for a person to check."""
    if not parcels:
        return None
    H = hull([ev.center(m["quad"]) for m in markers])
    if len(H) < 3:
        return max(range(len(parcels)), key=lambda i: parcels[i]["conf"])
    cx, cy = sum(p[0] for p in H) / len(H), sum(p[1] for p in H) / len(H)
    return min(range(len(parcels)), key=lambda i: math.dist(ev.center(parcels[i]["quad"]), (cx, cy)))


def analyse(im, map_box=None, head_box=None):
    im = im.convert("RGB")
    auto_map, auto_head = (None, None) if map_box and head_box else find_regions(im)
    src = dict(map="คนกำหนด" if map_box else "อัตโนมัติ" if auto_map else "ทั้งภาพ (หาไม่เจอ)",
               head="คนกำหนด" if head_box else "อัตโนมัติ" if auto_head else "หาไม่เจอ")
    map_box = map_box or auto_map or [0, 0, *im.size]
    head_box = head_box or auto_head
    scale, P = detect(im, map_box)
    parcels = [p for p in P if p["cls"] == 0]
    for p, (t, c) in zip(parcels, read(im, [p["quad"] for p in parcels], models()["read"])):
        p.update(text=t, value=ev.norm(t), sure=round(c, 2))
    k = pick_target(parcels, [p for p in P if p["cls"] == 2])
    head = None
    if head_box:
        crop = im.crop([int(v) for v in head_box])
        (t, c), = predict(models()["read"], fit(trim_ink(crop))[None], "cpu")
        head = dict(text=t, value=ev.norm(t), sure=round(c, 2))
    target = parcels[k]["value"] if k is not None else ""
    agree = bool(head and head["value"] and head["value"] == target)
    return dict(size=im.size, scale=scale, map_box=map_box, head_box=head_box, source=src,
                boxes=[dict(cls=CLASSES[p["cls"]], conf=round(p["conf"], 2), quad=p["quad"], text=p.get("text", ""),
                            value=p.get("value", ""), sure=p.get("sure", 0)) for p in parcels + [p for p in P if p["cls"] != 0]],
                target=k, head=head, agree=agree,
                suggestion=target or (head or {}).get("value", ""),
                advice="อ่านได้ตรงกันสองที่ แต่อาจผิดเหมือนกัน ดูภาพแล้วพิมพ์เลขที่ถูก" if agree else "ต้องตรวจ: ค่าไม่ตรงกันหรืออ่านได้ทางเดียว ดูภาพแล้วพิมพ์เลขที่ถูก")
