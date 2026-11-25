"""Score YOLO-OBB predictions on this Mac: detection, then 40 px crop + Apple Vision.

    python3 score_yolo.py --preds preds.json --split val --work <temp dir>

preds.json comes from the notebook: {page file name: [{cls, conf, quad}]}.
OBB boxes have no up/down direction, so a crop is read both ways and the reading that is
in the registry (then the surer one) wins.  Use val while developing; test only once.
"""
import argparse, json, math, random, subprocess
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import evaluate as ev

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v2")
ap.add_argument("--split", default="val")
ap.add_argument("--preds", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--conf", type=float, default=.25)
ap.add_argument("--out", default="")
ap.add_argument("--no-registry", action="store_true", help="choose the reading direction without the registry (fair against A/B)")
ap.add_argument("--baseline", action="store_true", help="also score Apple Vision alone on the same pages")
a = ap.parse_args()
DATA, WORK = Path(a.data).resolve(), Path(a.work)
WORK.mkdir(parents=True, exist_ok=True)
CLASSES = ["parcel", "struck", "marker", "length", "area", "text"]
preds = json.loads(Path(a.preds).read_text())
recs = [r for r in ev.load(DATA, a.split) if Path(r["image"]).name in preds]
assert recs, "no page in --preds belongs to this split"
missing = len(ev.load(DATA, a.split)) - len(recs)
reg = json.loads((DATA / "registry.json").read_text())


def true_cls(l):
    return ("struck" if l["void"] else "parcel") if l["cls"] == "parcel" else l["cls"] if l["cls"] in CLASSES else "text"


def upright(q):
    """Corners reordered so q[0]->q[1] is the long side, read left to right or bottom to top."""
    # Ultralytics lists corners counter-clockwise; without this every crop is mirrored
    if sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4)) < 0:
        q = q[::-1]
    if math.dist(q[0], q[1]) < math.dist(q[1], q[2]):
        q = q[1:] + q[:1]
    ang = math.degrees(math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]))
    if not -100 <= ang < 80:  # text normally runs right or upwards
        q = q[2:] + q[:2]
    return q


def crop40(im, q, path):
    w, h = ev.size(q)
    h = max(h, 4)
    pad, s = h * .25, h / 40
    ux, uy, vx, vy = (q[1][0] - q[0][0]) / w, (q[1][1] - q[0][1]) / w, (q[3][0] - q[0][0]) / h, (q[3][1] - q[0][1]) / h
    W, H = max(8, int((w + 2 * pad) / s)), max(8, int((h + 2 * pad) / s))
    bgc = im.getpixel((5, 5))
    c = im.transform((W, H), Image.AFFINE, (ux * s, vx * s, q[0][0] - pad * (ux + vx), uy * s, vy * s, q[0][1] - pad * (uy + vy)),
                     resample=Image.BICUBIC, fillcolor=bgc)
    bg = Image.new("RGB", (W + 120, H + 80), bgc)
    bg.paste(c, (60, 40))
    bg.save(path)


det, jobs, matched = defaultdict(Counter), [], {}
for r in recs:
    P = [p for p in preds[Path(r["image"]).name] if p["conf"] >= a.conf]
    for name in CLASSES:
        L = [l for l in r["labels"] if true_cls(l) == name]
        Q = [p for p in P if CLASSES[p["cls"]] == name]
        m = ev.match(Q, L)
        for k in (name, f"{name} {r['tier']}"):
            det[k].update(gt=len(L), pred=len(Q), hit=len(m))
        if name == "parcel":
            where = {id(p): i for i, p in m.items()}
            for p in Q:
                jobs.append((r, p, L[where[id(p)]] if id(p) in where else None))
cache, paths = {}, []
for n, (r, p, _) in enumerate(jobs):
    if cache.get("k") != r["image"]:
        cache = dict(k=r["image"], im=Image.open(DATA / r["image"]).convert("RGB"))
    q = upright([list(x) for x in p["quad"]])
    for d, qq in (("a", q), ("b", q[2:] + q[:2])):
        paths.append(WORK / f"{n}{d}.png")
        crop40(cache["im"], qq, paths[-1])
(WORK / "l.txt").write_text("\n".join(map(str, paths)))
subprocess.run(["./vision_ocr", str(WORK / "l.txt"), str(WORK / "o.jsonl"), "0", "0", "1"], check=True)
obs = [json.loads(l)["obs"] for l in (WORK / "o.jsonl").read_text().splitlines()]
assert len(obs) == len(paths)
for p in paths:
    p.unlink()


def read(o):
    t = "".join(x["text"] for x in sorted(o, key=lambda x: x["quad"][0][0]))
    return ev.norm(t), min([x["conf"] for x in o], default=0)


c = defaultdict(Counter)
for r in recs:
    for l in r["labels"]:
        if true_cls(l) == "parcel":
            for k in ("all", r["tier"], "rotated" if l["rotated"] else "upright", "thai" if l["thai"] else "arabic"):
                c[k]["gt"] += 1
called, conf_pairs, kinds = Counter(), Counter(), Counter()
for n, (r, p, l) in enumerate(jobs):
    (ta, ca), (tb, cb) = read(obs[2 * n]), read(obs[2 * n + 1])
    valid = set(map(str, reg[r["fields"]["amphoe"]][r["fields"]["tambon"]]))
    t = max(((ta, ca, 1), (tb, cb, 0)), key=lambda x: (x[0] in valid and not a.no_registry, bool(ev.NUM.match(x[0])), x[1], x[2]))[0]
    ok = l is not None and t == l["text"]
    if l is not None and not ok and len(t) == len(l["text"]):
        conf_pairs.update((g, x) for g, x in zip(l["text"], t) if g != x)
    if l is not None and not ok:
        kinds["no digits read" if not ev.NUM.match(t) and not any(ch.isdigit() for ch in t) else "right length, wrong digit" if len(t) == len(l["text"]) else "wrong length"] += 1
    called.update(n=1, right=int(ok), in_reg=int(t in valid), in_reg_right=int(ok and t in valid))
    if l is not None:
        for k in ("all", r["tier"], "rotated" if l["rotated"] else "upright", "thai" if l["thai"] else "arabic"):
            c[k].update(found=1, read=int(ok))
pct = lambda x, y: round(100 * x / y, 1) if y else None
out = dict(split=a.split, pages=len(recs), pages_without_prediction=missing, conf=a.conf,
           detection={k: dict(gt=v["gt"], recall=pct(v["hit"], v["gt"]), precision=pct(v["hit"], v["pred"])) for k, v in sorted(det.items())},
           parcel={k: dict(gt=v["gt"], found=pct(v["found"], v["gt"]), read=pct(v["read"], v["gt"])) for k, v in sorted(c.items())},
           called=dict(called, precision=pct(called["right"], called["n"]), wrong_if_in_registry=pct(called["in_reg"] - called["in_reg_right"], called["in_reg"])),
           misreads=dict(kinds), confusions=[[*k, v] for k, v in conf_pairs.most_common(12)],
           note="no registry used" if a.no_registry else "registry lookup uses the TRUE tambon, so this is slightly optimistic")
if a.baseline:  # A whole page and B tiles (480 px x2) on the very same pages, no training
    paths = [DATA / r["image"] for r in recs]
    (WORK / "p.txt").write_text("\n".join(map(str, paths)))
    P = {}
    for n, cfg in (("A_whole", ("0", "0", "1")), ("tiles", ("480", "160", "2"))):
        subprocess.run(["./vision_ocr", str(WORK / "p.txt"), str(WORK / f"{n}.jsonl"), *cfg], check=True)
        P[n] = {x["image"]: x["obs"] for x in map(json.loads, (WORK / f"{n}.jsonl").read_text().splitlines())}
    P["B_tiled"] = {k: ev.dedupe([o for o in v if ev.center(o["quad"])[1] >= 350]) for k, v in P.pop("tiles").items()}
    out["baseline"] = {}
    for n, pred in P.items():
        b = defaultdict(Counter)
        for r in recs:
            L = [l for l in r["labels"] if true_cls(l) == "parcel"]
            m = ev.match(pred[str(DATA / r["image"])], L)
            for i, l in enumerate(L):
                for k in ("all", r["tier"], "rotated" if l["rotated"] else "upright", "thai" if l["thai"] else "arabic"):
                    b[k].update(gt=1, found=int(i in m), read=int(i in m and ev.norm(m[i]["text"]) == l["text"]))
        out["baseline"][n] = {k: dict(gt=v["gt"], found=pct(v["found"], v["gt"]), read=pct(v["read"], v["gt"])) for k, v in sorted(b.items())}
print(json.dumps(out, ensure_ascii=False, indent=1))
if a.out:
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
