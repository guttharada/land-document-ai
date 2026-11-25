"""Baselines on v2, scored per label class.  Apple Vision, offline.

A whole  : full page once
B tiled  : overlapping tiles for the map (settings chosen on val)
C oracle : crop each parcel number from its TRUE box, upright  (not a usable system and not a
           ceiling: the 64 px recipe here is poor, see crop_variants.py)
C+map    : C, plus letters mapped to the digit they stand for (map learned on train crops)

    python3 run_v2.py --data data_v2 --work <temp dir>
"""
import argparse, json, math, random, re, subprocess, time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import evaluate as ev

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v2")
ap.add_argument("--work", required=True)
ap.add_argument("--val-per-tier", type=int, default=20)
ap.add_argument("--crops-per-page", type=int, default=10)
ap.add_argument("--train-crops", type=int, default=1500)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
DATA, WORK, OUT = Path(a.data).resolve(), Path(a.work), Path("results")
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)
reg = json.loads((DATA / "registry.json").read_text())
TH = "๐๑๒๓๔๕๖๗๘๙"
TOP = 350  # header / map boundary


def vision(paths, tag, tile=0, overlap=0, scale=1):
    lst, out = WORK / f"{tag}.txt", WORK / f"{tag}.jsonl"
    lst.write_text("\n".join(map(str, paths)))
    subprocess.run(["./vision_ocr", str(lst), str(out), str(tile), str(overlap), str(scale)], check=True)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == len(paths) and not any(r["error"] for r in recs)
    return {r["image"]: r["obs"] for r in recs}, sum(r["sec"] for r in recs) / len(recs)


def run(recs, tag, *cfg):
    p, sec = vision([DATA / r["image"] for r in recs], tag, *cfg)
    return {r["image"]: p[str(DATA / r["image"])] for r in recs}, sec


def combine(whole, tiled):
    return {k: [o for o in whole[k] if ev.center(o["quad"])[1] < TOP]
            + ev.dedupe([o for o in tiled[k] if ev.center(o["quad"])[1] >= TOP]) for k in whole}


def crop(im, q, path):
    w, h = ev.size(q)
    pad, s = h * .25, h / 64
    ux, uy, vx, vy = (q[1][0] - q[0][0]) / w, (q[1][1] - q[0][1]) / w, (q[3][0] - q[0][0]) / h, (q[3][1] - q[0][1]) / h
    W, H = int((w + 2 * pad) / s), int((h + 2 * pad) / s)
    bgc = im.getpixel((5, 5))
    c = im.transform((W, H), Image.AFFINE, (ux * s, vx * s, q[0][0] - pad * (ux + vx), uy * s, vy * s, q[0][1] - pad * (uy + vy)),
                     resample=Image.BICUBIC, fillcolor=bgc)
    bg = Image.new("RGB", (W + 120, H + 80), bgc)
    bg.paste(c, (60, 40))
    bg.save(path)


def squash(t):
    return re.sub(r"\s+", "", t)


def is_parcel(t, rule):
    t = squash(t)
    if rule == "any number":
        return bool(ev.NUM.match(ev.norm(t)))
    return 0 < len(t) <= 4 and all(c in TH for c in t)  # "thai digits"


def calls(obs, rec, rule):
    """Judge every box the rule would file as a parcel number."""
    live = [l for l in rec["labels"] if l["cls"] == "parcel" and not l["void"]]
    cand = [o for o in obs if ev.center(o["quad"])[1] >= TOP and is_parcel(o["text"], rule)]
    m = ev.match(cand, live)
    right = {id(o) for i, o in m.items() if ev.norm(o["text"]) == live[i]["text"]}
    mis = {id(o) for o in m.values()} - right
    out = []
    for o in cand:
        why = "right" if id(o) in right else "misread parcel" if id(o) in mis else None
        if why is None:
            for l in rec["labels"]:
                if ev.inside(ev.center(o["quad"]), l["quad"], 4):
                    why = "struck-out" if l["void"] else "duplicate/fragment" if l["cls"] == "parcel" else l["cls"]
                    break
        out.append((o, why or "nothing there"))
    return out


def score(pred, recs):
    """Per tier and per numeral system, only on pages of that system for parcels."""
    c = defaultdict(Counter)
    why = defaultdict(Counter)
    for r in recs:
        obs = pred[r["image"]]
        sysname = "thai" if r["thai_page"] else "arabic"
        groups = [r["tier"], "all"]
        live = [l for l in r["labels"] if l["cls"] == "parcel" and not l["void"]]
        g = ev.match(obs, live)
        for i, l in enumerate(live):
            ok = i in g and ev.norm(g[i]["text"]) == l["text"]
            keys = [f"parcel {sysname} {t}" for t in groups] + [f"parcel {t}" for t in groups]
            keys += ["parcel rotated"] if l["rotated"] else ["parcel upright"]
            keys += ["parcel " + l["role"]] if l["role"] == "target" else []
            for k in keys:
                c[k].update(n=1, found=int(i in g), read=int(ok))
        for l in r["labels"]:
            if l["cls"] not in ("marker", "length", "area"):
                continue
            near = [ev.norm(o["text"]) for o in obs if ev.inside(ev.center(o["quad"]), l["quad"], 3)]
            want = l["text"][-4:] if l["cls"] == "marker" else l["text"].split(" ")[0].replace("ม.", "")
            for k in (l["cls"], f"{l['cls']} {r['tier']}"):
                c[k].update(n=1, found=int(bool(near)), read=int(any(want in t for t in near)))
        for rule in ("any number", "thai digits"):
            if rule == "thai digits" and not r["thai_page"]:
                continue
            for o, w in calls(obs, r, rule):
                why[rule][w] += 1
        raw, snap = ev.header(obs, reg)
        for k, v in r["fields"].items():
            c["header " + k].update(n=1, found=int(bool(raw[k])), read=int(snap[k] == ev.norm(v)))
    pct = {k: dict(n=v["n"], found=round(100 * v["found"] / v["n"], 1), read=round(100 * v["read"] / v["n"], 1))
           for k, v in sorted(c.items())}
    rules = {k: dict(called=sum(v.values()), precision=round(100 * v["right"] / max(1, sum(v.values())), 1), **v)
             for k, v in why.items()}
    return dict(classes=pct, rules=rules)


t0 = time.time()
val, pick = ev.load(DATA, "val"), []
for t in "SHPQ":
    pick += random.Random(1).sample([r for r in val if r["tier"] == t], a.val_per_tier)
vw, _ = run(pick, "val_whole")
grid = []
for tile in (480, 640):
    for scale in (1, 2):
        vt, sec = run(pick, f"val_{tile}_{scale}", tile, 160, scale)
        s = score(combine(vw, vt), pick)
        R, P = s["classes"]["parcel thai all"]["read"], s["rules"]["thai digits"]["precision"]
        grid.append(dict(tile=tile, scale=scale, sec=round(sec, 2), read=R, precision=P, f1=round(2 * R * P / (R + P), 1)))
        print(grid[-1], flush=True)
best = max(grid, key=lambda g: g["f1"])

test = ev.load(DATA, "test")
test = random.Random(3).sample(test, a.limit) if a.limit else test
A, sa = run(test, "test_whole")
tl, sb = run(test, "test_tiled", best["tile"], 160, best["scale"])
B = combine(A, tl)
res = dict(A_whole=score(A, test), B_tiled=score(B, test))
print("whole/tiled done", round((time.time() - t0) / 60, 1), flush=True)

# oracle crops: test sample, and a train sample to learn letter -> digit look-alikes
cdir = WORK / "crops"
cdir.mkdir(exist_ok=True)


def crops(recs, per_page, cap, seed):
    r_, items = random.Random(seed), []
    for r in recs:
        live = [i for i, l in enumerate(r["labels"]) if l["cls"] == "parcel" and not l["void"]]
        if cap and len(items) >= cap:
            break
        im = Image.open(DATA / r["image"]).convert("RGB")
        for i in r_.sample(live, min(per_page, len(live))):
            p = cdir / f"{Path(r['image']).stem}_{i}.png"
            crop(im, r["labels"][i]["quad"], p)
            items.append((r, i, p))
    return items


def joined(obs):
    return squash("".join(o["text"] for o in sorted(obs, key=lambda o: o["quad"][0][0])))


train = ev.load(DATA, "train")
random.Random(5).shuffle(train)
tr = [x for x in crops(train, 4, a.train_crops, 6) if x[0]["thai_page"]]
tp, _ = vision([p for *_, p in tr], "train_crops")
votes = defaultdict(Counter)
for r, i, p in tr:
    t, g = joined(tp[str(p)]).translate(ev.TH), r["labels"][i]["text"]
    if len(t) == len(g):
        for x, y in zip(t, g):
            if not x.isdigit():
                votes[x][y] += 1
lookalike = {x: v.most_common(1)[0][0] for x, v in votes.items() if sum(v.values()) >= 3 and v.most_common(1)[0][1] / sum(v.values()) >= .6}

it = crops(test, a.crops_per_page, 0, 2)
cp, sc = vision([p for *_, p in it], "test_crops")
gm = {}
for name, P in (("A", A), ("B", B)):
    for r in test:
        live = [(i, l) for i, l in enumerate(r["labels"]) if l["cls"] == "parcel" and not l["void"]]
        for k, o in ev.match(P[r["image"]], [l for _, l in live]).items():
            gm[name, r["image"], live[k][0]] = ev.norm(o["text"])
same, conf = defaultdict(Counter), Counter()
for r, i, p in it:
    l = r["labels"][i]
    c = joined(cp[str(p)]).translate(ev.TH)
    c2 = "".join(lookalike.get(x, x) for x in c) if l["thai"] else c
    if l["thai"] and len(c2) == len(l["text"]):
        conf.update((g, x) for g, x in zip(l["text"], c2) if g != x)
    num = "thai" if l["thai"] else "arabic"
    for k in ("all", num, f"{num} {'hand' if r['tier'] in 'HQ' else 'printed'}", f"{num} tier {r['tier']}",
              "rotated" if l["rotated"] else "upright"):
        same[k].update(n=1, A=int(gm.get(("A", r["image"], i)) == l["text"]), B=int(gm.get(("B", r["image"], i)) == l["text"]),
                       C=int(c == l["text"]), C_map=int(c2 == l["text"]))
same = {k: dict(n=v["n"], **{x: round(100 * v[x] / v["n"], 1) for x in ("A", "B", "C", "C_map")}) for k, v in sorted(same.items())}

# auto-pass: which boxes may skip a human, and how many of those are wrong (Thai pages)
ap_, total = defaultdict(Counter), 0
for r in test:
    if not r["thai_page"]:
        continue
    total += sum(l["cls"] == "parcel" and not l["void"] for l in r["labels"])
    _, snap = ev.header(B[r["image"]], reg)
    valid = set(map(str, reg.get(snap["amphoe"], {}).get(snap["tambon"], [])))
    an = [(ev.center(o["quad"]), ev.norm(o["text"])) for o in A[r["image"]]]
    for o, w in calls(B[r["image"]], r, "thai digits"):
        t, c = ev.norm(o["text"]), ev.center(o["quad"])
        agree = any(t == x and math.dist(c, y) < 25 for y, x in an)
        for name, cond in (("all", True), ("in registry", t in valid), ("A agrees", agree), ("A agrees + registry", agree and t in valid)):
            if cond:
                ap_[name][w] += 1
auto = {k: dict(passed=sum(v.values()), coverage=round(100 * v["right"] / total, 1),
                wrong=round(100 * (1 - v["right"] / sum(v.values())), 2), **v) for k, v in ap_.items()}

out = dict(val_grid=grid, chosen=best, test_pages=len(test), thai_pages=sum(r["thai_page"] for r in test),
           sec_per_page=dict(A=round(sa, 2), B=round(sa + sb, 2)), test=res, same_labels=same,
           lookalike_map=lookalike, remaining_confusions=[[*k, v] for k, v in conf.most_common(12)],
           autopass=auto, minutes=round((time.time() - t0) / 60, 1))
(OUT / "baseline_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
