"""Three baselines, all offline with Apple Vision.

A whole   : OCR the full page once
B tiled   : overlapping enlarged tiles for the map (header still from A)
C oracle  : crop each number using the TRUE box, rotate upright, then OCR.
            Not a usable system: it is the ceiling a perfect detector would give.

Tile settings are chosen on val, then every arm runs once on test.
    python3 run_baselines.py --data data --work <temp dir>
"""
import argparse, json, random, subprocess, time
from pathlib import Path

from PIL import Image

import evaluate as ev

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--work", required=True)
ap.add_argument("--val-per-tier", type=int, default=20)
ap.add_argument("--crops-per-page", type=int, default=10)
ap.add_argument("--limit", type=int, default=0, help="test pages (0 = all), for smoke runs")
a = ap.parse_args()
DATA, WORK = Path(a.data).resolve(), Path(a.work)
WORK.mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)
reg = json.loads((DATA / "registry.json").read_text())


def vision(paths, tag, tile, overlap, scale):
    lst, out = WORK / f"{tag}.txt", WORK / f"{tag}.jsonl"
    lst.write_text("\n".join(map(str, paths)))
    subprocess.run(["./vision_ocr", str(lst), str(out), str(tile), str(overlap), str(scale)], check=True)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    bad = [r["image"] for r in recs if r["error"]]
    assert len(recs) == len(paths) and not bad, (len(recs), len(paths), bad[:3])
    return {r["image"]: r["obs"] for r in recs}, sum(r["sec"] for r in recs) / len(recs)


def run(recs, tag, tile=0, overlap=0, scale=1):
    p, sec = vision([DATA / r["image"] for r in recs], tag, tile, overlap, scale)
    return {r["image"]: p[str(DATA / r["image"])] for r in recs}, sec


def combine(whole, tiled):
    """Header from the whole page, map numbers from deduplicated tiles."""
    return {k: [o for o in whole[k] if ev.center(o["quad"])[1] < 350]
            + ev.dedupe([o for o in tiled[k] if ev.center(o["quad"])[1] >= 350]) for k in whole}


def crop(im, q, path):
    """Rectified upright crop, 64 px text height, padded."""
    w, h = ev.size(q)
    pad = h * .25
    ux, uy = (q[1][0] - q[0][0]) / w, (q[1][1] - q[0][1]) / w
    vx, vy = (q[3][0] - q[0][0]) / h, (q[3][1] - q[0][1]) / h
    s = h / 64
    W, H = int((w + 2 * pad) / s), int((h + 2 * pad) / s)
    ox, oy = q[0][0] - pad * (ux + vx), q[0][1] - pad * (uy + vy)
    c = im.transform((W, H), Image.AFFINE, (ux * s, vx * s, ox, uy * s, vy * s, oy),
                     resample=Image.BICUBIC, fillcolor=im.getpixel((5, 5)))
    bg = Image.new("RGB", (W + 120, H + 80), im.getpixel((5, 5)))
    bg.paste(c, (60, 40))
    bg.save(path)


t_start = time.time()
# 1) choose tile settings on val
val, pick = ev.load(DATA, "val"), []
for t in "ABC":
    pick += random.Random(1).sample([r for r in val if r["tier"] == t], a.val_per_tier)
vw, _ = run(pick, "val_whole")
grid = []
for tile in (480, 640, 960):
    for scale in (1, 2):
        vt, sec = run(pick, f"val_{tile}_{scale}", tile, 160, scale)
        s = ev.score(combine(vw, vt), pick, reg)["all"]
        grid.append(dict(tile=tile, scale=scale, sec=round(sec, 2), read=s["read"],
                         found=s["found"], precision=s["precision"],
                         f1=round(2 * s["read"] * s["precision"] / (s["read"] + s["precision"]), 1)))
        print(grid[-1], flush=True)
best = max(grid, key=lambda g: g["f1"])

# 2) test, once
test = ev.load(DATA, "test")[: a.limit or None]
whole, sa = run(test, "test_whole")
tiled, sb = run(test, "test_tiled", best["tile"], 160, best["scale"])
B = combine(whole, tiled)
results = {"A_whole": ev.score(whole, test, reg), "B_tiled": ev.score(B, test, reg)}

# 3) oracle crops, and A/B on the same sampled labels
cdir = WORK / "crops"
cdir.mkdir(exist_ok=True)
items, r_ = [], random.Random(2)
for r in test:
    im = Image.open(DATA / r["image"]).convert("RGB")
    live = [i for i, l in enumerate(r["labels"]) if not l["void"]]
    for i in r_.sample(live, min(a.crops_per_page, len(live))):
        p = cdir / f"{Path(r['image']).stem}_{i}.png"
        crop(im, r["labels"][i]["quad"], p)
        items.append((r, i, p))
cp, sc = vision([p for *_, p in items], "test_crops", 0, 0, 1)
same = {}
gA, gB = {}, {}
for r in test:
    live = [(i, l) for i, l in enumerate(r["labels"]) if not l["void"]]
    for g, P in ((gA, whole), (gB, B)):
        m = ev.match(P[r["image"]], [l for _, l in live])
        g[r["image"]] = {live[k][0]: o["text"] for k, o in m.items()}
for r, i, p in items:
    gt = r["labels"][i]["text"]
    c = "".join(o["text"] for o in sorted(cp[str(p)], key=lambda o: o["quad"][0][0]))
    keys = [r["tier"], "all"] + (["rotated"] if r["labels"][i]["rotated"] else [])
    keys += ["thai_digit"] if r["labels"][i]["thai"] else []
    for k in keys:
        s = same.setdefault(k, dict(n=0, A=0, B=0, C=0))
        s["n"] += 1
        s["A"] += ev.norm(gA[r["image"]].get(i, "")) == gt
        s["B"] += ev.norm(gB[r["image"]].get(i, "")) == gt
        s["C"] += ev.norm(c) == gt
for s in same.values():
    for k in "ABC":
        s[k] = round(100 * s[k] / s["n"], 1)

# 4) auto-pass rules on B's numbers: how many pass, how many of those are wrong
rules = {k: [0, 0, 0] for k in ("all numbers", "in registry", "A agrees", "A agrees + in registry")}
total = 0
for r in test:
    live = [l for l in r["labels"] if not l["void"]]
    void = [l for l in r["labels"] if l["void"]]
    nums = [o for o in B[r["image"]] if ev.NUM.match(ev.norm(o["text"])) and ev.center(o["quad"])[1] > 350]
    total += len(live)
    ok = {id(o) for i, o in ev.match(nums, live).items() if ev.norm(o["text"]) == live[i]["text"]}
    vd = {id(o) for o in ev.match(nums, void).values()} - ok
    _, snap = ev.header(B[r["image"]], reg)
    valid = set(map(str, reg.get(snap["amphoe"], {}).get(snap["tambon"], [])))
    an = [(ev.center(o["quad"]), ev.norm(o["text"])) for o in whole[r["image"]]]
    for o in nums:
        t, c = ev.norm(o["text"]), ev.center(o["quad"])
        agree = any(t == x and ev.math.dist(c, y) < 25 for y, x in an)
        for name, cond in (("all numbers", True), ("in registry", t in valid), ("A agrees", agree),
                           ("A agrees + in registry", agree and t in valid)):
            if cond:
                rules[name][0] += 1
                rules[name][1] += id(o) not in ok
                rules[name][2] += id(o) in vd
autopass = {k: dict(passed=v[0], coverage=round(100 * (v[0] - v[1]) / total, 1),
                    wrong=round(100 * v[1] / v[0], 2) if v[0] else None, wrong_count=v[1],
                    wrong_struck_out=v[2]) for k, v in rules.items()}

out = dict(val_grid=grid, chosen=best, sec_per_page=dict(A=round(sa, 2), B=round(sa + sb, 2)),
           sec_per_crop=round(sc, 3), test=results, same_labels=same, autopass=autopass,
           test_pages=len(test), minutes=round((time.time() - t_start) / 60, 1))
Path("results/baseline.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
