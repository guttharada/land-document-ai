"""How should one parcel number be cropped before OCR?  Variants compared on VAL only;
the best one (by Thai accuracy) then runs once on the same test labels as run_v2's arm C.

    python3 crop_variants.py --work <temp dir>
"""
import argparse, json, math, os, random, re, subprocess
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

import evaluate as ev

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v2")
ap.add_argument("--work", required=True)
ap.add_argument("--per-tier", type=int, default=300)
a = ap.parse_args()
DATA, WORK = Path(a.data).resolve(), Path(a.work)
WORK.mkdir(parents=True, exist_ok=True)
ANCHOR = ImageFont.truetype("/System/Library/Fonts/Supplemental/Silom.ttf", 64)

# pad: margin in text heights · h: text height px · flat: plain border px · anchor: Thai word at the left
V = {
    "base (pad .25, 64px)": dict(),
    "40px": dict(h=40),
    "96px": dict(h=96),
    "pad .6": dict(pad=.6),
    "context 1.0": dict(pad=1.0, flat=0),
    "context 2.0": dict(pad=2.0, flat=0),
    "context 2.0, 40px": dict(pad=2.0, flat=0, h=40),
    "autocontrast": dict(auto=True),
    "thai only": dict(langs="th-TH"),
    "thai word anchor": dict(anchor=True),
    "anchor + thai only": dict(anchor=True, langs="th-TH"),
    "anchor, 40px": dict(anchor=True, h=40),
    "anchor, pad .6": dict(anchor=True, pad=.6),
    "anchor, pad .6, 40px": dict(anchor=True, pad=.6, h=40),
}


def make(im, q, path, pad=.25, h=64, flat=60, auto=False, anchor=False, langs=None):
    """Write the crop, return the box (x0, y0, x1, y1) the number occupies in it."""
    tw, th = ev.size(q)
    s = th / h
    p = th * pad
    ux, uy, vx, vy = (q[1][0] - q[0][0]) / tw, (q[1][1] - q[0][1]) / tw, (q[3][0] - q[0][0]) / th, (q[3][1] - q[0][1]) / th
    W, H = int((tw + 2 * p) / s), int((th + 2 * p) / s)
    bgc = im.getpixel((5, 5))
    c = im.transform((W, H), Image.AFFINE, (ux * s, vx * s, q[0][0] - p * (ux + vx), uy * s, vy * s, q[0][1] - p * (uy + vy)),
                     resample=Image.BICUBIC, fillcolor=bgc)
    left = flat
    if anchor:
        word = "เลขที่ดิน"
        aw = int(ANCHOR.getlength(word))
        left = flat + aw + 40
    bg = Image.new("RGB", (W + left + flat, H + 2 * int(flat * .67)), bgc)
    if anchor:
        ImageDraw.Draw(bg).text((flat, (bg.height - 90) // 2), word, font=ANCHOR, fill=(25, 25, 30))
    top = int(flat * .67)
    bg.paste(c, (left, top))
    if auto:
        bg = ImageOps.autocontrast(bg.convert("L"), cutoff=1).convert("RGB")
    bg.save(path)
    m = p / s
    return left + m, top + m, left + W - m, top + H - m


def vision(paths, tag, langs):
    lst, out = WORK / f"{tag}.txt", WORK / f"{tag}.jsonl"
    lst.write_text("\n".join(map(str, paths)))
    env = dict(os.environ, **({"OCR_LANGS": langs} if langs else {}))
    subprocess.run(["./vision_ocr", str(lst), str(out), "0", "0", "1"], check=True, env=env)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == len(paths) and not any(r["error"] for r in recs)
    return [r["obs"] for r in recs]


def read(obs, box):
    """Text whose centre lies on the number itself (ignores neighbours and the anchor word)."""
    x0, y0, x1, y1 = box
    my = (y1 - y0) * .35
    hit = [o for o in obs if x0 - my <= ev.center(o["quad"])[0] <= x1 + my and y0 - my <= ev.center(o["quad"])[1] <= y1 + my]
    if not hit:  # anchor and number returned as one line
        hit = [o for o in obs if y0 - my <= ev.center(o["quad"])[1] <= y1 + my]
    t = "".join(o["text"] for o in sorted(hit, key=lambda o: o["quad"][0][0]))
    return re.sub(r"\s+", "", t).replace("เลขที่ดิน", "").translate(ev.TH)


def run(name, items, tag):
    cfg = V[name]
    d = WORK / tag
    d.mkdir(exist_ok=True)
    cache, paths, boxes = {}, [], []
    for r, i in items:
        if cache.get("k") != r["image"]:
            cache = dict(k=r["image"], im=Image.open(DATA / r["image"]).convert("RGB"))
        paths.append(d / f"{Path(r['image']).stem}_{i}.png")
        boxes.append(make(cache["im"], r["labels"][i]["quad"], paths[-1], **cfg))
    out = [read(o, b) for o, b in zip(vision(paths, tag, cfg.get("langs")), boxes)]
    for p in paths:
        p.unlink()
    return out


def live(r):
    return [i for i, l in enumerate(r["labels"]) if l["cls"] == "parcel" and not l["void"]]


def table(items, texts):
    c = defaultdict(Counter)
    for (r, i), t in zip(items, texts):
        l = r["labels"][i]
        num = "thai" if l["thai"] else "arabic"
        for k in (num, f"{num} {r['tier']}", f"{num} {'rotated' if l['rotated'] else 'upright'}"):
            c[k].update(n=1, ok=int(t == l["text"]))
    return {k: (v["n"], round(100 * v["ok"] / v["n"], 1)) for k, v in sorted(c.items())}


val, rnd, items = ev.load(DATA, "val"), random.Random(11), []
for t in "SHPQ":
    pool = [(r, i) for r in val if r["tier"] == t for i in live(r)]
    items += rnd.sample(pool, min(a.per_tier, len(pool)))
items.sort(key=lambda x: x[0]["image"])
res = {}
for k, name in enumerate(V):
    res[name] = table(items, run(name, items, f"val{k}"))
    print(f"{name:24s}", {x: res[name][x][1] for x in res[name] if "upright" not in x}, flush=True)

# balanced over tiers so dense sheets do not decide alone
score = {n: sum(v[f"thai {t}"][1] for t in "SHPQ") / 4 for n, v in res.items()}
best = max(score, key=score.get)
print("best on val:", best, round(score[best], 1), "| base", round(score["base (pad .25, 64px)"], 1), flush=True)

test, r_, ti = ev.load(DATA, "test"), random.Random(2), []
for r in test:  # same draw as run_v2 arm C
    L = live(r)
    ti += [(r, i) for i in r_.sample(L, min(10, len(L)))]
out = dict(val_n=len(items), val=res, val_score=score, best=best, test_n=len(ti),
           test_best=table(ti, run(best, ti, "test_best")), test_base=table(ti, run("base (pad .25, 64px)", ti, "test_base")))
Path("results/crop_variants.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("test base", out["test_base"])
print("test best", out["test_best"])
