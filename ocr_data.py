"""Training images for the number reader: 48x192 grey, 1-4 digits, Thai or Arabic.

page  : parcel numbers cut from data_v2 (train or val) with a jittered box
pen   : pseudo-handwriting. Glyph skeletons are bent, redrawn as one pen stroke, joined and
        given long tails, as seen on real deeds.  Still not real handwriting.
Fonts here never include the val fonts (Thonburi, Sathu) or test fonts (ChulaCharas, Arial Unicode).

    .venv/bin/python ocr_data.py --work <temp dir>
"""
import argparse, json, math, random
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from generate import STROKES, spline

H, W = 48, 192
TH = "๐๑๒๓๔๕๖๗๘๙"
VOCAB = TH + "0123456789"  # index + 1 in CTC, 0 = blank
SUP, LIB = "/System/Library/Fonts/Supplemental/", str(Path.home() / "Library/Fonts") + "/"
FONTS = [SUP + f for f in ("Silom.ttf", "Ayuthaya.ttf", "Tahoma.ttf", "Microsoft Sans Serif.ttf", "Krungthep.ttf", "SukhumvitSet.ttc")]
FONTS += [LIB + f for f in ("Anuphan-Light.ttf", "Anuphan-Regular.ttf", "LINESeedSansTH_Rg.ttf", "LINESeedSansTH_Th.ttf")]
FONTS = [f for f in FONTS if Path(f).exists()]


def fit(im):
    """Grey, contrast-stretched, scaled to fit 48x192, padded with the background."""
    a = np.asarray(im.convert("L"), np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 90)
    a = np.clip((a - lo) / max(hi - lo, 8) * 255, 0, 255)
    im = Image.fromarray(a.astype(np.uint8))
    k = min((H - 8) / im.height, (W - 8) / im.width)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.BILINEAR)
    out = Image.new("L", (W, H), 255)
    out.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return np.asarray(out, np.uint8)


def cut(im, q, pad=.25, height=40):
    """Rectified crop of quad q (q[0]->q[1] runs along the text)."""
    w, h = math.dist(q[0], q[1]), max(4, math.dist(q[1], q[2]))
    s, p = h / height, h * pad
    ux, uy, vx, vy = (q[1][0] - q[0][0]) / w, (q[1][1] - q[0][1]) / w, (q[3][0] - q[0][0]) / h, (q[3][1] - q[0][1]) / h
    size = max(8, int((w + 2 * p) / s)), max(8, int((h + 2 * p) / s))
    return im.transform(size, Image.AFFINE, (ux * s, vx * s, q[0][0] - p * (ux + vx), uy * s, vy * s, q[0][1] - p * (uy + vy)),
                        resample=Image.BICUBIC, fillcolor=im.getpixel((5, 5)))


def thin(b):
    """Zhang-Suen thinning of a boolean array."""
    b = np.pad(b, 1).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p = [np.roll(b, s, (0, 1)) for s in ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))]
            n = sum(x.astype(int) for x in p)
            t = sum(((p[i] == 0) & (p[(i + 1) % 8] == 1)).astype(int) for i in range(8))
            c = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0) if step == 0 else (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            kill = (b == 1) & (n >= 2) & (n <= 6) & (t == 1) & c
            if kill.any():
                b[kill] = 0
                changed = True
        if not changed:
            return b[1:-1, 1:-1].astype(bool)


@lru_cache(None)
def skeleton(font, ch):
    """Skeleton points (x, y in units of glyph height) and endpoints with outgoing direction."""
    f = ImageFont.truetype(font, 80)
    b = f.getbbox(ch)
    im = Image.new("L", (b[2] - b[0] + 8, b[3] - b[1] + 8), 0)
    ImageDraw.Draw(im).text((4 - b[0], 4 - b[1]), ch, font=f, fill=255)
    ys, xs = np.nonzero(thin(np.asarray(im) > 110))
    h = max(ys.max() - ys.min(), 1)
    P = np.stack([xs - xs.min(), ys - ys.min()], 1) / h
    ends = []
    S = set(zip(xs.tolist(), ys.tolist()))
    for x, y in S:
        if sum((x + i, y + j) in S for i in (-1, 0, 1) for j in (-1, 0, 1)) == 2:
            near = [(a, c) for a, c in S if 3 <= math.hypot(a - x, c - y) <= 7]
            if near:
                ax, ay = x - np.mean([n[0] for n in near]), y - np.mean([n[1] for n in near])
                k = math.hypot(ax, ay) or 1
                ends.append(((x - xs.min()) / h, (y - ys.min()) / h, ax / k, ay / k))
    return P, ends


def pen_text(text, r):
    """Draw text as joined pen strokes; returns an ink mask (L)."""
    hh = 110
    im = Image.new("L", (int(hh * 1.25 * (len(text) + 2)), hh * 4), 0)
    d = ImageDraw.Draw(im)
    font, rad = r.choice(FONTS), r.uniform(2.2, 6)
    slant, sq = r.uniform(-.25, .5), r.uniform(.75, 1.2)
    long_tail, x = r.random() < .5, hh * 1.0
    for ch in text:
        k = r.uniform(.85, 1.15) * hh
        y0 = hh * 1.5 + r.uniform(-.1, .1) * hh
        a1, a2, f1, f2 = r.uniform(0, .07), r.uniform(0, .07), r.uniform(2, 5), r.uniform(2, 5)
        p1, p2, rot = r.uniform(0, 6.3), r.uniform(0, 6.3), r.gauss(0, .06)

        def place(u, v):
            u2, v2 = u + a1 * math.sin(f1 * v + p1), v + a2 * math.sin(f2 * u + p2)
            u2 += slant * (1 - v2)
            c, s = math.cos(rot), math.sin(rot)
            return x + (u2 * c - v2 * s) * k * sq, y0 + (u2 * s + v2 * c) * k

        if ch in STROKES and r.random() < .7:
            width = .75
            for st in STROKES[ch]:
                d.line([place(px * .6 + r.gauss(0, .02), py + r.gauss(0, .02)) for px, py in spline(st, 6)],
                       fill=255, width=int(2 * rad), joint="curve")
        else:
            P, ends = skeleton(font, ch)
            width = P[:, 0].max() + .12
            for u, v in P:
                a, b = place(u, v)
                d.ellipse((a - rad, b - rad, a + rad, b + rad), fill=255)
            for u, v, ax, ay in ends:  # tails on ๓ ๔ ๕ ๖ ๗ ๘ ๙ get longer, as in quick handwriting
                if ch in TH and v < .45 and r.random() < (.75 if long_tail else .15):
                    n, L, bend = 14, r.uniform(.1, .75), r.uniform(-1.2, 1.2)
                    ang = math.atan2(ay, ax)
                    pts, cu, cv = [place(u, v)], u, v
                    for i in range(n):
                        ang += bend / n
                        cu, cv = cu + math.cos(ang) * L / n, cv + math.sin(ang) * L / n
                        pts.append(place(cu, cv))
                    d.line(pts, fill=255, width=max(2, int(1.7 * rad)), joint="curve")
        x += (width * sq + r.uniform(-.1, .28)) * k
    return im.crop(im.getbbox())


def pen_sample(seed):
    r, nr = random.Random(seed), np.random.default_rng(seed)
    thai = r.random() < .8
    n = r.choices([1, 2, 3, 4], [.08, .2, .42, .3])[0]
    v = str(r.randint(10 ** (n - 1) if n > 1 else 0, 10 ** n - 1))
    text = "".join(TH[int(c)] for c in v) if thai else v
    m = pen_text(text, r)
    if r.random() < .5:  # YOLO boxes often hug the digit bodies and cut the tails
        top = int(m.height * r.uniform(0, .25))
        m = m.crop((0, top, m.width, m.height))
    pw, ph = int(m.height * r.uniform(.05, .35)), int(m.height * r.uniform(.05, .35))
    page = Image.new("L", (m.width + 2 * pw, m.height + 2 * ph), r.randint(190, 255))
    ink = r.randint(0, 110)
    page.paste(ink, (pw, ph), m)
    d = ImageDraw.Draw(page)
    for _ in range(r.choice([0, 0, 0, 1, 2])):  # boundary lines, paper fibres, dotted form lines
        x0, y0 = r.uniform(0, page.width), r.uniform(0, page.height)
        a = r.uniform(0, 6.3)
        L = page.height * r.uniform(.3, 1.5)
        d.line((x0, y0, x0 + math.cos(a) * L, y0 + math.sin(a) * L), fill=r.randint(ink, 190), width=r.randint(1, 6))
    if r.random() < .12:
        y = page.height - r.randint(2, max(3, ph + 2))
        for xx in range(0, page.width, 12):
            d.ellipse((xx, y, xx + 4, y + 4), fill=ink)
    k = r.uniform(16, 44) / page.height  # low scan resolution, then blur and noise
    page = page.resize((max(6, int(page.width * k)), max(6, int(page.height * k))), Image.BILINEAR)
    if r.random() < .5:
        page = page.filter(ImageFilter.GaussianBlur(r.uniform(.2, .9)))
    a = np.asarray(page, np.float32) + nr.normal(0, r.uniform(0, 14), (page.height, page.width))
    return fit(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))), text


def page_samples(job):
    path, root, per_page, jitter = job
    rec = json.loads(Path(path).read_text())
    r = random.Random(path)
    L = [l for l in rec["labels"] if l["cls"] == "parcel" and not l["void"]]
    im = Image.open(Path(root) / rec["image"]).convert("RGB")
    out = []
    for l in r.sample(L, min(per_page, len(L))):
        q = l["quad"]
        if jitter:
            h = math.dist(q[1], q[2])
            q = [(x + r.gauss(0, .05) * h, y + r.gauss(0, .05) * h) for x, y in q]
        t = "".join(TH[int(c)] for c in l["text"]) if l["thai"] else l["text"]
        out.append((fit(cut(im, q, r.uniform(.1, .4) if jitter else .25)), t, rec["tier"], l["rotated"]))
    return out


def save(path, items):
    np.savez(path, x=np.stack([i[0] for i in items]), t=np.array([i[1] for i in items]),
             tier=np.array([i[2] if len(i) > 2 else "pen" for i in items]),
             rot=np.array([bool(i[3]) if len(i) > 3 else False for i in items]))
    print(Path(path).name, len(items), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v2")
    ap.add_argument("--work", required=True)
    ap.add_argument("--pen", type=int, default=40000)
    ap.add_argument("--per-page", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    pages = lambda s: sorted(map(str, (Path(a.data) / s).glob("*.json")))
    with Pool(4) as p:
        for name, split, n, j in (("page_train", "train", a.per_page, True), ("page_val", "val", 12, False)):
            rows = [x for part in p.imap(page_samples, [(f, a.data, n, j) for f in pages(split)], 8) for x in part]
            save(out / f"{name}.npz", rows)
        save(out / "pen_train.npz", list(p.imap(pen_sample, range(10 ** 6, 10 ** 6 + a.pen), 64)))
        save(out / "pen_val.npz", list(p.imap(pen_sample, range(5 * 10 ** 6, 5 * 10 ** 6 + 2000), 64)))
