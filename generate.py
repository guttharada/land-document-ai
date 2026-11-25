"""Generate synthetic survey-map pages with exact ground truth.

Every name, number and layout is invented. Pages carry a "SYNTHETIC" mark and
no government emblem, so they cannot be mistaken for real documents.

    python3 generate.py --out data --per-office 200

Split is BY OFFICE (train 1-6, val 7-8, test 9-10): test offices have styles
never seen in train, which is how accuracy drops when a new site opens.
Tiers: A = newer/printed, B = older handwriting, C = old dense subdivision.
"""
import argparse, json, math, random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2339, 1654  # A4 landscape, 200 dpi
MAP = (110, 400, 2229, 1540)
F = "/System/Library/Fonts/Supplemental/"
THAI_FONTS = ["Sathu.ttf", "Silom.ttf", "Ayuthaya.ttf", "Krungthep.ttf"]
THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
DOC_TYPES = ["ต้นร่างแผนที่", "รายการรังวัด", "แผนที่แบ่งแยก"]
MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
          "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
A1 = ["ทุ่ง", "หนอง", "บึง", "ดอน", "เขา", "คลอง"]
A2 = ["สมมติ", "จำลอง", "น้ำใส", "ดาวเรือง", "แสงทอง", "ใบไผ่", "ลมโชย", "ฟ้าคราม"]

# Handwritten digits as control points in a unit box (x right, y down).
STROKES = {
    "0": [[(.5, 0), (.1, .3), (.1, .7), (.5, 1), (.9, .7), (.9, .3), (.5, 0)]],
    "1": [[(.35, .2), (.55, 0), (.55, .5), (.55, 1)]],
    "2": [[(.1, .25), (.45, 0), (.85, .25), (.55, .6), (.1, 1), (.5, .96), (.95, 1)]],
    "3": [[(.15, .1), (.6, 0), (.85, .25), (.45, .47), (.9, .7), (.55, 1), (.1, .88)]],
    "4": [[(.6, 0), (.3, .35), (.05, .68), (.5, .68), (.95, .66)], [(.68, .35), (.68, .7), (.66, 1)]],
    "5": [[(.85, .02), (.5, .02), (.25, .03), (.2, .4), (.6, .4), (.9, .68), (.55, 1), (.1, .9)]],
    "6": [[(.75, .02), (.35, .3), (.12, .7), (.45, 1), (.85, .78), (.55, .5), (.15, .68)]],
    "7": [[(.08, .05), (.5, .04), (.9, .05), (.6, .5), (.4, 1)]],
    "8": [[(.55, .48), (.2, .25), (.5, 0), (.8, .25), (.5, .5), (.12, .75), (.5, 1), (.88, .75), (.5, .48)]],
    "9": [[(.82, .3), (.5, 0), (.15, .28), (.5, .52), (.82, .3), (.8, .65), (.7, 1)]],
}


def registry(seed=7):
    """Invented amphoe -> tambon -> valid parcel numbers."""
    r = random.Random(seed)
    names = [a + b for a in A1 for b in A2]
    r.shuffle(names)
    reg = {}
    for i in range(8):
        reg[names[i]] = {t: sorted(r.sample(range(1, 10000), 1500))
                         for t in names[8 + i * 5: 13 + i * 5]}
    return reg


def office_style(o):
    r = random.Random(1000 + o)
    return dict(
        font=r.choice(THAI_FONTS),
        ink=tuple(r.randint(0, 70) for _ in range(2)) + (r.randint(20, 120),),
        paper=(r.randint(232, 252), r.randint(222, 245), r.randint(190, 232)),
        slant=r.uniform(-.25, .45), jitter=r.uniform(.02, .042),
        pen=r.uniform(.07, .12), thai_p=r.choice([0, 0, .2, .5]),
        noise=r.uniform(3, 9), hpos=r.randint(-30, 60))


def spline(pts, n=8):
    p = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(p) - 2):
        for k in range(n):
            t = k / n
            out.append(tuple(.5 * (2 * p[i][d] + (p[i + 1][d] - p[i - 1][d]) * t
                + (2 * p[i - 1][d] - 5 * p[i][d] + 4 * p[i + 1][d] - p[i + 2][d]) * t * t
                + (-p[i - 1][d] + 3 * p[i][d] - 3 * p[i + 1][d] + p[i + 2][d]) * t ** 3)
                for d in (0, 1)))
    return out + [pts[-1]]


def hand_mask(text, h, st, r):
    """Alpha mask of a handwritten number of height h px."""
    s = 3
    hh, cw = h * s, h * s * .58
    im = Image.new("L", (int(cw * len(text) * 1.25 + hh), int(hh * 1.5)), 0)
    d = ImageDraw.Draw(im)
    x = hh * .35
    for ch in text:
        k, dy = r.uniform(.85, 1.1), r.uniform(-.06, .06) * hh
        for stroke in STROKES[ch]:
            pts = []
            for px, py in stroke:
                px += r.gauss(0, st["jitter"])
                py += r.gauss(0, st["jitter"])
                pts.append((x + (px + st["slant"] * (1 - py)) * cw * k, hh * .25 + py * hh * k + dy))
            d.line(spline(pts), fill=255, width=max(2, int(st["pen"] * hh)), joint="curve")
        x += cw * r.uniform(1.0, 1.25)
    im = im.crop(im.getbbox())
    return im.resize((max(1, im.width // s), max(1, im.height // s)), Image.LANCZOS)


def font_mask(text, h, font):
    f = ImageFont.truetype(F + font, int(h * 1.35))
    b = f.getbbox(text)
    im = Image.new("L", (b[2] - b[0] + 4, b[3] - b[1] + 4), 0)
    ImageDraw.Draw(im).text((2 - b[0], 2 - b[1]), text, font=f, fill=255)
    return im


def stamp(page, mask, cx, cy, ang, ink, alpha=1.0):
    """Paste mask centred at cx,cy rotated ang degrees (ccw). Returns its quad."""
    w, h = mask.size
    rot = mask.rotate(ang, expand=True, resample=Image.BICUBIC)
    if alpha < 1:
        rot = rot.point(lambda v: int(v * alpha))
    page.paste(ink, (int(cx - rot.width / 2), int(cy - rot.height / 2)), rot)
    a = math.radians(-ang)
    c, s = math.cos(a), math.sin(a)
    return [(cx + x * c - y * s, cy + x * s + y * c)
            for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def make(job):
    idx, office, out, split = job
    r = random.Random(50000 + idx)
    nr = np.random.default_rng(50000 + idx)
    st, reg = office_style(office), registry()
    tier = r.choices("ABC", [.35, .35, .30])[0]
    page = Image.new("RGB", (W, H), st["paper"])
    d = ImageDraw.Draw(page)
    ink = st["ink"] if tier != "A" else (25, 25, 30)
    amphoe = r.choice(sorted(reg))
    tambon = r.choice(sorted(reg[amphoe]))
    year = r.randint(2545, 2565) if tier == "A" else r.randint(2505, 2540)
    fields = {"doc_type": r.choice(DOC_TYPES),
              "date": f"{r.randint(1, 28)} {r.choice(MONTHS)} {year}",
              "amphoe": amphoe, "tambon": tambon}
    thai_page = r.random() < st["thai_p"]
    to_thai = lambda s: "".join(THAI_DIGITS[int(c)] if c.isdigit() else c for c in s)

    # header
    fnt = ImageFont.truetype(F + st["font"], 46)
    small = ImageFont.truetype(F + "Silom.ttf", 24)
    date = to_thai(fields["date"]) if thai_page else fields["date"]
    hx = 150 + st["hpos"]
    for (x, y), t in (((hx, 90), fields["doc_type"]), ((hx + 1150, 90), "วันที่ " + date),
                      ((hx, 200), "อำเภอ " + amphoe), ((hx + 1150, 200), "ตำบล " + tambon)):
        d.text((x + r.randint(-12, 12), y + r.randint(-10, 10)), t, font=fnt, fill=ink)
    d.text((W - 620, H - 60), "SYNTHETIC / เอกสารจำลอง", font=small, fill=(120, 120, 120))

    # warped parcel grid
    cols, rows = {"A": (r.randint(4, 7), r.randint(2, 4)), "B": (r.randint(6, 10), r.randint(3, 5)),
                  "C": (r.randint(14, 20), r.randint(6, 9))}[tier]
    x0, y0, x1, y1 = MAP
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
    amp, ph = r.uniform(0, 35), r.uniform(0, 6)
    V = [[(x0 + i * cw + r.uniform(-.1, .1) * cw + amp * math.sin(j + ph),
           y0 + j * ch + r.uniform(-.1, .1) * ch + amp * math.sin(i / 3 + ph))
          for i in range(cols + 1)] for j in range(rows + 1)]
    nums = r.sample(reg[amphoe][tambon], rows * cols * 2)
    cells = [(i, j) for j in range(rows) for i in range(cols) if r.random() > .04]
    target = r.choice(cells)
    labels, leaders = [], []
    for i, j in cells:
        poly = [V[j][i], V[j][i + 1], V[j + 1][i + 1], V[j + 1][i]]
        d.line(poly + [poly[0]], fill=ink, width=6 if (i, j) == target else 2)
    for k, (i, j) in enumerate(cells):
        poly = [V[j][i], V[j][i + 1], V[j + 1][i + 1], V[j + 1][i]]
        cx, cy = sum(p[0] for p in poly) / 4, sum(p[1] for p in poly) / 4
        top = math.degrees(math.atan2(poly[1][1] - poly[0][1], poly[1][0] - poly[0][0]))
        base = {"A": 34, "B": 34, "C": min(26, int(min(cw, ch) * .24))}[tier]
        h = max(13, int(base * r.uniform(.85, 1.15)))
        ang = -top + r.uniform(-4, 4) + (r.choice([90, -90]) if tier == "C" and r.random() < .3 else 0)
        leader = tier == "C" and r.random() < .06
        if leader:
            lx, ly = cx + r.uniform(-.5, .5) * cw, cy + r.choice([-1, 1]) * ch * .5
            d.line([(cx, cy), (lx, ly)], fill=ink, width=2)
            cx, cy = lx, ly
        thai = thai_page and tier != "C"

        def put(num, x, y, void):
            txt = str(num)
            if tier == "A" or thai:
                m = font_mask(to_thai(txt) if thai else txt, h, st["font"])
            else:
                m = hand_mask(txt, h, st, r)
            q = stamp(page, m, x, y, ang, ink)
            labels.append(dict(text=txt, quad=q, void=void, thai=thai, leader=leader,
                               rotated=abs(ang) > 45, target=(i, j) == target and not void))
            return q

        if tier != "A" and r.random() < (.12 if tier == "C" else .06):
            t = math.radians(-ang)          # stack along the text's own "down"
            nx, ny = -math.sin(t) * h, math.cos(t) * h
            q = put(nums[-1 - k], cx - nx * .8, cy - ny * .8, True)
            mid = [((q[a][0] + q[b][0]) / 2, (q[a][1] + q[b][1]) / 2) for a, b in ((0, 3), (1, 2))]
            d.line(mid, fill=ink, width=3)
            cx, cy = cx + nx * .8, cy + ny * .8
        put(nums[k], cx, cy, False)
        # edge lengths: distractors that are not parcel numbers
        if r.random() < (.15 if tier == "C" else .4):
            m = font_mask(f"{r.uniform(4, 60):.2f}", max(11, int(h * .6)), "Sathu.ttf")
            ex, ey = (poly[0][0] + poly[1][0]) / 2, (poly[0][1] + poly[1][1]) / 2
            stamp(page, m, ex, ey + h * .6, -top, ink)

    if r.random() < {"A": .2, "B": .5, "C": .5}[tier]:
        sx, sy, rad = r.uniform(400, W - 400), r.uniform(500, H - 300), r.randint(90, 150)
        m = Image.new("L", (2 * rad + 8,) * 2, 0)
        md = ImageDraw.Draw(m)
        md.ellipse((4, 4, 2 * rad, 2 * rad), outline=255, width=7)
        md.text((rad - 62, rad - 25), "ตัวอย่าง", font=ImageFont.truetype(F + "Silom.ttf", 40), fill=255)
        stamp(page, m, sx, sy, r.uniform(-30, 30), (190, 30, 40), .6)

    # scan: skew, fade, blur, noise, jpeg
    skew = r.uniform(-.8, .8) if tier == "A" else r.uniform(-3, 3)
    page = page.rotate(skew, resample=Image.BICUBIC, fillcolor=st["paper"])
    a = math.radians(-skew)
    for l in labels:
        l["quad"] = [[round(W / 2 + (x - W / 2) * math.cos(a) - (y - H / 2) * math.sin(a), 1),
                      round(H / 2 + (x - W / 2) * math.sin(a) + (y - H / 2) * math.cos(a), 1)]
                     for x, y in l["quad"]]
    if tier != "A":
        page = page.filter(ImageFilter.GaussianBlur(r.uniform(.3, 1.0)))
    arr = np.asarray(page).astype(np.float32)
    if tier != "A":
        fade = r.uniform(0, .45)
        low = Image.fromarray(nr.integers(0, 255, (7, 9), dtype=np.uint8)).resize((W, H), Image.BICUBIC)
        f = (fade * np.asarray(low, np.float32) / 255)[..., None]
        arr = arr * (1 - f) + np.array(st["paper"], np.float32) * f
    arr += nr.normal(0, st["noise"], (H, W, 1))
    page = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    name = f"{split}/o{office:02d}_{idx:05d}_{tier}"
    page.save(Path(out) / f"{name}.jpg", quality=r.randint(60, 88))
    rec = dict(image=f"{name}.jpg", split=split, office=office, tier=tier, fields=fields,
               thai_page=thai_page, labels=labels)
    (Path(out) / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False))
    return len(labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-office", type=int, default=200)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    split_of = lambda o: "train" if o <= 6 else "val" if o <= 8 else "test"
    for s in ("train", "val", "test"):
        (Path(a.out) / s).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "registry.json").write_text(json.dumps(registry(), ensure_ascii=False))
    jobs = [((o - 1) * 10000 + k, o, a.out, split_of(o)) for o in range(1, 11) for k in range(a.per_office)]
    with Pool(a.procs) as p:
        n = sum(p.imap_unordered(make, jobs, chunksize=4))
    print(f"{len(jobs)} pages, {n} labels -> {a.out}")
