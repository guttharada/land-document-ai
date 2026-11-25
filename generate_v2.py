"""Synthetic survey maps, v2: numeral systems and label classes follow real maps.

  parcel  เลขที่ดิน      Thai digits (a few pages Arabic)   <- what we want
  marker  หลักเขต        two lines: "3ก" over "8580"        Arabic + Thai letter
  length  ระยะ           "26.000ม."                         Arabic
  area    ไร่-งาน-วา     "๐-๐-๔๔" + stacked fraction        Thai
  road / misc            Thai text, "น.ส.๓ก.เลขที่ ๓๑๖", scale

Tiers: S single parcel, printed      H single parcel, old handwriting
       P subdivision, printed        Q old dense subdivision, handwriting
Split by office; val and test use fonts that never appear in train, and every
split contains both Thai- and Arabic-numbered pages.

    python3 generate_v2.py --out data_v2 --per-office 200
All content is invented. Pages carry a SYNTHETIC mark and no emblem.
Handwritten Thai digits are warped font glyphs, not real handwriting.
"""
import argparse, json, math, random
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates

from generate import A1, A2, DOC_TYPES, H, MONTHS, STROKES, THAI_DIGITS, W, spline

SUP, SYS, LIB = "/System/Library/Fonts/Supplemental/", "/System/Library/Fonts/", str(Path.home() / "Library/Fonts") + "/"
DIGIT_FONTS = {  # disjoint across splits
    "train": [SUP + "Silom.ttf", SUP + "Ayuthaya.ttf", SUP + "Tahoma.ttf", SUP + "Microsoft Sans Serif.ttf", LIB + "Anuphan-Light.ttf"],
    "val": [SUP + "Thonburi.ttc", SUP + "Sathu.ttf"],
    "test": [LIB + "ChulaCharasNewReg.ttf", SUP + "Arial Unicode.ttf"],
    # offices 11-12: made after training, fonts used nowhere else (also not in ocr_data.py)
    "test2": [LIB + "CHULALONGKORNReg.otf", LIB + "SFProTH_regular.woff.ttf"]}
TEXT_FONTS = [SUP + f for f in ("Silom.ttf", "Ayuthaya.ttf", "Sathu.ttf")]  # shape Thai vowels correctly
ARABIC_FONT = SUP + "Arial.ttf"
LETTERS = "กขคงจฉชซดตถทนบปพมยรลวสอ"
# share of pages whose parcel numbers are Arabic, fixed per office (not random)
ARABIC_P = {1: 0, 2: .2, 3: 0, 4: .1, 5: .3, 6: 0, 7: 0, 8: .3, 9: 0, 10: .3, 11: 0, 12: .2}


def split_of(o):
    return "train" if o <= 6 else "val" if o <= 8 else "test" if o <= 10 else "test2"


def th(s):
    return "".join(THAI_DIGITS[int(c)] if c.isascii() and c.isdigit() else c for c in s)


@lru_cache(512)
def font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(1)
def registry():
    r = random.Random(7)
    names = [a + b for a in A1 for b in A2]
    r.shuffle(names)
    return {names[i]: {t: sorted(r.sample(range(1, 4000), 900)) for t in names[8 + i * 5:13 + i * 5]}
            for i in range(8)}


def style(o):
    r = random.Random(2000 + o)
    red = r.randint(212, 248)
    green = red - r.randint(2, 10)
    return dict(digit=DIGIT_FONTS[split_of(o)][o % len(DIGIT_FONTS[split_of(o)])],
                text=r.choice(TEXT_FONTS), ink=(r.randint(0, 50), r.randint(0, 50), r.randint(10, 90)),
                paper=(red, green, green - r.randint(4, 30)),
                slant=r.uniform(-.15, .35), jitter=r.uniform(.02, .04), pen=r.uniform(.07, .11),
                warp=r.uniform(.035, .06), noise=r.uniform(3, 8), fibre=r.randint(20, 160))


def printed(text, h, path):
    f = font(path, max(8, int(h * 1.3)))
    b = f.getbbox(text)
    im = Image.new("L", (b[2] - b[0] + 4, b[3] - b[1] + 4), 0)
    ImageDraw.Draw(im).text((2 - b[0], 2 - b[1]), text, font=f, fill=255)
    return im


def warp(im, amt, slant, nr):
    """Elastic distortion + slant, so glyphs look hand drawn."""
    a = np.asarray(im, np.float32)
    h = a.shape[0]
    pad = int(h * .35)
    a = np.pad(a, pad)
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]].astype(np.float32)
    sig = h * .28
    dx, dy = (gaussian_filter(nr.normal(size=a.shape), sig) for _ in "xy")
    dx, dy = (d / (np.abs(d).max() + 1e-6) * amt * 3 * h for d in (dx, dy))
    out = map_coordinates(a, [yy + dy, xx + dx + slant * (yy - a.shape[0] / 2)], order=1)
    im = Image.fromarray(out.astype(np.uint8))
    return im.crop(im.getbbox() or (0, 0, 1, 1))


def hand(text, h, st, r, nr):
    """Arabic digits are drawn stroke by stroke; Thai glyphs are warped."""
    s, parts = 3, []
    hh = h * s
    for ch in text:
        if ch in STROKES:
            cw = hh * .6
            im = Image.new("L", (int(cw * 1.9), int(hh * 1.5)), 0)
            d = ImageDraw.Draw(im)
            for stroke in STROKES[ch]:
                d.line(spline([(cw * .4 + (px + r.gauss(0, st["jitter"]) + st["slant"] * (1 - py)) * cw,
                                hh * .25 + (py + r.gauss(0, st["jitter"])) * hh) for px, py in stroke]),
                       fill=255, width=max(3, int(st["pen"] * hh)), joint="curve")
            im = im.crop(im.getbbox())
        elif ch in THAI_DIGITS or ch in LETTERS:
            im = warp(printed(ch, hh, st["digit"]), st["warp"], st["slant"], nr)
        else:
            im = printed(ch, hh, st["digit"])
        parts.append(im.resize((max(1, int(im.width * r.uniform(.88, 1.12))), im.height)))
    gap = int(hh * .12)
    mh = max(p.height for p in parts)
    out = Image.new("L", (sum(p.width for p in parts) + gap * len(parts), mh + int(hh * .1)), 0)
    x = 0
    for p, ch in zip(parts, text):
        y = mh - p.height if ch != "-" else (mh - p.height) // 2
        out.paste(p, (x, y + r.randint(0, int(hh * .1))))
        x += p.width + gap
    out = out.crop(out.getbbox())
    return out.resize((max(1, out.width // s), max(1, out.height // s)), Image.LANCZOS)


class Page:
    def __init__(self, st, ink, old, r, nr):
        self.im = Image.new("RGB", (W, H), st["paper"])
        self.d = ImageDraw.Draw(self.im)
        self.st, self.ink, self.old, self.r, self.nr = st, ink, old, r, nr
        self.labels, self.boxes = [], []

    def mask(self, text, h, thai):
        if self.old:
            return hand(text, h, self.st, self.r, self.nr)
        return printed(text, h, self.st["digit"] if thai else ARABIC_FONT)

    def free(self, x, y, w, h, ang, slack=0):
        a = math.radians(ang)
        bw, bh = abs(w * math.cos(a)) + abs(h * math.sin(a)), abs(w * math.sin(a)) + abs(h * math.cos(a))
        b = (x - bw / 2 + slack, y - bh / 2 + slack, x + bw / 2 - slack, y + bh / 2 - slack)
        ok = b[0] > 20 and b[1] > 20 and b[2] < W - 20 and b[3] < H - 20
        return ok and all(b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3] for o in self.boxes), b

    def put(self, cls, text, mask, x, y, ang=0, force=False, **kw):
        """Place a label centred at x,y, rotated ang degrees ccw. False if it would overlap."""
        ok, b = self.free(x, y, mask.width, mask.height, ang)
        if not ok and not force:
            return False
        rot = mask.rotate(ang, expand=True, resample=Image.BICUBIC)
        self.im.paste(self.ink, (int(x - rot.width / 2), int(y - rot.height / 2)), rot)
        a = math.radians(-ang)
        c, s, w, h = math.cos(a), math.sin(a), mask.width / 2, mask.height / 2
        q = [(x + u * c - v * s, y + u * s + v * c) for u, v in ((-w, -h), (w, -h), (w, h), (-w, h))]
        self.boxes.append(b)
        self.labels.append(dict(cls=cls, text=text, quad=q, rotated=45 < abs(ang) % 180 < 135,
                                **{"void": False, "thai": False, "role": "", **kw}))
        return True

    def strike(self):
        q = self.labels[-1]["quad"]
        self.d.line([((q[a][0] + q[b][0]) / 2, (q[a][1] + q[b][1]) / 2) for a, b in ((0, 3), (1, 2))],
                    fill=self.ink, width=3)

    def parcel(self, num, x, y, h, ang, thai, role, void_num=None):
        mk = lambda n: self.mask(th(str(n)) if thai else str(n), h, thai)
        if void_num is not None:
            t = math.radians(-ang)
            nx, ny = -math.sin(t) * h * .85, math.cos(t) * h * .85
            self.put("parcel", str(void_num), mk(void_num), x - nx, y - ny, ang, True, void=True, thai=thai, role=role)
            self.strike()
            x, y = x + nx, y + ny
        return self.put("parcel", str(num), mk(num), x, y, ang, True, thai=thai, role=role)

    def marker(self, vx, vy, h, dirs):
        """Circle at a corner plus its two-line label on the first free side."""
        rad = max(5, int(h * .38))
        self.d.ellipse((vx - rad, vy - rad, vx + rad, vy + rad), outline=self.ink, width=max(2, h // 11))
        a, b = f"{self.r.randint(1, 9)}{self.r.choice(LETTERS)}", f"{self.r.randint(1000, 9999)}"
        m1 = self.mask(a, h, True) if self.old else printed(a, h, self.st["digit"])
        m2 = self.mask(b, h, False)
        m = Image.new("L", (max(m1.width, m2.width), m1.height + m2.height + 3), 0)
        m.paste(m1, ((m.width - m1.width) // 2, 0))
        m.paste(m2, ((m.width - m2.width) // 2, m1.height + 3))
        for dx, dy in dirs:
            if self.put("marker", a + b, m, vx + dx * (m.width / 2 + rad + 4), vy + dy * (m.height / 2 + rad + 2)):
                return

    def text(self, cls, s, x, y, h, ang=0):
        f = font(self.st["text"], int(h * 1.5))
        b = f.getbbox(s)
        m = Image.new("L", (b[2] - b[0] + 6, b[3] - b[1] + 6), 0)
        ImageDraw.Draw(m).text((3 - b[0], 3 - b[1]), s, font=f, fill=255)
        if self.old:
            m = warp(m, self.st["warp"] * .5, self.st["slant"] * .5, self.nr)
        return self.put(cls, s, m, x, y, ang, True)


def area_mask(pg, wa, h):
    """'๐-๐-๔๔' with an optional stacked tenth, as on subdivision sheets."""
    whole, tenth = int(wa), int(round((wa - int(wa)) * 10))
    if tenth == 10:
        whole, tenth = whole + 1, 0
    s = f"{whole // 400}-{whole % 400 // 100}-{whole % 100:02d}"
    m = pg.mask(th(s), h, True)
    if tenth:
        n, d_ = (pg.mask(th(t), int(h * .6), True) for t in (str(tenth), "10"))
        f = Image.new("L", (max(n.width, d_.width) + 4, n.height + d_.height + 5), 0)
        f.paste(n, ((f.width - n.width) // 2, 0))
        f.paste(d_, ((f.width - d_.width) // 2, n.height + 5))
        ImageDraw.Draw(f).line((0, n.height + 2, f.width, n.height + 2), fill=255, width=1)
        o = Image.new("L", (m.width + 5 + f.width, max(m.height, f.height)), 0)
        o.paste(m, (0, (o.height - m.height) // 2))
        o.paste(f, (m.width + 5, (o.height - f.height) // 2))
        m = o
    return s + (f" {tenth}/10" if tenth else ""), m


def single(pg, nums, thai, r):
    n, cx, cy = r.randint(4, 8), W / 2 + r.uniform(-200, 200), 960 + r.uniform(-60, 60)
    angs = sorted(2 * math.pi * (k + r.uniform(-.28, .28)) / n for k in range(n))
    P = [(cx + math.cos(a) * 1.35 * R, cy + math.sin(a) * R * .82) for a in angs for R in [r.uniform(300, 520)]]
    ink, h = pg.ink, r.randint(30, 38)
    pg.d.line(P + [P[0]], fill=ink, width=3, joint="curve")
    it = iter(nums)
    tgt = next(it)
    void = next(it) if pg.old and r.random() < .15 else None
    mx, my = sum(p[0] for p in P) / n, sum(p[1] for p in P) / n
    pg.parcel(tgt, mx, my, int(h * r.uniform(1, 1.35)), r.uniform(-5, 5), thai, "target", void)
    for x, y in P:  # stubs of neighbouring boundaries
        ux, uy = (x - mx) / math.dist((x, y), (mx, my)), (y - my) / math.dist((x, y), (mx, my))
        if r.random() < .7:
            pg.d.line([(x, y), (x + ux * 70, y + uy * 70)], fill=ink, width=2)
    road = r.randrange(n) if r.random() < .7 else -1
    for k in range(n):
        a, b = P[k], P[(k + 1) % n]
        ex, ey = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        L = math.dist(a, b)
        nx, ny = (b[1] - a[1]) / L, -(b[0] - a[0]) / L
        if (ex - mx) * nx + (ey - my) * ny < 0:
            nx, ny = -nx, -ny
        tilt = -math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        tilt = (tilt + 90) % 180 - 90
        if k == road:
            off = 100
            for t in np.arange(0, 1, .12):
                q = [(a[0] + (b[0] - a[0]) * u + nx * off, a[1] + (b[1] - a[1]) * u + ny * off) for u in (t, t + .07)]
                pg.d.line(q, fill=ink, width=3)
            pg.text("road", r.choice(["ทางหลวงท้องถิ่น", "ทางสาธารณประโยชน์", "ลำรางสาธารณะ", "ทางหลวงชนบท"]),
                    ex + nx * 52, ey + ny * 52, 26, tilt)
        elif r.random() < .75:
            d = r.uniform(110, 170)
            if r.random() < .18:
                pg.text("misc", "น.ส.๓ก.เลขที่ " + th(str(r.randint(10, 999))), ex + nx * d, ey + ny * d, 26)
            else:
                pg.parcel(next(it), ex + nx * d, ey + ny * d, h, r.uniform(-5, 5), thai, "neighbor")
        if r.random() < .3:
            s = f"{L / 14:.3f}ม."
            pg.put("length", s, printed(s, 20, pg.st["text"]) if not pg.old else pg.mask(s[:-2], 20, False),
                   ex - nx * 26, ey - ny * 26, tilt)
    for x, y in P:
        sx, sy = (1 if x > mx else -1), (1 if y > my else -1)
        pg.marker(x, y, int(h * .85), [(sx, sy * .3), (sx * .3, sy), (sx, -sy * .6), (-sx, sy)])


def subdivision(pg, nums, thai, r, dense):
    ppm = r.uniform(7.2, 9) if dense else r.uniform(13, 15.5)
    hp = r.randint(18, 22) if dense else r.randint(26, 31)
    hs = r.randint(11, 13) if dense else r.randint(15, 18)
    x0, y, x1 = 150, 420, 2200
    it, corners, lots = iter(nums), {}, []
    while True:
        dep = r.uniform(17, 24)
        if y + dep * ppm > 1530:
            break
        x = x0 + r.uniform(0, 60)
        while True:
            w = r.choice([10, 10, 10, 12.5, 13.5, 15, r.uniform(9, 20)])
            if x + w * ppm > x1:
                break
            if r.random() > .03:
                lots.append((x, y, w, dep))
            x += w * ppm
        y += dep * ppm
        if r.random() < .6:
            road = 7 * ppm
            pg.text("road", "ทางหลวงท้องถิ่น", r.uniform(500, 1700), y + road / 2, int(hs * 1.3))
            y += road
    for x, yy, w, dep in lots:
        X, Y = x + w * ppm, yy + dep * ppm
        pg.d.rectangle((x, yy, X, Y), outline=pg.ink, width=2)
        for c in ((x, yy), (X, yy), (x, Y), (X, Y)):
            corners[(round(c[0] / 4), round(c[1] / 4))] = c
    lead = []
    for x, yy, w, dep in lots:
        mx, my = x + w * ppm / 2, yy + dep * ppm / 2
        ang = r.choice([90, 90, 0]) if w < 13 else 0
        num = next(it)
        if not ang and hp * .8 * len(str(num)) > w * ppm * .85:
            ang = 90  # too wide for the lot: written vertically
        ang_j = ang + r.uniform(-4, 4) * pg.old
        s, am = area_mask(pg, w * dep / 4, int(hp * .72))
        off = hp * 1.15
        px, py, ax, ay = (mx - off, my, mx + off, my) if ang else (mx, my - off, mx, my + off)
        if dense and r.random() < .05:  # number written outside with a leader line
            lx, ly = mx + r.uniform(-.6, .6) * w * ppm, yy + r.choice([0, dep * ppm])
            pg.d.line([(mx, my), (lx, ly)], fill=pg.ink, width=2)
            px, py = lx, ly
        pg.parcel(num, px, py, hp, ang_j, thai, "lot",
                  next(it) if pg.old and r.random() < (.12 if dense else .06) else None)
        if r.random() < .8:
            pg.put("area", s, am, ax, ay, ang, thai=True)
    for vx, vy in corners.values():
        if r.random() < (.45 if dense else .85):
            pg.marker(vx, vy, hs, r.sample([(1, 1), (-1, 1), (1, -1), (-1, -1)], 4))
        else:
            q = max(4, hs // 3)
            pg.d.ellipse((vx - q, vy - q, vx + q, vy + q), outline=pg.ink, width=2)
    for x, yy, w, dep in lots:
        for s, lx, ly, a in ((f"{w:.3f}ม.", x + w * ppm / 2, yy + hs * .9, 0),
                             (f"{dep:.3f}ม.", x + hs * .9, yy + dep * ppm / 2, 90)):
            if r.random() < (.35 if dense else .7):
                pg.put("length", s, printed(s, hs, pg.st["text"]), lx, ly, a)


def paper_fx(im, st, r, nr, old):
    arr = np.asarray(im).astype(np.float32)
    low = lambda a, b: np.asarray(Image.fromarray(nr.integers(0, 255, (a, b), dtype=np.uint8)).resize((W, H), Image.BICUBIC), np.float32) / 255
    if old:
        stain = np.clip(low(9, 13) - .55, 0, 1) * r.uniform(40, 130)
        speck = np.clip(low(H // 9, W // 9) - .72, 0, 1) * r.uniform(60, 220)
        arr -= (stain + speck)[..., None] * np.array([.75, .85, 1.0], np.float32)
        f = (r.uniform(0, .45) * low(6, 8))[..., None]
        arr = arr * (1 - f) + np.array(st["paper"], np.float32) * f
    else:
        arr -= (np.clip(low(H // 12, W // 12) - .8, 0, 1) * 50)[..., None]
    arr += nr.normal(0, st["noise"], (H, W, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make(job):
    idx, office, out = job
    split = split_of(office)
    r, nr = random.Random(90000 + idx), np.random.default_rng(90000 + idx)
    st, reg = style(office), registry()
    tier = "SHPQ"[idx % 10000 % 4]
    old = tier in "HQ"
    st = dict(st)
    ink = st["ink"] if old else (20, 20, 25)
    pg = Page(st, ink, old, r, nr)
    for _ in range(st["fibre"] if old else st["fibre"] // 4):  # paper fibres under the drawing
        x, y, a = r.uniform(0, W), r.uniform(0, H), r.uniform(0, 6.3)
        tone = tuple(int(c * .88) for c in st["paper"])
        pg.d.line(spline([(x + math.cos(a + k * r.uniform(-.2, .2)) * 25 * k, y + math.sin(a) * 25 * k) for k in range(4)]),
                  fill=tone, width=2)
    amphoe = r.choice(sorted(reg))
    tambon = r.choice(sorted(reg[amphoe]))
    year = r.randint(2505, 2540) if old else r.randint(2541, 2565)
    fields = dict(doc_type=r.choice(DOC_TYPES), date=f"{r.randint(1, 28)} {r.choice(MONTHS)} {year}",
                  amphoe=amphoe, tambon=tambon)
    nums = r.sample(reg[amphoe][tambon], 900)
    thai = r.random() >= ARABIC_P[office]
    hx = r.randint(420, 520)
    for x, y, t in ((hx, 90, fields["doc_type"]), (hx + 1150, 90, "วันที่ " + th(fields["date"])),
                    (hx, 190, "อำเภอ " + amphoe), (hx + 1150, 190, "ตำบล " + tambon),
                    (W / 2, 300, "มาตราส่วน " + th(r.choice(["1:1000", "1:500", "1:4000"])))):
        pg.text("misc", t, x + r.randint(-12, 12), y + r.randint(-8, 8), 32)
    pg.d.line([(90, 330), (90, 150)], fill=ink, width=3)
    pg.d.polygon([(90, 135), (78, 172), (102, 172)], fill=ink)
    (single if tier in "SH" else subdivision)(pg, nums, thai, r, *([tier == "Q"] if tier in "PQ" else []))
    pg.d.text((W - 620, H - 60), "SYNTHETIC / เอกสารจำลอง", font=font(SUP + "Silom.ttf", 24), fill=(120, 120, 120))
    if r.random() < (.5 if old else .2):  # red stamp over the map
        rad = r.randint(90, 140)
        m = Image.new("L", (2 * rad + 8,) * 2, 0)
        md = ImageDraw.Draw(m)
        md.ellipse((4, 4, 2 * rad, 2 * rad), outline=150, width=7)
        md.text((rad - 62, rad - 25), "ตัวอย่าง", font=font(SUP + "Silom.ttf", 40), fill=150)
        m = m.rotate(r.uniform(-30, 30))
        pg.im.paste((190, 30, 40), (int(r.uniform(400, W - 600)), int(r.uniform(450, H - 500))), m)

    skew = r.uniform(-3, 3) if old else r.uniform(-.8, .8)
    im = pg.im.rotate(skew, resample=Image.BICUBIC, fillcolor=st["paper"])
    a = math.radians(-skew)
    for l in pg.labels:
        l["quad"] = [[round(W / 2 + (x - W / 2) * math.cos(a) - (y - H / 2) * math.sin(a), 1),
                      round(H / 2 + (x - W / 2) * math.sin(a) + (y - H / 2) * math.cos(a), 1)] for x, y in l["quad"]]
    if old:
        im = im.filter(ImageFilter.GaussianBlur(r.uniform(.3, 1.0)))
    im = paper_fx(im, st, r, nr, old)
    name = f"{split}/o{office:02d}_{idx:05d}_{tier}"
    im.save(Path(out) / f"{name}.jpg", quality=r.randint(60, 88))
    (Path(out) / f"{name}.json").write_text(json.dumps(dict(
        image=f"{name}.jpg", split=split, office=office, tier=tier, fields=fields, thai_page=thai,
        labels=pg.labels), ensure_ascii=False))
    return len(pg.labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_v2")
    ap.add_argument("--per-office", type=int, default=200)
    ap.add_argument("--offices", default="1-10")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    lo, hi = map(int, a.offices.split("-"))
    for s in {split_of(o) for o in range(lo, hi + 1)}:
        (Path(a.out) / s).mkdir(parents=True, exist_ok=True)
        for f in DIGIT_FONTS[s]:
            assert Path(f).exists(), f"font missing: {f}"
    (Path(a.out) / "registry.json").write_text(json.dumps(registry(), ensure_ascii=False))
    jobs = [((o - 1) * 10000 + k, o, a.out) for o in range(lo, hi + 1) for k in range(a.per_office)]
    with Pool(a.procs) as p:
        n = sum(p.imap_unordered(make, jobs, chunksize=4))
    print(f"{len(jobs)} pages, {n} labels -> {a.out}")
