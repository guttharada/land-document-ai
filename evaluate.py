"""Score OCR output against ground truth.  Imported by run_baselines.py."""
import difflib, json, math, re
from collections import defaultdict
from pathlib import Path

TH = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
NUM = re.compile(r"^\d{1,4}$")
TYPES = ["ต้นร่างแผนที่", "รายการรังวัด", "แผนที่แบ่งแยก"]


def norm(t):
    return re.sub(r"\s+", "", t.translate(TH))


def center(q):
    return sum(p[0] for p in q) / 4, sum(p[1] for p in q) / 4


def size(q):
    return math.dist(q[0], q[1]), math.dist(q[1], q[2])


def inside(pt, q, pad):
    """Is pt inside quad q grown by pad px, tested in the quad's own frame."""
    cx, cy = center(q)
    w, h = size(q)
    a = math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0])
    dx, dy = pt[0] - cx, pt[1] - cy
    u, v = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
    return abs(u) <= w / 2 + pad and abs(v) <= h / 2 + pad


def dedupe(obs):
    """Tiles overlap: where centres nearly coincide keep the longer, surer box."""
    keep = []
    for o in sorted(obs, key=lambda o: (-o["conf"], -len(norm(o["text"])))):
        c, h = center(o["quad"]), min(size(o["quad"]))
        if all(math.dist(c, center(k["quad"])) > max(h, min(size(k["quad"]))) * .7 for k in keep):
            keep.append(o)
    return keep


def match(obs, labels):
    """One-to-one, nearest first. Returns {label index: prediction}."""
    pairs = []
    for li, l in enumerate(labels):
        pad = min(size(l["quad"])) * .35
        for oi, o in enumerate(obs):
            c = center(o["quad"])
            if inside(c, l["quad"], pad):
                pairs.append((math.dist(c, center(l["quad"])), li, oi))
    got, used = {}, set()
    for _, li, oi in sorted(pairs):
        if li not in got and oi not in used:
            got[li] = obs[oi]
            used.add(oi)
    return got


def header(obs, reg):
    """Rule-based header parse. raw = as read, snap = nearest registry value."""
    top = [o for o in obs if center(o["quad"])[1] < 400]
    raw = dict.fromkeys(("doc_type", "date", "amphoe", "tambon"), "")
    for o in top:
        t = norm(o["text"])
        for key, kw in (("date", "วันที่"), ("amphoe", "อำเภอ"), ("tambon", "ตำบล")):
            if kw in t and not raw[key]:
                raw[key] = t.split(kw, 1)[1]
                break
        else:
            m = difflib.get_close_matches(t, TYPES, 1, .6)
            if m and not raw["doc_type"]:
                raw["doc_type"] = t
    snap = dict(raw)
    snap["doc_type"] = (difflib.get_close_matches(raw["doc_type"], TYPES, 1, .6) or [""])[0]
    snap["amphoe"] = (difflib.get_close_matches(raw["amphoe"], list(reg), 1, .6) or [""])[0]
    pool = list(reg.get(snap["amphoe"], {})) or [t for a in reg.values() for t in a]
    snap["tambon"] = (difflib.get_close_matches(raw["tambon"], pool, 1, .6) or [""])[0]
    return raw, snap


def score(pred, recs, reg, dd=False):
    """pred: {image: [obs]}. Returns {tier or 'all': metrics}."""
    c = defaultdict(lambda: defaultdict(int))
    for r in recs:
        obs = pred[r["image"]]
        obs = dedupe(obs) if dd else obs
        live = [l for l in r["labels"] if not l["void"]]
        g = match(obs, live)
        raw, snap = header(obs, reg)
        nums = [o for o in obs if NUM.match(norm(o["text"])) and center(o["quad"])[1] > 350]
        void_hit = match(nums, [l for l in r["labels"] if l["void"]])
        target = False
        res = defaultdict(int)
        for i, l in enumerate(live):
            ok = i in g and norm(g[i]["text"]) == l["text"]
            res["n"] += 1
            res["found"] += i in g
            res["read"] += ok
            if l["target"]:
                target = ok
            if l["rotated"]:
                res["rot_n"] += 1
                res["rot_read"] += ok
        res["pred"] += len(nums)
        res["void_as_live"] += len(void_hit)
        res["pages"] += 1
        res["target"] += target
        oks = []
        for k, v in r["fields"].items():
            gt = norm(v)
            res["raw_" + k] += raw[k] == gt
            res["snap_" + k] += snap[k] == gt
            oks.append(snap[k] == gt)
        res["page_all"] += all(oks) and target
        for t in (r["tier"], "all"):
            for k, v in res.items():
                c[t][k] += v
    out = {}
    for t, m in c.items():
        p = lambda a, b: round(100 * m[a] / m[b], 1) if m[b] else None
        out[t] = dict(pages=m["pages"], labels=m["n"], found=p("found", "n"), read=p("read", "n"),
                      read_if_found=p("read", "found"), precision=p("read", "pred"),
                      rotated_read=p("rot_read", "rot_n"), target=p("target", "pages"),
                      page_all=p("page_all", "pages"), void_as_live=m["void_as_live"],
                      **{k: p(k, "pages") for k in list(m) if k.startswith(("raw_", "snap_"))})
    return out


def load(root, split):
    return [json.loads(p.read_text()) for p in sorted((Path(root) / split).glob("*.json"))]
