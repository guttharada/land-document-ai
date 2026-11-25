"""Recall, precision, F1 and accuracy of YOLO + reader on one split.

    .venv/bin/python final_metrics.py --split test2 --preds preds.json --out results/test2.json

Detection hit: the predicted centre lies on the true box (grown by 35% of its height), same class,
one to one.  Reading is right only if every digit of the number is right.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import evaluate as ev
from read_numbers import CLASSES, read
from train_ocr import load

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v2")
ap.add_argument("--split", default="test2")
ap.add_argument("--preds", required=True)
ap.add_argument("--reader", default="ocr_thai.pt")
ap.add_argument("--conf", type=float, default=.25)
ap.add_argument("--out", default="")
a = ap.parse_args()
preds, model = json.loads(Path(a.preds).read_text()), load(a.reader)


def cls(l):
    return ("struck" if l["void"] else "parcel") if l["cls"] == "parcel" else l["cls"] if l["cls"] in CLASSES else "text"


def prf(tp, pred, gt):
    p, r = tp / pred if pred else 0, tp / gt if gt else 0
    return dict(gt=gt, predicted=pred, right=tp, recall=round(100 * r, 2), precision=round(100 * p, 2),
                f1=round(200 * p * r / (p + r), 2) if p + r else 0)


det, e2e, page, digits, sure, crossed = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter(), [], Counter()
recs = ev.load(a.data, a.split)
assert len(recs) == len(preds), (len(recs), len(preds))
for r in recs:
    P = [p for p in preds[Path(r["image"]).name] if p["conf"] >= a.conf]
    for n in CLASSES:
        L, Q = [l for l in r["labels"] if cls(l) == n], [p for p in P if CLASSES[p["cls"]] == n]
        for k in (n, f"{n} {r['tier']}"):
            det[k].update(gt=len(L), pred=len(Q), tp=len(ev.match(Q, L)))
    L, Q = [l for l in r["labels"] if cls(l) == "parcel"], [p for p in P if p["cls"] == 0]
    im = Image.open(Path(a.data) / r["image"]).convert("RGB")
    got = [(ev.norm(t), c) for t, c in read(im, [p["quad"] for p in Q], model)]
    at = {id(p): i for i, p in ev.match(Q, L).items()}
    voids = [l for l in r["labels"] if cls(l) == "struck"]
    crossed.update(called_parcel=len(ev.match([p for p in Q if id(p) not in at], voids)), total=len(voids))
    right, target = 0, None
    num = "thai" if r["thai_page"] else "arabic"
    groups = ("all", r["tier"], num)
    for k in groups:
        e2e[k].update(gt=len(L), pred=len(Q))
    for p, (t, c) in zip(Q, got):
        l = L[at[id(p)]] if id(p) in at else None
        ok = l is not None and t == l["text"]
        right += ok
        sure.append((c, ok))
        if l is not None:
            n = max(len(t), len(l["text"]))
            digits.update(total=n, ok=sum(x == y for x, y in zip(t, l["text"])))
            if l["role"] == "target":
                target = ok
        for k in groups:
            e2e[k].update(tp=int(ok))
    for k in ("all", r["tier"]):
        page[k].update(n=1, perfect=int(right == len(L) == len(Q)))
    if r["tier"] in "SH":
        page["target (true position)"].update(n=1, perfect=int(bool(target)))

out = dict(split=a.split, pages=len(recs), conf=a.conf,
           detection={k: prf(v["tp"], v["pred"], v["gt"]) for k, v in sorted(det.items()) if v["gt"]},
           end_to_end={k: prf(v["tp"], v["pred"], v["gt"]) for k, v in sorted(e2e.items())},
           digit_accuracy_on_found=round(100 * digits["ok"] / digits["total"], 2),
           page_perfect={k: dict(pages=v["n"], accuracy=round(100 * v["perfect"] / v["n"], 1)) for k, v in page.items()},
           struck_out_filed_as_parcel=dict(crossed),
           by_confidence={str(t): dict(kept=sum(c >= t for c, _ in sure), coverage=round(100 * sum(o for c, o in sure if c >= t) / e2e["all"]["gt"], 1),
                                       wrong=round(100 * sum(not o for c, o in sure if c >= t) / max(1, sum(c >= t for c, _ in sure)), 2))
                          for t in (0, .8, .9, .95, .99)})
print(json.dumps(out, ensure_ascii=False, indent=1))
if a.out:
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
