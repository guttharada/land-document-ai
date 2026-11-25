"""Small CRNN + CTC that reads 1-4 digit numbers, Thai or Arabic.

    .venv/bin/python train_ocr.py --work <dir from ocr_data.py> --out ocr_thai.pt

The kept epoch is the best on page_val (data_v2 val, fonts unseen in train), averaged over tiers.
"""
import argparse, random, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

VOCAB = "๐๑๒๓๔๕๖๗๘๙0123456789"


class Reader(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o, pool):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(pool))
        self.cnn = nn.Sequential(block(1, 32, 2), block(32, 64, 2), block(64, 128, (2, 1)), block(128, 160, (2, 1)),
                                 nn.Conv2d(160, 192, (3, 3), padding=(0, 1)), nn.ReLU(), nn.Dropout(.2))
        self.rnn = nn.LSTM(192, 128, bidirectional=True, batch_first=True)
        self.out = nn.Linear(256, len(VOCAB) + 1)

    def forward(self, x):  # N,1,48,192 -> N,48,21
        f = self.cnn(x).squeeze(2).transpose(1, 2)
        return self.out(self.rnn(f)[0]).log_softmax(-1)


def decode(lp):
    """Greedy CTC. Returns [(text, mean probability of the emitted characters)].

    One number is written in one script, so the Thai-only and Arabic-only readings are
    compared and the likelier one is kept; a mix such as 2๓๘ cannot come out.
    """
    best = None
    for keep in (range(1, 11), range(11, 21)):
        mask = torch.full((lp.shape[-1],), -1e9)
        mask[[0, *keep]] = 0
        v, idx = (lp + mask).max(-1)
        cand = list(zip(v.sum(-1).tolist(), idx.tolist(), v.exp().tolist()))
        best = cand if best is None else [max(a, b, key=lambda z: z[0]) for a, b in zip(best, cand)]
    out = []
    for _, row, q in best:
        s, c, prev = "", [], 0
        for k, v in zip(row, q):
            if k and k != prev:
                s += VOCAB[k - 1]
                c.append(v)
            prev = k
        out.append((s, float(np.mean(c)) if c else 0.0))
    return out


def to_input(x, dev):
    return (1 - torch.from_numpy(np.array(x)).to(dev).float() / 255).unsqueeze(1)


def predict(model, x, dev, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out += decode(model(to_input(x[i:i + bs], dev)).cpu())
    return out


def load(path, dev="cpu"):
    m = Reader()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(dev).eval()


def report(pred, t, tier, rot):
    g = defaultdict(list)
    for (p, _), gt, ti, ro in zip(pred, t, tier, rot):
        num = "thai" if gt[0] in VOCAB[:10] else "arabic"
        for k in (num, f"{num} {ti}") + ((f"{num} {'rotated' if ro else 'upright'}",) if ti != "pen" else ()):
            g[k].append(p == gt)
    return {k: (len(v), round(100 * float(np.mean(v)), 1)) for k, v in sorted(g.items())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="ocr_thai.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    D = {n: dict(np.load(Path(a.work) / f"{n}.npz")) for n in ("page_train", "pen_train", "page_val", "pen_val")}
    X = np.concatenate([D["page_train"]["x"], D["pen_train"]["x"]])
    T = np.concatenate([D["page_train"]["t"], D["pen_train"]["t"]])
    if a.limit:
        k = np.random.default_rng(0).permutation(len(X))[: a.limit]
        X, T = X[k], T[k]
    enc = [[VOCAB.index(c) + 1 for c in s] for s in T.tolist()]
    model = Reader().to(dev)
    print(dev, len(X), "images", sum(p.numel() for p in model.parameters()), "weights", flush=True)
    steps = a.epochs * (len(X) // a.batch)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    ctc, best, rng = nn.CTCLoss(zero_infinity=True), -1, np.random.default_rng(1)
    for ep in range(a.epochs):
        model.train()
        t0, tot, order = time.time(), 0.0, rng.permutation(len(X))
        for b in range(len(X) // a.batch):
            k = np.sort(order[b * a.batch:(b + 1) * a.batch])
            x = to_input(X[k], dev)
            x = torch.roll(x, (int(rng.integers(-3, 4)), int(rng.integers(-6, 7))), (2, 3))
            lp = model(x).transpose(0, 1).cpu()  # CTC runs on CPU
            tg = [enc[i] for i in k]
            loss = ctc(lp, torch.tensor([c for s in tg for c in s]), torch.full((len(k),), lp.shape[0]),
                       torch.tensor([len(s) for s in tg]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            sch.step()
            tot += loss.item()
        r = {n: report(predict(model, D[n]["x"], dev), D[n]["t"].tolist(), D[n]["tier"].tolist(), D[n]["rot"].tolist())
             for n in ("page_val", "pen_val")}
        s = np.mean([r["page_val"][f"thai {t}"][1] for t in "SHPQ"])
        print(ep, f"loss {tot / (len(X) // a.batch):.3f} {time.time() - t0:.0f}s val-mean {s:.1f}",
              {k: v[1] for k, v in r["page_val"].items() if "right" not in k and "rot" not in k}, "pen", r["pen_val"]["thai"][1], flush=True)
        if s > best:
            best = s
            torch.save(model.state_dict(), a.out)
    print("best val-mean", round(float(best), 1))
