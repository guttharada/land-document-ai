"""Local prototype: upload a deed, check the suggested parcel number, confirm it.

    .venv/bin/python app.py        then open http://127.0.0.1:8765

Only this computer can reach it.  The uploaded page stays in memory and is never written
to disk.  When the person ticks consent, only the small crop of the confirmed number and
its correct value are saved to feedback/ for later fine-tuning.
"""
import io, json, re, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps

import pipeline
from ocr_data import cut
from read_numbers import upright

HERE, PORT, MAX = Path(__file__).parent, 8765, 25_000_000
FEEDBACK = HERE / "feedback"
HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
Image.MAX_IMAGE_PIXELS = 60_000_000
pages, lock = {}, threading.Lock()  # id -> (image, result); last few only
TH = "๐๑๒๓๔๕๖๗๘๙"


def box_of(v, size):
    if v is None:
        return None
    x0, y0, x1, y1 = (float(n) for n in v)
    x0, x1, y0, y1 = max(0, min(x0, x1)), min(size[0], max(x0, x1)), max(0, min(y0, y1)), min(size[1], max(y0, y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("กรอบเล็กเกินไป")
    return [x0, y0, x1, y1]


def run(pid, im, map_box=None, head_box=None):
    with lock:  # one page at a time: 8 GB machines
        r = pipeline.analyse(im, map_box, head_box)
        pages[pid] = (im, r)
        while len(pages) > 3:
            pages.pop(next(iter(pages)))
    return dict(r, id=pid)


def save(d):
    im, r = pages[d["id"]]
    value = d["value"].translate(str.maketrans(TH, "0123456789")).strip()
    if not re.fullmatch(r"\d{1,6}", value):
        raise ValueError("เลขที่ดินต้องเป็นตัวเลข 1-6 หลัก")
    k = d.get("box")
    rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), value=value, suggested=r["suggestion"], agree=r["agree"],
               changed=value != r["suggestion"], source=r["source"])
    if d.get("consent") is True:
        FEEDBACK.mkdir(exist_ok=True)
        name = uuid.uuid4().hex[:12]
        if isinstance(k, int) and 0 <= k < len(r["boxes"]) and r["boxes"][k]["cls"] in ("parcel", "struck"):
            cut(im, upright([list(p) for p in r["boxes"][k]["quad"]])).save(FEEDBACK / f"{name}.png")
            rec.update(crop=f"{name}.png", read=r["boxes"][k]["text"], script=d.get("script", ""))
        with open(FEEDBACK / "log.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return dict(ok=True, saved=d.get("consent") is True, crop="crop" in rec)


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, kind="application/json; charset=utf-8"):
        body = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        for k, v in (("Content-Type", kind), ("Content-Length", str(len(body))), ("Cache-Control", "no-store"),
                     ("X-Content-Type-Options", "nosniff"),
                     ("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def trusted(self):  # other web pages must not be able to drive this server
        o = self.headers.get("Origin")
        return self.headers.get("Host") in HOSTS and (o is None or o.removeprefix("http://") in HOSTS)

    def do_GET(self):
        if not self.trusted():
            return self.send(403, dict(error="forbidden"))
        if self.path == "/":
            return self.send(200, (HERE / "app.html").read_bytes(), "text/html; charset=utf-8")
        self.send(404, dict(error="not found"))

    def do_POST(self):
        if not self.trusted() or self.headers.get("X-App") != "deed":
            return self.send(403, dict(error="forbidden"))
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX:
                return self.send(413, dict(error="ไฟล์ใหญ่เกิน 25 MB หรือว่าง"))
            body = self.rfile.read(n)
            if self.path == "/api/read":
                try:
                    im = ImageOps.exif_transpose(Image.open(io.BytesIO(body))).convert("RGB")
                except Exception:
                    raise ValueError("เปิดภาพไม่ได้ ใช้ JPG, PNG หรือ WEBP")
                if max(im.size) > 4000:
                    im.thumbnail((4000, 4000))
                return self.send(200, run(uuid.uuid4().hex, im))
            d = json.loads(body)
            if d.get("id") not in pages:
                raise ValueError("หมดอายุ กรุณาเปิดภาพใหม่")
            if self.path == "/api/again":
                im = pages[d["id"]][0]
                return self.send(200, run(d["id"], im, box_of(d.get("map"), im.size), box_of(d.get("head"), im.size)))
            if self.path == "/api/save":
                return self.send(200, save(d))
            self.send(404, dict(error="not found"))
        except (ValueError, KeyError, TypeError) as e:
            self.send(400, dict(error=str(e)))
        except Exception as e:
            print("error:", type(e).__name__, e, flush=True)
            self.send(500, dict(error="ประมวลผลไม่สำเร็จ"))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("loading models...", flush=True)
    pipeline.models()
    print(f"open http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
