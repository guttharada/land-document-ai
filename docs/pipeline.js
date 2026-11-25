// Browser port of yolo_tiles.py, read_numbers.py, ocr_data.fit/cut and pipeline.detect.
// Image arithmetic follows Pillow so that results stay close to the Python version.
export const CLASSES = ["parcel", "struck", "marker", "length", "area", "text"];
const VOCAB = "๐๑๒๓๔๕๖๗๘๙0123456789", TILE = 960, OVERLAP = 160, IH = 48, IW = 192;
const clip8 = v => v < 0 ? 0 : v > 255 ? 255 : v;
const dist = (a, b) => Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);
const center = q => [(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4];
const shortSide = q => Math.min(dist(q[0], q[1]), dist(q[1], q[2]));

// ---------- images: {w, h, n, d} with n channels, 8 bit ----------
const FILTERS = {
  bilinear: [1, x => Math.max(0, 1 - Math.abs(x))],
  bicubic: [2, x => { x = Math.abs(x); return x < 1 ? (1.5 * x - 2.5) * x * x + 1 : x < 2 ? ((-.5 * x + 2.5) * x - 4) * x + 2 : 0; }],
  lanczos: [3, x => { if (x === 0) return 1; if (Math.abs(x) >= 3) return 0; const p = Math.PI * x; return Math.sin(p) / p * Math.sin(p / 3) / (p / 3); }],
};
function coeffs(inSize, outSize, [support, f]) {
  const scale = inSize / outSize, fs = Math.max(scale, 1), sup = support * fs, out = [];
  for (let i = 0; i < outSize; i++) {
    const c = (i + .5) * scale, lo = Math.max(0, Math.trunc(c - sup + .5)), hi = Math.min(inSize, Math.trunc(c + sup + .5));
    const k = new Float64Array(hi - lo);
    let s = 0;
    for (let j = lo; j < hi; j++) s += k[j - lo] = f((j + .5 - c) / fs);
    for (let j = 0; j < k.length; j++) k[j] /= s;
    out.push([lo, k]);
  }
  return out;
}
export function resize(im, w, h, name) {
  const { n } = im, f = FILTERS[name];
  if (w !== im.w) {
    const C = coeffs(im.w, w, f), d = new Uint8ClampedArray(w * im.h * n);
    for (let y = 0; y < im.h; y++) for (let x = 0; x < w; x++) {
      const [lo, k] = C[x];
      for (let b = 0; b < n; b++) {
        let s = .5, p = (y * im.w + lo) * n + b;
        for (let j = 0; j < k.length; j++, p += n) s += im.d[p] * k[j];
        d[(y * w + x) * n + b] = Math.floor(clip8(s));
      }
    }
    im = { w, h: im.h, n, d };
  }
  if (h !== im.h) {
    const C = coeffs(im.h, h, f), d = new Uint8ClampedArray(w * h * n);
    for (let y = 0; y < h; y++) {
      const [lo, k] = C[y];
      for (let x = 0; x < w * n; x++) {
        let s = .5, p = lo * w * n + x;
        for (let j = 0; j < k.length; j++, p += w * n) s += im.d[p] * k[j];
        d[y * w * n + x] = Math.floor(clip8(s));
      }
    }
    im = { w, h, n, d };
  }
  return im;
}
export function fromRGBA(data, w, h) {
  const d = new Uint8ClampedArray(w * h * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) { d[j] = data[i]; d[j + 1] = data[i + 1]; d[j + 2] = data[i + 2]; }
  return { w, h, n: 3, d };
}
function crop(im, x0, y0, x1, y1) {
  const w = x1 - x0, h = y1 - y0, d = new Uint8ClampedArray(w * h * im.n);
  for (let y = 0; y < h; y++) d.set(im.d.subarray(((y + y0) * im.w + x0) * im.n, ((y + y0) * im.w + x1) * im.n), y * w * im.n);
  return { w, h, n: im.n, d };
}
function padTo(im, w, h) {
  if (im.w >= w && im.h >= h) return im;
  w = Math.max(w, im.w); h = Math.max(h, im.h);
  const c = resize(im, 1, 1, "bicubic").d, d = new Uint8ClampedArray(w * h * 3);
  for (let i = 0; i < d.length; i++) d[i] = c[i % 3];
  for (let y = 0; y < im.h; y++) d.set(im.d.subarray(y * im.w * 3, (y + 1) * im.w * 3), y * w * 3);
  return { w, h, n: 3, d };
}

// ---------- YOLO-OBB ----------
const starts = size => {
  if (size <= TILE) return [0];
  const s = [];
  for (let v = 0; v < size - TILE; v += TILE - OVERLAP) s.push(v);
  return [...s, size - TILE];
};
function nms(out, n, conf = .15, iou = .5, maxDet = 600) {  // Ultralytics fast NMS with probIoU, per class
  const B = [];
  for (let i = 0; i < n; i++) {
    let c = 0, s = -1;
    for (let k = 0; k < 6; k++) if (out[(4 + k) * n + i] > s) { s = out[(4 + k) * n + i]; c = k; }
    if (s <= conf) continue;
    const w = out[2 * n + i], h = out[3 * n + i], t = out[10 * n + i], A = w * w / 12, D = h * h / 12, co = Math.cos(t), si = Math.sin(t);
    B.push({ x: out[i], y: out[n + i], w, h, t, c, s, co, si, r: Math.hypot(w, h),
      a: A * co * co + D * si * si, b: A * si * si + D * co * co, k: (A - D) * co * si });
  }
  B.sort((p, q) => q.s - p.s);
  const E = 1e-7, keep = [];
  for (let j = 0; j < B.length && keep.length < maxDet; j++) {
    const q = B[j], hit = B.some((p, i) => {
      if (i >= j || p.c !== q.c || Math.hypot(p.x - q.x, p.y - q.y) > p.r + q.r) return false;
      const a = p.a + q.a, b = p.b + q.b, k = p.k + q.k, dx = p.x - q.x, dy = p.y - q.y, det = a * b - k * k;
      const t1 = (a * dy * dy + b * dx * dx) / (det + E) * .25, t2 = k * -dx * dy / (det + E) * .5;
      const t3 = Math.log(det / (4 * Math.sqrt(Math.max(0, p.a * p.b - p.k * p.k) * Math.max(0, q.a * q.b - q.k * q.k)) + E) + E) * .5;
      const bd = Math.min(100, Math.max(E, t1 + t2 + t3));
      return 1 - Math.sqrt(1 - Math.exp(-bd) + E) > iou;
    });
    if (!hit) keep.push(q);
  }
  return keep.map(p => {
    const ax = p.w / 2 * p.co, ay = p.w / 2 * p.si, bx = -p.h / 2 * p.si, by = p.h / 2 * p.co;
    return { cls: p.c, conf: p.s, quad: [[p.x + ax + bx, p.y + ay + by], [p.x + ax - bx, p.y + ay - by], [p.x - ax - bx, p.y - ay - by], [p.x - ax + bx, p.y - ay + by]] };
  });
}
async function predictPage(yolo, im, tick) {
  const found = [], Y = starts(im.h), X = starts(im.w);
  for (const y of Y) for (const x of X) {
    const t = new Float32Array(3 * TILE * TILE), A = TILE * TILE;
    for (let r = 0, i = 0; r < TILE; r++) for (let c = 0, p = ((y + r) * im.w + x) * 3; c < TILE; c++, i++, p += 3) {
      t[i] = im.d[p] / 255; t[A + i] = im.d[p + 1] / 255; t[2 * A + i] = im.d[p + 2] / 255;
    }
    const { data, n } = await yolo(t);
    for (const p of nms(data, n)) {
      const xs = p.quad.map(q => q[0]), ys = p.quad.map(q => q[1]), m = 3;  // cut by an inner tile edge
      if ((x > 0 && Math.min(...xs) < m) || (y > 0 && Math.min(...ys) < m) ||
        (x + TILE < im.w && Math.max(...xs) > TILE - m) || (y + TILE < im.h && Math.max(...ys) > TILE - m)) continue;
      found.push({ ...p, quad: p.quad.map(q => [q[0] + x, q[1] + y]) });
    }
    tick();
  }
  const keep = [];
  for (const d of found.sort((a, b) => b.conf - a.conf)) {
    const c = center(d.quad), s = shortSide(d.quad);
    if (keep.every(k => dist(c, center(k.quad)) > .5 * Math.min(s, shortSide(k.quad)))) keep.push(d);
  }
  return keep;
}

// ---------- reader ----------
function upright(q) {
  let s = 0;
  for (let i = 0; i < 4; i++) { const a = q[i], b = q[(i + 1) % 4]; s += a[0] * b[1] - b[0] * a[1]; }
  if (s < 0) q = [...q].reverse();
  if (dist(q[0], q[1]) < dist(q[1], q[2])) q = [...q.slice(1), q[0]];
  const ang = Math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]) * 180 / Math.PI;
  return ang >= -100 && ang < 80 ? q : [...q.slice(2), ...q.slice(0, 2)];
}
function cut(im, q, pad = .25, height = 40) {  // PIL transform(AFFINE, BICUBIC), then convert("L")
  const w = dist(q[0], q[1]), h = Math.max(4, dist(q[1], q[2])), s = h / height, p = h * pad;
  const ux = (q[1][0] - q[0][0]) / w, uy = (q[1][1] - q[0][1]) / w, vx = (q[3][0] - q[0][0]) / h, vy = (q[3][1] - q[0][1]) / h;
  const W = Math.max(8, Math.trunc((w + 2 * p) / s)), H = Math.max(8, Math.trunc((h + 2 * p) / s)), d = new Uint8ClampedArray(W * H);
  const cx = q[0][0] - p * (ux + vx), cy = q[0][1] - p * (uy + vy), f = (Math.min(5, im.h - 1) * im.w + Math.min(5, im.w - 1)) * 3;
  const cubic = (a, b, c, e, t) => b + t * (-a + c + t * (2 * (a - b) + c - e + t * (-a + b - c + e)));
  const lim = (v, n) => v < 0 ? 0 : v >= n ? n - 1 : v, rgb = [0, 0, 0];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let sx = ux * s * (x + .5) + vx * s * (y + .5) + cx, sy = uy * s * (x + .5) + vy * s * (y + .5) + cy;
    if (sx < 0 || sx >= im.w || sy < 0 || sy >= im.h) for (let b = 0; b < 3; b++) rgb[b] = im.d[f + b];
    else {
      sx -= .5; sy -= .5;
      const x0 = Math.floor(sx), y0 = Math.floor(sy), dx = sx - x0, dy = sy - y0;
      const X = [-1, 0, 1, 2].map(i => lim(x0 + i, im.w));
      for (let b = 0; b < 3; b++) {
        const R = [];
        for (let i = -1; i < 3; i++) { const o = lim(y0 + i, im.h) * im.w; R.push(cubic(...X.map(v => im.d[(o + v) * 3 + b]), dx)); }
        rgb[b] = Math.trunc(clip8(cubic(...R, dy)));
      }
    }
    d[y * W + x] = (rgb[0] * 19595 + rgb[1] * 38470 + rgb[2] * 7471 + 0x8000) >> 16;
  }
  return { w: W, h: H, n: 1, d };
}
function fit(im) {
  const v = Float32Array.from(im.d).sort(), pct = p => { const k = (v.length - 1) * p, i = Math.floor(k); return v[i] + (v[Math.min(i + 1, v.length - 1)] - v[i]) * (k - i); };
  const lo = pct(.01), hi = pct(.9), z = Math.max(hi - lo, 8), g = { ...im, d: Uint8ClampedArray.from(im.d, a => Math.trunc(clip8((a - lo) / z * 255))) };
  const k = Math.min((IH - 8) / im.h, (IW - 8) / im.w), r = resize(g, Math.max(1, Math.trunc(im.w * k)), Math.max(1, Math.trunc(im.h * k)), "bilinear");
  const out = new Float32Array(IW * IH), ox = Math.floor((IW - r.w) / 2), oy = Math.floor((IH - r.h) / 2);  // 0 = paper, 1 = ink
  for (let y = 0; y < r.h; y++) for (let x = 0; x < r.w; x++) out[(y + oy) * IW + x + ox] = 1 - r.d[y * r.w + x] / 255;
  return out;
}
function decode(lp) {  // 48 steps x 21; Thai-only and Arabic-only readings compared, the likelier kept
  let best = null;
  for (const from of [1, 11]) {
    let sum = 0, prev = 0, text = "", q = [];
    for (let t = 0; t < 48; t++) {
      let k = 0, v = lp[t * 21];
      for (let i = from; i < from + 10; i++) if (lp[t * 21 + i] > v) { v = lp[t * 21 + i]; k = i; }
      sum += v;
      if (k && k !== prev) { text += VOCAB[k - 1]; q.push(Math.exp(v)); }
      prev = k;
    }
    if (!best || sum > best.sum) best = { sum, text, sure: q.length ? q.reduce((a, b) => a + b) / q.length : 0 };
  }
  return best;
}

// ---------- whole page ----------
/** im: RGB image, box: map area or null, scales: sizes to try. models: {yolo, reader}. */
export async function analyse(im, box, scales, models, progress = () => {}) {
  box = box ? box.map(Math.trunc) : [0, 0, im.w, im.h];
  const c = crop(im, ...box), todo = [];
  for (const s of scales) {
    const w = Math.trunc(c.w * s), h = Math.trunc(c.h * s), m = Math.max(w, h);
    if (scales.length > 1 && (m > 3200 || (m < 500 && s < 2))) continue;
    todo.push([s, w, h, starts(Math.max(w, TILE)).length * starts(Math.max(h, TILE)).length]);
  }
  if (!todo.length) todo.push([1, c.w, c.h, starts(Math.max(c.w, TILE)).length * starts(Math.max(c.h, TILE)).length]);
  const total = todo.reduce((s, t) => s + t[3], 0);
  let done = 0, best = null;
  for (const [s, w, h] of todo) {
    const P = (await predictPage(models.yolo, padTo(resize(c, w, h, "lanczos"), TILE, TILE), () => progress(++done, total))).filter(p => p.conf >= .25);
    const score = P.reduce((t, p) => t + ((p.cls === 0 || p.cls === 2) && p.conf >= .5 ? p.conf : 0), 0);
    if (!best || score > best.score) best = { score, s, P };
  }
  for (const p of best.P) p.quad = p.quad.map(q => [q[0] / best.s + box[0], q[1] / best.s + box[1]]);
  const parcels = best.P.filter(p => p.cls === 0);
  for (const p of parcels) { const r = decode(await models.reader(fit(cut(im, upright(p.quad))))); p.text = r.text; p.sure = r.sure; }
  return { size: [im.w, im.h], box, scale: best.s, tiles: total, boxes: [...parcels, ...best.P.filter(p => p.cls)] };
}
