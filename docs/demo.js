const $ = id => document.getElementById(id), NS = "http://www.w3.org/2000/svg", TH = "๐๑๒๓๔๕๖๗๘๙";
const COLOR = { parcel: "#d92d20", struck: "#0a8fb0", marker: "#17803d", length: "#8a5a00", area: "#8a2bb8", text: "#667085" };
const NAME = { struck: "เลขที่ดินถูกขีดฆ่า", marker: "หลักเขต", length: "ระยะ", area: "เนื้อที่", text: "ข้อความอื่น" };
const SIZES = [.75, 1, 1.5, 2], LIMIT = 4000;
const worker = new Worker("worker.js", { type: "module" });
let file = null, size = null, box = null, R = null, pick = null, run = 0, busy = false, drawing = false, url = null;

const say = t => { $("msg").textContent = t; };
const arabic = t => t.replace(/[๐-๙]/g, c => TH.indexOf(c));
function setBusy(b) { busy = b; for (const e of document.querySelectorAll("button")) e.disabled = b; $("multi").disabled = b; if (!b) $("bar").hidden = true; }
function setDraw(b) { drawing = b; $("stage").classList.toggle("draw", b); $("area").setAttribute("aria-pressed", b); }

async function load(f) {
  if (!f || busy) return;
  let bm;
  try { bm = await decodeImage(f); } catch { return say("เปิดภาพไม่ได้ ใช้ JPG, PNG หรือ WEBP"); }
  file = f; size = [bm.width, bm.height]; box = null; setDraw(false);
  bm.close();
  if (url) URL.revokeObjectURL(url);
  $("img").src = url = URL.createObjectURL(f);
  $("drop").hidden = true; $("stage").hidden = $("opts").hidden = false; $("whole").hidden = true;
  read();
}
async function decodeImage(f) {
  const bm = await createImageBitmap(f, { colorSpaceConversion: "none" }), k = LIMIT / Math.max(bm.width, bm.height);
  if (k >= 1) return bm;
  const small = await createImageBitmap(bm, { resizeWidth: Math.round(bm.width * k), resizeHeight: Math.round(bm.height * k), resizeQuality: "high" });
  bm.close();
  return small;
}
async function read() {
  if (!file || busy) return;
  setBusy(true); R = pick = null; $("result").hidden = true; draw(); say("กำลังเตรียม...");
  const bitmap = await decodeImage(file);
  worker.postMessage({ id: ++run, bitmap, box, scales: $("multi").checked ? SIZES : [1], cpuOnly: new URLSearchParams(location.search).has("cpu") }, [bitmap]);
}
worker.onmessage = ({ data: m }) => {
  if (m.id !== run) return;
  if (m.type === "load") say("กำลังโหลดโมเดลครั้งแรก...");
  else if (m.type === "progress") { say(`กำลังอ่าน ส่วนที่ ${m.n} จาก ${m.of}`); $("bar").hidden = false; $("bar").value = m.n / m.of; }
  else if (m.type === "error") { setBusy(false); say("อ่านไม่สำเร็จ: " + m.message); }
  else { R = m; setBusy(false); show(); }
};
worker.onerror = () => { setBusy(false); say("เบราว์เซอร์นี้รันโมเดลไม่ได้ ลอง Chrome, Edge หรือ Safari รุ่นใหม่"); };

function show() {
  const P = R.boxes.filter(b => b.cls === 0), N = R.boxes.reduce((o, b) => (o[b.cls] = (o[b.cls] || 0) + 1, o), {});
  say(P.length ? "เสร็จแล้ว ตรวจเลขกับภาพทุกครั้ง" : "หาเลขที่ดินไม่เจอ ลองลากเลือกเฉพาะแผนที่ หรือลองหลายขนาด");
  $("count").textContent = P.length ? `(${P.length})` : "";
  $("list").replaceChildren(...P.map((b, i) => {
    const li = document.createElement("li"), bt = document.createElement("button"), n = document.createElement("b"), s = document.createElement("span");
    n.textContent = b.text || "อ่านไม่ออก";
    s.textContent = b.text ? `${arabic(b.text)} · มั่นใจ ${Math.round(b.sure * 100)}%` : "";
    bt.setAttribute("aria-pressed", "false"); bt.onclick = () => select(i); bt.append(n, s); li.append(bt);
    return li;
  }));
  $("legend").replaceChildren(...["struck", "marker", "length", "area", "text"].map((k, i) => {
    const li = document.createElement("li"), c = document.createElement("i");
    c.style.borderColor = COLOR[k]; li.append(c, `${NAME[k]} ${N[i + 1] || 0}`);
    return li;
  }));
  $("time").textContent = `ใช้ ${(R.ms / 1000).toFixed(1)} วินาที (${R.ep === "webgpu" ? "GPU" : "CPU"}) ขนาดที่ใช้ ${R.scale} เท่า`;
  $("result").hidden = false;
  draw();
}
function select(i) {
  pick = pick === i ? null : i;
  [...$("list").querySelectorAll("button")].forEach((b, k) => { b.setAttribute("aria-pressed", k === pick); if (k === pick) b.scrollIntoView({ block: "nearest" }); });
  draw();
}
function el(n, a) { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); return e; }
function frame(b, c) { return el("rect", { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1], fill: "none", stroke: c, "stroke-width": size[0] / 400, "stroke-dasharray": size[0] / 90 }); }
function draw() {
  const s = $("svg");
  if (!size) return;
  s.setAttribute("viewBox", `0 0 ${size[0]} ${size[1]}`); s.replaceChildren();
  if (box) s.append(frame(box, "#6b4fd8"));
  if (!R) return;
  const w = size[0] / 600, K = ["parcel", "struck", "marker", "length", "area", "text"];
  R.boxes.forEach((b, i) => {
    const k = K[b.cls], on = i === pick, p = el("polygon", { points: b.quad.map(q => q.join()).join(" "), stroke: COLOR[k], "stroke-width": on ? w * 2.4 : w, fill: on ? "rgba(217,45,32,.25)" : "transparent" });
    if (b.cls === 0) {
      p.setAttribute("class", "hit"); p.addEventListener("click", () => { if (!drawing) select(i); });
      const t = el("text", { x: Math.min(...b.quad.map(q => q[0])), y: Math.min(...b.quad.map(q => q[1])) - w * 2, fill: COLOR[k], "font-size": w * (on ? 13 : 9), "font-weight": 600 });
      t.textContent = b.text; s.append(t);
    }
    s.append(p);
  });
}

let start = null, ghost = null;
const at = e => { const r = $("svg").getBoundingClientRect(); return [(e.clientX - r.left) / r.width * size[0], (e.clientY - r.top) / r.height * size[1]].map((v, i) => Math.max(0, Math.min(size[i], v))); };
const span = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
$("svg").addEventListener("pointerdown", e => { if (!drawing || busy) return; e.preventDefault(); start = at(e); $("svg").setPointerCapture(e.pointerId); });
$("svg").addEventListener("pointermove", e => { if (!start) return; ghost?.remove(); ghost = frame(span(start, at(e)), "#6b4fd8"); $("svg").append(ghost); });
const cancel = () => { start = null; ghost?.remove(); ghost = null; };
$("svg").addEventListener("pointercancel", cancel);
$("svg").addEventListener("pointerup", e => {
  if (!start) return;
  const b = span(start, at(e)); cancel();
  if (b[2] - b[0] < 20 || b[3] - b[1] < 20) return;
  box = b; setDraw(false); $("whole").hidden = false; read();
});

$("open").onclick = () => $("file").click();
$("file").onchange = e => { load(e.target.files[0]); e.target.value = ""; };
$("area").onclick = () => setDraw(!drawing);
$("whole").onclick = () => { box = null; $("whole").hidden = true; read(); };
$("again").onclick = read;
for (const b of document.querySelectorAll("[data-s]")) b.onclick = async () => {
  try { const r = await fetch(`samples/${b.dataset.s}.jpg`); if (!r.ok) throw 0; load(await r.blob()); } catch { say("โหลดภาพจำลองไม่ได้"); }
};
for (const n of ["dragover", "dragleave", "drop"]) document.addEventListener(n, e => { e.preventDefault(); $("drop").classList.toggle("on", n === "dragover"); if (n === "drop") load(e.dataTransfer.files[0]); });
