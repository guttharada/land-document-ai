// Runs both models off the main thread.  The image is never sent anywhere.
import { analyse, fromRGBA } from "./pipeline.js";
let ready = null;
async function open(cpuOnly) {
  const gpu = !cpuOnly && "gpu" in navigator && await navigator.gpu.requestAdapter().catch(() => null);
  const ort = await import(gpu ? "./ort/ort.webgpu.min.mjs" : "./ort/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = new URL("./ort/", import.meta.url).href;
  ort.env.wasm.numThreads = 1;
  const make = (f, ep) => ort.InferenceSession.create(new URL(`./models/${f}`, import.meta.url).href, { executionProviders: [ep] });
  let ep = gpu ? "webgpu" : "wasm", yolo;
  try { yolo = await make("yolo.onnx", ep); } catch (e) { if (!gpu) throw e; console.warn("WebGPU failed:", e.message); yolo = await make("yolo.onnx", ep = "wasm"); }
  const reader = await make("reader.onnx", "wasm");
  return { ep, yolo: async t => { const o = (await yolo.run({ images: new ort.Tensor("float32", t, [1, 3, 960, 960]) })).output0; return { data: o.data, n: o.dims[2] }; },
    reader: async t => (await reader.run({ x: new ort.Tensor("float32", t, [1, 1, 48, 192]) })).y.data };
}
onmessage = async ({ data: m }) => {
  try {
    if (!ready) postMessage({ id: m.id, type: "load" });
    ready ??= open(m.cpuOnly);
    const models = await ready.catch(e => { ready = null; throw e; }), t0 = performance.now();
    const cv = new OffscreenCanvas(m.bitmap.width, m.bitmap.height), g = cv.getContext("2d", { willReadFrequently: true });
    g.fillStyle = "#fff"; g.fillRect(0, 0, cv.width, cv.height); g.drawImage(m.bitmap, 0, 0); m.bitmap.close();
    const im = fromRGBA(g.getImageData(0, 0, cv.width, cv.height).data, cv.width, cv.height);
    const r = await analyse(im, m.box, m.scales, models, (n, of) => postMessage({ id: m.id, type: "progress", n, of }));
    postMessage({ id: m.id, type: "done", ...r, ep: models.ep, ms: Math.round(performance.now() - t0) });
  } catch (e) { postMessage({ id: m.id, type: "error", message: String(e?.message || e) }); }
};
