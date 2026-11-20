/* Headless smoke test for app/static/app.js using jsdom.
 * Run: node tests/frontend_smoke.js
 * Requires `jsdom` (installed in /tmp/node_modules for this session).
 */
const fs = require("fs");
const path = require("path");
const { JSDOM } = require(process.env.JSDOM_PATH || "/tmp/node_modules/jsdom");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "app/templates/index.html"), "utf8");
const js = fs.readFileSync(path.join(root, "app/static/app.js"), "utf8");

const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true, url: "http://localhost/" });
const { window } = dom;
const { document } = window;

// --- browser APIs jsdom lacks ---------------------------------------------
const listeners = {};
const makeCtx = () => ({
  setTransform() {}, clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
  fillRect() {}, strokeRect() {}, fillText() {}, fill() {},
  set fillStyle(v) {}, set strokeStyle(v) {}, set lineWidth(v) {}, set globalAlpha(v) {},
  set font(v) {},
});
Object.defineProperty(window.HTMLCanvasElement.prototype, "getContext", { value: () => makeCtx() });
window.HTMLCanvasElement.prototype.getBoundingClientRect = function () {
  return { left: 0, top: 0, width: 1000, height: 190, right: 1000, bottom: 190 };
};
window.HTMLCanvasElement.prototype.setPointerCapture = function () {};
window.HTMLCanvasElement.prototype.releasePointerCapture = function () {};
window.URL.createObjectURL = () => "blob:mock";
window.URL.revokeObjectURL = () => {};
window.HTMLMediaElement.prototype.load = function () {};
window.HTMLMediaElement.prototype.play = function () { return Promise.resolve(); };

// --- fetch mock ------------------------------------------------------------
const ANALYSIS = {
  channels: 2, sample_rate: 16000, samples: 96000, duration_s: 6.0,
  delay_samples: 24, delay_ms: 1.5, delay_confirmed: true,
  delay_confidence: 0.95, delay_correlation: 0.1, delay_reason: "",
  regions: [
    { start_sample: 22400, end_sample: 40000, start_ms: 1400, end_ms: 2500, duration_ms: 1100, loss_db: 59, correlation: -0.99 },
    { start_sample: 57600, end_sample: 71200, start_ms: 3600, end_ms: 4450, duration_ms: 850, loss_db: 59, correlation: -0.99 },
  ],
  notes: ["确认右声道比左声道晚 1.5 ms。"],
  downmix_rms_dbfs: -25, repairable: true,
};
const calls = [];
window.fetch = async (url, opts) => {
  calls.push({ url, opts });
  if (url === "/api/analyze") {
    return {
      ok: true, status: 200,
      json: async () => ({
        job_id: "job1", analysis: ANALYSIS,
        waveform: Array.from({ length: 500 }, (_, i) => [-0.1 - (i % 7) / 100, 0.1 + (i % 5) / 100]),
        downmix_url: "/api/jobs/job1/downmix.wav", original_url: "/api/jobs/job1/original.wav",
      }),
    };
  }
  if (url.endsWith("/render")) {
    const body = JSON.parse(opts.body);
    const mode = body.spans.length ? "repaired" : "downmix";
    return {
      ok: true, status: 200,
      headers: { get: (k) => ({
        "X-Result-Mode": mode,
        "X-Total-Samples": "96000",
        "X-Merged-Ranges": body.spans.map((s) => `${s.start_sample}-${s.end_sample}`).join(","),
      }[k] || null) },
      blob: async () => ({ size: 192044, type: "audio/wav" }),
    };
  }
  throw new Error("unexpected fetch " + url);
};

window.eval(js);

const results = [];
function check(name, cond, extra = "") {
  results.push({ name, ok: !!cond, extra });
}

function fire(el, type, init = {}) {
  const ev = new window.Event(type, { bubbles: true, cancelable: true });
  Object.assign(ev, init);
  el.dispatchEvent(ev);
}

function pointer(el, type, clientX) {
  const ev = new window.MouseEvent(type, { bubbles: true, cancelable: true, clientX, clientY: 50 });
  Object.defineProperty(ev, "pointerId", { value: 1 });
  el.dispatchEvent(ev);
}

const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const $ = (s) => document.querySelector(s);
  check("result/editor hidden initially", $("#result-panel").hidden && $("#editor-panel").hidden);

  // select a file
  const file = new window.File([new Uint8Array([1, 2, 3])], "demo.wav", { type: "audio/wav" });
  Object.defineProperty($("#file-input"), "files", { value: [file], configurable: true });
  fire($("#file-input"), "change");
  check("analyze enabled after file pick", !$("#analyze-btn").disabled);
  check("file name shown", $("#file-name").textContent.includes("demo.wav"));

  // analyze
  $("#analyze-btn").click();
  await tick(20);
  check("analyze posted multipart", calls.some((c) => c.url === "/api/analyze" && c.opts.body instanceof window.FormData));
  check("result panel visible", !$("#result-panel").hidden);
  check("editor visible", !$("#editor-panel").hidden);
  check("facts rendered", $("#facts").textContent.includes("1.5 ms"));
  check("notes rendered", $("#notes").textContent.includes("1.5 ms"));
  check("original audio src set", $("#original-audio").getAttribute("src") === "/api/jobs/job1/original.wav");
  check("no spans yet", $("#sel-list").textContent.includes("未选择区间"));

  // select suspects
  $("#select-suspects").click();
  const rows = document.querySelectorAll(".sel-row");
  check("two spans after select suspects", rows.length === 2, `rows=${rows.length}`);
  check("rows show suspect tag", $("#sel-list").textContent.includes("疑似"));

  // edit an endpoint numerically -> overlapping spans
  const startInput = rows[1].querySelector('input[data-field="start"]');
  startInput.value = "30000"; // now overlaps the first span (ends 40000)
  fire(startInput, "change");
  check("overlap accepted", document.querySelectorAll(".sel-row").length === 2);

  // render -> should merge into one range
  $("#render-btn").click();
  await tick(20);
  const renderCall = calls.filter((c) => c.url.endsWith("/render")).pop();
  check("render posted spans", JSON.parse(renderCall.opts.body).spans.length === 2);
  check("result box visible", !$("#result").hidden);
  check("result mode reported", $("#status").textContent.includes("修复完成"));
  check("download link set", $("#download-link").getAttribute("href") === "blob:mock");
  check("download filename", $("#download-link").getAttribute("download") === "repaired-mono.wav");
  check("preview src equals download blob", $("#result-audio").getAttribute("src") === $("#download-link").getAttribute("href"));
  check("merged count mentioned", $("#result-info").textContent.includes("22400-40000"), $("#result-info").textContent);

  // clear selection and render -> downmix mode
  $("#clear-selection").click();
  $("#render-btn").click();
  await tick(20);
  check("downmix mode after clear", $("#result-info").textContent.includes("未选择区间"), $("#result-info").textContent);
  check("downmix status", $("#status").textContent.includes("直接下混"), $("#status").textContent);
  check("downmix filename", $("#download-link").getAttribute("download") === "downmix-mono.wav");

  // drag-create a selection on the canvas
  const canvas = $("#timeline");
  pointer(canvas, "pointerdown", 100);
  pointer(canvas, "pointermove", 300);
  pointer(canvas, "pointerup", 300);
  check("drag creates a span", document.querySelectorAll(".sel-row").length === 1,
    `rows=${document.querySelectorAll(".sel-row").length}`);

  // drag the right edge of that span
  const before = document.querySelector('.sel-row input[data-field="end"]').value;
  pointer(canvas, "pointerdown", 300);
  pointer(canvas, "pointermove", 500);
  pointer(canvas, "pointerup", 500);
  const after = document.querySelector('.sel-row input[data-field="end"]').value;
  check("edge drag extends span", Number(after) > Number(before), `${before}->${after}`);

  // click inside a suspect region away from existing spans -> toggles it on
  $("#clear-selection").click();
  const regionX = Math.round((22400 / 96000) * 1000) + 50; // inside first region
  pointer(canvas, "pointerdown", regionX);
  pointer(canvas, "pointerup", regionX);
  check("click on suspect selects it", document.querySelectorAll(".sel-row").length === 1,
    `rows=${document.querySelectorAll(".sel-row").length}`);
  // clicking a *different* suspect region adds it as a selection
  const region2X = Math.round((64000 / 96000) * 1000); // inside second region
  pointer(canvas, "pointerdown", region2X);
  pointer(canvas, "pointerup", region2X);
  check("click on second suspect adds it", document.querySelectorAll(".sel-row").length === 2,
    `rows=${document.querySelectorAll(".sel-row").length}`);
  // dragging inside an existing selection moves it rather than re-toggling
  const firstStart = document.querySelector('.sel-row input[data-field="start"]').value;
  pointer(canvas, "pointerdown", regionX);
  pointer(canvas, "pointermove", regionX + 30);
  pointer(canvas, "pointerup", regionX + 30);
  const movedStart = document.querySelector('.sel-row input[data-field="start"]').value;
  check("drag inside selection moves it", movedStart !== firstStart, `${firstStart}->${movedStart}`);

  // reset
  $("#reset-btn").click();
  check("reset hides panels", $("#result-panel").hidden && $("#editor-panel").hidden);
  check("reset clears status", $("#status").textContent.includes("等待上传"));

  // unconfirmed-delay path
  const html2 = document.querySelector("#result-panel");
  const resp = await window.fetch("/api/analyze"); // sanity: mock reachable
  check("fetch mock reachable", resp.ok);

  let failed = 0;
  for (const r of results) {
    console.log(`${r.ok ? "PASS" : "FAIL"}  ${r.name}${r.extra ? "  [" + r.extra + "]" : ""}`);
    if (!r.ok) failed++;
  }
  console.log(`\n${results.length - failed}/${results.length} frontend checks passed`);
  process.exit(failed ? 1 : 0);
})();
