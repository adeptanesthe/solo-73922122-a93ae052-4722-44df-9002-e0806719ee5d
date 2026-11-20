/* 反相修复台 — upload, inspect, select spans, render one mono WAV. */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const statusEl = $("#status");
  const fileInput = $("#file-input");
  const dropzone = $("#dropzone");
  const fileName = $("#file-name");
  const analyzeBtn = $("#analyze-btn");
  const resetBtn = $("#reset-btn");
  const resultPanel = $("#result-panel");
  const editorPanel = $("#editor-panel");
  const factsEl = $("#facts");
  const notesEl = $("#notes");
  const originalAudio = $("#original-audio");
  const canvas = $("#timeline");
  const ruler = $("#ruler");
  const selList = $("#sel-list");
  const resultBox = $("#result");
  const resultAudio = $("#result-audio");
  const downloadLink = $("#download-link");
  const resultInfo = $("#result-info");

  const state = {
    file: null,
    jobId: null,
    analysis: null,
    peaks: [],
    sampleRate: 16000,
    total: 0,
    spans: [],          // { id, start, end, kind }
    nextId: 1,
    drag: null,
    resultUrl: null,
    renderSeq: 0,
  };

  const MIN_SPAN = 160; // 10 ms at 16 kHz
  const EDGE_PX = 7;

  // ---------- helpers ----------
  function setStatus(text, cls = "") {
    statusEl.textContent = text;
    statusEl.className = "status " + cls;
  }

  function fmtMs(samples, rate = state.sampleRate) {
    return (samples * 1000 / rate).toFixed(1);
  }

  function fmtTime(samples, rate = state.sampleRate) {
    const t = samples / rate;
    const m = Math.floor(t / 60);
    const s = t - m * 60;
    return `${m}:${s.toFixed(2).padStart(5, "0")}`;
  }

  function clampSpan(s) {
    let start = Math.max(0, Math.min(Math.round(s.start), state.total));
    let end = Math.max(0, Math.min(Math.round(s.end), state.total));
    if (end < start) [start, end] = [end, start];
    if (end - start < MIN_SPAN) {
      end = Math.min(state.total, start + MIN_SPAN);
      start = Math.min(start, end - MIN_SPAN);
    }
    s.start = Math.max(0, start);
    s.end = Math.max(0, end);
    return s;
  }

  // ---------- canvas ----------
  function setupCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.round(190 * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: rect.width, height: 190 };
  }

  function xToSample(x, width) {
    return Math.round(Math.max(0, Math.min(1, x / width)) * state.total);
  }

  function sampleToX(s, width) {
    return state.total ? (s / state.total) * width : 0;
  }

  function drawTimeline() {
    if (!state.total) return;
    const { ctx, width, height } = setupCanvas();
    ctx.clearRect(0, 0, width, height);

    const mid = height * 0.52;
    const half = height * 0.36;

    // grid
    ctx.strokeStyle = "rgba(255,255,255,.05)";
    ctx.lineWidth = 1;
    const secs = state.total / state.sampleRate;
    const step = secs > 20 ? 5 : secs > 8 ? 2 : 1;
    for (let t = 0; t <= secs; t += step) {
      const x = sampleToX(t * state.sampleRate, width);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(255,255,255,.09)";
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();

    // suspect regions
    const regions = state.analysis?.regions || [];
    for (const r of regions) {
      const x1 = sampleToX(r.start_sample, width);
      const x2 = sampleToX(r.end_sample, width);
      ctx.fillStyle = "rgba(240,180,94,.13)";
      ctx.fillRect(x1, 0, Math.max(1, x2 - x1), height);
      ctx.strokeStyle = "rgba(240,180,94,.55)";
      ctx.strokeRect(x1 + .5, .5, Math.max(1, x2 - x1 - 1), height - 1);
    }

    // waveform
    const peaks = state.peaks;
    if (peaks.length) {
      ctx.fillStyle = "#7fe3d8";
      ctx.globalAlpha = 0.9;
      const bw = width / peaks.length;
      for (let i = 0; i < peaks.length; i++) {
        const [lo, hi] = peaks[i];
        const y1 = mid - hi * half;
        const y2 = mid - lo * half;
        ctx.fillRect(i * bw, y1, Math.max(bw, 1), Math.max(1, y2 - y1));
      }
      ctx.globalAlpha = 1;
    }

    // selections
    for (const s of state.spans) {
      const x1 = sampleToX(s.start, width);
      const x2 = sampleToX(s.end, width);
      const w = Math.max(2, x2 - x1);
      ctx.fillStyle = "rgba(61,123,247,.22)";
      ctx.fillRect(x1, 0, w, height);
      ctx.strokeStyle = "rgba(110,160,255,.95)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x1 + .75, .75, w - 1.5, height - 1.5);
      // handles
      ctx.fillStyle = "#a9c6ff";
      ctx.fillRect(x1, 0, 3, height);
      ctx.fillRect(x2 - 3, 0, 3, height);
      // fade zones
      const fadeSamples = Math.min(80, (s.end - s.start) / 2);
      const fw = sampleToX(s.start + fadeSamples, width) - x1;
      if (fw > 1) {
        ctx.fillStyle = "rgba(90,209,194,.20)";
        ctx.fillRect(x1, height - 8, fw, 8);
        ctx.fillRect(x2 - fw, height - 8, fw, 8);
      }
      ctx.fillStyle = "rgba(169,198,255,.9)";
      ctx.font = "11px ui-monospace, monospace";
      const label = `${fmtMs(s.start)}–${fmtMs(s.end)} ms`;
      ctx.fillText(label, Math.min(x1 + 6, width - 90), 13);
    }

    // ruler labels
    ruler.innerHTML = "";
    for (let t = 0; t <= secs + 0.001; t += step) {
      const el = document.createElement("span");
      el.textContent = `${t}s`;
      el.style.position = "absolute";
      el.style.left = `${(t / secs) * 100}%`;
      el.style.transform = "translateX(-50%)";
      ruler.appendChild(el);
    }
    if (!ruler.style.position || ruler.style.position === "static") {
      ruler.style.position = "relative";
      ruler.style.height = "16px";
    }
  }

  function hitTest(px, width) {
    for (let i = state.spans.length - 1; i >= 0; i--) {
      const s = state.spans[i];
      const x1 = sampleToX(s.start, width);
      const x2 = sampleToX(s.end, width);
      if (Math.abs(px - x1) <= EDGE_PX) return { span: s, mode: "start" };
      if (Math.abs(px - x2) <= EDGE_PX) return { span: s, mode: "end" };
      if (px > x1 && px < x2) return { span: s, mode: "move" };
    }
    return null;
  }

  function regionAt(px, width) {
    for (const r of state.analysis?.regions || []) {
      const x1 = sampleToX(r.start_sample, width);
      const x2 = sampleToX(r.end_sample, width);
      if (px >= x1 && px <= x2) return r;
    }
    return null;
  }

  function addSpan(start, end, kind = "manual") {
    const span = clampSpan({ id: state.nextId++, start, end, kind });
    state.spans.push(span);
    syncFromSpans();
    return span;
  }

  // ---------- pointer editing ----------
  canvas.addEventListener("pointerdown", (ev) => {
    if (!state.total || !state.analysis?.delay_confirmed) return;
    const rect = canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const hit = hitTest(x, rect.width);
    canvas.setPointerCapture(ev.pointerId);
    if (hit) {
      state.drag = {
        mode: hit.mode,
        span: hit.span,
        startX: x,
        orig: { start: hit.span.start, end: hit.span.end },
        moved: false,
        width: rect.width,
      };
    } else {
      state.drag = {
        mode: "create",
        span: null,
        anchor: xToSample(x, rect.width),
        startX: x,
        cur: xToSample(x, rect.width),
        moved: false,
        width: rect.width,
      };
    }
  });

  canvas.addEventListener("pointermove", (ev) => {
    if (!state.drag) {
      const rect = canvas.getBoundingClientRect();
      const hit = hitTest(ev.clientX - rect.left, rect.width);
      canvas.style.cursor = hit
        ? hit.mode === "move" ? "grab" : "ew-resize"
        : state.analysis?.regions?.length ? "cell" : "crosshair";
      return;
    }
    const d = state.drag;
    const x = ev.clientX - canvas.getBoundingClientRect().left;
    if (Math.abs(x - d.startX) > 2) d.moved = true;

    if (d.mode === "create") {
      d.cur = xToSample(x, d.width);
    } else if (d.mode === "move") {
      const delta = xToSample(x, d.width) - xToSample(d.startX, d.width);
      const len = d.orig.end - d.orig.start;
      let start = d.orig.start + delta;
      start = Math.max(0, Math.min(state.total - len, start));
      d.span.start = start;
      d.span.end = start + len;
    } else if (d.mode === "start") {
      d.span.start = Math.min(xToSample(x, d.width), d.span.end - MIN_SPAN);
    } else if (d.mode === "end") {
      d.span.end = Math.max(xToSample(x, d.width), d.span.start + MIN_SPAN);
    }
    drawTimeline();
  });

  function endDrag() {
    const d = state.drag;
    if (!d) return;
    state.drag = null;
    if (d.mode === "create") {
      if (d.moved && Math.abs(d.cur - d.anchor) >= MIN_SPAN) {
        addSpan(d.anchor, d.cur);
      } else if (!d.moved) {
        const rect = canvas.getBoundingClientRect();
        const x = d.startX;
        const r = regionAt(x, rect.width);
        if (r) toggleRegion(r);
      }
      return;
    }
    clampSpan(d.span);
    syncFromSpans();
  }

  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  window.addEventListener("resize", () => drawTimeline());

  // ---------- selections ----------
  function toggleRegion(r) {
    const existing = state.spans.find(
      (s) => Math.abs(s.start - r.start_sample) <= 1 && Math.abs(s.end - r.end_sample) <= 1
    );
    if (existing) {
      state.spans = state.spans.filter((s) => s !== existing);
    } else {
      addSpan(r.start_sample, r.end_sample, "suspect");
    }
    syncFromSpans();
  }

  function syncFromSpans() {
    // drop degenerate spans, keep order by start
    state.spans = state.spans.filter((s) => s.end > s.start);
    state.spans.sort((a, b) => a.start - b.start);
    renderSpanList();
    drawTimeline();
  }

  function renderSpanList() {
    selList.innerHTML = "";
    if (!state.spans.length) {
      const empty = document.createElement("div");
      empty.className = "sel-empty";
      empty.textContent = "未选择区间。点击橙色疑似区间、拖动波形空白处，或使用“全选疑似区间”。";
      selList.appendChild(empty);
      return;
    }
    state.spans.forEach((s) => {
      const row = document.createElement("div");
      row.className = "sel-row";
      row.innerHTML = `
        <span class="tag">${s.kind === "suspect" ? "疑似" : "手动"} · ${fmtTime(s.start)} – ${fmtTime(s.end)}</span>
        <label>起点 <input type="number" min="0" max="${state.total}" step="1" value="${s.start}" data-field="start"></label>
        <label>终点 <input type="number" min="0" max="${state.total}" step="1" value="${s.end}" data-field="end"></label>
        <span class="tag">${fmtMs(s.end - s.start)} ms</span>
        <button class="ghost" type="button" title="删除该选区">删除</button>`;
      row.querySelectorAll("input").forEach((input) => {
        input.addEventListener("change", () => {
          const v = parseInt(input.value, 10);
          if (!Number.isFinite(v)) return;
          s[input.dataset.field] = v;
          clampSpan(s);
          syncFromSpans();
        });
      });
      row.querySelector("button").addEventListener("click", () => {
        state.spans = state.spans.filter((x) => x !== s);
        syncFromSpans();
      });
      selList.appendChild(row);
    });
  }

  // ---------- analysis / rendering ----------
  function factsFromAnalysis(a) {
    const delayText = a.delay_confirmed
      ? `${a.delay_ms > 0 ? "+" : ""}${a.delay_ms.toFixed(1)} ms (${a.delay_samples} 采样)`
      : "未确认";
    const delayCls = a.delay_confirmed ? "good" : "bad";
    const items = [
      ["时长", `${a.duration_s.toFixed(2)} s`, ""],
      ["格式", `${a.sample_rate} Hz · ${a.channels} ch · 16-bit`, ""],
      ["固定时差", delayText, delayCls],
      ["相关强度", `${(a.delay_confidence * 100).toFixed(1)}%`, a.delay_confidence >= 0.8 ? "good" : "warn"],
      ["疑似反相区间", `${a.regions.length} 段`, a.regions.length ? "warn" : "good"],
      ["直接下混 RMS", `${a.downmix_rms_dbfs.toFixed(1)} dBFS`, ""],
    ];
    factsEl.innerHTML = items
      .map(([k, v, cls]) => `<div class="fact ${cls}"><dt>${k}</dt><dd>${v}</dd></div>`)
      .join("");
  }

  function notesFromAnalysis(a) {
    notesEl.innerHTML = "";
    const notes = [...(a.notes || [])];
    if (!a.delay_confirmed && a.delay_reason) notes.push(`原因：${a.delay_reason}`);
    for (const n of notes) {
      const div = document.createElement("div");
      div.className = "note" + (n.includes("未能确认") ? " bad" : "");
      div.textContent = n;
      notesEl.appendChild(div);
    }
  }

  async function analyze() {
    if (!state.file) return;
    setStatus("正在分析…", "busy");
    analyzeBtn.disabled = true;
    resetBtn.disabled = true;
    const form = new FormData();
    form.append("file", state.file, state.file.name);
    try {
      const resp = await fetch("/api/analyze", { method: "POST", body: form });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `分析失败（HTTP ${resp.status}）`);

      state.jobId = data.job_id;
      state.analysis = data.analysis;
      state.peaks = data.waveform || [];
      state.sampleRate = data.analysis.sample_rate;
      state.total = data.analysis.samples;
      state.spans = [];
      state.nextId = 1;

      factsFromAnalysis(data.analysis);
      notesFromAnalysis(data.analysis);
      originalAudio.src = data.original_url;
      resultPanel.hidden = false;
      editorPanel.hidden = false;
      resultBox.hidden = true;
      if (state.resultUrl) { URL.revokeObjectURL(state.resultUrl); state.resultUrl = null; }

      const editable = data.analysis.delay_confirmed;
      $("#select-suspects").disabled = !editable;
      $("#add-region").disabled = !editable;
      $("#clear-selection").disabled = !editable;
      drawTimeline();
      renderSpanList();
      setStatus(
        editable
          ? `分析完成 · ${data.analysis.regions.length} 段疑似反相`
          : "分析完成 · 时差未确认，仅可导出直接下混",
        editable ? "ok" : "err"
      );
    } catch (err) {
      setStatus(`错误：${err.message}`, "err");
    } finally {
      analyzeBtn.disabled = !state.file;
      resetBtn.disabled = false;
    }
  }

  async function render() {
    if (!state.jobId) return;
    const editable = state.analysis?.delay_confirmed;
    const spans = editable ? state.spans.map((s) => ({ start_sample: s.start, end_sample: s.end })) : [];
    const seq = ++state.renderSeq;
    setStatus("正在生成…", "busy");
    $("#render-btn").disabled = true;
    try {
      const resp = await fetch(`/api/jobs/${state.jobId}/render`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spans }),
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || `生成失败（HTTP ${resp.status}）`);
      }
      const blob = await resp.blob();
      if (seq !== state.renderSeq) return; // a newer render won
      const mode = resp.headers.get("X-Result-Mode") || "downmix";
      const total = resp.headers.get("X-Total-Samples");
      const merged = resp.headers.get("X-Merged-Ranges") || "";

      if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
      state.resultUrl = URL.createObjectURL(blob);
      resultAudio.src = state.resultUrl;
      downloadLink.href = state.resultUrl;
      downloadLink.download = mode === "repaired" ? "repaired-mono.wav" : "downmix-mono.wav";
      resultBox.hidden = false;

      const bytes = blob.size;
      const bits = `单声道 ${state.sampleRate} Hz · ${total} 采样 · ${(bytes / 1024).toFixed(1)} KB`;
      if (mode === "repaired") {
        resultInfo.textContent = `已修复 ${merged ? merged.split(",").length : 0} 个合并区间（${merged || "—"}）· ${bits}`;
        setStatus("修复完成 · 试听与下载为同一份 WAV", "ok");
      } else {
        resultInfo.textContent = `直接下混${state.analysis?.delay_confirmed ? "（未选择区间）" : "（时差未确认）"} · ${bits}`;
        setStatus(
          state.analysis?.delay_confirmed ? "已导出直接下混" : "时差未确认 · 仅输出直接下混",
          state.analysis?.delay_confirmed ? "ok" : "err"
        );
      }
      // Keep the preview and the download in lockstep: same object URL.
      resultAudio.load();
    } catch (err) {
      if (seq === state.renderSeq) setStatus(`错误：${err.message}`, "err");
    } finally {
      if (seq === state.renderSeq) $("#render-btn").disabled = false;
    }
  }

  // ---------- events ----------
  function pickFile(file) {
    if (!file) return;
    state.file = file;
    fileName.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
    analyzeBtn.disabled = false;
    setStatus("文件已选择，点击分析");
  }

  dropzone.addEventListener("click", (ev) => {
    if (ev.target !== fileInput) fileInput.click();
  });
  dropzone.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => pickFile(fileInput.files[0]));
  dropzone.addEventListener("dragover", (ev) => { ev.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    dropzone.classList.remove("dragover");
    pickFile(ev.dataTransfer.files[0]);
  });

  analyzeBtn.addEventListener("click", analyze);
  resetBtn.addEventListener("click", () => {
    state.file = null;
    state.jobId = null;
    state.analysis = null;
    state.peaks = [];
    state.spans = [];
    state.total = 0;
    state.renderSeq++;
    fileInput.value = "";
    fileName.textContent = "尚未选择文件";
    analyzeBtn.disabled = true;
    resetBtn.disabled = true;
    resultPanel.hidden = true;
    editorPanel.hidden = true;
    resultBox.hidden = true;
    originalAudio.removeAttribute("src");
    resultAudio.removeAttribute("src");
    if (state.resultUrl) { URL.revokeObjectURL(state.resultUrl); state.resultUrl = null; }
    setStatus("等待上传");
  });

  $("#select-suspects").addEventListener("click", () => {
    for (const r of state.analysis?.regions || []) {
      const exists = state.spans.some(
        (s) => Math.abs(s.start - r.start_sample) <= 1 && Math.abs(s.end - r.end_sample) <= 1
      );
      if (!exists) addSpan(r.start_sample, r.end_sample, "suspect");
    }
    syncFromSpans();
  });

  $("#clear-selection").addEventListener("click", () => {
    state.spans = [];
    syncFromSpans();
  });

  $("#add-region").addEventListener("click", () => {
    if (!state.total) return;
    const start = Math.min(state.total - MIN_SPAN, Math.round(state.total / 2));
    addSpan(start, start + MIN_SPAN, "manual");
  });

  $("#render-btn").addEventListener("click", render);

  window.addEventListener("beforeunload", () => {
    if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  });
})();
