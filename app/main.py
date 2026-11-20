"""FastAPI app: upload a stereo WAV, inspect anti-phase spans, render a repair.

All rendered output is a single mono WAV of exactly the input length; the
preview in the browser and the download link share the same rendered blob.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from . import dsp

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
JOB_TTL_SECONDS = 30 * 60
MAX_JOBS = 16

app = FastAPI(title="反相修复台", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
# NumPy work runs on a worker thread so a 60 s file never blocks the loop.
cpu_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dsp")


@dataclass
class Job:
    job_id: str
    pcm: np.ndarray
    analysis: dsp.Analysis
    original_bytes: bytes
    direct_wav: bytes
    created: float


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Job] = {}

    def put(self, job: Job) -> None:
        with self._lock:
            self._purge_locked()
            while len(self._items) >= MAX_JOBS:
                oldest = min(self._items.values(), key=lambda j: j.created)
                self._items.pop(oldest.job_id, None)
            self._items[job.job_id] = job

    def get(self, job_id: str) -> Job:
        with self._lock:
            self._purge_locked()
            job = self._items.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="会话已过期，请重新上传音频。")
        return job

    def _purge_locked(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        for key in [k for k, v in self._items.items() if v.created < cutoff]:
            self._items.pop(key, None)


jobs = JobStore()


class Span(BaseModel):
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=0)


class RenderRequest(BaseModel):
    spans: list[Span] = Field(default_factory=list, max_length=500)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/health")
def health():
    return {"ok": True, "version": "1.0"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件超过 16 MB 上限。")
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    loop = asyncio.get_running_loop()
    try:
        pcm, analysis, direct, direct_wav = await loop.run_in_executor(
            cpu_pool, _analyze_sync, data
        )
    except dsp.AudioError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job_id = secrets.token_urlsafe(12)
    jobs.put(
        Job(
            job_id=job_id,
            pcm=pcm,
            analysis=analysis,
            original_bytes=data,
            direct_wav=direct_wav,
            created=time.time(),
        )
    )
    return {
        "job_id": job_id,
        "analysis": analysis_json(analysis),
        "waveform": dsp.waveform_peaks(direct),
        "downmix_url": f"/api/jobs/{job_id}/downmix.wav",
        "original_url": f"/api/jobs/{job_id}/original.wav",
    }


def _analyze_sync(data: bytes):
    pcm, analysis = dsp.analyze(data)
    direct = 0.5 * (pcm[:, 0].astype(np.float64) + pcm[:, 1].astype(np.float64))
    return pcm, analysis, direct, dsp.encode_wav(direct, analysis.sample_rate)


def analysis_json(a: dsp.Analysis) -> dict:
    return {
        "channels": a.channels,
        "sample_rate": a.sample_rate,
        "samples": a.samples,
        "duration_s": a.duration_s,
        "delay_samples": a.delay_samples,
        "delay_ms": a.delay_ms,
        "delay_confirmed": a.delay_confirmed,
        "delay_confidence": a.delay_confidence,
        "delay_correlation": a.delay_correlation,
        "delay_reason": a.delay_reason,
        "regions": a.regions,
        "notes": a.notes,
        "downmix_rms_dbfs": a.downmix_rms_dbfs,
        "repairable": a.delay_confirmed,
    }


@app.get("/api/jobs/{job_id}/downmix.wav")
def downmix(job_id: str):
    job = jobs.get(job_id)
    return wav_response(job.direct_wav, f"downmix-{job_id}.wav", mode="downmix")


@app.get("/api/jobs/{job_id}/original.wav")
def original(job_id: str):
    """The validated upload, byte-for-byte, as an A/B reference."""
    job = jobs.get(job_id)
    return Response(
        content=job.original_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="original.wav"',
            "Cache-Control": "no-store",
            "X-Audio-Mode": "original",
        },
    )


@app.post("/api/jobs/{job_id}/render")
async def render(job_id: str, body: RenderRequest):
    job = jobs.get(job_id)
    a = job.analysis
    n = a.samples
    merged: list[tuple[int, int]] = []

    if not a.delay_confirmed or not body.spans:
        mode = "downmix"
        wav = job.direct_wav
    else:
        merged = dsp.merge_intervals(
            [(s.start_sample, s.end_sample) for s in body.spans], n
        )
        if not merged:
            mode = "downmix"
            wav = job.direct_wav
        else:
            loop = asyncio.get_running_loop()
            wav = await loop.run_in_executor(
                cpu_pool, _render_sync, job.pcm, merged, a.delay_samples, a.sample_rate
            )
            mode = "repaired"

    resp = wav_response(wav, f"repair-{job_id}.wav", mode=mode)
    resp.headers["X-Result-Mode"] = mode
    resp.headers["X-Total-Samples"] = str(n)
    resp.headers["X-Input-Spans"] = str(len(body.spans))
    resp.headers["X-Merged-Spans"] = str(len(merged))
    if merged:
        resp.headers["X-Merged-Ranges"] = ",".join(f"{s}-{e}" for s, e in merged)
    return resp


def _render_sync(pcm, merged, delay, sample_rate) -> bytes:
    mono = dsp.repair(pcm, merged, delay, sample_rate)
    return dsp.encode_wav(mono, sample_rate)


def wav_response(data: bytes, filename: str, *, mode: str) -> Response:
    return Response(
        content=data,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "no-store",
            "X-Audio-Mode": mode,
        },
    )


@app.exception_handler(dsp.AudioError)
async def audio_error_handler(request: Request, exc: dsp.AudioError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
