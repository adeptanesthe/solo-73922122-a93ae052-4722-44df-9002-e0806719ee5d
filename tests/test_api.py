import io
import re
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import dsp
from app.main import app

RATE = 16_000


def make_wav(left, right, rate=RATE):
    n = min(len(left), len(right))
    pcm = np.stack([left[:n], right[:n]], axis=1)
    pcm = np.rint(np.clip(pcm, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def source(seconds=4.0, seed=5):
    rng = np.random.default_rng(seed)
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    x = rng.standard_normal(n) * 0.12
    return (x * (0.55 + 0.45 * np.sin(2 * np.pi * 1.7 * t) ** 2)).astype(np.float32)


def voiced(seconds=4.0, seed=11, f0=200.0, noise=0.05):
    """Periodic pseudo-voice: in-place cancellation is real at 1 ms delay."""
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    rng = np.random.default_rng(seed)
    x = (
        0.85 * np.sin(2 * np.pi * f0 * t)
        + 0.22 * np.sin(2 * np.pi * 2 * f0 * t + 0.4)
        + 0.08 * np.sin(2 * np.pi * 3 * f0 * t + 1.1)
    )
    x = x + noise * rng.standard_normal(n)
    x = x * (0.72 + 0.28 * np.sin(2 * np.pi * 2.1 * t) ** 2)
    return (x / np.max(np.abs(x)) * 0.55).astype(np.float32)


def scenario(d=8, invert=True, seconds=4.0):
    x = voiced(seconds)
    r = np.zeros_like(x)
    r[d:] = x[: len(x) - d]
    if invert:
        r[20000:40000] *= -1
    return make_wav(x, r)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def decode(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == RATE
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
        return wf.getnframes(), pcm


def analyze(client, data, name="x.wav"):
    resp = client.post("/api/analyze", files={"file": (name, data, "audio/wav")})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health(client):
    assert client.get("/api/health").json()["ok"] is True


def test_index_and_static(client):
    html = client.get("/")
    assert html.status_code == 200
    assert "反相修复台" in html.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_analyze_returns_metadata_and_waveform(client):
    body = analyze(client, scenario())
    a = body["analysis"]
    assert a["delay_confirmed"] is True
    assert a["delay_samples"] == 8
    assert a["samples"] == RATE * 4
    assert len(a["regions"]) == 1
    assert 19000 <= a["regions"][0]["start_sample"] <= 20320
    assert a["regions"][0]["start_sample"] % 160 == 0
    assert a["regions"][0]["end_sample"] % 160 == 0
    assert len(body["waveform"]) > 100
    assert body["downmix_url"].endswith("/downmix.wav")
    assert body["original_url"].endswith("/original.wav")


def test_rejects_bad_upload(client):
    resp = client.post("/api/analyze", files={"file": ("x.wav", b"junk", "audio/wav")})
    assert resp.status_code == 422
    mono = make_wav(source(), source())[:0]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(np.zeros(RATE, dtype="<i2").tobytes())
    resp = client.post("/api/analyze", files={"file": ("m.wav", buf.getvalue(), "audio/wav")})
    assert resp.status_code == 422
    assert "双声道" in resp.json()["detail"]


def test_over_60s_rejected(client):
    n = RATE * 61
    data = make_wav(np.zeros(n), np.zeros(n))
    resp = client.post("/api/analyze", files={"file": ("long.wav", data, "audio/wav")})
    assert resp.status_code == 422
    assert "60" in resp.json()["detail"]


def test_render_repairs_and_preserves_length(client):
    body = analyze(client, scenario())
    job = body["job_id"]
    n = body["analysis"]["samples"]
    spans = body["analysis"]["regions"]
    payload = {"spans": [{"start_sample": s["start_sample"], "end_sample": s["end_sample"]} for s in spans]}
    resp = client.post(f"/api/jobs/{job}/render", json=payload)
    assert resp.status_code == 200
    assert resp.headers["X-Result-Mode"] == "repaired"
    frames, pcm = decode(resp.content)
    assert frames == n

    # outside the selected spans must equal the direct downmix sample-for-sample
    dm = client.get(body["downmix_url"])
    assert dm.status_code == 200
    dm_frames, dm_pcm = decode(dm.content)
    assert dm_frames == n
    mask = np.ones(n, dtype=bool)
    for s in spans:
        # spans are half-open [start_sample, end_sample)
        mask[s["start_sample"] : s["end_sample"]] = False
    assert mask.any()
    assert np.array_equal(pcm[mask], dm_pcm[mask])


def test_select_all_regions_leaves_every_unselected_unit_bit_exact(client):
    """Contract: after 'select all suspects', every unit not in a region equals
    the direct downmix sample-for-sample, and regions cover only whole units."""
    x = voiced(5.0)
    d = 8
    r = np.zeros_like(x)
    r[d:] = x[: len(x) - d]
    r[RATE : 2 * RATE] *= -1
    left = x.copy()
    left[int(RATE * 1.5) : int(RATE * 1.5) + 160] = 0.0  # one-sided unit in the middle
    body = analyze(client, make_wav(left, r))
    a = body["analysis"]
    assert a["delay_confirmed"] is True
    regions = a["regions"]
    assert len(regions) == 2, regions
    for reg in regions:
        assert reg["start_sample"] % 160 == 0
        assert reg["end_sample"] % 160 == 0

    # the one-sided unit must not be inside any region
    a0, a1 = int(RATE * 1.5), int(RATE * 1.5) + 160
    assert all(not (reg["start_sample"] < a1 and a0 < reg["end_sample"]) for reg in regions)

    job = body["job_id"]
    resp = client.post(
        f"/api/jobs/{job}/render",
        json={"spans": [{"start_sample": g["start_sample"], "end_sample": g["end_sample"]} for g in regions]},
    )
    assert resp.headers["X-Result-Mode"] == "repaired"
    frames, pcm = decode(resp.content)
    assert frames == len(x)
    _, dm_pcm = decode(client.get(body["downmix_url"]).content)

    unit = 160
    n_units = len(x) // unit
    inside = np.zeros(n_units, dtype=bool)
    for reg in regions:
        inside[reg["start_sample"] // unit : reg["end_sample"] // unit] = True
    assert not inside[a0 // unit]  # the one-sided unit is excluded
    for u in np.flatnonzero(~inside):
        s = u * unit
        assert np.array_equal(pcm[s : s + unit], dm_pcm[s : s + unit]), f"unit {s} changed"
    # tail samples after the last whole unit are untouched too
    tail = n_units * unit
    assert np.array_equal(pcm[tail:], dm_pcm[tail:])


def test_render_without_spans_is_downmix(client):
    body = analyze(client, scenario())
    job = body["job_id"]
    resp = client.post(f"/api/jobs/{job}/render", json={"spans": []})
    assert resp.status_code == 200
    assert resp.headers["X-Result-Mode"] == "downmix"
    dm = client.get(body["downmix_url"])
    assert resp.content == dm.content


def test_unconfirmed_delay_only_offers_downmix(client):
    x = source()
    y = source(seed=2)
    body = analyze(client, make_wav(x, y))
    assert body["analysis"]["delay_confirmed"] is False
    job = body["job_id"]
    resp = client.post(
        f"/api/jobs/{job}/render",
        json={"spans": [{"start_sample": 0, "end_sample": 1000}]},
    )
    assert resp.status_code == 200
    assert resp.headers["X-Result-Mode"] == "downmix"
    assert resp.content == client.get(body["downmix_url"]).content


def test_render_merges_spans(client):
    body = analyze(client, scenario())
    job = body["job_id"]
    resp = client.post(
        f"/api/jobs/{job}/render",
        json={
            "spans": [
                {"start_sample": 20000, "end_sample": 30000},
                {"start_sample": 30000, "end_sample": 40000},
            ]
        },
    )
    assert resp.headers["X-Merged-Ranges"] == "20000-40000"


def test_render_unknown_job(client):
    resp = client.post("/api/jobs/nope/render", json={"spans": []})
    assert resp.status_code == 404


def test_render_clamps_out_of_bounds(client):
    body = analyze(client, scenario())
    job = body["job_id"]
    n = body["analysis"]["samples"]
    resp = client.post(
        f"/api/jobs/{job}/render",
        json={"spans": [{"start_sample": 0, "end_sample": n + 5000}]},
    )
    assert resp.status_code == 200
    frames, _ = decode(resp.content)
    assert frames == n


def test_original_is_byte_identical(client):
    data = scenario()
    body = analyze(client, data)
    resp = client.get(body["original_url"])
    assert resp.content == data


def test_preview_and_download_are_same_blob(client):
    """The browser plays and downloads the same object URL; the API guarantees
    deterministic bytes for identical input."""
    body = analyze(client, scenario())
    job = body["job_id"]
    payload = {"spans": [{"start_sample": 20000, "end_sample": 40000}]}
    first = client.post(f"/api/jobs/{job}/render", json=payload).content
    second = client.post(f"/api/jobs/{job}/render", json=payload).content
    assert first == second


def test_render_is_16bit_mono_at_input_rate(client):
    body = analyze(client, scenario())
    resp = client.post(
        f"/api/jobs/{body['job_id']}/render",
        json={"spans": [{"start_sample": 20000, "end_sample": 40000}]},
    )
    frames, pcm = decode(resp.content)
    assert pcm.dtype == np.int16
    assert resp.headers["X-Total-Samples"] == str(frames)
