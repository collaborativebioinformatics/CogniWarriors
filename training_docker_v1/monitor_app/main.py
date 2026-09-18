"""FLARE-only dashboard API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLARE_OUTPUT_DIR = Path(os.environ.get("FLARE_OUTPUT_DIR", PROJECT_ROOT / "flare_outputs")).resolve()
FLARE_DEFAULT_SITES = int(os.environ.get("FLARE_DEFAULT_SITES", "2"))
FLARE_DEFAULT_ROUNDS = int(os.environ.get("FLARE_DEFAULT_ROUNDS", "2"))
FLARE_DEFAULT_LOCAL_EPOCHS = int(os.environ.get("FLARE_DEFAULT_LOCAL_EPOCHS", "1"))

app = FastAPI(title="FLARE Federated Training Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

events: List[Dict[str, Any]] = []
flare_lock = threading.RLock()
flare_job: Dict[str, Any] = {
    "status": "idle",
    "running": False,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "parameters": {
        "n_sites": FLARE_DEFAULT_SITES,
        "rounds": FLARE_DEFAULT_ROUNDS,
        "local_epochs": FLARE_DEFAULT_LOCAL_EPOCHS,
        "split": "iid",
        "loss": "mse",
        "mu": 0.0,
    },
}


def event(message: str, level: str = "info"):
    events.insert(0, {"at": round(time.time(), 3), "level": level, "message": message})
    del events[120:]


def set_flare_job(**updates):
    with flare_lock:
        flare_job.update(updates)


def flare_job_snapshot() -> Dict[str, Any]:
    with flare_lock:
        return dict(flare_job)


def latest_summary() -> Dict[str, Any] | None:
    path = FLARE_OUTPUT_DIR / "latest_summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_flare_job(parameters: Dict[str, Any]):
    FLARE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_flare_job(
        status="running",
        running=True,
        started_at=round(time.time(), 3),
        finished_at=None,
        error=None,
        parameters=parameters,
    )
    event(
        "FLARE started: "
        f"{parameters['n_sites']} site(s), {parameters['rounds']} round(s), "
        f"{parameters['local_epochs']} local epoch(s), split={parameters['split']}, loss={parameters['loss']}"
    )
    command = [
        sys.executable,
        "-m",
        "federated.flare.run_flare_sim",
        "--n-sites",
        str(parameters["n_sites"]),
        "--rounds",
        str(parameters["rounds"]),
        "--local-epochs",
        str(parameters["local_epochs"]),
        "--split",
        parameters["split"],
        "--loss",
        parameters["loss"],
        "--mu",
        str(parameters["mu"]),
        "--output-dir",
        str(FLARE_OUTPUT_DIR),
    ]

    env = os.environ.copy()
    env.setdefault("NBBH_DATA_ROOT", str(PROJECT_ROOT / "data"))
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (FLARE_OUTPUT_DIR / "latest_run.log").write_text(completed.stdout or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"FLARE run failed with exit code {completed.returncode}")
        set_flare_job(status="completed", running=False, finished_at=round(time.time(), 3), error=None)
        event("FLARE completed")
    except Exception as exc:
        set_flare_job(status="failed", running=False, finished_at=round(time.time(), 3), error=str(exc))
        event(f"FLARE failed: {exc}", level="error")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/flare/status")
def flare_status():
    return {
        "job": flare_job_snapshot(),
        "summary": latest_summary(),
        "events": events,
        "output_dir": str(FLARE_OUTPUT_DIR),
        "defaults": {
            "n_sites": FLARE_DEFAULT_SITES,
            "rounds": FLARE_DEFAULT_ROUNDS,
            "local_epochs": FLARE_DEFAULT_LOCAL_EPOCHS,
        },
    }


@app.post("/api/flare/run")
async def start_flare(request: Request):
    payload = await request.json()
    parameters = {
        "n_sites": int(payload.get("n_sites") or FLARE_DEFAULT_SITES),
        "rounds": int(payload.get("rounds") or FLARE_DEFAULT_ROUNDS),
        "local_epochs": int(payload.get("local_epochs") or FLARE_DEFAULT_LOCAL_EPOCHS),
        "split": payload.get("split") or "iid",
        "loss": payload.get("loss") or "mse",
        "mu": float(payload.get("mu") or 0.0),
    }
    if parameters["n_sites"] < 1:
        raise HTTPException(status_code=400, detail="n_sites must be at least 1")
    if parameters["rounds"] < 1:
        raise HTTPException(status_code=400, detail="rounds must be at least 1")
    if parameters["local_epochs"] < 1:
        raise HTTPException(status_code=400, detail="local_epochs must be at least 1")
    if parameters["split"] not in {"iid", "age"}:
        raise HTTPException(status_code=400, detail="split must be iid or age")
    if parameters["loss"] not in {"mse", "lmmnn"}:
        raise HTTPException(status_code=400, detail="loss must be mse or lmmnn")

    with flare_lock:
        if flare_job.get("running"):
            raise HTTPException(status_code=409, detail="FLARE is already running")

    thread = threading.Thread(target=run_flare_job, args=(parameters,), name="dashboard-flare-run", daemon=True)
    thread.start()
    return {"status": "started", "parameters": parameters}
