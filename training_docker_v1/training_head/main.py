"""FastAPI entry point for one federated Training Head."""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, HTTPException

from .schemas import EvaluateRequest, InitializeRequest, TrainRequest
from .serialization import WEIGHTS_FORMAT
from .settings import TrainingHeadSettings
from .trainer import DuplicateTrainingRequest, TrainingHeadError, TrainingHeadService, WeightsUnavailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

settings = TrainingHeadSettings.from_env()
service = TrainingHeadService(settings)
app = FastAPI(title=f"Federated Training Head - {settings.center_id}")


def _raise_http(exc: Exception):
    if isinstance(exc, DuplicateTrainingRequest):
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if isinstance(exc, WeightsUnavailable):
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, TrainingHeadError):
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    return service.health()


@app.post("/initialize")
def initialize(request: InitializeRequest):
    if request.weights_format != WEIGHTS_FORMAT:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported weights_format '{request.weights_format}'; expected '{WEIGHTS_FORMAT}'",
        )
    try:
        return service.initialize(request.round, request.model_version, request.weights)
    except Exception as exc:
        _raise_http(exc)


@app.post("/train")
def train(request: TrainRequest):
    try:
        return service.start_training(
            fed_round=request.round,
            model_version=request.model_version,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
        )
    except Exception as exc:
        _raise_http(exc)


@app.get("/status")
def status():
    return service.snapshot()


@app.get("/weights")
def weights():
    try:
        return service.weights()
    except Exception as exc:
        _raise_http(exc)


@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    try:
        return service.evaluate(split=request.split)
    except Exception as exc:
        _raise_http(exc)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)

