"""Environment-driven settings for one Training Head instance."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class TrainingHeadSettings:
    center_id: str
    data_dir: Path
    model_dir: Path
    federation_head_url: Optional[str]
    host: str
    port: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_fraction: float
    seed: int
    device: str

    @classmethod
    def from_env(cls) -> "TrainingHeadSettings":
        center_id = os.environ.get("CENTER_ID", "center1")
        return cls(
            center_id=center_id,
            data_dir=Path(os.environ.get("DATA_DIR", f"data/centers/{center_id}")).resolve(),
            model_dir=Path(os.environ.get("MODEL_DIR", f"checkpoints/{center_id}")).resolve(),
            federation_head_url=os.environ.get("FEDERATION_HEAD_URL"),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=_env_int("PORT", 8001),
            batch_size=_env_int("BATCH_SIZE", 32),
            epochs=_env_int("EPOCHS", 10),
            learning_rate=_env_float("LEARNING_RATE", 1e-3),
            weight_decay=_env_float("WEIGHT_DECAY", 1e-3),
            validation_fraction=_env_float("VALIDATION_FRACTION", 0.2),
            seed=_env_int("SEED", 0),
            device=os.environ.get("DEVICE", "auto"),
        )

