"""Local training orchestration for a federated Training Head."""

from __future__ import annotations

import itertools
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import config as model_config
from checkpoint import save_checkpoint
from fusion_model import FusionRegressor
from lmmnn_loss import LMMNNLoss
from multimodal_data import MultimodalDataset, SubjectGroupedBatchSampler, load_multimodal_dataframe

from .serialization import WEIGHTS_FORMAT, decode_state_dict, encode_state_dict
from .settings import TrainingHeadSettings

LOGGER = logging.getLogger(__name__)


class TrainingHeadError(RuntimeError):
    status_code = 400


class DuplicateTrainingRequest(TrainingHeadError):
    status_code = 409


class WeightsUnavailable(TrainingHeadError):
    status_code = 409


@dataclass
class TrainingStatus:
    center_id: str
    round: int = 0
    model_version: str = ""
    status: str = "idle"
    epoch: int = 0
    total_epochs: int = 0
    train_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    num_samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    updated_at: float = field(default_factory=time.time)

    def snapshot(self) -> Dict[str, object]:
        data = asdict(self)
        data["updated_at"] = round(self.updated_at, 3)
        return data


def _scale_columns(train_df, other_dfs, columns):
    scaler = StandardScaler().fit(train_df[columns].to_numpy())
    train_df = train_df.copy()
    train_df[columns] = scaler.transform(train_df[columns].to_numpy())
    scaled_others = []
    for df in other_dfs:
        df = df.copy()
        df[columns] = scaler.transform(df[columns].to_numpy())
        scaled_others.append(df)
    return train_df, scaled_others, scaler


def _masked_mse(preds, targets) -> torch.Tensor:
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return preds.new_tensor(float("nan"))
    diff = (preds - targets)[mask]
    return (diff ** 2).mean()


class TrainingHeadService:
    def __init__(self, settings: TrainingHeadSettings):
        self.settings = settings
        self.lock = threading.RLock()
        self.status = TrainingStatus(center_id=settings.center_id)
        self.model: Optional[FusionRegressor] = None
        self.loss_fn: Optional[LMMNNLoss] = None
        self.train_ds: Optional[MultimodalDataset] = None
        self.val_ds: Optional[MultimodalDataset] = None
        self.scalers = None
        self.phenotype_cols = None
        self.covariate_cols = None
        self.image_cols = None
        self.target_cols = model_config.TARGET_COLUMNS
        self.model_kwargs = None
        self.training_thread: Optional[threading.Thread] = None
        self.current_round = 0
        self.model_version = ""

        if settings.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(settings.device)

    def health(self) -> Dict[str, object]:
        return {
            "status": "healthy",
            "center_id": self.settings.center_id,
            "data_dir": str(self.settings.data_dir),
            "device": str(self.device),
        }

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return self.status.snapshot()

    def initialize(self, fed_round: int, model_version: str, weights: Optional[str]) -> Dict[str, object]:
        self._validate_round(fed_round)
        with self.lock:
            if self.status.status == "training":
                raise DuplicateTrainingRequest("cannot initialize while local training is running")
            self._set_status(status="initializing", round=fed_round, model_version=model_version, error=None)

        try:
            self._prepare()
            state_dict = decode_state_dict(weights)
            if state_dict is not None:
                assert self.model is not None
                self.model.load_state_dict(state_dict, strict=True)
                LOGGER.info("[%s] Received global model for round %s", self.settings.center_id, fed_round)
            else:
                LOGGER.info("[%s] Initialized local model for round %s", self.settings.center_id, fed_round)

            with self.lock:
                self.current_round = fed_round
                self.model_version = model_version
                self._set_status(status="idle", round=fed_round, model_version=model_version, epoch=0)
            return self.snapshot()
        except Exception as exc:
            with self.lock:
                self._set_status(status="failed", error=str(exc))
            raise

    def start_training(
        self,
        fed_round: Optional[int] = None,
        model_version: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, object]:
        fed_round = self.current_round if fed_round is None else fed_round
        model_version = model_version or self.model_version or f"local_round_{fed_round}"
        epochs = epochs or self.settings.epochs
        batch_size = batch_size or self.settings.batch_size
        learning_rate = learning_rate or self.settings.learning_rate
        self._validate_round(fed_round)

        with self.lock:
            if self.status.status in {"initializing", "training"}:
                raise DuplicateTrainingRequest("training is already running")
            self._set_status(status="initializing", round=fed_round, model_version=model_version, error=None)

        try:
            self._prepare()
        except Exception as exc:
            with self.lock:
                self._set_status(status="failed", error=str(exc))
            raise

        with self.lock:
            self._set_status(
                status="training",
                round=fed_round,
                model_version=model_version,
                epoch=0,
                total_epochs=epochs,
                train_loss=None,
                validation_loss=None,
                error=None,
                metrics={},
            )
            self.current_round = fed_round
            self.model_version = model_version

        LOGGER.info("[%s] Round %s training started", self.settings.center_id, fed_round)
        thread = threading.Thread(
            target=self._train_worker,
            args=(fed_round, model_version, epochs, batch_size, learning_rate),
            name=f"{self.settings.center_id}-round-{fed_round}",
            daemon=True,
        )
        self.training_thread = thread
        thread.start()

        return {
            "status": "training_started",
            "center_id": self.settings.center_id,
            "round": fed_round,
        }

    def weights(self) -> Dict[str, object]:
        with self.lock:
            if self.status.status != "completed":
                raise WeightsUnavailable("local weights are only available after training completes")
            assert self.model is not None
            weights = encode_state_dict(self.model.state_dict())
            metrics = dict(self.status.metrics)
            return {
                "center_id": self.settings.center_id,
                "round": self.status.round,
                "num_samples": self.status.num_samples,
                "weights": weights,
                "weights_format": WEIGHTS_FORMAT,
                "metrics": metrics,
                "model_version": self.status.model_version,
            }

    def evaluate(self, split: str = "validation") -> Dict[str, object]:
        self._prepare()
        with self.lock:
            if self.status.status == "training":
                raise DuplicateTrainingRequest("cannot evaluate while training is running")
            dataset = self.train_ds if split == "train" else self.val_ds
            round_id = self.status.round
        metrics = self._evaluate_dataset(dataset)
        return {"center_id": self.settings.center_id, "round": round_id, "metrics": metrics}

    def _prepare(self):
        if self.model is not None:
            return
        if not self.settings.data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {self.settings.data_dir}")

        df, phenotype_cols, covariate_cols, image_cols = load_multimodal_dataframe(
            data_root=self.settings.data_dir,
            target_columns=self.target_cols,
        )
        if df.empty:
            raise TrainingHeadError(f"no trainable rows found in {self.settings.data_dir}")

        train_df, val_df = self._split_local_data(df)
        train_df, (val_df,), phenotype_scaler = _scale_columns(train_df, [val_df], phenotype_cols)
        train_df, (val_df,), image_scaler = _scale_columns(train_df, [val_df], image_cols)
        train_df, (val_df,), covariate_scaler = _scale_columns(train_df, [val_df], covariate_cols)

        self.train_ds = MultimodalDataset(train_df, phenotype_cols, covariate_cols, image_cols, self.target_cols)
        self.val_ds = MultimodalDataset(val_df, phenotype_cols, covariate_cols, image_cols, self.target_cols)
        self.scalers = {
            "phenotype": phenotype_scaler,
            "image": image_scaler,
            "covariates": covariate_scaler,
        }
        self.phenotype_cols = phenotype_cols
        self.covariate_cols = covariate_cols
        self.image_cols = image_cols
        self.model_kwargs = dict(
            image_input_dim=len(image_cols),
            phenotype_input_dim=len(phenotype_cols),
            n_covariates=len(covariate_cols),
            n_targets=len(self.target_cols),
            **model_config.FUSION_ARCHITECTURE,
        )
        self.model = FusionRegressor(**self.model_kwargs).to(self.device)
        self.loss_fn = LMMNNLoss(n_targets=len(self.target_cols)).to(self.device)

        with self.lock:
            self.status.num_samples = len(self.train_ds)
            self.status.updated_at = time.time()

        LOGGER.info(
            "[%s] Loaded %s train rows and %s validation rows from %s",
            self.settings.center_id,
            len(self.train_ds),
            len(self.val_ds),
            self.settings.data_dir,
        )

    def _split_local_data(self, df):
        groups = df["participant_id"].to_numpy()
        if df["participant_id"].nunique() < 2 or len(df) < 3:
            return df.reset_index(drop=True), df.reset_index(drop=True)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.settings.validation_fraction,
            random_state=self.settings.seed,
        )
        train_idx, val_idx = next(splitter.split(df, groups=groups))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

    def _train_worker(self, fed_round: int, model_version: str, epochs: int, batch_size: int, learning_rate: float):
        try:
            assert self.model is not None
            assert self.loss_fn is not None
            assert self.train_ds is not None

            optimizer = torch.optim.Adam(
                itertools.chain(self.model.parameters(), self.loss_fn.parameters()),
                lr=learning_rate,
                weight_decay=self.settings.weight_decay,
            )
            sampler = SubjectGroupedBatchSampler(
                self.train_ds.subject_ids,
                batch_size=batch_size,
                seed=self.settings.seed,
            )
            loader = DataLoader(self.train_ds, batch_sampler=sampler)

            for epoch in range(1, epochs + 1):
                self.model.train()
                losses = []
                for image_x, pheno_x, covariates, targets, subject_ids in loader:
                    image_x = image_x.to(self.device)
                    pheno_x = pheno_x.to(self.device)
                    covariates = covariates.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    preds = self.model(image_x, pheno_x, covariates)
                    loss, per_target = self.loss_fn(preds, targets, subject_ids)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite training loss at epoch {epoch}: {loss.item()}")
                    for item in per_target:
                        if item is None:
                            continue
                        if not (math.isfinite(item["sigma2_subject"]) and math.isfinite(item["sigma2_error"])):
                            raise RuntimeError(f"non-finite variance scalar at epoch {epoch}: {item}")

                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item() / len(subject_ids))

                train_loss = float(np.mean(losses))
                val_metrics = self._evaluate_dataset(self.val_ds)
                with self.lock:
                    self._set_status(
                        status="training",
                        epoch=epoch,
                        total_epochs=epochs,
                        train_loss=train_loss,
                        validation_loss=val_metrics["loss"],
                        metrics={"train_loss": train_loss, **val_metrics},
                    )
                LOGGER.info(
                    "[%s] Round %s epoch %s/%s loss=%.6f",
                    self.settings.center_id,
                    fed_round,
                    epoch,
                    epochs,
                    train_loss,
                )

            checkpoint_path = self._save_local_checkpoint(fed_round, model_version)
            final_metrics = self._evaluate_dataset(self.val_ds)
            with self.lock:
                self._set_status(
                    status="completed",
                    epoch=epochs,
                    total_epochs=epochs,
                    validation_loss=final_metrics["loss"],
                    metrics={"train_loss": train_loss, **final_metrics},
                    checkpoint_path=str(checkpoint_path),
                    num_samples=len(self.train_ds),
                )
            LOGGER.info("[%s] Round %s training completed", self.settings.center_id, fed_round)
        except Exception as exc:
            LOGGER.exception("[%s] Round %s training failed", self.settings.center_id, fed_round)
            with self.lock:
                self._set_status(status="failed", error=str(exc))

    def _evaluate_dataset(self, dataset: Optional[MultimodalDataset]) -> Dict[str, float]:
        if dataset is None:
            raise TrainingHeadError("dataset has not been prepared")
        assert self.model is not None

        image_x = torch.from_numpy(dataset.image).to(self.device)
        pheno_x = torch.from_numpy(dataset.phenotype).to(self.device)
        covariates = torch.from_numpy(dataset.covariates).to(self.device)
        targets = torch.from_numpy(dataset.targets).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(image_x, pheno_x, covariates)
            loss = _masked_mse(preds, targets)
            mae_mask = ~torch.isnan(targets)
            mae = torch.abs(preds - targets)[mae_mask].mean() if mae_mask.sum() else preds.new_tensor(float("nan"))

        return {
            "loss": float(loss.detach().cpu().item()),
            "mae": float(mae.detach().cpu().item()),
        }

    def _save_local_checkpoint(self, fed_round: int, model_version: str) -> Path:
        assert self.model is not None
        assert self.scalers is not None
        assert self.phenotype_cols is not None
        assert self.covariate_cols is not None
        assert self.image_cols is not None
        assert self.model_kwargs is not None

        self.settings.model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.settings.model_dir / f"{self.settings.center_id}_round_{fed_round}.pt"
        save_checkpoint(
            checkpoint_path,
            self.model,
            self.scalers,
            self.phenotype_cols,
            self.covariate_cols,
            self.image_cols,
            self.target_cols,
            self.model_kwargs,
        )
        latest_path = self.settings.model_dir / "latest.pt"
        save_checkpoint(
            latest_path,
            self.model,
            self.scalers,
            self.phenotype_cols,
            self.covariate_cols,
            self.image_cols,
            self.target_cols,
            self.model_kwargs,
        )
        LOGGER.info("[%s] Local weights saved to %s", self.settings.center_id, checkpoint_path)
        return checkpoint_path

    def _validate_round(self, fed_round: int):
        if fed_round < 0:
            raise TrainingHeadError("round must be non-negative")
        if self.current_round and fed_round < self.current_round:
            raise TrainingHeadError(
                f"round {fed_round} is older than current round {self.current_round}"
            )

    def _set_status(self, **updates):
        for key, value in updates.items():
            setattr(self.status, key, value)
        self.status.updated_at = time.time()
