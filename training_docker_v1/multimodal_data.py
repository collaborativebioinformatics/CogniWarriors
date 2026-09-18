"""Joins phenotype features + EF targets + precomputed image embeddings into
one per-session table, and provides the Dataset/Sampler pair needed to train
with subject-grouped minibatches (required by the LMMNN loss in
lmmnn_loss.py -- it needs a subject's repeated rows to co-occur in a batch).
"""

import ast
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

import config


def _load_image_embeddings(data_root=None, embedding_files=None):
    data_root = Path(data_root) if data_root is not None else config.DATA_ROOT
    embedding_files = embedding_files or config.IMAGE_EMBEDDING_FILES
    embedding_dir = data_root / "images" / "embeddings"
    frames = []
    for filename in embedding_files:
        df = pd.read_csv(embedding_dir / filename)
        frames.append(df)
    images = pd.concat(frames, ignore_index=True)
    images = images.rename(columns={"subject": "participant_id", "session": "session_id"})

    vectors = np.stack(images["vector"].apply(ast.literal_eval).to_numpy())
    if vectors.shape[1] != config.IMAGE_EMBEDDING_DIM:
        raise ValueError(
            f"expected {config.IMAGE_EMBEDDING_DIM}-dim image vectors, got {vectors.shape[1]}"
        )
    image_cols = [f"img_{i}" for i in range(config.IMAGE_EMBEDDING_DIM)]
    images = pd.concat(
        [images[config.ID_COLS], pd.DataFrame(vectors, columns=image_cols)], axis=1
    )
    return images, image_cols


def load_multimodal_dataframe(data_root=None, target_columns=None):
    """Inner-joins phenotype features, EF targets, and image embeddings on
    (participant_id, session_id). Drops rows missing any target column.

    Returns (df, phenotype_cols, covariate_cols, image_cols).
    """
    data_root = Path(data_root) if data_root is not None else config.DATA_ROOT
    target_columns = target_columns or config.TARGET_COLUMNS

    phenotype = pd.read_csv(data_root / "processed" / "phenotype_features.tsv", sep="\t")
    with open(data_root / "processed" / "phenotype_feature_manifest.json") as f:
        manifest = json.load(f)
    phenotype_cols = manifest["embedder_input_columns"]
    covariate_cols = manifest["covariate_columns"]

    composite = pd.read_csv(data_root / "processed" / "ef_composite.tsv", sep="\t")
    images, image_cols = _load_image_embeddings(data_root=data_root)

    df = phenotype.merge(
        composite[config.ID_COLS + target_columns], on=config.ID_COLS, how="inner"
    )
    df = df.merge(images, on=config.ID_COLS, how="inner")
    # Drop only rows with EVERY target missing -- partial missingness across
    # target columns is expected (e.g. per-task z-scores) and is handled
    # per-target by NaN masking in LMMNNLoss/masked_mse instead.
    df = df.dropna(subset=target_columns, how="all").reset_index(drop=True)

    return df, phenotype_cols, covariate_cols, image_cols


class MultimodalDataset(Dataset):
    """One row per session. __getitem__ returns
    (img_vec, pheno_vec, covariates, targets, subject_id)."""

    def __init__(self, df, phenotype_cols, covariate_cols, image_cols, target_cols):
        self.subject_ids = df["participant_id"].to_numpy()
        self.image = df[image_cols].to_numpy(dtype=np.float32)
        self.phenotype = df[phenotype_cols].to_numpy(dtype=np.float32)
        self.covariates = df[covariate_cols].to_numpy(dtype=np.float32)
        self.targets = df[target_cols].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.image[idx]),
            torch.from_numpy(self.phenotype[idx]),
            torch.from_numpy(self.covariates[idx]),
            torch.from_numpy(self.targets[idx]),
            self.subject_ids[idx],
        )


class SubjectGroupedBatchSampler(Sampler):
    """Greedily packs each subject's row-indices into the same batch, so a
    subject's repeated sessions always co-occur (needed to build the LMMNN
    per-subject covariance block at all). Batch size is a soft cap: a single
    subject with more rows than batch_size still gets its own, larger batch
    rather than being split.
    """

    def __init__(self, subject_ids, batch_size, shuffle=True, seed=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.groups = {}
        for idx, subject in enumerate(subject_ids):
            self.groups.setdefault(subject, []).append(idx)
        self.subjects = list(self.groups.keys())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        subjects = list(self.subjects)
        if self.shuffle:
            rng.shuffle(subjects)

        batches = []
        current = []
        for subject in subjects:
            if current and len(current) + len(self.groups[subject]) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(self.groups[subject])
        if current:
            batches.append(current)

        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        total = sum(len(v) for v in self.groups.values())
        return max(1, round(total / self.batch_size))
