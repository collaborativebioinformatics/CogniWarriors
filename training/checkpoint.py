"""I/O boundary for the fusion model: save/load a trained model, and run
inference from raw (unscaled) feature values. A checkpoint bundles
everything needed to go from raw inputs to a prediction without re-running
any training code: model weights, the exact architecture kwargs used, the
fitted preprocessing scalers, and the column lists/order training used --
so a checkpoint is self-describing and doesn't depend on config.py still
matching what it was trained with.
"""

from pathlib import Path

import torch

from fusion_model import FusionRegressor


def save_checkpoint(path, model, scalers, phenotype_cols, covariate_cols, image_cols, target_cols, model_kwargs):
    """scalers: dict with keys 'image', 'phenotype', 'covariates', each a
    fitted sklearn StandardScaler. model_kwargs: the kwargs FusionRegressor
    needs to be reconstructed identically (image_input_dim,
    phenotype_input_dim, n_covariates, n_targets, plus config.FUSION_ARCHITECTURE)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "scalers": scalers,
            "phenotype_cols": phenotype_cols,
            "covariate_cols": covariate_cols,
            "image_cols": image_cols,
            "target_cols": target_cols,
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Returns (model, checkpoint_dict). model is in eval mode on `device`.
    checkpoint_dict carries the scalers and column lists needed for predict()."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = FusionRegressor(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict(model, checkpoint, image_features, phenotype_features, covariates):
    """Raw (unscaled) inputs in, predictions out -- the actual inference
    endpoint. image_features/phenotype_features/covariates: 2D arrays
    (n_rows, n_cols), columns in the same order as
    checkpoint['image_cols']/['phenotype_cols']/['covariate_cols'].

    Returns a (n_rows, n_targets) numpy array in the target's natural
    units -- targets are never scaled during training, so no inverse
    transform is needed. Fixed-effect predictions only (f_NN(x)); no BLUP
    subject-specific correction (see handoff doc's inference section).
    """
    scalers = checkpoint["scalers"]
    image_x = scalers["image"].transform(image_features)
    pheno_x = scalers["phenotype"].transform(phenotype_features)
    cov_x = scalers["covariates"].transform(covariates)

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        preds = model(
            torch.tensor(image_x, dtype=torch.float32, device=device),
            torch.tensor(pheno_x, dtype=torch.float32, device=device),
            torch.tensor(cov_x, dtype=torch.float32, device=device),
        )
    return preds.cpu().numpy()


def predict_from_checkpoint_path(path, image_features, phenotype_features, covariates, device="cpu"):
    """Convenience one-shot version of predict() for callers that don't
    already hold a loaded model (e.g. a future serving/FLARE inference
    wrapper) -- loads the checkpoint fresh each call."""
    model, checkpoint = load_checkpoint(path, device=device)
    return predict(model, checkpoint, image_features, phenotype_features, covariates)
