"""LMMNN-style random-effects loss (Simchoni & Rosset, NeurIPS 2021).

Replaces MSE with the negative log-likelihood of a Gaussian whose
covariance has a per-subject random-intercept term plus i.i.d. error:

    V = sigma2_subject * Z @ Z.T + sigma2_error * I

Everything (network weights + the two variance scalars per target) is
optimized jointly by backprop/Adam -- no EM, no alternating closed-form
steps.

Because minibatches are subject-grouped (see multimodal_data.py), Z is
block-diagonal with one all-ones column per subject, so V is block-diagonal
too. Within a subject's block of size n, Z@Z.T is the all-ones matrix J_n,
so V_g = sigma2_error * I_n + sigma2_subject * J_n is exactly a
"compound symmetry" matrix. Its inverse and log-determinant have a closed
form (Sherman-Morrison), used here instead of a dense per-block inversion:

    V_g^-1 r = (1/sigma2_error) * r
               - (sigma2_subject / (sigma2_error * (sigma2_error + n*sigma2_subject))) * sum(r) * 1
    r^T V_g^-1 r = sum(r^2)/sigma2_error
                   - (sigma2_subject / (sigma2_error*(sigma2_error + n*sigma2_subject))) * sum(r)^2
    log|V_g| = (n-1)*log(sigma2_error) + log(sigma2_error + n*sigma2_subject)

This is computed for every subject group in a batch via scatter-add (no
per-group Python loop, no matrix inversion).
"""

import math

import numpy as np
import torch
import torch.nn as nn


class LMMNNLoss(nn.Module):
    def __init__(self, n_targets):
        super().__init__()
        self.n_targets = n_targets
        # log-space so exp() keeps variances positive without constrained
        # optimization; initialized to log(1)=0.
        self.log_sigma2_subject = nn.Parameter(torch.zeros(n_targets))
        self.log_sigma2_error = nn.Parameter(torch.zeros(n_targets))

    def forward(self, preds, targets, subject_ids):
        """preds, targets: (batch, n_targets). subject_ids: numpy array of
        length batch (or anything np.unique accepts). Returns
        (total_loss, per_target_info)."""
        subject_ids = np.asarray(subject_ids)
        total_loss = preds.new_zeros(())
        per_target = []

        for t in range(self.n_targets):
            y_t = targets[:, t]
            mask = ~torch.isnan(y_t)
            if mask.sum() == 0:
                per_target.append(None)
                continue

            r = (y_t - preds[:, t])[mask]
            mask_np = mask.detach().cpu().numpy()
            _, group_idx_np = np.unique(subject_ids[mask_np], return_inverse=True)
            group_idx = torch.from_numpy(group_idx_np).long().to(r.device)
            n_groups = int(group_idx_np.max()) + 1

            sum_r = torch.zeros(n_groups, dtype=r.dtype, device=r.device).scatter_add_(0, group_idx, r)
            sum_r2 = torch.zeros(n_groups, dtype=r.dtype, device=r.device).scatter_add_(0, group_idx, r ** 2)
            counts = torch.zeros(n_groups, dtype=r.dtype, device=r.device).scatter_add_(
                0, group_idx, torch.ones_like(r)
            )

            sigma2_subject = torch.exp(self.log_sigma2_subject[t])
            sigma2_error = torch.exp(self.log_sigma2_error[t])

            denom = sigma2_error + counts * sigma2_subject
            quad = sum_r2 / sigma2_error - (sigma2_subject / (sigma2_error * denom)) * sum_r ** 2
            logdet = (counts - 1) * torch.log(sigma2_error) + torch.log(denom)

            loss_t = 0.5 * torch.sum(quad + logdet + counts * math.log(2 * math.pi))
            total_loss = total_loss + loss_t

            per_target.append(
                {
                    "loss": loss_t.item(),
                    "sigma2_subject": sigma2_subject.item(),
                    "sigma2_error": sigma2_error.item(),
                    "n_groups": n_groups,
                }
            )

        return total_loss, per_target
