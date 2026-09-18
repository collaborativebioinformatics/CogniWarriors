"""Server-side workflow: NVFlare's FedAvg with one extra step in front.

Phase 0 (once): ask every site for column sums, sums of squares and row
counts of its training data, combine them into ONE global mean/std per
feature, and write those into the initial model's scaling buffers.

Then: NVFlare's standard FedAvg, unchanged (weighted by each site's
NUM_STEPS_CURRENT_ROUND = number of training sessions).
"""

import numpy as np

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.fedavg import FedAvg

BLOCKS = ("image", "phenotype", "covariates")


def _to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class GlobalScalingFedAvg(FedAvg):
    def run(self) -> None:
        self.info("Phase 0: collecting per-site feature statistics (sums and counts only)")
        results = self.send_model_and_wait(
            task_name=self.task_name, targets=self.sample_clients(self.num_clients),
            data=FLModel(params={}, meta={"phase": "stats"}),
        )
        if len(results) < self.num_clients:
            raise RuntimeError(f"only {len(results)}/{self.num_clients} sites returned statistics")

        stats = [{k: _to_numpy(v).astype(np.float64) for k, v in r.params.items()} for r in results]
        n_total = sum(float(s["n"][0]) for s in stats)

        model = self.model if isinstance(self.model, FLModel) else (
            FLModel(params=self.model) if self.model is not None else self.load_model()
        )
        for block in BLOCKS:
            mean = sum(s[f"{block}_sum"] for s in stats) / n_total
            var = np.maximum(sum(s[f"{block}_sumsq"] for s in stats) / n_total - mean ** 2, 0.0)
            std = np.sqrt(var)
            std[std < 1e-8] = 1.0  # constant column -> leave unscaled
            dtype = _to_numpy(model.params[f"{block}_mean"]).dtype
            model.params[f"{block}_mean"] = mean.astype(dtype)
            model.params[f"{block}_std"] = std.astype(dtype)
        self.info(f"Global scaling built from {int(n_total)} training sessions across {len(stats)} sites")

        self.model = model  # FedAvg.run() starts from this model
        super().run()
