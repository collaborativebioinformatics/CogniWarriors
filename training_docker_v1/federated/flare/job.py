#!/usr/bin/env python3
"""Build and run the NVFlare FedAvg job for the fusion model.

Modes
    sim     all sites as threads in one process (fastest; for development)
    poc     real server + client processes on this machine (after `nvflare poc prepare/start`)
    export  write the job folder to disk, to submit to a real deployment
    prod    submit to a real deployment using the admin startup kit

Examples (from the repo root)
    python federated/flare/prepare_site_data.py --n-sites 4
    python federated/flare/job.py --mode sim --n-sites 4 --rounds 50
    python federated/flare/job.py --mode export --n-sites 4 --job-dir jobs/
    python federated/flare/job.py --mode prod --startup-kit /path/to/admin@nvidia.com --data-root /data
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAINING = REPO
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

from controller import GlobalScalingFedAvg

# code that must be shipped with the job (NVFlare copies it into each site's
# and the server's "custom" folder) -- code only, never data
SHARED_FILES = [TRAINING / f for f in ("config.py", "fusion_model.py", "phenotype_model.py",
                                        "lmmnn_loss.py", "multimodal_data.py")] + [HERE / "cogni_model.py"]


class GlobalScalingFedAvgRecipe(FedAvgRecipe):
    """NVFlare's PyTorch FedAvg recipe, with the server controller swapped for
    GlobalScalingFedAvg (adds the federated feature-statistics step).
    Same arguments as the base FedAvg._create_controller in NVFlare 2.9."""

    def _create_controller(self, persistor_id, model_params, model_aggregator):
        return GlobalScalingFedAvg(
            num_clients=self.min_clients,
            num_rounds=self.num_rounds,
            persistor_id=persistor_id,
            model=model_params,
            save_filename=self.save_filename,
            aggregator=model_aggregator,
            stop_cond=self.stop_cond,
            patience=self.patience,
            task_name="train",
            exclude_vars=self.exclude_vars,
            aggregation_weights=self.aggregation_weights,
            memory_gc_rounds=self.server_memory_gc_rounds,
            enable_tensor_disk_offload=self.enable_tensor_disk_offload,
            **self._get_controller_kwargs(),
        )


def build_recipe(args):
    train_args = (f"--data-root {args.data_root} --local-epochs {args.local_epochs} "
                  f"--loss {args.loss} --mu {args.mu} --seed {args.seed}")
    recipe = GlobalScalingFedAvgRecipe(
        name=args.name,
        model={"class_path": "cogni_model.FedFusionModel",
               "args": {"image_dim": args.image_dim, "phenotype_dim": args.phenotype_dim,
                        "n_covariates": args.n_covariates, "n_targets": args.n_targets,
                        "use_lmmnn": args.loss == "lmmnn"}},
        min_clients=args.n_sites,
        num_rounds=args.rounds,
        train_script=str(HERE / "client.py"),
        train_args=train_args,
        key_metric="val_mse",
        # "val_mse < 0" can never be true, so it never stops early by itself;
        # it tells NVFlare lower val_mse = better (best model + patience)
        stop_cond="val_mse < 0",
        patience=args.patience,
    )
    for f in SHARED_FILES:
        recipe.add_client_file(str(f))
        recipe.add_server_file(str(f))
    recipe.add_server_file(str(HERE / "controller.py"))
    return recipe


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sim", "poc", "export", "prod"], default="sim")
    p.add_argument("--name", default="cogniwarriors_fedavg")
    p.add_argument("--n-sites", type=int, default=4, help="sites that must join each round")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=20, help="stop after N rounds without val improvement")
    p.add_argument("--loss", choices=["mse", "lmmnn"], default="mse")
    p.add_argument("--mu", type=float, default=0.0, help="FedProx strength; 0 = plain FedAvg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default=str(HERE / "data"),
                   help="folder with one sub-folder per site (in prod each site can override with SITE_DATA_DIR)")
    p.add_argument("--image-dim", type=int, default=80)
    p.add_argument("--phenotype-dim", type=int, default=91)
    p.add_argument("--n-covariates", type=int, default=2)
    p.add_argument("--n-targets", type=int, default=1)
    p.add_argument("--workspace", default="/tmp/nvflare/cogniwarriors", help="simulator workspace")
    p.add_argument("--job-dir", default=str(REPO / "jobs"), help="where --mode export writes the job")
    p.add_argument("--startup-kit", help="admin startup kit folder (--mode prod)")
    args = p.parse_args()

    recipe = build_recipe(args)

    if args.mode == "export":
        recipe.export(args.job_dir)
        print(f"Job written to {args.job_dir}/{args.name}")
        return

    if args.mode == "sim":
        from nvflare.recipe import SimEnv
        env = SimEnv(num_clients=args.n_sites, workspace_root=args.workspace)
    elif args.mode == "poc":
        from nvflare.recipe import PocEnv
        env = PocEnv(num_clients=args.n_sites)
    else:
        if not args.startup_kit:
            p.error("--mode prod needs --startup-kit")
        from nvflare.recipe import ProdEnv
        env = ProdEnv(startup_kit_location=args.startup_kit)

    run = recipe.execute(env)
    print(f"Job status: {run.get_status()}")
    print(f"Results: {run.get_result()}")


if __name__ == "__main__":
    main()
