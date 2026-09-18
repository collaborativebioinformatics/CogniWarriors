#!/usr/bin/env python3
"""Dashboard-friendly wrapper for an NVIDIA FLARE simulator run."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TARGET_RE = re.compile(
    r"^(?P<target>[^:]+): MSE=(?P<mse>[-+0-9.eE]+)\s+MAE=(?P<mae>[-+0-9.eE]+)\s+R.\S*=(?P<r2>[-+0-9.eE]+)"
)


def command_text(command: List[str]) -> str:
    return " ".join(str(item) for item in command)


def run_command(command: List[str], env: Dict[str, str] | None = None) -> Dict[str, Any]:
    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout or ""
    return {
        "command": command_text(command),
        "returncode": completed.returncode,
        "duration_seconds": round(time.time() - started_at, 3),
        "output": output,
        "tail": "\n".join(output.splitlines()[-80:]),
    }


def latest_file(root: Path, name: str) -> str | None:
    matches = sorted(root.rglob(name), key=lambda path: path.stat().st_mtime, reverse=True)
    return str(matches[0]) if matches else None


def parse_evaluation(output: str) -> Dict[str, Any]:
    targets = []
    for line in output.splitlines():
        match = TARGET_RE.match(line.strip())
        if not match:
            continue
        targets.append(
            {
                "target": match.group("target"),
                "mse": float(match.group("mse")),
                "mae": float(match.group("mae")),
                "r2": float(match.group("r2")),
            }
        )
    return {"targets": targets, "raw_output": output}


def write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def default_output_dir() -> Path:
    env_output_dir = os.environ.get("FLARE_OUTPUT_DIR")
    return Path(env_output_dir) if env_output_dir else REPO / "flare_outputs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sites", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--split", choices=["iid", "age"], default="iid")
    parser.add_argument("--loss", choices=["mse", "lmmnn"], default="mse")
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    args = parser.parse_args()

    if args.n_sites < 1:
        parser.error("--n-sites must be at least 1")
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.local_epochs < 1:
        parser.error("--local-epochs must be at least 1")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir.resolve()
    run_dir = output_dir / "runs" / run_id
    site_data_dir = run_dir / "site_data"
    workspace_dir = run_dir / "workspace"
    summary_path = run_dir / "summary.json"
    latest_summary_path = output_dir / "latest_summary.json"
    env = os.environ.copy()
    env.setdefault("NBBH_DATA_ROOT", str(REPO / "data"))

    summary: Dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "started_at": round(time.time(), 3),
        "finished_at": None,
        "parameters": vars(args) | {"output_dir": str(output_dir)},
        "site_data_dir": str(site_data_dir),
        "workspace_dir": str(workspace_dir),
        "commands": [],
        "models": {},
        "evaluation": None,
        "error": None,
    }
    write_json(summary_path, summary)
    write_json(latest_summary_path, summary)

    try:
        prep = [
            sys.executable,
            str(HERE / "prepare_site_data.py"),
            "--n-sites",
            str(args.n_sites),
            "--split",
            args.split,
            "--seed",
            str(args.seed),
            "--out",
            str(site_data_dir),
        ]
        prep_result = run_command(prep, env=env)
        summary["commands"].append({"name": "prepare_site_data", **prep_result})
        write_json(summary_path, summary)
        write_json(latest_summary_path, summary)
        if prep_result["returncode"] != 0:
            raise RuntimeError("site data preparation failed")

        job = [
            sys.executable,
            str(HERE / "job.py"),
            "--mode",
            "sim",
            "--n-sites",
            str(args.n_sites),
            "--rounds",
            str(args.rounds),
            "--local-epochs",
            str(args.local_epochs),
            "--loss",
            args.loss,
            "--mu",
            str(args.mu),
            "--seed",
            str(args.seed),
            "--data-root",
            str(site_data_dir),
            "--workspace",
            str(workspace_dir),
        ]
        job_result = run_command(job, env=env)
        summary["commands"].append({"name": "nvflare_sim", **job_result})
        write_json(summary_path, summary)
        write_json(latest_summary_path, summary)
        if job_result["returncode"] != 0:
            raise RuntimeError("NVFLARE simulator failed")

        best_model = latest_file(workspace_dir, "best_FL_global_model.pt")
        final_model = latest_file(workspace_dir, "FL_global_model.pt")
        eval_model = best_model or final_model
        summary["models"] = {"best": best_model, "final": final_model, "evaluated": eval_model}
        if best_model:
            shutil.copyfile(best_model, output_dir / "best_FL_global_model.pt")
        if final_model:
            shutil.copyfile(final_model, output_dir / "FL_global_model.pt")

        if eval_model:
            evaluate = [
                sys.executable,
                str(HERE / "evaluate_global.py"),
                "--model",
                eval_model,
                "--test-dir",
                str(site_data_dir / "test"),
            ]
            eval_result = run_command(evaluate, env=env)
            summary["commands"].append({"name": "evaluate_global", **eval_result})
            summary["evaluation"] = parse_evaluation(eval_result["output"])
            if eval_result["returncode"] != 0:
                raise RuntimeError("global model evaluation failed")

        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
    finally:
        summary["finished_at"] = round(time.time(), 3)
        write_json(summary_path, summary)
        write_json(latest_summary_path, summary)

    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
