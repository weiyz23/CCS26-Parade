"""Compatibility dispatch for the original positional experiment entry point."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core.execution import run_configs
from experiments.core.preparation import validate_training_data
from experiments.core.runtime import CONFIGS, absolute, read_json
from experiments.scripts.run_experiment import main as run_experiment, positive


def main():
    parser = argparse.ArgumentParser(description="Legacy dispatch; prefer run_main.sh, run_dimension.sh and run_k.sh")
    parser.add_argument("mode", nargs="?", default="hyperbolic", choices=[
        "hyperbolic", "euclidean", "compare", "sweep-k-h", "sweep-k-e", "sweep-dim-h", "hyperbolic-dim"])
    parser.add_argument("dataset", nargs="?", default="ae_snapshot_20260101")
    parser.add_argument("k", nargs="?", type=positive, default=2)
    parser.add_argument("dim", nargs="?", type=positive, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dataset_dir = absolute(args.dataset) if "/" in args.dataset else ROOT / "dataset" / args.dataset
    data_dir = dataset_dir if (dataset_dir / "meta.json").is_file() else dataset_dir / "processed_data"
    output_dir = dataset_dir / "experiments"
    extra = ["--dry-run"] if args.dry_run else []
    if args.mode.startswith("sweep-k-"):
        geometry = "hyperbolic" if args.mode.endswith("-h") else "euclidean"
        return run_experiment(["k", "--data-dir", str(data_dir), "--output-dir", str(output_dir / f"{geometry}_sweep"),
                               "--geometry", geometry, "--embed-dim", str(args.dim)] + extra)
    if args.mode == "sweep-dim-h":
        return run_experiment(["dimension", "--data-dir", str(data_dir), "--output-dir", str(output_dir / "hyperbolic_dim_sweep"),
                               "--num-vectors", str(args.k)] + extra)
    geometries = ["hyperbolic", "euclidean"] if args.mode == "compare" else ["euclidean" if args.mode == "euclidean" else "hyperbolic"]
    configs = []
    for geometry in geometries:
        cfg = read_json(CONFIGS / f"{geometry}_base.json")
        cfg["model"].update(embed_dim=args.dim, num_vectors=args.k)
        cfg["paths"] = {"data_dir": str(data_dir), "model_dir": str(output_dir / geometry)}
        configs.append(cfg)
    if not args.dry_run:
        validate_training_data(data_dir)
    return run_configs(configs, output_dir, axis="k", dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
