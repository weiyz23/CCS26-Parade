"""Run geometry comparison with explicit shared data and path overrides."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core.execution import run_configs
from experiments.core.preparation import validate_training_data
from experiments.core.runtime import absolute, read_json


def main():
    parser = argparse.ArgumentParser(description="Compare hyperbolic and Euclidean training")
    parser.add_argument("--hyper_config", type=absolute, required=True)
    parser.add_argument("--euc_config", type=absolute, required=True)
    parser.add_argument("--data_dir", type=absolute)
    parser.add_argument("--output_root", type=absolute, default=ROOT / "experiments" / "results" / "compare")
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--num_vectors", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = []
    for geometry, config in [("hyperbolic", args.hyper_config), ("euclidean", args.euc_config)]:
        cfg = read_json(config)
        data_dir = args.data_dir or absolute(cfg["paths"]["data_dir"])
        cfg["paths"] = {"data_dir": str(data_dir), "model_dir": str(args.output_root / geometry)}
        cfg["model"]["geometry"] = geometry
        for key in ("embed_dim", "num_vectors"):
            if getattr(args, key) is not None:
                cfg["model"][key] = getattr(args, key)
        if not args.dry_run:
            validate_training_data(data_dir)
        configs.append(cfg)
    return run_configs(configs, args.output_root, axis="k", dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
