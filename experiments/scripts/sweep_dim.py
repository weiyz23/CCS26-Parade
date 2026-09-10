"""Compatibility CLI forwarding to the shared AE sweep implementation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core.runtime import absolute, read_json
from experiments.scripts.run_experiment import main as run_experiment


def main():
    parser = argparse.ArgumentParser(description="Legacy sweep arguments; prefer the AE shell entry point")
    parser.add_argument("--config", type=absolute, required=True)
    parser.add_argument("--output_root", type=absolute)
    parser.add_argument("--data_dir", type=absolute)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dim_values", default="12,16,24,32,48,64")
    parser.add_argument("--geometry", default="hyperbolic", choices=["hyperbolic", "euclidean"])
    parser.add_argument("--num_vectors", type=int, default=2)
    args = parser.parse_args()
    cfg = read_json(args.config)
    data_dir = args.data_dir or absolute(cfg["paths"]["data_dir"])
    output_dir = args.output_root or absolute(cfg["paths"]["model_dir"])
    forwarded = ["dimension", "--config", str(args.config), "--data-dir", str(data_dir),
                 "--output-dir", str(output_dir), "--seeds", args.seeds]
    forwarded += ["--dimensions", args.dim_values, "--num-vectors", str(args.num_vectors), "--geometry", args.geometry]
    if args.dry_run:
        forwarded.append("--dry-run")
    return run_experiment(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
