#!/usr/bin/env python3
"""Compatibility plotting entry point using the shared AE metric reporter."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core.reporting import FIELDS, summarize
from experiments.core.runtime import absolute, read_json


def collect_rows(runs_dir):
    summary = runs_dir / "summary.json"
    if summary.is_file():
        return read_json(summary)
    rows = []
    for metrics_file in sorted(runs_dir.rglob("train_metrics.json")):
        model_dir = metrics_file.parent
        status_file = model_dir / "run_status.json"
        if status_file.is_file() and read_json(status_file).get("status") != "complete":
            continue
        if not (model_dir / "best_model.pth").is_file():
            continue
        metrics = read_json(metrics_file)
        row = {key: metrics.get(key) for key in FIELDS}
        row.update(run=model_dir.name, model_dir=str(model_dir), status="complete", error="")
        # Older manifests omitted the seed. Keep that measurement explicitly unknown.
        row["seed"] = metrics.get("seed", "unknown")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=absolute, help="Legacy root containing experiments/")
    parser.add_argument("--runs-dir", type=absolute, help="AE experiment result directory")
    parser.add_argument("--output", type=absolute, help="PNG/PDF output path; both formats are written with this stem")
    parser.add_argument("--axis", choices=["k", "dimension"], default="k")
    args = parser.parse_args()
    runs_dir = args.runs_dir or (args.dataset_root / "experiments" if args.dataset_root else ROOT / "experiments/results/k")
    output = args.output or runs_dir / "geometry_metrics_comparison.png"
    rows = collect_rows(runs_dir)
    if not rows:
        parser.error(f"No completed training measurements found under {runs_dir}")
    summarize(rows, output.parent, args.axis, figure_stem=output.stem)
    print(f"Saved summary.csv and {output.stem}.png/.pdf in {output.parent}")


if __name__ == "__main__":
    main()
