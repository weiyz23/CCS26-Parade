#!/usr/bin/env python3
"""Public AE CLI. Heavy dependencies are imported only after argument parsing."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.core.runtime import CONFIGS, absolute, read_json


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return parsed


def csv_ints(value):
    try:
        values = [positive(item.strip()) for item in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers") from exc
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Values must be unique")
    return values


def csv_seeds(value):
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integer seeds") from exc
    if not values or any(seed < 0 or seed >= 2**32 for seed in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Seeds must be unique integers in [0, 2**32)")
    return values


def build_parser():
    parser = argparse.ArgumentParser(description="PARADE Artifact Evaluation experiments")
    commands = parser.add_subparsers(dest="experiment", required=True)
    for name in ("main", "dimension", "k"):
        cmd = commands.add_parser(name)
        cmd.add_argument("-B", "--raw-data-base", type=absolute,
                         default=absolute(os.environ.get("RAW_DATA_BASE", ROOT / "data" / "raw")))
        cmd.add_argument("--output-dir", type=absolute, default=ROOT / "experiments" / "results" / name)
        cmd.add_argument("--dry-run", action="store_true", help="Print the plan without preparing data or writing outputs")
        if name == "main":
            cmd.add_argument("--datasets", help="Comma-separated event IDs; default: all eight in paper order")
            continue
        cmd.add_argument("--data-dir", type=absolute, help="Use existing tensors instead of automatically preparing the paper RIB")
        cmd.add_argument("--config", type=absolute, help="Override the base training JSON (relative paths use the caller's directory)")
        cmd.add_argument("--seeds", type=csv_seeds, default=[42])
        cmd.add_argument("--device", choices=["auto", "cpu"], default="auto")
        cmd.add_argument("--epochs", type=positive)
        cmd.add_argument("--batch-size", type=positive)
        cmd.add_argument("--micro-batch-size", type=positive)
        cmd.add_argument("--num-samples-per-profile", type=positive)
        if name == "dimension":
            cmd.add_argument("--dimensions", dest="values", type=csv_ints, default=[12, 16, 24, 32, 48, 64])
            cmd.add_argument("--num-vectors", type=positive, default=2)
            cmd.add_argument("--geometry", choices=["hyperbolic", "euclidean"], default="hyperbolic")
        else:
            cmd.add_argument("--k-values", dest="values", type=csv_ints, default=[1, 2, 3, 4, 5, 6])
            cmd.add_argument("--embed-dim", type=positive, default=24)
            cmd.add_argument("--geometry", choices=["both", "hyperbolic", "euclidean"], default="both")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from experiments.core.execution import run_main, run_sensitivity
    try:
        if args.experiment == "main":
            events = read_json(CONFIGS / "historical_events.json")
            if args.datasets is not None:
                selected = args.datasets.split(",")
                selected = [name.strip() for name in selected]
                unknown = set(selected) - {event["dataset"] for event in events}
                if unknown or len(set(selected)) != len(selected):
                    parser.error(f"Unknown or duplicate dataset IDs: {args.datasets}")
                events = [event for event in events if event["dataset"] in selected]
            return run_main(events, args.raw_data_base, args.output_dir, dry_run=args.dry_run)
        if args.experiment == "k":
            args.geometries = ["hyperbolic", "euclidean"] if args.geometry == "both" else [args.geometry]
        else:
            args.geometries = [args.geometry]
        return run_sensitivity(args)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
