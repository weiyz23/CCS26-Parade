#!/usr/bin/env python3
"""Run optional plots independently and report their status separately."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys


SCRIPTS = ("visualize.py", "visualize_as_dim_reduction.py")


def _write_status(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_visualizations(data_dir, model_dir, viz_dir, embed_dim=24, num_vectors=2, *, env=None, log_path=None):
    data_dir, model_dir, viz_dir = (Path(path).resolve() for path in (data_dir, model_dir, viz_dir))
    viz_dir.mkdir(parents=True, exist_ok=True)
    status_path = viz_dir / "visualization_status.json"
    report = {"status": "running", "error": "", "steps": []}
    _write_status(status_path, report)
    context = Path(log_path).open("a", encoding="utf-8") if log_path else nullcontext(None)
    with context as log:
        for script in SCRIPTS:
            command = [sys.executable, str(Path(__file__).parent / script),
                       "--data_dir", str(data_dir), "--model_dir", str(model_dir),
                       "--viz_dir", str(viz_dir), "--embed_dim", str(embed_dim),
                       "--num_vectors", str(num_vectors)]
            print(f"Running visualization: {script}", file=log or sys.stdout, flush=True)
            step = {"script": script, "status": "complete", "exit_code": 0, "error": ""}
            try:
                subprocess.run(command, env=env, stdout=log,
                               stderr=subprocess.STDOUT if log else None, check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                step.update(status="failed", exit_code=getattr(exc, "returncode", None), error=str(exc))
                print(f"Visualization failed: {script}: {exc}", file=log or sys.stderr, flush=True)
            report["steps"].append(step)
            _write_status(status_path, report)
    failures = [step for step in report["steps"] if step["status"] == "failed"]
    report["status"] = "failed" if failures else "complete"
    report["error"] = "; ".join(f"{step['script']}: {step['error']}" for step in failures)
    _write_status(status_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--viz_dir", type=Path, required=True)
    parser.add_argument("--embed_dim", type=int, default=24)
    parser.add_argument("--num_vectors", type=int, default=2)
    args = parser.parse_args()
    report = run_visualizations(args.data_dir, args.model_dir, args.viz_dir, args.embed_dim, args.num_vectors)
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
