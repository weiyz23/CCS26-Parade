"""Experiment orchestration; training and detection stay in their existing modules."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import math
import os
from pathlib import Path
import sys
import time

from .preparation import prepare_as_metadata, prepare_snapshot, validate_training_data
from .runtime import CONFIGS, ROOT, ensure_parser, exclusive_lock, read_json, run_logged, write_csv, write_json


def run_main(events, raw_root, output_dir, *, dry_run=False):
    planned = []
    for event in events:
        dataset_dir = ROOT / "dataset" / event["dataset"]
        window = f"{event['start_utc']},{event['end_utc']}"
        command = ["bash", str(ROOT / "preprocess" / "run.sh"), "--raw-data-base", str(raw_root),
                   "full-once", event["dataset"], window]
        planned.append((event, dataset_dir, command))
    if dry_run:
        for event, dataset_dir, command in planned:
            print(f"{event['dataset']} ({event['label']}): prepare AS metadata for {event['start_utc'][:7]}")
            print(f"  {command!r}\n  dataset={dataset_dir}\n  log={output_dir / event['dataset'] / 'run.log'}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(ROOT / ".pids" / "ae-main.lock"):
        ensure_parser(output_dir / "build.log")
        rows = []
        for event, dataset_dir, command in planned:
            started = time.monotonic()
            log_path = output_dir / event["dataset"] / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            row = dict(event, status="running", seconds=None, log=str(log_path),
                       results=str(dataset_dir / "results"), model=str(dataset_dir / "saved_models"), error="")
            env = os.environ.copy()
            env["RAW_DATA_BASE"] = str(raw_root)
            env["PARADE_LOG_DIR"] = str(log_path.parent / "pipeline")
            env["SKIP_PIPELINE_DETECTION"] = "0"
            env["MPLBACKEND"] = "Agg"
            write_json(log_path.parent / "run_config.json", {
                "event": event, "command": command, "raw_data_base": str(raw_root),
                "python": sys.executable,
                "training_environment": {key: env[key] for key in (
                    "EMBEDDING_DIM", "NUM_VECTORS", "TRAIN_EPOCHS", "TRAIN_PATIENCE",
                    "TRAIN_DELTA", "TRAIN_LR", "TRAIN_SEED") if key in env},
            })
            try:
                with log_path.open("a") as log, redirect_stdout(log), redirect_stderr(log):
                    prepare_as_metadata(event["start_utc"][:10], dataset_dir, raw_root)
                run_logged(command, log_path, env=env)
                required = [dataset_dir / "saved_models" / name for name in ("best_model.pth", "thresholds.json")]
                required.append(dataset_dir / "results" / "anomalous_raw.csv")
                if not all(path.is_file() for path in required):
                    raise RuntimeError("Pipeline exited without expected model/detection outputs")
                row["status"] = "complete"
            except Exception as exc:
                row.update(status="failed", error=str(exc))
                with log_path.open("a") as log:
                    log.write(f"\nAE FAILURE: {exc}\n")
            row["seconds"] = time.monotonic() - started
            rows.append(row)
            write_json(output_dir / "summary.json", rows)
            write_csv(output_dir / "summary.csv", rows,
                      ["dataset", "label", "category", "start_utc", "end_utc", "status", "seconds", "log", "results", "model", "error"])
            print(f"{event['dataset']}: {row['status']} ({row['seconds']:.1f}s)", flush=True)
    return int(any(row["status"] != "complete" for row in rows))


def sweep_configs(args, data_dir):
    configs = []
    for geometry in args.geometries:
        base_path = args.config or (CONFIGS / f"{geometry}_base.json")
        base = read_json(base_path)
        for value in args.values:
            for seed in args.seeds:
                cfg = copy.deepcopy(base)
                cfg["model"].update(geometry=geometry,
                                    embed_dim=value if args.experiment == "dimension" else args.embed_dim,
                                    num_vectors=args.num_vectors if args.experiment == "dimension" else value)
                cfg["train"]["seed"] = seed
                for name in ("epochs", "batch_size", "num_samples_per_profile", "micro_batch_size"):
                    value_override = getattr(args, name, None)
                    if value_override is not None:
                        cfg["train"][name] = value_override
                run_id = f"{geometry}_{'dim' if args.experiment == 'dimension' else 'k'}{value}_seed{seed}"
                cfg["paths"] = {"data_dir": str(data_dir), "model_dir": str(args.output_dir / run_id)}
                train = cfg["train"]
                if any(int(train[name]) <= 0 for name in ("epochs", "batch_size", "num_samples_per_profile", "patience")):
                    raise ValueError("Training epochs, batch size, samples and patience must be positive")
                if not 0 < int(train.get("micro_batch_size", train["batch_size"])) <= int(train["batch_size"]):
                    raise ValueError("micro_batch_size must be in (0, batch_size]")
                configs.append(cfg)
    return configs


def run_configs(configs, output_dir, *, axis, dry_run=False, device="auto"):
    if dry_run:
        for cfg in configs:
            print(f"{cfg['paths']['model_dir']}: {cfg}")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with exclusive_lock(output_dir / ".run.lock"):
        for cfg in configs:
            model_dir = Path(cfg["paths"]["model_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            config_path = model_dir / "train_config.json"
            write_json(config_path, cfg)
            row = {"run": model_dir.name, "geometry": cfg["model"]["geometry"],
                   "embed_dim": cfg["model"]["embed_dim"], "num_vectors": cfg["model"]["num_vectors"],
                   "seed": cfg["train"]["seed"], "model_dir": str(model_dir), "status": "running", "error": ""}
            write_json(model_dir / "run_status.json", row)
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            if device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            started = time.monotonic()
            try:
                run_logged([sys.executable, ROOT / "experiments" / "scripts" / "train.py", "--config", config_path],
                           model_dir / "training.log", env=env)
                metrics = read_json(model_dir / "train_metrics.json")
                if not math.isfinite(metrics.get("best_loss", float("nan"))):
                    raise RuntimeError("Training did not produce a finite best loss")
                if not (model_dir / "best_model.pth").is_file():
                    raise RuntimeError("Training did not produce best_model.pth")
                row.update({key: metrics.get(key) for key in (
                    "best_loss", "total_train_time_sec", "early_stop_mark_epoch", "early_stop_mark_cum_time_sec",
                    "max_cuda_peak_alloc_mb", "max_cuda_peak_reserved_mb", "configured_epochs")})
                row["status"] = "complete"
            except Exception as exc:
                row.update(status="failed", error=str(exc))
            row["wall_time_sec"] = time.monotonic() - started
            write_json(model_dir / "run_status.json", row)
            rows.append(row)
            write_json(output_dir / "summary.json", rows)
            print(f"{model_dir.name}: {row['status']}", flush=True)
        from .reporting import summarize
        summarize(rows, output_dir, axis)
    return int(any(row["status"] != "complete" for row in rows))


def run_sensitivity(args):
    data_dir = args.data_dir or (ROOT / "dataset" / "ae_snapshot_20260101")
    configs = sweep_configs(args, data_dir)
    if args.dry_run:
        if args.data_dir is None:
            print(f"Prepare/reuse 2026-01-01 00:00 UTC RIB (74 collectors): {data_dir}; raw={args.raw_data_base}")
        return run_configs(configs, args.output_dir, axis=args.experiment, dry_run=True)
    if args.data_dir is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ensure_parser(args.output_dir / "preparation.log")
        prepare_snapshot(data_dir, args.raw_data_base, args.output_dir / "preparation.log")
    validate_training_data(data_dir)
    return run_configs(configs, args.output_dir, axis=args.experiment, device=args.device)
