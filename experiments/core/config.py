import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def override_from_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.num_vectors is not None:
        cfg["model"]["num_vectors"] = args.num_vectors
    if args.embed_dim is not None:
        cfg["model"]["embed_dim"] = args.embed_dim
    if args.geometry is not None:
        cfg["model"]["geometry"] = args.geometry
    if args.data_dir is not None:
        cfg["paths"]["data_dir"] = args.data_dir
    if args.model_dir is not None:
        cfg["paths"]["model_dir"] = args.model_dir
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified experiment training entry")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config")
    parser.add_argument("--num_vectors", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--geometry", type=str, choices=["hyperbolic", "euclidean"], default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    return parser
