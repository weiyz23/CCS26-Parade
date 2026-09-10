#!/usr/bin/env python3
"""Embedding dispersion analysis for euclidean/hyperbolic k-sweeps.

Tasks:
1) Distribution of semantic distances for randomly sampled AS pairs (>= 1M by default).
2) Distribution of nearest-neighbor distances for randomly sampled anchor ASes (>= 10K by default).
3) Distribution of semantic distances among ASes in the first 3 propagation layers of a random prefix.

Outputs:
- Three boxplot figures (one per task) faceted by k.
- Summary CSV files with count/mean/median/IQR/P90/P95.
- Run metadata JSON (seed, source, rank strategy, sampled prefix, etc.).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.models.base import BORDER_VALUE
from experiments.models.factory import build_model


RANK_ORDER = ["0-500", "501-1500", "1501-5000", "5001-15000", "15000+", "Unknown"]


@dataclass
class RunSpec:
    geometry: str
    k: int
    run_dir: Path
    manifest_path: Path
    model_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_runs(exp_root: Path, *, model_seed=None, embed_dim=None) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for manifest_path in sorted(exp_root.rglob("train_manifest.json")):
        run_dir = manifest_path.parent
        model_path = run_dir / "best_model.pth"
        status_path = run_dir / "run_status.json"
        if status_path.is_file() and load_json(status_path).get("status") != "complete":
            continue
        if model_path.is_file():
            manifest = load_json(manifest_path)
            if model_seed is not None and manifest.get("seed") != model_seed:
                continue
            if embed_dim is not None and manifest["embed_dim"] != embed_dim:
                continue
            specs.append(
                RunSpec(geometry=manifest["geometry"], k=int(manifest["num_vectors"]),
                        run_dir=run_dir, manifest_path=manifest_path, model_path=model_path)
            )
    return sorted(specs, key=lambda x: (x.k, x.geometry))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rank_by_aid(data_root: Path, as_map: Dict[int, int], rank_file: Optional[Path] = None) -> Tuple[np.ndarray, int]:
    rank_file = rank_file or data_root / "as_info.jsonl"
    if not rank_file.exists():
        raise FileNotFoundError(f"Rank file not found: {rank_file}")

    rank_by_aid = np.full((len(as_map),), np.nan, dtype=np.float64)
    loaded_rows = 0
    with rank_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            asn = obj.get("asn")
            rank = obj.get("rank")
            if asn is None or rank is None:
                continue
            aid = as_map.get(int(asn))
            if aid is None:
                continue
            rank_by_aid[aid] = float(rank)
            loaded_rows += 1

    if loaded_rows == 0:
        raise ValueError(f"No valid rank rows found in: {rank_file}")

    return rank_by_aid, loaded_rows


def build_rank_bins(rank_values: np.ndarray) -> np.ndarray:
    out = np.full((len(rank_values),), "Unknown", dtype=object)
    valid_mask = np.isfinite(rank_values) & (rank_values > 0)
    if valid_mask.sum() == 0:
        return out

    rv = rank_values
    out[(rv >= 0) & (rv <= 500) & valid_mask] = "0-500"
    out[(rv >= 501) & (rv <= 1500) & valid_mask] = "501-1500"
    out[(rv >= 1501) & (rv <= 5000) & valid_mask] = "1501-5000"
    out[(rv >= 5001) & (rv <= 15000) & valid_mask] = "5001-15000"
    out[(rv >= 15001) & valid_mask] = "15000+"
    return out


def load_model_for_run(run: RunSpec, num_as: int, num_profiles: int, device: torch.device):
    manifest = load_json(run.manifest_path)
    model = build_model(
        geometry=manifest["geometry"],
        num_as=num_as,
        num_profiles=num_profiles,
        embed_dim=manifest["embed_dim"],
        num_vectors=manifest["num_vectors"],
    ).to(device)
    state = torch.load(run.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _distance_pairwise(
    model,
    geometry: str,
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
) -> torch.Tensor:
    if geometry == "hyperbolic":
        d = model.manifold.dist(emb_a, emb_b)
    else:
        d = torch.linalg.norm(emb_a - emb_b, dim=-1)
    return torch.clamp(d, max=BORDER_VALUE)


def _distance_matrix(
    model,
    geometry: str,
    anchor_emb: torch.Tensor,
    all_emb: torch.Tensor,
) -> torch.Tensor:
    if geometry == "hyperbolic":
        d = model.manifold.dist(anchor_emb.unsqueeze(1), all_emb.unsqueeze(0))
    else:
        d = torch.linalg.norm(anchor_emb.unsqueeze(1) - all_emb.unsqueeze(0), dim=-1)
    return torch.clamp(d, max=BORDER_VALUE)


def sample_distances_task1(
    model,
    geometry: str,
    as_emb: torch.Tensor,
    rank_bins: np.ndarray,
    n_pairs: int,
    pair_batch: int,
) -> pd.DataFrame:
    num_as = as_emb.shape[0]
    left = torch.randint(0, num_as, (n_pairs,), device=as_emb.device)
    right = torch.randint(0, num_as, (n_pairs,), device=as_emb.device)

    same_mask = left == right
    while same_mask.any():
        right[same_mask] = torch.randint(0, num_as, (int(same_mask.sum().item()),), device=as_emb.device)
        same_mask = left == right

    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n_pairs, pair_batch):
            j = min(i + pair_batch, n_pairs)
            d = _distance_pairwise(model, geometry, as_emb[left[i:j]], as_emb[right[i:j]])
            chunks.append(d.detach().cpu().numpy())
    dists = np.concatenate(chunks, axis=0)
    left_np = left.cpu().numpy()

    return pd.DataFrame(
        {
            "distance": dists,
            "rank_bin": rank_bins[left_np],
        }
    )


def sample_distances_task2(
    model,
    geometry: str,
    as_emb: torch.Tensor,
    rank_bins: np.ndarray,
    n_anchors: int,
    nn_batch: int,
) -> pd.DataFrame:
    num_as = as_emb.shape[0]
    anchors = torch.randperm(num_as, device=as_emb.device)[: min(n_anchors, num_as)]
    min_d_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, anchors.shape[0], nn_batch):
            j = min(i + nn_batch, anchors.shape[0])
            idx = anchors[i:j]
            mat = _distance_matrix(model, geometry, as_emb[idx], as_emb)
            local_rows = torch.arange(idx.shape[0], device=as_emb.device)
            mat[local_rows, idx] = float("inf")
            min_d = torch.min(mat, dim=1).values
            min_d_chunks.append(min_d.detach().cpu().numpy())

    anchor_np = anchors.cpu().numpy()
    return pd.DataFrame(
        {
            "distance": np.concatenate(min_d_chunks, axis=0),
            "rank_bin": rank_bins[anchor_np],
        }
    )


def find_prefix_top3_aids(
    prefix_to_profile: torch.Tensor,
    triplets: torch.Tensor,
    prefix_map: Dict[str, int],
    rng: random.Random,
    max_trials: int = 200,
) -> Tuple[int, str, torch.Tensor, int]:
    inv_prefix_map = [None] * len(prefix_map)
    for pfx, pid in prefix_map.items():
        inv_prefix_map[pid] = pfx

    for _ in range(max_trials):
        prefix_id = rng.randrange(0, len(prefix_to_profile))
        profile_id = int(prefix_to_profile[prefix_id].item())
        mask = (triplets[:, 0] == profile_id) & (triplets[:, 2] < 3)
        aids = torch.unique(triplets[mask, 1].long())
        if aids.numel() >= 2:
            prefix_str = inv_prefix_map[prefix_id] if prefix_id < len(inv_prefix_map) else str(prefix_id)
            return prefix_id, prefix_str, aids, profile_id
    raise RuntimeError("Could not find a prefix with >=2 AS in hop<3 within max_trials.")


def sample_distances_task3(
    model,
    geometry: str,
    as_emb: torch.Tensor,
    aids_top3: torch.Tensor,
    n_pairs: int,
    pair_batch: int,
) -> pd.DataFrame:
    aids_top3 = aids_top3.to(as_emb.device)
    n = aids_top3.shape[0]
    left_idx = torch.randint(0, n, (n_pairs,), device=as_emb.device)
    right_idx = torch.randint(0, n, (n_pairs,), device=as_emb.device)

    same_mask = left_idx == right_idx
    while same_mask.any():
        right_idx[same_mask] = torch.randint(0, n, (int(same_mask.sum().item()),), device=as_emb.device)
        same_mask = left_idx == right_idx

    left = aids_top3[left_idx]
    right = aids_top3[right_idx]

    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n_pairs, pair_batch):
            j = min(i + pair_batch, n_pairs)
            d = _distance_pairwise(model, geometry, as_emb[left[i:j]], as_emb[right[i:j]])
            chunks.append(d.detach().cpu().numpy())

    return pd.DataFrame({"distance": np.concatenate(chunks, axis=0)})


def summarize(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    def p90(x: pd.Series) -> float:
        return float(np.quantile(x, 0.90))

    def p95(x: pd.Series) -> float:
        return float(np.quantile(x, 0.95))

    g = df.groupby(list(group_cols), dropna=False)["distance"]
    out = g.agg(
        count="count",
        mean="mean",
        std="std",
        median="median",
        q1=lambda x: float(np.quantile(x, 0.25)),
        q3=lambda x: float(np.quantile(x, 0.75)),
        p90=p90,
        p95=p95,
    ).reset_index()
    out["iqr"] = out["q3"] - out["q1"]
    return out


def _iter_k_axes(k_values: List[int]):
    n = len(k_values)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.8 * rows), dpi=120)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    else:
        axes = [axes]
    return fig, axes


def plot_task12(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    k_values = sorted(df["k"].unique().tolist())
    fig, axes = _iter_k_axes(k_values)
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        cur = df[df["k"] == k]
        sns.boxplot(
            data=cur,
            x="rank_bin",
            y="distance",
            hue="geometry",
            order=RANK_ORDER,
            hue_order=["euclidean", "hyperbolic"],
            ax=ax,
            showfliers=False,
        )
        ax.set_title(f"k={k}")
        ax.set_xlabel("Rank Bin")
        ax.set_ylabel("Semantic Distance")
        if idx == 0:
            ax.legend(title="Geometry")
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    for ax in axes[len(k_values) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_task3(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    k_values = sorted(df["k"].unique().tolist())
    fig, axes = _iter_k_axes(k_values)
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        cur = df[df["k"] == k]
        sns.boxplot(
            data=cur,
            x="geometry",
            y="distance",
            order=["euclidean", "hyperbolic"],
            ax=ax,
            showfliers=False,
        )
        ax.set_title(f"k={k}")
        ax.set_xlabel("Geometry")
        ax.set_ylabel("Semantic Distance")

    for ax in axes[len(k_values) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze embedding dispersion across euclidean/hyperbolic sweeps")
    parser.add_argument("--dataset-root", type=Path, help="Legacy dataset root containing experiments/")
    parser.add_argument("--data-dir", type=Path, help="Directory containing meta.json and tensors")
    parser.add_argument("--runs-dir", type=Path, help="Directory containing K-sweep model runs")
    parser.add_argument("--as-info", type=Path, help="AS Rank JSONL for the same historical snapshot")
    parser.add_argument("--model-seed", type=int, help="Select one training seed from a multi-seed sweep")
    parser.add_argument("--embed-dim", type=int, help="Select one embedding dimension")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs-task1", type=int, default=1_000_000)
    parser.add_argument("--anchors-task2", type=int, default=10_000)
    parser.add_argument("--pairs-task3", type=int, default=200_000)
    parser.add_argument("--pair-batch", type=int, default=200_000)
    parser.add_argument("--nn-batch", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="For debug only; 0 means all runs")
    args = parser.parse_args()

    if args.quick_test:
        args.pairs_task1 = min(args.pairs_task1, 20_000)
        args.anchors_task2 = min(args.anchors_task2, 500)
        args.pairs_task3 = min(args.pairs_task3, 20_000)

    set_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu")
    print(f"[Info] device={device}")

    data_root = (args.dataset_root or ROOT / "dataset/ae_snapshot_20260101").resolve()
    exp_root = (args.runs_dir or (data_root / "experiments" if args.dataset_root else ROOT / "experiments/results/k")).resolve()
    processed_root = args.data_dir or (data_root if (data_root / "meta.json").is_file() else data_root / "processed_data")
    processed_root = processed_root.resolve()
    if args.data_dir is not None:
        data_root = processed_root.parent if processed_root.name == "processed_data" else processed_root
    output_dir = args.output_dir or (exp_root / "analysis")

    runs = discover_runs(exp_root, model_seed=args.model_seed, embed_dim=args.embed_dim)
    if args.max_runs > 0:
        runs = runs[: args.max_runs]
    if not runs:
        raise FileNotFoundError(f"No sweep runs found under: {exp_root}")
    settings = {(load_json(run.manifest_path)["embed_dim"], load_json(run.manifest_path).get("seed")) for run in runs}
    if len(settings) != 1 or len({(run.geometry, run.k) for run in runs}) != len(runs):
        raise ValueError("Select a single K sweep using --model-seed, --embed-dim or --runs-dir; mixed settings cannot be pooled")
    if args.data_dir is None and args.dataset_root is None:
        data_paths = {Path(load_json(run.manifest_path)["data_dir"]).resolve() for run in runs}
        if len(data_paths) != 1:
            raise ValueError("Runs use multiple datasets; select a single sweep with --runs-dir")
        processed_root = data_paths.pop()
        data_root = processed_root.parent if processed_root.name == "processed_data" else processed_root
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] discovered {len(runs)} runs")

    meta = load_json(processed_root / "meta.json")
    num_as = int(meta["num_asns"])
    num_profiles = int(meta["num_profiles"])

    as_map: Dict[int, int] = torch.load(processed_root / "as_map.pt", map_location="cpu")
    prefix_map: Dict[str, int] = torch.load(processed_root / "prefix_map.pt", map_location="cpu")
    prefix_to_profile: torch.Tensor = torch.load(processed_root / "prefix_to_profile.pt", map_location="cpu")
    triplets: torch.Tensor = torch.load(processed_root / "profile_triplets.pt", map_location="cpu")

    rank_by_aid, rank_rows = load_rank_by_aid(data_root, as_map, args.as_info)
    rank_source = "as_info_rank"
    rank_bins = build_rank_bins(rank_by_aid)

    prefix_id, prefix_str, aids_top3, profile_id = find_prefix_top3_aids(
        prefix_to_profile=prefix_to_profile,
        triplets=triplets,
        prefix_map=prefix_map,
        rng=rng,
    )
    print(
        f"[Info] selected prefix={prefix_str} (id={prefix_id}, profile_id={profile_id}) "
        f"with {aids_top3.numel()} AS in hop<3"
    )

    all_task1: List[pd.DataFrame] = []
    all_task2: List[pd.DataFrame] = []
    all_task3: List[pd.DataFrame] = []

    for idx, run in enumerate(runs, start=1):
        print(f"[Run {idx}/{len(runs)}] {run.geometry} k={run.k} | loading model...")
        model = load_model_for_run(run, num_as=num_as, num_profiles=num_profiles, device=device)
        as_emb = model.as_embedding.detach()

        print(f"[Run {idx}/{len(runs)}] task1 sampling {args.pairs_task1} AS pairs...")
        t1 = sample_distances_task1(
            model=model,
            geometry=run.geometry,
            as_emb=as_emb,
            rank_bins=rank_bins,
            n_pairs=args.pairs_task1,
            pair_batch=args.pair_batch,
        )
        t1["geometry"] = run.geometry
        t1["k"] = run.k
        t1["run"] = run.run_dir.name
        all_task1.append(t1)

        print(f"[Run {idx}/{len(runs)}] task2 sampling {args.anchors_task2} anchors...")
        t2 = sample_distances_task2(
            model=model,
            geometry=run.geometry,
            as_emb=as_emb,
            rank_bins=rank_bins,
            n_anchors=args.anchors_task2,
            nn_batch=args.nn_batch,
        )
        t2["geometry"] = run.geometry
        t2["k"] = run.k
        t2["run"] = run.run_dir.name
        all_task2.append(t2)

        print(f"[Run {idx}/{len(runs)}] task3 sampling {args.pairs_task3} pairs within hop<3 AS set...")
        t3 = sample_distances_task3(
            model=model,
            geometry=run.geometry,
            as_emb=as_emb,
            aids_top3=aids_top3,
            n_pairs=args.pairs_task3,
            pair_batch=args.pair_batch,
        )
        t3["geometry"] = run.geometry
        t3["k"] = run.k
        t3["run"] = run.run_dir.name
        t3["prefix"] = prefix_str
        t3["prefix_id"] = prefix_id
        t3["profile_id"] = profile_id
        all_task3.append(t3)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df1 = pd.concat(all_task1, ignore_index=True)
    df2 = pd.concat(all_task2, ignore_index=True)
    df3 = pd.concat(all_task3, ignore_index=True)

    plot_task12(
        df1,
        title="Task1: Random AS Pair Semantic Distance by Rank (Euclidean vs Hyperbolic)",
        output_path=output_dir / "task1_as_pair_distance_boxplot.png",
    )
    plot_task12(
        df2,
        title="Task2: Nearest-AS Semantic Distance by Rank (Euclidean vs Hyperbolic)",
        output_path=output_dir / "task2_nearest_as_distance_boxplot.png",
    )
    plot_task3(
        df3,
        title=f"Task3: Prefix Top-3-Layer AS Semantic Distance (prefix={prefix_str})",
        output_path=output_dir / "task3_prefix_top3_distance_boxplot.png",
    )

    summary1 = summarize(df1, ["k", "geometry", "rank_bin"])
    summary2 = summarize(df2, ["k", "geometry", "rank_bin"])
    summary3 = summarize(df3, ["k", "geometry"])

    summary1.to_csv(output_dir / "task1_summary.csv", index=False)
    summary2.to_csv(output_dir / "task2_summary.csv", index=False)
    summary3.to_csv(output_dir / "task3_summary.csv", index=False)

    run_meta = {
        "dataset_root": str(data_root),
        "data_dir": str(processed_root),
        "runs_dir": str(exp_root),
        "as_info": str((args.as_info or data_root / "as_info.jsonl").resolve()),
        "num_runs": len(runs),
        "seed": args.seed,
        "device": str(device),
        "pairs_task1": args.pairs_task1,
        "anchors_task2": args.anchors_task2,
        "pairs_task3": args.pairs_task3,
        "pair_batch": args.pair_batch,
        "nn_batch": args.nn_batch,
        "rank_source": rank_source,
        "rank_rows_from_file": rank_rows,
        "rank_bin_scheme": ["0-500", "501-1500", "1501-5000", "5001-15000", "15000+"],
        "selected_prefix": prefix_str,
        "selected_prefix_id": prefix_id,
        "selected_profile_id": profile_id,
        "selected_prefix_top3_as_count": int(aids_top3.numel()),
        "runs": [
            {
                "geometry": r.geometry,
                "k": r.k,
                "run_dir": str(r.run_dir),
                "manifest_path": str(r.manifest_path),
                "model_path": str(r.model_path),
            }
            for r in runs
        ],
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("[Done] Outputs:")
    print(f"  - {output_dir / 'task1_as_pair_distance_boxplot.png'}")
    print(f"  - {output_dir / 'task2_nearest_as_distance_boxplot.png'}")
    print(f"  - {output_dir / 'task3_prefix_top3_distance_boxplot.png'}")
    print(f"  - {output_dir / 'task1_summary.csv'}")
    print(f"  - {output_dir / 'task2_summary.csv'}")
    print(f"  - {output_dir / 'task3_summary.csv'}")
    print(f"  - {output_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
