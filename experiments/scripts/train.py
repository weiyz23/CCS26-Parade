import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.core.config import build_arg_parser, load_json_config, override_from_cli
from experiments.core.data import CompactSampler, load_training_artifacts
from experiments.core.similarity import compute_affinity, recalibrate_affinity
from experiments.losses.parade_loss import ParadeLoss
from experiments.models.factory import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: torch.nn.Module, geometry: str, lr: float, weight_decay: float):
    if geometry == "hyperbolic":
        import geoopt

        return geoopt.optim.RiemannianAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def run_training(cfg: dict) -> None:
    process_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    # Record the effective settings after all command-line overrides.
    with (model_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    artifacts = load_training_artifacts(data_dir=data_dir, device=device)
    meta = artifacts["meta"]
    triplets = artifacts["triplets"]
    fingerprints = artifacts["fingerprints"]

    sampler = CompactSampler(triplets, meta["num_profiles"], device=device)
    valid_profiles = torch.nonzero(sampler.counts > 0).squeeze(1)
    num_train_profiles = valid_profiles.size(0)
    if num_train_profiles == 0:
        raise ValueError("Training data contains no profile-AS samples")

    model = build_model(
        geometry=cfg["model"]["geometry"],
        num_as=meta["num_asns"],
        num_profiles=meta["num_profiles"],
        embed_dim=cfg["model"]["embed_dim"],
        num_vectors=cfg["model"]["num_vectors"],
    ).to(device)

    optimizer = build_optimizer(
        model=model,
        geometry=cfg["model"]["geometry"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    loss_fn = ParadeLoss(
        hier_margin=cfg["loss"]["hier_margin"],
        rank_weight=cfg["loss"]["ranking_weight"],
        sim_margin=cfg["loss"]["sim_margin"],
    )

    best_loss = float("inf")
    patience_counter = 0
    early_stop_mark_epoch = None
    early_stop_mark_cum_time_sec = None

    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]
    micro_batch_size = int(cfg["train"].get("micro_batch_size", batch_size))
    if micro_batch_size <= 0 or micro_batch_size > batch_size:
        raise ValueError("micro_batch_size must be in (0, batch_size]")
    gradient_accumulation_steps = (batch_size + micro_batch_size - 1) // micro_batch_size
    m = cfg["train"]["num_samples_per_profile"]
    sim_weight = cfg["loss"]["sim_weight"]
    diversity_weight = cfg["loss"]["diversity_weight"]
    mark_only_early_stop = cfg["train"].get("mark_only_early_stop", True)

    epoch_records = []
    total_train_start = time.time()
    cumulative_time_sec = 0.0

    print(
        "Start Training | "
        f"geometry={cfg['model']['geometry']} "
        f"K={cfg['model']['num_vectors']} "
        f"dim={cfg['model']['embed_dim']}"
    )

    for epoch in range(epochs):
        model.train()
        loss_fn.train()
        total_loss = 0.0
        total_hier = 0.0
        total_sim = 0.0
        total_div = 0.0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        epoch_start = time.time()

        perm = torch.randperm(num_train_profiles, device=device)
        shuffled_indices = valid_profiles[perm]
        num_batches = (num_train_profiles + batch_size - 1) // batch_size

        progress = tqdm(range(num_batches), desc=f"Ep {epoch + 1}")
        for i in progress:
            start = i * batch_size
            end = min(start + batch_size, num_train_profiles)

            p_idxs_a_full = shuffled_indices[start:end]
            curr_full_bs = p_idxs_a_full.size(0)
            p_idxs_b_full = p_idxs_a_full[torch.randperm(curr_full_bs, device=device)]
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_hier = 0.0
            batch_sim = 0.0
            batch_div = 0.0
            for micro_start in range(0, curr_full_bs, micro_batch_size):
                micro_stop = min(micro_start + micro_batch_size, curr_full_bs)
                p_idxs_a = p_idxs_a_full[micro_start:micro_stop]
                p_idxs_b = p_idxs_b_full[micro_start:micro_stop]
                curr_bs = p_idxs_a.size(0)

                pos_as, pos_hops, pos_weights = sampler.sample(p_idxs_a, num_samples=m)
                neg_as = torch.randint(0, meta["num_asns"], (curr_bs, 3 * m), device=device)

                d_pos = model(p_idxs_a, pos_as)
                d_neg = model(p_idxs_a, neg_as)
                loss_hier = loss_fn.forward_hierarchical(
                    d_pos, d_neg, pos_hops, sample_weights=pos_weights
                )

                fp_a = fingerprints[p_idxs_a]
                fp_b = fingerprints[p_idxs_b]
                affinity = recalibrate_affinity(compute_affinity(fp_a, fp_b))

                dist_p2p = model.compute_profile_chamfer_distance(p_idxs_a, p_idxs_b)
                loss_sim = loss_fn.forward_similarity(dist_p2p, affinity)

                loss_div = torch.tensor(0.0, device=device)
                if diversity_weight > 0 and cfg["model"]["num_vectors"] > 1:
                    loss_div = model.get_diversity_reg(p_idxs_a)

                loss = loss_hier + sim_weight * loss_sim + diversity_weight * loss_div
                micro_weight = curr_bs / curr_full_bs
                (loss * micro_weight).backward()
                batch_loss += loss.item() * micro_weight
                batch_hier += loss_hier.item() * micro_weight
                batch_sim += loss_sim.item() * micro_weight
                batch_div += loss_div.item() * micro_weight
            optimizer.step()

            total_loss += batch_loss
            total_hier += batch_hier
            total_sim += batch_sim
            total_div += batch_div
            progress.set_postfix({"L": f"{batch_loss:.3f}", "H": f"{batch_hier:.3f}", "S": f"{batch_sim:.3f}"})

        avg_loss = total_loss / max(1, num_batches)
        avg_hier = total_hier / max(1, num_batches)
        avg_sim = total_sim / max(1, num_batches)
        avg_div = total_div / max(1, num_batches)

        epoch_time_sec = time.time() - epoch_start
        cumulative_time_sec += epoch_time_sec

        if device.type == "cuda":
            current_alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            current_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
            peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        else:
            current_alloc_mb = None
            current_reserved_mb = None
            peak_alloc_mb = None
            peak_reserved_mb = None

        scheduler.step(avg_loss)
        print(
            f"Ep {epoch + 1}: avg_loss={avg_loss:.4f} "
            f"avg_hier={avg_hier:.4f} avg_sim={avg_sim:.4f} avg_div={avg_div:.4f} "
            f"time={epoch_time_sec:.2f}s"
        )

        if avg_loss < best_loss - cfg["train"]["delta"]:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "best_model.pth")
        else:
            patience_counter += 1

        would_early_stop = patience_counter >= cfg["train"]["patience"]
        if would_early_stop and early_stop_mark_epoch is None:
            early_stop_mark_epoch = epoch + 1
            early_stop_mark_cum_time_sec = cumulative_time_sec
            print(f"Early stop condition first met at epoch {early_stop_mark_epoch} (mark only)")

        epoch_records.append(
            {
                "epoch": epoch + 1,
                "avg_total_loss": avg_loss,
                "avg_hier_loss": avg_hier,
                "avg_sim_loss": avg_sim,
                "avg_div_loss": avg_div,
                "epoch_time_sec": epoch_time_sec,
                "cumulative_time_sec": cumulative_time_sec,
                "process_cumulative_time_sec": time.time() - process_start,
                "early_stop_condition_met": would_early_stop,
                "cuda_mem_current_alloc_mb": current_alloc_mb,
                "cuda_mem_current_reserved_mb": current_reserved_mb,
                "cuda_mem_peak_alloc_mb": peak_alloc_mb,
                "cuda_mem_peak_reserved_mb": peak_reserved_mb,
            }
        )

        if would_early_stop and (not mark_only_early_stop):
            print("Early stopping triggered")
            break

    torch.save(model.state_dict(), model_dir / "final_model.pth")

    total_train_time_sec = time.time() - total_train_start

    peak_alloc_series = [r["cuda_mem_peak_alloc_mb"] for r in epoch_records if r["cuda_mem_peak_alloc_mb"] is not None]
    peak_reserved_series = [r["cuda_mem_peak_reserved_mb"] for r in epoch_records if r["cuda_mem_peak_reserved_mb"] is not None]

    train_metrics = {
        "seed": seed,
        "geometry": cfg["model"]["geometry"],
        "embed_dim": cfg["model"]["embed_dim"],
        "num_vectors": cfg["model"]["num_vectors"],
        "effective_batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "configured_epochs": epochs,
        "mark_only_early_stop": mark_only_early_stop,
        "best_loss": best_loss,
        "early_stop_mark_epoch": early_stop_mark_epoch,
        "early_stop_mark_cum_time_sec": early_stop_mark_cum_time_sec,
        "total_train_time_sec": total_train_time_sec,
        "total_process_time_sec": time.time() - process_start,
        "max_cuda_peak_alloc_mb": max(peak_alloc_series) if peak_alloc_series else None,
        "max_cuda_peak_reserved_mb": max(peak_reserved_series) if peak_reserved_series else None,
        "epoch_records": epoch_records,
    }

    with open(model_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)

    with open(model_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "geometry": cfg["model"]["geometry"],
                "embed_dim": cfg["model"]["embed_dim"],
                "num_vectors": cfg["model"]["num_vectors"],
                "effective_batch_size": batch_size,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "best_loss": best_loss,
                "data_dir": str(data_dir),
                "configured_epochs": epochs,
                "mark_only_early_stop": mark_only_early_stop,
                "early_stop_mark_epoch": early_stop_mark_epoch,
                "early_stop_mark_cum_time_sec": early_stop_mark_cum_time_sec,
                "total_train_time_sec": total_train_time_sec,
                "train_metrics_file": str(model_dir / "train_metrics.json"),
            },
            f,
            indent=2,
        )

    print(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_json_config(args.config)
    config = override_from_cli(config, args)
    run_training(config)
