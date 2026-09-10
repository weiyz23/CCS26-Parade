import gc
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

from .similarity import unpack_128bit_to_binary


class CompactSampler:
    """GPU sampler for profile->AS triplets with compressed indexing."""

    def __init__(self, triplets_tensor: torch.Tensor, num_profiles: int, device: torch.device):
        t0 = time.time()
        sorted_indices = torch.argsort(triplets_tensor[:, 0])
        sorted_triplets = triplets_tensor[sorted_indices]

        profile_ids = sorted_triplets[:, 0].to(dtype=torch.long, device="cpu")
        as_ids = sorted_triplets[:, 1].to(dtype=torch.long, device="cpu")
        hop_values = sorted_triplets[:, 2].to(dtype=torch.float32, device="cpu")

        self.val_as = as_ids.to(device=device, dtype=torch.long)
        self.val_hops = hop_values.clamp(min=0, max=255).to(device=device, dtype=torch.uint8)
        if sorted_triplets.shape[1] >= 4:
            self.val_weights = sorted_triplets[:, 3].to(device=device, dtype=torch.float32)
        else:
            self.val_weights = torch.ones_like(self.val_hops, dtype=torch.float32, device=device)

        full_counts = torch.bincount(profile_ids, minlength=num_profiles)

        self.counts = full_counts.to(device=device, dtype=torch.long)
        self.starts = (torch.cumsum(full_counts, dim=0) - full_counts).to(device=device, dtype=torch.long)
        self.device = device

        del sorted_indices, sorted_triplets, profile_ids, as_ids, hop_values, full_counts
        gc.collect()
        print(f"Sampler initialized in {time.time() - t0:.2f}s")

    def sample(self, batch_profile_indices: torch.Tensor, num_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = batch_profile_indices.size(0)
        starts = self.starts[batch_profile_indices]
        counts = self.counts[batch_profile_indices]
        safe_counts = torch.clamp(counts, min=1).unsqueeze(1)

        rand_offsets = torch.randint(0, 1_000_000, (bsz, num_samples), device=self.device) % safe_counts
        global_indices = starts.unsqueeze(1) + rand_offsets
        flat_indices = global_indices.view(-1)

        sampled_as = self.val_as[flat_indices].view(bsz, num_samples)
        sampled_hops = self.val_hops[flat_indices].view(bsz, num_samples).float()
        sampled_weights = self.val_weights[flat_indices].view(bsz, num_samples)
        return sampled_as, sampled_hops, sampled_weights


def load_training_artifacts(data_dir: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    fp_path = data_dir / "profile_fingerprints.pt"
    if not fp_path.exists():
        raise FileNotFoundError(f"Missing {fp_path}")

    raw_fp = torch.load(fp_path, map_location=device)
    fingerprints = unpack_128bit_to_binary(raw_fp.to(device))

    triplets_path = data_dir / "profile_triplets.pt"
    if not triplets_path.exists():
        raise FileNotFoundError(f"Missing {triplets_path}")
    triplets = torch.load(triplets_path, map_location="cpu")

    return {
        "meta": meta,
        "triplets": triplets,
        "fingerprints": fingerprints,
    }
