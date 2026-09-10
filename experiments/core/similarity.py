import torch


def unpack_128bit_to_binary(tensor_n_2: torch.Tensor) -> torch.Tensor:
    """Unpack (N, 2) int64 tensors (128-bit SimHash) into (N, 128) binary float vectors."""
    device = tensor_n_2.device
    mask = 1 << torch.arange(63, -1, -1, device=device)
    high = tensor_n_2[:, 0].unsqueeze(1)
    low = tensor_n_2[:, 1].unsqueeze(1)
    high_bits = (high.bitwise_and(mask) != 0).float()
    low_bits = (low.bitwise_and(mask) != 0).float()
    return torch.cat([high_bits, low_bits], dim=1)


def compute_affinity(fp_a: torch.Tensor, fp_b: torch.Tensor) -> torch.Tensor:
    """Compute normalized Hamming similarity in [0, 1]."""
    hamming_dist = torch.sum(torch.abs(fp_a - fp_b), dim=-1)
    length = fp_a.size(-1)
    similarity = 1.0 - (hamming_dist / length)
    return torch.clamp(similarity, 0.0, 1.0)


def recalibrate_affinity(raw_sim: torch.Tensor, baseline: float = 0.50) -> torch.Tensor:
    """Re-scale raw similarity so near-baseline noise maps close to zero."""
    val = (raw_sim - baseline) / (1.0 - baseline)
    return torch.clamp(val, min=0.0, max=1.0)
