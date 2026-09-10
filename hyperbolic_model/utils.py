import torch

def unpack_128bit_to_binary(tensor_n_2):
    """
    Helper: Unpack (N, 2) Int64 tensors (128-bit SimHash) into (N, 128) binary float vectors.
    """
    device = tensor_n_2.device
    # Prepare mask: 2^63, ..., 2^0
    mask = 1 << torch.arange(63, -1, -1, device=device)
    
    # Process high and low 64 bits separately
    high = tensor_n_2[:, 0].unsqueeze(1)
    low  = tensor_n_2[:, 1].unsqueeze(1)
    
    # Bitwise AND to extract bits -> Convert to float 0.0/1.0
    high_bits = (high.bitwise_and(mask) != 0).float()
    low_bits  = (low.bitwise_and(mask)  != 0).float()
    
    # Concatenate to (N, 128)
    return torch.cat([high_bits, low_bits], dim=1)

def recalibrate_affinity(raw_sim, baseline=0.50):
    """    
    Logic:
    - Raw ~0.50 (Noise)      -> Mapped ~0.00 (Ignorable)
    - Raw ~0.70 (Weak Rel)   -> Mapped ~0.40 (Weak Pull)
    - Raw ~0.90 (Strong Rel) -> Mapped ~0.80 (Strong Pull)
    """
    val = (raw_sim - baseline) / (1.0 - baseline)
    return torch.clamp(val, min=0.0, max=1.0)

def compute_affinity(fp_a, fp_b):
    """
    Compute SimHash Affinity (Normalized Hamming Similarity).
    fp: (Batch, L) float tensor
    Returns: affinity in [0, 1]
    """
    L = fp_a.size(-1)
    # L1 distance on 0/1 tensors is equivalent to Hamming distance
    hamming_dist = torch.sum(torch.abs(fp_a - fp_b), dim=-1)
    
    # Similarity = 1 - (Hamming / L)
    similarity = 1.0 - (hamming_dist / L)
    
    # Clamp to ensure numerical stability
    return torch.clamp(similarity, 0.0, 1.0)
