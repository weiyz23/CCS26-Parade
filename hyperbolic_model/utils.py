import torch

FINGERPRINT_SIMILARITY_THRESHOLD = 0.75

@torch.no_grad()
def profile_as_distances(model, profile_ids, as_ids, *, batch_size=8192):
    """Return unscaled minimum-head distances on the model's device for paired IDs."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if profile_ids.ndim != 1 or as_ids.ndim != 1 or profile_ids.shape != as_ids.shape:
        raise ValueError("profile_ids and as_ids must have matching shape (B,)")
    device = model.as_embedding.device
    profile_ids = profile_ids.to(device=device, dtype=torch.long)
    as_ids = as_ids.to(device=device, dtype=torch.long)
    if not len(profile_ids):
        return model.as_embedding.new_empty((0,))
    distances = []
    for start in range(0, len(profile_ids), batch_size):
        b_p = profile_ids[start:start + batch_size]
        b_a = as_ids[start:start + batch_size]
        distances.append(model(b_p, b_a.unsqueeze(1)).squeeze(2).min(dim=1)[0])
    return torch.cat(distances)

def unpack_128bit_to_binary(tensor_n_2):
    """Unpack two int64 words, most-significant bit first, into 128 binary floats."""
    device = tensor_n_2.device
    mask = 1 << torch.arange(63, -1, -1, device=device)

    high = tensor_n_2[:, 0].unsqueeze(1)
    low  = tensor_n_2[:, 1].unsqueeze(1)

    high_bits = (high.bitwise_and(mask) != 0).float()
    low_bits  = (low.bitwise_and(mask)  != 0).float()

    return torch.cat([high_bits, low_bits], dim=1)

def recalibrate_affinity(raw_sim, baseline=0.50):
    """Map the baseline similarity to zero and clamp to [0, 1]."""
    val = (raw_sim - baseline) / (1.0 - baseline)
    return torch.clamp(val, min=0.0, max=1.0)

def compute_affinity(fp_a, fp_b):
    """Return one minus normalized Hamming distance for binary fingerprints."""
    L = fp_a.size(-1)
    hamming_dist = torch.sum(torch.abs(fp_a - fp_b), dim=-1)

    similarity = 1.0 - (hamming_dist / L)

    return torch.clamp(similarity, 0.0, 1.0)


@torch.no_grad()
def fingerprint_matches(profile_fps, packed_anchors, *, batch_size=8192):
    """Match packed AS anchors at raw 128-bit Hamming similarity strictly above 0.75."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if profile_fps.ndim == 1:
        profile_fps = profile_fps.unsqueeze(0)
    if profile_fps.ndim != 2 or profile_fps.shape[1] != 128:
        raise ValueError("profile_fps must have shape (B, 128)")
    if packed_anchors.ndim != 2 or packed_anchors.shape[1] != 2:
        raise ValueError("packed_anchors must have shape (M, 2)")

    matches = torch.zeros(len(profile_fps), dtype=torch.bool, device=profile_fps.device)
    if not len(profile_fps) or not len(packed_anchors):
        return matches

    # Signed and unsigned words contain the same bits. Normalize before unpacking.
    anchors = unpack_128bit_to_binary(packed_anchors.to(device=profile_fps.device, dtype=torch.int64))
    profiles = profile_fps.to(dtype=torch.float32)
    # Bound the profile-by-anchor broadcast.
    query_batch_size = min(batch_size, max(1, 65536 // len(anchors)))
    for start in range(0, len(profiles), query_batch_size):
        affinities = compute_affinity(profiles[start:start + query_batch_size].unsqueeze(1), anchors.unsqueeze(0))
        matches[start:start + query_batch_size] = (affinities > FINGERPRINT_SIMILARITY_THRESHOLD).any(dim=1)
    return matches
