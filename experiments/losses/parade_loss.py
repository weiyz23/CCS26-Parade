import torch
import torch.nn as nn


LEAK_PROB = 0.20


class ParadeLoss(nn.Module):
    """Geometry-agnostic PARADE loss used by both hyperbolic and euclidean models."""

    def __init__(self, hier_margin: float = 1.0, rank_weight: float = 0.2, sim_margin: float = 4.0):
        super().__init__()
        self.hier_margin = hier_margin
        self.rank_weight = rank_weight
        self.sim_margin = sim_margin
        self.relu = nn.ReLU()

    def forward_hierarchical(
        self,
        d_pos: torch.Tensor,
        d_neg: torch.Tensor,
        pos_hops: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, _, m_pos = d_pos.shape
        min_vals, min_indices = torch.min(d_pos, dim=1)

        ownership_mask = torch.zeros_like(d_pos, dtype=torch.bool)
        ownership_mask.scatter_(1, min_indices.unsqueeze(1), 1)

        if self.training:
            random_mask = torch.bernoulli(torch.full_like(d_pos, LEAK_PROB)).bool()
            final_mask = ownership_mask | random_mask
        else:
            final_mask = ownership_mask

        raw_loss = self.relu(d_pos.unsqueeze(-1) - d_neg.unsqueeze(2) + self.hier_margin)
        per_sample_loss = torch.sum(raw_loss, dim=-1)

        mask_float = final_mask.float()
        if sample_weights is not None:
            norm_weights = sample_weights / (sample_weights.mean() + 1e-9)
            weighted_mask = mask_float * norm_weights.unsqueeze(1)
        else:
            weighted_mask = mask_float

        masked_loss = per_sample_loss * weighted_mask
        basic_loss = masked_loss.sum() / (weighted_mask.sum() + 1e-9)

        rank_loss_val = torch.tensor(0.0, device=d_pos.device)
        if self.rank_weight > 0:
            d_i = min_vals.unsqueeze(2)
            d_j = min_vals.unsqueeze(1)
            h_i = pos_hops.view(bsz, m_pos, 1)
            h_j = pos_hops.view(bsz, 1, m_pos)
            hop_mask = (h_i < h_j).float()
            rank_penalty = self.relu(d_i - d_j + 0.1 * self.hier_margin)
            rank_loss_val = (rank_penalty * hop_mask).sum() / (hop_mask.sum() + 1e-9)

        return basic_loss + self.rank_weight * rank_loss_val

    def forward_similarity(self, dist_p2p: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
        pull_weight = affinity
        push_weight = 1.0 - affinity

        loss_pull = pull_weight * torch.pow(dist_p2p, 2)
        dist_hinge = self.relu(self.sim_margin - dist_p2p)
        loss_push = push_weight * torch.pow(dist_hinge, 2)

        return torch.mean(loss_pull + loss_push)
