import torch
import geoopt

from .base import BORDER_VALUE, RoutingModelBase


class HyperbolicRoutingModel(RoutingModelBase):
    def __init__(self, num_as: int, num_profiles: int, embed_dim: int = 24, curvature: float = 1.0, num_vectors: int = 2):
        super().__init__(num_as, num_profiles, embed_dim, num_vectors)
        self.c = torch.tensor([curvature])
        self.manifold = geoopt.PoincareBall(c=self.c)

        self.as_embedding = geoopt.ManifoldParameter(
            self.manifold.random(num_as, embed_dim) * 0.5,
            manifold=self.manifold,
        )
        self.profile_embedding = geoopt.ManifoldParameter(
            self.manifold.random(num_profiles, num_vectors, embed_dim) * 0.01,
            manifold=self.manifold,
        )

    def pairwise_profile_as_dist(self, p_indices: torch.Tensor, as_indices: torch.Tensor) -> torch.Tensor:
        p_emb = self.profile_embedding[p_indices]
        a_emb = self.as_embedding[as_indices]
        dists = self.manifold.dist(p_emb.unsqueeze(2), a_emb.unsqueeze(1))
        return torch.clamp(dists, max=BORDER_VALUE)

    def compute_profile_chamfer_distance(self, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        emb_a = self.profile_embedding[idx_a]
        emb_b = self.profile_embedding[idx_b]
        dists = self.manifold.dist(emb_a.unsqueeze(2), emb_b.unsqueeze(1))
        dists = torch.clamp(dists, max=BORDER_VALUE)

        min_a, _ = torch.min(dists, dim=2)
        min_b, _ = torch.min(dists, dim=1)
        return (torch.mean(min_a, dim=1) + torch.mean(min_b, dim=1)) / 2.0

    def get_diversity_reg(self, p_indices: torch.Tensor) -> torch.Tensor:
        if self.num_vectors <= 1:
            return torch.tensor(0.0, device=p_indices.device)

        p_emb = self.profile_embedding[p_indices]
        if self.num_vectors == 2:
            dist = self.manifold.dist(p_emb[:, 0], p_emb[:, 1])
        else:
            dists = self.manifold.dist(p_emb.unsqueeze(2), p_emb.unsqueeze(1))
            mask = ~torch.eye(self.num_vectors, dtype=torch.bool, device=p_indices.device)
            dist = dists[:, mask]

        dist = torch.clamp(dist, max=BORDER_VALUE)
        return torch.mean(torch.exp(-dist))
