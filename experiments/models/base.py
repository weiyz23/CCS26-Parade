from abc import ABC, abstractmethod

import torch
import torch.nn as nn


BORDER_VALUE = 13.0


class RoutingModelBase(nn.Module, ABC):
    def __init__(self, num_as: int, num_profiles: int, embed_dim: int, num_vectors: int):
        super().__init__()
        self.num_as = num_as
        self.num_profiles = num_profiles
        self.embed_dim = embed_dim
        self.num_vectors = num_vectors

    @abstractmethod
    def pairwise_profile_as_dist(self, p_indices: torch.Tensor, as_indices: torch.Tensor) -> torch.Tensor:
        """Return distances with shape (B, K, M)."""

    @abstractmethod
    def compute_profile_chamfer_distance(self, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        """Return profile-to-profile distance with shape (B,)."""

    @abstractmethod
    def get_diversity_reg(self, p_indices: torch.Tensor) -> torch.Tensor:
        """Return scalar regularizer to avoid multi-head collapse."""

    def forward(self, p_indices: torch.Tensor, as_indices: torch.Tensor) -> torch.Tensor:
        return self.pairwise_profile_as_dist(p_indices, as_indices)
