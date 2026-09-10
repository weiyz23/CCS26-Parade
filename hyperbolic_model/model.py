"""
Hyperbolic BGP Embedding Model & Loss Definitions.
Includes:
- HyperbolicRoutingModel: Manifold, Embeddings, Distance Calcs
- HyperbolicLoss: Hierarchical (Ranking) & Propagation (Similarity) Losses
"""

import torch
import torch.nn as nn
import geoopt

BORDER_VALUE = 13.0
LEAK_PROB = 0.20

class HyperbolicRoutingModel(nn.Module):
    def __init__(self, num_as, num_profiles, embed_dim=24, curvature=1.0, num_vectors=2):
        super().__init__()
        self.num_as = num_as
        self.num_profiles = num_profiles
        self.num_vectors = num_vectors
        self.c = torch.tensor([curvature])
        
        # 1. Define Manifold: Poincaré Ball
        self.manifold = geoopt.PoincareBall(c=self.c)

        # 2. Define Embeddings
        # AS Embedding (Leaf Nodes): Initialize further from origin (0.5) to encourage boundary span
        self.as_embedding = geoopt.ManifoldParameter(
            self.manifold.random(num_as, embed_dim) * 0.5, 
            manifold=self.manifold
        )
        # Profile Embedding (Root Nodes): Initialize near origin (0.01)
        # Using Multi-vector representation (Set of K vectors)
        self.profile_embedding = geoopt.ManifoldParameter(
            self.manifold.random(num_profiles, num_vectors, embed_dim) * 0.01, 
            manifold=self.manifold
        )

    def forward(self, p_indices, as_indices):
        """
        Standard Forward for AS-Prefix Affinity (Distance Calculation).
        Uses Min-Pooling (Winner-Takes-All) to select the closest vector in the profile set.
        
        Args:
            p_indices: (B)
            as_indices: (B, M)
        Returns:
            min_dists: (B, M) Distance to the closest vector for each AS sample
        """
        # (B, K, D)
        p_emb = self.profile_embedding[p_indices]
        # (B, M, D)
        a_emb = self.as_embedding[as_indices]
        
        # Broadcast for pairwise dist: (B, K, 1, D) vs (B, 1, M, D)
        p_exp = p_emb.unsqueeze(2)
        a_exp = a_emb.unsqueeze(1)
        
        dists = self.manifold.dist(p_exp, a_exp)
        # Clamp distances to avoid numerical issues
        dists = torch.clamp(dists, max=BORDER_VALUE)
        
        # (B, M, K) -> Return all the distances for further processing
        return dists

    def compute_profile_chamfer_distance(self, idx_a, idx_b):
        """
        Compute Hyperbolic Chamfer Distance between two sets of Profile vectors.
        Used for Prefix-Prefix Affinity Loss.
        
        Args:
            idx_a: (Batch_Size) Indices of first set of profiles
            idx_b: (Batch_Size) Indices of second set of profiles
        Returns:
            chamfer_dist: (Batch_Size) Bidirectional Chamfer Distance
        """
        # (Batch, K, Dim)
        emb_a = self.profile_embedding[idx_a]
        emb_b = self.profile_embedding[idx_b]
        
        # Pairwise Distance Matrix: (Batch, K, K)
        # emb_a: (B, K, 1, D) vs emb_b: (B, 1, K, D)
        dists = self.manifold.dist(emb_a.unsqueeze(2), emb_b.unsqueeze(1))
        dists = torch.clamp(dists, max=BORDER_VALUE)
        
        # Chamfer Distance: Average of symmetric min distances
        # For each vec in A, find closest in B
        min_a, _ = torch.min(dists, dim=2) # (Batch, K)
        # For each vec in B, find closest in A
        min_b, _ = torch.min(dists, dim=1) # (Batch, K)
        
        # Average over the set size K
        term1 = torch.mean(min_a, dim=1)
        term2 = torch.mean(min_b, dim=1)
        
        return (term1 + term2) / 2.0

    def get_diversity_reg(self, p_indices):
        """
        Intra-Profile Diversity Regularization.
        Prevents the K vectors of a single profile from collapsing into a single point.
        
        Args:
            p_indices: (Batch_Size)
        Returns:
            reg_val: Scalar regularization value
        """
        if self.num_vectors <= 1: 
            return torch.tensor(0.0, device=p_indices.device)
        
        p_emb = self.profile_embedding[p_indices] # (B, K, D)
        
        if self.num_vectors == 2:
            # Maximize distance between the pair => Minimize exp(-dist)
            dist = self.manifold.dist(p_emb[:, 0], p_emb[:, 1])
        else:
            # Pairwise distance for K > 2
            p_exp = p_emb.unsqueeze(2)
            p_tr = p_emb.unsqueeze(1)
            dists = self.manifold.dist(p_exp, p_tr)
            # Mask diagonal
            mask = ~torch.eye(self.num_vectors, dtype=torch.bool, device=p_indices.device)
            dist = dists[:, mask]
            
        dist = torch.clamp(dist, max=BORDER_VALUE)
        # Regularization: minimize exp(-dist)
        return torch.mean(torch.exp(-dist))


class HyperbolicLoss(nn.Module):
    """
    Composite Loss Function for PARADE Framework.
    """
    def __init__(self, hier_margin=1.0, rank_weight=0.2, sim_margin=4.0):
        super().__init__()
        self.hier_margin = hier_margin
        self.rank_weight = rank_weight
        self.sim_margin = sim_margin
        self.relu = nn.ReLU()

    def forward_hierarchical(self, d_pos, d_neg, pos_hops, sample_weights=None):
        """
        Hierarchical Affinity Loss with Stochastic Masking for Multi-Head Specialization.
        Includes Margin Loss (Pos vs Neg) and Ranking Loss (Pos vs Pos by Hops)
        """
        B, K, M_pos = d_pos.shape
        # --- 1. Check head ownership and create masks ---
        min_vals, min_indices = torch.min(d_pos, dim=1)
        ownership_mask = torch.zeros_like(d_pos, dtype=torch.bool)
        # (B, 1, M_pos)
        ownership_mask.scatter_(1, min_indices.unsqueeze(1), 1)
        
        # --- 2. Stochastic Leakage ---
        if self.training:
            random_mask = torch.bernoulli(torch.full_like(d_pos, LEAK_PROB)).bool()
            final_mask = ownership_mask | random_mask
        else:
            final_mask = ownership_mask

        # --- 3. Compute Basic Margin Loss with Masking ---
        d_p = d_pos.unsqueeze(-1)
        d_n = d_neg.unsqueeze(2)
        
        # Raw Loss: (B, K, M_pos, M_neg)
        raw_loss = self.relu(d_p - d_n + self.hier_margin)
        per_sample_loss = torch.sum(raw_loss, dim=-1)
        
        # --- 4. Apply Mask for Basic Loss ---
        mask_float = final_mask.float()
        if sample_weights is not None:
            # Normalize per-batch weights to keep gradient scale stable.
            norm_weights = sample_weights / (sample_weights.mean() + 1e-9)
            weight_matrix = norm_weights.unsqueeze(1)
            weighted_mask = mask_float * weight_matrix
        else:
            weighted_mask = mask_float

        masked_loss = per_sample_loss * weighted_mask
        valid_counts = weighted_mask.sum() + 1e-9
        basic_loss = masked_loss.sum() / valid_counts

        # --- 5. Apply Masking for Ranking Loss ---
        rank_loss_val = torch.tensor(0.0, device=d_pos.device)
        if self.rank_weight > 0:
            d_i = min_vals.unsqueeze(2) # (B, M, 1)
            d_j = min_vals.unsqueeze(1) # (B, 1, M)
            
            h_i = pos_hops.view(B, M_pos, 1)
            h_j = pos_hops.view(B, 1, M_pos)           
            
            # Rank Mask: Hop i < Hop j
            # (B, 1, M, M)
            hop_mask = (h_i < h_j).float()
            
            rank_penalty = self.relu(d_i - d_j + 0.1 * self.hier_margin)
            masked_rank_loss = rank_penalty * hop_mask
            valid_pairs = hop_mask.sum() + 1e-9
            rank_loss_val = masked_rank_loss.sum() / valid_pairs

        return basic_loss + self.rank_weight * rank_loss_val
    

    def forward_similarity(self, dist_p2p, affinity):
        """
        Propagation Affinity Loss with soft contrastive loss based on similarity.
        
        Args:
            dist_p2p: (Batch) Hyperbolic Chamfer Distance
            affinity: (Batch) Similarity score [0, 1] (1 = identical)
        """
        # Weighting:
        # If affinity=1 (Similar) -> pull_weight=1, push_weight=0 -> Minimize dist
        # If affinity=0 (Dissimilar) -> pull_weight=0, push_weight=1 -> Push to margin
        pull_weight = affinity
        push_weight = 1.0 - affinity
        
        # 1. Pull Term
        loss_pull = pull_weight * torch.pow(dist_p2p, 2)
        
        # 2. Push Term (Hinge Loss)
        dist_hinge = self.relu(self.sim_margin - dist_p2p)
        loss_push = push_weight * torch.pow(dist_hinge, 2)
        
        return torch.mean(loss_pull + loss_push)
