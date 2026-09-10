"""Training script for PARADE Framework."""

import torch
import torch.nn as nn
import geoopt
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import random
import numpy as np
# Import both Model and Loss from model.py
from model import HyperbolicRoutingModel, HyperbolicLoss
from utils import unpack_128bit_to_binary, recalibrate_affinity, compute_affinity

class CompactSampler:
    """
    GPU-Accelerated Sampler using Compressed Indexing.
    Efficiently samples neighbors for a batch of profiles directly on GPU.
    """
    def __init__(self, triplets_tensor, num_profiles, device):
        print(f"--- Initializing GPU Sampler (Edges: {len(triplets_tensor):,}) ---")
        t0 = time.time()

        # 1. Sort
        sorted_indices = torch.argsort(triplets_tensor[:, 0])
        sorted_triplets = triplets_tensor[sorted_indices]
        
        # Robust dtype normalization: profile/as ids must be integer tensors.
        profile_ids = sorted_triplets[:, 0].to(dtype=torch.long, device="cpu")
        as_ids = sorted_triplets[:, 1].to(dtype=torch.long, device="cpu")
        hop_values = sorted_triplets[:, 2].to(dtype=torch.float32, device="cpu")

        # 2. Prepare Data (No Padding)
        self.val_as = as_ids.to(device=device, dtype=torch.long)
        self.val_hops = hop_values.clamp(min=0, max=255).to(device=device, dtype=torch.uint8)
        if sorted_triplets.shape[1] >= 4:
            self.val_weights = sorted_triplets[:, 3].to(device=device, dtype=torch.float32)
        else:
            self.val_weights = torch.ones_like(self.val_hops, dtype=torch.float32, device=device)
        
        # 3. Build Index
        full_counts = torch.bincount(profile_ids, minlength=num_profiles)
        
        # --- Print Statistics ---
        max_degree = full_counts.max().item()
        avg_degree = full_counts.float().mean().item()
        zero_neighbors = (full_counts == 0).sum().item()
        
        print(f"--- Graph Statistics ---")
        print(f"Total Profiles : {num_profiles:,}")
        print(f"Max AS/Profile : {max_degree}") 
        print(f"Avg AS/Profile : {avg_degree:.2f}")
        print(f"Zero-Neighbor  : {zero_neighbors}")
        
        # 4. Upload Index to GPU
        self.counts = full_counts.to(device=device, dtype=torch.long)
        self.starts = (torch.cumsum(full_counts, dim=0) - full_counts).to(device=device, dtype=torch.long)
        self.device = device
        
        print(f"Sampler initialized in {time.time()-t0:.2f}s.")
        del sorted_indices, sorted_triplets, profile_ids, as_ids, hop_values, full_counts

    def sample(self, batch_profile_indices, num_samples=4):
        """
        batch_profile_indices: (B,)
        returns:
            sampled_as: (B, M) LongTensor
            sampled_hops: (B, M) FloatTensor
            sampled_weights: (B, M) FloatTensor
        """
        B = batch_profile_indices.size(0)
        M = num_samples
        
        starts = self.starts[batch_profile_indices] # (B,)
        counts = self.counts[batch_profile_indices] # (B,)
        
        # Defensive: If any count is 0, this will error or produce garbage. 
        # We assume caller filters them out, but to be safe against div/0:
        safe_counts = torch.clamp(counts, min=1).unsqueeze(1)
        
        # Random Offset
        rand_offsets = torch.randint(0, 1000000, (B, M), device=self.device) % safe_counts
        
        # Global Index
        global_indices = starts.unsqueeze(1) + rand_offsets
        flat_indices = global_indices.view(-1)
        
        # Fetch Data
        sampled_as = self.val_as[flat_indices].view(B, M)
        sampled_hops = self.val_hops[flat_indices].view(B, M).float()
        sampled_weights = self.val_weights[flat_indices].view(B, M)
        
        return sampled_as, sampled_hops, sampled_weights

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Using device: {device} ({device_name})")

    data_dir = args.data_dir
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "train_config.json").open("w") as f:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, f, indent=2)
    
    # --- 1. Load Metadata & Data ---
    with open(data_dir / "meta.json", "r") as f:
        meta = json.load(f)
    num_profiles = meta["num_profiles"]
    num_asns = meta["num_asns"]
    if num_profiles <= 0 or num_asns <= 0:
        raise ValueError("Training requires at least one profile and one AS.")
    
    # Load fingerprints directly to GPU for fast similarity calc
    fp_path = data_dir / "profile_fingerprints.pt"
    if not fp_path.exists():
        raise FileNotFoundError(f"{fp_path} not found!")
    
    raw_fp = torch.load(fp_path) # Load to CPU first
    profile_fingerprints = unpack_128bit_to_binary(raw_fp.to(device))

    print(f"Fingerprints ready. Shape: {profile_fingerprints.shape}")
    del raw_fp
    
    triplets_tensor = torch.load(data_dir / "profile_triplets.pt")
    sampler = CompactSampler(triplets_tensor, num_profiles, device)
    del triplets_tensor
    import gc; gc.collect()

    # Filter valid Profiles (those with edges)
    valid_profiles = torch.nonzero(sampler.counts > 0).squeeze(1)
    num_train_profiles = valid_profiles.size(0)
    if num_train_profiles == 0:
        raise ValueError("Training data contains no profile-AS samples.")
    print(f"Valid Profiles for training: {num_train_profiles}")
    
    # --- 2. Model & Optimizer ---
    model = HyperbolicRoutingModel(
        num_asns, num_profiles, 
        embed_dim=args.embed_dim, 
        num_vectors=args.num_vectors
    ).to(device)
    
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Loss Config
    loss_fn = HyperbolicLoss(
        hier_margin=args.hier_margin, 
        rank_weight=args.ranking_weight, 
        sim_margin=args.sim_margin
    )
    
    best_loss = float('inf')
    patience_counter = 0
    batch_size = args.batch_size
    M = args.num_samples_per_profile
    
    print(f"Start Training. Epochs: {args.epochs}, Batch: {batch_size}, Samples(M): {M}")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_hier = 0
        total_sim = 0
        
        # Shuffle profiles for this epoch
        perm = torch.randperm(num_train_profiles, device=device)
        shuffled_indices = valid_profiles[perm]
        
        num_batches = (num_train_profiles + batch_size - 1) // batch_size
        progress = tqdm(range(num_batches), desc=f"Ep {epoch+1}")
        
        for i in progress:
            start = i * batch_size
            end = min(start + batch_size, num_train_profiles)
            
            # --- Batch Construction ---
            # 1. Anchor Profiles
            p_idxs_a = shuffled_indices[start:end]
            curr_bs = p_idxs_a.size(0)
            
            # 2. Comparison Profiles (Randomly permuted from current batch)
            # This creates random pairs efficiently without N^2 compute
            perm_local = torch.randperm(curr_bs, device=device)
            p_idxs_b = p_idxs_a[perm_local]
            
            optimizer.zero_grad()
            
            # --- Part A: Hierarchical Affinity (AS-Prefix) ---
            # Sample Pos/Neg AS (1:3 ratio)
            pos_as, pos_hops, pos_weights = sampler.sample(p_idxs_a, num_samples=M)
            neg_as = torch.randint(0, num_asns, (curr_bs, 3*M), device=device)
            
            d_pos = model(p_idxs_a, pos_as)
            d_neg = model(p_idxs_a, neg_as)

            loss_hier = loss_fn.forward_hierarchical(d_pos, d_neg, pos_hops, sample_weights=pos_weights)
            
            # --- Part B: Propagation Affinity (Prefix-Prefix) ---
            # 1. Ground Truth Affinity from Fingerprints
            fp_a = profile_fingerprints[p_idxs_a]
            fp_b = profile_fingerprints[p_idxs_b]
            raw_affinity = compute_affinity(fp_a, fp_b)
            affinity = recalibrate_affinity(raw_affinity)
            
            # 2. Embedding Distance (Set-to-Set Chamfer)
            dist_p2p = model.compute_profile_chamfer_distance(p_idxs_a, p_idxs_b)
            
            # 3. Soft Contrastive Loss
            loss_sim = loss_fn.forward_similarity(dist_p2p, affinity)
            
            # --- Part C: Regularizations ---
            loss_div_norm = torch.tensor(0.0, device=device)
            if args.diversity_weight > 0 and args.num_vectors > 1:
                loss_div_norm = model.get_diversity_reg(p_idxs_a)
            
            # --- Total Loss ---
            loss = loss_hier + args.sim_weight * loss_sim + args.diversity_weight * loss_div_norm
            
            loss.backward()
            optimizer.step()
            
            # --- Logging ---
            total_loss += loss.item()
            total_hier += loss_hier.item()
            total_sim += loss_sim.item()
            
            progress.set_postfix({
                "L": f"{loss.item():.2f}",
                "H": f"{loss_hier.item():.2f}",
                "S": f"{loss_sim.item():.2f}"
            })
        
        avg_loss = total_loss / num_batches
        print(f"Ep {epoch+1}: Avg={avg_loss:.4f} | Hier={total_hier/num_batches:.4f} | Sim={total_sim/num_batches:.4f}")
        scheduler.step(avg_loss)
        
        # --- Periodic Evaluation (Radius Check) ---
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Check AS Radii (Goal: Large)
                idx = torch.randperm(num_asns)[:5000]
                as_emb = model.as_embedding[idx]
                as_r = model.manifold.dist(torch.zeros_like(as_emb), as_emb).cpu()
                
                # Check Profile Radii (Goal: Small/Medium)
                perm_idx = torch.randperm(num_train_profiles)[:5000]
                sampled_p_ids = valid_profiles[perm_idx]
                
                # Flatten K vectors
                p_emb = model.profile_embedding[sampled_p_ids].view(-1, args.embed_dim)
                p_r = model.manifold.dist(torch.zeros_like(p_emb), p_emb).cpu()
                
                print(f"Radii Stats: AS_Mean={as_r.mean():.2f} (Target>4) | Prof_Mean={p_r.mean():.2f}")
            model.train()
        
        # --- Checkpointing & Early Stopping ---
        if avg_loss < best_loss - args.delta:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "best_model.pth")
            print(f"New Best Model: {best_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print("Early Stopping.")
            break
            
        if (epoch + 1) % 50 == 0:
             torch.save(model.state_dict(), model_dir / f"model_ep{epoch+1}.pth")

    torch.save(model.state_dict(), model_dir / "final_model.pth")
    print("Training Complete.")
    
    # remove all the checkpoint files except best_model.pth
    for file in model_dir.glob("model_ep*.pth"):
        file.unlink()
    print("Checkpoint files cleaned up.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("./data"))
    parser.add_argument("--model_dir", type=Path, default=Path("saved_models"))

    # Loss Params
    parser.add_argument("--hier_margin", type=float, default=1, help="Margin for Hierarchical Loss")
    parser.add_argument("--sim_margin", type=float, default=4.0, help="Margin for Similarity Push")
    parser.add_argument("--ranking_weight", type=float, default=0.5)
    parser.add_argument("--sim_weight", type=float, default=0.8, help="Weight for Prefix Similarity")
    parser.add_argument("--diversity_weight", type=float, default=0.0, help="Norm weight for multi-vector")
    
    # Model Params
    parser.add_argument("--num_vectors", type=int, default=2)
    parser.add_argument("--num_samples_per_profile", type=int, default=64)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--embed_dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
