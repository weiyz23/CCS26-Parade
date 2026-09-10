import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys

# Append path to ensure we can import model
sys.path.append(str(Path(__file__).parent))
from model import HyperbolicRoutingModel

# Percentile Defaults (sigma-based logic)
PERC_POS_HIGH_SIGMA = 99.95     # (Positive distrib)
PERC_NEG_LOW_SIGMA = 1.6        # (Negative distrib)

def find_min_error_threshold(pos_dists, neg_dists, n_grid=10000):
    """
    Find threshold that minimizes (false_positive_rate + false_negative_rate).
    Decision rule: distance < threshold => predict positive.
    pos_dists: distances for true positives (numpy array)
    neg_dists: distances for true negatives (numpy array)
    Returns: threshold (float) or None if inputs invalid, and a small stats dict.
    """
    pos = np.asarray(pos_dists)
    neg = np.asarray(neg_dists)
    if pos.size == 0 or neg.size == 0:
        return None, None

    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)

    # Optimization: Only attempt exact unique values if total size is manageable.
    combined_size = pos.size + neg.size
    thresholds = None

    if combined_size <= 100000:  
        try:
            combined = np.unique(np.concatenate([pos_sorted, neg_sorted]))
            if combined.size <= n_grid:
                thresholds = combined
        except Exception:
            pass

    if thresholds is None:
        lo = float(min(pos_sorted[0], neg_sorted[0]))
        hi = float(max(pos_sorted[-1], neg_sorted[-1]))
        thresholds = np.linspace(lo, hi, n_grid)

    # Empirical CDFs via searchsorted
    pos_counts = np.searchsorted(pos_sorted, thresholds, side='right')
    neg_counts = np.searchsorted(neg_sorted, thresholds, side='right')

    cdf_pos = pos_counts / pos_sorted.size  # P(D <= t | positive)
    cdf_neg = neg_counts / neg_sorted.size  # P(D <= t | negative)

    fnr = 1.0 - cdf_pos  # false negative rate (missed positives)
    fpr = cdf_neg        # false positive rate (negatives predicted positive)

    errs = fnr + fpr
    idx = int(np.argmin(errs))
    best_t = float(thresholds[idx])
    stats = {'fnr': float(fnr[idx]), 'fpr': float(fpr[idx]), 'error_sum': float(errs[idx])}
        
    return best_t, stats

class ThresholdCalculator:
    def __init__(
        self,
        data_dir,
        model_dir,
        embed_dim,
        num_vectors,
        device=None,
        seed=42,
        update_data_meta=True,
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.embed_dim = embed_dim
        self.num_vectors = num_vectors
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)
        self.update_data_meta = bool(update_data_meta)

        # Threshold calibration contains bounded random negative sampling.  A
        # fixed seed makes independent learned spaces directly reproducible
        # without consulting any downstream detection result.
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Resources
        self.model = None
        self.as_map = None
        self.triplets = None
        self.triplets_df = None
        
        # Embeddings
        self.as_emb = None
        self.prof_emb = None
        self.fingerprints = None
        
        # Output
        self.calculated_thresholds = {}

        self._load_resources()

    def _load_resources(self):
        # Meta
        with open(self.data_dir / "meta.json", "r") as f:
            self.meta = json.load(f)
            
        # AS Map
        self.as_map = torch.load(self.data_dir / "as_map.pt")
        
        # Model
        self.model = HyperbolicRoutingModel(
            num_as=self.meta["num_asns"], 
            num_profiles=self.meta["num_profiles"], 
            embed_dim=self.embed_dim, 
            num_vectors=self.num_vectors
        ).to(self.device)
        
        model_path = self.model_dir / "best_model.pth"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            
        self.model.eval()
        
        # Embeddings (Keep on Device)
        self.as_emb = self.model.as_embedding.detach()
        self.prof_emb = self.model.profile_embedding.detach()
        
        # Triplets
        triplets_path = self.data_dir / "profile_triplets.pt"
        if triplets_path.exists():
            self.triplets = torch.load(triplets_path, map_location=self.device)
            if isinstance(self.triplets, np.ndarray):
                self.triplets = torch.from_numpy(self.triplets).to(self.device)
            if torch.is_tensor(self.triplets):
                if self.triplets.dim() != 2 or self.triplets.size(1) < 4:
                    raise ValueError(
                        "profile_triplets.pt must have columns [profile_id, as_id, hop, weight]. "
                        "Delete stale dataset tensors and rerun preprocess/build_dataset.py."
                    )
                total_samples = int(self.triplets.size(0))
                weight_col = self.triplets[:, 3].to(dtype=torch.float32)
                rib_mask = torch.isclose(weight_col, torch.ones_like(weight_col), atol=1e-6)
                self.triplets = self.triplets[rib_mask]
                kept_samples = int(self.triplets.size(0))
                print(f"Triplets filtered by weight==1 (RIB): kept {kept_samples}/{total_samples}")
                if kept_samples == 0:
                    raise ValueError("No triplets with weight==1 found; cannot calculate thresholds.")
                trips_cpu = self.triplets[:, :3].cpu().numpy()
                self.triplets_df = pd.DataFrame(trips_cpu, columns=['pid', 'aid', 'hop'])

    def _calc_dist_indices(self, idx_a, idx_b, emb_a, emb_b, batch_size=8192):
        """
        Compute distance between embeddings indexed by idx_a and idx_b.
        Batches the indexing to avoid OOM on GPU.
        """
        res = []
        # Ensure indices are on the correct device for slicing and valid dtype for indexing
        if isinstance(idx_a, np.ndarray):
            idx_a = torch.from_numpy(idx_a).to(self.device)
        elif isinstance(idx_a, torch.Tensor) and idx_a.device != self.device:
            idx_a = idx_a.to(self.device)
            
        if isinstance(idx_b, np.ndarray):
            idx_b = torch.from_numpy(idx_b).to(self.device)
        elif isinstance(idx_b, torch.Tensor) and idx_b.device != self.device:
            idx_b = idx_b.to(self.device)

        if torch.is_tensor(idx_a) and idx_a.dtype != torch.long:
            idx_a = idx_a.to(dtype=torch.long)
        if torch.is_tensor(idx_b) and idx_b.dtype != torch.long:
            idx_b = idx_b.to(dtype=torch.long)
            
        with torch.no_grad():
            for i in range(0, len(idx_a), batch_size):
                b_idx_a = idx_a[i:i+batch_size]
                b_idx_b = idx_b[i:i+batch_size]
                
                b_a = emb_a[b_idx_a]
                b_b = emb_b[b_idx_b]
                
                if b_a.dim() == 3 and b_b.dim() == 2:
                    b_b = b_b.unsqueeze(1)
                elif b_a.dim() == 2 and b_b.dim() == 3:
                    b_a = b_a.unsqueeze(1)
                    
                d = self.model.manifold.dist(b_a, b_b)
                
                if d.dim() > 1:
                    d, _ = d.min(dim=1)
                
                res.append(d.cpu())
        return torch.cat(res).numpy()

    def run(self):
        if self.triplets is None:
            print("No triplets data found.")
            return

        # Positive (Hop < 8)
        mask_pos = (self.triplets[:, 2] < 8)
        pos_triplets = self.triplets[mask_pos]
        pos_pids = pos_triplets[:, 0].to(dtype=torch.long)
        pos_aids = pos_triplets[:, 1].to(dtype=torch.long)
        # Use batched index lookup to avoid OOM
        pos_dists = self._calc_dist_indices(
            pos_pids,
            pos_aids,
            self.prof_emb,
            self.as_emb
        )
        
        # Negative
        n_neg = min(len(pos_dists), 10000000) # Limit neg samples
        neg_pids = torch.randint(0, len(self.prof_emb), (n_neg,), device=self.device)
        neg_aids = torch.randint(0, len(self.as_emb), (n_neg,), device=self.device)
        
        neg_dists = self._calc_dist_indices(
            neg_pids,
            neg_aids,
            self.prof_emb,
            self.as_emb
        )
        
        t1, stats1 = find_min_error_threshold(pos_dists, neg_dists)
        t1_pos_sigma = float(np.percentile(pos_dists, PERC_POS_HIGH_SIGMA))
        t1_neg_sigma = float(np.percentile(neg_dists, PERC_NEG_LOW_SIGMA))
        
        t1_err = t1 if t1 is not None else t1_pos_sigma

        # Logic: max(others) but capped at Neg -2sigma
        t1_final = min(t1_neg_sigma, max(t1_pos_sigma, t1_err))
        
        self.calculated_thresholds['threshold_profile_as'] = t1_final
        self.calculated_thresholds['threshold_profile_as_details'] = {
            'pos_sigma': t1_pos_sigma,
            'neg_sigma': t1_neg_sigma,
            'min_err': t1_err
        }

        # Origin (Hop == 0)
        mask_origin = (self.triplets[:, 2] == 0)
        origin_triplets = self.triplets[mask_origin]
        if len(origin_triplets) > 0:
            origin_pids = origin_triplets[:, 0].to(dtype=torch.long)
            origin_aids = origin_triplets[:, 1].to(dtype=torch.long)
            origin_dists = self._calc_dist_indices(
                origin_pids,
                origin_aids,
                self.prof_emb,
                self.as_emb
            )
            # Use same negatives
            t2, stats2 = find_min_error_threshold(origin_dists, neg_dists)
            t2_pos_sigma = float(np.percentile(origin_dists, PERC_POS_HIGH_SIGMA))
            t2_neg_sigma = float(np.percentile(neg_dists, PERC_NEG_LOW_SIGMA))
            
            t2_err = t2 if t2 is not None else t2_neg_sigma
            
            # Logic: max(others) but capped at Neg -2sigma
            t2_final = min(t2_neg_sigma, max(t2_pos_sigma, t2_err))
            
            self.calculated_thresholds['threshold_profile_origin'] = t2_final
            self.calculated_thresholds['threshold_profile_origin_details'] = {
                'pos_sigma': t2_pos_sigma,
                'neg_sigma': t2_neg_sigma,
                'min_err': t2_err
            }
        else:
             # Basic fallback if no origin data
             self.calculated_thresholds['threshold_profile_origin'] = 4.0
             self.calculated_thresholds['threshold_profile_origin_details'] = {}

        # Reuse logic from visualize.py
        origin_df = self.triplets_df[self.triplets_df['hop'] == 0]
        pid_to_origins = origin_df.groupby('pid')['aid'].apply(list).to_dict()
        pid_to_associated = self.triplets_df.groupby('pid')['aid'].apply(set).to_dict()
        
        valid_pids = [p for p in pid_to_origins.keys() if p in pid_to_associated]
        # Use a reasonable sample size for threshold calc
        MAX_SAMPLES = 100000 
        if len(valid_pids) > MAX_SAMPLES:
            sampled_pids = np.random.choice(valid_pids, MAX_SAMPLES, replace=False)
        else:
            sampled_pids = valid_pids
            
        assoc_dists = []
        unrelated_dists = []
        
        as_emb_device = self.as_emb.to(self.device)
        num_as = len(self.as_emb)
        
        for pid in sampled_pids:
            origins = pid_to_origins[pid]
            associated = list(pid_to_associated[pid])
            if not origins: continue
            
            origin_vecs = as_emb_device[origins]
            
            # Associated
            if associated:
                assoc_vecs = as_emb_device[associated]
                with torch.no_grad():
                    dists = self.model.manifold.dist(assoc_vecs.unsqueeze(1), origin_vecs.unsqueeze(0))
                    min_dists, _ = dists.min(dim=1)
                    assoc_dists.extend(min_dists.cpu().numpy())
            
            # Unrelated (Sampled) - Fewer per PID for speed, but aggregate is large
            needed = 25 
            candidates_pool = np.random.randint(0, num_as, needed * 5) 
            unrelated_indices = [c for c in candidates_pool if c not in pid_to_associated[pid]]
            if len(unrelated_indices) > needed:
                unrelated_indices = unrelated_indices[:needed]
            
            if unrelated_indices:
                unrelated_vecs = as_emb_device[unrelated_indices]
                with torch.no_grad():
                     dists_unrel = self.model.manifold.dist(unrelated_vecs.unsqueeze(1), origin_vecs.unsqueeze(0))
                     min_unrel, _ = dists_unrel.min(dim=1)
                     unrelated_dists.extend(min_unrel.cpu().numpy())

        if len(assoc_dists) > 0 and len(unrelated_dists) > 0:
            t3, stats3 = find_min_error_threshold(assoc_dists, unrelated_dists)
            t3_neg_sigma = float(np.percentile(unrelated_dists, PERC_NEG_LOW_SIGMA))
            t3_pos_sigma = float(np.percentile(assoc_dists, PERC_POS_HIGH_SIGMA)) 
            
            t3_err = t3 if t3 is not None else t3_neg_sigma

            # Logic: max(others) but capped at Neg -2sigma
            t3_final = min(t3_neg_sigma, max(t3_pos_sigma, t3_err))

            self.calculated_thresholds['threshold_as_origin'] = t3_final
            self.calculated_thresholds['threshold_as_origin_details'] = {
                'pos_sigma': t3_pos_sigma, 
                'neg_sigma': t3_neg_sigma,
                'min_err': t3_err
            }

        self._persist_thresholds()

    def _persist_thresholds(self):
        if self.update_data_meta:
            meta_path = self.data_dir / "meta.json"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            meta.update(self.calculated_thresholds)

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)

        # Also persist thresholds with the model version directory to support online multi-model inference.
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with open(self.model_dir / "thresholds.json", "w") as f:
            json.dump(self.calculated_thresholds, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--model_dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--embed_dim", type=int, default=24)
    parser.add_argument("--num_vectors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Write thresholds.json only under model_dir; leave data/meta.json unchanged.",
    )
    args = parser.parse_args()
    
    calc = ThresholdCalculator(
        args.data_dir,
        args.model_dir,
        args.embed_dim,
        args.num_vectors,
        seed=args.seed,
        update_data_meta=not args.model_only,
    )
    calc.run()
