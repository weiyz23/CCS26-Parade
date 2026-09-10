import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import argparse
import sys

# Append path to ensure we can import model
sys.path.append(str(Path(__file__).parent))
from model import HyperbolicRoutingModel
from utils import unpack_128bit_to_binary, compute_affinity

# --- Constants ---
TIER1_ASNS = {
    "701", "1299", "2914", "3257", "3320", "3356", "3491", 
    "5511", "6453", "6461", "6762", "6830", "7018", "12956"
}

# --- Plot Settings ---
FONT_SIZE = 16
FONT_SIZE_SMALL = 14

# Configure Seaborn to respect Times New Roman
sns.set_theme(style="whitegrid", rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE_SMALL,
    "ytick.labelsize": FONT_SIZE_SMALL,
    "legend.fontsize": FONT_SIZE_SMALL,
    "axes.grid": True,
    "grid.alpha": 0.3
})

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

# --- Main Visualizer Class ---

class HyperbolicVisualizer:
    def __init__(self, data_dir, model_dir, viz_dir, embed_dim, num_vectors, device=None):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.embed_dim = embed_dim
        self.num_vectors = num_vectors
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # internal state
        self.model = None
        self.as_map = None
        self.id2as = None
        self.triplets = None      # Torch tensor
        self.triplets_df = None   # Pandas DataFrame
        self.profile_fingerprints = None
        self.rank_data = {}       # asn -> rank
        self.metrics_data = {}    # asn (str) -> {rank, prefix_count}
        
        # Computed Properties
        self.as_emb = None       # detached cpu tensor
        self.prof_emb = None     # detached cpu tensor
        self.as_radii = None     # numpy array
        
        # Computed Metrics
        self.threshold_profile_as = None
        self.threshold_profile_origin = None 
        self.threshold_as_origin = None
        
        # Load all resources
        self._load_resources()

    def _load_resources(self):
        # 1. Meta & Mappings
        try:
            with open(self.data_dir / "meta.json", "r") as f:
                self.meta = json.load(f)
        except Exception:
            raise ValueError("Please run calc_thresholds.py to generate thresholds and meta information.")
            
        # Try to load pre-calculated thresholds
        self.threshold_profile_as = self.meta.get('threshold_profile_as')
        self.threshold_profile_origin = self.meta.get('threshold_profile_origin')
        self.threshold_as_origin = self.meta.get('threshold_as_origin')
        
        # Enforce thresholds existence
        if self.threshold_profile_as is None:
            raise ValueError("Meta file missing 'threshold_profile_as'. Please run calc_thresholds.py.")
            
        self.as_map = torch.load(self.data_dir / "as_map.pt")
        self.id2as = {v: k for k, v in self.as_map.items()}
        
        # 2. Model
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
        
        # 3. Embeddings (Cache as detached tensors on DEVICE for analysis)
        self.as_emb = self.model.as_embedding.detach()
        self.prof_emb = self.model.profile_embedding.detach()
        
        # 4. Triplets
        triplets_path = self.data_dir / "profile_triplets.pt"
        if triplets_path.exists():
            self.triplets = torch.load(triplets_path).cpu() # Keep on CPU for DF conversion
            if isinstance(self.triplets, np.ndarray):
                self.triplets = torch.from_numpy(self.triplets)
            self.triplets_df = pd.DataFrame(self.triplets.numpy(), columns=['pid', 'aid', 'hop'])
            
            # Create set of positive pairs (depth < 8)
            self.positive_pairs = set()
            mask_pos = self.triplets[:, 2] < 8
            pos_triplets = self.triplets[mask_pos]
            for row in pos_triplets:
                self.positive_pairs.add((int(row[0]), int(row[1])))
        
        # 5. Fingerprints (Optional)
        fp_path = self.data_dir / "profile_fingerprints.pt"
        if fp_path.exists():
            raw_fp = torch.load(fp_path, map_location='cpu')
            if raw_fp.dim() == 2 and raw_fp.shape[1] == 2:
                self.profile_fingerprints = unpack_128bit_to_binary(raw_fp)
            else:
                self.profile_fingerprints = raw_fp.float()
                if self.profile_fingerprints.max() > 1.0:
                     self.profile_fingerprints = (self.profile_fingerprints > 0).float()
            
            # Move to device for calculation to speed up massive pair mining
            if self.profile_fingerprints is not None:
                self.profile_fingerprints = self.profile_fingerprints.to(self.device)
        
        # 6. Rank Data
        self._load_rank_data()

    def _load_rank_data(self):
        search_dirs = [self.data_dir, self.data_dir.parent]
        filenames = ["as_rank.jsonl", "as_info.jsonl"]
        rank_file = None
        for d in search_dirs:
            for fname in filenames:
                f = d / fname
                if f.exists(): 
                    rank_file = f
                    break
            if rank_file: break
            
        if rank_file:
            with open(rank_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        asn = int(data['asn'])
                        self.rank_data[asn] = data['rank']
                    except Exception: continue
        else:
            print("Warning: AS Rank file not found.")

    def _compute_hyperbolic_radius(self, embeddings_tensor):
        """Batch compute hyperbolic radius."""
        # embeddings: (N, D) or (N, M, D)
        original_shape = embeddings_tensor.shape
        flat_emb = embeddings_tensor.reshape(-1, original_shape[-1]).to(self.device)
        
        zeros = torch.zeros_like(flat_emb)
        dists = []
        batch_size = 65536
        
        with torch.no_grad():
            for i in range(0, len(flat_emb), batch_size):
                b_emb = flat_emb[i:i+batch_size]
                b_zeros = zeros[i:i+batch_size]
                d = self.model.manifold.dist(b_zeros, b_emb)
                dists.append(d.cpu())
                
        radii = torch.cat(dists).numpy()
        if len(original_shape) > 2:
            return radii.reshape(original_shape[:-1])
        return radii
    
    def _calc_dist_batched(self, tensor_a, tensor_b, batch_size=4096):
        """
        Compute pairwise distances between aligned tensors.
        tensor_a: (N, ...) 
        tensor_b: (N, ...)
        Returns: (N,) numpy array
        """        
        res = []
        with torch.no_grad():
            for i in range(0, len(tensor_a), batch_size):
                b_a = tensor_a[i:i+batch_size].to(self.device)
                b_b = tensor_b[i:i+batch_size].to(self.device)
                
                # Check shapes for broadcasting needs
                # e.g. Prof (B, M, D) vs AS (B, 1, D)
                if b_a.dim() == 3 and b_b.dim() == 2:
                    b_b = b_b.unsqueeze(1)
                elif b_a.dim() == 2 and b_b.dim() == 3:
                    b_a = b_a.unsqueeze(1)
                    
                d = self.model.manifold.dist(b_a, b_b)
                
                # If outcome is (B, M), we typically want min if it's multi-vector
                if d.dim() > 1:
                    d, _ = d.min(dim=1)
                
                res.append(d.cpu())
        return torch.cat(res).numpy()

    def _sample_negative_pairs(self, n_samples, num_profiles, num_as):
        """
        Sample negative pairs (pid, aid) that are not in positive_pairs.
        """
        neg_pairs = []
        attempts = 0
        max_attempts = n_samples * 10  # Allow some retries
        while len(neg_pairs) < n_samples and attempts < max_attempts:
            pids = torch.randint(0, num_profiles, (n_samples - len(neg_pairs),))
            aids = torch.randint(0, num_as, (n_samples - len(neg_pairs),))
            for pid, aid in zip(pids.tolist(), aids.tolist()):
                if (pid, aid) not in self.positive_pairs:
                    neg_pairs.append((pid, aid))
            attempts += 1
        if len(neg_pairs) < n_samples:
            print(f"Warning: Could only sample {len(neg_pairs)} negative pairs out of {n_samples} requested.")
        return neg_pairs

    def set_adaptive_ticks(self, ax=None):
        """
        Automatically set x and y axis ticks to have 4-6 ticks with nice intervals.
        Hides the last tick label for cleaner appearance.
        """
        if ax is None:
            ax = plt.gca()
        
        # Get current limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # If y_max != 1, add 10% padding to y_max
        if y_max != 1:
            y_range = y_max - y_min
            y_max += 0.1 * y_range
            ax.set_ylim(y_min, y_max)
        
        def compute_nice_ticks(min_val, max_val, target_ticks=5):
            range_val = max_val - min_val
            if range_val <= 0:
                return [min_val]
            
            # Possible step sizes
            nice_steps = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
            
            best_step = nice_steps[0]
            best_diff = float('inf')

            for step in nice_steps:
                count = round(range_val / step) + 1
                diff = abs(count - target_ticks)
                
                if 4 <= count <= 6:
                    best_step = step
                    break
                elif diff < best_diff:
                    best_step = step
                    best_diff = diff
            
            start_tick = np.ceil(min_val / best_step) * best_step
            start_tick = round(min_val / best_step) * best_step
            
            ticks = []
            i = 0
            while True:
                val = start_tick + i * best_step
                
                if val > max_val + 1e-8:
                    break
                
                if val >= min_val - 1e-8:
                    val = round(val, 10)
                    ticks.append(val)
                
                i += 1
                
            return ticks

        def format_labels(ticks):
            labels = [f'{t:g}' for t in ticks]
            return labels

        # Set x ticks
        x_ticks = compute_nice_ticks(x_min, x_max)
        ax.set_xticks(x_ticks)
        if len(x_ticks) > 0:
            ax.set_xticklabels(format_labels(x_ticks))
        
        # Set y ticks
        y_ticks = compute_nice_ticks(y_min, y_max)
        ax.set_yticks(y_ticks)
        if len(y_ticks) > 0:
            ax.set_yticklabels(format_labels(y_ticks))

    def plot_cdf(self, data, label, color, linestyle='-', ax=None):
        if len(data) == 0: return
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        if ax:
            ax.plot(sorted_data, yvals, label=label, color=color, linestyle=linestyle, linewidth=2)
        else:
            plt.plot(sorted_data, yvals, label=label, color=color, linestyle=linestyle, linewidth=2)

    # --- Plotting Logic ---

    def plot_profile_pair_distributions(self):
        """
        Refactored Visualization:
        Focuses on distinguishing "True Signals" (High Affinity) from "Noise" (Random).
        Uses massive pre-sampling to find rare high-similarity pairs.
        """
        if self.prof_emb is None: return
        if self.profile_fingerprints is None:
             print("Skipping similarity PDF: No profile fingerprints found.")
             return

        SCAN_BATCH_SIZE = 10000 
        MAX_SCAN_PAIRS = 15000000
        TARGET_SAMPLES_PER_BIN = 5000
        
        num_profiles = len(self.prof_emb)
        
        bins = {
            "Strangers [0, 0.2)": [],
            "Neighbors [0.2, 0.5)": [],
            "Siblings  [0.5, 1]": []
        }
        
        def calc_sim_chunk(idx1, idx2):
            fp1 = self.profile_fingerprints[idx1]
            fp2 = self.profile_fingerprints[idx2]
            hamming = torch.sum(torch.abs(fp1 - fp2), dim=1)
            return 1.0 - (hamming / fp1.size(1))

        # --- Phase 1: Massive Mining ---
        total_scanned = 0
        with torch.no_grad():
            while total_scanned < MAX_SCAN_PAIRS:
                idx1 = torch.randint(0, num_profiles, (SCAN_BATCH_SIZE,), device=self.device)
                idx2 = torch.randint(0, num_profiles, (SCAN_BATCH_SIZE,), device=self.device)
                
                sims = calc_sim_chunk(idx1, idx2)
                
                # 0-0.6 -> 0-0.2
                mask_stranger = sims < 0.60
                # 0.6-0.75 -> 0.2-0.5
                mask_neighbor = (sims >= 0.60) & (sims < 0.75)
                # 0.75-1.0 -> 0.5-1.0
                mask_sibling  = sims >= 0.75
                
                if len(bins["Strangers [0, 0.2)"]) < TARGET_SAMPLES_PER_BIN:
                    k = "Strangers [0, 0.2)"
                    sel = torch.nonzero(mask_stranger).squeeze()
                    if sel.numel() > 0:
                        if sel.dim() == 0: sel = sel.unsqueeze(0)
                        keep = sel[:TARGET_SAMPLES_PER_BIN // 10]
                        bins[k].append(torch.stack([idx1[keep], idx2[keep]], dim=1))
                        
                if len(bins["Neighbors [0.2, 0.5)"]) < TARGET_SAMPLES_PER_BIN:
                    k = "Neighbors [0.2, 0.5)"
                    sel = torch.nonzero(mask_neighbor).squeeze()
                    if sel.numel() > 0:
                        if sel.dim() == 0: sel = sel.unsqueeze(0)
                        bins[k].append(torch.stack([idx1[sel], idx2[sel]], dim=1))
                        
                if len(bins["Siblings  [0.5, 1]"]) < TARGET_SAMPLES_PER_BIN:
                    k = "Siblings  [0.5, 1]"
                    sel = torch.nonzero(mask_sibling).squeeze()
                    if sel.numel() > 0:
                        if sel.dim() == 0: sel = sel.unsqueeze(0)
                        bins[k].append(torch.stack([idx1[sel], idx2[sel]], dim=1))
                
                total_scanned += SCAN_BATCH_SIZE
                
                full = all(len(torch.cat(v)) >= TARGET_SAMPLES_PER_BIN if v else False for v in bins.values())
                if full:
                    break
        
        # --- Phase 2: Compute Hyperbolic Distances & Plot ---
        plt.figure(figsize=(6, 4), facecolor='white')
        
        for label, idx_list in bins.items():
            if not idx_list:
                print(f"Warning: No samples found for {label}")
                continue
                
            pairs = torch.cat(idx_list, dim=0)
            if pairs.size(0) > TARGET_SAMPLES_PER_BIN:
                pairs = pairs[:TARGET_SAMPLES_PER_BIN]
            
            idx1 = pairs[:, 0]
            idx2 = pairs[:, 1]
            real_count = idx1.size(0)
            
            vecs1 = self.prof_emb[idx1].to(self.device)
            vecs2 = self.prof_emb[idx2].to(self.device)
            
            if vecs1.dim() == 3:
                 v1_exp = vecs1.unsqueeze(2) # (B, K, 1, D)
                 v2_exp = vecs2.unsqueeze(1) # (B, 1, K, D)
                 d_mat = self.model.manifold.dist(v1_exp, v2_exp) # (B, K, K)
                 
                 # Min-Linkage Distance: min_{i, j} d(a_i, b_j)
                 dists, _ = d_mat.view(d_mat.shape[0], -1).min(dim=1)
            else:
                dists = self.model.manifold.dist(vecs1, vecs2)
            
            dists_np = dists.detach().cpu().numpy()
            
            c = 'blue'
            if 'Strangers' in label: c = 'gray'
            elif 'Neighbors' in label: c = 'orange'
            elif 'Siblings' in label: c = 'red'
            
            # Plot CDF
            self.plot_cdf(dists_np, f"{label}", c, '-')
            
        plt.xlabel("Hyperbolic Distance", fontsize=FONT_SIZE)
        plt.ylabel("CDF", fontsize=FONT_SIZE)
        plt.ylim(0,1)
        plt.xlim(left=0)
        self.set_adaptive_ticks()
        plt.legend(fontsize=FONT_SIZE_SMALL)
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0.0, color='k', linestyle='--', alpha=0.1)
        out_path = self.viz_dir / "profile_semantic_separation_cdf.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Bonus: Correlation Hexbin Plot ---
        all_sims = []
        all_dists = []
        
        for label, idx_list in bins.items():
            if not idx_list: continue
            pairs = torch.cat(idx_list, dim=0)
            # Subsample for hexbin to avoid memory issues if bins are huge, though we capped at 5000
            if len(pairs) > 2000:
                pairs = pairs[:2000]
                
            idx1, idx2 = pairs[:, 0], pairs[:, 1]
            
            # Recalc Sim & Dist
            fp1 = self.profile_fingerprints[idx1].to(self.device)
            fp2 = self.profile_fingerprints[idx2].to(self.device)
            
            # Remapping
            hamming = torch.sum(torch.abs(fp1 - fp2), dim=1)
            s = 1.0 - 2 * (hamming / fp1.size(1))
            s = torch.clamp(s, 0.0, 1.0)
            
            vecs1 = self.prof_emb[idx1].to(self.device)
            vecs2 = self.prof_emb[idx2].to(self.device)
            
            if vecs1.dim() == 3:
                v1_exp = vecs1.unsqueeze(2)
                v2_exp = vecs2.unsqueeze(1)
                d_mat = self.model.manifold.dist(v1_exp, v2_exp)
                
                # Min-Linkage Distance
                d, _ = d_mat.view(d_mat.shape[0], -1).min(dim=1)
            else:
                d = self.model.manifold.dist(vecs1, vecs2)
        
            all_sims.append(s.cpu().numpy())
            all_dists.append(d.detach().cpu().numpy())
             
        if all_sims:
            plt.figure(figsize=(6, 4))
            x = np.concatenate(all_sims)
            y = np.concatenate(all_dists)
            
            from matplotlib.colors import LogNorm
            from matplotlib.ticker import LogLocator, LogFormatterMathtext
            
            # Use LogNorm for logarithmic color scaling, with LogLocator for nice ticks (1, 10, 100...)
            plt.hexbin(x, y, gridsize=30, cmap='inferno', norm=LogNorm(), mincnt=1)
            cbar = plt.colorbar()
            cbar.set_label('Count', fontsize=FONT_SIZE_SMALL)
            
            # Force major ticks only at powers of 10
            cbar.locator = LogLocator(base=10, subs=(1.0,))
            cbar.formatter = LogFormatterMathtext(base=10) # Use 10^x formatting
            cbar.update_ticks()
            # Explicitly turn off minor ticks to remove the clutter
            cbar.minorticks_off()
            
            plt.xlabel("Fingerprint Similarity (Ground Truth)", fontsize=FONT_SIZE)
            plt.ylabel("Hyperbolic Distance (Learned)", fontsize=FONT_SIZE)
            plt.ylim(bottom=0)
            plt.xlim(0,1)
            self.set_adaptive_ticks()
            plt.gca().invert_xaxis() # Invert x-axis so SimHash 1.0 (near) is left, 0.0 (far) is right
            
            plt.savefig(self.viz_dir / "profile_correlation_hexbin.png", dpi=300, bbox_inches='tight')
            plt.close()

    def plot_radius_distribution(self):
        self.as_radii = self._compute_hyperbolic_radius(self.as_emb)
        prof_radii_flat = self._compute_hyperbolic_radius(self.prof_emb).flatten()
        
        plt.figure(figsize=(6, 4), facecolor='white')
        sns.kdeplot(self.as_radii, fill=True, label='AS Embeddings', color='gray')
        sns.kdeplot(prof_radii_flat, fill=True, label='Profile Embeddings', color='steelblue')
        plt.xlabel("Hyperbolic Radius", fontsize=FONT_SIZE)
        plt.xlim(1,6)
        plt.ylim(bottom=0)
        plt.ylabel("Density", fontsize=FONT_SIZE)
        plt.legend(fontsize=FONT_SIZE_SMALL)
        plt.grid(True, alpha=0.3)
        self.set_adaptive_ticks()  # Add adaptive ticks
        plt.savefig(self.viz_dir / "metric_radius_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_distance_distributions_pdf(self):
        MAX_SAMPLES_PLOT = 50000  # Only affects plotting speed, not threshold accuracy
        
        # --- 1. PDF of Hyperbolic Distances (AS-Profile) ---
        if self.triplets is None: return

        # A. Prepare Data
        mask_pos = (self.triplets[:, 2] < 8)
        pos_triplets = self.triplets[mask_pos]
        
        # Optimize: Threshold is known, only sample for plotting
        if len(pos_triplets) > MAX_SAMPLES_PLOT:
             indices = torch.randperm(len(pos_triplets))[:MAX_SAMPLES_PLOT]
             pos_triplets_sample = pos_triplets[indices]
             pos_dists = self._calc_dist_batched(
                self.prof_emb[pos_triplets_sample[:, 0]], 
                self.as_emb[pos_triplets_sample[:, 1]]
             )
        else:
             pos_dists = self._calc_dist_batched(
                self.prof_emb[pos_triplets[:, 0]], 
                self.as_emb[pos_triplets[:, 1]]
             )

        # Origin Samples (Depth == 0)
        mask_origin = (self.triplets[:, 2] == 0)
        origin_triplets = self.triplets[mask_origin]
        
        origin_dists = np.array([])
        if len(origin_triplets) > 0:
            if len(origin_triplets) > MAX_SAMPLES_PLOT:
                 indices = torch.randperm(len(origin_triplets))[:MAX_SAMPLES_PLOT]
                 origin_sample = origin_triplets[indices]
                 origin_dists = self._calc_dist_batched(
                     self.prof_emb[origin_sample[:, 0]],
                     self.as_emb[origin_sample[:, 1]]
                 )
            else:
                 origin_dists = self._calc_dist_batched(
                    self.prof_emb[origin_triplets[:, 0]], 
                    self.as_emb[origin_triplets[:, 1]]
                 )
        
        # Negative Samples (Match Positive Count for balance in Threshold Calc)
        n_neg = len(pos_dists)
        if n_neg > MAX_SAMPLES_PLOT: n_neg = MAX_SAMPLES_PLOT
            
        neg_pairs = self._sample_negative_pairs(n_neg, len(self.prof_emb), len(self.as_emb))
        neg_pids = torch.tensor([p for p, a in neg_pairs])
        neg_aids = torch.tensor([a for p, a in neg_pairs])
        neg_dists = self._calc_dist_batched(
            self.prof_emb[neg_pids],
            self.as_emb[neg_aids]
        )

        # B. Plot 1: Positive (All Depths) vs Negative
        plt.figure(figsize=(6, 4), facecolor='white')
        
        # Downsample ONLY for plotting (KDE is slow on millions of points)
        pos_dists_plot = pos_dists
        if len(pos_dists) > MAX_SAMPLES_PLOT:
            pos_dists_plot = np.random.choice(pos_dists, MAX_SAMPLES_PLOT, replace=False)
            
        neg_dists_plot = neg_dists
        if len(neg_dists) > MAX_SAMPLES_PLOT:
            neg_dists_plot = np.random.choice(neg_dists, MAX_SAMPLES_PLOT, replace=False)
            
        sns.kdeplot(pos_dists_plot, fill=True, label="Related AS", color="green")
        sns.kdeplot(neg_dists_plot, fill=True, label="Unrelated AS", color="red")
        
        plt.xlabel("Hyperbolic Distance", fontsize=FONT_SIZE)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.ylabel("Density", fontsize=FONT_SIZE)
        plt.legend(loc="best", fontsize=FONT_SIZE_SMALL)
        plt.grid(True, alpha=0.3)
        self.set_adaptive_ticks()  # Add adaptive ticks
        plt.savefig(self.viz_dir / "metric_distance_pdf_positive.png", dpi=300, bbox_inches='tight')
        plt.close()

        # C. Plot 2: Origin (Depth 0) vs Negative
        if len(origin_dists) > 0:
            plt.figure(figsize=(6, 4), facecolor='white')
            
            origin_dists_plot = origin_dists
            if len(origin_dists) > MAX_SAMPLES_PLOT:
                origin_dists_plot = np.random.choice(origin_dists, MAX_SAMPLES_PLOT, replace=False)
                
            sns.kdeplot(origin_dists_plot, fill=True, label="Origin AS", color="blue")
            sns.kdeplot(neg_dists_plot, fill=True, label="Unrelated AS", color="red") # Reuse negative plot sample
            
            plt.xlabel("Hyperbolic Distance", fontsize=FONT_SIZE)
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.ylabel("Density", fontsize=FONT_SIZE)
            plt.legend(loc="best", fontsize=FONT_SIZE_SMALL)
            plt.grid(True, alpha=0.3)
            self.set_adaptive_ticks()  # Add adaptive ticks
            plt.savefig(self.viz_dir / "metric_distance_pdf_origin.png", dpi=300, bbox_inches='tight')
            plt.close()

        # --- 2. PDF AS Alignment: Distance to Origin AS ---
        # Identify origins
        origin_df = self.triplets_df[self.triplets_df['hop'] == 0]
        pid_to_origins = origin_df.groupby('pid')['aid'].apply(list).to_dict()
        pid_to_associated = self.triplets_df.groupby('pid')['aid'].apply(set).to_dict()
        
        valid_pids = [p for p in pid_to_origins.keys() if p in pid_to_associated]
        # REDUCE SAMPLE SIZE for loop-heavy operations
        MAX_SAMPLES_ORIGIN_LOOP = 10000 
        if len(valid_pids) > MAX_SAMPLES_ORIGIN_LOOP:
            sampled_pids = np.random.choice(valid_pids, MAX_SAMPLES_ORIGIN_LOOP, replace=False)
        else:
            sampled_pids = valid_pids
            
        assoc_dists = []
        unrelated_dists = []
        
        as_emb_device = self.as_emb.to(self.device)
        num_as = len(self.as_emb)
        
        for pid in sampled_pids:
            origins = pid_to_origins[pid]
            associated = [x for x in pid_to_associated[pid] if x not in origins]
            if not origins: continue
            
            origin_vecs = as_emb_device[origins] # (K, D)
            
            # Associated
            if associated:
                assoc_vecs = as_emb_device[associated] # (M, D)
                with torch.no_grad():
                    dists = self.model.manifold.dist(assoc_vecs.unsqueeze(1), origin_vecs.unsqueeze(0))
                    min_dists, _ = dists.min(dim=1)
                    assoc_dists.extend(min_dists.cpu().numpy())
            
            # Unrelated (Sampled)
            needed = 5000
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

        plt.figure(figsize=(6, 4), facecolor='white')
        sns.kdeplot(assoc_dists, fill=True, label="Related AS", color="green")
        sns.kdeplot(unrelated_dists, fill=True, label="Unrelated AS", color="red")
        
        plt.xlabel("Hyperbolic Distance to Nearest Origin AS", fontsize=FONT_SIZE)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.ylabel("Density", fontsize=FONT_SIZE)
        plt.legend(loc="best", fontsize=FONT_SIZE_SMALL)
        plt.grid(True, alpha=0.3)
        self.set_adaptive_ticks()  # Add adaptive ticks
        plt.savefig(self.viz_dir / "metric_as_origin_proximity_pdf.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_as_prefix_distance_cdf(self):
        if self.triplets is None: return

        # 1. Positive Samples (Depth < 8)
        mask_pos = (self.triplets[:, 2] < 8)
        pos_triplets = self.triplets[mask_pos]
        
        # Batch Calculate Positive Dists
        # Use torch slicing directly
        pos_res = self._calc_dist_batched(
            self.prof_emb[pos_triplets[:, 0]], 
            self.as_emb[pos_triplets[:, 1]]
        )
        
        # 2. Negative Samples (Random)
        n_neg = len(pos_res)
        neg_pairs = self._sample_negative_pairs(n_neg, len(self.prof_emb), len(self.as_emb))
        neg_pids = torch.tensor([p for p, a in neg_pairs])
        neg_aids = torch.tensor([a for p, a in neg_pairs])
        neg_res = self._calc_dist_batched(
            self.prof_emb[neg_pids],
            self.as_emb[neg_aids]
        )

        # 3. Origin Samples (Depth == 0)
        mask_origin = (self.triplets[:, 2] == 0)
        origin_triplets = self.triplets[mask_origin]
        origin_res = np.array([])
        if len(origin_triplets) > 0:
            origin_res = self._calc_dist_batched(
                self.prof_emb[origin_triplets[:, 0]],
                self.as_emb[origin_triplets[:, 1]]
            )

        # Plot
        plt.figure(figsize=(6, 4), facecolor='white')
        self.plot_cdf(pos_res, "Positive Pairs", "green")
        if len(origin_res) > 0:
            self.plot_cdf(origin_res, "Origin Pairs", "blue")
        self.plot_cdf(neg_res, "Negative Pairs", "red")
        
        # Threshold Logic
        plt.axvline(x=self.threshold_profile_as, color='black', linestyle='--', 
                    label=f'All Pos Threshold: {self.threshold_profile_as:.2f}')
                        
        if len(origin_res) > 0:
            plt.axvline(x=self.threshold_profile_origin, color='blue', linestyle=':', 
                        label=f'Origin Threshold: {self.threshold_profile_origin:.2f}')
        
        plt.xlabel("Hyperbolic Distance", fontsize=FONT_SIZE)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.ylabel("CDF", fontsize=FONT_SIZE)
        plt.legend(loc="lower right", fontsize=FONT_SIZE_SMALL)
        plt.grid(True, alpha=0.3)
        self.set_adaptive_ticks()  # Add adaptive ticks
        plt.savefig(self.viz_dir / "metric_as_prefix_distance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_moas_metrics(self):
        if self.triplets_df is None: return
        
        df = self.triplets_df
        # origin_df = df[df['hop'] == 0]
        # origin_counts = origin_df.groupby('pid')['aid'].nunique()
        
        # moas_pids = set(origin_counts[origin_counts > 1].index)
        # non_moas_pids = set(origin_counts[origin_counts == 1].index)
        
        # --- Plot 1: Vector Specialization ---
        # Logic: Compare Min Distance (Nearest Vector) vs Max Distance (Furthest Vector) for MOAS vs Non-MOAS
        
        # def compute_min_max_dists(target_pids):
        #     # Subset dataframe
        #     subset = origin_df[origin_df['pid'].isin(target_pids)]
        #     if subset.empty: return np.array([]), np.array([])
            
        #     # Limit sample size for memory
        #     if len(subset) > 50000:
        #         subset = subset.sample(50000)
                
        #     pids = torch.tensor(subset['pid'].values, dtype=torch.long)
        #     aids = torch.tensor(subset['aid'].values, dtype=torch.long)
            
        #     # Batch calculate
        #     # Need (N, K, D) from profiles and (N, D) from AS
        #     all_min = []
        #     all_max = []
        #     batch_size = 2048
            
        #     p_emb_dev = self.prof_emb.to(self.device)
        #     a_emb_dev = self.as_emb.to(self.device)
            
        #     with torch.no_grad():
        #         for i in range(0, len(pids), batch_size):
        #             b_pids = pids[i:i+batch_size]
        #             b_aids = aids[i:i+batch_size]
                    
        #             b_p_vecs = p_emb_dev[b_pids] # (B, K, D)
        #             b_a_vecs = a_emb_dev[b_aids].unsqueeze(1) # (B, 1, D)
                    
        #             dists = self.model.manifold.dist(b_p_vecs, b_a_vecs) # (B, K)
                    
        #             min_d, _ = dists.min(dim=1)
        #             max_d, _ = dists.max(dim=1)
                    
        #             all_min.append(min_d.cpu())
        #             all_max.append(max_d.cpu())
                    
        #     return torch.cat(all_min).numpy(), torch.cat(all_max).numpy()

        # moas_min, moas_max = compute_min_max_dists(moas_pids)
        # non_moas_min, non_moas_max = compute_min_max_dists(non_moas_pids)
        
        # plt.figure(figsize=(7, 4), facecolor='white')
        # self.plot_cdf(moas_min, "MOAS: Nearest (Active)", "red", "-")
        # self.plot_cdf(moas_max, "MOAS: Furthest (Idle)", "red", "--")
        # self.plot_cdf(non_moas_min, "Non-MOAS: Nearest", "blue", "-")
        # self.plot_cdf(non_moas_max, "Non-MOAS: Furthest", "blue", "--")
        
        # plt.xlabel("Distance to Origin AS", fontsize=FONT_SIZE)
        # plt.ylabel("CDF", fontsize=FONT_SIZE)
        # plt.xlim(0,8)
        # plt.ylim(0,1)
        # plt.legend(loc="upper left", fontsize=FONT_SIZE_SMALL)
        # plt.grid(True, alpha=0.3)
        # plt.savefig(self.viz_dir / "metric_moas_specialization.png", dpi=300, bbox_inches='tight')
        # plt.close()
        
        # --- Plot 2: Topological Decay ---
        # Compare distance CDF by Depth
        hops = [0, 1, 2, 3, 4, 5]
        plt.figure(figsize=(6, 4), facecolor='white')
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(hops)))
        
        for i, h in enumerate(hops):
            # Filter triplets
            subset = df[df['hop'] == h]
            if subset.empty: continue
            
            # Sample
            if len(subset) > 5000: subset = subset.sample(5000)
            
            # Compute distances
            dists = self._calc_dist_batched(
                self.prof_emb[subset['pid'].values],
                self.as_emb[subset['aid'].values]
            )
            
            # Plot CDF manually
            sorted_d = np.sort(dists)
            y = np.arange(len(sorted_d)) / (len(sorted_d) - 1)
            plt.plot(sorted_d, y, label=f"Depth h={h}", color=colors[i], linewidth=2)
        
        plt.xlabel("Hyperbolic Distance", fontsize=FONT_SIZE)
        plt.xlim(1,7)
        plt.ylim(0,1)
        plt.ylabel("CDF", fontsize=FONT_SIZE)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right", fontsize=FONT_SIZE_SMALL)
        plt.savefig(self.viz_dir / "metric_hop_decay.png", dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_metrics_distribution(self):
        # Prepare Data
        if self.as_radii is None:
            self.as_radii = self._compute_hyperbolic_radius(self.as_emb)
            
        # Correlate Rank, Radius, Prefix Count
        # We need prefix counts first
        if self.triplets_df is not None:
            prefix_counts_series = self.triplets_df[['pid', 'aid']].drop_duplicates()['aid'].value_counts()
            prefix_counts_dict = prefix_counts_series.to_dict()
        else:
            prefix_counts_dict = {}

        data_list = []
        for asn_int, as_id in self.as_map.items():
            rank = self.rank_data.get(asn_int, -1)
            p_count = prefix_counts_dict.get(as_id, 0)
            radius = self.as_radii[as_id]
            if rank > 0:
                data_list.append({
                    'asn': asn_int, 
                    'rank': rank, 
                    'radius': radius, 
                    'p_count': p_count
                })
        
        df_metrics = pd.DataFrame(data_list)
        if df_metrics.empty:
            print("No rank data available for correlation plots.")
            return

        # Plot: Prefix Count vs Radius
        mask_pfx = df_metrics['p_count'] > 0
        if mask_pfx.any():
            plt.figure(figsize=(6, 4), facecolor='white')
            plt.scatter(df_metrics.loc[mask_pfx, 'p_count'], 
                        df_metrics.loc[mask_pfx, 'radius'], 
                        alpha=0.5, s=10)
            plt.xscale('log')
            plt.xlabel("Associated Prefix Count (Log Scale)", fontsize=FONT_SIZE)
            plt.ylabel("Hyperbolic Radius", fontsize=FONT_SIZE)
            plt.grid(True, alpha=0.3)
            self.set_adaptive_ticks()  # Add adaptive ticks
            plt.savefig(self.viz_dir / "radius_distribution_by_prefix_count.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Plot: CDF of Radius by Rank Group
        groups = {
            'Top 1000': (1, 1000),
            '1k - 5k': (1001, 5000),
            '5k - 15k': (5001, 15000),
            '15k+': (15001, float('inf'))
        }
        
        plt.figure(figsize=(6, 4), facecolor='white')
        for label, (low, high) in groups.items():
            subset = df_metrics[(df_metrics['rank'] >= low) & (df_metrics['rank'] <= high)]
            self.plot_cdf(subset['radius'].values, label, None) # Auto color
        
        plt.legend(fontsize=FONT_SIZE_SMALL)
        plt.xlabel("Hyperbolic Radius", fontsize=FONT_SIZE)
        plt.ylabel("CDF", fontsize=FONT_SIZE)
        plt.xlim(left=0)
        plt.ylim(0,1)
        plt.grid(True, alpha=0.3)
        self.set_adaptive_ticks()  # Add adaptive ticks
        plt.savefig(self.viz_dir / "rank_groups_radius_cdf.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Core AS Analysis (Radius-based)
        self._analyze_core_ases(df_metrics)
            
    def _analyze_core_ases(self, df_metrics):
        core_limit = self.threshold_profile_as / 2
        core_ases = df_metrics[df_metrics['radius'] <= core_limit].sort_values('radius')
        
        core_file = self.viz_dir / "core_ases.csv"
        core_ases.to_csv(core_file, index=False)
        suspicious = core_ases[core_ases['rank'] > 15000].head(20)
        if not suspicious.empty:
            print("\n--- Suspicious Low-Radius High-Rank ASes ---")
            print(suspicious[['asn', 'rank', 'radius', 'p_count']].to_string(index=False))
            print("-" * 40)

    def plot_semantic_convergence(self): 
        # Ground Truth Degrees
        p_ids = self.triplets[:, 0].numpy()
        degrees = np.bincount(p_ids, minlength=len(self.prof_emb))
        
        # Stratified Sample
        bins = [(1, 10), (10, 100), (100, 200), (200, 300), (300, 99999)]
        labels = ["1-10", "10-100", "100-200", "200-300", "300+"]
        
        sampled_rows = []
        for (low, high), label in zip(bins, labels):
            cands = np.where((degrees >= low) & (degrees < high))[0]
            if len(cands) > 5000:
                cands = np.random.choice(cands, 5000, replace=False)
            
            # Convert numpy indices to tensor for GPU indexing
            cands_tensor = torch.from_numpy(cands).long().to(self.device)
             
            p_vecs = self.prof_emb[cands_tensor] # (B, K, D)
            a_vecs = self.as_emb.to(self.device).unsqueeze(1) # (N_AS, 1, D)
            
            bs = 10 
            for i in range(0, len(cands), bs):
                sub_p = p_vecs[i:i+bs] # (10, K, D)
                for j in range(len(sub_p)):
                    # One profile against all ASes
                    # (1, K, D) vs (N_AS, 1, D) broadcast -> (N_AS, K)
                    target_p = sub_p[j].unsqueeze(0) 
                    
                    with torch.no_grad():
                        d = self.model.manifold.dist(target_p, a_vecs) # (N_AS, K)
                        d_min, _ = d.min(dim=1)
                        count = (d_min < self.threshold_profile_as).sum().item()
                        
                    sampled_rows.append({
                        'bin': label,
                        'pred_count': count
                    })

        if not sampled_rows: return
        df = pd.DataFrame(sampled_rows)
        
        plt.figure(figsize=(9, 4), facecolor='white')
        sns.boxplot(x='bin', y='pred_count', hue='bin', data=df, palette="Blues", legend=False, showfliers=False, whis=[1, 99], width=0.6)
        
        # Trend line
        medians = df.groupby('bin', sort=False)['pred_count'].median().reindex(labels)
        plt.plot(range(len(labels)), medians.values, marker='o', color='red', linewidth=2)
        
        plt.xlabel("Ground Truth Related AS Count", fontsize=FONT_SIZE)
        plt.ylabel(f"Acceptable AS Count", fontsize=FONT_SIZE)
        plt.ylim((450, 1100))
        # Manually set yticks
        y_ticks = np.arange(500, 1100, 150)
        plt.yticks(y_ticks)
        plt.grid(True, axis='y', alpha=0.3)        
        plt.savefig(self.viz_dir / "profile_as_semantic_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()

    def run_all(self):
        self.plot_radius_distribution()
        self.plot_distance_distributions_pdf()
        self.plot_moas_metrics()
        self.analyze_metrics_distribution()
        self.plot_semantic_convergence()
        self.plot_profile_pair_distributions()

# --- Entry Point ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--model_dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--viz_dir", type=Path, default=Path("output/visualizations"), help="Path to save visualizations")
    parser.add_argument("--embed_dim", type=int, default=24)
    parser.add_argument("--num_vectors", type=int, default=2)
    args = parser.parse_args()
    
    viz = HyperbolicVisualizer(
        args.data_dir, 
        args.model_dir, 
        args.viz_dir, 
        args.embed_dim, 
        args.num_vectors
    )
    viz.run_all()

if __name__ == "__main__":
    main()
