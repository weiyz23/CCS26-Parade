import torch
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import sys
import time
import ipaddress                            
from collections import Counter

sys.path.append(str(Path(__file__).parent))
from model import HyperbolicRoutingModel, BORDER_VALUE
from utils import unpack_128bit_to_binary, compute_affinity

class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        # Directed union: x points to y (child to parent)
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

SIMILARITY_THRESHOLD = 0.75
MISSING_VALUE = BORDER_VALUE + 1.0


def build_origin_split_mask(df):
    # Semantic level is produced upstream by parser; origin anomalies are level == 1.
    if 'level' in df.columns:
        levels = pd.to_numeric(df['level'], errors='coerce')
        return (levels == 1).fillna(False)

    # Backward compatibility for legacy inputs without level.
    if 'asn' in df.columns and 'origin' in df.columns:
        asn_vals = pd.to_numeric(df['asn'], errors='coerce')
        origin_vals = pd.to_numeric(df['origin'], errors='coerce')
        return (asn_vals == origin_vals).fillna(False)

    return pd.Series(False, index=df.index)

def load_thresholds(data_dir, model_dir=None):
    """
    Returns: (theta_1, theta_2, theta_3) and optionally (theta_semantic)
    """
    if model_dir is not None:
        threshold_path = model_dir / "thresholds.json"
        if threshold_path.exists():
            with open(threshold_path, "r") as f:
                meta = json.load(f)
                t1 = meta.get('threshold_profile_as')
                t2 = meta.get('threshold_profile_origin')
                t3 = meta.get('threshold_as_origin')
                if t1 is not None and t2 is not None and t3 is not None:
                    print(f"Loaded thresholds from {threshold_path}: T1={t1:.4f}, T2={t2:.4f}, T3={t3:.4f}")
                    return t1, t2, t3

    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                t1 = meta.get('threshold_profile_as')
                t2 = meta.get('threshold_profile_origin')
                t3 = meta.get('threshold_as_origin')
                
                if t1 is not None and t2 is not None and t3 is not None:
                    print(f"Loaded optimal thresholds from meta.json: T1={t1:.4f}, T2={t2:.4f}, T3={t3:.4f}")
                    return t1, t2, t3
        except Exception as e:
            raise ValueError("Thresholds not found in meta.json. Please run calc_thresholds.py first.")

def load_resources(data_dir, model_dir, embed_dim, num_vectors):
    # Load mappings
    if not (data_dir / "prefix_map.pt").exists():
        raise FileNotFoundError(f"Mapping files not found in {data_dir}")

    prefix_map = torch.load(data_dir / "prefix_map.pt")
    as_map = torch.load(data_dir / "as_map.pt")
    
    prefix_to_profile = torch.load(data_dir / "prefix_to_profile.pt")
        
    with open(data_dir / "meta.json", "r") as f:
        meta = json.load(f)
        
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HyperbolicRoutingModel(
        num_as=meta["num_asns"],
        num_profiles=meta["num_profiles"],
        embed_dim=embed_dim,
        num_vectors=num_vectors
    ).to(device)
    
    model_path = model_dir / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError("No model checkpoint found.")
            
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    fp_path = data_dir / "profile_fingerprints.pt"
    if not fp_path.exists():
        raise FileNotFoundError(f"Fingerprints not found at {fp_path}")
    # Load to GPU (or CPU if VRAM is tight, but GPU is recommended for the comparison)
    raw_fps = torch.load(fp_path, map_location=device)
    fingerprints = unpack_128bit_to_binary(raw_fps)
    del raw_fps

    # Load Anchor DB (Ensure it loads the NEW fingerprint db)
    as_anchor_db = None
    
    # Check data_dir first as analyze_anchors.py saves there by default now
    anchor_path = data_dir / "fingerprint_anchor_db.pt"
    if anchor_path.exists():
        as_anchor_db = torch.load(anchor_path, map_location=device)
    else:
        print("[WARNING] Fingerprint Anchor DB not found. Skipping filter.")
    
    return model, prefix_map, as_map, prefix_to_profile, device, as_anchor_db, fingerprints

def detect_anomalies(input_csv, output_anomalous_csv, output_missing_csv, data_dir, model_dir, embed_dim, num_vectors):
    model, prefix_map, as_map, prefix_to_profile, device, as_anchor_db, fingerprints = load_resources(data_dir, model_dir, embed_dim, num_vectors)
    
    # 1. Compute Thresholds
    theta_1, theta_2, theta_3 = load_thresholds(data_dir, model_dir)
    df = pd.read_csv(input_csv, dtype={'timestamp': int, 'prefix': str, 'asn': int, 'level': int, 'origin': int, 'upstream_asns': str})
    # Default to 0 for backward compatibility.
    if 'is_blackhole_route' not in df.columns:
        df['is_blackhole_route'] = 0
    else:
        df['is_blackhole_route'] = pd.to_numeric(df['is_blackhole_route'], errors='coerce').fillna(0).astype(int)
    if 'is_ip_leasing' not in df.columns:
        df['is_ip_leasing'] = 0
    else:
        df['is_ip_leasing'] = pd.to_numeric(df['is_ip_leasing'], errors='coerce').fillna(0).astype(int)
    
    # Check for required columns
    required_cols = ['prefix', 'asn', 'origin']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain '{col}' column")
            
    # 2. Resolve Mappings
    # (a) Prefix -> Profile ID
    unique_prefixes = df['prefix'].astype(str).unique()
    prefix_to_pid = {}
    num_profiles = len(prefix_to_profile)
    
    # Precompute all possible prefixes from prefix_map
    prefix_to_profile_dict = {}
    for p_str, p_idx in prefix_map.items():
        if p_idx < num_profiles:
            pid = prefix_to_profile[p_idx].item()
            if pid != -1:
                prefix_to_profile_dict[p_str] = pid
    
    for prefix in unique_prefixes:
        profile_id = -1
        try:
            net = ipaddress.ip_network(prefix, strict=False)
            candidates = [net]
            curr = net
            while curr.prefixlen > 8:
                curr = curr.supernet()
                candidates.append(curr)
            
            for cand in candidates:
                is_v4 = (cand.version == 4)
                if is_v4 and cand.prefixlen < 8:
                    break
                if not is_v4 and cand.prefixlen < 32:
                    break
                
                cand_str = str(cand)
                if cand_str in prefix_to_profile_dict:
                    profile_id = prefix_to_profile_dict[cand_str]
                    break
        except ValueError:
            pass
            
        prefix_to_pid[prefix] = profile_id

    # (b) ASN -> AS ID
    def resolve_asn_safe(x):
        try:
            val = int(x)
            res = as_map.get(val)
            return res if res is not None else -1
        except (ValueError, TypeError):
            return -1

    df_prefix_str = df['prefix'].astype(str)
    
    # Map all relevant IDs
    prof_indices = df_prefix_str.map(prefix_to_pid).fillna(-1).astype(int)
    as_indices = df['asn'].apply(resolve_asn_safe).astype(int)
    origin_indices = df['origin'].apply(resolve_asn_safe).astype(int)
    
    # Initialize distances
    df['dist_pa'] = float('nan') # Profile - AS
    df['dist_po'] = float('nan') # Profile - Origin
    df['dist_ao'] = float('nan') # AS - Origin
    
    # Mask definitions
    mask_p = (prof_indices != -1)
    mask_a = (as_indices != -1)
    mask_o = (origin_indices != -1)
    
    # 1. Missing: P missing
    mask_res_missing = (~mask_p)
    df.loc[mask_res_missing, 'dist_pa'] = MISSING_VALUE
    
    # 2. Computation - Compute Partial Distances
    # To optimize, we compute unique pairs needed
    
    # Compute PA (Needs P & A)
    compute_pa_idx = df[mask_p & mask_a].index.tolist()
    if compute_pa_idx:
        p_tensor = torch.tensor(prof_indices.loc[compute_pa_idx].values, device=device)
        a_tensor = torch.tensor(as_indices.loc[compute_pa_idx].values, device=device)
        
        dists = []
        batch_size = 8192  # Increased for better GPU utilization
        with torch.no_grad():
            for i in range(0, len(p_tensor), batch_size):
                b_p = p_tensor[i:i+batch_size]
                b_a = a_tensor[i:i+batch_size]
                d = model(b_p, b_a.unsqueeze(1)).squeeze(2).min(dim=1)[0]
                dists.append(d.cpu())
        df.loc[compute_pa_idx, 'dist_pa'] = torch.cat(dists).numpy()

    # Compute PO (Needs P & O)
    compute_po_idx = df[mask_p & mask_o].index.tolist()
    if compute_po_idx:
        p_tensor = torch.tensor(prof_indices.loc[compute_po_idx].values, device=device)
        o_tensor = torch.tensor(origin_indices.loc[compute_po_idx].values, device=device)
        
        dists = []
        batch_size = 8192  # Increased for better GPU utilization
        with torch.no_grad():
            for i in range(0, len(p_tensor), batch_size):
                b_p = p_tensor[i:i+batch_size]
                b_o = o_tensor[i:i+batch_size]
                d = model(b_p, b_o.unsqueeze(1)).squeeze(2).min(dim=1)[0]
                dists.append(d.cpu())
        df.loc[compute_po_idx, 'dist_po'] = torch.cat(dists).numpy()

    # Compute AO (Needs A & O)
    compute_ao_idx = df[mask_a & mask_o].index.tolist()
    if compute_ao_idx:
        a_tensor = torch.tensor(as_indices.loc[compute_ao_idx].values, device=device)
        o_tensor = torch.tensor(origin_indices.loc[compute_ao_idx].values, device=device)
        
        dists = []
        batch_size = 8192  # Increased for better GPU utilization
        with torch.no_grad():
            for i in range(0, len(a_tensor), batch_size):
                b_a = a_tensor[i:i+batch_size]
                b_o = o_tensor[i:i+batch_size]
                
                vec_a = model.as_embedding[b_a]
                vec_o = model.as_embedding[b_o]
                # (B, D) -> (B, 1, D) vs (B, 1, D)
                # dist returns (B, 1) usually. We want (B,)
                d = model.manifold.dist(vec_a.unsqueeze(1), vec_o.unsqueeze(1)).view(-1)
                dists.append(d.cpu())
        df.loc[compute_ao_idx, 'dist_ao'] = torch.cat(dists).numpy()

    # --- Decision Logic ---
    
    # Compute Scaling Factors per AS
    with torch.no_grad():
        # norm of (N_as, D) -> (N_as)
        as_norms = model.as_embedding.norm(p=2, dim=1)
        
        factors = torch.ones_like(as_norms)
        
        # Piecewise function logic
        mask_low = (as_norms < 0.8)
        mask_high = (as_norms > 0.95)
        mask_mid = (~mask_low) & (~mask_high)
        
        factors[mask_low] = 0.8
        factors[mask_high] = 1.0
        factors[mask_mid] = 0.8 + (as_norms[mask_mid] - 0.8) * (4.0 / 3.0)
        scaling_factors_np = factors.cpu().numpy()

    def get_factors(indices):
        f = np.ones(len(indices), dtype=np.float32)
        valid = (indices != -1)
        if valid.any():
             f[valid] = scaling_factors_np[indices[valid]]
        return f

    factor_as = get_factors(as_indices.values)
    factor_origin = get_factors(origin_indices.values)

    # ONLY apply scaling when Origin == AS (context: missing AO link metric, so tighten PA/PO)
    mask_diff_source = (df['asn'] != df['origin']).values
    factor_as[mask_diff_source] = 1.0
    factor_origin[mask_diff_source] = 1.0

    c_pa_high = (df['dist_pa'] > theta_1 * factor_as).fillna(False)
    c_pa_low  = (df['dist_pa'] <= theta_1 * factor_as).fillna(False)
    c_po_high = (df['dist_po'] > theta_2 * factor_origin).fillna(False)
    c_ao_high = (df['dist_ao'] > theta_3 * factor_as).fillna(False)

    mask_source = (df['origin'] == df['asn']).fillna(False)

    # 1. Missing
    mask_res_missing = (~mask_p) | ((~mask_a) & (~mask_o))

    # 2. Anomaly
    # Case 1: Full Info (P, A, O exist)
    case_full = mask_p & mask_a & mask_o
    anom_full = case_full & ((mask_source & (c_pa_high | c_po_high)) | (~mask_source & c_pa_high & (c_po_high | c_ao_high)))

    # Case 2a: Origin Missing (A exists, O missing)
    case_origin_missing = mask_p & mask_a & (~mask_o)
    anom_origin_missing = case_origin_missing & c_pa_high

    # Case 2b: AS Missing (A missing, O exists)
    case_as_missing = mask_p & (~mask_a) & mask_o
    anom_as_missing = case_as_missing & c_po_high
    
    mask_res_anomaly = anom_full | anom_origin_missing | anom_as_missing

    # 4. Anchor Filter (Fingerprint Version)
    if as_anchor_db is not None:
        mask_anchor_normal = pd.Series(False, index=df.index)
        
        # Filter strictly those rows that are considered anomalies so far
        candidate_mask = mask_res_anomaly & mask_a & mask_p
        
        if candidate_mask.any():
             cand_df = pd.DataFrame({
                 'row_idx': df.index[candidate_mask],
                 'as_id': as_indices[candidate_mask],
                 'pid': prof_indices[candidate_mask]
             })
             
             # Group by AS to batch process
             for internal_as_id, group in cand_df.groupby('as_id'):
                 as_id_key = int(internal_as_id)
                 
                 # Check if this AS has anchors in DB
                 if as_id_key not in as_anchor_db:
                     continue
                     
                 # Anchors are now raw Fingerprint Tensors (M, 128)
                 anchors_fps = as_anchor_db[as_id_key].to(device)
                 if anchors_fps.size(0) == 0: continue
                 
                 # Unpack anchors from (M, 2) to (M, 128)
                 anchors_fps = unpack_128bit_to_binary(anchors_fps)
                 
                 # Prepare Candidate Fingerprints
                 # Get PIDs for current anomalies
                 group_pids = torch.tensor(group['pid'].values, dtype=torch.long, device=device)
                 # Look up their fingerprints from the global tensor
                 group_fps = fingerprints[group_pids] # (B, 128)
                 
                 # Batch processing
                 # SimHash comparison (Broadcasting optimized)
                 BATCH_SIZE = 8192  # Increased for better GPU utilization
                 group_indices = group['row_idx'].values
                 
                 for i in range(0, len(group), BATCH_SIZE):
                     b_fps = group_fps[i : i+BATCH_SIZE] # (b, 128)
                     
                     # --- SimHash / Hamming Similarity Calculation ---
                     # Compute hamming affinity efficiently via broadcasting using utils.compute_affinity
                     sims = compute_affinity(b_fps.unsqueeze(1), anchors_fps.unsqueeze(0)) # Result: (b, M)
                     
                     # Find best match for each candidate
                     # max_sims: (b,)
                     max_sims, _ = sims.max(dim=1)
                     
                     # Check threshold
                     is_normal = (max_sims > SIMILARITY_THRESHOLD).cpu().numpy()
                     
                     # Update mask
                     if is_normal.any():
                         normal_row_indices = group_indices[i : i+BATCH_SIZE][is_normal]
                         mask_anchor_normal.loc[normal_row_indices] = True
        
        # Apply exemption
        exempt_count = mask_anchor_normal.sum()
        if exempt_count > 0:
            print(f"Fingerprint Filter exempted {exempt_count} anomalies.")
            mask_res_anomaly = mask_res_anomaly & (~mask_anchor_normal)

    # Define Categories
    categories = {
        'anomalous': mask_res_anomaly, # "Anomalous" contains the detected high-risk samples
        'missing': mask_res_missing
    }

    # Output Files
    output_files = {
        'anomalous': output_anomalous_csv,
        'missing': output_missing_csv
    }
    
    Path(output_anomalous_csv).parent.mkdir(parents=True, exist_ok=True)

    summary_stats = []

    for cat_name, mask in categories.items():
        if output_files[cat_name] is None:
            continue
            
        df_cat = df[mask].copy()
        if 'dist_pa' in df_cat.columns:
            df_cat = df_cat.sort_values('dist_pa', ascending=False)
        
        # For stats aggregation (only for anomalous)
        if cat_name == 'anomalous':
            df_agg = df_cat.sort_values('dist_pa', ascending=False).drop_duplicates(subset='prefix')
            # Aggregation Logic on df_agg
            if 'upstream_asns' in df_agg.columns:
                if not df_agg.empty:
                    # Group by asn to get all anomalies per AS in df_agg
                    asn_groups = df_agg.groupby('asn')
                    uf = UnionFind()
                    for asn, group in asn_groups:
                        upstream_series = group['upstream_asns'].dropna().str.split(':').explode().dropna()
                        upstream_series = upstream_series[upstream_series != '']
                        upstream_lists = upstream_series.astype(int).tolist()
                        if upstream_lists:
                            upstream_counts = Counter(upstream_lists)
                            total_anoms = len(group)
                            for up_as, count in upstream_counts.items():
                                if count / total_anoms >= 0.7:
                                    uf.union(asn, up_as)
                    
                    # Keep original anomaly ASN for row semantics; use grouped ASN only for aggregation stats.
                    df_agg['group_asn'] = df_agg['asn'].apply(uf.find)
            
            # Aggregated stats
            rows_agg = len(df_agg)
            prefix_count_agg = df_agg['prefix'].nunique()
            as_count_agg = df_agg['group_asn'].nunique() if 'group_asn' in df_agg.columns else df_agg['asn'].nunique()
            summary_stats.append({
                'category': cat_name,
                'subtype': 'aggregated',
                'rows': rows_agg,
                'prefix_count': prefix_count_agg,
                'as_count': as_count_agg
            })
        else:
            df_agg = df_cat
        
        base_filename = str(output_files[cat_name]).replace('.csv', '')
        
        # Output raw (all initial results without filtering)
        df_cat.to_csv(f"{base_filename}_raw.csv", index=False)
        rows_raw = len(df_cat)
        prefix_count_raw = df_cat['prefix'].nunique()
        as_count_raw = df_cat['asn'].nunique()
        summary_stats.append({
            'category': cat_name,
            'subtype': 'raw',
            'rows': rows_raw,
            'prefix_count': prefix_count_raw,
            'as_count': as_count_raw
        })
        
        # Split aggregated for output and stats using true origin semantics when available.
        origin_mask = build_origin_split_mask(df_agg)
        df_agg_origin = df_agg[origin_mask]
        df_agg_non_origin = df_agg[~origin_mask]
            
        # Origin
        df_agg_origin.to_csv(f"{base_filename}_origin.csv", index=False)
        rows_origin = len(df_agg_origin)
        unique_as_origin = df_agg_origin['group_asn'].nunique() if 'group_asn' in df_agg_origin.columns else df_agg_origin['asn'].nunique()
        prefix_count_origin = df_agg_origin['prefix'].nunique()
        as_count_origin = unique_as_origin
        summary_stats.append({
            'category': cat_name,
            'subtype': 'origin',
            'rows': rows_origin,
            'prefix_count': prefix_count_origin,
            'as_count': as_count_origin
        })
        
        # Non-Origin
        df_agg_non_origin.to_csv(f"{base_filename}_non_origin.csv", index=False)
        rows_non = len(df_agg_non_origin)
        unique_as_non = df_agg_non_origin['group_asn'].nunique() if 'group_asn' in df_agg_non_origin.columns else df_agg_non_origin['asn'].nunique()
        prefix_count_non = df_agg_non_origin['prefix'].nunique()
        as_count_non = unique_as_non
        summary_stats.append({
            'category': cat_name,
            'subtype': 'non_origin',
            'rows': rows_non,
            'prefix_count': prefix_count_non,
            'as_count': as_count_non
        })

    # Save summary stats
    if summary_stats:
        stats_path = Path(output_anomalous_csv).parent / "detection_stats.csv"
        pd.DataFrame(summary_stats).to_csv(stats_path, index=False)

if __name__ == "__main__":
    total_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Detect anomalies using Hyperbolic BGP Model")
    parser.add_argument("--input", type=str, default="./processed_data/anomaly_links.csv", help="Input CSV file path")
    parser.add_argument("--anomalous-output", type=str, default="./output/anomalous.csv", help="Output CSV file path for anomalies")
    parser.add_argument("--missing-output", type=str, default="./output/missing.csv", help="Output CSV file path for missing profiles")
    parser.add_argument("--data_dir", type=Path, default=Path("processed_data"), help="Path to processed data")
    parser.add_argument("--model_dir", type=Path, default=Path("saved_models"), help="Path to load models")
    parser.add_argument("--embed_dim", type=int, default=24, help="Embedding dimension")
    parser.add_argument("--num_vectors", type=int, default=2, help="Number of vectors per profile")
    
    args = parser.parse_args()
    
    detect_anomalies(args.input, args.anomalous_output, args.missing_output, args.data_dir, args.model_dir, args.embed_dim, args.num_vectors)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total execution time: {total_time:.2f} seconds")
