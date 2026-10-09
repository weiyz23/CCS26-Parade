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
from utils import unpack_128bit_to_binary, fingerprint_matches, profile_as_distances
from utils import FINGERPRINT_SIMILARITY_THRESHOLD as SIMILARITY_THRESHOLD

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

MISSING_VALUE = BORDER_VALUE + 1.0


def build_origin_split_mask(df):
    # Parser level 1 identifies origin anomalies.
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
    """Load PA, PO and AO thresholds, preferring the model's calibration."""
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
    if not (data_dir / "prefix_map.pt").exists():
        raise FileNotFoundError(f"Mapping files not found in {data_dir}")

    prefix_map = torch.load(data_dir / "prefix_map.pt")
    as_map = torch.load(data_dir / "as_map.pt")

    prefix_to_profile = torch.load(data_dir / "prefix_to_profile.pt")

    with open(data_dir / "meta.json", "r") as f:
        meta = json.load(f)

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
    raw_fps = torch.load(fp_path, map_location=device)
    fingerprints = unpack_128bit_to_binary(raw_fps)
    del raw_fps

    as_anchor_db = None

    anchor_path = data_dir / "fingerprint_anchor_db.pt"
    if anchor_path.exists():
        as_anchor_db = torch.load(anchor_path, map_location=device)
    else:
        print("[WARNING] Fingerprint Anchor DB not found. Skipping filter.")

    return model, prefix_map, as_map, prefix_to_profile, device, as_anchor_db, fingerprints

def score_candidates(df, resources, thresholds, *, anchor_for_all=False, batch_size=8192):
    """Score every input row, preserving its index and missing mappings.

    prediction includes anchor exemptions; geometry_prediction does not.
    anchor_for_all also records exemptions for geometric negatives."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model, prefix_map, as_map, prefix_to_profile, device, as_anchor_db, fingerprints = resources
    theta_1, theta_2, theta_3 = thresholds
    original_index = df.index.copy()
    df = df.copy().reset_index(drop=True)
    # Default to 0 for backward compatibility.
    if 'is_blackhole_route' not in df.columns:
        df['is_blackhole_route'] = 0
    else:
        df['is_blackhole_route'] = pd.to_numeric(df['is_blackhole_route'], errors='coerce').fillna(0).astype(int)
    if 'is_ip_leasing' not in df.columns:
        df['is_ip_leasing'] = 0
    else:
        df['is_ip_leasing'] = pd.to_numeric(df['is_ip_leasing'], errors='coerce').fillna(0).astype(int)

    required_cols = ['prefix', 'asn', 'origin']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain '{col}' column")

    unique_prefixes = df['prefix'].astype(str).unique()
    prefix_to_pid = {}
    num_profiles = len(prefix_to_profile)

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
            # Honor exact reference prefixes before fallback length limits.
            exact = str(net)
            if exact in prefix_to_profile_dict:
                prefix_to_pid[prefix] = prefix_to_profile_dict[exact]
                continue
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

    def resolve_asn_safe(x):
        try:
            val = int(x)
            res = as_map.get(val)
            return res if res is not None else -1
        except (ValueError, TypeError):
            return -1

    df_prefix_str = df['prefix'].astype(str)

    prof_indices = df_prefix_str.map(prefix_to_pid).fillna(-1).astype(int)
    as_indices = df['asn'].apply(resolve_asn_safe).astype(int)
    origin_indices = df['origin'].apply(resolve_asn_safe).astype(int)
    df['profile_id'] = prof_indices
    df['as_id'] = as_indices
    df['origin_id'] = origin_indices

    df['dist_pa'] = float('nan')
    df['dist_po'] = float('nan')
    df['dist_ao'] = float('nan')

    mask_p = (prof_indices != -1)
    mask_a = (as_indices != -1)
    mask_o = (origin_indices != -1)

    mask_res_missing = (~mask_p)
    df.loc[mask_res_missing, 'dist_pa'] = MISSING_VALUE

    compute_pa_idx = df[mask_p & mask_a].index.tolist()
    if compute_pa_idx:
        p_tensor = torch.tensor(prof_indices.loc[compute_pa_idx].values, device=device)
        a_tensor = torch.tensor(as_indices.loc[compute_pa_idx].values, device=device)

        df.loc[compute_pa_idx, 'dist_pa'] = profile_as_distances(model, p_tensor, a_tensor, batch_size=batch_size).cpu().numpy()

    compute_po_idx = df[mask_p & mask_o].index.tolist()
    if compute_po_idx:
        p_tensor = torch.tensor(prof_indices.loc[compute_po_idx].values, device=device)
        o_tensor = torch.tensor(origin_indices.loc[compute_po_idx].values, device=device)

        df.loc[compute_po_idx, 'dist_po'] = profile_as_distances(model, p_tensor, o_tensor, batch_size=batch_size).cpu().numpy()

    compute_ao_idx = df[mask_a & mask_o].index.tolist()
    if compute_ao_idx:
        a_tensor = torch.tensor(as_indices.loc[compute_ao_idx].values, device=device)
        o_tensor = torch.tensor(origin_indices.loc[compute_ao_idx].values, device=device)

        dists = []
        with torch.no_grad():
            for i in range(0, len(a_tensor), batch_size):
                b_a = a_tensor[i:i+batch_size]
                b_o = o_tensor[i:i+batch_size]

                vec_a = model.as_embedding[b_a]
                vec_o = model.as_embedding[b_o]
                d = model.manifold.dist(vec_a.unsqueeze(1), vec_o.unsqueeze(1)).view(-1)
                dists.append(d.cpu())
        df.loc[compute_ao_idx, 'dist_ao'] = torch.cat(dists).numpy()


    with torch.no_grad():
        as_norms = model.as_embedding.norm(p=2, dim=1)

        factors = torch.ones_like(as_norms)

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

    # Scale thresholds only for source anomalies (asn == origin).
    mask_diff_source = (df['asn'] != df['origin']).values
    factor_as[mask_diff_source] = 1.0
    factor_origin[mask_diff_source] = 1.0
    df['factor_as'] = factor_as
    df['factor_origin'] = factor_origin

    c_pa_high = (df['dist_pa'] > theta_1 * factor_as).fillna(False)
    c_po_high = (df['dist_po'] > theta_2 * factor_origin).fillna(False)
    c_ao_high = (df['dist_ao'] > theta_3 * factor_as).fillna(False)

    mask_source = (df['origin'] == df['asn']).fillna(False)

    mask_res_missing = (~mask_p) | ((~mask_a) & (~mask_o))

    case_full = mask_p & mask_a & mask_o
    anom_full = case_full & ((mask_source & (c_pa_high | c_po_high)) | (~mask_source & c_pa_high & (c_po_high | c_ao_high)))

    case_origin_missing = mask_p & mask_a & (~mask_o)
    anom_origin_missing = case_origin_missing & c_pa_high

    case_as_missing = mask_p & (~mask_a) & mask_o
    anom_as_missing = case_as_missing & c_po_high

    mask_res_anomaly = anom_full | anom_origin_missing | anom_as_missing
    df['geometry_prediction'] = mask_res_anomaly

    mask_anchor_normal = pd.Series(False, index=df.index)
    if as_anchor_db is not None:
        candidate_mask = mask_a & mask_p
        if not anchor_for_all:
            candidate_mask &= mask_res_anomaly

        if candidate_mask.any():
             cand_df = pd.DataFrame({
                 'row_idx': df.index[candidate_mask],
                 'as_id': as_indices[candidate_mask],
                 'pid': prof_indices[candidate_mask]
             })

             for internal_as_id, group in cand_df.groupby('as_id'):
                 as_id_key = int(internal_as_id)

                 if as_id_key not in as_anchor_db:
                     continue

                 group_pids = torch.tensor(group['pid'].values, dtype=torch.long, device=device)
                 group_fps = fingerprints[group_pids] # (B, 128)

                 group_indices = group['row_idx'].values
                 is_normal = fingerprint_matches(group_fps, as_anchor_db[as_id_key], batch_size=batch_size).cpu().numpy()
                 mask_anchor_normal.loc[group_indices[is_normal]] = True

    df['missing'] = mask_res_missing
    df['anchor_filtered'] = mask_anchor_normal
    df['prediction'] = mask_res_anomaly & (~mask_anchor_normal)
    df.index = original_index
    return df


def detect_anomalies(input_csv, output_anomalous_csv, output_missing_csv, data_dir, model_dir, embed_dim, num_vectors):
    resources = load_resources(data_dir, model_dir, embed_dim, num_vectors)
    thresholds = load_thresholds(data_dir, model_dir)
    df = pd.read_csv(input_csv, dtype={'timestamp': int, 'prefix': str, 'asn': int, 'level': int, 'origin': int, 'upstream_asns': str})
    input_columns = set(df.columns)
    df = score_candidates(df, resources, thresholds)
    mask_res_anomaly = df['prediction'].copy()
    mask_res_missing = df['missing'].copy()
    exempt_count = df['anchor_filtered'].sum()
    if exempt_count:
        print(f"Fingerprint Filter exempted {exempt_count} anomalies.")
    # Keep the established CSV schema; detailed scores are available separately.
    scoring_columns = {
        'profile_id', 'as_id', 'origin_id', 'factor_as', 'factor_origin',
        'missing', 'anchor_filtered', 'geometry_prediction', 'prediction',
    }
    df = df.drop(columns=sorted(scoring_columns - input_columns))

    categories = {
        'anomalous': mask_res_anomaly,
        'missing': mask_res_missing
    }

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

        if cat_name == 'anomalous':
            df_agg = df_cat.sort_values('dist_pa', ascending=False).drop_duplicates(subset='prefix')
            if 'upstream_asns' in df_agg.columns:
                if not df_agg.empty:
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

                    # Preserve the anomaly ASN; group_asn is only for aggregation.
                    df_agg['group_asn'] = df_agg['asn'].apply(uf.find)

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

        origin_mask = build_origin_split_mask(df_agg)
        df_agg_origin = df_agg[origin_mask]
        df_agg_non_origin = df_agg[~origin_mask]

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
