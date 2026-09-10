import pandas as pd
import json
import torch
import argparse
from pathlib import Path
import sys
import requests
import time
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. Resource Loading & Mapping Construction
# ==========================================
def find_dataset_root(data_dir):
    data_dir = Path(data_dir).resolve()
    if data_dir.name == "processed_data":
        return data_dir.parent
    if data_dir.parent.name == "processed_data":
        return data_dir.parent.parent
    return data_dir


def resolve_as_info_path(data_dir):
    dataset_root = find_dataset_root(data_dir)
    fixed_name = dataset_root / "as_info.jsonl"
    if fixed_name.exists() and fixed_name.is_file():
        return fixed_name

    monthly_files = sorted(dataset_root.glob("as_info.[0-9][0-9][0-9][0-9].jsonl"))
    if monthly_files:
        return monthly_files[-1]

    raise FileNotFoundError(
        f"No AS info file found under {dataset_root}. "
        "Expected as_info.jsonl or as_info.YYMM.jsonl"
    )


def normalize_org_name(name):
    if not name:
        return None

    text = str(name).strip().lower()
    if not text:
        return None

    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(inc|llc|ltd|limited|corp|corporation|company|co|holdings|group)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_profile_org_history(cache_data, org_id_to_name_norm):
    profile_org_ids = {}
    profile_org_names = {}

    if isinstance(cache_data, dict) and "org_ids" in cache_data:
        raw_ids = cache_data.get("org_ids", {})
        raw_names = cache_data.get("org_names", {})
        for pid, org_ids in raw_ids.items():
            pid_int = int(pid)
            ids = {str(org_id) for org_id in org_ids if org_id}
            names = {
                normalize_org_name(name)
                for name in raw_names.get(pid, raw_names.get(pid_int, []))
                if normalize_org_name(name)
            }
            for org_id in ids:
                if org_id in org_id_to_name_norm:
                    names.add(org_id_to_name_norm[org_id])
            if ids:
                profile_org_ids[pid_int] = ids
            if names:
                profile_org_names[pid_int] = names
        return profile_org_ids, profile_org_names

    if isinstance(cache_data, dict):
        for pid, org_ids in cache_data.items():
            pid_int = int(pid)
            ids = {str(org_id) for org_id in org_ids if org_id}
            names = {
                org_id_to_name_norm[org_id]
                for org_id in ids
                if org_id in org_id_to_name_norm
            }
            if ids:
                profile_org_ids[pid_int] = ids
            if names:
                profile_org_names[pid_int] = names

    return profile_org_ids, profile_org_names


def load_resources(data_dir, as_info_path):    
    # 1. Load AS Info (ASN -> Org ID)
    as_org = {}
    as_org_name = {}
    as_org_name_norm = {}
    org_id_to_name_norm = {}
    
    if as_info_path.exists():
        with open(as_info_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    asn = int(item['asn'])
                    
                    if 'organization' in item and item['organization']:
                        org_id = item['organization'].get('orgId')
                        org_name = item['organization'].get('orgName')
                        org_name_norm = normalize_org_name(org_name)
                        
                        if org_id:
                            as_org[asn] = org_id
                        if org_name:
                            as_org_name[asn] = org_name
                        if org_name_norm:
                            as_org_name_norm[asn] = org_name_norm
                            if org_id and org_id not in org_id_to_name_norm:
                                org_id_to_name_norm[org_id] = org_name_norm
                except:
                    continue
    else:
        print(f"Warning: AS info file not found at {as_info_path}")

    # 2. Load Torch Mappings & History Cache
    try:
        prefix_map = torch.load(data_dir / "prefix_map.pt")
        prefix_to_profile = torch.load(data_dir / "prefix_to_profile.pt") # Tensor [N_prefix] -> ProfileID
        
        # Load pre-computed history (Created by build_dataset.py)
        cache_path = data_dir / "history_cache.pt"
        if not cache_path.exists():
             raise FileNotFoundError(f"History cache not found at {cache_path}. Please run preprocess/build_dataset.py.")

        cache_data = torch.load(cache_path)
        profile_org_ids, profile_org_names = normalize_profile_org_history(cache_data, org_id_to_name_norm)

    except Exception as e:
        print(f"Error loading torch files: {e}")
        sys.exit(1)

    return prefix_map, prefix_to_profile, profile_org_ids, profile_org_names, as_org, as_org_name, as_org_name_norm

# ==========================================
# 2. RPKI Validation Logic
# ==========================================
RIPE_API_URL = "https://stat.ripe.net/data/rpki-validation/data.json"

def validate_rpki_single(prefix, asn, timestamp, session=None):
    """Query RIPEstat API for current RPKI status.

    The RIPEstat rpki-validation endpoint does not honor a historical timestamp,
    so the timestamp is retained only to preserve the caller contract and cache key.
    """
    params = {
        'resource': asn,
        'prefix': prefix,
    }
    
    try:
        # Use session if provided for connection pooling
        requester = session if session else requests
        response = requester.get(RIPE_API_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'status' in data['data']:
                return data['data']['status'], data['data'].get('validating_roas', [])
    except Exception:
        return None, None
    return 'error', None

def process_rpki_batch(chunk_keys):
    """Process a batch of (prefix, asn, timestamp) keys for RPKI validation"""
    results_map = {} 
    
    # Configure retry strategy for robustness
    retry_strategy = requests.adapters.Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    
    # Creates a session for this batch (thread)
    with requests.Session() as session:
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        for prefix, asn, timestamp in chunk_keys:
            status, roas = validate_rpki_single(prefix, asn, timestamp, session)
            results_map[(prefix, asn, timestamp)] = (status, roas)
            # Reduced sleep time as we have retry logic now
            time.sleep(0.02) 
    return results_map

def filter_false_positives(df, prefix_map, prefix_to_profile, profile_org_ids, profile_org_names, as_org, as_org_name, as_org_name_norm, check_rpki=False):
    fp_indices = []
    fp_reasons = []
    # --- RPKI Validation (Only if requested, usually for origin anomalies) ---
    if check_rpki:
        # 1. Deduplicate & Group by ASN
        queries_by_asn = defaultdict(set)
        for row in df.itertuples(index=False):
            try:
                queries_by_asn[int(row.asn)].add((str(row.prefix), int(row.asn), row.timestamp))
            except ValueError:
                continue
        
        queries_to_run = []
        skipped_queries_by_asn = {}
        asn_samples = {}
        
        # Sampling Config
        SAMPLING_THRESHOLD = 20
        SAMPLE_SIZE = 10
        
        total_unique_queries = 0
        for asn, q_set in queries_by_asn.items():
            total_unique_queries += len(q_set)
            q_list = list(q_set)
            
            if len(q_list) > SAMPLING_THRESHOLD:
                # Sample a subset
                sample = random.sample(q_list, SAMPLE_SIZE)
                queries_to_run.extend(sample)
                asn_samples[asn] = sample
                # Mark others as skipped
                skipped_queries_by_asn[asn] = list(q_set - set(sample))
            else:
                queries_to_run.extend(q_list)
        
        # 2. Batch Processing
        # Dynamic chunk size: For small batches, use smaller chunks to utilize concurrency
        total_q = len(queries_to_run)
        if total_q < 50:
            chunk_size = 2
        else:
            chunk_size = 5

        chunks = [queries_to_run[i:i + chunk_size] for i in range(0, total_q, chunk_size)]
        rpki_results = {}
        max_w = min(24, len(chunks)) if len(chunks) > 0 else 1
        
        with ThreadPoolExecutor(max_workers=max_w) as executor:
            futures = [executor.submit(process_rpki_batch, chunk) for chunk in chunks]
            for future in as_completed(futures):
                try:
                    rpki_results.update(future.result())
                except Exception as e:
                    print(f"RPKI Batch Error: {e}")
                    
        # 3. Inference for Skipped Queries
        inferred_valid_count = 0
        for asn, skipped in skipped_queries_by_asn.items():
            samples = asn_samples.get(asn, [])
            if not samples: continue
            
            valid_count = 0
            for q in samples:
                status, _ = rpki_results.get(q, (None, None))
                if status == 'valid':
                    valid_count += 1
            
            # Heuristic: If > 90% of samples are valid, assume all skipped are valid
            if valid_count / len(samples) >= 0.8:
                for q in skipped:
                    rpki_results[q] = ('valid', ['INFERRED_BY_SAMPLING'])
                    inferred_valid_count += 1
        
        if inferred_valid_count > 0:
            print(f"Inferred {inferred_valid_count} 'valid' RPKI results via sampling logic.")
            
    else:
        rpki_results = {}

    # --- Iteration ---
    # Pre-resolve profiles for all prefixes to avoid repeated lookups (Exact Match ONLY)
    unique_prefixes = df['prefix'].astype(str).unique()
    prefix_to_pid_map = {}
    
    num_profiles = len(prefix_to_profile)
    
    for p_str in unique_prefixes:
        # Exact Match Only
        if p_str in prefix_map:
            p_idx = prefix_map[p_str]
            if p_idx < num_profiles:
                pid = prefix_to_profile[p_idx].item()
                if pid != -1:
                    prefix_to_pid_map[p_str] = pid


    # Initialize fp_reasons if not present
    if "fp_reasons" not in df.columns:
        df["fp_reasons"] = ""

    # Iterate rows using itertuples (Significantly faster than iterrows)
    for row in df.itertuples(index=True):
        is_fp = False
        reasons = [] 
        
        idx = row.Index
        prefix = str(row.prefix)
        try:
            anomaly_asn = int(row.asn)
            anomaly_hop = int(row.level) 
        except ValueError:
            continue
            
        timestamp = row.timestamp

        # 1. RPKI Filter (Valid => False Positive)
        if check_rpki:
            key = (prefix, anomaly_asn, timestamp)
            status, roas = rpki_results.get(key, (None, None))
            if status == 'valid':
                is_fp = True
                reasons.append(f"RPKI_VALID_CURRENT (ROAs: {len(roas) if roas else 0})")

        # 2. Organization/Org-Hop Filter (Optimized)
        anom_org_id = as_org.get(anomaly_asn)
        anom_org_name_norm = as_org_name_norm.get(anomaly_asn)

        # Find pre-resolved pid
        pid = prefix_to_pid_map.get(prefix)
        if pid is not None:
            org_id_hist = profile_org_ids.get(pid, set())
            org_name_hist = profile_org_names.get(pid, set())

            if anom_org_id and anom_org_id in org_id_hist:
                is_fp = True
                reasons.append(f"ORG_ID_MATCH (Org {as_org_name.get(anomaly_asn)} found in history)")
            elif anom_org_name_norm and anom_org_name_norm in org_name_hist:
                is_fp = True
                reasons.append(f"ORG_NAME_MATCH (Org {as_org_name.get(anomaly_asn)} found in history)")

        if is_fp:
            fp_indices.append(idx)
            df.at[idx, "fp_reasons"] = "; ".join(reasons)
    
    # Split DataFrames
    fp_df = df.loc[fp_indices].copy()
    anomalies_df = df.drop(index=fp_indices).copy()
    
    return fp_df, anomalies_df

# ==========================================
# 3. Main Execution Logic
# ==========================================
def process_and_update_stats(results_dir, prefix_map, prefix_to_profile, profile_org_history, as_org, as_org_name):
    """
    Process all anomaly files (anomalous, missing, unknown pairs for origin/non_origin),
    apply filters, and update statistics.
    """
    categories = ['anomalous', 'missing', 'unknown']
    subtypes = ['origin', 'non_origin']
    
    stats_records = []

    for category in categories:
        for subtype in subtypes:
            filename = f"{category}_{subtype}.csv"
            # Origin types need ROV (RPKI) validation
            check_rpki = (subtype == 'origin')
            
            filepath = results_dir / filename
            if not filepath.exists():
                # print(f"Skipping {filename} (File not found)")
                continue

            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            if 'is_ip_leasing' not in df.columns:
                df['is_ip_leasing'] = 0
            else:
                df['is_ip_leasing'] = pd.to_numeric(df['is_ip_leasing'], errors='coerce').fillna(0).astype(int)
                
            if df.empty:
                stats_records.append({
                    'category': category,
                    'subtype': subtype,
                    'total_rows': 0,
                    'false_positives': 0,
                    'final_rows': 0,
                    'unique_asns_total': 0,
                    'unique_asns_final': 0
                })
                continue

            # Apply Filters (RPKI + Org/Hop)
            # If mappings are missing, anomalies are preserved (is_fp=False)
            fp_df, anom_df = filter_false_positives(
                df, 
                prefix_map, 
                prefix_to_profile, 
                profile_org_history[0],
                profile_org_history[1],
                as_org, 
                as_org_name,
                profile_org_history[2],
                check_rpki=check_rpki
            )
            
            stem = filepath.stem 
            fp_out = results_dir / f"{stem}_false_pos.csv"
            anom_out = results_dir / f"{stem}_anom.csv"
            
            # Save results
            fp_df.to_csv(fp_out, index=False)
            anom_df.to_csv(anom_out, index=False)
            
            # Calculate Stats
            total_asns = df['asn'].nunique() if 'asn' in df.columns else 0
            final_asns = anom_df['asn'].nunique() if 'asn' in anom_df.columns else 0
            
            stats_records.append({
                'category': category,
                'subtype': subtype,
                'total_rows': len(df),
                'false_positives': len(fp_df),
                'final_rows': len(anom_df),
                'unique_asns_total': total_asns,
                'unique_asns_final': final_asns
            })
            
    # Update detection_stats.csv
    if stats_records:
        stats_df = pd.DataFrame(stats_records)
        stats_path = results_dir / "detection_stats.csv"
        stats_df.to_csv(stats_path, index=False)
    else:
        print("\nNo statistics generated (no files processed).")


def is_results_window_dir(path):
    target_files = [
        "anomalous_origin.csv",
        "anomalous_non_origin.csv",
        "missing_origin.csv",
        "missing_non_origin.csv",
    ]
    return any((path / name).exists() for name in target_files)


def iter_results_dirs(results_root):
    if is_results_window_dir(results_root):
        return [results_root]

    result_dirs = set()
    for pattern in ("anomalous_origin.csv", "anomalous_non_origin.csv", "missing_origin.csv", "missing_non_origin.csv"):
        for match in results_root.rglob(pattern):
            result_dirs.add(match.parent)
    return sorted(result_dirs)

def main():
    parser = argparse.ArgumentParser(description="Filter False Positives and Aggregate BGP Anomaly Results")
    parser.add_argument("--data_dir", type=Path, default=Path("../dataset/klayswap/processed_data"), help="Path to processed data directory")
    parser.add_argument("--results_dir", type=Path, default=None, help="Path to a single window results directory or a results root")
    
    args = parser.parse_args()
    
    PROCESSED_DIR = args.data_dir
    BASE_DIR = find_dataset_root(PROCESSED_DIR)
    RESULTS_DIR = args.results_dir if args.results_dir is not None else BASE_DIR / "results"
    AS_INFO_PATH = resolve_as_info_path(PROCESSED_DIR)
    
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory {RESULTS_DIR} does not exist.")
        sys.exit(1)

    prefix_map, prefix_to_profile, profile_org_ids, profile_org_names, as_org, as_org_name, as_org_name_norm = load_resources(PROCESSED_DIR, AS_INFO_PATH)
    
    result_dirs = iter_results_dirs(RESULTS_DIR)
    if not result_dirs:
        print(f"No result directories found under {RESULTS_DIR}")
        sys.exit(0)

    profile_org_history = (profile_org_ids, profile_org_names, as_org_name_norm)

    for result_dir in result_dirs:
        process_and_update_stats(
            result_dir,
            prefix_map,
            prefix_to_profile,
            profile_org_history,
            as_org,
            as_org_name
        )

if __name__ == "__main__":
    main()
