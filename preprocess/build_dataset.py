"""Dataset building script for Hyperbolic BGP embedding."""

import json
import torch
from pathlib import Path
import numpy as np
import struct
import pandas as pd
import os

class EfficientAnchorBuilder:
    def __init__(self, device, similarity_threshold=0.9, max_anchors=2000):
        self.device = device
        self.max_hamming_dist = int(128 * (1.0 - similarity_threshold))
        self.max_anchors = max_anchors
        # Pairwise mode is bounded even with packed-byte Hamming distances;
        # larger groups use the iterative exact comparison.
        self.max_node_limit = 1500
        self.byte_popcount = torch.tensor(
            [value.bit_count() for value in range(256)],
            dtype=torch.int16,
            device=device,
        )
        
        # Mask for unpacking (1, 1, 64)
        self.bit_mask = (1 << torch.arange(63, -1, -1, device=device)).unsqueeze(0).unsqueeze(0)

    def _unpack_batch(self, packed_tensor):
        """Unpack (N, 2) Int64 to (N, 128) Float32 binary."""
        bits = (packed_tensor.unsqueeze(-1).bitwise_and(self.bit_mask) != 0).float()
        return bits.view(bits.shape[0], -1)

    def cluster_as(self, packed_fps):
        N = packed_fps.size(0)
        if N == 0: return packed_fps.cpu()
        
        unique_packed = torch.unique(packed_fps, dim=0)
        n_unique = unique_packed.size(0)
        
        if n_unique <= 1:
            return unique_packed.cpu()

        perm = torch.randperm(n_unique, device=self.device)
        unique_packed = unique_packed[perm]
        
        if n_unique <= self.max_node_limit:
            anchors = self._cluster_matrix_mode(unique_packed)
        else:
            anchors = self._cluster_iterative_mode(unique_packed)
            
        return anchors

    def _packed_hamming(self, left, right):
        xor = torch.bitwise_xor(left, right).contiguous()
        byte_values = xor.view(torch.uint8).reshape(*xor.shape[:-1], -1)
        return self.byte_popcount[byte_values.long()].sum(dim=-1)

    def _cluster_matrix_mode(self, unique_packed):
        """
        Matrix Mode: Fast for small N.
        Computes N*N distance matrix.
        """
        dists = self._packed_hamming(
            unique_packed.unsqueeze(1), unique_packed.unsqueeze(0)
        )
        coverage_matrix = dists <= self.max_hamming_dist
        del dists # Free memory immediately

        selected_indices = []
        covered_mask = torch.zeros(unique_packed.size(0), dtype=torch.bool, device=self.device)
        
        # Greedy Selection
        while not covered_mask.all():
            if len(selected_indices) >= self.max_anchors:
                break
                
            remaining = (~covered_mask).nonzero()
            if remaining.numel() == 0: break
            
            leader_idx = remaining[0].item()
            selected_indices.append(leader_idx)
            
            # Mask neighbors
            covered_mask |= coverage_matrix[leader_idx]
            
        return unique_packed[selected_indices].cpu()

    def _cluster_iterative_mode(self, unique_packed):
        """
        Iterative Mode: Low memory for Large N.
        Never computes N*N. Computes N * K (K <= 1024).
        """
        n_unique = unique_packed.size(0)
        covered_mask = torch.zeros(n_unique, dtype=torch.bool, device=self.device)
        selected_indices = []
        
        while True:
            if len(selected_indices) >= self.max_anchors:
                break
                
            uncovered_indices = (~covered_mask).nonzero(as_tuple=True)[0]
            
            if len(uncovered_indices) == 0:
                break
            
            leader_idx = uncovered_indices[0].item()
            selected_indices.append(leader_idx)
            
            leader = unique_packed[leader_idx]
            candidates = unique_packed[uncovered_indices]
            dists = self._packed_hamming(candidates, leader)
            
            is_close = dists <= self.max_hamming_dist
            
            newly_covered_global_indices = uncovered_indices[is_close]
            covered_mask[newly_covered_global_indices] = True
            
        return unique_packed[selected_indices].cpu()

def build_dataset(output_dir: Path, seed: int = 42):
    # Anchor selection uses a randomized traversal order.  Fixing it makes
    # preprocessing benchmarks and regenerated detector caches reproducible.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # If anchor cache exists, skip heavy anchor rebuild but still refresh light artifacts.
    skip_anchor_rebuild = (output_dir / "fingerprint_anchor_db.pt").exists()
    if skip_anchor_rebuild:
        print(f"Found existing anchor cache at {output_dir}. Will refresh UPD stats and metadata only.")

    prefix_map_pt = output_dir / "prefix_map.pt"
    
    # Variables that need to be populated for Stage 2
    df_triplets = None
    as_map = None
    prefix_map = None
    num_profiles = 0

    triplets_pt = output_dir / "profile_triplets.pt"
    if prefix_map_pt.exists() and triplets_pt.exists():
        cached_triplets = torch.load(triplets_pt, map_location="cpu")
        if not torch.is_tensor(cached_triplets) or cached_triplets.dim() != 2 or cached_triplets.size(1) < 4:
            print("Found stale unweighted profile_triplets.pt. Deleting generated PT caches for rebuild...")
            for stale_name in (
                "prefix_map.pt",
                "as_map.pt",
                "profile_triplets.pt",
                "prefix_to_profile.pt",
                "profile_fingerprints.pt",
                "fingerprint_anchor_db.pt",
                "history_cache.pt",
                "upd_counts.pt",
                "meta.json",
            ):
                stale_path = output_dir / stale_name
                if stale_path.exists():
                    stale_path.unlink()
            skip_anchor_rebuild = False

    if prefix_map_pt.exists():
        print(f"Found {prefix_map_pt}. Skipping Stage 1 (Binary -> PT)...")
        
        # Load maps
        with (output_dir / "as_map.pt").open("rb") as f:
            as_map = torch.load(f)
            
        with (output_dir / "prefix_map.pt").open("rb") as f:
            prefix_map = torch.load(f)

        # Load triplets
        triplets_tensor = torch.load(output_dir / "profile_triplets.pt")
        if not torch.is_tensor(triplets_tensor) or triplets_tensor.dim() != 2 or triplets_tensor.size(1) < 4:
            raise ValueError("profile_triplets.pt must have columns [profile_id, as_id, hop, weight]. Rebuild the dataset tensors.")

        # Convert to numpy and then DataFrame
        # Tensor columns: [profile_id, as_id, hop, weight]
        triplets_arr = triplets_tensor.numpy()
        df_triplets = pd.DataFrame(triplets_arr[:, :4], columns=['pid', 'aid', 'hop', 'weight'])
        
        num_profiles = int(df_triplets['pid'].max()) + 1
        
    else:
        # 1. Load Maps
        with (output_dir / "prefix_map.txt").open("r") as f:
            prefix_list = [line.strip() for line in f]
        prefix_map = {p: i for i, p in enumerate(prefix_list)}
        
        with (output_dir / "as_map.txt").open("r") as f:
            as_list = [line.strip() for line in f]
        as_map = {int(a): i for i, a in enumerate(as_list)}
        
        print(f"Num Prefixes: {len(prefix_map)}")
        print(f"Num ASNs: {len(as_map)}")
        
        # 2. Load Prefix -> Profile ID
        prefix_to_profile_id = np.fromfile(output_dir / "prefix_to_profile.bin", dtype=np.int32)
        
        # 3. Load Profile Triplets
        with (output_dir / "profile_triplets.bin").open("rb") as f:
            count_bytes = f.read(8)
            count = struct.unpack("Q", count_bytes)[0]
            print(f"Total triplets: {count}")
            
            remaining_bytes = os.fstat(f.fileno()).st_size - 8
            weighted_item_size = np.dtype([('pid', 'i4'), ('aid', 'i4'), ('hop', 'u1'), ('weight', 'f4')]).itemsize

            if remaining_bytes != count * weighted_item_size:
                raise ValueError(
                    "profile_triplets.bin is not in weighted format [pid, aid, hop, weight]. "
                    "Delete the old binary outputs and rerun preprocessing with the current processor."
                )

            dt = np.dtype([('pid', 'i4'), ('aid', 'i4'), ('hop', 'u1'), ('weight', 'f4')])

            triplets_np = np.fromfile(f, dtype=dt, count=count)
        
        # profile_triplets: (profile_id, as_id, hop, weight)
        # Convert structured array to regular array.
        p_ids = triplets_np['pid'].astype(np.int64)
        a_ids = triplets_np['aid'].astype(np.int64)
        hops = triplets_np['hop'].astype(np.int64)
        weights = triplets_np['weight'].astype(np.float32)
        triplets_tensor = torch.from_numpy(np.column_stack([p_ids, a_ids, hops, weights]))
            
        # 4. Load Profile Fingerprints (128-bit, stored as pairs of uint64)
        profile_fingerprints_raw = np.fromfile(output_dir / "profile_fingerprints.bin", dtype=np.uint64)
        # Reshape to (num_profiles, 2) since each fingerprint is low + high
        num_profiles_fp = len(profile_fingerprints_raw) // 2
        profile_fingerprints = profile_fingerprints_raw.reshape(num_profiles_fp, 2)
        profile_fingerprints_tensor = torch.from_numpy(profile_fingerprints.astype(np.int64))
            
        # Save    
        with (output_dir / "prefix_map.pt").open("wb") as f:
            torch.save(prefix_map, f)
        with (output_dir / "as_map.pt").open("wb") as f:
            torch.save(as_map, f)
            
        torch.save(triplets_tensor, output_dir / "profile_triplets.pt")
        
        # Save Prefix -> Profile Mapping (Tensor)
        # Index is prefix_id, Value is profile_id
        prefix_to_profile_tensor = torch.from_numpy(prefix_to_profile_id).long()
        torch.save(prefix_to_profile_tensor, output_dir / "prefix_to_profile.pt")

        # Save Profile Fingerprints
        torch.save(profile_fingerprints_tensor, output_dir / "profile_fingerprints.pt")
        
        # Prepare for Stage 2
        df_triplets = pd.DataFrame(triplets_np)
        df_triplets = df_triplets[['pid', 'aid', 'hop', 'weight']]
        num_profiles = int(np.max(triplets_np['pid'])) + 1
        
    # ==========================================
    # Extra: Build History Cache
    # ==========================================
    print("Building history cache...")
    
    # 1. Load AS Info for Org Mapping
    as_info_path = output_dir / "as_info.jsonl"
    if not as_info_path.exists():
        as_info_path = output_dir.parent / "as_info.jsonl"
    as_org = {}
    print(f"Loading AS info from {as_info_path}...")
    if as_info_path.exists():
        with open(as_info_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    asn = int(item['asn'])
                    if 'organization' in item and item['organization']:
                        org_id = item['organization'].get('orgId')
                        if org_id:
                            as_org[asn] = org_id
                except:
                    continue
    else:
        shard_files = sorted(output_dir.glob("as_info.*.jsonl"))
        if not shard_files:
            shard_files = sorted(output_dir.parent.glob("as_info.*.jsonl"))

        if shard_files:
            print(f"as_info.jsonl not found, loading shard files: {len(shard_files)}")
            for shard in shard_files:
                with open(shard, 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            asn = int(item['asn'])
                            if 'organization' in item and item['organization']:
                                org_id = item['organization'].get('orgId')
                                if org_id:
                                    as_org[asn] = org_id
                        except:
                            continue
        else:
            print(f"Warning: {as_info_path} not found, continue without org mapping cache.")
    
    # 2. Process Triplets for History
    # df_triplets should have columns ['pid', 'aid', 'hop']
    idx_to_asn_map = {v: k for k, v in as_map.items()}
    
    # Group by pid, aid -> min hop
    grouped = df_triplets.groupby(['pid', 'aid'])['hop'].min().reset_index()
    
    # --- OPTIMIZATION: Keep only first 5 hops (0, 1, 2, 3, 4) ---
    grouped = grouped[grouped['hop'] < 5]
    
    profile_history = {} # as_id -> list[pid]
    profile_org_history = {} # pid -> list[org_id]

    # Columns: pid, aid, hop
    for row in grouped.itertuples(index=False):
        pid = row.pid
        aid = int(row.aid)
        hop = row.hop
        
        # Resolve real ASN for Org lookup
        asn = idx_to_asn_map.get(aid, -1)

        # 1. New History: ASID -> List[PID]
        if aid not in profile_history:
            profile_history[aid] = []
        profile_history[aid].append(pid)

        # 2. Org History
        org_id = as_org.get(asn)
        if org_id:
            if pid not in profile_org_history:
                profile_org_history[pid] = []
            profile_org_history[pid].append(org_id)

    torch.save({pid: list(set(orgs)) for pid, orgs in profile_org_history.items()}, output_dir / "history_cache.pt")

    # ==========================================
    # Build Fingerprint Anchors
    # ==========================================
    if skip_anchor_rebuild:
        print("Skipping fingerprint anchor rebuild (cache exists).")
    else:
        print("Building fingerprint anchors...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fp_path = output_dir / "profile_fingerprints.pt"
        if not fp_path.exists():
            raise FileNotFoundError(f"Fingerprints not found at {fp_path}")

        # Keep raw data as Int64 on GPU to prevent OOM
        raw_fps = torch.load(fp_path, map_location=device)
        builder = EfficientAnchorBuilder(device, similarity_threshold=0.9)
        anchor_db = {}

        total_profiles = 0
        total_anchors = 0

        total_as_groups = len(profile_history)
        for group_index, (as_id, pids) in enumerate(profile_history.items(), start=1):
            if not pids:
                continue

            pid_tensor = torch.tensor(pids, dtype=torch.long, device=device)
            current_packed_fps = raw_fps[pid_tensor]

            anchors = builder.cluster_as(current_packed_fps)
            anchor_db[as_id] = anchors

            total_profiles += len(pids)
            total_anchors += anchors.size(0)
            if group_index % 1_000 == 0 or group_index == total_as_groups:
                print(
                    f"Anchor groups: {group_index}/{total_as_groups}; "
                    f"profiles={total_profiles}, anchors={total_anchors}",
                    flush=True,
                )

        print(f"\n--- Anchor Clustering Result ---")
        print(f"Profiles: {total_profiles}, Anchors: {total_anchors}, Compression: {total_profiles / total_anchors:.2f}x")

        torch.save(anchor_db, output_dir / "fingerprint_anchor_db.pt")
    
    meta = {
        "num_prefixes": len(prefix_map),
        "num_asns": len(as_map),
        "num_profiles": num_profiles,
        "num_triplets": len(triplets_tensor),
        "build_seed": seed,
        "triplet_weighted": True,
        "triplet_default_rib_weight": 1.0,
        "triplet_default_upd_weight": 0.2,
    }
    
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)
        
    print("Preprocessing complete.")

    # Cleanup intermediate files
    print("Cleaning up intermediate files...")
    for f in ["prefix_map.txt", "as_map.txt", "prefix_to_profile.bin", "profile_triplets.bin", "profile_fingerprints.bin"]:
        try:
            (output_dir / f).unlink()
        except FileNotFoundError:
            pass
    print("Cleanup complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Hyperbolic BGP Dataset")
    parser.add_argument("--data_dir", type=Path, default=Path("processed_data"), help="Data directory for processed tensors")
    parser.add_argument("--seed", type=int, default=42, help="Seed used by fingerprint-anchor construction")
    args = parser.parse_args()

    build_dataset(args.data_dir, seed=args.seed)
