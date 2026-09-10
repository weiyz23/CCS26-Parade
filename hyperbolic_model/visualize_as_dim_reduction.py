import torch
import json
import numpy as np
import argparse
import csv
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import umap
from pathlib import Path
from tqdm import tqdm

# Import model
try:
    from model import HyperbolicRoutingModel
except ImportError:
    print("Error: Could not import HyperbolicRoutingModel from model.py")
    exit(1)

warnings.filterwarnings("ignore")

# Set global font to serif family (prioritizing Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# ==========================================
# 0. Constant definitions
# ==========================================

TIER1_ASNS = {
    "701", "1299", "2914", "3257", "3320", "3356", "3491", 
    "5511", "6453", "6461", "6762", "6830", "7018", "12956"
}

AS_PATH_COLOR = "#d62728"

# ==========================================
# 1. Core math tools
# ==========================================

def get_poincare_distance_matrix(model, x_tensor, max_dist=15.0):
    if x_tensor.device.type != 'cpu':
        x_tensor = x_tensor.cpu()
    x_i = x_tensor.unsqueeze(1)
    x_j = x_tensor.unsqueeze(0)
    with torch.no_grad():
        dists = model.manifold.dist(x_i, x_j)
    dists = torch.clamp(dists, max=max_dist)
    return dists.numpy()

def get_hyperbolic_radius(model, embeddings):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    zeros = torch.zeros_like(embeddings) 
    with torch.no_grad():
        embeddings = embeddings.to(model.as_embedding.device)
        zeros = zeros.to(model.as_embedding.device)
        radii = model.manifold.dist(zeros, embeddings)
    return radii.cpu().numpy()

def apply_radial_preserving(embedding_euclidean, original_norms):
    current_norms = np.linalg.norm(embedding_euclidean, axis=1, keepdims=True)
    current_norms = np.clip(current_norms, 1e-9, None)
    directions = embedding_euclidean / current_norms
    return directions * original_norms

def parse_as_path(values):
    if not values:
        return []
    asns = []
    for value in values:
        for item in str(value).replace(",", " ").split():
            item = item.strip()
            if item.upper().startswith("AS"):
                item = item[2:]
            if item:
                asns.append(item)
    return asns

def resolve_asn_index(asn, as_map):
    if asn in as_map:
        return int(as_map[asn])
    asn_str = str(asn)
    if asn_str in as_map:
        return int(as_map[asn_str])
    try:
        asn_int = int(asn_str)
    except ValueError:
        return None
    if asn_int in as_map:
        return int(as_map[asn_int])
    return None

def resolve_as_path(as_path, as_map):
    records = []
    missing = []
    for step, asn in enumerate(as_path, start=1):
        global_idx = resolve_asn_index(asn, as_map)
        if global_idx is None:
            missing.append(asn)
        records.append({"step": step, "asn": asn, "global_idx": global_idx})
    return records, missing

def build_as_path_overlay(path_records, orig_to_local, full_radii):
    overlay_records = []
    for record in path_records:
        global_idx = record["global_idx"]
        if global_idx is None or global_idx not in orig_to_local:
            continue
        local_idx = int(orig_to_local[global_idx])
        overlay_records.append(
            {
                "step": record["step"],
                "asn": record["asn"],
                "global_idx": int(global_idx),
                "local_idx": local_idx,
                "radius": float(full_radii[global_idx]),
            }
        )
    return overlay_records

def write_as_path_coordinates(path_overlay, data, output_path, dims):
    if not path_overlay:
        return

    fieldnames = ["step", "asn", "global_idx", "local_idx", "radius", "x", "y"]
    if dims == 3:
        fieldnames.append("z")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in path_overlay:
            point = data[record["local_idx"]]
            row = {
                "step": record["step"],
                "asn": record["asn"],
                "global_idx": record["global_idx"],
                "local_idx": record["local_idx"],
                "radius": record["radius"],
                "x": float(point[0]),
                "y": float(point[1]),
            }
            if dims == 3:
                row["z"] = float(point[2])
            writer.writerow(row)

def draw_as_path_overlay(ax, data, path_overlay, is_3d=False, annotate=True):
    if not path_overlay:
        return

    local_indices = [record["local_idx"] for record in path_overlay]
    coords = data[local_indices]
    labels = [record["asn"] for record in path_overlay]

    if is_3d:
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            color=AS_PATH_COLOR,
            linewidth=2.5,
            alpha=0.95,
            zorder=30,
            label="AS Path",
        )
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            marker="X",
            color=AS_PATH_COLOR,
            s=120,
            edgecolors="white",
            linewidth=1.2,
            zorder=31,
        )
        return

    for start, end in zip(coords[:-1], coords[1:]):
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={
                "arrowstyle": "->",
                "color": AS_PATH_COLOR,
                "lw": 2.0,
                "alpha": 0.95,
                "shrinkA": 7,
                "shrinkB": 7,
            },
            zorder=30,
        )

    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color=AS_PATH_COLOR,
        linewidth=1.3,
        alpha=0.55,
        zorder=29,
    )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        marker="X",
        color=AS_PATH_COLOR,
        s=150,
        edgecolors="white",
        linewidth=1.5,
        zorder=31,
        label="AS Path",
    )

    if annotate:
        for record, label, point in zip(path_overlay, labels, coords):
            ax.annotate(
                f"{record['step']}:{label}",
                xy=(point[0], point[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color=AS_PATH_COLOR,
                weight="bold",
                zorder=32,
            )

# ==========================================
# 2. Auto region search algorithm
# ==========================================

def find_best_zoom_region(data, highlight_groups, search_box_size=0.04, min_points=4, view_padding=0.03):
    candidate_indices = []
    tier3_indices = set()
    transit_indices = set()
    stub_indices = set()
    
    for group in highlight_groups:
        name = group['name'].lower()
        if "1000-5000" in name:
            tier3_indices.update(group['indices'])
            candidate_indices.extend(group['indices'])
        if "5000-15000" in name:
            transit_indices.update(group['indices'])
            candidate_indices.extend(group['indices'])
        elif "15000+" in name:
            stub_indices.update(group['indices'])
            candidate_indices.extend(group['indices'])
            
    if not candidate_indices:
        return None
        
    candidate_indices = np.unique(candidate_indices)
    
    # Pre-calculate category masks relative to candidate_indices
    is_tier3 = np.array([idx in tier3_indices for idx in candidate_indices])
    is_transit = np.array([idx in transit_indices for idx in candidate_indices])
    is_stub = np.array([idx in stub_indices for idx in candidate_indices])
    
    candidate_points = data[candidate_indices]
    
    best_count = 0
    best_center = None
    r = search_box_size / 2.0
    
    for p in candidate_points:
        x_min, x_max = p[0] - r, p[0] + r
        y_min, y_max = p[1] - r, p[1] + r
        
        in_x = (candidate_points[:, 0] >= x_min) & (candidate_points[:, 0] <= x_max)
        in_y = (candidate_points[:, 1] >= y_min) & (candidate_points[:, 1] <= y_max)
        in_region = in_x & in_y
        
        count = np.sum(in_region)
        
        # Check diversity constraint: must contain at least one of each type
        has_tier3 = np.any(in_region & is_tier3)
        has_transit = np.any(in_region & is_transit)
        has_stub = np.any(in_region & is_stub)
        
        if has_tier3 and has_transit and has_stub:
            if count > best_count:
                best_count = count
                best_center = p

    if best_center is not None:
        print(f"    [Auto-Zoom] Found region with {best_count} points.")
        final_half_size = (search_box_size / 2.0) + view_padding
        cx, cy = best_center
        return (cx - final_half_size, cx + final_half_size, 
                cy - final_half_size, cy + final_half_size)
    else:
        return None

# ==========================================
# 3. Plotting engine
# ==========================================

def create_custom_cmap():
    colors = [
        (0.0,  "#f7f7f7"),  # Reserved Semantic Space (Lightest Gray/White)
        (0.34, "#f7f7f7"),
        (0.35,  "#2171b5"),  # Tier-1
        (0.55,  "#41ab5d"),  # Tier-2 (Light Green)
        (0.85,  "#feb24c"),  # Tier-3 (Orange Yellow)
        (0.95,  "#ffeda0"),  # Transit (Light Yellow)
        (1.0,   "#bdbdbd"),  # Stub (Gray)
    ]
    return LinearSegmentedColormap.from_list("hyperbolic_hierarchy", colors)

def plot_embedding(
    data,
    radii,
    title,
    filename,
    is_3d=False,
    draw_boundary=False,
    highlight_groups=None,
    zoom_rect=None,
    as_path_overlay=None,
):
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    
    custom_cmap = create_custom_cmap()    
    radii_flat = radii.flatten()
    base_filename = str(filename).rsplit('.', 1)[0]
    ext = ".png"
    
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
        if draw_boundary:
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
            xs = np.cos(u)*np.sin(v)
            ys = np.sin(u)*np.sin(v)
            zs = np.cos(v)
            ax.plot_wireframe(xs, ys, zs, color="#dddddd", alpha=0.3, linewidth=0.5)
            ax.set_axis_off() 
        else:
            ax.grid(False)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=radii_flat, cmap=custom_cmap, vmin=0.0, vmax=1.0, s=8, alpha=0.8)
        if highlight_groups:
            for group in highlight_groups:
                idx = group["indices"]
                if len(idx)>0: 
                    ax.scatter(data[idx,0], data[idx,1], data[idx,2], marker=group["marker"], color=group["color"], s=group["s"], edgecolors='black', linewidth=0.5, zorder=20)
        draw_as_path_overlay(ax, data, as_path_overlay, is_3d=True)
        plt.savefig(filename, dpi=200)
        plt.close()
        return

    # --- 2D Mode ---
    # 1. Global (Base plot)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_aspect('equal')
    
    if draw_boundary:
        wedge = patches.Wedge((0, 0), 1.0, 0, 180, fill=False, color='black', linewidth=1.5, alpha=0.8)
        ax.add_artist(wedge)
        # Draw diameter (horizontal axis)
        ax.plot([-1, 1], [0, 0], color='black', linewidth=1.5, alpha=0.8)
        
        # Draw prefix space boundary (r=0.35)
        prefix_wedge = patches.Wedge((0, 0), 0.35, 0, 180, fill=False, color='#636363', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.add_artist(prefix_wedge)
        
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
        ax.axis('off')
    else:
        ax.axis('on'); ax.grid(True, linestyle=':', alpha=0.2)

    scatter = ax.scatter(data[:, 0], data[:, 1], c=radii_flat, cmap=custom_cmap, vmin=0.0, vmax=1.0, s=8, alpha=0.7, edgecolors='none')
    
    # Origin
    ax.scatter(0, 0, c='black', marker='o', s=60, edgecolors='white', linewidth=1.5, zorder=25)

    if highlight_groups:
        for group in highlight_groups:
            idx = group["indices"]
            if len(idx) > 0:
                ax.scatter(data[idx, 0], data[idx, 1], marker=group["marker"], color=group["color"], s=group["s"], label=group["name"], edgecolors='black', linewidth=1.0, zorder=20)

    draw_as_path_overlay(ax, data, as_path_overlay, annotate=True)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    print(f"  Saving: {base_filename}_Global{ext}")
    plt.savefig(f"{base_filename}_Global{ext}", dpi=200)

    # Create separate horizontal colorbar
    fig_cbar, ax_cbar = plt.subplots(figsize=(12, 1), facecolor='white')
    mappable = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    cbar = plt.colorbar(mappable, ax=ax_cbar, orientation='horizontal', shrink=1.0, aspect=40)
    cbar.set_label('Hyperbolic Radius', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks([0.0, 0.35, 1.0])
    cbar.set_ticklabels(['Prefix Space', 'Core', 'Stub'])
    ax_cbar.axis('off')
    plt.savefig(f"{base_filename}_Colorbar{ext}", dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig_cbar)
    print(f"  Saving: {base_filename}_Colorbar{ext}")

    # Create separate legend
    if highlight_groups:
        handles, labels = ax.get_legend_handles_labels()
        fig_legend, ax_legend = plt.subplots(figsize=(4, 3), facecolor='white')
        ax_legend.legend(handles, labels, loc='center', fontsize=16, framealpha=0.9, edgecolor='#cccccc', fancybox=True)
        ax_legend.axis('off')
        plt.savefig(f"{base_filename}_Legend{ext}", dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close(fig_legend)
        print(f"  Saving: {base_filename}_Legend{ext}")

    # 2. Boxed
    if zoom_rect is not None:
        x1, x2, y1, y2 = zoom_rect
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2.0, edgecolor='#41ab5d', facecolor='none', zorder=30)
        ax.add_patch(rect)
        
        print(f"  Saving: {base_filename}_Boxed{ext}")
        plt.savefig(f"{base_filename}_Boxed{ext}", dpi=200)
        
    plt.close(fig)

    # 3. Detail (Local zoomed view)
    if zoom_rect is not None:
        x1, x2, y1, y2 = zoom_rect
        fig_z, ax_z = plt.subplots(figsize=(6, 6), facecolor='white')
        ax_z.set_facecolor('#f0f9e8')
        
        ax_z.scatter(data[:, 0], data[:, 1], c=radii_flat, cmap=custom_cmap, vmin=0.0, vmax=1.0, s=40, alpha=0.9, edgecolors='none')
        
        # Origin in Detail view
        ax_z.scatter(0, 0, c='black', marker='o', s=80, edgecolors='white', linewidth=1.5, zorder=25)
        ax_z.text(0.02, 0.02, 'Origin', fontsize=12, color='black', fontweight='bold', ha='left', va='bottom', zorder=25)
        
        prefix_wedge_z = patches.Wedge((0, 0), 0.35, 0, 180, fill=False, color='#636363', linestyle='--', linewidth=1.5, alpha=0.6, zorder=10)
        ax_z.add_artist(prefix_wedge_z)
        
        if highlight_groups:
            for group in highlight_groups:
                idx = group["indices"]
                if len(idx) > 0:
                    sc = ax_z.scatter(data[idx, 0], data[idx, 1], marker=group["marker"], color=group["color"], s=group["s"]*2.8, edgecolors='black', linewidth=1.5, zorder=20, label=group["name"])            

        draw_as_path_overlay(ax_z, data, as_path_overlay, annotate=True)
        
        ax_z.set_xlim(x1, x2)
        ax_z.set_ylim(y1, y2)
        
        for spine in ax_z.spines.values():
            spine.set_edgecolor('#41ab5d'); spine.set_linewidth(3.0)
        ax_z.set_xticks([]); ax_z.set_yticks([])
        
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        
        print(f"  Saving: {base_filename}_Detail{ext}")
        plt.savefig(f"{base_filename}_Detail{ext}", dpi=200, pad_inches=0)
        plt.close(fig_z)

# ==========================================
# 4. Main process
# ==========================================

def run_grid_search(args, max_samples=15000):
    device = torch.device("cpu")
    viz_dir = args.viz_dir / "grid_search"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.data_dir / "meta.json", "r") as f: meta = json.load(f)
    as_map = torch.load(args.data_dir / "as_map.pt")
    as_path = parse_as_path(args.as_path)
    path_records, missing_path_asns = resolve_as_path(as_path, as_map)
    path_force_indices = np.array(
        sorted({record["global_idx"] for record in path_records if record["global_idx"] is not None}),
        dtype=int,
    )
    if as_path:
        print(f"AS path requested: {' -> '.join(as_path)}")
        print(f"Resolved {len(path_force_indices)}/{len(as_path)} unique ASNs from the path.")
        if missing_path_asns:
            print(f"Warning: ASNs not found in as_map.pt: {', '.join(missing_path_asns)}")
        
    model = HyperbolicRoutingModel(
        num_as=meta["num_asns"], num_profiles=meta["num_profiles"], 
        embed_dim=args.embed_dim, num_vectors=args.num_vectors
    ).to(device)
    
    model_path = args.model_dir / "best_model.pth"
    if not model_path.exists(): raise FileNotFoundError(f"Model checkpoint not found")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    full_emb = model.as_embedding.detach()
    full_radii = get_hyperbolic_radius(model, full_emb).flatten()
    
    # --- Sampling ---
    global_tier1_indices = []
    for asn_str in TIER1_ASNS:
        idx = None
        if asn_str in as_map: idx = as_map[asn_str]
        elif int(asn_str) in as_map: idx = as_map[int(asn_str)]
        if idx is not None: global_tier1_indices.append(idx)
    target_tier1_set = set(global_tier1_indices)
    
    # Calculate finer percentiles for stratified sampling
    p02, p1, p1_5, p3, p4_5, p6, p15 = np.percentile(full_radii, [0.2, 1.0, 1.5, 3.0, 4.5, 6, 15])
    
    # Define Main Pools (for compatibility with highlighting)
    pool_02_15 = np.where((full_radii >= p02) & (full_radii < p1_5))[0]
    pool_15_6 = np.where((full_radii >= p1_5) & (full_radii < p6))[0]
    pool_6_15 = np.where((full_radii >= p6) & (full_radii < p15))[0]
    pool_15_100 = np.where(full_radii >= p15)[0]
    
    # Define Sub-Pools for Stratified Sampling
    pool_02_10 = np.where((full_radii >= p02) & (full_radii < p1))[0]
    pool_10_15 = np.where((full_radii >= p1) & (full_radii < p1_5))[0]
    
    pool_15_30 = np.where((full_radii >= p1_5) & (full_radii < p3))[0]
    pool_30_45 = np.where((full_radii >= p3) & (full_radii < p4_5))[0]
    pool_45_60 = np.where((full_radii >= p4_5) & (full_radii < p6))[0]
    
    N_total = full_emb.shape[0]
    print(f"Sampling data (Target: {max_samples})...")
    
    stratum_1 = np.array(global_tier1_indices, dtype=int)
    
    # --- Stratum 2: Tier-2 (0.2-1.5%) ---
    # Split into 0.2-1.0 and 1.0-1.5, ensure balanced sampling
    sub_02_10 = np.setdiff1d(pool_02_10, stratum_1)
    sub_10_15 = np.setdiff1d(pool_10_15, stratum_1)
    
    s2_1 = np.random.choice(sub_02_10, min(len(sub_02_10), 50), replace=False)
    s2_2 = np.random.choice(sub_10_15, min(len(sub_10_15), 50), replace=False)
    stratum_2 = np.concatenate([s2_1, s2_2])
    
    # --- Stratum 3: Tier-3 (1.5-6%) ---
    # Split into 1.5-3, 3-4.5, 4.5-6
    sel_so_far = np.concatenate([stratum_1, stratum_2])
    
    sub_15_30 = np.setdiff1d(pool_15_30, sel_so_far)
    sub_30_45 = np.setdiff1d(pool_30_45, sel_so_far)
    sub_45_60 = np.setdiff1d(pool_45_60, sel_so_far)
    
    s3_1 = np.random.choice(sub_15_30, min(len(sub_15_30), 166), replace=False)
    s3_2 = np.random.choice(sub_30_45, min(len(sub_30_45), 167), replace=False)
    s3_3 = np.random.choice(sub_45_60, min(len(sub_45_60), 167), replace=False)
    
    stratum_3 = np.concatenate([s3_1, s3_2, s3_3])

    base_sel = np.unique(np.concatenate([stratum_1, stratum_2, stratum_3, path_force_indices]))
    
    if N_total > max_samples:
        mask = np.ones(N_total, dtype=bool)
        mask[base_sel] = False
        pool_rem = np.arange(N_total)[mask]
        n_need = max(0, max_samples - len(base_sel))
        stratum_4 = np.random.choice(pool_rem, min(len(pool_rem), n_need), replace=False)
        final_indices = np.concatenate([base_sel, stratum_4])
    else:
        final_indices = np.arange(N_total)
        
    np.random.shuffle(final_indices)
    data_subset = full_emb[final_indices]
    
    # --- Highlight Groups ---
    orig_to_local = { int(orig): loc for loc, orig in enumerate(final_indices) }
    as_path_overlay = build_as_path_overlay(path_records, orig_to_local, full_radii)
    if as_path and len(as_path_overlay) < len(as_path):
        print(f"Warning: only {len(as_path_overlay)}/{len(as_path)} AS path hops can be plotted.")
    highlight_groups = []
    
    # Tier-1
    t1_valid_orig = [x for x in target_tier1_set if x in orig_to_local]
    if t1_valid_orig:
        # User Rule: Remove the Tier-1 node with the largest radius (likely an outlier or misclassification)
        max_r_node = max(t1_valid_orig, key=lambda x: full_radii[x])
        t1_valid_orig.remove(max_r_node)

    local_t1 = [orig_to_local[x] for x in t1_valid_orig]
    if local_t1:
        highlight_groups.append({"name": "Top-tier", "indices": np.array(local_t1), "marker": "*", "color": "#2171b5", "s": 180})
        
    # Tier-2 (0.3-1.5%)
    pool_2_valid = [x for x in pool_02_15 if x in orig_to_local and x not in target_tier1_set]
    if pool_2_valid:
        count = min(len(pool_2_valid), 10)
        cands = sorted(pool_2_valid, key=lambda x: full_radii[x])[:count]
        highlight_groups.append({"name": "Rank < 1000", "indices": np.array([orig_to_local[x] for x in cands]), "marker": "s", "color": "#41ab5d", "s": 120})

    # Tier-3 (1.5-7%)
    pool_3_valid = [x for x in pool_15_6 if x in orig_to_local and x not in target_tier1_set]
    if pool_3_valid:
        count = min(len(pool_3_valid), 20)
        cands = sorted(pool_3_valid, key=lambda x: full_radii[x])[:count]
        highlight_groups.append({"name": "Rank 1000-5000", "indices": np.array([orig_to_local[x] for x in cands]), "marker": "D", "color": "#feb24c", "s": 110})

    # Transit (7-20%)
    pool_4_valid = [x for x in pool_6_15 if x in orig_to_local and x not in target_tier1_set]
    if pool_4_valid:
        count = min(len(pool_4_valid), 30)
        cands = sorted(pool_4_valid, key=lambda x: full_radii[x])[:count]
        highlight_groups.append({"name": "Rank 5000-15000", "indices": np.array([orig_to_local[x] for x in cands]), "marker": "o", "color": "#ffeda0", "s": 110})

    # Stub (20-100%)
    pool_5_valid = [x for x in pool_15_100 if x in orig_to_local]
    if pool_5_valid:
        cands = np.random.choice(pool_5_valid, min(len(pool_5_valid), 50), replace=False)
        highlight_groups.append({"name": "Rank 15000+", "indices": np.array([orig_to_local[x] for x in cands]), "marker": "^", "color": "#bdbdbd", "s": 110})

    # --- Normalization & Distance Calculation ---
    raw_norms_tensor = torch.norm(data_subset, p=2, dim=1, keepdim=True)
    if raw_norms_tensor.max() >= 1.0:
        data_subset = data_subset / (raw_norms_tensor.max() + 1e-5) * 0.9999
    raw_norms_np = raw_norms_tensor.numpy()
    
    print("Pre-computing distance matrices...")
    dist_matrix_hyp = get_poincare_distance_matrix(model, data_subset)
    
    dims = [2]
    methods = ["UMAP"]
    output_styles = ["Poincare"]
    
    pbar = tqdm(total=len(dims)*len(methods)*len(output_styles))

    for dim in dims:
        for method in methods:
            try:
                reducer = umap.UMAP(n_components=dim, metric="precomputed", n_neighbors=30, min_dist=0.1, n_jobs=-1)
                embedding_result = reducer.fit_transform(dist_matrix_hyp)
            except Exception as e:
                print(e); continue

            for out_style in output_styles:
                final_data = embedding_result
                draw_boundary = False
                zoom_roi = None 

                if out_style == "Poincare":
                    final_data = apply_radial_preserving(embedding_result, raw_norms_np)
                    draw_boundary = True
                    
                    if dim == 2:
                        zoom_roi = find_best_zoom_region(
                            final_data, 
                            highlight_groups, 
                            search_box_size=0.03, 
                            min_points=4,
                            view_padding=0.02 
                        )
                        if zoom_roi: print(f"    Target ROI: {zoom_roi}")

                else:
                    final_data = embedding_result
                    draw_boundary = False
                
                fname = f"{dim}D_{method}_Out-{out_style}.png"
                title = f"{dim}D {method} | Output: {out_style}"
                
                plot_embedding(
                    final_data, raw_norms_np, 
                    title, viz_dir / fname, 
                    is_3d=False, draw_boundary=draw_boundary,
                    highlight_groups=highlight_groups,
                    zoom_rect=zoom_roi,
                    as_path_overlay=as_path_overlay,
                )
                if as_path_overlay:
                    path_coordinate_file = viz_dir / f"{dim}D_{method}_Out-{out_style}_ASPath.csv"
                    write_as_path_coordinates(as_path_overlay, final_data, path_coordinate_file, dim)
                    print(f"  Saving: {path_coordinate_file}")
                pbar.update(1)

    pbar.close()
    print(f"\nAll visualizations saved to: {viz_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--model_dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--viz_dir", type=Path, default=Path("visualizations"))
    parser.add_argument("--embed_dim", type=int, default=24)
    parser.add_argument("--num_vectors", type=int, default=2)
    parser.add_argument(
        "--as-path",
        nargs="*",
        default=None,
        help="Optional AS path to highlight, for example: --as-path 3356 6453 7018 or --as-path '3356,6453,7018'.",
    )
    args = parser.parse_args()
    run_grid_search(args, max_samples=20000)
