import argparse
import csv
import ipaddress
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.manifold import TSNE

sys.path.append(str(Path(__file__).parent))

try:
    from model import HyperbolicRoutingModel
except ImportError:
    print("Error: Could not import HyperbolicRoutingModel from model.py")
    sys.exit(1)


warnings.filterwarnings("ignore")

FONT_SIZE = 16
FONT_SIZE_SMALL = 14
MAIN_LEGEND_FONT_SIZE = 16
MAIN_AXIS_MARGIN = 0.01
OPAQUE_LEGEND_FACE_COLOR = (1.0, 1.0, 1.0, 1.0)
HIGHLIGHT_LEGEND_PREFIX = "Sub-prefixes within"
DEFAULT_BACKGROUND_ALPHA = 0.15
DEFAULT_ZOOM_BACKGROUND_ALPHA = 0.4
DEFAULT_SECONDARY_ALPHA = 0.4
DEFAULT_SECONDARY_LINE_ALPHA = 0.15

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": ["Times New Roman"] + plt.rcParams["font.serif"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
)

DEFAULT_HIGHLIGHT_PREFIXES = [
    "36.152.0.0/16",
    "110.96.0.0/13",
    "58.200.0.0/13",
    "202.192.0.0/15",
]

HIGHLIGHT_COLORS = [
    "#ff00cc",
    "#0b8f6a",
    "#ff5a1f",
    "#1f3bff",
    "#8c564b",
    "#9467bd",
    "#d62728",
    "#17becf",
]

DEFAULT_ZOOM_BOXES_2D = [
    ("left_cluster", -92, -84.0, -7, -1),
    ("right_cluster", 48, 56, -21.0, -15.0),
    ("middle_lower_cluster", 25, 31, -73.0, -68.0),
    ("bottom_cluster", 4, 8, -90, -85),
]


def parse_highlight_prefixes(values):
    if not values:
        return DEFAULT_HIGHLIGHT_PREFIXES
    prefixes = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                prefixes.append(item)
    return prefixes


def parse_zoom_boxes(values):
    if not values:
        return DEFAULT_ZOOM_BOXES_2D
    boxes = []
    for i, value in enumerate(values, start=1):
        parts = [part.strip() for part in str(value).split(",")]
        if len(parts) == 4:
            name = f"zoom_{i}"
            coords = parts
        elif len(parts) == 5:
            name = parts[0]
            coords = parts[1:]
        else:
            raise ValueError(
                "Each zoom box must be 'x_min,x_max,y_min,y_max' or "
                "'name,x_min,x_max,y_min,y_max'."
            )
        x_min, x_max, y_min, y_max = [float(v) for v in coords]
        boxes.append((name, x_min, x_max, y_min, y_max))
    return boxes


def load_model(data_dir, model_dir, embed_dim, num_vectors, device):
    with (data_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    model_path = model_dir / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    profile_shape = state_dict["profile_embedding"].shape
    checkpoint_num_vectors = int(profile_shape[1])
    checkpoint_embed_dim = int(profile_shape[-1])

    if checkpoint_embed_dim != embed_dim:
        print(
            f"Checkpoint embed_dim is {checkpoint_embed_dim}; "
            f"using it instead of CLI value {embed_dim}."
        )
        embed_dim = checkpoint_embed_dim
    if checkpoint_num_vectors != num_vectors:
        print(
            f"Checkpoint num_vectors is {checkpoint_num_vectors}; "
            f"using it instead of CLI value {num_vectors}."
        )
        num_vectors = checkpoint_num_vectors

    model = HyperbolicRoutingModel(
        num_as=meta["num_asns"],
        num_profiles=meta["num_profiles"],
        embed_dim=embed_dim,
        num_vectors=num_vectors,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, meta


def load_prefix_resources(data_dir):
    prefix_map = torch.load(data_dir / "prefix_map.pt", map_location="cpu")
    prefix_to_profile = torch.load(data_dir / "prefix_to_profile.pt", map_location="cpu")
    return prefix_map, prefix_to_profile


def coerce_profile_id(prefix_index, prefix_to_profile):
    prefix_index = int(prefix_index)
    if prefix_index < 0 or prefix_index >= len(prefix_to_profile):
        return None
    profile_id = int(prefix_to_profile[prefix_index])
    return profile_id if profile_id >= 0 else None


def build_parsed_prefix_entries(prefix_map):
    entries = []
    for prefix, prefix_index in prefix_map.items():
        try:
            network = ipaddress.ip_network(str(prefix), strict=False)
        except ValueError:
            continue
        entries.append((str(prefix), int(prefix_index), network))
    return entries


def collect_highlight_groups(prefix_map, prefix_to_profile, highlight_prefixes, max_profiles, seed):
    parsed_entries = build_parsed_prefix_entries(prefix_map)
    groups = []
    rng = np.random.default_rng(seed)

    for group_idx, root_prefix in enumerate(highlight_prefixes):
        root_network = ipaddress.ip_network(root_prefix, strict=False)
        child_prefixes = []
        profile_ids = set()
        prefix_to_profile_id = {}

        for prefix, prefix_index, network in parsed_entries:
            if network.version != root_network.version:
                continue
            if network.prefixlen < root_network.prefixlen:
                continue
            if not network.subnet_of(root_network):
                continue

            profile_id = coerce_profile_id(prefix_index, prefix_to_profile)
            if profile_id is None:
                continue

            child_prefixes.append(prefix)
            profile_ids.add(profile_id)
            prefix_to_profile_id[prefix] = profile_id

        all_profile_ids = sorted(profile_ids)
        if max_profiles > 0 and len(all_profile_ids) > max_profiles:
            root_profile_id = prefix_to_profile_id.get(str(root_network))
            sampled = set()
            if root_profile_id is not None:
                sampled.add(root_profile_id)

            remaining = np.array(
                [profile_id for profile_id in all_profile_ids if profile_id not in sampled],
                dtype=np.int64,
            )
            n_needed = max_profiles - len(sampled)
            if n_needed > 0 and len(remaining) > 0:
                sampled.update(
                    int(profile_id)
                    for profile_id in rng.choice(
                        remaining,
                        size=min(n_needed, len(remaining)),
                        replace=False,
                    )
                )
            selected_profile_ids = sorted(sampled)
        else:
            selected_profile_ids = all_profile_ids

        color = HIGHLIGHT_COLORS[group_idx % len(HIGHLIGHT_COLORS)]
        groups.append(
            {
                "root": str(root_network),
                "prefixes": sorted(
                    child_prefixes,
                    key=lambda p: (
                        ipaddress.ip_network(p, strict=False).prefixlen,
                        ipaddress.ip_network(p, strict=False).network_address,
                    ),
                ),
                "profile_ids": selected_profile_ids,
                "total_profile_count": len(all_profile_ids),
                "prefix_to_profile_id": prefix_to_profile_id,
                "color": color,
            }
        )

    return groups


def sample_background_profiles(prefix_map, prefix_to_profile, sample_prefixes, seed):
    rng = np.random.default_rng(seed)
    items = list(prefix_map.items())
    if not items:
        return set(), 0

    n_sample = min(len(items), max(0, int(sample_prefixes)))
    chosen_indices = rng.choice(len(items), size=n_sample, replace=False)
    profile_ids = set()
    prefix_records = []

    for item_index in chosen_indices:
        prefix, prefix_index = items[int(item_index)]
        profile_id = coerce_profile_id(prefix_index, prefix_to_profile)
        if profile_id is not None:
            profile_ids.add(profile_id)
            prefix_records.append(
                {
                    "prefix": str(prefix),
                    "profile_id": profile_id,
                    "group": "background",
                }
            )

    return profile_ids, n_sample, prefix_records


def build_tsne_input(profile_embedding, profile_ids):
    profile_ids = np.array(sorted(profile_ids), dtype=np.int64)
    selected = profile_embedding[torch.from_numpy(profile_ids)].detach().cpu().float().numpy()
    num_profiles, num_vectors, embed_dim = selected.shape
    flat = selected.reshape(num_profiles * num_vectors, embed_dim)
    return profile_ids, flat, num_vectors


def fit_tsne(points, dims, perplexity, max_iter, seed):
    if len(points) <= 2:
        raise ValueError("Need at least three profile component points for t-SNE.")

    effective_perplexity = min(float(perplexity), max(1.0, (len(points) - 1) / 3.0))
    if effective_perplexity != float(perplexity):
        print(f"Adjusted perplexity from {perplexity} to {effective_perplexity:.2f}.")

    reducer = TSNE(
        n_components=dims,
        perplexity=effective_perplexity,
        max_iter=max_iter,
        init="pca",
        learning_rate="auto",
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(points)


def default_primary_diagnostics(num_profiles, num_vectors, method):
    diagnostics = {
        "primary_head_method": np.array([method] * num_profiles, dtype=object),
        "primary_score_margin": np.zeros(num_profiles, dtype=np.float64),
    }
    for head_idx in range(num_vectors):
        diagnostics[f"head{head_idx}_score"] = np.zeros(num_profiles, dtype=np.float64)
    return diagnostics


def compute_primary_heads(
    model,
    data_dir,
    profile_ids,
    max_depth,
    method,
    chunk_size=1_000_000,
    batch_size=65_536,
):
    triplets_path = data_dir / "profile_triplets.pt"
    num_vectors = int(model.profile_embedding.shape[1])
    default_heads = np.zeros(len(profile_ids), dtype=np.int64)
    diagnostics = default_primary_diagnostics(len(profile_ids), num_vectors, method)

    if method == "head0":
        print("Primary-head scoring disabled by method=head0; using head 0 as the primary head.")
        return default_heads, diagnostics

    if max_depth < 0:
        print("Primary-head scoring disabled; using head 0 as the primary head.")
        diagnostics["primary_head_method"][:] = "head0"
        return default_heads, diagnostics
    if not triplets_path.exists():
        print(f"Primary-head scoring skipped; missing {triplets_path}.")
        diagnostics["primary_head_method"][:] = "head0"
        return default_heads, diagnostics

    print(f"Scoring primary heads by {method} from {triplets_path.name} with hop <= {max_depth}...")
    triplets = torch.load(triplets_path, map_location="cpu")
    profile_lookup = torch.full(
        (int(model.profile_embedding.shape[0]),),
        -1,
        dtype=torch.long,
    )
    profile_lookup[torch.from_numpy(profile_ids)] = torch.arange(len(profile_ids), dtype=torch.long)

    best_hops = {}
    matched_rows = 0

    for start in range(0, len(triplets), chunk_size):
        chunk = triplets[start : start + chunk_size]
        pids = chunk[:, 0].to(dtype=torch.long)
        hops = chunk[:, 2]
        local_ids = profile_lookup[pids]
        mask = (local_ids >= 0) & (hops <= max_depth)
        if not torch.any(mask):
            continue

        selected_local = local_ids[mask].tolist()
        selected_aids = chunk[mask, 1].to(dtype=torch.long).tolist()
        selected_hops = hops[mask].to(dtype=torch.float32).tolist()
        matched_rows += len(selected_local)

        for local_id, as_id, hop in zip(selected_local, selected_aids, selected_hops):
            key = (int(local_id), int(as_id))
            old_hop = best_hops.get(key)
            if old_hop is None or hop < old_hop:
                best_hops[key] = float(hop)

    scored_pairs = len(best_hops)
    if scored_pairs == 0:
        print("No sampled profiles had hop-limited triplets; using head 0 as the primary head.")
        diagnostics["primary_head_method"][:] = "head0"
        return default_heads, diagnostics

    local_ids = torch.tensor([key[0] for key in best_hops.keys()], dtype=torch.long)
    as_ids = torch.tensor([key[1] for key in best_hops.keys()], dtype=torch.long)
    hops = torch.tensor(list(best_hops.values()), dtype=torch.float64)

    scores = torch.zeros((len(profile_ids), num_vectors), dtype=torch.float64)
    score_counts = torch.zeros(len(profile_ids), dtype=torch.float64)

    with torch.no_grad():
        for b_start in range(0, len(local_ids), batch_size):
            b_end = b_start + batch_size
            b_local = local_ids[b_start:b_end]
            b_as = as_ids[b_start:b_end]
            b_hops = hops[b_start:b_end]

            b_pids = torch.from_numpy(profile_ids[b_local.numpy()]).to(dtype=torch.long)
            p_emb = model.profile_embedding[b_pids].detach()
            a_emb = model.as_embedding[b_as].detach().unsqueeze(1)
            dists = model.manifold.dist(p_emb, a_emb).cpu().to(dtype=torch.float64)

            if method == "responsibility":
                winner_heads = torch.argmin(dists, dim=1)
                weights = 1.0 / (b_hops + 1.0)
                vote_matrix = torch.zeros((len(b_local), num_vectors), dtype=torch.float64)
                vote_matrix.scatter_(1, winner_heads.unsqueeze(1), weights.unsqueeze(1))
                scores.index_add_(0, b_local, vote_matrix)
            elif method == "mean_distance":
                scores.index_add_(0, b_local, dists)
            else:
                raise ValueError(f"Unknown primary head method: {method}")

            score_counts.index_add_(0, b_local, torch.ones(len(b_local), dtype=torch.float64))

    scored_mask = score_counts > 0
    if not torch.any(scored_mask):
        print("No sampled profiles had hop-limited triplets; using head 0 as the primary head.")
        diagnostics["primary_head_method"][:] = "head0"
        return default_heads, diagnostics

    if method == "responsibility":
        final_scores = scores
        final_scores[~scored_mask] = -torch.inf
        primary_heads = torch.argmax(final_scores, dim=1).numpy().astype(np.int64)
        sorted_scores, _ = torch.sort(final_scores, dim=1, descending=True)
    else:
        final_scores = scores / score_counts.clamp_min(1.0).unsqueeze(1)
        final_scores[~scored_mask] = torch.inf
        primary_heads = torch.argmin(final_scores, dim=1).numpy().astype(np.int64)
        sorted_scores, _ = torch.sort(final_scores, dim=1, descending=False)

    primary_heads[~scored_mask.numpy()] = 0
    if num_vectors >= 2:
        margin = (sorted_scores[:, 0] - sorted_scores[:, 1]).numpy()
        if method == "mean_distance":
            margin = -margin
    else:
        margin = np.zeros(len(profile_ids), dtype=np.float64)
    margin[~scored_mask.numpy()] = 0.0

    score_np = final_scores.numpy()
    score_np[~scored_mask.numpy()] = 0.0
    for head_idx in range(num_vectors):
        diagnostics[f"head{head_idx}_score"] = score_np[:, head_idx]
    diagnostics["primary_score_margin"] = margin
    diagnostics["primary_head_method"][~scored_mask.numpy()] = "head0"

    print(
        f"Primary-head scoring matched {matched_rows:,} triplet rows, "
        f"deduplicated to {scored_pairs:,} (profile, AS) pairs "
        f"for {int(scored_mask.sum()):,}/{len(profile_ids):,} sampled profiles."
    )
    return primary_heads, diagnostics


def build_point_roles(num_profiles, num_vectors, primary_heads):
    roles = np.full(num_profiles * num_vectors, "secondary", dtype=object)
    for local_profile, primary_head in enumerate(primary_heads):
        roles[local_profile * num_vectors + int(primary_head)] = "primary"
    return roles


def make_highlight_prefix_records(groups):
    records = []
    for group in groups:
        selected_profiles = set(group["profile_ids"])
        for prefix in group["prefixes"]:
            profile_id = group["prefix_to_profile_id"].get(prefix)
            if profile_id not in selected_profiles:
                continue
            records.append(
                {
                    "prefix": prefix,
                    "profile_id": profile_id,
                    "group": group["root"],
                }
            )
    return records


def build_prefix_coordinate_rows(
    prefix_records,
    profile_to_local,
    primary_heads,
    primary_diagnostics,
    projection,
    num_vectors,
    dims,
):
    rows = []
    seen = set()
    for record in prefix_records:
        profile_id = int(record["profile_id"])
        if profile_id not in profile_to_local:
            continue
        key = (record["prefix"], profile_id, record["group"])
        if key in seen:
            continue
        seen.add(key)

        local_profile = profile_to_local[profile_id]
        primary_head = int(primary_heads[local_profile])
        head_indices = [primary_head] if record["group"] == "background" else list(range(num_vectors))
        for head_idx in head_indices:
            point_idx = local_profile * num_vectors + head_idx
            point = projection[point_idx]
            row = {
                "prefix": record["prefix"],
                "group": record["group"],
                "profile_id": profile_id,
                "head_index": head_idx,
                "is_primary": head_idx == primary_head,
                "primary_head": primary_head,
                "primary_head_method": primary_diagnostics["primary_head_method"][local_profile],
                "primary_score_margin": float(primary_diagnostics["primary_score_margin"][local_profile]),
                "x": float(point[0]),
                "y": float(point[1]),
            }
            for key, values in primary_diagnostics.items():
                if key.startswith("head") and key.endswith("_score"):
                    row[key] = float(values[local_profile])
            if dims == 3:
                row["z"] = float(point[2])
            rows.append(row)

    rows.sort(key=lambda row: (row["x"], row["y"], row.get("z", 0.0), row["prefix"]))
    return rows


def write_prefix_coordinate_rows(rows, output_path, dims):
    score_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key.startswith("head") and key.endswith("_score")
        }
    )
    fieldnames = [
        "prefix",
        "group",
        "profile_id",
        "head_index",
        "is_primary",
        "primary_head",
        "primary_head_method",
        "primary_score_margin",
        *score_fields,
        "x",
        "y",
    ]
    if dims == 3:
        fieldnames.append("z")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def filter_rows_in_zoom_box(rows, zoom_box):
    _, x_min, x_max, y_min, y_max = zoom_box
    zoom_rows = [
        row
        for row in rows
        if x_min <= row["x"] <= x_max and y_min <= row["y"] <= y_max
    ]
    zoom_rows.sort(key=lambda row: (row["x"], row["y"], row["profile_id"], row["prefix"], row["head_index"]))
    return zoom_rows


def make_group_plot_indices(groups, profile_to_local, num_vectors, primary_heads):
    plot_groups = []
    for group in groups:
        local_profiles = [
            profile_to_local[profile_id]
            for profile_id in group["profile_ids"]
            if profile_id in profile_to_local
        ]
        primary_indices = []
        secondary_indices = []
        line_indices = []
        for local_profile in local_profiles:
            component_indices = [
                local_profile * num_vectors + component_idx
                for component_idx in range(num_vectors)
            ]
            primary_component = int(primary_heads[local_profile])
            for component_idx, point_idx in enumerate(component_indices):
                if component_idx == primary_component:
                    primary_indices.append(point_idx)
                else:
                    secondary_indices.append(point_idx)
            if len(component_indices) >= 2:
                line_indices.append(component_indices)

        plot_group = dict(group)
        plot_group["primary_indices"] = np.array(primary_indices, dtype=np.int64)
        plot_group["secondary_indices"] = np.array(secondary_indices, dtype=np.int64)
        plot_group["line_indices"] = line_indices
        plot_groups.append(plot_group)

    return plot_groups


def format_2d_axes(ax, max_ticks, prune=None):
    max_intervals = max(1, int(max_ticks) - 1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_intervals, prune=prune))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_intervals, prune=prune))
    ax.tick_params(axis="both", which="both", length=0, labelsize=FONT_SIZE_SMALL)


def make_legend_background_opaque(legend):
    legend.set_zorder(100)
    frame = legend.get_frame()
    frame.set_fill(True)
    frame.set_facecolor(OPAQUE_LEGEND_FACE_COLOR)
    frame.set_edgecolor("#cccccc")
    frame.set_alpha(1.0)


def build_main_legend_entries(plot_groups, head_display):
    if head_display == "all":
        background_label = "Other profile sub-vectors"
    else:
        background_label = "Other primary sub-vectors"

    handles = []
    labels = []
    for group in plot_groups:
        primary_handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=group["color"],
            markeredgecolor="black",
            alpha=0.95,
            markersize=7,
        )
        if head_display == "primary_only":
            handles.append(primary_handle)
        else:
            secondary_handle = Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=group["color"],
                markeredgecolor="black",
                alpha=DEFAULT_SECONDARY_ALPHA,
                markersize=7,
            )
            handles.append((primary_handle, secondary_handle))
        labels.append(f"{HIGHLIGHT_LEGEND_PREFIX} {group['root']}")

    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#9e9e9e",
            markeredgecolor="none",
            alpha=0.35,
            markersize=7,
        )
    )
    labels.append(background_label)
    return handles, labels


def plot_2d(
    projection,
    plot_groups,
    point_roles,
    head_display,
    secondary_alpha,
    output_path,
):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    primary_mask = point_roles == "primary"
    secondary_mask = point_roles == "secondary"

    if head_display == "all":
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c="#9e9e9e",
            s=8,
            alpha=DEFAULT_BACKGROUND_ALPHA,
            edgecolors="none",
            zorder=5,
        )
    else:
        ax.scatter(
            projection[primary_mask, 0],
            projection[primary_mask, 1],
            c="#9e9e9e",
            s=9,
            alpha=DEFAULT_BACKGROUND_ALPHA,
            edgecolors="none",
            zorder=6,
        )

    if head_display != "primary_only":
        for group in plot_groups:
            for component_indices in group["line_indices"]:
                xy = projection[component_indices, :2]
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    color=group["color"],
                    alpha=0.45 if head_display == "all" else DEFAULT_SECONDARY_LINE_ALPHA,
                    linewidth=0.65,
                    solid_capstyle="butt",
                    zorder=10,
                )

    for group in plot_groups:
        if head_display != "primary_only":
            secondary_indices = group["secondary_indices"]
            if len(secondary_indices) > 0:
                ax.scatter(
                    projection[secondary_indices, 0],
                    projection[secondary_indices, 1],
                    c="white",
                    s=34,
                    alpha=1.0,
                    edgecolors="none",
                    zorder=17,
                )
                ax.scatter(
                    projection[secondary_indices, 0],
                    projection[secondary_indices, 1],
                    c=group["color"],
                    s=22,
                    alpha=0.9 if head_display == "all" else secondary_alpha,
                    edgecolors="black",
                    linewidth=0.25,
                    zorder=18,
                )

        primary_indices = group["primary_indices"]
        if len(primary_indices) > 0:
            ax.scatter(
                projection[primary_indices, 0],
                projection[primary_indices, 1],
                c="white",
                s=44,
                alpha=1.0,
                edgecolors="none",
                zorder=21,
            )
            ax.scatter(
                projection[primary_indices, 0],
                projection[primary_indices, 1],
                c=group["color"],
                s=30,
                alpha=0.94,
                edgecolors="black",
                linewidth=0.35,
                zorder=22,
            )

    ax.set_title("t-SNE Projection of Profile Embeddings", fontsize=FONT_SIZE)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=FONT_SIZE)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=FONT_SIZE)
    ax.margins(x=MAIN_AXIS_MARGIN, y=MAIN_AXIS_MARGIN)
    format_2d_axes(ax, 6)
    ax.grid(True, linestyle=":", alpha=0.25)
    legend_handles, legend_labels = build_main_legend_entries(plot_groups, head_display)
    legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        frameon=True,
        facecolor=OPAQUE_LEGEND_FACE_COLOR,
        framealpha=1.0,
        fontsize=MAIN_LEGEND_FONT_SIZE,
        title_fontsize=MAIN_LEGEND_FONT_SIZE,
        handlelength=1.4,
        handletextpad=0.5,
        borderpad=0.35,
        labelspacing=0.3,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
    )
    make_legend_background_opaque(legend)
    plt.tight_layout(pad=0.35)
    plt.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _indices_in_box(points, x_min, x_max, y_min, y_max):
    return np.where(
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )[0]


def plot_zoom_2d(
    projection,
    plot_groups,
    point_roles,
    head_display,
    secondary_alpha,
    zoom_box,
    figure_size,
    output_path,
):
    name, x_min, x_max, y_min, y_max = zoom_box
    visible_point_count = 0

    fig, ax = plt.subplots(figsize=figure_size, facecolor="white")
    primary_mask = point_roles == "primary"
    secondary_mask = point_roles == "secondary"

    primary_indices = np.where(primary_mask)[0]
    primary_indices = primary_indices[
        (projection[primary_indices, 0] >= x_min)
        & (projection[primary_indices, 0] <= x_max)
        & (projection[primary_indices, 1] >= y_min)
        & (projection[primary_indices, 1] <= y_max)
    ]
    visible_point_count += len(primary_indices)
    ax.scatter(
        projection[primary_indices, 0],
        projection[primary_indices, 1],
        c="#bdbdbd",
        s=24,
        alpha=DEFAULT_ZOOM_BACKGROUND_ALPHA,
        edgecolors="none",
        zorder=5,
    )

    if head_display == "all":
        secondary_indices = np.where(secondary_mask)[0]
        secondary_indices = secondary_indices[
            (projection[secondary_indices, 0] >= x_min)
            & (projection[secondary_indices, 0] <= x_max)
            & (projection[secondary_indices, 1] >= y_min)
            & (projection[secondary_indices, 1] <= y_max)
        ]
        ax.scatter(
            projection[secondary_indices, 0],
            projection[secondary_indices, 1],
            c="#cccccc",
            s=18,
            alpha=DEFAULT_ZOOM_BACKGROUND_ALPHA,
            edgecolors="none",
            zorder=4,
        )

    if head_display != "primary_only":
        for group in plot_groups:
            for component_indices in group["line_indices"]:
                xy = projection[component_indices, :2]
                if not (
                    ((xy[:, 0] >= x_min) & (xy[:, 0] <= x_max) & (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)).any()
                ):
                    continue
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    color=group["color"],
                    alpha=0.45 if head_display == "all" else DEFAULT_SECONDARY_LINE_ALPHA,
                    linewidth=0.65,
                    solid_capstyle="butt",
                    zorder=10,
                )

    for group in plot_groups:
        if head_display != "primary_only":
            secondary_indices = group["secondary_indices"]
            secondary_indices = secondary_indices[
                (projection[secondary_indices, 0] >= x_min)
                & (projection[secondary_indices, 0] <= x_max)
                & (projection[secondary_indices, 1] >= y_min)
                & (projection[secondary_indices, 1] <= y_max)
            ]
            visible_point_count += len(secondary_indices)
            if len(secondary_indices) > 0:
                ax.scatter(
                    projection[secondary_indices, 0],
                    projection[secondary_indices, 1],
                    c="white",
                    s=52,
                    alpha=1.0,
                    edgecolors="none",
                    zorder=17,
                )
                ax.scatter(
                    projection[secondary_indices, 0],
                    projection[secondary_indices, 1],
                    c=group["color"],
                    s=34,
                    alpha=0.9 if head_display == "all" else secondary_alpha,
                    edgecolors="black",
                    linewidth=0.35,
                    zorder=18,
                )

        primary_indices = group["primary_indices"]
        primary_indices = primary_indices[
            (projection[primary_indices, 0] >= x_min)
            & (projection[primary_indices, 0] <= x_max)
            & (projection[primary_indices, 1] >= y_min)
            & (projection[primary_indices, 1] <= y_max)
        ]
        visible_point_count += len(primary_indices)
        if len(primary_indices) > 0:
            ax.scatter(
                projection[primary_indices, 0],
                projection[primary_indices, 1],
                c="white",
                s=62,
                alpha=1.0,
                edgecolors="none",
                zorder=21,
            )
            ax.scatter(
                projection[primary_indices, 0],
                projection[primary_indices, 1],
                c=group["color"],
                s=42,
                alpha=0.95,
                edgecolors="black",
                linewidth=0.45,
                zorder=22,
            )

        if head_display != "primary_only":
            secondary_indices = group["secondary_indices"]
            secondary_indices = secondary_indices[
                (projection[secondary_indices, 0] >= x_min)
                & (projection[secondary_indices, 0] <= x_max)
                & (projection[secondary_indices, 1] >= y_min)
                & (projection[secondary_indices, 1] <= y_max)
            ]
    ax.set_aspect("auto")
    format_2d_axes(ax, 4, prune="both")
    ax.grid(True, linestyle=":", alpha=0.25)
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.98)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.savefig(output_path, dpi=240)
    plt.close(fig)
    print(f"Saved zoom subplot to: {output_path} ({visible_point_count:,} visible plotted points)")


def plot_3d(
    projection,
    plot_groups,
    point_roles,
    head_display,
    secondary_alpha,
    output_path,
):
    fig = plt.figure(figsize=(11, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    primary_mask = point_roles == "primary"
    secondary_mask = point_roles == "secondary"

    if head_display == "all":
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            projection[:, 2],
            c="#9e9e9e",
            s=8,
            alpha=DEFAULT_BACKGROUND_ALPHA,
            edgecolors="none",
            label="Other profile sub-vectors",
            zorder=5,
        )
    else:
        ax.scatter(
            projection[primary_mask, 0],
            projection[primary_mask, 1],
            projection[primary_mask, 2],
            c="#9e9e9e",
            s=9,
            alpha=DEFAULT_BACKGROUND_ALPHA,
            edgecolors="none",
            label="Other primary sub-vectors",
            zorder=6,
        )

    if head_display != "primary_only":
        for group in plot_groups:
            for component_indices in group["line_indices"]:
                xyz = projection[component_indices, :3]
                ax.plot(
                    xyz[:, 0],
                    xyz[:, 1],
                    xyz[:, 2],
                    color=group["color"],
                    alpha=0.45 if head_display == "all" else DEFAULT_SECONDARY_LINE_ALPHA,
                    linewidth=0.7,
                    zorder=12,
                )

    for group in plot_groups:
        if head_display != "primary_only":
            secondary_indices = group["secondary_indices"]
            if len(secondary_indices) > 0:
                ax.scatter(
                    projection[secondary_indices, 0],
                    projection[secondary_indices, 1],
                    projection[secondary_indices, 2],
                    c=group["color"],
                    s=24,
                    alpha=0.9 if head_display == "all" else secondary_alpha,
                    edgecolors="black",
                    linewidth=0.25,
                    zorder=18,
                )

        primary_indices = group["primary_indices"]
        if len(primary_indices) > 0:
            ax.scatter(
                projection[primary_indices, 0],
                projection[primary_indices, 1],
                projection[primary_indices, 2],
                c=group["color"],
                s=32,
                alpha=0.94,
                edgecolors="black",
                linewidth=0.35,
                label=f"{HIGHLIGHT_LEGEND_PREFIX} {group['root']}",
                zorder=22,
            )

    ax.set_title("3D t-SNE Projection of Profile Embeddings", fontsize=18)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_zlabel("t-SNE Dimension 3", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.25)
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor=OPAQUE_LEGEND_FACE_COLOR,
        framealpha=1.0,
        fontsize=FONT_SIZE_SMALL,
        title="Sub-vectors",
    )
    make_legend_background_opaque(legend)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240)
    plt.close(fig)


def run_profile_dim_reduction(args):
    device = torch.device("cpu")
    output_dir = args.viz_dir / "profile_dim_reduction"
    output_dir.mkdir(parents=True, exist_ok=True)

    highlight_prefixes = parse_highlight_prefixes(args.highlight_prefixes)
    zoom_boxes = parse_zoom_boxes(args.zoom_boxes)

    print("Loading model and prefix resources...")
    model, meta = load_model(args.data_dir, args.model_dir, args.embed_dim, args.num_vectors, device)
    prefix_map, prefix_to_profile = load_prefix_resources(args.data_dir)

    print(
        "Loaded "
        f"{len(prefix_map):,} prefixes, {len(prefix_to_profile):,} prefix-profile links, "
        f"and {meta['num_profiles']:,} profiles."
    )

    print("Collecting highlighted prefixes and child prefixes...")
    highlight_groups = collect_highlight_groups(
        prefix_map,
        prefix_to_profile,
        highlight_prefixes,
        args.max_highlight_profiles,
        args.seed,
    )
    highlighted_profiles = set()
    for group in highlight_groups:
        highlighted_profiles.update(group["profile_ids"])
        print(
            f"  {group['root']}: "
            f"{len(group['prefixes']):,} child prefixes, "
            f"{len(group['profile_ids']):,}/{group['total_profile_count']:,} highlighted profiles."
        )

    background_profiles, sampled_prefix_count, background_prefix_records = sample_background_profiles(
        prefix_map,
        prefix_to_profile,
        args.sample_prefixes,
        args.seed,
    )
    print(f"Sampled prefixes: {sampled_prefix_count:,}")
    print(f"Unique sampled profiles: {len(background_profiles):,}")

    final_profiles = background_profiles | highlighted_profiles
    print(f"Final profile sample after forced highlights: {len(final_profiles):,}")

    profile_ids, points, num_vectors = build_tsne_input(model.profile_embedding, final_profiles)
    profile_to_local = {int(profile_id): local_idx for local_idx, profile_id in enumerate(profile_ids)}
    primary_heads, primary_diagnostics = compute_primary_heads(
        model,
        args.data_dir,
        profile_ids,
        args.primary_head_max_depth,
        args.primary_head_method,
    )
    point_roles = build_point_roles(len(profile_ids), num_vectors, primary_heads)
    plot_groups = make_group_plot_indices(highlight_groups, profile_to_local, num_vectors, primary_heads)

    print(f"Running {args.dims}D t-SNE on {len(points):,} profile component points...")
    projection = fit_tsne(points, args.dims, args.perplexity, args.max_iter, args.seed)

    highlight_prefix_records = make_highlight_prefix_records(highlight_groups)
    prefix_rows = build_prefix_coordinate_rows(
        background_prefix_records + highlight_prefix_records,
        profile_to_local,
        primary_heads,
        primary_diagnostics,
        projection,
        num_vectors,
        args.dims,
    )
    coordinate_path = output_dir / f"profile_tsne_{args.dims}d_prefix_coordinates.csv"
    write_prefix_coordinate_rows(prefix_rows, coordinate_path, args.dims)
    print(f"Saved prefix coordinate table to: {coordinate_path}")

    output_path = output_dir / f"profile_tsne_{args.dims}d.png"
    if args.dims == 2:
        plot_2d(
            projection,
            plot_groups,
            point_roles,
            args.head_display,
            args.secondary_alpha,
            output_path,
        )
    else:
        plot_3d(
            projection,
            plot_groups,
            point_roles,
            args.head_display,
            args.secondary_alpha,
            output_path,
        )

    print(f"Saved visualization to: {output_path}")

    if args.dims == 2 and not args.disable_zoom_plots:
        if args.zoom_figure_size is None:
            zoom_figure_size = (args.zoom_figure_width, args.zoom_figure_height)
        else:
            zoom_figure_size = (args.zoom_figure_size, args.zoom_figure_size)
        for zoom_box in zoom_boxes:
            zoom_name = zoom_box[0]
            zoom_output_path = output_dir / f"profile_tsne_2d_zoom_{zoom_name}.png"
            zoom_coordinate_path = output_dir / f"profile_tsne_2d_zoom_{zoom_name}_prefix_coordinates.csv"
            zoom_coordinate_rows = filter_rows_in_zoom_box(prefix_rows, zoom_box)
            write_prefix_coordinate_rows(zoom_coordinate_rows, zoom_coordinate_path, args.dims)
            print(
                f"Saved zoom coordinate table to: {zoom_coordinate_path} "
                f"({len(zoom_coordinate_rows):,} rows)"
            )
            plot_zoom_2d(
                projection,
                plot_groups,
                point_roles,
                args.head_display,
                args.secondary_alpha,
                zoom_box,
                zoom_figure_size,
                zoom_output_path,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--model_dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--viz_dir", type=Path, default=Path("visualizations"))
    parser.add_argument("--embed_dim", type=int, default=24)
    parser.add_argument("--num_vectors", type=int, default=2)
    parser.add_argument("--sample_prefixes", type=int, default=20000)
    parser.add_argument("--dims", type=int, choices=[2, 3], default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument(
        "--primary_head_max_depth",
        type=int,
        default=3,
        help="Use triplets with hop <= this value to choose each profile's primary head. Use -1 to disable.",
    )
    parser.add_argument(
        "--primary_head_method",
        choices=["responsibility", "mean_distance", "head0"],
        default="responsibility",
        help="Method used to choose the primary head for each profile.",
    )
    parser.add_argument(
        "--head_display",
        choices=["fade_secondary", "primary_only", "all"],
        default="fade_secondary",
        help="How to display non-primary heads after primary-head scoring.",
    )
    parser.add_argument(
        "--secondary_alpha",
        type=float,
        default=DEFAULT_SECONDARY_ALPHA,
        help="Alpha for secondary highlighted heads and connecting lines in fade_secondary mode.",
    )
    parser.add_argument(
        "--disable_zoom_plots",
        action="store_true",
        help="Disable extra 2D zoom subplot generation.",
    )
    parser.add_argument(
        "--zoom_boxes",
        nargs="*",
        default=None,
        help=(
            "Optional zoom boxes. Each item is 'x_min,x_max,y_min,y_max' or "
            "'name,x_min,x_max,y_min,y_max'. Defaults to four fixed boxes."
        ),
    )
    parser.add_argument(
        "--zoom_figure_width",
        type=float,
        default=3.2,
        help="Figure width in inches for each zoom subplot.",
    )
    parser.add_argument(
        "--zoom_figure_height",
        type=float,
        default=3.2,
        help="Figure height in inches for each zoom subplot.",
    )
    parser.add_argument(
        "--zoom_figure_size",
        type=float,
        default=None,
        help="Legacy square figure size in inches for each zoom subplot. Overrides width and height if set.",
    )
    parser.add_argument(
        "--max_highlight_profiles",
        type=int,
        default=50,
        help="Maximum highlighted profiles sampled per prefix group. Use 0 to keep all.",
    )
    parser.add_argument(
        "--highlight-prefixes",
        nargs="*",
        default=None,
        help="Optional comma- or space-separated root prefixes to highlight.",
    )
    args = parser.parse_args()
    run_profile_dim_reduction(args)


if __name__ == "__main__":
    main()
