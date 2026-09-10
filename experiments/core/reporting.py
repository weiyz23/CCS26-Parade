"""Export training measurements without inventing missing CPU/GPU statistics."""

import math

from .runtime import write_csv


FIELDS = ["run", "geometry", "embed_dim", "num_vectors", "seed", "status", "best_loss",
          "configured_epochs", "total_train_time_sec", "early_stop_mark_epoch", "early_stop_mark_cum_time_sec",
          "max_cuda_peak_alloc_mb", "max_cuda_peak_reserved_mb", "wall_time_sec", "model_dir", "error"]


def summarize(rows, output_dir, axis, *, figure_stem="metrics"):
    write_csv(output_dir / "summary.csv", rows, FIELDS)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_key = "embed_dim" if axis == "dimension" else "num_vectors"
    panels = [("best_loss", "Best training loss"), ("total_train_time_sec", "Training time (s)"),
              ("early_stop_mark_cum_time_sec", "Early-stop mark time (s)"),
              ("max_cuda_peak_reserved_mb", "Peak reserved CUDA memory (MiB)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    completed = [row for row in rows if row["status"] == "complete"]
    groups = sorted({(row["geometry"], row["seed"]) for row in completed}, key=lambda group: (group[0], str(group[1])))
    for ax, (metric, label) in zip(axes.flat, panels):
        plotted = False
        for geometry, seed in groups:
            points = sorted((row for row in completed if row["geometry"] == geometry and row["seed"] == seed),
                            key=lambda row: row[x_key])
            ys = [row.get(metric) for row in points]
            if not any(y is not None and math.isfinite(y) for y in ys):
                continue
            ax.plot([row[x_key] for row in points], [float("nan") if y is None else y for y in ys],
                    marker="o", label=f"{geometry}, seed={seed}")
            plotted = True
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Embedding dimension" if axis == "dimension" else "Profile vector count K")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{axis.capitalize()} sensitivity: {len(completed)}/{len(rows)} runs completed")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{figure_stem}.{extension}", dpi=160)
    plt.close(fig)
