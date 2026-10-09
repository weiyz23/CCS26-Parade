#!/usr/bin/env python3
"""Replay the archived mixed benchmark."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core.runtime import absolute, read_json, sha256, write_json

POLICIES = ("archived", "aligned")
TIERS = ("[1,500]", "(500,2000]", "(2000,5000]", "(5000,15000]", ">15000")
DATA_FILES = ("meta.json", "as_map.pt", "prefix_map.pt", "prefix_to_profile.pt",
              "profile_fingerprints.pt", "fingerprint_anchor_db.pt", "history_cache.pt")
MODEL_FILES = ("best_model.pth", "thresholds.json")
RESULT_FILES = ("simulation_manifest.parquet", "mixed_manifest.parquet", "mixed_scores.parquet",
                "train_config.json", "run_config.json", "mixed_metrics.csv", "mixed_summary.json",
                "mixed_recall.csv", "mixed_heatmaps_archived.png", "mixed_heatmaps_archived.pdf",
                "miss_attribution.json", "miss_attribution.csv", "archive_comparison.json")


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=absolute, help="Revision directory or inference bundle")
    parser.add_argument("--data-dir", type=absolute, help="Inference tensors")
    parser.add_argument("--model-dir", type=absolute, help="Checkpoint and thresholds")
    parser.add_argument("--as-info", type=absolute, help="AS organization metadata")
    parser.add_argument("--embed-dim", type=positive)
    parser.add_argument("--num-vectors", type=positive)
    parser.add_argument("--output-dir", type=absolute,
                        default=ROOT / "experiments" / "results" / "simulation_archived_mixed" / "results")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_inputs(args):
    local = ROOT.parent / "ReprLearningForBGP" / "dataset" / "revision_20260101"
    archive = args.archive_dir or (local if local.is_dir() else ROOT / "dataset" / "revision_20260101")
    return dict(archive=archive,
                data_dir=args.data_dir or (archive / "data" if (archive / "data").is_dir() else archive / "coverage" / "c100" / "data"),
                model_dir=args.model_dir or (archive / "model" if (archive / "model").is_dir() else archive / "coverage" / "c100" / "model"),
                as_info=args.as_info or archive / "as_info.2601.jsonl")


def confusion_metrics(labels, predictions):
    import numpy as np

    y, p = np.asarray(labels, dtype=bool), np.asarray(predictions, dtype=bool)
    tp, fn = int((y & p).sum()), int((y & ~p).sum())
    fp, tn = int((~y & p).sum()), int((~y & ~p).sum())
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return dict(tn=tn, fp=fp, fn=fn, tp=tp,
                recall=tp / (tp + fn) if tp + fn else None,
                precision=tp / (tp + fp) if tp + fp else None,
                fpr=fp / (fp + tn) if fp + tn else None,
                mcc=(tp * tn - fp * fn) / denominator if denominator else None)


def add_history_policy(scored, history, organizations, history_names=None, organization_names=None):
    from hyperbolic_model.filter import organization_exemption

    scored = scored.copy()
    reasons = [organization_exemption(pid, asn, organizations, organization_names or {},
                                      history, history_names or {})
               for asn, pid in zip(scored["asn"], scored["profile_id"])]
    scored["history_filtered"] = [reason == "org_id" for reason in reasons]
    scored["organization_name_filtered"] = [reason == "org_name" for reason in reasons]
    scored["organization_filtered"] = scored["history_filtered"] | scored["organization_name_filtered"]
    scored["prediction_archived"] = scored["prediction"] & ~scored["history_filtered"]
    scored["prediction_aligned"] = scored["prediction"] & ~scored["organization_filtered"]
    return scored


def compare_saved_scores(actual, expected):
    import pandas as pd

    keys = ["sample_id", "repeat_seed"]
    for name, frame in (("replayed", actual), ("saved", expected)):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name} scores have null or duplicate sample keys")
    replayed, saved = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if not replayed.index.equals(saved.index):
        raise ValueError("Replayed and saved scores have different complete sample key sets")
    if (replayed["label"].to_numpy() != saved["label"].to_numpy()).any():
        raise ValueError("Replayed and saved sample labels differ")
    old_prediction = "detected_alpha1" if "detected_alpha1" in saved else "prediction_archived"
    differences = int((replayed["prediction_archived"].to_numpy() != saved[old_prediction].to_numpy()).sum())
    if differences:
        raise ValueError(f"Archived prediction replay differs for {differences} of {len(actual)} samples")
    metrics = pd.DataFrame([confusion_metrics(group["label"], group[old_prediction])
                            for _, group in expected.groupby("repeat_seed", sort=True)])
    return dict(samples_compared=len(actual), prediction_differences=differences,
                saved_mean=metrics[["recall", "precision", "fpr", "mcc"]].mean().to_dict())


def report_mixed(scored, output_dir):
    import pandas as pd

    rows = [dict(policy=policy, repeat_seed=int(seed), samples=len(group),
                 missing=int(group["missing"].sum()),
                 **confusion_metrics(group["label"], group[f"prediction_{policy}"]))
            for policy in POLICIES for seed, group in scored.groupby("repeat_seed", sort=True)]
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "mixed_metrics.csv", index=False)
    means = table.groupby("policy")[["recall", "precision", "fpr", "mcc"]].mean()
    write_json(output_dir / "mixed_summary.json", {
        "repeats": rows, "mean": json.loads(means.to_json(orient="index")),
        "aggregation": "Mean over repeats with fixed positives",
        "recall_definition": "Candidate-level TP / (TP + FN)",
        "policies": {"archived": "Fingerprint and orgId exemptions",
                     "aligned": "Additional organization-name exemptions"},
    })
    return means


def report_misses(mixed, output_dir):
    import pandas as pd

    positives = mixed[(mixed["label"] == 1) & (mixed["repeat_seed"] == mixed["repeat_seed"].min())]

    def counts(frame, **group):
        geom, fp = frame["geometry_prediction"].astype(bool), frame["anchor_filtered"].astype(bool)
        org_id, org_name = frame["history_filtered"].astype(bool), frame["organization_name_filtered"].astype(bool)
        org = org_id | org_name
        row = dict(group, samples=len(frame), missing=int(frame["missing"].sum()),
                   geometry_misses=int((~geom).sum()), fingerprint_only_misses=int((geom & fp & ~org).sum()),
                   organization_id_only_misses=int((geom & ~fp & org_id).sum()),
                   organization_name_only_misses=int((geom & ~fp & org_name).sum()),
                   both_exemptions_misses=int((geom & fp & org).sum()),
                   fingerprint_flags_on_geometry_negatives=int((~geom & fp).sum()),
                   organization_flags_on_geometry_negatives=int((~geom & org).sum()),
                   detected_geometry=int(geom.sum()), detected_archived=int(frame["prediction_archived"].sum()),
                   detected_aligned=int(frame["prediction_aligned"].sum()),
                   exemption_caused_misses=int((geom & (fp | org)).sum()),
                   misses_aligned=len(frame) - int(frame["prediction_aligned"].sum()))
        if row["geometry_misses"] + row["exemption_caused_misses"] != row["misses_aligned"]:
            raise ValueError("Miss attribution does not match final predictions")
        return row

    summary = counts(positives, dataset="mixed_base")
    scenarios = [counts(group, dataset="mixed_base", scenario=scenario)
                 for scenario, group in positives.groupby("scenario")]
    pd.DataFrame([summary] + scenarios).to_csv(output_dir / "miss_attribution.csv", index=False)
    write_json(output_dir / "miss_attribution.json", {
        "method": "Geometry first; exemption categories are disjoint",
        "mixed_counting": "Fixed positives from the first repeat",
        "summary": summary, "by_scenario": scenarios,
    })


def report_mixed_heatmaps(scored, output_dir, policy="archived"):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positives = scored[(scored["label"] == 1) & (scored["repeat_seed"] == scored["repeat_seed"].min())]
    rows = [dict(policy=policy, scenario=scenario, rank_tier=tier, depth=int(depth),
                 samples=len(group), tp=int(group[f"prediction_{policy}"].sum()),
                 fn=len(group) - int(group[f"prediction_{policy}"].sum()),
                 missing=int(group["missing"].sum()), recall=float(group[f"prediction_{policy}"].mean()))
            for (scenario, tier, depth), group in positives.groupby(["scenario", "rank_tier", "depth"])]
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "mixed_recall.csv", index=False)
    tiers = [tier for tier in TIERS if tier in set(table["rank_tier"])]
    scenarios = [("origin_hijack", [-1], "Origin hijack"),
                 ("path_manipulation", list(range(5)), "Path manipulation"),
                 ("route_leak", list(range(5)), "One-shot route leak")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 5, 5]})
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("lightgray")
    for ax, (scenario, depths, title) in zip(axes, scenarios):
        cells = table[table["scenario"] == scenario]
        matrix = np.full((len(tiers), len(depths)), np.nan)
        counts = np.zeros(matrix.shape, dtype=int)
        for row in cells.itertuples():
            i, j = tiers.index(row.rank_tier), depths.index(row.depth)
            matrix[i, j], counts[i, j] = row.recall, row.samples
        plot = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="equal")
        for i in range(len(tiers)):
            for j in range(len(depths)):
                value = matrix[i, j]
                label = "N/A" if np.isnan(value) else f"{value:.3f}\nn={counts[i, j]}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color="white" if value < .2 or value > .8 else "black")
        recall = cells["tp"].sum() / cells["samples"].sum() if len(cells) else float("nan")
        ax.set(xticks=range(len(depths)), xticklabels=["Origin"] if depths == [-1] else depths,
               yticks=range(len(tiers)), yticklabels=[TIERS.index(tier) + 1 for tier in tiers],
               xlabel="" if depths == [-1] else "Propagation depth", ylabel="Rank tier",
               title=f"{title}\nRecall {recall:.3f}")
        ax.set_xticks(np.arange(-.5, len(depths)), minor=True)
        ax.set_yticks(np.arange(-.5, len(tiers)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="both", length=0)
    fig.colorbar(plot, ax=axes, label="Recall")
    fig.suptitle(f"Archived mixed-set recall — {policy} policy\n{len(positives):,} unique anomalies; repeated seeds counted once", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"mixed_heatmaps_{policy}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return table


def export_bundle(output_dir, data_dir, model_dir, as_info, history, profile_ids, history_names):
    import torch

    bundle = output_dir / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    for part in ("data", "model"):
        (bundle / part).mkdir(parents=True)
    for name in DATA_FILES:
        if name != "history_cache.pt":
            shutil.copy2(data_dir / name, bundle / "data" / name)
    torch.save({"org_ids": {pid: history[pid] for pid in profile_ids if pid in history},
                "org_names": {pid: history_names[pid] for pid in profile_ids if pid in history_names}},
               bundle / "data" / "history_cache.pt")
    for name in MODEL_FILES:
        shutil.copy2(model_dir / name, bundle / "model" / name)
    shutil.copy2(as_info, bundle / "as_info.2601.jsonl")
    for name in RESULT_FILES:
        source = output_dir / name
        if source.is_file():
            shutil.copy2(source, bundle / name)
    files = [f"data/{name}" for name in DATA_FILES] + [f"model/{name}" for name in MODEL_FILES]
    files += ["as_info.2601.jsonl"] + [name for name in RESULT_FILES if (bundle / name).is_file()]
    write_json(bundle / "manifest.json", {
        "files": {name: sha256(bundle / name) for name in files},
        "usage": "bash experiments/run_simulation.sh --archive-dir BUNDLE --output-dir RESULTS",
    })
    with tarfile.open(output_dir / "simulation_bundle.tar.gz", "w:gz") as package:
        for name in files + ["manifest.json"]:
            package.add(bundle / name, arcname=f"bundle/{name}", recursive=False)
    return bundle


def run(args):
    import pandas as pd
    import torch
    from hyperbolic_model.detect import load_resources, load_thresholds, score_candidates
    from hyperbolic_model.filter import load_resources as load_organization_resources

    inputs = resolve_inputs(args)
    archive, data_dir, model_dir, as_info = (inputs[name] for name in ("archive", "data_dir", "model_dir", "as_info"))
    out = args.output_dir
    if any(source == out or source.is_relative_to(out) for source in (archive, data_dir, model_dir, as_info)):
        raise ValueError("Output directory must not contain any input directory or file")
    for path in [archive / "mixed_manifest.parquet", as_info] + [data_dir / name for name in DATA_FILES] + [model_dir / name for name in MODEL_FILES]:
        if not path.is_file():
            raise FileNotFoundError(f"Required replay input is missing: {path}")
    cfg_path = model_dir.parent / "train_config.json"
    cfg = read_json(cfg_path if cfg_path.is_file() else ROOT / "experiments" / "configs" / "hyperbolic_base.json")
    embed_dim, num_vectors = args.embed_dim or cfg["model"]["embed_dim"], args.num_vectors or cfg["model"]["num_vectors"]
    cfg["model"].update(embed_dim=embed_dim, num_vectors=num_vectors)
    cfg["paths"] = dict(data_dir="data", model_dir="model")
    torch.set_num_threads(2)
    mixed = pd.read_parquet(archive / "mixed_manifest.parquet")
    print(f"Loading model and scoring {len(mixed):,} frozen mixed candidates", flush=True)
    resources, thresholds = load_resources(data_dir, model_dir, embed_dim, num_vectors), load_thresholds(data_dir, model_dir)
    _, _, history, history_names, organizations, _, organization_names = load_organization_resources(data_dir, as_info)
    scored = add_history_policy(score_candidates(mixed, resources, thresholds, anchor_for_all=True),
                                history, organizations, history_names, organization_names)
    old_scores = archive / "coverage" / "c100" / "mixed_scores.parquet"
    if not old_scores.is_file():
        old_scores = archive / "mixed_scores.parquet"
    comparison = compare_saved_scores(scored, pd.read_parquet(old_scores)) if old_scores.is_file() else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "archive_comparison.json").unlink(missing_ok=True)
    mixed.to_parquet(out / "mixed_manifest.parquet", index=False)
    scored.to_parquet(out / "mixed_scores.parquet", index=False)
    mixed[(mixed["label"] == 1) & (mixed["repeat_seed"] == mixed["repeat_seed"].min())].to_parquet(out / "simulation_manifest.parquet", index=False)
    write_json(out / "train_config.json", cfg)
    write_json(out / "run_config.json", {
        "mode": "frozen_mixed_replay", "source_time_utc": "2026-01-01 00:00:00", "live_rpki": False,
        "source_paths": {name: str(path) for name, path in inputs.items()},
        "model": dict(embed_dim=embed_dim, num_vectors=num_vectors), "policies": list(POLICIES),
        "repeat_seeds": sorted(int(seed) for seed in mixed["repeat_seed"].unique()),
        "inputs": {"checkpoint": sha256(model_dir / "best_model.pth"), "thresholds": sha256(model_dir / "thresholds.json"),
                   "as_info": sha256(as_info), **{name: sha256(data_dir / name) for name in DATA_FILES}},
        "manifests": {"mixed_manifest.parquet": sha256(archive / "mixed_manifest.parquet")},
        "saved_scores": sha256(old_scores) if old_scores.is_file() else None,
        "comparison": "keys, labels and archived predictions" if comparison is not None else "saved scores unavailable",
    })
    if comparison is not None:
        write_json(out / "archive_comparison.json", comparison)
    means = report_mixed(scored, out)
    report_mixed_heatmaps(scored, out)
    report_misses(scored, out)
    bundle = export_bundle(out, data_dir, model_dir, as_info, history,
                           set(int(pid) for pid in scored["profile_id"] if pid >= 0), history_names)
    print(means.to_string(float_format=lambda value: f"{value:.7f}"), flush=True)
    print(f"Results: {out}\nInference bundle: {bundle}", flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.dry_run:
            inputs = resolve_inputs(args)
            print("Mode: frozen mixed replay")
            for name, path in inputs.items():
                print(f"{name}: {path}")
            print(f"Output: {args.output_dir}\nSaved thresholds; archived and aligned policies.")
            return 0
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Mixed replay failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
