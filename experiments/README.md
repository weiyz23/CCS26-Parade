# AE experiments

PARADE experiments use the [repository environment](../README.md#environment).
Each launcher supports `--help`, `--dry-run` and `--output-dir`.

```bash
# All eight historical events, or KLAYswap alone.
bash experiments/run_main.sh
bash experiments/run_main.sh --datasets klayswap
```

| Entry point | Default workload |
| --- | --- |
| `run_main.sh` | Eight historical events |
| `run_dimension.sh` | Hyperbolic dimension 12/16/24/32/48/64, K=2 |
| `run_k.sh` | K=1–6, dimension 24, hyperbolic and Euclidean |
| `run_simulation.sh` | Saved mixed benchmark, five 4:1 repeats |

## Historical events

```bash
bash experiments/run_main.sh --datasets klayswap,celer-bridge \
  --raw-data-base /path/to/bgp --output-dir experiments/results/main_subset
```

Events and UTC windows are defined in
[historical_events.json](configs/historical_events.json); `rostelecom-2` is the
2020 event. Each run downloads/reuses observations and AS metadata, then executes
preprocessing, training, calibration, detection, filtering and visualization.
Defaults are dimension 24, K=2 and seed 42; `TRAIN_*` environment overrides apply.

Raw RIB/UPDATE files are downloaded from [RIPE RIS](https://data.ris.ripe.net/)
and [RouteViews](https://archive.routeviews.org/) into `data/raw/<event-id>/`
by default; `--raw-data-base` changes that root. Parsed data/caches are under
`dataset/<event-id>/`, models under its `saved_models/`, and detection outputs
under its `results/`. Logs and `summary.csv/json` are under
`experiments/results/main/`. **For historical events, `--output-dir` changes
only logs and summaries, not dataset/cache/model paths.** The author's full
experiments accumulated over 1 TB of raw files and over 200 GB of intermediate
caches, not per-event minimum requirements.

Plotting has separate `visualization_status`
and `visualization_error` fields. Main-pipeline failures return a nonzero exit code.
Run one historical replay at a time.

To summarize existing outputs without rerunning an event:

```bash
python experiments/scripts/summarize_events.py \
  --datasets rostelecom-2,vodafone,celer-bridge,safehost,tic,voxility,cloudflare,klayswap \
  --dataset-root dataset --output-dir experiments/results/historical_summary
```

`historical_metrics.csv` reports candidate/detection rows and unique prefixes,
ASNs and saved AS groups before and after organization/RPKI filtering.
Fingerprint exemptions already apply to detector outputs; filtering-reason counts
can overlap. These are run statistics rather than Table 1 ground-truth metrics.

## Dimension and K

```bash
bash experiments/run_dimension.sh
bash experiments/run_k.sh

bash experiments/run_dimension.sh --data-dir /path/to/tensors --dimensions 12,24,48
bash experiments/run_k.sh --data-dir /path/to/tensors --k-values 1,2 --geometry hyperbolic
```

Default input is the 2026-01-01 00:00 UTC RIB snapshot from 74 collectors
(23 RIS, 51 RouteViews), specified in [snapshot_20260101.json](configs/snapshot_20260101.json).
It is prepared once under `dataset/ae_snapshot_20260101/` without UPDATE paths.
`--data-dir` uses existing `meta.json`, `profile_triplets.pt` and
`profile_fingerprints.pt` instead.

Training settings come from [hyperbolic_base.json](configs/hyperbolic_base.json)
and [euclidean_base.json](configs/euclidean_base.json). Sensitivity runs use 800
epochs and record the early-stop point while continuing training. Use `--config`,
`--seeds`, `--epochs` or `--batch-size` to override settings.

Outputs under `experiments/results/{dimension,k}/` include checkpoints, effective
`train_config.json`, `summary.csv/json` and `metrics.png/pdf`. Metrics cover
training loss, time and memory.

## Archived mixed benchmark

The saved January 1, 2026 benchmark contains 2,000 OH, 2,000 PM and 2,000 RL
anomalies, with attacker/recipient rank > 2,000. Five control sets use seeds 42–46;
each repeat has 6,000 fixed anomalies and 24,000 controls. Replay needs only the
Python dependencies, saved samples and inference bundle.

Download the two assets from the fixed
[v1.1.0 release](https://github.com/weiyz23/CCS26-Parade/releases/tag/v1.1.0):
[simulation_bundle.tar.gz](https://github.com/weiyz23/CCS26-Parade/releases/download/v1.1.0/simulation_bundle.tar.gz)
(179 MB) and
[SHA256SUMS](https://github.com/weiyz23/CCS26-Parade/releases/download/v1.1.0/SHA256SUMS).

```bash
mkdir -p data/ae-release-v1.1.0
curl -fL https://github.com/weiyz23/CCS26-Parade/releases/download/v1.1.0/simulation_bundle.tar.gz \
  -o data/ae-release-v1.1.0/simulation_bundle.tar.gz
curl -fL https://github.com/weiyz23/CCS26-Parade/releases/download/v1.1.0/SHA256SUMS \
  -o data/ae-release-v1.1.0/SHA256SUMS
(cd data/ae-release-v1.1.0 && sha256sum -c SHA256SUMS)
tar -xzf data/ae-release-v1.1.0/simulation_bundle.tar.gz -C data/ae-release-v1.1.0
bash experiments/run_simulation.sh \
  --archive-dir data/ae-release-v1.1.0/bundle --output-dir experiments/results/mixed
```

The bundle contains `mixed_manifest.parquet`, `data/`, `model/`,
`as_info.2601.jsonl` and `train_config.json`. Override locations with `--data-dir`,
`--model-dir` and `--as-info`; dimensions with `--embed-dim` and `--num-vectors`.
Data and model must use the same mappings. Without `--archive-dir`, defaults use
the adjacent `ReprLearningForBGP/dataset/revision_20260101`, or
`dataset/revision_20260101` inside this repository.

Recall is `TP / (TP + FN)` over positive candidates. MCC is averaged over the
five repeats. The `archived` policy uses fingerprint and orgId exemptions:
Recall **0.8541667**, mean MCC **0.8975295**. `aligned` additionally applies
organization-name exemptions and is reported separately.

Outputs include all 150,000 scored rows, per-repeat metrics, miss attribution,
`mixed_heatmaps_archived.{png,pdf}` and an inference bundle. The heatmap counts
the 6,000 anomalies once. Available saved predictions are compared by complete
sample keys and labels; mismatches fail the replay.

Figure 9's independent tier–depth diagnostic manifests were not recovered;
those diagnostics are outside this artifact's reproduction scope.
