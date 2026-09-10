# Artifact Evaluation experiments

This directory contains the experiment orchestration for PARADE. Run the three
entry points below after installing the environment described in the
[root README](../README.md#environment). They reuse the existing preprocessing,
training, calibration, detection, filtering, and visualization implementations.

| Entry point | Experiment | Default workload |
| --- | --- | --- |
| `run_main.sh` | Historical event evaluation | Eight events, sequentially; download through detection, filtering, and visualization |
| `run_dimension.sh` | Embedding dimension sensitivity | Hyperbolic geometry, K=2, dimensions 12/16/24/32/48/64 |
| `run_k.sh` | Profile-vector count and geometry comparison | Dimension 24, K=1–6, both hyperbolic and Euclidean geometry |

The main experiment runs PARADE on the historical events. It does not run the
other systems compared in the paper. The sensitivity experiments report
training loss and resource costs; they do not calculate labeled detection
Recall, Precision, FPR, or MCC.

## Environment and paths

Use Linux, Python 3.12, the Python dependencies in `requirements.txt`, CMake,
a C++17 compiler, and `libbgpstream` headers/libraries. A CUDA-capable PyTorch
installation is recommended for full experiments. The entry points prepend
`<repo>/.conda/bin` to PATH when that environment exists, otherwise use `python`
from PATH. Needed parser binaries are built automatically with CMake.

The scripts work from any current directory. Paths supplied as arguments are
resolved relative to the caller's working directory, including paths with spaces.
The defaults below are resolved relative to the repository:

| Content | Default location |
| --- | --- |
| Downloaded raw observations and monthly AS metadata cache | `data/raw/` (or `RAW_DATA_BASE`) |
| Main experiment dataset and tensors | `dataset/<event-id>/` |
| Main models, calibrated thresholds, detections, plots | `dataset/<event-id>/{saved_models,results,visualizations}/` |
| Shared sensitivity snapshot and tensors | `dataset/ae_snapshot_20260101/` |
| Experiment logs, models for sensitivity runs, summaries, and summary plots | `experiments/results/{main,dimension,k}/` |

Full downloads and preprocessing require substantial disk space, host memory,
and GPU time. Historical archives and monthly CAIDA AS Rank metadata require
network access. The retained origin filter also queries **current** RIPEstat
RPKI validation; its results can change independently of the historical BGP
observations. Existing detection and filtering rules are unchanged.

Only run one main experiment at a time: the underlying replay pipeline shares
PID files and shared-memory names. The AE entry point uses a lock to reject a
second main run. Avoid running legacy preprocessing daemons concurrently.
Sensitivity runs also execute sequentially, and both scripts share a locked
snapshot preparation step.

## 1. Eight historical events

The authoritative AE event list is `configs/historical_events.json`. Times below
are UTC and retain the existing replay windows.

| Event ID | Paper event | Type | Date | Start | End |
| --- | --- | --- | --- | --- | --- |
| `rostelecom-2` | Rostelecom (2020) | OH | 2020-04-01 | 18:00:00 | 21:00:00 |
| `vodafone` | Vodafone | OH | 2021-04-16 | 12:00:00 | 15:00:00 |
| `celer-bridge` | Celer bridge | OH | 2022-08-17 | 18:00:00 | 22:00:00 |
| `safehost` | Safehost | RL | 2019-06-06 | 08:00:00 | 13:00:00 |
| `tic` | TIC | RL | 2023-03-12 | 12:00:00 | 16:00:00 |
| `voxility` | Voxility | RL | 2025-04-01 | 06:00:00 | 10:00:00 |
| `cloudflare` | Cloudflare | RL | 2026-01-22 | 20:00:00 | 23:59:00 |
| `klayswap` | KLAYswap | PM | 2022-02-03 | 00:00:00 | 05:00:00 |

`rostelecom-2` is the existing repository ID for the **2020** event. The 2017
Rostelecom event (`rostelecom`) and CANTV are case studies, not part of these eight.

From the repository root:

```bash
# Inspect all planned commands without downloading or writing experiment outputs.
bash experiments/run_main.sh --dry-run

# Run all eight events in the table order.
bash experiments/run_main.sh

# Run a subset; raw-data paths are configurable.
bash experiments/run_main.sh --datasets klayswap,celer-bridge \
  --raw-data-base /mnt/bgp_raw --output-dir experiments/results/main_subset
```

For each event, the script prepares AS metadata for the first day of the event's
month, installs it as `as_info.jsonl` in that event's dataset directory, then
calls `preprocess/run.sh full-once`. Organization history is built after the
metadata is available. Existing raw downloads and preprocessing caches are
reused according to the existing pipeline. Model training, threshold calibration,
detection, filtering, and visualization execute for each invocation.

The main model defaults are dimension **24**, K=**2**, 128 positive samples and
384 negative samples per profile, batch size **4096**, learning rate **0.002**,
and seed **42**. Training has a **1000**-epoch maximum and stops with patience
**80**, delta **1e-5**. Calibration uses the **99.95th** positive and **1.6th**
negative percentiles. Existing `EMBEDDING_DIM`, `NUM_VECTORS`, and `TRAIN_*`
environment overrides are retained; for example, `TRAIN_EPOCHS=2` reduces the
training duration for an execution check, not a paper-scale reproduction.

Each event gets `run_config.json`, `run.log`, and a `pipeline/` log directory
under the chosen experiment output directory. The effective training settings
are also saved in `dataset/<event-id>/saved_models/train_config.json`.
`summary.csv` and `summary.json` record status, elapsed time, output paths, and
errors. A failed event does not prevent later events from running; any failure
makes the final exit status nonzero. No model is retrained based on whether a
known culprit or prefix was detected.

## 2. Dimension sensitivity

```bash
bash experiments/run_dimension.sh --dry-run
bash experiments/run_dimension.sh
```

The first real invocation downloads the **2026-01-01 00:00 UTC RIB** for the
74 collectors listed in `configs/snapshot_20260101.json` (23 RIS, 51 RouteViews),
then runs the existing processor and tensor/anchor builder. It does not add
UPDATE messages or use the historical-event bootstrap window for this snapshot.
The collector list was extracted from the full collector fold of the existing
`dataset/revision_20260101/collector_manifest.json`; the portable configuration
does not depend on that local manifest or its absolute archive paths.
All expected collectors must be present. Archive decompression, per-RIB usable
observations, and generated artifacts are checked before preparation is marked
complete. This experiment never silently substitutes a smaller collector set.

`download_manifest.json` records archive URLs, selected collectors, and raw file
sizes. `preparation_manifest.json` records the snapshot specification,
preprocessing source hashes, artifact sizes/timestamps, and preparation time.
The shared tensors are reused when that completed manifest still matches.
Incomplete or changed preparation is rebuilt in the managed AE snapshot
location. Existing published revision experiment directories are not modified.

The default sweep runs six models, all with K=2. Each model has a separate
`hyperbolic_dim<D>_seed<S>/` output directory.

```bash
# Use already prepared tensors; no snapshot download or parser build is needed.
bash experiments/run_dimension.sh --data-dir dataset/klayswap \
  --dimensions 12,24,48 --output-dir experiments/results/dimension_klayswap

# A small execution check, using an existing small dataset.
bash experiments/run_dimension.sh --data-dir /tmp/parade-small-data \
  --dimensions 12,24 --epochs 2 --batch-size 4 \
  --num-samples-per-profile 2 --device cpu \
  --output-dir /tmp/parade-dimension-check
```

`--data-dir` must contain `meta.json`, `profile_triplets.pt`, and
`profile_fingerprints.pt`; point to the tensor directory itself, not necessarily
`processed_data/`. Root-level tensors are the current repository layout.

## 3. K sensitivity and geometry comparison

```bash
bash experiments/run_k.sh --dry-run
bash experiments/run_k.sh

# Select part of the sweep, or one geometry.
bash experiments/run_k.sh --k-values 1,2 --geometry hyperbolic \
  --data-dir dataset/ae_snapshot_20260101
```

The default invocation reuses the same automatically prepared snapshot as the
dimension experiment, fixes the dimension at **24**, and trains K=1–6 for
**both geometries** (12 models). Outputs are named
`hyperbolic_k<K>_seed<S>/` and `euclidean_k<K>_seed<S>/`.

Both sensitivity scripts use the existing base JSON configurations: **800**
epochs, batch size **4096**, 128/384 samples, learning rate **0.002**, seed **42**,
patience **60**, and delta **1e-5**. They **record the early-stop point and keep
training to the epoch limit**, unlike the main experiment's actual early stop.
Hyperbolic training retains RiemannianAdam; Euclidean training retains AdamW.
The loss weights and margins are unchanged.

Use `--config FILE` to supply a base configuration, `--seeds 42,43` for multiple
training seeds, or `--epochs`, `--batch-size`, `--micro-batch-size`, and
`--num-samples-per-profile` for explicit execution overrides. Dataset/output
paths, swept parameters, selected geometry, and seeds are set by the launcher.
Use `--help` on each script for all options. The saved `train_config.json`
contains the final effective settings, rather than a temporary source config.

### Sensitivity outputs and interpretation

Each model directory contains `train_config.json`, `training.log`,
`run_status.json`, `train_manifest.json`, `train_metrics.json`,
`best_model.pth`, and `final_model.pth` on success. The experiment root contains:

- `summary.json` and `summary.csv`: parameter values, status, best loss,
  training time, early-stop mark, peak allocated/reserved CUDA memory, and paths.
- `metrics.png` and `metrics.pdf`: loss, training time, early-stop mark time,
  and peak reserved CUDA memory versus the scanned parameter.
- `preparation.log`: snapshot preparation/build log when automatic preparation
  was needed.

CUDA memory fields on CPU runs and unreached early-stop marks are JSON `null`, empty CSV
cells, and “Not available” plot panels when no measurements exist. These values
are not reported as zero. Multi-seed plots show each seed separately. Failed
runs remain in the summary and do not contribute stale model metrics to plots.
A rerun of the same output location retrains the selected models and replaces
its current summary; logs append. Choose another `--output-dir` to retain an
independent experiment. No temporary sweep configurations are written into the
source `configs/` directory.

Loss is the training objective, not a labeled detection metric. The figures
summarize the measured training behavior and costs; they do not by themselves
reproduce every paper figure or establish detection accuracy.

## Optional AS Rank semantic analysis

The existing embedding-dispersion analysis remains available separately. Provide
AS Rank metadata for the same month (for example, a prepared historical
`as_info.jsonl`), since the RIB-only sensitivity preparation does not require
external AS metadata for training.

```bash
python experiments/scripts/analyze_embedding_dispersion.py \
  --data-dir dataset/ae_snapshot_20260101 \
  --runs-dir experiments/results/k \
  --as-info /path/to/as_info.2601.jsonl \
  --output-dir experiments/results/k/semantic_analysis
```

The command reads model manifests rather than assuming old model-directory
names. It emits the existing AS-pair, nearest-AS, and prefix-layer distance
boxplots and summary CSVs. `--quick-test` reduces its analysis sample counts.
Select one K sweep with a fixed dimension and seed per invocation. Use
`--model-seed 42 --embed-dim 24` to select from multiple settings; the analyzer
rejects mixed settings rather than pooling them.