# PARADE

BGP anomaly detection with hyperbolic prefix–AS embeddings. This repository
provides preprocessing, offline training and historical replay. Historical launchers
download their inputs automatically; frozen mixed-benchmark inputs are Release attachments.
See [AE experiments](experiments/README.md)
for historical events, sensitivity experiments and the archived mixed benchmark.

## Start here

After [environment setup](#environment), run from the repository root:

```bash
# All eight historical events.
bash experiments/run_main.sh

# KLAYswap only.
bash experiments/run_main.sh --datasets klayswap
```

For the saved mixed benchmark, use the [v1.1.0 bundle download and replay](experiments/README.md#archived-mixed-benchmark).

## Environment

Requires Linux, Python 3.12, CMake, a C++17 compiler and libbgpstream headers/libraries.
For Ubuntu 22.04 and earlier, use the [official installation instructions](https://bgpstream.caida.org/docs/install/bgpstream).
For Ubuntu 24.04, follow the author's
[libbgpstream build guide](https://github.com/weiyz23/libbgpstream-guide-for-Ubuntu24.04):

```bash
git clone --recurse-submodules https://github.com/weiyz23/libbgpstream-guide-for-Ubuntu24.04.git
```

Follow its wandio and libbgpstream build/install steps, then run `sudo ldconfig`.
Return to this repository before the commands below.
Training uses CUDA when available; choose a compatible
[PyTorch build](https://pytorch.org/get-started/locally/).

```bash
conda create --prefix ./.conda python=3.12
conda activate ./.conda
python -m pip install -r requirements.txt

cmake -S preprocess/parser -B preprocess/parser/build
cmake --build preprocess/parser/build --parallel 2
mkdir -p preprocess/parser/bin
cp preprocess/parser/build/bin/* preprocess/parser/bin/
```

Run the following commands from the repository root. The shell entry points
use `.conda/bin` when present, otherwise Python from `PATH`.

## Run

```bash
# Train and detect using an existing dataset.
bash hyperbolic_model/run.sh "$(pwd)/dataset/klayswap"

# Supply a different raw-data location.
bash experiments/run_main.sh --datasets klayswap --raw-data-base /path/to/bgp

# Replay a custom UTC window.
mkdir -p dataset/my_dataset
bash preprocess/run.sh full-once my_dataset \
  "2022-02-03 00:00:00,2022-02-03 05:00:00"
```

Preprocessing also supports `download-once`, `run-once` and
`start|stop|status|restart`; see `bash preprocess/run.sh --help`.
Run one replay job at a time because the pipeline shares PID files and memory.

The model defaults to dimension 24, K=2, batch size 4096, 128 positive and
384 negative samples per profile, and up to 1,000 epochs with early stopping.
Use `EMBEDDING_DIM`, `NUM_VECTORS` and `TRAIN_*` environment variables to override
training settings:

```bash
TRAIN_EPOCHS=100 EMBEDDING_DIM=24 NUM_VECTORS=2 \
  bash hyperbolic_model/run.sh "$(pwd)/dataset/klayswap"
```

Individual stages are `hyperbolic_model/{train,calc_thresholds,detect,filter}.py`;
use their `--help` options for direct invocation.

## Inputs and outputs

Raw RIB/UPDATE files come from [RIPE RIS](https://data.ris.ripe.net/) and
[RouteViews](https://archive.routeviews.org/). The author's complete experiments
accumulated **over 1 TB of raw files and over 200 GB of intermediate caches**;
these are cumulative totals, not a minimum for each event.

For the historical-event launcher:

| Location | Contents |
| --- | --- |
| `data/raw/<event-id>/` | Default raw-data location; override with `--raw-data-base` |
| `dataset/<event-id>/` | Parsed caches, mappings, tensors and candidate paths |
| `dataset/<event-id>/saved_models/` | Checkpoints and calibrated thresholds |
| `dataset/<event-id>/results/`, `visualizations/` | Detection/filter outputs and plots |
| `experiments/results/main/` | Run logs, configuration and summary files |

`run_main.sh --output-dir` changes the log/summary location only; datasets,
caches and models remain under `dataset/`. These directories may be symlinked
to larger storage volumes.

```text
dataset/<name>/
  meta.json
  prefix_map.pt
  as_map.pt
  prefix_to_profile.pt
  profile_triplets.pt             # [profile_id, as_id, hop, weight]
  profile_fingerprints.pt         # packed 128-bit fingerprints
  fingerprint_anchor_db.pt
  history_cache.pt
  rib_records_cache.bin
  processed_data/anomaly/anomaly_links.csv
  saved_models/{best_model.pth,final_model.pth,thresholds.json}
  results/
  visualizations/
```

Training tensors reside in the dataset root. Candidate CSVs contain `timestamp`,
`prefix`, `asn`, `level`, `origin` and `upstream_asns`. Detection adds PA, PO and
AO distances and writes origin/non-origin anomaly and missing-embedding CSVs
under `results/`. Thresholds are saved beside the model; `--model-only` on
`calc_thresholds.py` avoids updating dataset metadata.

Visualization runs after detection. Its status is saved separately in
`visualization_status.json`; plotting failures leave successful detection complete.
For individual plots, use `hyperbolic_model/visualize*.py --help`.

Summarize existing event results without rerunning the pipeline:

```bash
python experiments/scripts/summarize_events.py --datasets klayswap \
  --dataset-root dataset --output-dir experiments/results/historical_summary
```

`historical_metrics.csv` contains descriptive run statistics, not the paper's
Table 1 ground-truth evaluation. See [AE experiments](experiments/README.md).

## Implementation scope

Profiles use AS/minimum-hop signatures. The offline bootstrap is three days,
and UPDATE selection uses a 14-day lookback. Post-filtering applies organization
exemptions and current RPKI validation. The daily model-refresh orchestrator
is not included.
