# PARADE: BGP Anomaly Detection with Hyperbolic Embeddings

This repository implements the core pipeline for **PARADE: Unified and
Data-Driven Prefix-AS Semantics for BGP Anomaly Detection**: build propagation
profiles, learn AS/profile embeddings in a Poincaré ball, extract suspicious
routing changes, and score their semantic consistency.

The available workflow is offline training and historical replay. Raw BGP data,
generated datasets, and trained checkpoints are not included. Differences from
the paper are described under [Implementation scope](#implementation-scope).

## Quick start with the run scripts

For Artifact Evaluation, see [the experiment guide](experiments/README.md).
It provides separate launchers for the eight historical events, embedding
dimension sensitivity, and K/geometry comparison, including automatic snapshot
preparation and metric summaries.

After [environment setup](#environment), use the two main entry points:

| Script | Usage |
| --- | --- |
| `preprocess/run.sh` | `bash preprocess/run.sh [--raw-data-base PATH] ACTION DATASET_NAME [TIME_WINDOW]` |
| `hyperbolic_model/run.sh` | `bash hyperbolic_model/run.sh DATASET_DIR` |

Run from the repository root. This example uses the predefined KLAYswap window,
`2022-02-03 00:00:00,2022-02-03 05:00:00` (UTC):

```bash
PARADE_ROOT="$PWD"
RAW_DATA_BASE="$PARADE_ROOT/data/raw"
DATA_DIR="$PARADE_ROOT/dataset/klayswap"
mkdir -p "$DATA_DIR" "$RAW_DATA_BASE"

# Download, preprocess, replay, train, calibrate, and detect.
bash preprocess/run.sh --raw-data-base "$RAW_DATA_BASE" full-once klayswap
```

`full-once` calls `hyperbolic_model/run.sh` automatically after preparing the
data. If the dataset tensors and candidate CSV already exist, run the modeling
script directly:

```bash
bash hyperbolic_model/run.sh "$DATA_DIR"
```

The modeling script trains embeddings, calibrates thresholds, and writes
detection results under `$DATA_DIR/results`. Training settings can be adjusted
with environment variables; for example:

```bash
TRAIN_EPOCHS=100 EMBEDDING_DIM=24 NUM_VECTORS=2 \
  bash hyperbolic_model/run.sh "$DATA_DIR"
```

Other preprocessing actions are `download-once` (download only), `run-once`
(replay existing data), and `start|stop|status|restart` for the background
download/parser/monitor processes. Omitting the action defaults to `start`;
that mode does not run tensor construction or model training.
Use `bash preprocess/run.sh --help` for all arguments.

For a new dataset, create `dataset/<name>` and pass an explicit UTC window:

```bash
mkdir -p "$PARADE_ROOT/dataset/my_dataset"
bash preprocess/run.sh --raw-data-base "$RAW_DATA_BASE" full-once my_dataset \
  "2022-02-03 00:00:00,2022-02-03 05:00:00"
```

Downloads cover the configured collectors and can require substantial disk
space and time. The [step-by-step workflow](#step-by-step-workflow) below explains
the individual stages.

## Environment

- Linux with Bash, a C++17 compiler, CMake, Make, and POSIX shared memory.
- Python 3.12 is the tested Python version.
- `libbgpstream`, including headers and the shared library. Follow the
  [official installation guide](https://bgpstream.caida.org/docs/install/bgpstream).
- An NVIDIA GPU with a compatible CUDA build of PyTorch is recommended for full
  datasets. Training falls back to CPU when CUDA is unavailable; use smaller
  batches for CPU runs. The local GPU smoke check used an RTX 4090, PyTorch
  `2.11.0+cu130`, and Geoopt `0.5.1`. Memory use depends on data and batch size.

Run setup commands from the repository root:

```bash
conda create --prefix ./.conda python=3.12
conda activate ./.conda

# Example for CUDA 13.0; choose the wheel that matches your GPU driver.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, geoopt, aiohttp; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Use the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
for other CUDA builds. `torchvision` and `torchaudio` are not used.
`requirements.txt` includes both core pipeline and visualization dependencies;
it is not an exact environment lock file.
`preprocess/run.sh` prepends the repository's `.conda/bin` to `PATH` when present.

Build the C++ programs before the first replay:

```bash
cmake -S preprocess/parser -B preprocess/parser/build
cmake --build preprocess/parser/build --parallel 2
mkdir -p preprocess/parser/bin
cp preprocess/parser/build/bin/* preprocess/parser/bin/
```

The shell entry points expect binaries in `preprocess/parser/bin`. For
`libbgpstream` installed outside system search paths, configure the compiler
and runtime library paths for that installation.

## Step-by-step workflow

These commands expand the script workflow into individual stages for debugging
or custom runs. They use the same predefined KLAYswap window as the quick start.

Run from the repository root after setup and compilation:

```bash
PARADE_ROOT="$PWD"
RAW_DATA_BASE="$PARADE_ROOT/data/raw"
DATA_DIR="$PARADE_ROOT/dataset/klayswap"
mkdir -p "$DATA_DIR" "$RAW_DATA_BASE"

# 1. Download raw observations for the replay window and bootstrap.
bash preprocess/run.sh --raw-data-base "$RAW_DATA_BASE" download-once klayswap

# 2. Generate training binaries and the baseline cache.
preprocess/parser/bin/processor "$RAW_DATA_BASE/klayswap" "$DATA_DIR"

# 3. Replay updates and extract candidate tuples.
bash preprocess/run.sh --raw-data-base "$RAW_DATA_BASE" run-once klayswap

# 4. Generate tensors and historical fingerprint anchors.
python preprocess/build_dataset.py --data_dir "$DATA_DIR"

# 5. Train, calibrate thresholds, and detect anomalies.
bash hyperbolic_model/run.sh "$DATA_DIR"
```

Use one replay job at a time: scripts share PID files and shared-memory names.
Use a fresh dataset directory when changing historical inputs, since tensor and
anchor caches are reused by filename. `build_dataset.py` removes its intermediate
mapping/triplet/fingerprint binaries after conversion.

## Training and detection

`hyperbolic_model/run.sh` defaults to dimension 24, two profile vectors, 128
positive samples per profile, three times as many negative samples, and batch
size 4096. It trains for up to 1,000 epochs with early stopping. Environment
overrides include `EMBEDDING_DIM`, `NUM_VECTORS`, `TRAIN_EPOCHS`,
`TRAIN_PATIENCE`, `TRAIN_DELTA`, `TRAIN_LR`, and `TRAIN_SEED`.

For smaller batches, invoke the stages directly, keeping dimension and vector
count identical. These reduced settings check execution, not paper metrics:

```bash
python hyperbolic_model/train.py --data_dir "$DATA_DIR" \
  --model_dir "$DATA_DIR/saved_models" --embed_dim 24 --num_vectors 2 \
  --batch_size 64 --num_samples_per_profile 16 --epochs 10

python hyperbolic_model/calc_thresholds.py --data_dir "$DATA_DIR" \
  --model_dir "$DATA_DIR/saved_models" --embed_dim 24 --num_vectors 2

python hyperbolic_model/detect.py --data_dir "$DATA_DIR" \
  --model_dir "$DATA_DIR/saved_models" --embed_dim 24 --num_vectors 2 \
  --input "$DATA_DIR/processed_data/anomaly/anomaly_links.csv" \
  --anomalous-output "$DATA_DIR/results/anomalous.csv" \
  --missing-output "$DATA_DIR/results/missing.csv"
```

The wrapper appends a repository-local CUDA 13 library directory to
`LD_LIBRARY_PATH`; direct Python commands avoid that environment-specific setting.

## Data and outputs

Training tensors live directly in the dataset root, **not** in `processed_data`:

```text
dataset/<name>/
  meta.json
  prefix_map.pt                 # prefix string -> prefix index
  as_map.pt                     # ASN -> AS index
  prefix_to_profile.pt          # prefix index -> profile index
  profile_triplets.pt           # [profile_id, as_id, hop, weight]
  profile_fingerprints.pt       # packed 128-bit fingerprints
  fingerprint_anchor_db.pt      # historical anchors grouped by AS
  history_cache.pt              # optional organization history
  rib_records_cache.bin         # replay baseline
  processed_data/anomaly/anomaly_links.csv
  saved_models/
    best_model.pth
    final_model.pth
    thresholds.json
  results/
    anomalous_raw.csv
    anomalous_origin.csv
    anomalous_non_origin.csv
    missing_raw.csv
    missing_origin.csv
    missing_non_origin.csv
    detection_stats.csv
```

Candidates contain `timestamp`, `prefix`, `asn`, `level`, `origin`, and
`upstream_asns`; the monitor also emits topology levels, downstream ASes, and
blackhole/leasing flags. The detector adds `dist_pa`, `dist_po`, and `dist_ao`
for profile–AS, profile–origin, and AS–origin distances. More-specific prefixes
use an available historical parent profile. `missing_*` records indicate
insufficient reference embeddings and should be inspected separately.

The output arguments define filename stems; the detector writes the suffixed
CSVs shown above. Thresholds are saved beside the model and, by default, copied
into `meta.json`; `calc_thresholds.py --model-only` keeps them only in the model
directory. Replay logs are under `log/`.

## Optional visualization

Visualization dependencies are installed with `requirements.txt` during setup.

```bash
python hyperbolic_model/visualize.py --data_dir "$DATA_DIR" \
  --model_dir "$DATA_DIR/saved_models" --viz_dir "$DATA_DIR/visualizations" \
  --embed_dim 24 --num_vectors 2
```

`visualize_as_dim_reduction.py` and `visualize_profile_dim_reduction.py` provide
additional projections; consult their `--help` output for options. Visualization
is included in the shell pipeline and the AE main experiment.

## Core components

| Component | Purpose |
| --- | --- |
| `preprocess/data_adapter.py`, `bgp_downloader.py`, `download_replay_data.py` | Download RouteViews and RIPE RIS archives. Collectors are configured in `preprocess/completeness_policy.json`. |
| `preprocess/parser/src/processor.cpp` | Generate training profiles, weighted AS samples, 128-bit fingerprints, and the replay baseline cache. |
| `preprocess/parser/src/data_parser.cpp`, `monitor.cpp` | Replay routing records through a shared-memory ring buffer and extract candidate prefix/AS/origin tuples. |
| `preprocess/build_dataset.py` | Convert binary outputs to tensors and build historical fingerprint anchors. |
| `hyperbolic_model/model.py`, `train.py` | Learn one vector per AS and multiple vectors per profile with hierarchical and co-propagation losses. |
| `hyperbolic_model/calc_thresholds.py`, `detect.py` | Calibrate distance thresholds, score candidates, apply historical fingerprint exemptions, and output results. |

## Implementation scope

- Training profiles currently use AS/minimum-hop signatures. The paper specifies
  identical path sets and SCC-DAG longest-path depths.
- The offline downloader bootstraps three days before replay and keeps the
  earliest RIB per collector. The processor selects UPDATE files using a 14-day
  lookback from the latest downloaded RIB. These windows need alignment with the
  paper before claiming exact reproduction.
- Calibration uses the paper's 99.95th positive and 1.6th negative percentiles.
  Candidate selection and alert aggregation contain different heuristics.
- `filter.py` is an additional legacy utility retained in the shell pipeline.
  It queries current RPKI state and can suppress organization matches, whereas
  the paper retains organization-only matches as annotations. Optional CAIDA
  AS Rank metadata is not needed for training or distance-based detection.
- The daily model-refresh orchestrator is not included. 