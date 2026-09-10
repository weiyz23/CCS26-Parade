#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Set LD_LIBRARY_PATH to include nvidia libraries (fix for nvrtc error)
# Note: This path is relative to the project root; adjust if your conda environment differs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DIR/../.conda/lib/python3.12/site-packages/nvidia/cu13/lib/

# Default Dataset Directory
DATASET_DIR="${1:-../dataset/klayswap}"
DATA_DIR="$DATASET_DIR"
MODEL_DIR="$DATASET_DIR/saved_models"
VIZ_DIR="$DATASET_DIR/visualizations"
# Default parameters
EMBEDDING_DIM="${EMBEDDING_DIM:-24}"
NUM_VECTORS="${NUM_VECTORS:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"
TRAIN_PATIENCE="${TRAIN_PATIENCE:-80}"
TRAIN_DELTA="${TRAIN_DELTA:-0.00001}"
TRAIN_LR="${TRAIN_LR:-0.002}"
TRAIN_SEED="${TRAIN_SEED:-42}"

echo "==========================================="
echo "Starting Hyperbolic BGP Embedding Training"
echo "Dataset: $DATASET_DIR"
echo "Data:    $DATA_DIR"
echo "Models:  $MODEL_DIR"
echo "==========================================="

echo "[1/5] Running Training..."
python train.py --batch_size 4096 --num_samples_per_profile 128 \
    --data_dir "$DATA_DIR" --model_dir "$MODEL_DIR" \
    --embed_dim "$EMBEDDING_DIM" --num_vectors "$NUM_VECTORS" \
    --epochs "$TRAIN_EPOCHS" --patience "$TRAIN_PATIENCE" \
    --delta "$TRAIN_DELTA" --lr "$TRAIN_LR" --seed "$TRAIN_SEED"

echo "[2/5] Calculating Thresholds..."
python calc_thresholds.py --data_dir "$DATA_DIR" --model_dir "$MODEL_DIR" --embed_dim $EMBEDDING_DIM --num_vectors $NUM_VECTORS

echo "[3/5] Running Anomaly Detection..."
DETECT_OUTPUT_DIR="$DATASET_DIR/results"
mkdir -p "$DETECT_OUTPUT_DIR"

if [ "${SKIP_PIPELINE_DETECTION:-0}" = "1" ]; then
    echo "Skipping legacy detect.py pass; caller will score candidates directly."
    INPUT_CSV=""
else
    # Try to find anomaly_links.csv
    if [ -f "$DATASET_DIR/processed_data/anomaly/anomaly_links.csv" ]; then
        INPUT_CSV="$DATASET_DIR/processed_data/anomaly/anomaly_links.csv"
    elif [ -f "$DATASET_DIR/processed_data/anomaly_links.csv" ]; then
        INPUT_CSV="$DATASET_DIR/processed_data/anomaly_links.csv"
    elif [ -f "$DATASET_DIR/anomaly_links.csv" ]; then
        INPUT_CSV="$DATASET_DIR/anomaly_links.csv"
    else
        echo "Warning: anomaly_links.csv not found. Skipping detection."
        INPUT_CSV=""
    fi
fi

if [ -n "$INPUT_CSV" ]; then
    python detect.py \
        --input "$INPUT_CSV" \
        --data_dir "$DATA_DIR" \
        --model_dir "$MODEL_DIR" \
        --embed_dim $EMBEDDING_DIM \
        --num_vectors $NUM_VECTORS \
        --anomalous-output "$DETECT_OUTPUT_DIR/anomalous.csv" \
        --missing-output "$DETECT_OUTPUT_DIR/missing.csv"
fi
# Optional Steps Below
echo "[4/5] Running Filter for Anomaly Results..."
python filter.py --data_dir "$DATA_DIR"

echo "[5/5] Running Visualization..."
python visualize.py --data_dir "$DATA_DIR" --model_dir "$MODEL_DIR" --viz_dir "$VIZ_DIR" --embed_dim $EMBEDDING_DIM --num_vectors $NUM_VECTORS
python visualize_as_dim_reduction.py --data_dir "$DATA_DIR" --model_dir "$MODEL_DIR" --viz_dir "$VIZ_DIR" --embed_dim $EMBEDDING_DIM --num_vectors $NUM_VECTORS

echo "==========================================="
echo "Pipeline Completed Successfully!"
echo "Plots:    $VIZ_DIR"
echo "Detect:   $DETECT_OUTPUT_DIR"
echo "==========================================="
