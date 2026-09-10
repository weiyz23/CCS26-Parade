#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$DIR"
PROJECT_ROOT="$(realpath "$DIR/..")"
PY_ENV_BIN="$PROJECT_ROOT/.conda/bin"

if [[ -d "$PY_ENV_BIN" ]]; then
    export PATH="$PY_ENV_BIN:$PATH"
fi

usage() {
    cat <<EOF
Usage:
    $0 [--raw-data-base PATH] [start|stop|status|restart|run-once|download-once|full-once] [DATASET_NAME] [TIME]

Arguments:
    DATASET_NAME  Dataset folder name under ../dataset (default: test)
    TIME          For start/restart/download-once: optional replay window (YYYY-mm-dd HH:MM:SS,YYYY-mm-dd HH:MM:SS)
                  For run-once: required/optional time window (YYYY-mm-dd HH:MM:SS,YYYY-mm-dd HH:MM:SS)
                  For full-once: required/optional replay window (YYYY-mm-dd HH:MM:SS,YYYY-mm-dd HH:MM:SS)

Options:
    -B, --raw-data-base PATH
                  Raw data root directory (default: /data/parade)

Defaults:
        Dataset output dir: ../dataset/<DATASET_NAME>
    Raw data dir:       /data/parade/<DATASET_NAME>

Examples:
  $0
    $0 --raw-data-base /data/parade
  $0 start safehost
    $0 --raw-data-base /mnt/bgp_raw start safehost
    $0 start safehost "2019-06-06 08:00:00,2019-06-06 13:00:00"
        $0 download-once safehost "2019-06-06 08:00:00,2019-06-06 13:00:00"
    $0 run-once safehost "2019-06-06 08:00:00,2019-06-06 13:00:00"
  $0 status safehost
  $0 stop safehost
EOF
}

ACTION="start"
if [[ "${1:-}" == "__log_maintainer" ]]; then
    LOG_DIR_ARG="${2:-}"
    LOG_MAX_BYTES_ARG="${3:-104857600}"
    LOG_INTERVAL_ARG="${4:-5}"
    if [[ -z "$LOG_DIR_ARG" ]]; then
        exit 1
    fi
    while true; do
        for log_file in "$LOG_DIR_ARG"/*.log; do
            [[ -f "$log_file" ]] || continue
            size=$(stat -c%s "$log_file" 2>/dev/null || echo 0)
            if (( size > LOG_MAX_BYTES_ARG )); then
                tmp_file="${log_file}.truncate.tmp"
                tail -c "$LOG_MAX_BYTES_ARG" "$log_file" > "$tmp_file" 2>/dev/null || true
                cat "$tmp_file" > "$log_file" 2>/dev/null || true
                rm -f "$tmp_file"
            fi
        done
        sleep "$LOG_INTERVAL_ARG"
    done
fi

if [[ $# -gt 0 ]]; then
    RAW_DATA_BASE_INPUT=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help|help)
                usage
                exit 0
                ;;
            -B|--raw-data-base)
                if [[ $# -lt 2 ]]; then
                    echo "[ERROR] Missing value for $1"
                    usage
                    exit 1
                fi
                RAW_DATA_BASE_INPUT="$2"
                shift 2
                ;;
            start|stop|status|restart|run-once|download-once|full-once)
                ACTION="$1"
                shift
                break
                ;;
            *)
                break
                ;;
        esac
    done
fi

DATASET_INPUT="${1:-test}"
START_TIME_INPUT="${2:-}"
NOW_UTC="$(date -u '+%Y-%m-%d %H:%M:%S')"

floor_to_8h_utc() {
    local epoch="$1"
    local interval=$((8 * 3600))
    local aligned=$(( (epoch / interval) * interval ))
    date -u -d "@$aligned" '+%Y-%m-%d %H:%M:%S'
}

declare -A DATASET_START_HINTS
declare -A DATASET_TIME_WINDOWS
DATASET_START_HINTS["vodafone"]="2021-04-16 12:00:00"
DATASET_START_HINTS["safehost"]="2019-06-06 08:00:00"
DATASET_START_HINTS["klayswap"]="2022-02-03 00:00:00"
DATASET_START_HINTS["celer-bridge"]="2022-08-17 18:00:00"
DATASET_START_HINTS["tic"]="2023-03-12 12:00:00"
DATASET_START_HINTS["cantv"]="2026-01-02 14:00:00"
DATASET_START_HINTS["rostelecom"]="2017-04-26 20:00:00"
DATASET_START_HINTS["cloudflare"]="2026-01-22 20:00:00"
DATASET_START_HINTS["voxility"]="2025-04-01 06:00:00"
DATASET_START_HINTS["rostelecom-2"]="2020-04-01 18:00:00"
DATASET_TIME_WINDOWS["vodafone"]="2021-04-16 12:00:00,2021-04-16 15:00:00"
DATASET_TIME_WINDOWS["safehost"]="2019-06-06 08:00:00,2019-06-06 13:00:00"
DATASET_TIME_WINDOWS["klayswap"]="2022-02-03 00:00:00,2022-02-03 05:00:00"
DATASET_TIME_WINDOWS["celer-bridge"]="2022-08-17 18:00:00,2022-08-17 22:00:00"
DATASET_TIME_WINDOWS["tic"]="2023-03-12 12:00:00,2023-03-12 16:00:00"
DATASET_TIME_WINDOWS["cantv"]="2026-01-02 14:00:00,2026-01-02 18:00:00"
DATASET_TIME_WINDOWS["rostelecom"]="2017-04-26 20:00:00,2017-04-26 23:59:00"
DATASET_TIME_WINDOWS["cloudflare"]="2026-01-22 20:00:00,2026-01-22 23:59:00"
DATASET_TIME_WINDOWS["voxility"]="2025-04-01 06:00:00,2025-04-01 10:00:00"
DATASET_TIME_WINDOWS["rostelecom-2"]="2020-04-01 18:00:00,2020-04-01 21:00:00"

DATASET_DIR="$(realpath "../dataset/$DATASET_INPUT")"
RAW_DATA_BASE="${RAW_DATA_BASE_INPUT:-${RAW_DATA_BASE:-/data/parade}}"
RAW_DATA_DIR="$RAW_DATA_BASE/$DATASET_INPUT"
# Historical validation consumes rib_records_cache.bin directly for GraphDist
# and never reads reference_paths.bin.  Avoid materializing that multi-GiB
# supplemental simulation artifact in the sequential staging workspace.
if [[ "$DATASET_DIR" == /data/parade_revision_history/* ]]; then
    export PARADE_SKIP_REFERENCE_PATHS=1
fi
RIB_ARCHIVE_CACHE_ROOT="${RIB_ARCHIVE_CACHE_ROOT:-}"
RIB_RAW_BASE="${RIB_RAW_BASE:-}"
PROCESSED_DIR="$DATASET_DIR/processed_data"
PARSER_DIR="$DIR/parser"
LOG_DIR="${PARADE_LOG_DIR:-$PROJECT_ROOT/log}"
PID_DIR="$PROJECT_ROOT/.pids"
LOG_MAINTAINER_PID_FILE="$PID_DIR/log_maintainer.pid"

LOG_MAX_BYTES=$((100 * 1024 * 1024))
LOG_CHECK_INTERVAL=5

DOWNLOADER_LOG="$LOG_DIR/bgp_downloader.log"
PROCESSOR_LOG="$LOG_DIR/processor.log"
DOWNLOADER_PID_FILE="$PID_DIR/bgp_downloader.pid"
PROCESSOR_PID_FILE="$PID_DIR/processor.pid"

# Priority:
# 1) explicit TIME window argument (<start>,<end>)
# 2) dataset-specific predefined historical window
# 3) fail fast if neither is available
PIPELINE_START_TIME=""
PIPELINE_END_TIME=""
DOWNLOADER_START_TIME=""
DOWNLOADER_END_TIME=""
if [[ "$ACTION" == "start" || "$ACTION" == "restart" || "$ACTION" == "download-once" ]]; then
    PIPELINE_TIME_WINDOW=""
    if [[ -n "$START_TIME_INPUT" ]]; then
        PIPELINE_TIME_WINDOW="$START_TIME_INPUT"
    elif [[ -n "${DATASET_TIME_WINDOWS[$DATASET_INPUT]:-}" ]]; then
        PIPELINE_TIME_WINDOW="${DATASET_TIME_WINDOWS[$DATASET_INPUT]}"
    fi

    if [[ -z "$PIPELINE_TIME_WINDOW" ]]; then
        echo "[ERROR] start/restart requires replay window: <start>,<end>"
        echo "[ERROR] No predefined window found for dataset: $DATASET_INPUT"
        exit 1
    fi

    IFS=',' read -r _raw_start _raw_end <<< "$PIPELINE_TIME_WINDOW"
    PIPELINE_START_TIME="$(echo "${_raw_start:-}" | xargs)"
    PIPELINE_END_TIME="$(echo "${_raw_end:-}" | xargs)"

    if [[ -z "$PIPELINE_START_TIME" || -z "$PIPELINE_END_TIME" ]]; then
        echo "[ERROR] Invalid replay window format: $PIPELINE_TIME_WINDOW"
        echo "[ERROR] Expected: YYYY-mm-dd HH:MM:SS,YYYY-mm-dd HH:MM:SS"
        exit 1
    fi

    PIPELINE_START_EPOCH="$(date -u -d "$PIPELINE_START_TIME" '+%s')"
    PIPELINE_END_EPOCH="$(date -u -d "$PIPELINE_END_TIME" '+%s')"
    if (( PIPELINE_END_EPOCH <= PIPELINE_START_EPOCH )); then
        echo "[ERROR] Invalid replay window: end must be later than start"
        echo "[ERROR] start=$PIPELINE_START_TIME end=$PIPELINE_END_TIME"
        exit 1
    fi

    # Bootstrap downloader with 3-day UPD history coverage.
    DOWNLOADER_START_TIME="$(date -u -d "@$((PIPELINE_START_EPOCH - 259200))" '+%Y-%m-%d %H:%M:%S')"
    DOWNLOADER_END_TIME="$PIPELINE_END_TIME"
fi

get_pipeline_time_window() {
    local time_window="${1:-}"
    if [[ -z "$time_window" && -n "${DATASET_TIME_WINDOWS[$DATASET_INPUT]:-}" ]]; then
        time_window="${DATASET_TIME_WINDOWS[$DATASET_INPUT]}"
        echo "[INFO] Using predefined replay window for $DATASET_INPUT: $time_window" >&2
    fi
    if [[ -z "$time_window" ]]; then
        echo "[ERROR] This action requires replay window: <start>,<end>"
        exit 1
    fi
    echo "$time_window"
}

is_running() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null || true)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_python_daemon() {
    local name="$1"
    local pid_file="$2"
    local log_file="$3"
    shift 3

    # Recover from stale root-owned pid files left by accidental sudo runs.
    if [[ -f "$pid_file" && ! -w "$pid_file" ]]; then
        rm -f "$pid_file" 2>/dev/null || true
    fi

    if is_running "$pid_file"; then
        echo "[INFO] $name already running (pid: $(cat "$pid_file"))"
        return
    fi

    echo "[INFO] Starting $name ..."
    nohup "$@" >> "$log_file" 2>&1 &
    echo $! > "$pid_file"
    sleep 1

    if is_running "$pid_file"; then
        echo "[SUCCESS] $name started (pid: $(cat "$pid_file"))"
    else
        if [[ "$name" == "processor" ]]; then
            echo "[WARN] processor exited quickly; continuing."
            echo "[WARN] This may happen when there is no new raw data to process yet. See log: $log_file"
            rm -f "$pid_file"
            return
        fi
        echo "[ERROR] Failed to start $name. See log: $log_file"
        exit 1
    fi
}

start_log_maintainer() {
    if [[ -f "$LOG_MAINTAINER_PID_FILE" && ! -w "$LOG_MAINTAINER_PID_FILE" ]]; then
        rm -f "$LOG_MAINTAINER_PID_FILE" 2>/dev/null || true
    fi
    if is_running "$LOG_MAINTAINER_PID_FILE"; then
        echo "[INFO] log_maintainer already running (pid: $(cat "$LOG_MAINTAINER_PID_FILE"))"
        return
    fi
    echo "[INFO] Starting log_maintainer ..."
    nohup "$SCRIPT_PATH" __log_maintainer "$LOG_DIR" "$LOG_MAX_BYTES" "$LOG_CHECK_INTERVAL" \
        >> "$LOG_DIR/log_maintainer.log" 2>&1 &
    echo $! > "$LOG_MAINTAINER_PID_FILE"
    sleep 1
    if is_running "$LOG_MAINTAINER_PID_FILE"; then
        echo "[SUCCESS] log_maintainer started (pid: $(cat "$LOG_MAINTAINER_PID_FILE"))"
    else
        echo "[ERROR] Failed to start log_maintainer. See log: $LOG_DIR/log_maintainer.log"
        exit 1
    fi
}

stop_one() {
    local pid_file="$1"
    local name="$2"
    if is_running "$pid_file"; then
        local pid
        pid=$(cat "$pid_file")
        echo "[INFO] Stopping $name (pid: $pid)"
        kill "$pid" 2>/dev/null || true
        for _ in {1..10}; do
            if kill -0 "$pid" 2>/dev/null; then
                sleep 1
            else
                break
            fi
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "[WARN] Force killing $name (pid: $pid)"
            kill -9 "$pid" 2>/dev/null || true
        fi
    else
        echo "[INFO] $name not running"
    fi
    rm -f "$pid_file"
}

status_one() {
    local pid_file="$1"
    local name="$2"
    if is_running "$pid_file"; then
        echo "[RUNNING] $name pid=$(cat "$pid_file")"
    else
        echo "[STOPPED] $name"
    fi
}

start_all() {
    mkdir -p "$DATASET_DIR" "$PROCESSED_DIR" "$LOG_DIR" "$PID_DIR"

    start_log_maintainer

    if ! mkdir -p "$RAW_DATA_DIR"; then
        echo "[ERROR] Failed to create raw data directory: $RAW_DATA_DIR"
        echo "[ERROR] Current RAW_DATA_BASE is: $RAW_DATA_BASE"
        echo "[ERROR] Please ensure write permission, e.g.: sudo mkdir -p $RAW_DATA_BASE && sudo chown -R $(id -un):$(id -gn) $RAW_DATA_BASE"
        exit 1
    fi

    if [[ ! -w "$RAW_DATA_DIR" ]]; then
        echo "[ERROR] Raw data directory is not writable: $RAW_DATA_DIR"
        echo "[ERROR] Ownership now: $(stat -c '%U:%G %a' "$RAW_DATA_DIR" 2>/dev/null || echo 'unknown')"
        echo "[ERROR] Please fix permission, e.g.: sudo chown -R $(id -un):$(id -gn) $RAW_DATA_BASE"
        exit 1
    fi

    if [[ ! -w "$DATASET_DIR" ]]; then
        echo "[ERROR] Dataset directory is not writable: $DATASET_DIR"
        echo "[ERROR] Ownership now: $(stat -c '%U:%G %a' "$DATASET_DIR" 2>/dev/null || echo 'unknown')"
        exit 1
    fi

    if [[ ! -x "$PARSER_DIR/bin/processor" || ! -x "$PARSER_DIR/bin/data_processor" || ! -x "$PARSER_DIR/bin/monitor" ]]; then
        echo "[INFO] Compiling parser binaries..."
        bash "$PARSER_DIR/build.sh"
    fi

    start_python_daemon "bgp_downloader" "$DOWNLOADER_PID_FILE" "$DOWNLOADER_LOG" \
        env BGP_DOWNLOADER_LOG_FILE="$DOWNLOADER_LOG" \
        python -u "$DIR/bgp_downloader.py" --dataset "$DATASET_DIR" --raw-data "$RAW_DATA_DIR" --start-time "$DOWNLOADER_START_TIME" --end-time "$DOWNLOADER_END_TIME"

    start_python_daemon "processor" "$PROCESSOR_PID_FILE" "$PROCESSOR_LOG" \
        env PROCESSOR_LOG_FILE="$PROCESSOR_LOG" \
        stdbuf -oL -eL "$PARSER_DIR/bin/processor" "$RAW_DATA_DIR" "$DATASET_DIR"

    if [[ -n "$PIPELINE_START_TIME" ]]; then
        env RAW_DATA_BASE="$RAW_DATA_BASE" bash "$PARSER_DIR/run_monitor.sh" start -D "$DATASET_DIR" -T "$PIPELINE_START_TIME"
    else
        env RAW_DATA_BASE="$RAW_DATA_BASE" bash "$PARSER_DIR/run_monitor.sh" start -D "$DATASET_DIR"
    fi

    # NOTE: dataset building (build_dataset.py) is now handled inline by the
    # online orchestrator (hyperbolic_model/run_online.sh).  No separate daemon
    # is needed here.

    echo "[SUCCESS] Pipeline daemons started for dataset: $DATASET_INPUT"
    echo "[INFO] Replay start time (UTC): $PIPELINE_START_TIME"
    echo "[INFO] Replay end time (UTC): $PIPELINE_END_TIME"
    echo "[INFO] Downloader bootstrap start time (UTC): $DOWNLOADER_START_TIME"
    echo "[INFO] Downloader end time (UTC): $DOWNLOADER_END_TIME"
    echo "[INFO] Raw data dir: $RAW_DATA_DIR"
    echo "[INFO] Dataset dir: $DATASET_DIR"
    echo "[INFO] Logs: $LOG_DIR"
}

stop_all() {
    env RAW_DATA_BASE="$RAW_DATA_BASE" bash "$PARSER_DIR/run_monitor.sh" stop -D "$DATASET_DIR" || true
    stop_one "$PROCESSOR_PID_FILE" "processor"
    stop_one "$DOWNLOADER_PID_FILE" "bgp_downloader"
    stop_one "$LOG_MAINTAINER_PID_FILE" "log_maintainer"
    echo "[SUCCESS] Pipeline daemons stopped for dataset: $DATASET_INPUT"
}

run_once() {
    local time_window="$START_TIME_INPUT"
    if [[ -z "$time_window" && -n "${DATASET_TIME_WINDOWS[$DATASET_INPUT]:-}" ]]; then
        time_window="${DATASET_TIME_WINDOWS[$DATASET_INPUT]}"
        echo "[INFO] Using predefined one-shot window for $DATASET_INPUT: $time_window"
    fi
    if [[ -z "$time_window" ]]; then
        echo "[ERROR] run-once requires a time window argument: <start>,<end>"
        exit 1
    fi

    mkdir -p "$DATASET_DIR" "$PROCESSED_DIR" "$LOG_DIR" "$PID_DIR"

    local raw_base="$RAW_DATA_BASE"
    local raw_dir="$RAW_DATA_DIR"
    if [[ ! -d "$raw_dir" ]]; then
        mkdir -p "$raw_dir" 2>/dev/null || true
    fi
    if [[ ! -d "$raw_dir" ]]; then
        echo "[ERROR] Raw data directory does not exist: $RAW_DATA_DIR"
        echo "[ERROR] Failed to create it under RAW_DATA_BASE: $RAW_DATA_BASE"
        echo "[ERROR] Please ensure write permission, e.g.: sudo mkdir -p $RAW_DATA_BASE && sudo chown -R $(id -un):$(id -gn) $RAW_DATA_BASE"
        exit 1
    fi

    local once_monitor_log="$LOG_DIR/monitor.once.log"
    local once_parser_log="$LOG_DIR/data_parser.once.log"
    echo "[INFO] Running one-shot monitor stack for dataset: $DATASET_INPUT"
    echo "[INFO] One-shot time window (UTC): $time_window"
    env RAW_DATA_BASE="$raw_base" RAW_DATA_DIR_OVERRIDE="$raw_dir" \
        bash "$PARSER_DIR/run_monitor.sh" run-once -D "$DATASET_DIR" -W "$time_window" \
        -l "$once_monitor_log" -n "$once_parser_log"
    echo "[SUCCESS] One-shot window run completed for dataset: $DATASET_INPUT"
}

download_once() {
    if [[ -z "$PIPELINE_START_TIME" || -z "$PIPELINE_END_TIME" ]]; then
        echo "[ERROR] download-once requires replay window: <start>,<end>"
        exit 1
    fi

    mkdir -p "$DATASET_DIR" "$PROCESSED_DIR" "$LOG_DIR" "$PID_DIR"

    local raw_dir="$RAW_DATA_DIR"
    if [[ ! -d "$raw_dir" ]]; then
        mkdir -p "$raw_dir" 2>/dev/null || true
    fi
    if [[ ! -d "$raw_dir" ]]; then
        echo "[ERROR] Raw data directory does not exist: $RAW_DATA_DIR"
        echo "[ERROR] Failed to create it under RAW_DATA_BASE: $RAW_DATA_BASE"
        echo "[ERROR] Please ensure write permission, e.g.: sudo mkdir -p $RAW_DATA_BASE && sudo chown -R $(id -un):$(id -gn) $RAW_DATA_BASE"
        exit 1
    fi

    local downloader_extra_args=()
    if [[ -n "$RIB_ARCHIVE_CACHE_ROOT" || -n "$RIB_RAW_BASE" ]]; then
        if [[ -z "$RIB_ARCHIVE_CACHE_ROOT" || -z "$RIB_RAW_BASE" ]]; then
            echo "[ERROR] RIB_ARCHIVE_CACHE_ROOT and RIB_RAW_BASE must be set together"
            exit 1
        fi
        if [[ ! -d "$RIB_ARCHIVE_CACHE_ROOT" ]]; then
            echo "[ERROR] RIB archive cache root does not exist: $RIB_ARCHIVE_CACHE_ROOT"
            exit 1
        fi

        local external_rib_dir="$RIB_RAW_BASE/$DATASET_INPUT/rib"
        local raw_rib_dir="$RAW_DATA_DIR/rib"
        mkdir -p "$external_rib_dir"

        if [[ -L "$raw_rib_dir" ]]; then
            local current_target
            current_target="$(readlink "$raw_rib_dir")"
            if [[ "$current_target" != "$external_rib_dir" ]]; then
                rm -f "$raw_rib_dir"
            fi
        elif [[ -e "$raw_rib_dir" ]]; then
            if [[ -d "$raw_rib_dir" && -z "$(find "$raw_rib_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
                rmdir "$raw_rib_dir"
            else
                echo "[ERROR] Cannot enable external RIB raw dir because a non-empty path exists: $raw_rib_dir"
                echo "[ERROR] Move or remove it first, then rerun. Target external RIB dir: $external_rib_dir"
                exit 1
            fi
        fi

        if [[ ! -e "$raw_rib_dir" ]]; then
            ln -s "$external_rib_dir" "$raw_rib_dir"
        fi

        downloader_extra_args+=(--rib-cache-root "$RIB_ARCHIVE_CACHE_ROOT" --no-rib-network-fallback)
        echo "[INFO] RIB archive cache root: $RIB_ARCHIVE_CACHE_ROOT"
        echo "[INFO] External raw RIB dir: $external_rib_dir"
        echo "[INFO] Raw RIB symlink: $raw_rib_dir -> $(readlink "$raw_rib_dir")"
    fi

    echo "[INFO] Running one-shot replay downloader for dataset: $DATASET_INPUT"
    echo "[INFO] Replay window start (UTC): $PIPELINE_START_TIME"
    echo "[INFO] Replay window end (UTC): $PIPELINE_END_TIME"
    echo "[INFO] Downloader bootstrap start (UTC): $DOWNLOADER_START_TIME"

    python -u "$DIR/download_replay_data.py" \
        --dataset "$DATASET_DIR" \
        --raw-data "$RAW_DATA_DIR" \
        --start-time "$DOWNLOADER_START_TIME" \
        --end-time "$PIPELINE_END_TIME" \
        "${downloader_extra_args[@]}"

    echo "[SUCCESS] One-shot replay download completed for dataset: $DATASET_INPUT"
}

build_rib_cache_once() {
    local dataset_root="$1"

    processor_outputs_ready() {
        local root="$1"
        local required
        for required in \
            "$root/rib_records_cache.bin" \
            "$root/prefix_map.txt" \
            "$root/as_map.txt" \
            "$root/prefix_to_profile.bin" \
            "$root/profile_triplets.bin" \
            "$root/profile_fingerprints.bin"; do
            [[ -s "$required" ]] || return 1
        done

        python3 - "$root/profile_triplets.bin" <<'PY'
import os
import struct
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    raw_count = f.read(8)
    if len(raw_count) != 8:
        sys.exit(1)
    count = struct.unpack("Q", raw_count)[0]
    remaining = os.fstat(f.fileno()).st_size - 8

sys.exit(0 if remaining == count * 13 else 1)
PY
    }

    if processor_outputs_ready "$dataset_root"; then
        echo "[INFO] Existing weighted processor outputs found, skip processor bootstrap: $dataset_root"
        return 0
    fi

    echo "[INFO] Weighted processor outputs missing or incomplete; rebuilding from existing raw files."
    rm -f \
        "$dataset_root/rib_records_cache.bin" \
        "$dataset_root/prefix_map.txt" \
        "$dataset_root/as_map.txt" \
        "$dataset_root/prefix_to_profile.bin" \
        "$dataset_root/profile_triplets.bin" \
        "$dataset_root/profile_fingerprints.bin" \
        "$dataset_root/reference_paths.bin" \
        "$dataset_root/prefix_map.pt" \
        "$dataset_root/as_map.pt" \
        "$dataset_root/profile_triplets.pt" \
        "$dataset_root/prefix_to_profile.pt" \
        "$dataset_root/profile_fingerprints.pt" \
        "$dataset_root/fingerprint_anchor_db.pt" \
        "$dataset_root/history_cache.pt" \
        "$dataset_root/meta.json"

    echo "[INFO] Building baseline cache via processor (one-shot bootstrap)"
    if ! stdbuf -oL -eL "$PARSER_DIR/bin/processor" "$RAW_DATA_DIR" "$dataset_root" >> "$PROCESSOR_LOG" 2>&1; then
        echo "[ERROR] processor failed while building weighted training outputs"
        tail -n 80 "$PROCESSOR_LOG" 2>/dev/null || true
        return 1
    fi

    if ! processor_outputs_ready "$dataset_root"; then
        echo "[ERROR] processor finished but weighted training outputs are missing or invalid under: $dataset_root"
        tail -n 80 "$PROCESSOR_LOG" 2>/dev/null || true
        return 1
    fi

    echo "[INFO] weighted processor outputs generated under: $dataset_root"
    return 0
}

cleanup_external_rib_raw_once() {
    if [[ -z "$RIB_RAW_BASE" ]]; then
        return 0
    fi

    local external_rib_dir="$RIB_RAW_BASE/$DATASET_INPUT/rib"
    if [[ ! -d "$external_rib_dir" ]]; then
        return 0
    fi

    echo "[INFO] Removing decompressed external RIB raw files: $external_rib_dir"
    find "$external_rib_dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}

full_once() {
    local time_window
    time_window="$(get_pipeline_time_window "$START_TIME_INPUT")"

    local dataset_root="$DATASET_DIR"
    local anomaly_csv="$dataset_root/processed_data/anomaly/anomaly_links.csv"
    local download_attempts=3
    local download_delay_seconds=10

    mkdir -p "$dataset_root" "$PROCESSED_DIR" "$LOG_DIR" "$PID_DIR"

    echo "[INFO] full-once pipeline start: dataset=$DATASET_INPUT window=$time_window"

    # Ensure no daemonized process from previous runs interferes with one-shot flow.
    stop_all || true

    local n=1
    while (( n <= download_attempts )); do
        echo "[INFO] download-once attempt $n/$download_attempts"
        if "$SCRIPT_PATH" --raw-data-base "$RAW_DATA_BASE" download-once "$DATASET_INPUT" "$time_window"; then
            break
        fi

        if (( n == download_attempts )); then
            echo "[ERROR] download-once failed after $download_attempts attempts"
            exit 1
        fi

        echo "[WARN] download-once failed, retrying in ${download_delay_seconds}s"
        sleep "$download_delay_seconds"
        ((n++))
    done

    build_rib_cache_once "$dataset_root"
    cleanup_external_rib_raw_once

    local monitor_attempt
    for monitor_attempt in 1 2; do
        echo "[INFO] run-once attempt $monitor_attempt/2"
        "$SCRIPT_PATH" --raw-data-base "$RAW_DATA_BASE" run-once "$DATASET_INPUT" "$time_window"

        if [[ -s "$anomaly_csv" ]]; then
            echo "[INFO] anomaly csv generated: $anomaly_csv"
            break
        fi

        if (( monitor_attempt == 2 )); then
            echo "[ERROR] anomaly csv is empty after retries: $anomaly_csv"
            exit 1
        fi

        echo "[WARN] anomaly csv empty, retrying run-once"
        rm -f "$anomaly_csv"
    done

    echo "[INFO] Building dataset tensors"
    python "$DIR/build_dataset.py" --data_dir "$dataset_root"

    echo "[INFO] Running hyperbolic pipeline"
    bash "$PROJECT_ROOT/hyperbolic_model/run.sh" "$dataset_root"

    echo "[SUCCESS] full-once pipeline completed for dataset: $DATASET_INPUT"
}

status_all() {
    status_one "$DOWNLOADER_PID_FILE" "bgp_downloader"
    status_one "$PROCESSOR_PID_FILE" "processor"
    env RAW_DATA_BASE="$RAW_DATA_BASE" bash "$PARSER_DIR/run_monitor.sh" status -D "$DATASET_DIR" || true
    status_one "$LOG_MAINTAINER_PID_FILE" "log_maintainer"
}

case "$ACTION" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        start_all
        ;;
    status)
        status_all
        ;;
    run-once)
        run_once
        ;;
    download-once)
        download_once
        ;;
    full-once)
        full_once
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
