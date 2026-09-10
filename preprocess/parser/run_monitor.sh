#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"

ACTION="start"
if [[ $# -gt 0 ]]; then
    case "$1" in
        start|stop|status|restart|run-once)
            ACTION="$1"
            shift
            ;;
    esac
fi

DEFAULT_DATA_DIR="../dataset/safehost"
DEFAULT_PARSER_THREADS="24"
DEFAULT_ANOMALY_THREADS="8"
DEFAULT_START_TIME=""

DATA_DIR="$DEFAULT_DATA_DIR"
RAW_DATA_DIR=""
RAW_DATA_BASE="${RAW_DATA_BASE:-/data/parade}"
RAW_DATA_DIR_OVERRIDE="${RAW_DATA_DIR_OVERRIDE:-}"
PARSER_THREADS="$DEFAULT_PARSER_THREADS"
ANOMALY_THREADS="$DEFAULT_ANOMALY_THREADS"
START_TIME="$DEFAULT_START_TIME"
TIME_WINDOW=""
MONITOR_LOG="$ROOT_DIR/log/monitor.daemon.log"
DATA_PARSER_LOG="$ROOT_DIR/log/data_parser.daemon.log"
PID_DIR="$ROOT_DIR/.pids"
MONITOR_PID_FILE="$PID_DIR/monitor.pid"
PARSER_PID_FILE="$PID_DIR/data_parser.pid"

usage() {
    cat <<EOF
Usage: $0 [start|stop|status|restart|run-once] [options]

Options:
    -D <dataset_dir>   Dataset directory containing processed_data and metadata
        -R <raw_data_base> Raw data base directory (default: /data/parade)
  -T <start_time>    Optional start time for data_processor, format: YYYY-mm-dd HH:MM[:SS]
    -W <time_window>   One-shot mode window, format: <start>,<end>
  -p <threads>       Monitor parser threads (default: ${DEFAULT_PARSER_THREADS})
  -a <threads>       Monitor anomaly threads (default: ${DEFAULT_ANOMALY_THREADS})
    -l <path>          Monitor stdout/stderr log path (default: $MONITOR_LOG)
    -n <path>          Data parser log path (default: $DATA_PARSER_LOG)
  -h                 Show this help

Examples:
  $0 start -D ../dataset/klayswap
    $0 start -D ../dataset/klayswap -R /data/parade
  $0 start -D ../dataset/klayswap -T "2026-03-19 00:00:00"
    $0 run-once -D ../dataset/klayswap -W "2026-03-19 00:00:00,2026-03-19 04:00:00"
  $0 status -D ../dataset/klayswap
  $0 stop -D ../dataset/klayswap
EOF
}

while getopts "D:R:T:W:p:a:l:n:h" opt; do
    case "$opt" in
        D) DATA_DIR="$OPTARG" ;;
                R) RAW_DATA_BASE="$OPTARG" ;;
        T) START_TIME="$OPTARG" ;;
                W) TIME_WINDOW="$OPTARG" ;;
        p) PARSER_THREADS="$OPTARG" ;;
        a) ANOMALY_THREADS="$OPTARG" ;;
        l) MONITOR_LOG="$OPTARG" ;;
        n) DATA_PARSER_LOG="$OPTARG" ;;
        h)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

mkdir -p "$ROOT_DIR/log" "$PID_DIR"
DATA_DIR="$(realpath "$DATA_DIR")"

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
    fi
    rm -f "$pid_file"
}

ensure_binaries() {
    if [[ ! -x "$SCRIPT_DIR/bin/monitor" || ! -x "$SCRIPT_DIR/bin/data_processor" ]]; then
        echo "[INFO] Compiling parser binaries..."
        bash "$SCRIPT_DIR/build.sh"
    fi
}

start_services() {
    if [[ ! -d "$DATA_DIR" ]]; then
        echo "[ERROR] Dataset directory does not exist: $DATA_DIR"
        exit 1
    fi

    ensure_binaries

    # Raw BGP files are stored under <RAW_DATA_BASE>/<dataset_name>
    if [[ -n "$RAW_DATA_DIR_OVERRIDE" ]]; then
        RAW_DATA_DIR="$RAW_DATA_DIR_OVERRIDE"
    else
        RAW_DATA_DIR="$RAW_DATA_BASE/$(basename "$DATA_DIR")"
    fi
    if [[ ! -d "$RAW_DATA_DIR" ]]; then
        mkdir -p "$RAW_DATA_DIR" 2>/dev/null || true
    fi
    if [[ ! -d "$RAW_DATA_DIR" ]]; then
        echo "[ERROR] Raw data directory does not exist and cannot be created: $RAW_DATA_DIR"
        echo "[ERROR] Please ensure write permission on RAW_DATA_BASE=$RAW_DATA_BASE"
        exit 1
    fi

    ANOMALY_OUTPUT_DIR="$DATA_DIR/processed_data/anomaly"
    mkdir -p "$ANOMALY_OUTPUT_DIR"
    ANOMALY_CSV_FILE="$ANOMALY_OUTPUT_DIR/anomaly_links.csv"

    if is_running "$MONITOR_PID_FILE"; then
        echo "[INFO] monitor already running (pid: $(cat "$MONITOR_PID_FILE"))"
    else
        echo "[INFO] Starting monitor..."
        nohup stdbuf -oL -eL "$SCRIPT_DIR/bin/monitor" \
            --parser-threads "$PARSER_THREADS" \
            --anomaly-threads "$ANOMALY_THREADS" \
            --monitor-log "$MONITOR_LOG" \
            --csv-filename "$ANOMALY_CSV_FILE" \
            >> "$MONITOR_LOG" 2>&1 &
        echo $! > "$MONITOR_PID_FILE"
        sleep 1
        if ! is_running "$MONITOR_PID_FILE"; then
            echo "[ERROR] monitor failed to start. Check log: $MONITOR_LOG"
            exit 1
        fi
    fi

    if is_running "$PARSER_PID_FILE"; then
        echo "[INFO] data_processor already running (pid: $(cat "$PARSER_PID_FILE"))"
    else
        echo "[INFO] Starting data_processor..."
        cmd=(stdbuf -oL -eL "$SCRIPT_DIR/bin/data_processor" "-D" "$RAW_DATA_DIR" "-O" "$DATA_DIR")
        if [[ -n "$START_TIME" ]]; then
            cmd+=("-T" "$START_TIME")
        fi
        nohup env DATA_PARSER_LOG_FILE="$DATA_PARSER_LOG" "${cmd[@]}" >> "$DATA_PARSER_LOG" 2>&1 &
        echo $! > "$PARSER_PID_FILE"
        sleep 1
        if ! is_running "$PARSER_PID_FILE"; then
            echo "[ERROR] data_processor failed to start. Check log: $DATA_PARSER_LOG"
            exit 1
        fi
    fi

    echo "[SUCCESS] monitor stack started"
}

run_once_window() {
    if [[ -z "$TIME_WINDOW" ]]; then
        echo "[ERROR] run-once requires -W <start,end>"
        exit 1
    fi

    if [[ ! -d "$DATA_DIR" ]]; then
        echo "[ERROR] Dataset directory does not exist: $DATA_DIR"
        exit 1
    fi

    ensure_binaries

    if [[ -n "$RAW_DATA_DIR_OVERRIDE" ]]; then
        RAW_DATA_DIR="$RAW_DATA_DIR_OVERRIDE"
    else
        RAW_DATA_DIR="$RAW_DATA_BASE/$(basename "$DATA_DIR")"
    fi
    if [[ ! -d "$RAW_DATA_DIR" ]]; then
        mkdir -p "$RAW_DATA_DIR" 2>/dev/null || true
    fi
    if [[ ! -d "$RAW_DATA_DIR" ]]; then
        echo "[ERROR] Raw data directory does not exist and cannot be created: $RAW_DATA_DIR"
        echo "[ERROR] Please ensure write permission on RAW_DATA_BASE=$RAW_DATA_BASE"
        exit 1
    fi

    ANOMALY_OUTPUT_DIR="$DATA_DIR/processed_data/anomaly"
    mkdir -p "$ANOMALY_OUTPUT_DIR"
    ANOMALY_CSV_FILE="$ANOMALY_OUTPUT_DIR/anomaly_links.csv"
    rm -f /dev/shm/bgp_* 2>/dev/null || true

    echo "[INFO] Starting monitor (one-shot)..."
    stdbuf -oL -eL "$SCRIPT_DIR/bin/monitor" \
        --parser-threads "$PARSER_THREADS" \
        --anomaly-threads "$ANOMALY_THREADS" \
        --monitor-log "$MONITOR_LOG" \
        --csv-filename "$ANOMALY_CSV_FILE" \
        >> "$MONITOR_LOG" 2>&1 &
    local monitor_pid=$!

    cleanup_once() {
        if kill -0 "$monitor_pid" 2>/dev/null; then
            kill "$monitor_pid" 2>/dev/null || true
            wait "$monitor_pid" 2>/dev/null || true
        fi
        rm -f /dev/shm/bgp_* 2>/dev/null || true
    }
    trap cleanup_once EXIT INT TERM

    sleep 1
    if ! kill -0 "$monitor_pid" 2>/dev/null; then
        echo "[ERROR] monitor failed to start. Check log: $MONITOR_LOG"
        exit 1
    fi

    echo "[INFO] Starting data_processor one-shot for window: $TIME_WINDOW"
    if ! env DATA_PARSER_LOG_FILE="$DATA_PARSER_LOG" \
        stdbuf -oL -eL "$SCRIPT_DIR/bin/data_processor" \
        -D "$RAW_DATA_DIR" -O "$DATA_DIR" -W "$TIME_WINDOW" >> "$DATA_PARSER_LOG" 2>&1; then
        echo "[ERROR] data_processor one-shot execution failed. Check log: $DATA_PARSER_LOG"
        exit 1
    fi

    echo "[INFO] Waiting for monitor to finish..."
    if ! wait "$monitor_pid"; then
        echo "[ERROR] monitor exited with failure. Check log: $MONITOR_LOG"
        exit 1
    fi

    trap - EXIT INT TERM
    rm -f /dev/shm/bgp_* 2>/dev/null || true
    echo "[SUCCESS] one-shot monitor stack completed"
}

status_services() {
    if is_running "$MONITOR_PID_FILE"; then
        echo "[RUNNING] monitor pid=$(cat "$MONITOR_PID_FILE")"
    else
        echo "[STOPPED] monitor"
    fi

    if is_running "$PARSER_PID_FILE"; then
        echo "[RUNNING] data_processor pid=$(cat "$PARSER_PID_FILE")"
    else
        echo "[STOPPED] data_processor"
    fi
}

case "$ACTION" in
    start)
        start_services
        ;;
    stop)
        stop_one "$PARSER_PID_FILE" "data_processor"
        stop_one "$MONITOR_PID_FILE" "monitor"
        echo "[SUCCESS] monitor stack stopped"
        ;;
    restart)
        stop_one "$PARSER_PID_FILE" "data_processor"
        stop_one "$MONITOR_PID_FILE" "monitor"
        start_services
        ;;
    status)
        status_services
        ;;
    run-once)
        run_once_window
        ;;
    *)
        usage
        exit 1
        ;;
esac
