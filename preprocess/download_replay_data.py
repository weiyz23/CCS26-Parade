#!/usr/bin/env python3
"""One-shot historical replay downloader (simple thread-pool edition).

Design goals:
- Generate all tasks for a fixed [start_time, end_time) window.
- Download with a direct ThreadPoolExecutor model.
- Reuse HTTP session per worker thread.
"""

import argparse
import bz2
import datetime
import gzip
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

try:
    from .data_adapter import RIPEAdapter, RouteViewsAdapter, create_retry_session
except ImportError:  # Direct script execution from preprocess/run.sh.
    from data_adapter import RIPEAdapter, RouteViewsAdapter, create_retry_session

_THREAD_LOCAL = threading.local()


@dataclass
class DownloadStats:
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


class DecompressionError(OSError):
    """Compressed payload is complete enough to try, but cannot be unpacked."""


def parse_time_arg(value: str, arg_name: str) -> datetime.datetime:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"Missing required argument: {arg_name}")

    try:
        dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        dt = None

    if dt is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.datetime.strptime(text, fmt)
                break
            except ValueError:
                continue

    if dt is None:
        raise ValueError(f"Invalid {arg_name} format: {value}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt


def get_thread_session():
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = create_retry_session(retries=6, backoff_factor=0.5)
        _THREAD_LOCAL.session = session
    return session


def deduplicate_tasks(tasks):
    seen = set()
    unique = []
    for url, path in tasks:
        if path in seen:
            continue
        seen.add(path)
        unique.append((url, path))
    return unique


def _uncompressed_path(path: str):
    if path.endswith(".gz"):
        return path[:-3]
    if path.endswith(".bz2"):
        return path[:-4]
    return None


def _task_is_already_available(path: str) -> bool:
    uncompressed = _uncompressed_path(path)
    if uncompressed:
        return (
            (os.path.exists(path) and os.path.getsize(path) > 0)
            or (os.path.exists(uncompressed) and os.path.getsize(uncompressed) > 0)
        )
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    return False


def filter_existing_tasks(tasks):
    remaining = []
    skipped = 0
    for url, path in tasks:
        uncompressed = _uncompressed_path(path)
        if _task_is_already_available(path):
            skipped += 1
            continue
        remaining.append((url, path))
    return remaining, skipped


def _decompress_archive(path: str) -> bool:
    out_path = _uncompressed_path(path)
    if not out_path:
        return True

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    tmp_out = out_path + ".tmp"
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f_in, open(tmp_out, "wb") as f_out:
                while True:
                    chunk = f_in.read(1024 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)
        elif path.endswith(".bz2"):
            with bz2.open(path, "rb") as f_in, open(tmp_out, "wb") as f_out:
                while True:
                    chunk = f_in.read(1024 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)

        os.replace(tmp_out, out_path)
        os.remove(path)
        return True
    except Exception:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        if os.path.exists(out_path):
            os.remove(out_path)
        # Corrupted compressed payload should not be reused.
        if os.path.exists(path):
            os.remove(path)
        return False


def _remove_if_exists(path: str | None):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _remove_download_artifacts(path: str, temp_path: str | None = None):
    _remove_if_exists(temp_path)
    _remove_if_exists(path)
    _remove_if_exists(_uncompressed_path(path))


def _parse_normalized_rib_task(path: str):
    name = os.path.basename(path)
    parts = name.split(".")
    if len(parts) < 6 or parts[0] != "rib":
        return None

    source = parts[1]
    date = parts[2]
    time_text = parts[3]
    collector = ".".join(parts[4:-1])
    ext = parts[-1]
    if source not in {"ris", "rv"} or len(date) != 8 or len(time_text) != 4 or not collector:
        return None
    if ext not in {"gz", "bz2"}:
        return None

    return {
        "source": source,
        "date": date,
        "time": time_text,
        "collector": collector,
        "month": f"{date[:4]}-{date[4:6]}",
    }


def select_earliest_rib_per_collector(tasks):
    """Keep one baseline snapshot per collector for one-shot replay.

    A replay window contains several periodic RIB snapshots per collector, but
    the parser needs only the earliest snapshot as its baseline; subsequent
    state is reconstructed from updates.  Keeping all snapshots needlessly
    multiplies the decompressed workspace by the number of snapshot periods.
    """
    selected = {}
    passthrough = []
    for task in tasks:
        _, path = task
        parsed = _parse_normalized_rib_task(path)
        if parsed is None:
            passthrough.append(task)
            continue
        key = (parsed["source"], parsed["collector"])
        clock = (parsed["date"], parsed["time"])
        previous = selected.get(key)
        if previous is None or clock < previous[0]:
            selected[key] = (clock, task)
    return passthrough + [value[1] for _, value in sorted(selected.items())]


def _rib_cache_candidates(cache_root: str, path: str) -> list[str]:
    parsed = _parse_normalized_rib_task(path)
    if not parsed:
        return []

    root = os.path.abspath(cache_root)
    date = parsed["date"]
    time_text = parsed["time"]
    month = parsed["month"]
    collector = parsed["collector"]

    if parsed["source"] == "ris":
        rrc = collector if collector.startswith("rrc") else f"rrc{collector.zfill(2)}"
        base = os.path.join(root, month, "ripe", rrc)
        names = [
            f"bview.{date}.{time_text}.gz",
            f"bview.{date}.{time_text}.bz2",
            f"rib.{date}.{time_text}.gz",
            f"rib.{date}.{time_text}.bz2",
        ]
    else:
        base = os.path.join(root, month, "routeviews", collector)
        names = [
            f"rib.{date}.{time_text}.bz2",
            f"rib.{date}.{time_text}.gz",
            f"bview.{date}.{time_text}.bz2",
            f"bview.{date}.{time_text}.gz",
        ]

    return [os.path.join(base, name) for name in names]


def _parse_cache_archive_time(file_name: str):
    parts = file_name.split(".")
    if len(parts) < 4:
        return None
    try:
        return datetime.datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M").replace(
            tzinfo=datetime.timezone.utc
        )
    except ValueError:
        return None


def _iter_month_dirs(start_time: datetime.datetime, end_time: datetime.datetime):
    current = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while current < end_time:
        yield current.strftime("%Y-%m")
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def generate_rib_tasks_from_cache(cache_root: str, start_time: datetime.datetime, end_time: datetime.datetime, output_dir: str):
    tasks = []
    root = os.path.abspath(cache_root)

    for month in _iter_month_dirs(start_time, end_time):
        ripe_root = os.path.join(root, month, "ripe")
        if os.path.isdir(ripe_root):
            for collector_name in sorted(os.listdir(ripe_root)):
                collector_dir = os.path.join(ripe_root, collector_name)
                if not os.path.isdir(collector_dir):
                    continue
                collector = collector_name[3:] if collector_name.startswith("rrc") else collector_name
                for file_name in sorted(os.listdir(collector_dir)):
                    if not file_name.startswith(("bview.", "rib.")) or not file_name.endswith((".gz", ".bz2")):
                        continue
                    file_time = _parse_cache_archive_time(file_name)
                    if file_time is None or not (start_time <= file_time < end_time):
                        continue
                    parts = file_name.split(".")
                    path = os.path.join(output_dir, f"rib.ris.{parts[1]}.{parts[2]}.{collector}.{parts[-1]}")
                    tasks.append((f"cache://{os.path.join(collector_dir, file_name)}", path))

        rv_root = os.path.join(root, month, "routeviews")
        if os.path.isdir(rv_root):
            for collector in sorted(os.listdir(rv_root)):
                collector_dir = os.path.join(rv_root, collector)
                if not os.path.isdir(collector_dir):
                    continue
                for file_name in sorted(os.listdir(collector_dir)):
                    if not file_name.startswith(("rib.", "bview.")) or not file_name.endswith((".gz", ".bz2")):
                        continue
                    file_time = _parse_cache_archive_time(file_name)
                    if file_time is None or not (start_time <= file_time < end_time):
                        continue
                    parts = file_name.split(".")
                    path = os.path.join(output_dir, f"rib.rv.{parts[1]}.{parts[2]}.{collector}.{parts[-1]}")
                    tasks.append((f"cache://{os.path.join(collector_dir, file_name)}", path))

    return tasks


def _decompress_cache_archive(cache_path: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_out = out_path + ".tmp"
    _remove_if_exists(tmp_out)

    if cache_path.endswith(".gz"):
        opener = gzip.open
    elif cache_path.endswith(".bz2"):
        opener = bz2.open
    else:
        opener = open

    try:
        with opener(cache_path, "rb") as f_in, open(tmp_out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        os.replace(tmp_out, out_path)
    except Exception:
        _remove_if_exists(tmp_out)
        _remove_if_exists(out_path)
        raise


def hydrate_rib_tasks_from_cache(tasks, cache_root: str, no_network_fallback: bool):
    stats = DownloadStats()
    remaining = []
    failed = []
    total = len(tasks)

    for index, (url, path) in enumerate(tasks, start=1):
        if _task_is_already_available(path):
            stats.skipped += 1
            continue

        cache_path = next(
            (
                candidate
                for candidate in _rib_cache_candidates(cache_root, path)
                if os.path.exists(candidate) and os.path.getsize(candidate) > 0
            ),
            None,
        )
        if cache_path is None:
            if no_network_fallback:
                failed.append((url, path))
            else:
                remaining.append((url, path))
            continue

        try:
            _remove_download_artifacts(path, path + ".part")
            # The historical runner preloads the zlib-enabled wandio 6.0.4,
            # allowing both gzip and bzip2 archives to remain read-only links.
            os.symlink(cache_path, path)
            stats.downloaded += 1
        except Exception as exc:
            print(
                f"Failed to hydrate RIB from cache: {cache_path} -> {path}: "
                f"{type(exc).__name__}: {exc}"
            )
            if no_network_fallback:
                failed.append((url, path))
            else:
                remaining.append((url, path))

        if index % 50 == 0 or index == total:
            print(
                f"RIB cache hydrate progress {index}/{total}: "
                f"hydrated={stats.downloaded} skipped={stats.skipped} "
                f"remaining={len(remaining)} failed={len(failed)}"
            )

    print(
        "RIB cache hydrate summary: "
        f"hydrated={stats.downloaded} skipped={stats.skipped} "
        f"remaining_network={len(remaining)} failed={len(failed)}"
    )

    if failed and no_network_fallback:
        for url, path in failed[:20]:
            print(f"  MISSING_RIB_CACHE {path} <- {url}")
        if len(failed) > 20:
            print(f"  ... {len(failed) - 20} more missing/failed RIB cache tasks omitted")
        raise RuntimeError(
            f"RIB cache hydration failed for {len(failed)} tasks and network fallback is disabled"
        )

    return stats, remaining


def _expected_total_size(headers, resume_pos: int):
    content_range = headers.get("Content-Range") if headers else None
    if content_range:
        try:
            total_text = content_range.split("/")[-1]
            if total_text and total_text != "*":
                return int(total_text)
        except (ValueError, IndexError):
            pass

    content_length = headers.get("Content-Length") if headers else None
    if content_length:
        try:
            length = int(content_length)
            return length + resume_pos if resume_pos > 0 else length
        except ValueError:
            return None

    return None


def download_one_task(url: str, path: str, max_retries: int) -> str:
    if _task_is_already_available(path):
        return "skipped"

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    session = get_thread_session()
    temp_path = path + ".part"

    for attempt in range(max_retries):
        try:
            resume_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

            response = session.get(url, headers=headers, stream=True, timeout=(30, 300), verify=False)

            if resume_pos > 0 and response.status_code == 200:
                response.close()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                resume_pos = 0
                headers = {}
                response = session.get(url, headers=headers, stream=True, timeout=(30, 300), verify=False)

            if response.status_code == 416:
                response.close()
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    os.replace(temp_path, path)
                    return "downloaded"
                raise IOError("Range not satisfiable and no partial payload is available")

            response.raise_for_status()
            expected_size = _expected_total_size(response.headers, resume_pos)

            mode = "ab" if resume_pos > 0 else "wb"
            with open(temp_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            response.close()

            written_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            if expected_size is not None and written_size < expected_size:
                raise IOError(f"Size too small: expected>={expected_size}, got={written_size}")
            if expected_size is not None and written_size > expected_size:
                print(
                    f"Warning: larger payload for {url} (expected={expected_size}, got={written_size}), "
                    "continue with decompression verification"
                )
            if written_size <= 0:
                raise IOError("Empty payload")

            os.replace(temp_path, path)
            return "downloaded"

        except Exception as e:
            if isinstance(e, DecompressionError):
                # Bad archives should not be resumed; start the next attempt from byte 0.
                _remove_download_artifacts(path, temp_path)
            else:
                # Keep .part for transport failures so the next attempt can use Range resume.
                _remove_if_exists(path)
                _remove_if_exists(_uncompressed_path(path))

            if attempt < max_retries - 1:
                wait_time = min(20.0, 1.5 * (2 ** attempt) + random.uniform(0.0, 1.5))
                print(
                    f"Retry {attempt + 1}/{max_retries} for {url} after {wait_time:.1f}s: "
                    f"{type(e).__name__}: {e}"
                )
                time.sleep(wait_time)
            else:
                print(f"Failed to download after {max_retries} attempts: {url} - Last error: {e}")
                return "failed"

    return "failed"


def download_tasks_threadpool(tasks, workers: int, max_retries: int):
    stats = DownloadStats()
    failed_tasks = []
    total = len(tasks)
    if total == 0:
        return stats, failed_tasks

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one_task, url, path, max_retries): (url, path)
            for url, path in tasks
        }

        for future in as_completed(futures):
            url, path = futures[future]
            result = future.result()
            if result == "downloaded":
                stats.downloaded += 1
            elif result == "skipped":
                stats.skipped += 1
            else:
                stats.failed += 1
                failed_tasks.append((url, path))

            completed += 1
            if completed % 200 == 0 or completed == total:
                print(
                    f"Progress {completed}/{total}: downloaded={stats.downloaded} "
                    f"skipped={stats.skipped} failed={stats.failed}"
                )

    return stats, failed_tasks


def run_task_group(label: str, tasks, workers: int, max_retries: int):
    print(f"Starting {label} download: tasks={len(tasks)} workers={workers}")
    return download_tasks_threadpool(tasks, workers=workers, max_retries=max_retries)


def run_with_retry_rounds(label: str, tasks, workers: int, max_retries: int, retry_rounds: int):
    aggregate = DownloadStats()
    remaining = list(tasks)

    for round_idx in range(1, retry_rounds + 1):
        if not remaining:
            break

        print(f"{label} round {round_idx}/{retry_rounds}: pending={len(remaining)}")
        round_stats, failed_tasks = run_task_group(
            label,
            remaining,
            workers=workers,
            max_retries=max_retries,
        )
        aggregate.downloaded += round_stats.downloaded
        aggregate.skipped += round_stats.skipped

        if not failed_tasks:
            remaining = []
            break

        remaining = failed_tasks
        if round_idx < retry_rounds:
            wait_seconds = min(10, 2 * round_idx)
            print(
                f"{label} still has {len(remaining)} failed tasks after round {round_idx}; "
                f"retrying in {wait_seconds}s"
            )
            time.sleep(wait_seconds)

    aggregate.failed = len(remaining)
    return aggregate, remaining


def run_once(args) -> int:
    start_time = parse_time_arg(args.start_time, "--start-time")
    end_time = parse_time_arg(args.end_time, "--end-time")
    if end_time <= start_time:
        raise ValueError(
            f"Invalid replay range: end_time({end_time.isoformat()}) must be later than "
            f"start_time({start_time.isoformat()})"
        )

    upd_dir = f"{args.raw_data}/upd"
    rib_dir = f"{args.raw_data}/rib"
    os.makedirs(upd_dir, exist_ok=True)
    os.makedirs(rib_dir, exist_ok=True)

    adapters = {
        "ripe": RIPEAdapter(),
        "rv": RouteViewsAdapter(),
    }

    upd_tasks = []
    rib_tasks = []

    if not args.only_rib:
        for source, adapter in adapters.items():
            source_tasks = adapter.generate_tasks(start_time, end_time, "upd", upd_dir)
            upd_tasks.extend(source_tasks)
            print(f"Planned UPD tasks from {source}: {len(source_tasks)}")

    if not args.only_upd and args.rib_cache_root:
        rib_tasks = generate_rib_tasks_from_cache(args.rib_cache_root, start_time, end_time, rib_dir)
        print(f"Planned RIB tasks from cache: {len(rib_tasks)}")
    elif not args.only_upd:
        for source, adapter in adapters.items():
            source_tasks = adapter.generate_tasks(start_time, end_time, "rib", rib_dir)
            rib_tasks.extend(source_tasks)
            print(f"Planned RIB tasks from {source}: {len(source_tasks)}")

    upd_tasks = deduplicate_tasks(upd_tasks)
    rib_tasks = deduplicate_tasks(select_earliest_rib_per_collector(rib_tasks))

    upd_tasks, upd_prefilter_skipped = filter_existing_tasks(upd_tasks)
    rib_tasks, rib_prefilter_skipped = filter_existing_tasks(rib_tasks)

    rib_cache_stats = DownloadStats()
    if args.rib_cache_root and rib_tasks:
        print(f"Hydrating RIB tasks from local cache: {args.rib_cache_root}")
        rib_cache_stats, rib_tasks = hydrate_rib_tasks_from_cache(
            rib_tasks,
            args.rib_cache_root,
            no_network_fallback=args.no_rib_network_fallback,
        )

    total_planned = (
        len(upd_tasks)
        + len(rib_tasks)
        + upd_prefilter_skipped
        + rib_prefilter_skipped
        + rib_cache_stats.downloaded
        + rib_cache_stats.skipped
    )
    print(
        f"Replay window UTC: [{start_time.isoformat()}, {end_time.isoformat()})\n"
        f"Planned unique tasks: upd={len(upd_tasks) + upd_prefilter_skipped} "
        f"rib={len(rib_tasks) + rib_prefilter_skipped + rib_cache_stats.downloaded + rib_cache_stats.skipped} "
        f"total={total_planned}\n"
        f"Prefilter skipped existing files: upd={upd_prefilter_skipped} rib={rib_prefilter_skipped}\n"
        f"RIB cache: hydrated={rib_cache_stats.downloaded} skipped={rib_cache_stats.skipped} "
        f"network_remaining={len(rib_tasks)}"
    )

    if len(upd_tasks) + len(rib_tasks) == 0:
        print("No tasks planned in target window. Nothing to download.")
        return 0

    overall = DownloadStats(
        downloaded=rib_cache_stats.downloaded,
        skipped=upd_prefilter_skipped + rib_prefilter_skipped + rib_cache_stats.skipped,
    )

    if args.serial_types:
        failed_tasks = []
        if rib_tasks:
            rib_stats, rib_failed_tasks = run_with_retry_rounds(
                "RIB",
                rib_tasks,
                workers=args.rib_workers,
                max_retries=args.max_retries,
                retry_rounds=args.retry_rounds,
            )
            overall.downloaded += rib_stats.downloaded
            overall.skipped += rib_stats.skipped
            overall.failed += rib_stats.failed
            failed_tasks.extend(rib_failed_tasks)
        if upd_tasks:
            upd_stats, upd_failed_tasks = run_with_retry_rounds(
                "UPD",
                upd_tasks,
                workers=args.upd_workers,
                max_retries=args.max_retries,
                retry_rounds=args.retry_rounds,
            )
            overall.downloaded += upd_stats.downloaded
            overall.skipped += upd_stats.skipped
            overall.failed += upd_stats.failed
            failed_tasks.extend(upd_failed_tasks)
    else:
        mixed = rib_tasks + upd_tasks
        random.shuffle(mixed)
        workers = max(args.rib_workers, args.upd_workers)
        mixed_stats, failed_tasks = run_with_retry_rounds(
            "ALL",
            mixed,
            workers=workers,
            max_retries=args.max_retries,
            retry_rounds=args.retry_rounds,
        )
        overall.downloaded += mixed_stats.downloaded
        overall.skipped += mixed_stats.skipped
        overall.failed += mixed_stats.failed

    print(
        "Replay download summary: "
        f"downloaded={overall.downloaded} skipped={overall.skipped} failed={overall.failed}"
    )

    failed_rib_tasks = [task for task in failed_tasks if f"{os.sep}rib{os.sep}" in task[1]]
    failed_upd_tasks = [task for task in failed_tasks if f"{os.sep}upd{os.sep}" in task[1]]

    if overall.failed > 0:
        print(
            "Warning: continuing despite failed downloads. "
            "Downstream stages will use the raw files that are available."
        )
        print(
            "Failed task breakdown: "
            f"rib={len(failed_rib_tasks)} upd={len(failed_upd_tasks)}"
        )
        for url, path in failed_tasks[:20]:
            print(f"  FAILED {path} <- {url}")
        if len(failed_tasks) > 20:
            print(f"  ... {len(failed_tasks) - 20} more failed tasks omitted")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-shot replay downloader for historical BGP data")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory (reserved for interface consistency)")
    parser.add_argument("--raw-data", type=str, required=True, help="Raw data root containing upd/rib")
    parser.add_argument("--start-time", type=str, required=True, help="Replay start time (UTC)")
    parser.add_argument("--end-time", type=str, required=True, help="Replay end time (UTC, exclusive)")
    parser.add_argument(
        "--rib-workers", type=int,
        default=int(os.environ.get("PARADE_RIB_WORKERS", "8")),
        help="Thread count for RIB download",
    )
    parser.add_argument(
        "--upd-workers", type=int,
        default=int(os.environ.get("PARADE_UPD_WORKERS", "32")),
        help="Thread count for UPD download",
    )
    parser.add_argument("--max-retries", type=int, default=6, help="Max retries per file")
    parser.add_argument("--retry-rounds", type=int, default=3, help="Retry rounds for failed task subsets")
    parser.add_argument("--only-rib", action="store_true", help="Download RIB only")
    parser.add_argument("--only-upd", action="store_true", help="Download UPD only")
    parser.add_argument("--serial-types", action="store_true", help="Download RIB then UPD sequentially")
    parser.add_argument(
        "--rib-cache-root",
        type=str,
        default=None,
        help="Local RIB archive cache root, e.g. /networkDisk/RIB. Matching archives are linked read-only instead of downloaded.",
    )
    parser.add_argument(
        "--no-rib-network-fallback",
        action="store_true",
        help="Fail if a RIB archive is missing from --rib-cache-root instead of downloading it.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.only_rib and args.only_upd:
        raise ValueError("--only-rib and --only-upd cannot be used together")
    if args.no_rib_network_fallback and not args.rib_cache_root:
        raise ValueError("--no-rib-network-fallback requires --rib-cache-root")

    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
