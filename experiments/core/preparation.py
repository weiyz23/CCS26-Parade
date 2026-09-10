"""Prepare the fixed paper snapshot and monthly AS metadata on demand."""

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import sys
import time

from .runtime import CONFIGS, ROOT, exclusive_lock, file_stats, read_json, run_logged, sha256, write_json


TRAINING_FILES = ("meta.json", "profile_triplets.pt", "profile_fingerprints.pt")
SNAPSHOT_FILES = TRAINING_FILES + (
    "prefix_map.pt", "as_map.pt", "prefix_to_profile.pt", "fingerprint_anchor_db.pt",
    "history_cache.pt", "rib_records_cache.bin",
)


def validate_training_data(data_dir):
    missing = [name for name in TRAINING_FILES if not (data_dir / name).is_file() or not (data_dir / name).stat().st_size]
    if missing:
        raise FileNotFoundError(f"Missing training artifacts in {data_dir}: {', '.join(missing)}")
    meta = read_json(data_dir / "meta.json")
    if any(int(meta.get(key, 0)) <= 0 for key in ("num_profiles", "num_asns")):
        raise ValueError(f"Empty training dataset: {data_dir}")


def prepare_as_metadata(date, dataset_dir, cache_root):
    # Use the existing monthly CAIDA query, with strict pagination and atomic publication.
    from preprocess.download_history_data import download_as_rank_data

    month = date[:7]
    cache_dir = cache_root / "as_metadata" / month
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "as_info.jsonl"
    marker = cache_dir / "manifest.json"
    with exclusive_lock(cache_dir / ".lock"):
        valid_cache = False
        if marker.is_file() and cached.is_file():
            try:
                prior = read_json(marker)
            except json.JSONDecodeError:
                prior = {}
            valid_cache = prior.get("month") == month and prior.get("sha256") == sha256(cached)
        if not valid_cache:
            # Only this helper's cache is replaced; a partially downloaded file is never published.
            pending = cache_dir / "download.jsonl"
            pending.unlink(missing_ok=True)
            download_as_rank_data(date, pending, strict=True)
            rows = [json.loads(line) for line in pending.read_text().splitlines() if line.strip()]
            if not rows or any("asn" not in row for row in rows):
                raise ValueError(f"Invalid AS metadata for {month}")
            pending.replace(cached)
            write_json(marker, {"month": month, "rows": len(rows), "sha256": sha256(cached)})
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target = dataset_dir / "as_info.jsonl"
        shutil.copy2(cached, target.with_suffix(".jsonl.tmp"))
        target.with_suffix(".jsonl.tmp").replace(target)
        write_json(dataset_dir / "as_metadata_manifest.json", read_json(marker))
    return target


def snapshot_tasks(spec, rib_dir):
    from preprocess.data_adapter import RIPEAdapter, RouteViewsAdapter
    from preprocess.download_replay_data import _parse_normalized_rib_task

    start = datetime.fromisoformat(spec["snapshot_utc"].replace("Z", "+00:00"))
    expected = set(spec["collector_keys"])
    if len(expected) != len(spec["collector_keys"]):
        raise ValueError("Duplicate snapshot collectors")
    adapters = {"ris": RIPEAdapter(), "rv": RouteViewsAdapter()}
    adapters["ris"].ava_rrcs = [key.split(":", 1)[1] for key in spec["collector_keys"] if key.startswith("ris:")]
    adapters["rv"].collectors = [key.split(":", 1)[1] for key in spec["collector_keys"] if key.startswith("rv:")]
    found = {}
    for source, adapter in adapters.items():
        for url, path in adapter.generate_tasks(start, start + timedelta(minutes=1), "rib", str(rib_dir)):
            parsed = _parse_normalized_rib_task(path)
            if parsed is None:
                raise ValueError(f"Unrecognized RIB filename: {path}")
            key = f"{source}:{parsed['collector']}"
            if key not in expected or parsed["date"] != start.strftime("%Y%m%d") or parsed["time"] != "0000":
                raise ValueError(f"Unexpected snapshot input: {path}")
            if key in found:
                raise ValueError(f"Multiple archives for {key}")
            found[key] = {"collector": key, "url": url, "archive": path}
    if set(found) != expected:
        raise RuntimeError(f"Missing snapshot collectors: {', '.join(sorted(expected - set(found)))}")
    return [found[key] for key in spec["collector_keys"]]


def prepare_snapshot(data_dir, raw_root, log_path):
    from preprocess.download_replay_data import _decompress_archive, _uncompressed_path, run_with_retry_rounds

    spec = read_json(CONFIGS / "snapshot_20260101.json")
    sources = [ROOT / "preprocess" / "build_dataset.py", Path(__file__).resolve()]
    sources += sorted((ROOT / "preprocess" / "parser" / "include").glob("*.hpp"))
    sources.append(ROOT / "preprocess" / "parser" / "src" / "processor.cpp")
    identity = {"snapshot": spec, "preparation_code": {str(p.relative_to(ROOT)): sha256(p) for p in sources}}
    marker = data_dir / "preparation_manifest.json"
    artifacts = [data_dir / name for name in SNAPSHOT_FILES]
    with exclusive_lock(data_dir.parent / ".ae-snapshot.lock"):
        if marker.is_file() and all(p.is_file() and p.stat().st_size for p in artifacts):
            try:
                previous = read_json(marker)
            except json.JSONDecodeError:
                previous = {}
            if previous.get("identity") == identity and previous.get("artifacts") == file_stats(artifacts):
                validate_training_data(data_dir)
                print(f"Reusing prepared snapshot: {data_dir}", flush=True)
                return data_dir
        marker.unlink(missing_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = raw_root / "ae_snapshot_20260101"
        rib_dir = raw_dir / "rib"
        rib_dir.mkdir(parents=True, exist_ok=True)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with Path(log_path).open("a", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            inventory = snapshot_tasks(spec, rib_dir)
            write_json(data_dir / "download_manifest.json", {"snapshot": spec, "status": "planned", "inputs": inventory})
            tasks = [(item["url"], item["archive"]) for item in inventory]
            _, failures = run_with_retry_rounds("AE snapshot RIB", tasks, workers=8, max_retries=6, retry_rounds=3)
            if failures:
                raise RuntimeError(f"Snapshot download failed for {len(failures)} collectors")
            for item in inventory:
                archive = item["archive"]
                if not _decompress_archive(archive):
                    raise RuntimeError(f"Cannot decompress snapshot archive: {archive}")
                path = Path(_uncompressed_path(archive))
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Empty or missing snapshot RIB: {path}")
                item.update(path=str(path), bytes=path.stat().st_size)
        # Isolate the exact collector set from any unrelated raw files or UPDATEs.
        selected = raw_dir / "selected" / "rib"
        selected.mkdir(parents=True, exist_ok=True)
        for old in selected.iterdir():
            if not old.is_symlink():
                raise RuntimeError(f"Unexpected file in managed snapshot links: {old}")
            old.unlink()
        for item in inventory:
            path = Path(item["path"])
            (selected / path.name).symlink_to(path.resolve())
        # The builder reuses tensors by filename. Invalidate only its known generated files
        # before rebuilding an incomplete or changed managed snapshot.
        for name in SNAPSHOT_FILES + ("upd_counts.pt",):
            (data_dir / name).unlink(missing_ok=True)
        processor = ROOT / "preprocess" / "parser" / "bin" / "processor"
        run_logged([processor, selected.parent, data_dir, "--require-nonempty-ribs"], log_path)
        run_logged([sys.executable, ROOT / "preprocess" / "build_dataset.py", "--data_dir", data_dir], log_path)
        validate_training_data(data_dir)
        if not all(p.is_file() and p.stat().st_size for p in artifacts):
            raise RuntimeError(f"Incomplete prepared snapshot: {data_dir}")
        write_json(data_dir / "download_manifest.json", {"snapshot": spec, "status": "complete", "inputs": inventory})
        write_json(marker, {"identity": identity, "artifacts": file_stats(artifacts), "seconds": time.monotonic() - started})
    return data_dir
