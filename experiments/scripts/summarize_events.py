#!/usr/bin/env python3
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core.runtime import CONFIGS, absolute, read_json, write_csv

FIELDS = [
    "dataset", "category", "start_utc", "end_utc", "candidate_rows", "detected_rows",
    "pre_filter_rows", "pre_filter_prefixes", "pre_filter_asns", "pre_filter_groups",
    "post_filter_rows", "post_filter_prefixes", "post_filter_asns", "post_filter_groups",
    "removed_rows", "org_id_removed", "org_name_removed", "rpki_removed",
]
REASONS = {"org_id_removed": "ORG_ID_MATCH", "org_name_removed": "ORG_NAME_MATCH",
           "rpki_removed": "RPKI_VALID_CURRENT"}


def count_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def read_stats(path):
    stats = dict(rows=0, prefixes=set(), asns=set(), groups=set(), reasons=Counter())
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row.get("prefix") or not row.get("asn"):
                raise ValueError(f"Missing prefix or asn in {path}")
            stats["rows"] += 1
            stats["prefixes"].add(row["prefix"])
            stats["asns"].add(int(row["asn"]))
            stats["groups"].add(int(row.get("group_asn") or row["asn"]))
            reasons = [part.strip() for part in (row.get("fp_reasons") or "").split(";")]
            for field, marker in REASONS.items():
                if any(reason.startswith(marker) for reason in reasons):
                    stats["reasons"][field] += 1
    return stats


def collect_event(event, dataset_dir):
    candidates = [dataset_dir / relative for relative in (
        "processed_data/anomaly/anomaly_links.csv", "processed_data/anomaly_links.csv", "anomaly_links.csv")]
    candidate = next((path for path in candidates if path.is_file()), None)
    if candidate is None:
        raise FileNotFoundError(f"No anomaly_links.csv found in the three pipeline locations under {dataset_dir}")
    results = dataset_dir / "results"
    row = {field: event[field] for field in ("dataset", "category", "start_utc", "end_utc")}
    row.update(candidate_rows=count_rows(candidate), detected_rows=count_rows(results / "anomalous_raw.csv"),
               removed_rows=0, **dict.fromkeys(REASONS, 0))
    pre, post = [], []
    for subtype in ("origin", "non_origin"):
        before_path = results / f"anomalous_{subtype}.csv"
        before = read_stats(before_path)
        pre.append(before)
        # The filter does not write output files for empty inputs.
        if not before["rows"]:
            continue
        after_path = results / f"anomalous_{subtype}_anom.csv"
        removed_path = results / f"anomalous_{subtype}_false_pos.csv"
        for path in (after_path, removed_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required filtered output is missing for nonempty {before_path}: {path}")
        after, removed = read_stats(after_path), read_stats(removed_path)
        if before["rows"] != after["rows"] + removed["rows"]:
            raise ValueError(f"Filtered row counts do not partition {before_path}")
        post.append(after)
        row["removed_rows"] += removed["rows"]
        for field in REASONS:
            row[field] += removed["reasons"][field]
    for stage, parts in (("pre_filter", pre), ("post_filter", post)):
        row[f"{stage}_rows"] = sum(part["rows"] for part in parts)
        for field in ("prefixes", "asns", "groups"):
            row[f"{stage}_{field}"] = len(set().union(*(part[field] for part in parts)))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize historical detection outputs without rescoring or regrouping ASNs")
    parser.add_argument("--datasets", help="Comma-separated event IDs; default: all eight in configuration order")
    parser.add_argument("--dataset-root", type=absolute, default=ROOT / "dataset")
    parser.add_argument("--output-dir", type=absolute, default=ROOT / "experiments" / "results" / "main")
    args = parser.parse_args(argv)
    events = read_json(CONFIGS / "historical_events.json")
    if args.datasets is not None:
        selected = [name.strip() for name in args.datasets.split(",")]
        if len(set(selected)) != len(selected) or set(selected) - {event["dataset"] for event in events}:
            parser.error(f"Unknown or duplicate dataset IDs: {args.datasets}")
        events = [event for event in events if event["dataset"] in selected]
    try:
        rows = [collect_event(event, args.dataset_root / event["dataset"]) for event in events]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "historical_metrics.csv"
        write_csv(output, rows, FIELDS)
        columns = ("dataset", "category", "candidate_rows", "detected_rows", "pre_filter_prefixes",
                   "post_filter_prefixes", "post_filter_groups", "removed_rows")
        widths = {column: max(len(column), *(len(str(row[column])) for row in rows)) for column in columns}
        print("  ".join(column.ljust(widths[column]) for column in columns))
        for row in rows:
            print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))
        print(f"Saved {output}")
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print(f"Historical summary failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
