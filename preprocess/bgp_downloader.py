# bgp_downloader.py
# BGP Data Downloader for continuous incremental updates

import os
import datetime
import asyncio
import json
import shutil
from data_adapter import (
    FILE_LIST_CACHE_DIRNAME,
    RIPEAdapter,
    RouteViewsAdapter,
    download_files_parallel_async,
    create_retry_session,
)

COMPLETENESS_POLICY_PATH = os.path.join(os.path.dirname(__file__), 'completeness_policy.json')


def load_completeness_policy() -> dict:
    with open(COMPLETENESS_POLICY_PATH, 'r', encoding='utf-8') as f:
        loaded = json.load(f)

    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid completeness policy root: {COMPLETENESS_POLICY_PATH}")

    for source in ('ris', 'rv'):
        src_cfg = loaded.get('sources', {}).get(source, {})
        collectors = int(src_cfg.get('collectors'))
        if collectors <= 0:
            raise ValueError(f"Invalid collectors for source={source} in {COMPLETENESS_POLICY_PATH}")
        collector_ids = src_cfg.get('collector_ids')
        if not isinstance(collector_ids, list) or not collector_ids:
            raise ValueError(f"Missing collector_ids for source={source} in {COMPLETENESS_POLICY_PATH}")
        collector_ids = [str(x) for x in collector_ids]
        if len(set(collector_ids)) != len(collector_ids):
            raise ValueError(f"Duplicate collector_ids for source={source} in {COMPLETENESS_POLICY_PATH}")
        if len(collector_ids) != collectors:
            raise ValueError(f"collector_ids size mismatch for source={source} in {COMPLETENESS_POLICY_PATH}")

    for dtype in ('upd', 'rib'):
        for source in ('ris', 'rv'):
            rule = loaded.get('rules', {}).get(dtype, {}).get(source)
            if not isinstance(rule, str) or not rule.strip():
                raise ValueError(f"Missing rule for {dtype}.{source} in {COMPLETENESS_POLICY_PATH}")

    return loaded


def parse_start_time(start_time_str: str | None) -> datetime.datetime:
    if not start_time_str:
        return datetime.datetime.now(datetime.timezone.utc)

    text = start_time_str.strip()
    try:
        dt = datetime.datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        dt = None

    if dt is None:
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S'):
            try:
                dt = datetime.datetime.strptime(text, fmt)
                break
            except ValueError:
                continue

    if dt is None:
        raise ValueError(f"Invalid --start-time format: {start_time_str}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt


def parse_required_time_arg(time_str: str | None, arg_name: str) -> datetime.datetime:
    if not time_str or not time_str.strip():
        raise ValueError(f"Missing required argument: {arg_name}")
    return parse_start_time(time_str)


def init_program_log_from_env(env_name: str):
    log_path = os.environ.get(env_name, '').strip()
    if not log_path:
        return
    try:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(log_path, 'w', encoding='utf-8'):
            pass
    except Exception as e:
        print(f"Failed to initialize log file from {env_name}: {e}")

class BGPDataDownloader:
    def __init__(
        self,
        dataset_dir: str,
        raw_data_dir: str,
        start_time: datetime.datetime | None = None,
        end_time: datetime.datetime | None = None,
    ):
        self.dataset_dir = dataset_dir
        self.raw_data_dir = raw_data_dir
        self.start_time = start_time or datetime.datetime.now(datetime.timezone.utc)
        if self.start_time.tzinfo is None:
            self.start_time = self.start_time.replace(tzinfo=datetime.timezone.utc)
        else:
            self.start_time = self.start_time.astimezone(datetime.timezone.utc)

        if end_time is None:
            raise ValueError("end_time is required for historical replay mode")
        if end_time.tzinfo is None:
            self.end_time = end_time.replace(tzinfo=datetime.timezone.utc)
        else:
            self.end_time = end_time.astimezone(datetime.timezone.utc)

        if self.end_time < self.start_time:
            raise ValueError(
                f"Invalid replay range: end_time({self.end_time.isoformat()}) "
                f"is earlier than start_time({self.start_time.isoformat()})"
            )
        self.ripe_adapter = RIPEAdapter()
        self.rv_adapter = RouteViewsAdapter()
        self.upd_dir = os.path.join(raw_data_dir, 'upd')
        self.rib_dir = os.path.join(raw_data_dir, 'rib')
        self.upd_scan_span = datetime.timedelta(days=7)
        self.rib_scan_span = datetime.timedelta(hours=5)
        self.upd_status_retention = datetime.timedelta(days=21)
        self.rib_status_retention = datetime.timedelta(hours=48)
        self.upd_file_retention_hours = 14 * 24
        self.rib_file_retention_hours = 48
        self.window_end = {
            'upd': self.start_time,
            'rib': self.start_time,
        }
        # `window_step` controls how quickly the anchor advances each cycle.
        # `window_span` controls how much history each cycle scans/downloads.
        self.window_step = {
            'upd': datetime.timedelta(minutes=20),
            'rib': datetime.timedelta(hours=1),
        }
        self.window_span = {
            'upd': self.upd_scan_span,
            'rib': self.rib_scan_span,
        }

        self.completeness_policy = load_completeness_policy()
        
        # Create directories with error handling
        try:
            os.makedirs(dataset_dir, exist_ok=True)
            os.makedirs(raw_data_dir, exist_ok=True)
            os.makedirs(self.upd_dir, exist_ok=True)
            os.makedirs(self.rib_dir, exist_ok=True)
        except PermissionError as e:
            print(f"Permission denied when creating directories: {e}")
            print(f"Please ensure you have write permissions for the directories: {dataset_dir} and {raw_data_dir}")
            raise

        # Reset remote file-list cache on each run to avoid stale index drift.
        cache_dir = os.path.join(raw_data_dir, FILE_LIST_CACHE_DIRNAME)
        try:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Reset file list cache directory: {cache_dir}")
        except Exception as e:
            print(f"Failed to reset file list cache directory {cache_dir}: {e}")
        
        # Completion status: (source, data_type, timestamp) -> set of collectors
        self.completion_status = {}
        # Use fixed operational thresholds instead of source list lengths.
        self.all_collectors = {
            'ripe': int(self.completeness_policy['sources']['ris']['collectors']),
            'rv': int(self.completeness_policy['sources']['rv']['collectors']),
        }
        self.collector_universe = {
            'ripe': set(str(x) for x in self.completeness_policy['sources']['ris']['collector_ids']),
            'rv': set(str(x) for x in self.completeness_policy['sources']['rv']['collector_ids']),
        }
        # Track per-timestamp completion by source to emit a single .done marker.
        self.completed_sources = {}
        # Track sources that reached completion by force-complete policy.
        self.forced_complete_sources = {}
        self.done_markers_written = set()

    def _parse_timestamp_text(self, timestamp: str):
        try:
            return datetime.datetime.strptime(timestamp, '%Y%m%d.%H%M').replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            return None

    def _rule_expected(self, rule: str, hour: int, minute: int) -> bool:
        if rule == 'always':
            return True
        if rule == 'never':
            return False
        if rule == 'minute_mod_5':
            return (minute % 5) == 0
        if rule == 'minute_mod_15':
            return (minute % 15) == 0
        if rule == 'hour_mod_4':
            return minute == 0 and (hour % 4) == 0
        if rule == 'hour_mod_8':
            return minute == 0 and (hour % 8) == 0
        return False

    def _is_source_expected_for_timestamp(self, source: str, data_type: str, timestamp: str) -> bool:
        ts_dt = self._parse_timestamp_text(timestamp)
        if ts_dt is None:
            return False

        if source == 'ripe':
            source_key = 'ris'
        elif source == 'rv':
            source_key = 'rv'
        else:
            return False

        rule = (
            self.completeness_policy
            .get('rules', {})
            .get(data_type, {})
            .get(source_key, '')
        )
        if not isinstance(rule, str):
            return False
        return self._rule_expected(rule, ts_dt.hour, ts_dt.minute)

    def _required_sources_for_timestamp(self, data_type: str, timestamp: str):
        required = {'ripe'}
        if self._is_source_expected_for_timestamp('rv', data_type, timestamp):
            required.add('rv')
        return required

    def _get_data_dir_by_type(self, data_type: str) -> str:
        return self.upd_dir if data_type == 'upd' else self.rib_dir

    def _build_collector_counts_by_source(self, data_type: str, timestamp: str):
        counts = {}
        for source in ('ripe', 'rv'):
            expected_for_timestamp = self._is_source_expected_for_timestamp(source, data_type, timestamp)
            expected = self.all_collectors[source] if expected_for_timestamp else 0
            received = len(self.completion_status.get((source, data_type, timestamp), set()))
            missing = max(expected - received, 0)
            counts[source] = {
                'expected_for_timestamp': expected_for_timestamp,
                'received': received,
                'expected': expected,
                'missing': missing,
            }
        return counts

    def _build_collector_details_by_source(self, data_type: str, timestamp: str):
        details = {}
        for source in ('ripe', 'rv'):
            expected_for_timestamp = self._is_source_expected_for_timestamp(source, data_type, timestamp)
            if not expected_for_timestamp:
                details[source] = {
                    'expected_for_timestamp': False,
                    'expected_collectors': 0,
                    'received_collectors': [],
                    'missing_collectors': [],
                }
                continue

            expected_collectors = self.collector_universe[source]
            received_collectors = self.completion_status.get((source, data_type, timestamp), set())
            missing_collectors = sorted(expected_collectors - received_collectors)
            details[source] = {
                'expected_for_timestamp': True,
                'expected_collectors': len(expected_collectors),
                'received_collectors': sorted(received_collectors),
                'missing_collectors': missing_collectors,
            }
        return details

    def _parse_bgp_filename_components(self, filename: str):
        parts = filename.split('.')
        if len(parts) < 5:
            return None, None, None, None
        data_type = parts[0]
        source_text = parts[1]
        if source_text == 'ris':
            source = 'ripe'
        elif source_text == 'rv':
            source = 'rv'
        else:
            return None, None, None, None

        if not (parts[2].isdigit() and len(parts[2]) == 8 and parts[3].isdigit() and len(parts[3]) == 4):
            return None, None, None, None

        timestamp = parts[2] + '.' + parts[3]
        collector_parts = parts[4:]
        if collector_parts and collector_parts[-1] in ('gz', 'bz2'):
            collector_parts = collector_parts[:-1]
        if not collector_parts:
            return None, None, None, None

        collector = '.'.join(collector_parts)
        return data_type, source, timestamp, collector

    def _scan_collectors_by_timestamp(self, data_type: str):
        data_dir = self._get_data_dir_by_type(data_type)
        collectors_by_ts = {}
        if not os.path.exists(data_dir):
            return collectors_by_ts

        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if not os.path.isfile(filepath):
                continue
            if filename.endswith('.done'):
                continue

            parsed_data_type, source, timestamp, collector = self._parse_bgp_filename_components(filename)
            if parsed_data_type != data_type or source is None or timestamp is None or collector is None:
                continue

            by_source = collectors_by_ts.setdefault(timestamp, {'ripe': set(), 'rv': set()})
            by_source[source].add(collector)

        return collectors_by_ts

    def _prime_completion_status_from_disk(self, data_type: str):
        collectors_by_ts = self._scan_collectors_by_timestamp(data_type)
        for timestamp, source_map in collectors_by_ts.items():
            for source in ('ripe', 'rv'):
                if not source_map[source]:
                    continue
                self.completion_status[(source, data_type, timestamp)] = set(source_map[source])

    def _force_complete_stale_timestamps(self, data_type: str, window_start: datetime.datetime):
        self._prime_completion_status_from_disk(data_type)

        data_dir = self._get_data_dir_by_type(data_type)
        stale_timestamps = set()
        if not os.path.exists(data_dir):
            return

        for key in self.completion_status.keys():
            source, key_data_type, timestamp = key
            if key_data_type != data_type:
                continue
            ts_dt = self._parse_timestamp_text(timestamp)
            if ts_dt is None:
                continue
            if ts_dt >= window_start:
                continue
            done_path = os.path.join(data_dir, f"{timestamp}.done")
            if os.path.exists(done_path):
                self.done_markers_written.add((data_type, timestamp))
                continue
            stale_timestamps.add(timestamp)

        for timestamp in sorted(stale_timestamps):
            required_sources = self._required_sources_for_timestamp(data_type, timestamp)
            details = self._build_collector_details_by_source(data_type, timestamp)
            details_summary = {}
            for source in sorted(required_sources):
                src = details[source]
                details_summary[source] = {
                    'received': len(src['received_collectors']),
                    'missing': len(src['missing_collectors']),
                    'missing_collectors': src['missing_collectors'],
                }
            print(
                "Force-done by window scan: "
                f"data_type={data_type} timestamp={timestamp} details={json.dumps(details_summary, ensure_ascii=True)}"
            )

            for source in sorted(required_sources):
                src = details[source]
                self.signal_data_complete(
                    source,
                    data_type,
                    timestamp,
                    forced=True,
                    attempts=0,
                    missing_collectors=len(src['missing_collectors']),
                    received_collectors=len(src['received_collectors']),
                    expected_collectors=src['expected_collectors'],
                    missing_collector_names=src['missing_collectors'],
                )

    def _write_done_marker_if_ready(self, data_type: str, timestamp: str):
        key = (data_type, timestamp)
        if key in self.done_markers_written:
            return

        required_sources = self._required_sources_for_timestamp(data_type, timestamp)
        completed = self.completed_sources.get(key, set())
        if not required_sources.issubset(completed):
            return

        collector_counts_by_source = self._build_collector_counts_by_source(data_type, timestamp)
        collector_details_by_source = self._build_collector_details_by_source(data_type, timestamp)
        forced_sources = sorted(self.forced_complete_sources.get(key, set()))

        data_dir = self._get_data_dir_by_type(data_type)
        done_path = os.path.join(data_dir, f"{timestamp}.done")
        tmp_path = done_path + ".tmp"
        payload = json.dumps(
            {
                'timestamp': timestamp,
                'data_type': data_type,
                'completed_sources': sorted(completed),
                'required_sources': sorted(required_sources),
                'forced_complete_sources': forced_sources,
                'collector_counts_by_source': collector_counts_by_source,
                'collector_details_by_source': collector_details_by_source,
                'missing_collectors_by_source': {
                    source: collector_details_by_source[source]['missing_collectors']
                    for source in ('ripe', 'rv')
                },
                'written_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            ensure_ascii=True,
        )
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(payload)
            f.write('\n')
        os.replace(tmp_path, done_path)
        self.done_markers_written.add(key)
        print(f"Wrote done marker: {done_path}")

        # Emit one delay record per timestamp, only after done marker is committed.
        now = datetime.datetime.now(datetime.timezone.utc)
        try:
            data_ts = datetime.datetime.strptime(timestamp, '%Y%m%d.%H%M').replace(tzinfo=datetime.timezone.utc)
            delay_seconds = (now - data_ts).total_seconds()
            delay_minutes = delay_seconds / 60
            print(f"Data delay: {delay_seconds:.1f} seconds ({delay_minutes:.1f} minutes) for {data_type} at {timestamp}")
        except ValueError:
            print(f"Data complete for {data_type} at {timestamp} (unable to calculate delay)")

    def _get_download_window(self, data_type: str):
        step = self.window_step[data_type]
        span = self.window_span[data_type]
        prev_end_time = self.window_end[data_type]

        if prev_end_time >= self.end_time:
            return None, None, True

        end_time = min(prev_end_time, self.end_time)
        self.window_end[data_type] = prev_end_time + step
        start_time = end_time - span
        return start_time, end_time, False

    def _parse_bgp_file_timestamp(self, filename: str):
        """Parse UTC timestamp from BGP filename."""
        _, _, timestamp, _ = self._parse_bgp_filename_components(filename)
        if timestamp is None:
            return None
        return self._parse_timestamp_text(timestamp)

    def _parse_bgp_filename_meta(self, filename: str):
        """Parse (timestamp, collector) from BGP filename."""
        _, _, timestamp, collector = self._parse_bgp_filename_components(filename)
        return timestamp, collector

    async def _download_source_window(self, source: str, data_type: str, start_time: datetime.datetime, end_time: datetime.datetime):
        if source == 'ripe':
            adapter = self.ripe_adapter
        elif source == 'rv':
            adapter = self.rv_adapter
        else:
            return False

        data_dir = self._get_data_dir_by_type(data_type)
        tasks = adapter.generate_tasks(start_time, end_time, data_type, data_dir)
        if not tasks:
            return False

        successful_paths, failed_paths, skipped_paths = await download_files_parallel_async(tasks, profile=data_type)
        self.update_completion_status(successful_paths + skipped_paths, source, data_type)
        return bool(successful_paths)

    async def download_upd_window(self):
        """Download UPD files in configured rolling window.

        Returns a tuple: (downloaded_any, reached_end_time).
        """
        start_time, end_time, reached_end_time = self._get_download_window('upd')
        if reached_end_time:
            return False, True

        downloaded_any = False

        for source in ('ripe', 'rv'):
            source_downloaded = await self._download_source_window(source, 'upd', start_time, end_time)
            downloaded_any = downloaded_any or source_downloaded

        return downloaded_any, False

    def update_completion_status(self, successful_paths, source, data_type):
        """Update completion status and check for completeness"""
        total_collectors = self.all_collectors[source]
        updated_keys = set()
        
        for path in successful_paths:
            filename = os.path.basename(path)
            timestamp, collector = self._parse_bgp_filename_meta(filename)
            if timestamp is None or collector is None:
                continue
            key = (source, data_type, timestamp)
            updated_keys.add(key)
            if key not in self.completion_status:
                self.completion_status[key] = set()
            self.completion_status[key].add(collector)
        
        # Check completeness for updated keys
        for key in updated_keys:
            timestamp = key[2]
            if not self._is_source_expected_for_timestamp(source, data_type, timestamp):
                continue

            current_size = len(self.completion_status[key])
            if current_size >= total_collectors:
                self.signal_data_complete(
                    source,
                    data_type,
                    timestamp,
                    forced=False,
                    attempts=0,
                    missing_collectors=0,
                    received_collectors=current_size,
                    expected_collectors=total_collectors,
                    missing_collector_names=[],
                )

    def signal_data_complete(
        self,
        source,
        data_type,
        timestamp,
        *,
        forced: bool,
        attempts: int,
        missing_collectors: int,
        received_collectors: int,
        expected_collectors: int,
        missing_collector_names: list[str],
    ):
        """Signal that data is complete - can be customized"""
        ts_key = (data_type, timestamp)
        if ts_key not in self.completed_sources:
            self.completed_sources[ts_key] = set()
        if source in self.completed_sources[ts_key]:
            return
        self.completed_sources[ts_key].add(source)

        if forced:
            if ts_key not in self.forced_complete_sources:
                self.forced_complete_sources[ts_key] = set()
            self.forced_complete_sources[ts_key].add(source)
            print(
                "Force-complete applied: "
                f"source={source} data_type={data_type} timestamp={timestamp} "
                f"attempt={attempts} missing={missing_collectors} "
                f"received={received_collectors} expected={expected_collectors} "
                f"missing_collectors={missing_collector_names}"
            )

        self._write_done_marker_if_ready(data_type, timestamp)

    def clean_old_status(self):
        """Remove stale status records with data-type specific retention."""
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff_by_type = {
            'upd': now - self.upd_status_retention,
            'rib': now - self.rib_status_retention,
        }
        to_remove = []
        ts_remove = []
        for key in self.completion_status:
            source, data_type, timestamp = key
            try:
                ts_dt = datetime.datetime.strptime(timestamp, '%Y%m%d.%H%M').replace(tzinfo=datetime.timezone.utc)
                if ts_dt < cutoff_by_type.get(data_type, cutoff_by_type['rib']):
                    to_remove.append(key)
            except ValueError:
                continue
        for key in to_remove:
            del self.completion_status[key]

        for key in list(self.completed_sources.keys()):
            data_type, timestamp = key
            try:
                ts_dt = datetime.datetime.strptime(timestamp, '%Y%m%d.%H%M').replace(tzinfo=datetime.timezone.utc)
                if ts_dt < cutoff_by_type.get(data_type, cutoff_by_type['rib']):
                    ts_remove.append(key)
            except ValueError:
                continue

        for key in ts_remove:
            del self.completed_sources[key]
            if key in self.forced_complete_sources:
                del self.forced_complete_sources[key]
            if key in self.done_markers_written:
                self.done_markers_written.remove(key)

    async def download_rib_window(self):
        """Download RIB files in configured rolling window.

        Returns a tuple: (downloaded_any, reached_end_time).
        """
        start_time, end_time, reached_end_time = self._get_download_window('rib')
        if reached_end_time:
            return False, True

        self._force_complete_stale_timestamps('rib', start_time)
        downloaded_any = False

        for source in ('ripe', 'rv'):
            source_downloaded = await self._download_source_window(source, 'rib', start_time, end_time)
            downloaded_any = downloaded_any or source_downloaded

        # Download AS rank once
        date_str = end_time.strftime('%Y-%m-01')
        as_rank_file = os.path.join(self.dataset_dir, f'as_info.{end_time.strftime("%y%m")}.jsonl')
        self.download_as_rank(date_str, as_rank_file)
        
        return downloaded_any, False

    def download_as_rank(self, date_str, output_file):
        # Copied from download_history_data.py
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"AS Rank data already exists at {output_file}, skipping download.")
            return

        url = "https://api.asrank.caida.org/v2/graphql"
        page_size = 1000

        session = create_retry_session()

        # Get total count
        query = """{
            asns(first:%d, offset:%d) {
                totalCount
                pageInfo {
                    first
                    hasNextPage
                }
                edges {
                    node {
                        asn
                        asnName
                        rank
                        organization {
                            orgId
                            orgName
                        }
                        cliqueMember
                        seen
                        longitude
                        latitude
                        cone {
                            numberAsns
                            numberPrefixes
                            numberAddresses
                        }
                        country {
                            iso
                            name
                        }
                        asnDegree {
                            provider
                            peer
                            customer
                            total
                            transit
                            sibling
                        }
                        announcing {
                            numberPrefixes
                            numberAddresses
                        }
                    }
                }
            }
        }"""

        try:
            query_str = query % (page_size, 0)
            response = session.post(url, json={'query': query_str}, timeout=120, verify=False)
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                print(f"GraphQL Errors: {result['errors']}")
                return

            if "data" not in result or "asns" not in result["data"]:
                print(f"Unexpected response format: {result}")
                return

            data = result["data"]["asns"]
            total_count = data["totalCount"]
            edges = data["edges"]

            if not edges:
                print(f"No AS Rank data available for {date_str}")
                return

            total_pages = (total_count + page_size - 1) // page_size
            print(f"Total ASNs: {total_count}, total pages: {total_pages}")

        except Exception as e:
            print(f"Error getting total count: {str(e)}")
            return

        # Parallel download pages
        from concurrent.futures import ThreadPoolExecutor, as_completed
        all_nodes = set()

        def download_page(page):
            offset = page * page_size
            try:
                query_str = query % (page_size, offset)
                response = session.post(url, json={'query': query_str}, timeout=120, verify=False)
                response.raise_for_status()
                result = response.json()

                if "errors" in result:
                    print(f"GraphQL Errors for page {page}: {result['errors']}")
                    return []

                if "data" not in result or "asns" not in result["data"]:
                    print(f"Unexpected response format for page {page}: {result}")
                    return []

                data = result["data"]["asns"]
                edges = data["edges"]
                return [node["node"] for node in edges]
            except Exception as e:
                print(f"Error downloading page {page}: {str(e)}")
                return []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(download_page, page) for page in range(total_pages)]
            for future in as_completed(futures):
                nodes = future.result()
                for node in nodes:
                    all_nodes.add(json.dumps(node))

        # Write to file
        try:
            with open(output_file, "w") as f:
                for node_json in all_nodes:
                    f.write(node_json + "\n")
            print(f"Successfully downloaded AS Rank data for {date_str} to {output_file} ({len(all_nodes)} ASNs)")
        except Exception as e:
            print(f"Error writing to file: {str(e)}")
            if os.path.exists(output_file):
                os.remove(output_file)

    def cleanup_old_files(
        self,
        raw_data_retention_hours_upd: int | None = None,
        raw_data_retention_hours_rib: int | None = None,
        as_info_retention_days: int = 45,
    ):
        """Remove old files with separate UPD/RIB retention windows."""
        if raw_data_retention_hours_upd is None:
            raw_data_retention_hours_upd = self.upd_file_retention_hours
        if raw_data_retention_hours_rib is None:
            raw_data_retention_hours_rib = self.rib_file_retention_hours

        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff_raw_data = {
            'upd': now - datetime.timedelta(hours=raw_data_retention_hours_upd),
            'rib': now - datetime.timedelta(hours=raw_data_retention_hours_rib),
        }
        cutoff_as_info = now - datetime.timedelta(days=as_info_retention_days)
        removed_raw_dates = set()
        removed_as_info_dates = set()

        for data_type in ['upd', 'rib']:
            if data_type == 'upd':
                data_dir = self.upd_dir
            elif data_type == 'rib':
                data_dir = self.rib_dir
            else:
                continue
            if not os.path.exists(data_dir):
                continue

            for filename in os.listdir(data_dir):
                filepath = os.path.join(data_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        file_time = self._parse_bgp_file_timestamp(filename)
                        if file_time is None:
                            continue
                        if file_time < cutoff_raw_data.get(data_type, cutoff_raw_data['rib']):
                            os.remove(filepath)
                            removed_raw_dates.add(file_time.strftime('%Y%m%d.%H%M'))
                    except (ValueError, IndexError):
                        continue

        # Cleanup old AS info files (older than 2 months)
        for filename in os.listdir(self.dataset_dir):
            if filename.startswith('as_info.') and filename.endswith('.jsonl'):
                try:
                    parts = filename.split('.')
                    if len(parts) == 3:
                        date_part = parts[1]  # e.g., '2603'
                        if len(date_part) == 4:
                            yy = int(date_part[:2]) + 2000
                            mm = int(date_part[2:])
                            file_time = datetime.datetime(yy, mm, 1).replace(tzinfo=datetime.timezone.utc)
                            if file_time < cutoff_as_info:
                                filepath = os.path.join(self.dataset_dir, filename)
                                os.remove(filepath)
                                removed_as_info_dates.add(date_part)
                except (ValueError, IndexError):
                    continue

        if removed_raw_dates:
            dates_str = ', '.join(sorted(removed_raw_dates))
            print(f"Removed old raw data timestamps: {dates_str}")
        if removed_as_info_dates:
            dates_str = ', '.join(sorted(removed_as_info_dates))
            print(f"Removed old as_info month tags: {dates_str}")

    async def run_upd_loop(self):
        """Async loop for UPD downloads"""
        while True:
            downloaded_any, reached_end_time = await self.download_upd_window()
            if reached_end_time:
                print(f"UPD replay finished at end time: {self.end_time.isoformat()}")
                break
            if not downloaded_any:
                await asyncio.sleep(0)

    async def run_rib_loop(self):
        """Async loop for RIB downloads"""
        while True:
            downloaded_any, reached_end_time = await self.download_rib_window()
            if reached_end_time:
                print(f"RIB replay finished at end time: {self.end_time.isoformat()}")
                break
            if not downloaded_any:
                await asyncio.sleep(0)

    async def run(self):
        """Main async run method"""
        await asyncio.gather(
            self.run_upd_loop(),
            self.run_rib_loop()
        )

if __name__ == "__main__":
    import argparse

    init_program_log_from_env('BGP_DOWNLOADER_LOG_FILE')

    parser = argparse.ArgumentParser(description='Continuous BGP Data Downloader')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory for AS info and other metadata')
    parser.add_argument('--raw-data', type=str, required=True, help='Raw data directory for RIB and UPD files')
    parser.add_argument('--start-time', type=str, default=None, help='Initial end-time anchor for historical backfill. UPD advances in 20-minute windows; RIB completeness targets 4-hour slots for RouteViews (8-hour slots for RIPE). Default: current UTC time.')
    parser.add_argument('--end-time', type=str, required=True, help='Replay end time in UTC. Downloader exits automatically when this boundary is reached.')

    args = parser.parse_args()

    start_time = parse_start_time(args.start_time)
    end_time = parse_required_time_arg(args.end_time, '--end-time')
    downloader = BGPDataDownloader(
        args.dataset,
        args.raw_data,
        start_time=start_time,
        end_time=end_time,
    )
    asyncio.run(downloader.run())
