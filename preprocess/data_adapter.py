# bgp_data_adapter.py
# Adapter for downloading BGP data from RIPE RIS and RouteViews

from abc import ABC, abstractmethod
import os
import random
import json
import requests
import datetime
import time
import urllib3
import re
import asyncio
import aiohttp
import gzip
import bz2
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

COMPLETENESS_POLICY_PATH = os.path.join(os.path.dirname(__file__), 'completeness_policy.json')


def _load_collectors_from_policy():
    with open(COMPLETENESS_POLICY_PATH, 'r', encoding='utf-8') as f:
        policy = json.load(f)

    try:
        ris = policy['sources']['ris']
        rv = policy['sources']['rv']
        ava_rrcs = [str(x) for x in ris['collector_ids']]
        collectors = [str(x) for x in rv['collector_ids']]
        ris_total = int(ris['collectors'])
        rv_total = int(rv['collectors'])
    except Exception as e:
        raise ValueError(f"Invalid collector config in {COMPLETENESS_POLICY_PATH}: {e}")

    if not ava_rrcs or not collectors:
        raise ValueError(f"collector_ids must be non-empty in {COMPLETENESS_POLICY_PATH}")
    if len(set(ava_rrcs)) != len(ava_rrcs) or len(set(collectors)) != len(collectors):
        raise ValueError(f"collector_ids must be unique in {COMPLETENESS_POLICY_PATH}")
    if len(ava_rrcs) != ris_total or len(collectors) != rv_total:
        raise ValueError(f"collector count mismatch in {COMPLETENESS_POLICY_PATH}")

    return ava_rrcs, collectors


AVA_RRCS, COLLECTORS = _load_collectors_from_policy()

FILE_LIST_CACHE_DIRNAME = ".file_list_cache"


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _cache_only_threshold(source: str, data_type: str) -> datetime.timedelta:
    dtype = data_type.lower()
    src = source.lower()
    if dtype == 'upd':
        return datetime.timedelta(hours=2)
    if dtype == 'rib' and src == 'ripe':
        return datetime.timedelta(hours=12)
    if dtype == 'rib' and src == 'rv':
        return datetime.timedelta(hours=4)
    return datetime.timedelta.max


def _should_use_cache_only(source: str, data_type: str, end_time: datetime.datetime) -> bool:
    end_time_utc = _ensure_utc(end_time)
    age = _utc_now() - end_time_utc
    return age > _cache_only_threshold(source, data_type)


def _cache_file_path(output_dir: str, cache_name: str) -> str:
    raw_data_dir = os.path.dirname(output_dir)
    cache_dir = os.path.join(raw_data_dir, FILE_LIST_CACHE_DIRNAME)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_name}.json")


def _load_cache_map(output_dir: str, cache_name: str):
    cache_path = _cache_file_path(output_dir, cache_name)
    if not os.path.exists(cache_path):
        return {}, cache_path
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data, cache_path
    except Exception:
        pass
    return {}, cache_path


def _save_cache_map(cache_map, cache_path: str):
    tmp_path = cache_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(cache_map, f, ensure_ascii=True, sort_keys=True)
    os.replace(tmp_path, cache_path)


def _serialize_file_list(file_list):
    out = []
    for file_name, file_time in file_list:
        out.append([file_name, _ensure_utc(file_time).strftime('%Y-%m-%dT%H:%M:%SZ')])
    return out


def _deserialize_file_list(payload):
    out = []
    if not isinstance(payload, list):
        return out
    for item in payload:
        if not isinstance(item, list) or len(item) != 2:
            continue
        file_name, ts_text = item
        try:
            file_time = datetime.datetime.fromisoformat(str(ts_text).replace('Z', '+00:00'))
        except Exception:
            continue
        out.append((str(file_name), _ensure_utc(file_time)))
    return out


def _iter_month_starts(start_time: datetime.datetime, end_time: datetime.datetime):
    current = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while current < end_time:
        yield current
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def _parse_archive_file_time(file_name: str):
    parts = file_name.split('.')
    if len(parts) < 4:
        return None
    try:
        return datetime.datetime.strptime(parts[-3] + parts[-2], '%Y%m%d%H%M').replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return None


def _build_local_task_path(output_dir: str, data_type: str, source_tag: str, collector: str, archive_file_name: str):
    parts = archive_file_name.split('.')
    if len(parts) < 4:
        return None
    date_str = parts[1]
    time_str = parts[2]
    ext = parts[-1]
    local_name = f'{data_type}.{source_tag}.{date_str}.{time_str}.{collector}.{ext}'
    return os.path.join(output_dir, local_name)


def _build_window_tasks(
    file_entries,
    *,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    url_prefix: str,
    output_dir: str,
    data_type: str,
    source_tag: str,
    collector: str,
    file_filter=None,
):
    tasks = []
    for file_name, file_time in sorted(file_entries, key=lambda x: x[1]):
        if not (start_time <= file_time < end_time):
            continue
        if file_filter is not None and not file_filter(file_time):
            continue

        path = _build_local_task_path(output_dir, data_type, source_tag, collector, file_name)
        if path is None:
            continue
        tasks.append((url_prefix + file_name, path))
    return tasks


def _get_file_list_cached_common(
    output_dir: str,
    cache_name: str,
    key: str,
    refresh: bool,
    cache_only: bool,
    fetch_fn,
):
    cache_map, cache_path = _load_cache_map(output_dir, cache_name)
    cached_entry = cache_map.get(key)

    if cache_only and cached_entry:
        return (
            _deserialize_file_list(cached_entry.get('ribs', [])),
            _deserialize_file_list(cached_entry.get('updates', [])),
        )

    if not refresh and cached_entry:
        return (
            _deserialize_file_list(cached_entry.get('ribs', [])),
            _deserialize_file_list(cached_entry.get('updates', [])),
        )

    try:
        ribs, updates = fetch_fn()
        cache_map[key] = {
            'updated_at': _utc_now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'ribs': _serialize_file_list(ribs),
            'updates': _serialize_file_list(updates),
        }
        _save_cache_map(cache_map, cache_path)
        return ribs, updates
    except Exception:
        if cached_entry:
            return (
                _deserialize_file_list(cached_entry.get('ribs', [])),
                _deserialize_file_list(cached_entry.get('updates', [])),
            )
        raise

def create_retry_session(retries=5, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry( 
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

class BGPDataAdapter(ABC):
    @abstractmethod
    def generate_tasks(self, start_time: datetime.datetime, end_time: datetime.datetime, data_type: str, output_dir: str) -> list:
        """
        Generate download tasks for the given time window and data type.
        
        :param start_time: Start time
        :param end_time: End time
        :param data_type: 'rib' or 'upd'
        :param output_dir: Output directory
        :param cache_file: Cache file path
        :param force_refresh: Force refresh cache
        :return: List of (url, file_path) tuples
        """
        pass

class RIPEAdapter(BGPDataAdapter):
    def __init__(self):
        self.dir_url = 'https://data.ris.ripe.net/rrc{}/{}'
        self.ava_rrcs = AVA_RRCS
        self.cache_name = 'ripe_file_lists_v1'

    def get_file_list(self, rrc_dir: str, request_timeout: int = 30):
        session = create_retry_session()
        
        for attempt in range(3):
            try:
                r = session.get(rrc_dir, timeout=request_timeout, verify=False)
                if r.status_code == 404:
                    return [], []
                r.raise_for_status()
                break
            except Exception as e:
                print(f"Failed to fetch file list (attempt {attempt + 1}): {str(e)}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        # Use regex to extract .gz and .bz2 file links
        file_links = re.findall(r'href="([^"]*\.(?:gz|bz2))"', r.text)
        
        ribs_file_list = []
        updates_file_list = []
        
        for file_name in file_links:
            file_time = _parse_archive_file_time(file_name)
            if file_time is None:
                continue

            if file_name.startswith(('bview.', 'rib.')):
                ribs_file_list.append((file_name, file_time))
            elif file_name.startswith('updates.'):
                updates_file_list.append((file_name, file_time))
        
        return ribs_file_list, updates_file_list

    def _get_file_list_cached(
        self,
        rrc_dir: str,
        output_dir: str,
        refresh: bool,
        cache_only: bool,
        request_timeout: int = 30,
    ):
        return _get_file_list_cached_common(
            output_dir=output_dir,
            cache_name=self.cache_name,
            key=rrc_dir,
            refresh=refresh,
            cache_only=cache_only,
            fetch_fn=lambda: self.get_file_list(rrc_dir, request_timeout=request_timeout),
        )

    def generate_tasks(self, start_time: datetime.datetime, end_time: datetime.datetime, data_type: str, output_dir: str) -> list:
        task_params_list = []
        cache_only = _should_use_cache_only('ripe', data_type, end_time)
        refresh = not cache_only
        is_rib = data_type.lower() == 'rib'
        list_timeout = 90 if is_rib else 30
        
        for month in _iter_month_starts(start_time, end_time):
            for rrc in self.ava_rrcs:
                rrc_dir = self.dir_url.format(rrc, month.strftime('%Y.%m'))
                
                try:
                    ribs_file_list, updates_file_list = self._get_file_list_cached(
                        rrc_dir,
                        output_dir,
                        refresh=refresh,
                        cache_only=cache_only,
                        request_timeout=list_timeout,
                    )
                    
                    if not ribs_file_list and not updates_file_list:
                        continue

                    selected_entries = ribs_file_list if is_rib else updates_file_list
                    task_params_list.extend(
                        _build_window_tasks(
                            selected_entries,
                            start_time=start_time,
                            end_time=end_time,
                            url_prefix=rrc_dir + '/',
                            output_dir=output_dir,
                            data_type='rib' if is_rib else 'upd',
                            source_tag='ris',
                            collector=rrc,
                        )
                    )
                            
                except Exception as e:
                    print(f"Error processing RRC {rrc}: {str(e)}")
                    continue
        
        return task_params_list

class RouteViewsAdapter(BGPDataAdapter):
    def __init__(self):
        self.collectors = COLLECTORS
        self.dir_url = 'https://archive.routeviews.org/{}/bgpdata/{}/'
        self.cache_name = 'routeviews_file_lists_v1'

    def get_file_list(
        self,
        dir_url: str,
        *,
        include_ribs: bool = True,
        include_updates: bool = True,
        request_timeout: int = 30,
    ):
        session = create_retry_session()

        def fetch_archive_list(archive_url: str, archive_name: str):
            result = []
            try:
                response = session.get(archive_url, timeout=request_timeout, verify=False)
                if response.status_code != 200:
                    return result

                file_links = re.findall(r'href="([^"]*\.(?:gz|bz2))"', response.text)
                for file_name in file_links:
                    file_time = _parse_archive_file_time(file_name)
                    if file_time is None:
                        continue
                    result.append((file_name, file_time))
            except Exception as e:
                print(f"Failed to fetch {archive_name} files: {str(e)}")
            return result

        ribs_file_list = fetch_archive_list(dir_url + 'RIBS/', 'RIBS') if include_ribs else []
        updates_file_list = fetch_archive_list(dir_url + 'UPDATES/', 'UPDATES') if include_updates else []

        return ribs_file_list, updates_file_list

    def _get_file_list_cached(
        self,
        dir_url: str,
        output_dir: str,
        refresh: bool,
        cache_only: bool,
        *,
        include_ribs: bool,
        include_updates: bool,
        request_timeout: int = 30,
    ):
        # A cached UPDATE-only listing says nothing about the month's RIBs.
        # Keep each requested scope separate, including when reading old archives.
        cache_key = f"{dir_url}|ribs={int(include_ribs)}|updates={int(include_updates)}"
        return _get_file_list_cached_common(
            output_dir=output_dir,
            cache_name=self.cache_name,
            key=cache_key,
            refresh=refresh,
            cache_only=cache_only,
            fetch_fn=lambda: self.get_file_list(
                dir_url,
                include_ribs=include_ribs,
                include_updates=include_updates,
                request_timeout=request_timeout,
            ),
        )

    def generate_tasks(self, start_time: datetime.datetime, end_time: datetime.datetime, data_type: str, output_dir: str) -> list:
        task_params_list = []
        cache_only = _should_use_cache_only('rv', data_type, end_time)
        refresh = not cache_only
        is_rib = data_type.lower() == 'rib'
        list_timeout = 90 if is_rib else 30
        
        for month in _iter_month_starts(start_time, end_time):
            for collector in self.collectors:
                dir_url = self.dir_url.format(collector, month.strftime('%Y.%m'))
                
                try:
                    ribs_file_list, updates_file_list = self._get_file_list_cached(
                        dir_url,
                        output_dir,
                        refresh=refresh,
                        cache_only=cache_only,
                        include_ribs=is_rib,
                        include_updates=not is_rib,
                        request_timeout=list_timeout,
                    )

                    selected_entries = ribs_file_list if is_rib else updates_file_list
                    url_prefix = dir_url + ('RIBS/' if is_rib else 'UPDATES/')
                    rib_slot_filter = (lambda ft: ft.minute == 0 and (ft.hour % 4) == 0) if is_rib else None
                    task_params_list.extend(
                        _build_window_tasks(
                            selected_entries,
                            start_time=start_time,
                            end_time=end_time,
                            url_prefix=url_prefix,
                            output_dir=output_dir,
                            data_type='rib' if is_rib else 'upd',
                            source_tag='rv',
                            collector=collector,
                            file_filter=rib_slot_filter,
                        )
                    )
                            
                except Exception as e:
                    print(f"Error processing RouteViews collector {collector}: {str(e)}")
                    continue
        
        return task_params_list
    
# Shared utility functions
def _safe_remove(path: str | None):
    if path and os.path.exists(path):
        os.remove(path)


def _decompress_to_output(compressed_path: str, output_path: str, opener):
    tmp_output_path = output_path + '.tmp'
    _safe_remove(tmp_output_path)
    with opener(compressed_path, 'rb') as f_in, open(tmp_output_path, 'wb') as f_out:
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)
    os.replace(tmp_output_path, output_path)


def _expected_total_size_from_headers(headers, resume_pos: int):
    content_range = headers.get('Content-Range') if headers else None
    if content_range:
        # Example: bytes 100-199/500
        try:
            total_text = content_range.split('/')[-1]
            if total_text and total_text != '*':
                return int(total_text)
        except (ValueError, IndexError):
            pass

    content_length = headers.get('Content-Length') if headers else None
    if content_length:
        try:
            length = int(content_length)
            return length + resume_pos if resume_pos > 0 else length
        except ValueError:
            return None
    return None


def _resolve_uncompressed_path(file_path: str):
    if file_path.endswith('.gz'):
        return file_path[:-3]
    if file_path.endswith('.bz2'):
        return file_path[:-4]
    return None


async def decompress_file_async(file_path, uncompressed_path=None):
    """
    Decompress a single file (.gz or .bz2) asynchronously.
    Returns True if decompression succeeds, and removes the compressed file.
    """
    if uncompressed_path is None:
        if file_path.endswith('.gz'):
            uncompressed_path = file_path[:-3]
        elif file_path.endswith('.bz2'):
            uncompressed_path = file_path[:-4]

    if uncompressed_path:
        try:
            if file_path.endswith('.gz'):
                await asyncio.to_thread(_decompress_to_output, file_path, uncompressed_path, gzip.open)
            elif file_path.endswith('.bz2'):
                await asyncio.to_thread(_decompress_to_output, file_path, uncompressed_path, bz2.open)
            else:
                return False

            _safe_remove(file_path)
            return True
        except Exception as e:
            print(f"Failed to decompress {file_path}: {str(e)}")
            _safe_remove(uncompressed_path)
            return False
    return False

async def download_one_file_async(
    session,
    url: str,
    file_path: str,
    request_timeout: aiohttp.ClientTimeout | None = None,
    max_retries: int = 5,
    retry_wait_cap_sec: float = 20.0,
    retry_jitter_sec: float = 2.0,
) -> str:
    """
    Download a file with integrity check and decompression asynchronously.
    Returns:
    - "downloaded": File was successfully downloaded and processed.
    - "skipped": File already exists and is valid, no download needed.
    - "failed": Download or processing failed.
    """
    uncompressed_path = _resolve_uncompressed_path(file_path)

    if uncompressed_path and os.path.exists(uncompressed_path) and os.path.getsize(uncompressed_path) > 0:
        # Uncompressed exists and no compressed file, assume valid
        return "skipped"

    # Download with retries using aiohttp.
    for attempt in range(max_retries):
        temp_path = file_path + '.part'
        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # Recover from stale/unwritable partial files (e.g., left by previous runs).
            if os.path.exists(temp_path) and not os.access(temp_path, os.W_OK):
                _safe_remove(temp_path)

            resume_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'

            req_timeout = request_timeout or aiohttp.ClientTimeout(total=3600, connect=60, sock_read=180)
            async with session.get(url, headers=headers, timeout=req_timeout) as response:
                if resume_pos > 0 and response.status == 200:
                    _safe_remove(temp_path)
                    resume_pos = 0
                    headers = {}
                    async with session.get(url, headers=headers, timeout=req_timeout) as response2:
                        response2.raise_for_status()
                        expected_size = _expected_total_size_from_headers(response2.headers, resume_pos)
                        with open(temp_path, 'wb') as f:
                            async for chunk in response2.content.iter_chunked(8192):
                                if chunk:
                                    f.write(chunk)
                elif response.status == 416:
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        os.replace(temp_path, file_path)
                        if not uncompressed_path or await decompress_file_async(file_path, uncompressed_path):
                            return "downloaded"
                    return "failed"
                else:
                    response.raise_for_status()
                    expected_size = _expected_total_size_from_headers(response.headers, resume_pos)
                    mode = 'ab' if resume_pos > 0 else 'wb'
                    with open(temp_path, mode) as f:
                        async for chunk in response.content.iter_chunked(8192):
                            if chunk:
                                f.write(chunk)

            written_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            if expected_size is not None and written_size != expected_size:
                raise IOError(f"Size mismatch: expected={expected_size}, got={written_size}")
            if written_size <= 0:
                raise IOError("Empty payload")

            os.replace(temp_path, file_path)

            # After download, check if it's compressed and decompress if needed
            if not uncompressed_path or await decompress_file_async(file_path, uncompressed_path):
                return "downloaded"
            raise IOError("Decompression failed")

        except Exception as e:
            _safe_remove(temp_path)
            _safe_remove(file_path)
            if uncompressed_path:
                _safe_remove(uncompressed_path)

            if attempt < max_retries - 1:
                base_wait = min(retry_wait_cap_sec, 1.5 * (2 ** attempt))
                wait_time = min(retry_wait_cap_sec, base_wait + random.uniform(0.0, retry_jitter_sec))
                print(
                    f"Retry {attempt + 1}/{max_retries} for {url} after {wait_time:.1f}s: {type(e).__name__}: {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                print(f"Failed to download after {max_retries} attempts: {url} - Last error: {str(e)}")
                return "failed"
    
    return "failed"

def _build_download_profile(profile: str):
    normalized = (profile or "default").strip().lower()
    if normalized in ("replay", "offline"):
        # One-shot historical replay prefers higher fan-out than online mode.
        return {
            "max_workers": 16,
            "connector_limit_per_host": 8,
            "connector_limit": 32,
            "request_timeout": aiohttp.ClientTimeout(total=3600, connect=90, sock_read=900),
            "session_timeout": aiohttp.ClientTimeout(total=3600, connect=90, sock_read=900),
            "force_close": False,
            "max_retries": 6,
            "retry_wait_cap_sec": 20.0,
            "retry_jitter_sec": 1.5,
        }
    if normalized == "rib":
        # Large snapshots benefit from lower parallelism and much looser read/connect timeouts.
        return {
            "max_workers": 4,
            "connector_limit_per_host": 2,
            "connector_limit": 8,
            "request_timeout": aiohttp.ClientTimeout(total=7200, connect=300, sock_read=1800),
            "session_timeout": aiohttp.ClientTimeout(total=7200, connect=300, sock_read=1800),
            "force_close": True,
            "max_retries": 8,
            "retry_wait_cap_sec": 20.0,
            "retry_jitter_sec": 1.5,
        }
    if normalized == "upd":
        return {
            "max_workers": 1,
            "connector_limit_per_host": 1,
            "connector_limit": 2,
            "request_timeout": aiohttp.ClientTimeout(total=1200, connect=60, sock_read=240),
            "session_timeout": aiohttp.ClientTimeout(total=1200, connect=60, sock_read=240),
            "force_close": False,
            "max_retries": 5,
            "retry_wait_cap_sec": 12.0,
            "retry_jitter_sec": 1.0,
        }

    return {
        "max_workers": 5,
        "connector_limit_per_host": 2,
        "connector_limit": 8,
        "request_timeout": aiohttp.ClientTimeout(total=1800, connect=60, sock_read=600),
        "session_timeout": aiohttp.ClientTimeout(total=1800, connect=60, sock_read=600),
        "force_close": False,
        "max_retries": 5,
        "retry_wait_cap_sec": 15.0,
        "retry_jitter_sec": 1.0,
    }

async def download_files_parallel_async(
    task_list: list,
    max_workers: int = 5,
    profile: str = "default",
):
    """
    Download files in parallel asynchronously.
    Returns:
    - successful_paths: newly downloaded files
    - failed_paths: failed downloads
    - skipped_paths: files already present and considered valid
    """
    cfg = _build_download_profile(profile)
    effective_workers = min(max_workers, cfg["max_workers"])
    semaphore = asyncio.Semaphore(effective_workers)  # Limit concurrent downloads

    async def download_with_semaphore(url, path, session):
        async with semaphore:
            return await download_one_file_async(
                session,
                url,
                path,
                request_timeout=cfg["request_timeout"],
                max_retries=cfg["max_retries"],
                retry_wait_cap_sec=cfg["retry_wait_cap_sec"],
                retry_jitter_sec=cfg["retry_jitter_sec"],
            )

    timeout = cfg["session_timeout"]
    connector_kwargs = {
        "limit_per_host": cfg["connector_limit_per_host"],
        "limit": cfg["connector_limit"],
        "force_close": cfg["force_close"],
        "enable_cleanup_closed": True,
    }
    if not cfg["force_close"]:
        connector_kwargs["keepalive_timeout"] = 60
    connector = aiohttp.TCPConnector(**connector_kwargs)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [download_with_semaphore(url, path, session) for url, path in task_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_paths = []
    failed_paths = []
    skipped_paths = []

    for (url, path), result in zip(task_list, results):
        if isinstance(result, Exception):
            print(f"Error downloading {url}: {result}")
            failed_paths.append(path)
        elif result == "downloaded":
            successful_paths.append(path)
        elif result == "failed":
            failed_paths.append(path)
        elif result == "skipped":
            skipped_paths.append(path)

    return successful_paths, failed_paths, skipped_paths
