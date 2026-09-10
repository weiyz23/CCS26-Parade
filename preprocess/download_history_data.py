# download history data from RIPE RIS Archive for all the RRCs
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import requests
import bs4
import datetime
import time
import hashlib
import json
import urllib3
import gzip
import subprocess
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(CUR_DIR), 'dataset')
DIR_URL = 'https://data.ris.ripe.net/rrc{}/{}'

AVA_RRCS = [
    "00", "01",         # missing 02
    "03", "04", "05", "06", "07",      # missing 08, 09
    "10", "11", "12", "13", "14", "15", "16",  # missing 17
    "18", "19", "20", "21", "22", "23", "24", "25", "26"
]

# Create a session with retry mechanism
def create_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
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

# Cache management
def load_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache, cache_file):
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_progress(progress, progress_file):
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def get_cache_key(url):
    return hashlib.md5(url.encode()).hexdigest()

# Check file integrity
def verify_file_integrity(file_path, expected_size=None):
    if not os.path.exists(file_path):
        return False
    
    # Check file size
    if expected_size and os.path.getsize(file_path) != expected_size:
        return False
    
    # Check if file is empty or corrupted
    if os.path.getsize(file_path) == 0:
        return False
    
    # For gz files, check file header and verify CRC integrity
    if file_path.endswith('.gz'):
        try:
            # Check file header first
            with open(file_path, 'rb') as f:
                header = f.read(3)
                if header != b'\x1f\x8b\x08':
                    return False
            return True
        except (gzip.BadGzipFile, OSError, IOError):
            return False
    
    # For bz2 files, check file header
    if file_path.endswith('.bz2'):
        try:
            with open(file_path, 'rb') as f:
                header = f.read(3)
                if header != b'BZh':
                    return False
            return True
        except (OSError, IOError):
            return False
    
    return True

# Resume download
def download_with_resume(url: str, file_path: str, session=None, max_retries=5) -> bool:
    if session is None:
        session = create_retry_session()
    
    # Check existing file
    resume_pos = 0
    if os.path.exists(file_path):
        resume_pos = os.path.getsize(file_path)
    
    headers = {}
    if resume_pos > 0:
        headers['Range'] = f'bytes={resume_pos}-'
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, stream=True, timeout=30, verify=False)
            
            # Check if resume is supported
            if resume_pos > 0 and response.status_code not in [206, 416]:
                resume_pos = 0
                headers = {}
                response = session.get(url, headers=headers, stream=True, timeout=30, verify=False)
            
            if response.status_code == 416:  # Range not satisfiable
                return True
            
            response.raise_for_status()
            
            # Get total file size
            content_length = response.headers.get('content-length')
            if content_length:
                total_size = int(content_length)
                if resume_pos > 0:
                    total_size += resume_pos
            else:
                total_size = None
            
            # Write to file
            mode = 'ab' if resume_pos > 0 else 'wb'
            with open(file_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify file integrity
            if verify_file_integrity(file_path, total_size):
                return True
            else:
                if os.path.exists(file_path):
                    os.remove(file_path)
                resume_pos = 0
                headers = {}
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Waiting {wait_time:.2f} seconds before retry...")
                time.sleep(wait_time)
            continue
    
    print(f"Failed to download after {max_retries} attempts: {url}")
    return False

# download file using wget with retry and resume
def download_one_file(url: str, file_path: str) -> bool:
    # Check if file exists and is intact
    if verify_file_integrity(file_path):
        return True
    
    # Download with retry mechanism
    session = create_retry_session()
    return download_with_resume(url, file_path, session)

# Get file list (with cache)
def get_file_list_with_cache(rrc_dir: str, cache_file: str, force_refresh=False):
    cache = load_cache(cache_file)
    cache_key = get_cache_key(rrc_dir)
    
    # Check if cache exists and is not expired (24 hours)
    if not force_refresh and cache_key in cache:
        cache_data = cache[cache_key]
        cache_time = datetime.datetime.fromisoformat(cache_data['timestamp'])
        if datetime.datetime.now() - cache_time < datetime.timedelta(hours=24):
            ribs_files = [(name, datetime.datetime.fromisoformat(time_str)) 
                         for name, time_str in cache_data['ribs_files']]
            updates_files = [(name, datetime.datetime.fromisoformat(time_str)) 
                           for name, time_str in cache_data['updates_files']]
            return ribs_files, updates_files
    
    # Get new data
    session = create_retry_session()
    
    for attempt in range(3):
        try:
            r = session.get(rrc_dir, timeout=30, verify=False)
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
    
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    ribs_file_list = []
    updates_file_list = []
    
    for link in soup.find_all('a'):
        file_name = link.get('href')
        if file_name and file_name.endswith('.gz'):
            try:
                file_spilt = file_name.split('.')
                file_time = datetime.datetime.strptime(file_spilt[1] + file_spilt[2], '%Y%m%d%H%M')
                
                if file_name.startswith('bview'):
                    ribs_file_list.append((file_name, file_time))
                elif file_name.startswith('updates'):
                    updates_file_list.append((file_name, file_time))
            except:
                continue
    
    # Update cache - convert datetime object to string
    cache[cache_key] = {
        'ribs_files': [(name, time_obj.isoformat()) for name, time_obj in ribs_file_list],
        'updates_files': [(name, time_obj.isoformat()) for name, time_obj in updates_file_list],
        'timestamp': datetime.datetime.now().isoformat()
    }
    save_cache(cache, cache_file)
    
    return ribs_file_list, updates_file_list

def download_as_rank_data(date_str, output_file, *, strict=False):
    output_file = os.fspath(output_file)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        return

    # Calculate start and end dates for the first day of the month
    try:
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        month_start = dt.replace(day=1).strftime('%Y-%m-%d')
        month_end = dt.replace(day=2).strftime('%Y-%m-%d')
    except ValueError:
        if strict:
            raise
        print(f"Invalid date format: {date_str}")
        return

    url = "https://api.asrank.caida.org/v2/graphql"
    page_size = 1000
    offset = 0
    has_next_page = True
    
    print(f"Downloading AS Rank data for {month_start} (using window {month_start} to {month_end}) to {output_file}")
    
    session = create_retry_session()
    temporary_file = output_file + ".part"
    expected_count = None
    seen_asns = set()
    
    try:
        with open(temporary_file, "w") as f:
            while has_next_page:
                query = """{
                    asns(first:%d, offset:%d, dateStart:"%s", dateEnd:"%s") {
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
                }""" % (page_size, offset, month_start, month_end)
                
                response = session.post(url, json={'query': query}, timeout=60, verify=False)
                response.raise_for_status()
                result = response.json()
                
                if "errors" in result:
                    raise RuntimeError(f"GraphQL Errors: {result['errors']}")
                    
                if "data" not in result or "asns" not in result["data"]:
                    raise RuntimeError(f"Unexpected response format: {result}")

                data = result["data"]["asns"]
                edges = data["edges"]
                if strict:
                    page_total = int(data["totalCount"])
                    if expected_count is not None and page_total != expected_count:
                        raise RuntimeError("AS Rank totalCount changed during pagination")
                    expected_count = page_total
                
                if not edges:
                    if strict:
                        raise RuntimeError("AS Rank returned an empty page before completion")
                    break
                    
                for node in edges:
                    if strict:
                        asn = int(node["node"]["asn"])
                        if asn in seen_asns:
                            raise RuntimeError(f"Duplicate ASN in AS Rank pages: {asn}")
                        seen_asns.add(asn)
                    f.write(json.dumps(node["node"]) + "\n")
                    
                has_next_page = data["pageInfo"]["hasNextPage"]
                offset += len(edges)
        if strict and (not offset or offset != expected_count):
            raise RuntimeError(f"Incomplete AS Rank download: {offset}/{expected_count}")
        os.replace(temporary_file, output_file)
        print(f"Successfully downloaded AS Rank data to {output_file}")
        
    except Exception as e:
        print(f"Error downloading AS Rank data: {str(e)}")
        if os.path.exists(temporary_file):
            os.remove(temporary_file)
        if strict:
            raise
    finally:
        session.close()

# download the data for all the RRCs
def generate_task_for_one_duration(start_str: str, end_str: str, sub_dir_name: str, force_refresh=False):
    start_time = datetime.datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
    
    # check if the sub directory exists
    sub_dir = os.path.join(DATASET_DIR, sub_dir_name)
    rib_dir = os.path.join(sub_dir, 'rib')
    upd_dir = os.path.join(sub_dir, 'upd')
    cache_file = os.path.join(sub_dir, 'download_cache.json')

    for d in [sub_dir, rib_dir, upd_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    task_params_list = []
    
    for rrc in AVA_RRCS:
        rrc_dir = DIR_URL.format(rrc, start_time.strftime('%Y.%m'))
        
        try:
            ribs_file_list, updates_file_list = get_file_list_with_cache(rrc_dir, cache_file, force_refresh)
            
            if not ribs_file_list and not updates_file_list:
                continue
            
            # Process ribs files
            ribs_file_list = sorted(ribs_file_list, key=lambda x: x[1])
            
            # Download all RIB files within 24 hours before start_time
            rib_start_window = start_time - datetime.timedelta(hours=24)
            
            for ribs_file, file_time in ribs_file_list:
                if rib_start_window < file_time <= start_time:
                    url = rrc_dir + '/' + ribs_file
                    file_name = f'{ribs_file[:-3]}-{rrc}.gz'
                    path = os.path.join(rib_dir, file_name)
                    task_params_list.append((url, path))

            # Process updates files
            updates_file_list = sorted(updates_file_list, key=lambda x: x[1])
            filtered_updates = []
            for file_name, file_time in updates_file_list:
                if start_time <= file_time < end_time:
                    filtered_updates.append((file_name, file_time))
            
            for updates_file, _ in filtered_updates:
                url = rrc_dir + '/' + updates_file
                file_name = f'{updates_file[:-3]}-{rrc}.gz'
                path = os.path.join(upd_dir, file_name)
                task_params_list.append((url, path))
                
        except Exception as e:
            print(f"Error processing RRC {rrc}: {str(e)}")
            continue
    
    return task_params_list

def download_all_files(task_info, force_refresh=False):
    sub_dir_name = task_info[2]
    sub_dir = os.path.join(DATASET_DIR, sub_dir_name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    progress_file = os.path.join(sub_dir, 'download_progress.json')

    progress = load_progress(progress_file)
    task_key = f"{task_info[0]}_{task_info[1]}_{task_info[2]}"
    
    # Check for previous progress
    if task_key in progress and not force_refresh:
        completed_files = set(progress[task_key].get('completed', []))
    else:
        completed_files = set()
    
    task_params_list = generate_task_for_one_duration(*task_info, force_refresh=force_refresh)
    
    # Download AS Rank data
    try:
        start_time = datetime.datetime.strptime(task_info[0], '%Y-%m-%d %H:%M:%S')
        date_str = start_time.strftime('%Y-%m-%d')
        as_rank_file = os.path.join(sub_dir, 'as_info.jsonl')
        download_as_rank_data(date_str, as_rank_file)
    except Exception as e:
        print(f"Failed to initiate AS Rank download: {e}")

    # Filter completed files
    remaining_tasks = []
    for url, path in task_params_list:
        # Check if decompressed file exists
        uncompressed_path = None
        if path.endswith('.gz'):
            uncompressed_path = path[:-3]
        elif path.endswith('.bz2'):
            uncompressed_path = path[:-4]
            
        if uncompressed_path and os.path.exists(uncompressed_path) and os.path.getsize(uncompressed_path) > 0:
            completed_files.add(path)
            continue

        if path not in completed_files or not verify_file_integrity(path):
            remaining_tasks.append((url, path))
        else:
            completed_files.add(path)
    
    if not remaining_tasks:
        return
    
    # download the files parallelly
    failed_count = 0
    success_count = len(completed_files)
    total_count = len(task_params_list)
    
    with ThreadPoolExecutor(max_workers=8) as executor:  # Reduce concurrency to avoid overload
        futures = [executor.submit(download_one_file, url, path) for url, path in remaining_tasks]
        
        for future in as_completed(futures):
            if future.result():
                success_count += 1
                # Update progress
                completed_files.add(remaining_tasks[list(futures).index(future)][1])
                progress[task_key] = {
                    'completed': list(completed_files),
                    'total': total_count,
                    'last_update': datetime.datetime.now().isoformat()
                }
                save_progress(progress, progress_file)
            else:
                failed_count += 1
                
    print(f"Task completed: {success_count}/{total_count} successful, {failed_count} failed for task: {task_info}")
    
    if failed_count > 0:
        print(f"Failed to download {failed_count} files for the task: {task_info}")
        print("You can rerun the script to retry failed downloads.")

def decompress_directory(target_dir):
    print(f"\nStarting to decompress files in {target_dir}...")
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)
            uncompressed_path = None
            cmd = None
            
            if file.endswith('.gz'):
                uncompressed_path = file_path[:-3]
                cmd = ['gunzip', '-c', file_path]
            elif file.endswith('.bz2'):
                uncompressed_path = file_path[:-4]
                cmd = ['bunzip2', '-c', file_path]
            
            if uncompressed_path and cmd:
                if os.path.exists(uncompressed_path) and os.path.getsize(uncompressed_path) > 0:
                    continue
                try:
                    with open(uncompressed_path, 'wb') as f_out:
                        subprocess.run(cmd, stdout=f_out, check=True)
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to decompress {file_path}: {str(e)}")
                    if os.path.exists(uncompressed_path):
                        os.remove(uncompressed_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download RIPE RIS historical data')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Target dataset name (e.g., vodafone)')
    parser.add_argument('--start', type=str, required=True,
                       help='Start time (e.g., "2021-04-16 12:00:00")')
    parser.add_argument('--end', type=str, required=True, 
                       help='End time (e.g., "2021-04-16 15:00:00")')
    
    args = parser.parse_args()
    
    selected_task = (args.start, args.end, args.dataset)

    print(f"\nStarting task: {selected_task}")
    download_all_files(selected_task)
    print(f"Completed task download: {selected_task}")
    
    sub_dir = os.path.join(DATASET_DIR, selected_task[2])
    decompress_directory(sub_dir)
