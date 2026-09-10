#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <arpa/inet.h>
#include <functional>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <tuple>
#include <ctime>
#include <cstdlib>
#include <malloc.h>
#include "helpers.hpp"
#include "path_extractor.hpp" // Changed: Include shared logic

extern "C" {
#include <bgpstream.h>
}

using namespace std;
namespace fs = filesystem;

static bool should_process_rib_file(const string& fname) {
    if (fname.empty()) {
        return false;
    }
    if (fname.size() > 5 && fname.rfind(".done") == (fname.size() - 5)) {
        return false;
    }
    if (fname.size() > 4 && fname.rfind(".tmp") == (fname.size() - 4)) {
        return false;
    }
    if (fname.size() > 5 && fname.rfind(".part") == (fname.size() - 5)) {
        return false;
    }

    // Support both legacy RouteViews names (bview.*) and normalized downloader names (rib.*).
    if (fname.rfind("bview.", 0) == 0 || fname.rfind("rib.", 0) == 0) {
        return true;
    }
    return false;
}

static bool should_process_upd_file(const string& fname) {
    if (fname.empty()) {
        return false;
    }
    if (fname.size() > 5 && fname.rfind(".done") == (fname.size() - 5)) {
        return false;
    }
    if (fname.size() > 4 && fname.rfind(".tmp") == (fname.size() - 4)) {
        return false;
    }
    if (fname.size() > 5 && fname.rfind(".part") == (fname.size() - 5)) {
        return false;
    }
    return fname.rfind("updates.", 0) == 0 || fname.rfind("upd.", 0) == 0;
}

// 1. Define a 128-bit struct to store the result
struct PropFingerprint {
    uint64_t low;  // Low 64 bits
    uint64_t high; // High 64 bits
};

// --- Data Structures ---

// Global Maps
unordered_map<string, int32_t> prefix_to_id;
vector<string> id_to_prefix;

// Vector of AS Pairs to represent a path
// using PathVector = vector<ASPair>; // Removed in favor of FixedPath

// Path Hash -> ID (Using BGPStream's 32-bit hash combined with origin AS)
unordered_map<uint64_t, int32_t> path_hash_to_id;

// Optimized Storage: Fixed arrays to avoid fragmentation and allow memcpy 
struct FixedPath {
    ASPair pairs[AS_PAIR_SIZE];
};
vector<FixedPath> id_to_fixed_path;
vector<uint8_t> id_to_path_len;
// Linear AS paths are retained once per path ID for supplemental simulations.
// The normal training/profile pipeline continues to use the compact FixedPath.
vector<vector<uint32_t>> id_to_linear_path;
vector<uint8_t> id_to_path_is_strict;

// Compact (prefix_id, path_id) keys.  The previous std::set representation
// allocated one tree node per route and could exceed host memory on large
// historical snapshots.  Append-only uint64_t vectors followed by in-place
// sort/unique preserve exactly the same deduplication semantics at a fraction
// of the memory cost.
vector<uint64_t> rib_entry_keys;
vector<uint64_t> upd_entry_keys;

static uint64_t make_entry_key(int32_t prefix_id, int32_t path_id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(prefix_id)) << 32)
        | static_cast<uint32_t>(path_id);
}

static pair<int32_t, int32_t> decode_entry_key(uint64_t key) {
    return {
        static_cast<int32_t>(static_cast<uint32_t>(key >> 32)),
        static_cast<int32_t>(static_cast<uint32_t>(key)),
    };
}

static void sort_unique_keys(vector<uint64_t>& keys) {
    sort(keys.begin(), keys.end());
    keys.erase(unique(keys.begin(), keys.end()), keys.end());
}

unordered_map<uint32_t, int32_t> asn_to_id;
vector<uint32_t> id_to_asn;

enum class PathSource : uint8_t {
    RIB = 0,
    UPD = 1,
};

struct HopSource {
    uint8_t hop;
    PathSource source;
};

// Prefix -> (AS_ID -> MinHop)
vector<unordered_map<int32_t, HopSource>> prefix_hops;
// Prefix -> list of (u_as_id, v_as_id, depth)
vector<vector<tuple<int32_t, int32_t, uint8_t>>> prefix_edges;

// --- Helpers ---

int32_t get_or_create_prefix_id(const string& prefix) {
    auto it = prefix_to_id.find(prefix);
    if (it != prefix_to_id.end()) {
        return it->second;
    }
    int32_t id = prefix_to_id.size();
    prefix_to_id[prefix] = id;
    id_to_prefix.push_back(prefix);
    // Resize prefix_hops to accommodate new prefix
    if (prefix_hops.size() <= id) {
        prefix_hops.resize(id + 1);
    }
    // Resize prefix_edges
    if (prefix_edges.size() <= id) {
        prefix_edges.resize(id + 1);
    }
    return id;
}

int32_t get_or_create_asn_id(uint32_t asn) {
    auto it = asn_to_id.find(asn);
    if (it != asn_to_id.end()) {
        return it->second;
    }
    int32_t id = asn_to_id.size();
    asn_to_id[asn] = id;
    id_to_asn.push_back(asn);
    return id;
}

// Helper to process AS Path from BGPStream
// Returns the number of pairs extracted into the fixed buffer
bool extract_as_path_pairs(bgpstream_elem_t* elem, FixedPath& fp, uint8_t& count) {
    if (!elem) return false;
    if (extract_as_path_info(elem, fp.pairs, count)) {
        return true;
    }
    return false;
}

// Helper to process AS Path from BGPStream
// Returns a vector of ASNs in the path (Original for fingerprint)
vector<uint32_t> extract_as_path(bgpstream_as_path_t* path) {
    vector<uint32_t> asns;
    if (!path) return asns;
    
    bgpstream_as_path_iter_t iter;
    bgpstream_as_path_iter_reset(&iter);
    bgpstream_as_path_seg_t* seg;
    
    uint32_t last_asn = 0;
    bool first = true;

    while ((seg = bgpstream_as_path_get_next_seg(path, &iter)) != NULL) {
        if (seg->type == BGPSTREAM_AS_PATH_SEG_ASN) {
            bgpstream_as_path_seg_asn_t* asn_seg = (bgpstream_as_path_seg_asn_t*)seg;
            if (first || asn_seg->asn != last_asn) {
                asns.push_back(asn_seg->asn);
                last_asn = asn_seg->asn;
                first = false;
            }
        } else {
            // Handle sets/sequences by iterating
            bgpstream_as_path_seg_set_t* set_seg = (bgpstream_as_path_seg_set_t*)seg;
            for (int i = 0; i < set_seg->asn_cnt; ++i) {
                if (first || set_seg->asn[i] != last_asn) {
                    asns.push_back(set_seg->asn[i]);
                    last_asn = set_seg->asn[i];
                    first = false;
                }
            }
        }
    }
    return asns;
}

bool is_strict_linear_as_path(bgpstream_as_path_t* path) {
    if (!path) return false;
    bgpstream_as_path_iter_t iter;
    bgpstream_as_path_iter_reset(&iter);
    bgpstream_as_path_seg_t* seg;
    while ((seg = bgpstream_as_path_get_next_seg(path, &iter)) != NULL) {
        if (seg->type != BGPSTREAM_AS_PATH_SEG_ASN) {
            return false;
        }
    }
    return true;
}

// Core logic to process a single entry
void process_entry(const string& prefix_str, const vector<uint32_t>& as_path, PathSource source) {
    if (as_path.empty()) return;

    int32_t p_id = get_or_create_prefix_id(prefix_str);
    
    // Calculate hops and extract edges
    // Path: [AS1, AS2, AS3, Origin]
    // Hops: AS1->3, AS2->2, AS3->1, Origin->0    
    int path_len = as_path.size();
    
    // Handle origin AS separately (no edges)
    // Hop is 0 for origin AS
    {
        uint32_t origin_as = as_path[path_len - 1];
        int32_t as_id = get_or_create_asn_id(origin_as);
        auto& hops_map = prefix_hops[p_id];
        auto it = hops_map.find(as_id);
        if (it == hops_map.end()) {
            hops_map[as_id] = {0, source};
        } else if (source == PathSource::RIB && it->second.source != PathSource::RIB) {
            it->second = {0, source};
        } else if (source == it->second.source && 0 < it->second.hop) {
            it->second.hop = 0;
        }
    }
    
    // Handle other ASes and extract edges
    for (int i = 0; i < path_len - 1; ++i) {
        uint8_t hop = (uint8_t)(path_len - 1 - i);
        uint32_t asn = as_path[i];
        
        int32_t as_id = get_or_create_asn_id(asn);
        // RIB has higher priority; within same source keep smaller hop.
        auto& hops_map = prefix_hops[p_id];
        auto it = hops_map.find(as_id);
        if (it == hops_map.end()) {
            hops_map[as_id] = {hop, source};
        } else {
            if (source == PathSource::RIB && it->second.source != PathSource::RIB) {
                it->second = {hop, source};
            } else if (source == it->second.source && hop < it->second.hop) {
                it->second.hop = hop;
            }
        }

        // Extract edges
        int32_t u_id = get_or_create_asn_id(as_path[i + 1]);
        prefix_edges[p_id].emplace_back(u_id, as_id, hop - 1);
    }
}

// --- File Processors ---
bool process_mrt_file(const string& filepath) {
    const size_t initial_entries = rib_entry_keys.size();
    bgpstream_t *bs = bgpstream_create();
    if (!bs) return false;

    bgpstream_set_data_interface(bs, BGPSTREAM_DATA_INTERFACE_SINGLEFILE);
    bgpstream_set_data_interface_option(bs, bgpstream_get_data_interface_option_by_name(bs, BGPSTREAM_DATA_INTERFACE_SINGLEFILE, "rib-file"), filepath.c_str());
    
    if (bgpstream_start(bs) < 0) {
        bgpstream_destroy(bs);
        return false;
    }

    bgpstream_record_t *rec;
    int read_status;
    bool corrupted = false;
    while ((read_status = bgpstream_get_next_record(bs, &rec)) > 0) {
        if (rec->status == BGPSTREAM_RECORD_STATUS_CORRUPTED_SOURCE ||
            rec->status == BGPSTREAM_RECORD_STATUS_CORRUPTED_RECORD) {
            corrupted = true;
        }
        if (rec->status != BGPSTREAM_RECORD_STATUS_VALID_RECORD) continue;
        bgpstream_elem_t *elem;
        while (bgpstream_record_get_next_elem(rec, &elem)) {
            if (elem->type == BGPSTREAM_ELEM_TYPE_RIB) {
                uint32_t origin_as = 0;
                int origin_result = bgpstream_as_path_get_origin_val(elem->as_path, &origin_as);
                if (origin_result != 0) {
                    continue; // Skip invalid AS paths
                }

                // Parse prefix string
                char prefix_str[INET6_ADDRSTRLEN + 5];
                bgpstream_pfx_snprintf(prefix_str, sizeof(prefix_str), &elem->prefix);
                string pfx_str(prefix_str);
                // Get Prefix ID
                int32_t p_id = get_or_create_prefix_id(pfx_str);

                // Utilize BGPStream hash combined with origin AS to find existing path ID
                uint64_t path_hash = ((uint64_t)origin_as << 32) | bgpstream_as_path_hash(elem->as_path);
                int32_t path_id;
                
                auto hash_it = path_hash_to_id.find(path_hash);
                if (hash_it != path_hash_to_id.end()) {
                    path_id = hash_it->second;
                } else {
                    FixedPath fp;
                    uint8_t cnt;
                    if(extract_as_path_pairs(elem, fp, cnt) == false || cnt == 0) {
                        continue; // Skip invalid or empty paths
                    }
                    path_id = id_to_path_len.size();
                    id_to_fixed_path.push_back(fp);
                    id_to_path_len.push_back(cnt);
                    id_to_linear_path.emplace_back();
                    id_to_path_is_strict.push_back(0);
                    path_hash_to_id[path_hash] = path_id;
                }

                // Extract linear AS path only when needed
                vector<uint32_t> as_path = extract_as_path(elem->as_path);
                if (!as_path.empty()) {
                    const bool is_strict = is_strict_linear_as_path(elem->as_path);
                    if (id_to_linear_path[path_id].empty()) {
                        id_to_linear_path[path_id] = as_path;
                    }
                    if (is_strict) {
                        id_to_path_is_strict[path_id] = 1;
                    }
                    rib_entry_keys.push_back(make_entry_key(p_id, path_id));
                }
            }
        }
    }
    bgpstream_destroy(bs);
    return !corrupted && read_status == 0 && rib_entry_keys.size() > initial_entries;
}

void process_upd_file(const string& filepath) {
    if (!fs::exists(filepath) || !fs::is_regular_file(filepath)) {
        fprintf(stderr, "Warning: UPD file is not accessible, skip: %s\n", filepath.c_str());
        return;
    }

    bgpstream_t *bs = bgpstream_create();
    if (!bs) return;

    bgpstream_set_data_interface(bs, BGPSTREAM_DATA_INTERFACE_SINGLEFILE);
    bgpstream_data_interface_option_t* upd_opt =
        bgpstream_get_data_interface_option_by_name(bs, BGPSTREAM_DATA_INTERFACE_SINGLEFILE, "upd-file");
    if (!upd_opt) {
        fprintf(stderr, "Warning: BGPStream singlefile option 'upd-file' unavailable, skip: %s\n", filepath.c_str());
        bgpstream_destroy(bs);
        return;
    }
    bgpstream_set_data_interface_option(bs, upd_opt, filepath.c_str());

    if (bgpstream_start(bs) < 0) {
        fprintf(stderr, "Warning: bgpstream_start failed for UPD file: %s\n", filepath.c_str());
        bgpstream_destroy(bs);
        return;
    }

    bgpstream_record_t *rec;
    while (bgpstream_get_next_record(bs, &rec) > 0) {
        if (rec->status != BGPSTREAM_RECORD_STATUS_VALID_RECORD) continue;
        bgpstream_elem_t *elem;
        while (bgpstream_record_get_next_elem(rec, &elem)) {
            if (elem->type == BGPSTREAM_ELEM_TYPE_WITHDRAWAL) {
                continue;
            }

            uint32_t origin_as = 0;
            int origin_result = bgpstream_as_path_get_origin_val(elem->as_path, &origin_as);
            if (origin_result != 0) {
                continue;
            }

            char prefix_str[INET6_ADDRSTRLEN + 5];
            bgpstream_pfx_snprintf(prefix_str, sizeof(prefix_str), &elem->prefix);
            string pfx_str(prefix_str);
            int32_t p_id = get_or_create_prefix_id(pfx_str);

            uint64_t path_hash = ((uint64_t)origin_as << 32) | bgpstream_as_path_hash(elem->as_path);
            int32_t path_id;

            auto hash_it = path_hash_to_id.find(path_hash);
            if (hash_it != path_hash_to_id.end()) {
                path_id = hash_it->second;
            } else {
                FixedPath fp;
                uint8_t cnt;
                if (extract_as_path_pairs(elem, fp, cnt) == false || cnt == 0) {
                    continue;
                }
                path_id = id_to_path_len.size();
                id_to_fixed_path.push_back(fp);
                id_to_path_len.push_back(cnt);
                id_to_linear_path.emplace_back();
                id_to_path_is_strict.push_back(0);
                path_hash_to_id[path_hash] = path_id;
            }

            vector<uint32_t> as_path = extract_as_path(elem->as_path);
            if (!as_path.empty()) {
                const bool is_strict = is_strict_linear_as_path(elem->as_path);
                if (id_to_linear_path[path_id].empty()) {
                    id_to_linear_path[path_id] = as_path;
                }
                if (is_strict) {
                    id_to_path_is_strict[path_id] = 1;
                }
                upd_entry_keys.push_back(make_entry_key(p_id, path_id));
            }
        }
    }
    bgpstream_destroy(bs);
}

void materialize_profile_inputs() {
    sort_unique_keys(rib_entry_keys);
    for (uint64_t key : rib_entry_keys) {
        const auto [prefix_id, path_id] = decode_entry_key(key);
        if (prefix_id < 0 || path_id < 0
                || static_cast<size_t>(prefix_id) >= id_to_prefix.size()
                || static_cast<size_t>(path_id) >= id_to_linear_path.size()) {
            continue;
        }
        const auto& path = id_to_linear_path[path_id];
        if (!path.empty()) {
            process_entry(id_to_prefix[prefix_id], path, PathSource::RIB);
        }
    }

    sort_unique_keys(upd_entry_keys);
    upd_entry_keys.erase(
        remove_if(
            upd_entry_keys.begin(), upd_entry_keys.end(),
            [](uint64_t key) {
                return binary_search(rib_entry_keys.begin(), rib_entry_keys.end(), key);
            }),
        upd_entry_keys.end());
    for (uint64_t key : upd_entry_keys) {
        const auto [prefix_id, path_id] = decode_entry_key(key);
        if (prefix_id < 0 || path_id < 0
                || static_cast<size_t>(prefix_id) >= id_to_prefix.size()
                || static_cast<size_t>(path_id) >= id_to_linear_path.size()) {
            continue;
        }
        const auto& path = id_to_linear_path[path_id];
        if (!path.empty()) {
            process_entry(id_to_prefix[prefix_id], path, PathSource::UPD);
        }
    }
}

void write_rib_artifacts(const string& output_dir) {
    fs::create_directories(output_dir);
    if (getenv("PARADE_SKIP_RIB_CACHE") == nullptr) {
        cout << "Constructing and saving BGPRecords from cache..." << '\n';
        ofstream ofs(output_dir + "/rib_records_cache.bin", ios::binary);
        const uint64_t count = rib_entry_keys.size();
        ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));

        for (uint64_t key : rib_entry_keys) {
            const auto [prefix_id, path_id] = decode_entry_key(key);
            const auto& path = id_to_linear_path[path_id];
            BGPRecord record;
            record.timestamp = 0;
            record.origin_as = path.empty() ? 0 : path.back();
            record.signal = 0;

            const string& pfx_str = id_to_prefix[prefix_id];
            const size_t slash_pos = pfx_str.find('/');
            if (slash_pos != string::npos) {
                const string addr_str = pfx_str.substr(0, slash_pos);
                const uint8_t prefix_len = stoi(pfx_str.substr(slash_pos + 1));
                uint32_t ipv4_addr;
                uint8_t ipv6_addr[16];
                if (inet_pton(AF_INET, addr_str.c_str(), &ipv4_addr) == 1) {
                    record.prefix = Prefix(4, ipv4_addr, prefix_len);
                } else if (inet_pton(AF_INET6, addr_str.c_str(), ipv6_addr) == 1) {
                    record.prefix = Prefix(6, ipv6_addr, prefix_len);
                }
            }

            const auto& fp = id_to_fixed_path[path_id];
            const uint8_t pair_count = id_to_path_len[path_id];
            record.as_pair_count = pair_count;
            if (pair_count > 0) {
                memcpy(record.as_pairs, fp.pairs, pair_count * sizeof(ASPair));
            }
            ofs.write(reinterpret_cast<const char*>(&record), sizeof(BGPRecord));
        }
    } else {
        cout << "Skipping deployment RIB cache for this coverage." << '\n';
    }

    // Compact supplemental reference: [count:u64], then repeated
    // [prefix_id:i32][path_len:u8][path_asns:u32 * path_len].
    if (getenv("PARADE_SKIP_REFERENCE_PATHS") == nullptr) {
        cout << "Saving compact simulation path reference..." << '\n';
        ofstream ofs(output_dir + "/reference_paths.bin", ios::binary);
        uint64_t count = 0;
        for (uint64_t key : rib_entry_keys) {
            const auto [prefix_id, path_id] = decode_entry_key(key);
            (void)prefix_id;
            const auto& path = id_to_linear_path[path_id];
            if (id_to_path_is_strict[path_id] && !path.empty() && path.size() <= 255) {
                ++count;
            }
        }
        ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (uint64_t key : rib_entry_keys) {
            const auto [prefix_id, path_id] = decode_entry_key(key);
            const auto& path = id_to_linear_path[path_id];
            if (!id_to_path_is_strict[path_id] || path.empty() || path.size() > 255) {
                continue;
            }
            const uint8_t path_len = static_cast<uint8_t>(path.size());
            ofs.write(reinterpret_cast<const char*>(&prefix_id), sizeof(prefix_id));
            ofs.write(reinterpret_cast<const char*>(&path_len), sizeof(path_len));
            ofs.write(
                reinterpret_cast<const char*>(path.data()),
                path.size() * sizeof(uint32_t));
        }
    } else {
        cout << "Skipping compact simulation path reference." << '\n';
    }
}

void release_path_storage() {
    unordered_map<uint64_t, int32_t>().swap(path_hash_to_id);
    vector<FixedPath>().swap(id_to_fixed_path);
    vector<uint8_t>().swap(id_to_path_len);
    vector<vector<uint32_t>>().swap(id_to_linear_path);
    vector<uint8_t>().swap(id_to_path_is_strict);
    vector<uint64_t>().swap(rib_entry_keys);
    vector<uint64_t>().swap(upd_entry_keys);
    malloc_trim(0);
}

// --- Main ---

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <dataset_dir> <output_dir> [--require-nonempty-ribs]\n";
        return 1;
    }

    string dataset_dir = argv[1];
    string output_dir = argv[2];
    const bool require_nonempty_ribs = argc == 4 && string(argv[3]) == "--require-nonempty-ribs";
    if (argc > 3 && !require_nonempty_ribs) {
        cerr << "Optional argument: --require-nonempty-ribs\n";
        return 1;
    }
    
    // 1. Process RIBs
    time_t latest_rib_ts = 0;
    string rib_dir = dataset_dir + "/rib";
    if (fs::exists(rib_dir)) {
        for (const auto& entry : fs::directory_iterator(rib_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            string fname = entry.path().filename().string();
            if (!should_process_rib_file(fname)) {
                continue;
            }
            time_t file_ts = 0;
            int rrc_id = 0;
            if (extract_timestamp_rrc(fname, file_ts, rrc_id, 0) == 0 && file_ts > latest_rib_ts) {
                latest_rib_ts = file_ts;
            }
            const bool valid_rib = process_mrt_file(entry.path().string());
            if (require_nonempty_ribs && !valid_rib) {
                cerr << "Required RIB has no usable observations or could not be read: " << entry.path() << '\n';
                return 1;
            }
        }
    }

    // 2. Process UPD ANNOUNCE samples from the recent lookback window.
    string upd_dir = dataset_dir + "/upd";
    const time_t upd_lookback_start = latest_rib_ts > 0 ? latest_rib_ts - TRAINING_UPD_LOOKBACK_SEC : 0;
    if (fs::exists(upd_dir)) {
        for (const auto& entry : fs::directory_iterator(upd_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            string fname = entry.path().filename().string();
            if (!should_process_upd_file(fname)) {
                continue;
            }

            time_t file_ts = 0;
            int rrc_id = 0;
            bool in_window = true;
            if (extract_timestamp_rrc(fname, file_ts, rrc_id, 1) == 0 && latest_rib_ts > 0) {
                in_window = (file_ts >= upd_lookback_start && file_ts <= latest_rib_ts);
            }
            if (in_window) {
                process_upd_file(entry.path().string());
            }
        }
    }

    materialize_profile_inputs();

    cout << "Data loading complete.\n";
    cout << "Mixed sample stats: rib_unique_paths=" << rib_entry_keys.size()
         << " upd_unique_paths=" << upd_entry_keys.size() << '\n';
    cout << "Total Prefixes: " << id_to_prefix.size() << '\n';
    cout << "Total ASNs: " << id_to_asn.size() << '\n';
    if (id_to_prefix.empty() || id_to_asn.empty() || rib_entry_keys.empty()) {
        cerr << "No usable RIB records were processed from raw data dir: " << dataset_dir << '\n';
        return 1;
    }

    // These artifacts need the parsed path table but not the later profile
    // aggregation.  Emit them now, then release path-level storage before the
    // next memory-intensive phase.
    write_rib_artifacts(output_dir);
    release_path_storage();

    // 2. Aggregation (Profile Generation)
    cout << "Aggregating profiles...\n";
    
    // Fingerprint -> Profile ID
    // Fingerprint is a sorted vector of (AS_ID, Hop)
    map<vector<pair<int32_t, uint8_t>>, int32_t> fingerprint_to_profile_id;
    vector<int32_t> prefix_to_profile_id(id_to_prefix.size());
    
    // Store profile triplets for output: ProfileID -> [(AS, Hop), ...]
    // We can just store them in a flat vector for binary output
    struct Triplet {
        int32_t profile_id;
        int32_t as_id;
        uint8_t hop;
        float weight;
    };
    vector<Triplet> profile_triplets;
    
    int32_t next_profile_id = 0;
    for (size_t p_id = 0; p_id < prefix_hops.size(); ++p_id) {
        const auto& hops_map = prefix_hops[p_id];
        if (hops_map.empty()) continue; // Should not happen if logic is correct
        
        // Create fingerprint
        vector<pair<int32_t, uint8_t>> fingerprint;
        fingerprint.reserve(hops_map.size());
        for (const auto& kv : hops_map) {
            fingerprint.push_back({kv.first, kv.second.hop});
        }
        sort(fingerprint.begin(), fingerprint.end());
        
        // Check if profile exists
        auto it = fingerprint_to_profile_id.find(fingerprint);
        int32_t profile_id;
        
        if (it == fingerprint_to_profile_id.end()) {
            profile_id = next_profile_id++;
            fingerprint_to_profile_id[fingerprint] = profile_id;
            
            // Record triplets for this new profile
            for (const auto& [as_id, hop] : fingerprint) {
                const auto it_hop = hops_map.find(as_id);
                const PathSource src = (it_hop == hops_map.end()) ? PathSource::RIB : it_hop->second.source;
                const float sample_weight =
                    (src == PathSource::RIB) ? TRAINING_RIB_SAMPLE_WEIGHT : TRAINING_UPD_SAMPLE_WEIGHT;
                profile_triplets.push_back({profile_id, as_id, hop, sample_weight});
            }
        } else {
            profile_id = it->second;
        }
        
        prefix_to_profile_id[p_id] = profile_id;
    }
    
    cout << "Aggregation complete.\n";
    cout << "Total Profiles: " << next_profile_id << '\n';
    if (next_profile_id == 0 || profile_triplets.empty()) {
        cerr << "No weighted profile triplets were generated from raw data dir: " << dataset_dir << '\n';
        return 1;
    }
    cout << "Compression Ratio: " << (double)id_to_prefix.size() / next_profile_id << "x\n";

    // Aggregate edges per profile
    cout << "Aggregating edges per profile..." << '\n';
    vector<vector<tuple<int32_t, int32_t, uint8_t>>> profile_edges(next_profile_id);
    for (size_t p_id = 0; p_id < prefix_edges.size(); ++p_id) {
        int32_t prof_id = prefix_to_profile_id[p_id];
        for (const auto& edge : prefix_edges[p_id]) {
            profile_edges[prof_id].push_back(edge);
        }
    }

    // Compute fingerprint for each profile
    cout << "Computing 128-bit fingerprint for profiles..." << '\n';
    const double alpha = 0.5;
    
    // Modify storage container: store PropFingerprint structs
    vector<PropFingerprint> profile_fingerprints(next_profile_id);
    
    // 128-dimensional accumulator
    vector<double> accum(128); 

    for (int32_t prof_id = 0; prof_id < next_profile_id; ++prof_id) {
        // Reset accumulator
        fill(accum.begin(), accum.end(), 0.0);

        for (const auto& [u_id, v_id, depth] : profile_edges[prof_id]) {
            double w = exp(-alpha * depth);

            // 2. Generate two independent 64-bit hashes
            // Part 1: Low 64 bits
            uint64_t h1 = (uint64_t)u_id * 6364136223846793005ULL + (uint64_t)v_id * 1442695040888963407ULL;
            
            // Part 2: High 64 bits (using different parameters and mixing logic to ensure orthogonality)
            uint64_t h2 = ((uint64_t)u_id ^ 0x9e3779b97f4a7c15ULL) * 0xc6a4a7935bd1e995ULL + 
                          ((uint64_t)v_id ^ 0x5555555555555555ULL) * 0x5bd1e9955bd1e995ULL;

            // 3. Update accumulator (split into two segments to avoid if-else check for k >= 64)
            
            // Process low 64 bits (corresponding to h1)
            for (int k = 0; k < 64; ++k) {
                if (h1 & (1ULL << k)) {
                    accum[k] += w;
                } else {
                    accum[k] -= w;
                }
            }

            // Process high 64 bits (corresponding to h2, indices from 64 to 127)
            for (int k = 0; k < 64; ++k) {
                if (h2 & (1ULL << k)) {
                    accum[k + 64] += w; // Note index offset +64
                } else {
                    accum[k + 64] -= w;
                }
            }
        }

        // 4. Generate 128-bit fingerprint
        uint64_t fp_low = 0;
        uint64_t fp_high = 0;

        // Generate low 64-bit part
        for (int k = 0; k < 64; ++k) {
            if (accum[k] > 0) {
                fp_low |= (1ULL << k);
            }
        }

        // Generate high 64-bit part
        for (int k = 0; k < 64; ++k) {
            if (accum[k + 64] > 0) { // Note index offset +64
                fp_high |= (1ULL << k);
            }
        }

        profile_fingerprints[prof_id] = {fp_low, fp_high};
    }
    cout << "fingerprint computation complete." << '\n';

    // 3. Output
    fs::create_directories(output_dir);
    {
        ofstream ofs(output_dir + "/as_map.txt");
        for (const auto& asn : id_to_asn) ofs << asn << "\n";
    }
    
    {
        ofstream ofs(output_dir + "/prefix_map.txt");
        for (const auto& p : id_to_prefix) ofs << p << "\n";
    }
    
    {
        ofstream ofs(output_dir + "/prefix_to_profile.bin", ios::binary);
        ofs.write(reinterpret_cast<const char*>(prefix_to_profile_id.data()), prefix_to_profile_id.size() * sizeof(int32_t));
    }
    
    // Save Profile Triplets (Binary struct {int32, int32, uint8, float})
    {
        ofstream ofs(output_dir + "/profile_triplets.bin", ios::binary);
        // Write count first
        uint64_t count = profile_triplets.size();
        ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        for (const auto& t : profile_triplets) {
            ofs.write(reinterpret_cast<const char*>(&t.profile_id), sizeof(int32_t));
            ofs.write(reinterpret_cast<const char*>(&t.as_id), sizeof(int32_t));
            ofs.write(reinterpret_cast<const char*>(&t.hop), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char*>(&t.weight), sizeof(float));
        }
    }

    {
        ofstream ofs(output_dir + "/profile_fingerprints.bin", ios::binary);
        // Cast pointer, data layout: [low_0, high_0, low_1, high_1, ..., low_N, high_N]
        uint64_t* ptr = reinterpret_cast<uint64_t*>(profile_fingerprints.data());
        ofs.write(reinterpret_cast<const char*>(ptr), profile_fingerprints.size() * sizeof(PropFingerprint));
    }
    
    // These process-wide containers may hold hundreds of millions of entries.
    // All output streams have been closed above, so let the OS reclaim them in
    // one operation instead of spending minutes walking global destructors.
    cout.flush();
    cerr.flush();
    std::_Exit(0);
}
