 #include <filesystem>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <cctype>
#include <regex>
#include "path_extractor.hpp"
#include "helpers.hpp"

static const char* optstring = "D:O:T:W:";

// State file path
string state_file_path = "data_parser_state.txt";

struct IncrementalState {
    time_t baseline_ts;
    time_t last_processed_upd_ts;
};

// State file format: <baseline_ts> <last_processed_upd_ts>
IncrementalState read_incremental_state() {
    ifstream ifs(state_file_path);
    if (!ifs) return {0, 0};

    long long first = 0;
    long long second = 0;
    if (!(ifs >> first)) {
        return {0, 0};
    }
    if (ifs >> second) {
        return {static_cast<time_t>(first), static_cast<time_t>(second)};
    }

    // Fallback: only last_processed_upd_ts was stored
    return {0, static_cast<time_t>(first)};
}

void write_incremental_state(const IncrementalState& state) {
    ofstream ofs(state_file_path);
    ofs << state.baseline_ts << " " << state.last_processed_upd_ts;
}

using namespace std;

void init_program_log_from_env(const char* env_name) {
    const char* log_path = getenv(env_name);
    if (!log_path || log_path[0] == '\0') {
        return;
    }
    ofstream ofs(log_path, ios::out | ios::trunc);
    if (!ofs) {
        fprintf(stderr, "Warning: failed to initialize log file from %s: %s\n", env_name, log_path);
    }
}

// Global RingBuffer manager. A smart pointer is used for simpler resource management.
unique_ptr<LockFreeRingBufferManager> g_ringbuffer_manager;

// Check whether the RIB cache exists and is valid.
bool check_rib_cache(const string& cache_path) {
    ifstream ifs(cache_path, ios::binary);
    if (!ifs) {
        printf("Cache file not found: %s\n", cache_path.c_str());
        return false;
    }
    
    size_t count;
    if (!ifs.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        return false;
    }
    
    // Basic validation: verify file size matches expected record count.
    ifs.seekg(0, ios::end);
    size_t filesize = ifs.tellg();
    size_t expected_size = sizeof(size_t) + count * sizeof(BGPRecord);
    
    if (filesize != expected_size) {
        printf("Cache file size mismatch (expected %zu, got %zu): %s\n", expected_size, filesize, cache_path.c_str());
        return false;
    }
    
    printf("Found valid RIB records cache with %zu records: %s\n", count, cache_path.c_str());
    return true;
}

// Stream-load RIB cache entries into the RingBuffer.
bool stream_rib_cache_to_ringbuffer(const string& cache_path) {
    ifstream ifs(cache_path, ios::binary);
    if (!ifs) return false;
    
    size_t count;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    printf("Streaming %zu RIB records from cache to RingBuffer...\n", count);
    
    size_t alignment_records = g_ringbuffer_manager->get_write_alignment_records_needed();
    size_t current_threshold = (alignment_records == 0) ? BATCH_SIZE : alignment_records;
    
    vector<BGPRecord> batch_buffer;
    // Reserve capacity to avoid repeated reallocations.
    batch_buffer.reserve(BATCH_SIZE);
    
    size_t processed = 0;
    while (processed < count) {
        size_t remaining = count - processed;
        // Determine read size for this iteration: bounded by remaining records and threshold.
        size_t to_read = (remaining < current_threshold) ? remaining : current_threshold;
        
        // Resize vector first, then read directly into the vector buffer.
        batch_buffer.resize(to_read);
        if (!ifs.read(reinterpret_cast<char*>(batch_buffer.data()), to_read * sizeof(BGPRecord))) {
            fprintf(stderr, "Failed to read from cache (expected %zu records)\n", to_read);
            return false;
        }
         
        // Write this chunk to RingBuffer in batch mode.
        if (!write_batch_to_ringbuffer(batch_buffer, false)) {
            fprintf(stderr, "Failed to flush batch during cache streaming\n");
            return false;
        }
            
        processed += to_read;
        current_threshold = BATCH_SIZE;
    }
    
    printf("Finished streaming RIB cache.\n");
    return true;
}

// Parse time string to timestamp (UTC)
// Supports formats: timestamp, 'Y-m-d', 'Y-m-d H:M', 'Y-m-d H:M:S'
time_t parse_time_string(const string& time_str) {
        // Try to parse as timestamp first - check if entire string is consumed
        try {
            size_t pos;
            long long timestamp = stoll(time_str, &pos);
            if (pos == time_str.length()) {  // Entire string was parsed as number
                    return static_cast<time_t>(timestamp);
            }
        } catch (...) {
            // Not a valid number, continue to date parsing
        }

        struct tm tm_time = {};
        istringstream ss(time_str);
        string date_part, time_part;
        
        // Split by space to get date and time parts
        if (!getline(ss, date_part, ' ')) {
            return -1;
        }
        
        // Parse date part (YYYY-MM-DD)
        int year, month, day;
        if (sscanf(date_part.c_str(), "%d-%d-%d", &year, &month, &day) != 3) {
            return -1;
        }
        
        tm_time.tm_year = year - 1900;
        tm_time.tm_mon = month - 1;
        tm_time.tm_mday = day;
        
        // Parse time part if exists (H:M or H:M:S)
        if (getline(ss, time_part)) {
            int hour, min, sec = 0;
            if (sscanf(time_part.c_str(), "%d:%d:%d", &hour, &min, &sec) >= 2) {
                tm_time.tm_hour = hour;
                tm_time.tm_min = min;
                tm_time.tm_sec = sec;
            }
        }
        return timegm(&tm_time);
}

// Convert timestamp to readable string
string timestamp_to_string(time_t timestamp) {
        if (timestamp == LLONG_MAX) {
            return "unlimited";
        }
        
        struct tm* tm_info = gmtime(&timestamp);
        char buffer[32];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S UTC", tm_info);
        return string(buffer);
}

// Parse start time parameter.
// Preferred format: <start>. Backward compatible with legacy <start,end> input.
time_t parse_start_time_param(const string& time_arg) {
    if (time_arg.empty()) {
        fprintf(stderr, "You can specify the UTC start time with -T option\n");
        fprintf(stderr, "Example: -T '2019-06-06 08:00'\n");
        fprintf(stderr, "Using current time as fallback start time.\n");
        return time(nullptr);
    }

    // Backward compatibility: allow legacy start,end input and only use start.
    string start_part = time_arg;
    size_t comma_pos = time_arg.find(',');
    if (comma_pos != string::npos) {
        start_part = time_arg.substr(0, comma_pos);
    }

    time_t tmp = parse_time_string(start_part);
    if (tmp != -1) {
        return tmp;
    } else {
        fprintf(stderr, "You can specify the UTC start time with -T option\n");
        fprintf(stderr, "Example: -T '2019-06-06 08:00'\n");
        fprintf(stderr, "Using current time as fallback start time.\n");
        return time(nullptr);
    }
}

bool parse_time_window_param(const string& time_arg, time_t& start_ts, time_t& end_ts) {
    if (time_arg.empty()) {
        return false;
    }

    size_t comma_pos = time_arg.find(',');
    if (comma_pos == string::npos) {
        fprintf(stderr, "Invalid time window format: %s\n", time_arg.c_str());
        fprintf(stderr, "Expected: <start>,<end>\n");
        return false;
    }

    string start_part = time_arg.substr(0, comma_pos);
    string end_part = time_arg.substr(comma_pos + 1);
    if (start_part.empty() || end_part.empty()) {
        fprintf(stderr, "Invalid time window format: %s\n", time_arg.c_str());
        fprintf(stderr, "Expected: <start>,<end>\n");
        return false;
    }

    start_ts = parse_time_string(start_part);
    end_ts = parse_time_string(end_part);
    if (start_ts == -1 || end_ts == -1) {
        fprintf(stderr, "Failed to parse time window: %s\n", time_arg.c_str());
        return false;
    }
    if (end_ts <= start_ts) {
        fprintf(stderr, "Invalid time window: end must be greater than start. start=%s end=%s\n",
                timestamp_to_string(start_ts).c_str(),
                timestamp_to_string(end_ts).c_str());
        return false;
    }
    return true;
}

string format_data_timestamp_key(time_t ts) {
    struct tm* tm_info = gmtime(&ts);
    char buffer[32];
    strftime(buffer, sizeof(buffer), "%Y%m%d.%H%M", tm_info);
    return string(buffer);
}

// Helper function to process files in a directory
void process_directory(string dir_path, 
                    unordered_map<time_t, vector<string>> *rib_files,
                    unordered_map<time_t, vector<string>> *upd_files,
                    bool check_rib, bool check_upd) {
    if (!filesystem::exists(dir_path)) return;
    
    for (const auto &entry: filesystem::directory_iterator(dir_path)) {
        if (entry.is_directory()) continue;
        
        const string filename = entry.path().filename().string();
        if (filename.size() > 5 && filename.rfind(".done") == (filename.size() - 5)) {
            continue;
        }
        time_t timestamp;
        int rrc_id;
        
        string file_path = dir_path;
        if (file_path.back() != '/') {
            file_path += "/";
        }
        file_path += filename;
        
        // Try to parse as RIB file
        if (check_rib) {
            int rib_res = extract_timestamp_rrc(filename, timestamp, rrc_id, 0);
            if (rib_res == 0) {
                if (rib_files->find(timestamp) == rib_files->end()) {
                    vector<string> file_list;
                    file_list.push_back(file_path);
                    rib_files->insert({timestamp, file_list});
                } else {
                    rib_files->at(timestamp).push_back(file_path);
                }
                continue;
            }
        }
        
        // Try to parse as UPD file
        if (check_upd) {
            int upd_res = extract_timestamp_rrc(filename, timestamp, rrc_id, 1);
            if (upd_res == 0) {
                if (upd_files->find(timestamp) == upd_files->end()) {
                    vector<string> file_list;
                    file_list.push_back(file_path);
                    upd_files->insert({timestamp, file_list});
                } else {
                    upd_files->at(timestamp).push_back(file_path);
                }
            }
        }
    }
}

// Get the files in the data directory, then group the files by the time
// file_type: 0 = rib only, 1 = upd only, 2 = both
int get_files_in_data_dir(string data_dir,
                        unordered_map<time_t, vector<string>> *rib_files,
                        unordered_map<time_t, vector<string>> *upd_files) {
    if (!filesystem::exists(data_dir)) {
        printf("The data directory does not exist\n");
        return -1;
    }
    
    string rib_dir = data_dir + "/rib";
    string upd_dir = data_dir + "/upd";
    bool found_any = false;
    
    if (filesystem::exists(rib_dir)) {
        process_directory(rib_dir, rib_files, upd_files, true, false);
        found_any = true;
    }
    
    if (filesystem::exists(upd_dir)) {
        process_directory(upd_dir, rib_files, upd_files, false, true);
        found_any = true;
    }
    
    if (!found_any) {
        printf("Error: 'rib' or 'upd' directory not found in %s\n", data_dir.c_str());
        return -1;
    }
    
    return 0;
}

bool parse_processed_dir_timestamp(const string& dirname, time_t& out_ts) {
    if (dirname.length() != 14) {
        return false;
    }
    if (dirname.find_first_not_of("0123456789") != string::npos) {
        return false;
    }

    try {
        int year = stoi(dirname.substr(0, 4));
        int month = stoi(dirname.substr(4, 2));
        int day = stoi(dirname.substr(6, 2));
        int hour = stoi(dirname.substr(8, 2));
        int min = stoi(dirname.substr(10, 2));
        int sec = stoi(dirname.substr(12, 2));

        struct tm tm_time = {};
        tm_time.tm_year = year - 1900;
        tm_time.tm_mon = month - 1;
        tm_time.tm_mday = day;
        tm_time.tm_hour = hour;
        tm_time.tm_min = min;
        tm_time.tm_sec = sec;
        out_ts = timegm(&tm_time);
        return out_ts > 0;
    } catch (...) {
        return false;
    }
}

string format_processed_dir_timestamp(time_t ts) {
    struct tm* tm_info = gmtime(&ts);
    char buffer[32];
    strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", tm_info);
    return string(buffer);
}

static constexpr time_t RIPE_RIB_INTERVAL_SEC = 8 * 3600;

bool is_ripe_rib_aligned_ts(time_t ts) {
    if (ts <= 0) {
        return false;
    }
    return (ts % RIPE_RIB_INTERVAL_SEC) == 0;
}

time_t floor_to_ripe_rib_slot(time_t ts) {
    if (ts <= 0) {
        return 0;
    }
    return ts - (ts % RIPE_RIB_INTERVAL_SEC);
}

// Maximum allowed rollback from startup anchor when selecting a previous baseline.
// Set DATA_PARSER_BASELINE_MAX_BACKTRACK_SEC <= 0 to disable the rollback limit.
time_t get_baseline_max_backtrack_seconds() {
    const time_t default_limit = 36 * 60 * 60;  // 36h
    const char* env = getenv("DATA_PARSER_BASELINE_MAX_BACKTRACK_SEC");
    if (!env || env[0] == '\0') {
        return default_limit;
    }

    char* end_ptr = nullptr;
    long long parsed = strtoll(env, &end_ptr, 10);
    if (end_ptr == env || (end_ptr && *end_ptr != '\0')) {
        fprintf(stderr,
                "Invalid DATA_PARSER_BASELINE_MAX_BACKTRACK_SEC='%s', using default %ld seconds\n",
                env, static_cast<long>(default_limit));
        return default_limit;
    }

    if (parsed <= 0) {
        return LLONG_MAX;
    }
    return static_cast<time_t>(parsed);
}

time_t find_latest_complete_baseline(const string& processed_data_dir, time_t current_baseline_ts) {
    if (!filesystem::exists(processed_data_dir)) {
        return current_baseline_ts;
    }

    time_t latest_ts = current_baseline_ts;
    for (const auto& entry : filesystem::directory_iterator(processed_data_dir)) {
        if (!entry.is_directory()) {
            continue;
        }

        const string dirname = entry.path().filename().string();
        if (dirname.empty()) {
            continue;
        }

        time_t candidate_ts = 0;
        if (!parse_processed_dir_timestamp(dirname, candidate_ts)) {
            continue;
        }

        if (candidate_ts <= latest_ts) {
            continue;
        }

        string cache_path = entry.path().string() + "/rib_records_cache.bin";
        if (check_rib_cache(cache_path)) {
            latest_ts = candidate_ts;
        }
    }

    return latest_ts;
}

// Find the largest complete and RIPE-8h-aligned baseline <= upper_bound_ts.
// Returns 0 when no valid aligned baseline cache is available.
time_t find_latest_complete_aligned_baseline_le(const string& processed_data_dir, time_t upper_bound_ts) {
    if (!filesystem::exists(processed_data_dir)) {
        return 0;
    }

    time_t best_ts = 0;
    for (const auto& entry : filesystem::directory_iterator(processed_data_dir)) {
        if (!entry.is_directory()) {
            continue;
        }

        const string dirname = entry.path().filename().string();
        if (dirname.empty()) {
            continue;
        }

        time_t candidate_ts = 0;
        if (!parse_processed_dir_timestamp(dirname, candidate_ts)) {
            continue;
        }
        if (!is_ripe_rib_aligned_ts(candidate_ts)) {
            continue;
        }
        if (candidate_ts > upper_bound_ts) {
            continue;
        }

        string cache_path = entry.path().string() + "/rib_records_cache.bin";
        if (!check_rib_cache(cache_path)) {
            continue;
        }

        if (best_ts == 0 || candidate_ts > best_ts) {
            best_ts = candidate_ts;
        }
    }

    return best_ts;
}

// Find the smallest complete and RIPE-8h-aligned baseline >= lower_bound_ts.
// Returns 0 when no valid aligned baseline cache is available.
time_t find_earliest_complete_aligned_baseline_ge(const string& processed_data_dir, time_t lower_bound_ts) {
    if (!filesystem::exists(processed_data_dir)) {
        return 0;
    }

    time_t best_ts = 0;
    for (const auto& entry : filesystem::directory_iterator(processed_data_dir)) {
        if (!entry.is_directory()) {
            continue;
        }

        const string dirname = entry.path().filename().string();
        if (dirname.empty()) {
            continue;
        }

        time_t candidate_ts = 0;
        if (!parse_processed_dir_timestamp(dirname, candidate_ts)) {
            continue;
        }
        if (!is_ripe_rib_aligned_ts(candidate_ts)) {
            continue;
        }
        if (candidate_ts < lower_bound_ts) {
            continue;
        }

        string cache_path = entry.path().string() + "/rib_records_cache.bin";
        if (!check_rib_cache(cache_path)) {
            continue;
        }

        if (best_ts == 0 || candidate_ts < best_ts) {
            best_ts = candidate_ts;
        }
    }

    return best_ts;
}

// Find the smallest complete baseline timestamp >= lower_bound_ts.
// Returns 0 when no valid baseline cache is available.
time_t find_earliest_complete_baseline_ge(const string& processed_data_dir, time_t lower_bound_ts) {
    if (!filesystem::exists(processed_data_dir)) {
        return 0;
    }

    time_t best_ts = 0;
    for (const auto& entry : filesystem::directory_iterator(processed_data_dir)) {
        if (!entry.is_directory()) {
            continue;
        }

        const string dirname = entry.path().filename().string();
        if (dirname.empty()) {
            continue;
        }

        time_t candidate_ts = 0;
        if (!parse_processed_dir_timestamp(dirname, candidate_ts)) {
            continue;
        }

        if (candidate_ts < lower_bound_ts) {
            continue;
        }

        string cache_path = entry.path().string() + "/rib_records_cache.bin";
        if (!check_rib_cache(cache_path)) {
            continue;
        }

        if (best_ts == 0 || candidate_ts < best_ts) {
            best_ts = candidate_ts;
        }
    }

    return best_ts;
}

// Find the largest complete baseline timestamp <= upper_bound_ts.
// Returns 0 when no valid baseline cache is available.
time_t find_latest_complete_baseline_le(const string& processed_data_dir, time_t upper_bound_ts) {
    if (!filesystem::exists(processed_data_dir)) {
        return 0;
    }

    time_t best_ts = 0;
    for (const auto& entry : filesystem::directory_iterator(processed_data_dir)) {
        if (!entry.is_directory()) {
            continue;
        }

        const string dirname = entry.path().filename().string();
        if (dirname.empty()) {
            continue;
        }

        time_t candidate_ts = 0;
        if (!parse_processed_dir_timestamp(dirname, candidate_ts)) {
            continue;
        }

        if (candidate_ts > upper_bound_ts) {
            continue;
        }

        string cache_path = entry.path().string() + "/rib_records_cache.bin";
        if (!check_rib_cache(cache_path)) {
            continue;
        }

        if (best_ts == 0 || candidate_ts > best_ts) {
            best_ts = candidate_ts;
        }
    }

    return best_ts;
}

// Find earliest timestamp currently available in raw rib/upd inputs.
// Returns 0 when no parseable raw files exist yet.
time_t find_earliest_available_raw_ts(const string& raw_data_dir) {
    auto rib_files = make_unique<unordered_map<time_t, vector<string>>>();
    auto upd_files = make_unique<unordered_map<time_t, vector<string>>>();

    if (get_files_in_data_dir(raw_data_dir, rib_files.get(), upd_files.get()) != 0) {
        return 0;
    }

    time_t earliest_ts = 0;
    for (const auto& [ts, _] : *rib_files) {
        if (earliest_ts == 0 || ts < earliest_ts) {
            earliest_ts = ts;
        }
    }
    for (const auto& [ts, _] : *upd_files) {
        if (earliest_ts == 0 || ts < earliest_ts) {
            earliest_ts = ts;
        }
    }
    return earliest_ts;
}

struct BaselineCollectorFilter {
    unordered_set<string> ripe_missing;
    unordered_set<string> rv_missing;
};

string format_data_timestamp_key_for_done(time_t ts) {
    struct tm* tm_info = gmtime(&ts);
    char buffer[32];
    strftime(buffer, sizeof(buffer), "%Y%m%d.%H%M", tm_info);
    return string(buffer);
}

bool parse_upd_filename_source_collector(const string& file_path, string& source, string& collector) {
    string filename = filesystem::path(file_path).filename().string();
    vector<string> parts;
    size_t pos = 0;
    string token;
    string text = filename;
    while ((pos = text.find('.')) != string::npos) {
        token = text.substr(0, pos);
        parts.push_back(token);
        text.erase(0, pos + 1);
    }
    parts.push_back(text);

    if (parts.size() < 6) {
        return false;
    }
    if (parts[0] != "upd") {
        return false;
    }
    if (!(parts[2].size() == 8 && parts[3].size() == 4)) {
        return false;
    }

    if (parts[1] == "ris") {
        source = "ripe";
    } else if (parts[1] == "rv") {
        source = "rv";
    } else {
        return false;
    }

    size_t end_index = parts.size();
    if (!parts.empty() && (parts.back() == "gz" || parts.back() == "bz2")) {
        end_index -= 1;
    }
    if (end_index <= 4) {
        return false;
    }

    collector.clear();
    for (size_t i = 4; i < end_index; i++) {
        if (!collector.empty()) {
            collector += ".";
        }
        collector += parts[i];
    }
    return !collector.empty();
}

string read_text_file(const string& path) {
    ifstream ifs(path);
    if (!ifs) {
        return "";
    }
    ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

unordered_set<string> parse_json_string_array_field(const string& text, const string& field_name) {
    unordered_set<string> values;
    regex field_regex("\"" + field_name + "\"\\s*:\\s*\\[(.*?)\\]");
    smatch match;
    if (!regex_search(text, match, field_regex)) {
        return values;
    }

    string array_text = match[1].str();
    regex str_regex("\"([^\"]+)\"");
    auto begin = sregex_iterator(array_text.begin(), array_text.end(), str_regex);
    auto end = sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        values.insert((*it)[1].str());
    }
    return values;
}

BaselineCollectorFilter load_rib_missing_collectors_for_baseline(const string& raw_data_dir, time_t baseline_ts) {
    BaselineCollectorFilter filter;
    string done_name = format_data_timestamp_key_for_done(baseline_ts) + ".done";
    string done_path = raw_data_dir + "/rib/" + done_name;
    string text = read_text_file(done_path);
    if (text.empty()) {
        printf("RIB done metadata unavailable for baseline %s, no collector filter applied\n",
               timestamp_to_string(baseline_ts).c_str());
        return filter;
    }

    size_t key_pos = text.find("\"missing_collectors_by_source\"");
    if (key_pos == string::npos) {
        printf("RIB done metadata missing collector field for baseline %s, no collector filter applied\n",
               timestamp_to_string(baseline_ts).c_str());
        return filter;
    }

    size_t obj_start = text.find('{', key_pos);
    if (obj_start == string::npos) {
        return filter;
    }
    int depth = 0;
    size_t obj_end = string::npos;
    for (size_t i = obj_start; i < text.size(); i++) {
        if (text[i] == '{') {
            depth += 1;
        } else if (text[i] == '}') {
            depth -= 1;
            if (depth == 0) {
                obj_end = i;
                break;
            }
        }
    }
    if (obj_end == string::npos || obj_end <= obj_start) {
        return filter;
    }

    string missing_object = text.substr(obj_start, obj_end - obj_start + 1);
    filter.ripe_missing = parse_json_string_array_field(missing_object, "ripe");
    filter.rv_missing = parse_json_string_array_field(missing_object, "rv");

    printf("Loaded RIB collector filter for baseline %s: ripe_missing=%zu rv_missing=%zu\n",
           timestamp_to_string(baseline_ts).c_str(),
           filter.ripe_missing.size(),
           filter.rv_missing.size());
    return filter;
}

bool should_skip_upd_file_by_rib_filter(const string& file_path,
                                        time_t baseline_ts,
                                        const BaselineCollectorFilter& filter) {
    string source;
    string collector;
    if (!parse_upd_filename_source_collector(file_path, source, collector)) {
        return false;
    }

    bool skip = false;
    if (source == "ripe") {
        skip = filter.ripe_missing.find(collector) != filter.ripe_missing.end();
    } else if (source == "rv") {
        skip = filter.rv_missing.find(collector) != filter.rv_missing.end();
    }

    if (skip) {
        printf("Filtered UPD file by RIB-missing collector: baseline=%s source=%s collector=%s file=%s\n",
               timestamp_to_string(baseline_ts).c_str(),
               source.c_str(),
               collector.c_str(),
               file_path.c_str());
    }
    return skip;
}

bool process_next_upd_batch(const string& data_dir,
                            time_t baseline_ts,
                            time_t& last_processed_upd,
                            time_t max_exclusive_ts,
                            const BaselineCollectorFilter& filter) {
    auto rib_files = make_unique<unordered_map<time_t, vector<string>>>();
    auto upd_files = make_unique<unordered_map<time_t, vector<string>>>();

    int res = get_files_in_data_dir(data_dir, rib_files.get(), upd_files.get());
    if (res != 0) {
        fprintf(stderr, "Failed to get files\n");
        return false;
    }

    set<time_t> new_upd_timestamps;
    for (const auto& [timestamp, file_list] : *upd_files) {
        if (timestamp <= last_processed_upd) {
            continue;
        }
        if (timestamp < baseline_ts) {
            continue;
        }
        if (timestamp >= max_exclusive_ts) {
            continue;
        }
        new_upd_timestamps.insert(timestamp);
    }

    if (new_upd_timestamps.empty()) {
        return false;
    }

    time_t batch_start = *new_upd_timestamps.begin();
    time_t batch_end = batch_start + 5 * 60;

    vector<BGPRecord> batch_records;
    batch_records.reserve(100000);

    for (time_t ts : new_upd_timestamps) {
        if (ts >= batch_start && ts < batch_end) {
            const auto& file_list = upd_files->at(ts);
            for (const auto& file_path : file_list) {
                if (should_skip_upd_file_by_rib_filter(file_path, baseline_ts, filter)) {
                    continue;
                }
                parse_one_file_to_container(file_path, ts, false, batch_records);
            }
        }
    }

    if (batch_records.empty()) {
        last_processed_upd = batch_end - 1;
        return true;
    }

    sort(batch_records.begin(), batch_records.end(),
        [](const BGPRecord& a, const BGPRecord& b) {
            return a.timestamp < b.timestamp;
        });

    if (!write_batch_to_ringbuffer(batch_records, true)) {
        fprintf(stderr, "Failed to write UPD batch\n");
        return false;
    }

    printf("Processed UPD batch from %s to %s (RIB: %s)\n",
           timestamp_to_string(batch_start).c_str(),
           timestamp_to_string(batch_end).c_str(),
           timestamp_to_string(baseline_ts).c_str());

    last_processed_upd = batch_end - 1;
    return true;
}

bool process_warmup_upd_batches(const string& data_dir,
                                time_t baseline_ts,
                                time_t warmup_seconds,
                                const BaselineCollectorFilter& filter) {

    const time_t warmup_start = baseline_ts - warmup_seconds;
    const time_t warmup_end = baseline_ts;

    auto rib_files = make_unique<unordered_map<time_t, vector<string>>>();
    auto upd_files = make_unique<unordered_map<time_t, vector<string>>>();
    int res = get_files_in_data_dir(data_dir, rib_files.get(), upd_files.get());
    if (res != 0) {
        fprintf(stderr, "Failed to get files for warmup\n");
        return false;
    }

    set<time_t> warmup_timestamps;
    for (const auto& [timestamp, _] : *upd_files) {
        if (timestamp < warmup_start || timestamp >= warmup_end) {
            continue;
        }
        warmup_timestamps.insert(timestamp);
    }

    if (warmup_timestamps.empty()) {
        printf("Warmup: no UPD data in [%s, %s)\n",
               timestamp_to_string(warmup_start).c_str(),
               timestamp_to_string(warmup_end).c_str());
        return true;
    }

    time_t batch_start = *warmup_timestamps.begin();
    size_t total_records = 0;
    size_t total_batches = 0;

    while (batch_start < warmup_end) {
        time_t batch_end = batch_start + 5 * 60;
        if (batch_end > warmup_end) {
            batch_end = warmup_end;
        }

        vector<BGPRecord> batch_records;
        batch_records.reserve(100000);
        for (time_t ts : warmup_timestamps) {
            if (ts < batch_start || ts >= batch_end) {
                continue;
            }
            const auto& file_list = upd_files->at(ts);
            for (const auto& file_path : file_list) {
                if (should_skip_upd_file_by_rib_filter(file_path, baseline_ts, filter)) {
                    continue;
                }
                parse_one_file_to_container(file_path, ts, false, batch_records);
            }
        }

        if (!batch_records.empty()) {
            sort(batch_records.begin(), batch_records.end(),
                 [](const BGPRecord& a, const BGPRecord& b) {
                     return a.timestamp < b.timestamp;
                 });
            if (!write_batch_to_ringbuffer(batch_records, true)) {
                fprintf(stderr, "Failed to write warmup UPD batch\n");
                return false;
            }
            total_records += batch_records.size();
            total_batches += 1;
        }

        batch_start = batch_end;
    }

    printf("Warmup complete for baseline %s: batches=%zu records=%zu window=[%s, %s)\n",
           timestamp_to_string(baseline_ts).c_str(),
           total_batches,
           total_records,
           timestamp_to_string(warmup_start).c_str(),
           timestamp_to_string(warmup_end).c_str());
    return true;
}


int main(int argc, char *argv[]) {
    init_program_log_from_env("DATA_PARSER_LOG_FILE");
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IOLBF, 0);

    int opt;
    string raw_data_dir = "";
    string dataset_dir = "";
    string start_time_arg = "";
    string time_window_arg = "";
    
    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 'D':
                raw_data_dir = optarg;
                break;
            case 'O':
                dataset_dir = optarg;
                break;
            case 'T':
                start_time_arg = optarg;
                break;
            case 'W':
                time_window_arg = optarg;
                break;
            default:
                break;
        }
    }

    if (raw_data_dir == "") {
        fprintf(stderr, "Please specify <raw_data_dir> with -D option\n");
        return 1;
    }

    if (dataset_dir == "") {
        fprintf(stderr, "Please specify <dataset_dir> with -O option\n");
        return 1;
    }

    if (!filesystem::exists(raw_data_dir)) {
        fprintf(stderr, "raw_data_dir does not exist: %s\n", raw_data_dir.c_str());
        return 1;
    }

    if (!filesystem::exists(dataset_dir)) {
        fprintf(stderr, "dataset_dir does not exist: %s\n", dataset_dir.c_str());
        return 1;
    }

    bool one_shot_mode = !time_window_arg.empty();
    time_t one_shot_end_time = 0;
    time_t start_time = 0;
    if (one_shot_mode) {
        if (!parse_time_window_param(time_window_arg, start_time, one_shot_end_time)) {
            return 1;
        }
        if (!start_time_arg.empty()) {
            printf("Warning: -W is set, ignoring -T. Window start is used as pipeline start.\n");
        }
    } else {
        start_time = parse_start_time_param(start_time_arg);
    }

    if (one_shot_mode) {
        printf("Running one-shot window mode: [%s, %s)\n",
               timestamp_to_string(start_time).c_str(),
               timestamp_to_string(one_shot_end_time).c_str());

        string processed_data_dir = dataset_dir + "/processed_data";
        const time_t warmup_seconds = 8 * 60 * 60;
        const time_t max_backtrack_sec = get_baseline_max_backtrack_seconds();
        const time_t baseline_target_slot = floor_to_ripe_rib_slot(start_time);

        time_t current_baseline_ts = 0;
        time_t candidate_prev = find_latest_complete_aligned_baseline_le(processed_data_dir, baseline_target_slot);
        bool prev_within_limit = false;
        if (candidate_prev > 0) {
            time_t rollback = baseline_target_slot - candidate_prev;
            if (rollback < 0) {
                rollback = 0;
            }
            prev_within_limit = (rollback <= max_backtrack_sec);
        }

        if (candidate_prev > 0 && prev_within_limit) {
            current_baseline_ts = candidate_prev;
        } else {
            time_t candidate_next = find_earliest_complete_aligned_baseline_ge(processed_data_dir, baseline_target_slot);
            if (candidate_next > 0) {
                current_baseline_ts = candidate_next;
            }
        }

        if (current_baseline_ts <= 0) {
            string legacy_cache = dataset_dir + "/rib_records_cache.bin";
            if (filesystem::exists(legacy_cache)) {
                current_baseline_ts = start_time;
                printf("No aligned baseline directory cache found. Falling back to legacy cache: %s\n",
                       legacy_cache.c_str());
            } else {
                fprintf(stderr, "No aligned baseline cache found for one-shot window near %s\n",
                        timestamp_to_string(start_time).c_str());
                return 1;
            }
        }

        printf("One-shot baseline selected: %s (%ld)\n",
               timestamp_to_string(current_baseline_ts).c_str(), current_baseline_ts);

        BaselineCollectorFilter baseline_filter =
            load_rib_missing_collectors_for_baseline(raw_data_dir, current_baseline_ts);

        g_ringbuffer_manager = make_unique<LockFreeRingBufferManager>();
        if (!g_ringbuffer_manager->init_producer()) {
            fprintf(stderr, "Failed to initialize RingBuffer producer\n");
            return 1;
        }

        string cache_path = processed_data_dir + "/" + format_processed_dir_timestamp(current_baseline_ts) + "/rib_records_cache.bin";
        if (!filesystem::exists(cache_path)) {
            string legacy_cache = dataset_dir + "/rib_records_cache.bin";
            if (filesystem::exists(legacy_cache)) {
                cache_path = legacy_cache;
            }
        }
        if (!check_rib_cache(cache_path) || !stream_rib_cache_to_ringbuffer(cache_path)) {
            fprintf(stderr, "Failed to stream one-shot baseline cache to RingBuffer\n");
            return 1;
        }

        if (warmup_seconds > 0) {
            if (!process_warmup_upd_batches(raw_data_dir, current_baseline_ts, warmup_seconds, baseline_filter)) {
                fprintf(stderr, "Failed to process one-shot warmup window\n");
                return 1;
            }
        }

        if (!write_signal_to_ringbuffer(current_baseline_ts, STAGE_END_AF)) {
            fprintf(stderr, "Failed to send RIB end signal in one-shot mode\n");
            return 1;
        }

        time_t last_processed_upd = current_baseline_ts - 1;
        while (process_next_upd_batch(raw_data_dir, current_baseline_ts, last_processed_upd, one_shot_end_time, baseline_filter)) {
        }

        if (!write_signal_to_ringbuffer(one_shot_end_time, static_cast<uint8_t>(UPD_WINDOW_END_AF | STAGE_END_AF))) {
            fprintf(stderr, "Failed to send UPD end signal in one-shot mode\n");
            return 1;
        }

        g_ringbuffer_manager->set_finished();
        printf("One-shot window completed. baseline=%s, end=%s\n",
               timestamp_to_string(current_baseline_ts).c_str(),
               timestamp_to_string(one_shot_end_time).c_str());
        return 0;
    }

    // Online mode with baseline switching.
    // Producer writes: [BASELINE_SWITCH_AF] -> RIB cache -> STAGE_END_AF -> UPD batches.
    time_t current_baseline_ts = 0;
    string processed_data_dir = dataset_dir + "/processed_data";
    time_t earliest_raw_ts = find_earliest_available_raw_ts(raw_data_dir);
    time_t startup_anchor = start_time;
    if (earliest_raw_ts > 0 && earliest_raw_ts > startup_anchor) {
        startup_anchor = earliest_raw_ts;
    }

    if (earliest_raw_ts > 0) {
        printf("Earliest raw data timestamp: %s (%ld); startup anchor: %s (%ld)\n",
               timestamp_to_string(earliest_raw_ts).c_str(), earliest_raw_ts,
               timestamp_to_string(startup_anchor).c_str(), startup_anchor);
    } else {
        printf("No raw timestamp detected at startup. Using requested start as anchor: %s (%ld)\n",
               timestamp_to_string(startup_anchor).c_str(), startup_anchor);
    }

    // Persist state next to processed_data to survive restarts.
    state_file_path = processed_data_dir + "/data_parser_state.txt";
    IncrementalState state = read_incremental_state();
    bool use_state_baseline = false;
    if (state.baseline_ts > 0) {
        string state_cache_path = processed_data_dir + "/" +
            format_processed_dir_timestamp(state.baseline_ts) + "/rib_records_cache.bin";
        bool state_cache_valid = check_rib_cache(state_cache_path);
        bool state_stale = (startup_anchor > 0 && state.baseline_ts < startup_anchor);
        bool state_aligned = is_ripe_rib_aligned_ts(state.baseline_ts);

        if (state_cache_valid && !state_stale && state_aligned) {
            current_baseline_ts = state.baseline_ts;
            use_state_baseline = true;
            printf("Using baseline from state file: %s (%ld)\n",
                   timestamp_to_string(current_baseline_ts).c_str(), current_baseline_ts);
        } else {
            printf("Ignoring stale/invalid state baseline: %s (%ld), anchor: %s (%ld), cache_valid=%s, aligned=%s\n",
                   timestamp_to_string(state.baseline_ts).c_str(), state.baseline_ts,
                   timestamp_to_string(startup_anchor).c_str(), startup_anchor,
                   (state_cache_valid ? "true" : "false"),
                   (state_aligned ? "true" : "false"));
        }
    }

    if (!use_state_baseline) {
        time_t max_backtrack_sec = get_baseline_max_backtrack_seconds();
        time_t baseline_target_slot = floor_to_ripe_rib_slot(startup_anchor);
        printf("Searching startup baseline cache in %s around startup anchor %s (%ld)\n",
               processed_data_dir.c_str(),
               timestamp_to_string(startup_anchor).c_str(), startup_anchor);
        printf("RIPE 8h baseline target slot (<= anchor): %s (%ld)\n",
               timestamp_to_string(baseline_target_slot).c_str(), baseline_target_slot);
        if (max_backtrack_sec == LLONG_MAX) {
            printf("Previous baseline rollback limit: disabled\n");
        } else {
            printf("Previous baseline rollback limit: %ld seconds\n", static_cast<long>(max_backtrack_sec));
        }

        while (true) {
            // Prefer previous baseline for better coverage of the requested start period.
            time_t candidate_prev = find_latest_complete_aligned_baseline_le(processed_data_dir, baseline_target_slot);
            bool prev_within_limit = false;
            if (candidate_prev > 0) {
                time_t rollback = baseline_target_slot - candidate_prev;
                if (rollback < 0) {
                    rollback = 0;
                }
                prev_within_limit = (rollback <= max_backtrack_sec);
            }

            if (candidate_prev > 0 && prev_within_limit) {
                current_baseline_ts = candidate_prev;
                break;
            }
            // Fallback when no previous baseline exists.
            time_t candidate_next = find_earliest_complete_aligned_baseline_ge(processed_data_dir, baseline_target_slot);
            if (candidate_next > 0) {
                current_baseline_ts = candidate_next;
                if (candidate_prev > 0 && !prev_within_limit) {
                    printf("Warning: previous baseline %s (%ld) is too old for target slot %s (%ld), using earliest aligned >= slot: %s (%ld)\n",
                           timestamp_to_string(candidate_prev).c_str(), candidate_prev,
                           timestamp_to_string(baseline_target_slot).c_str(), baseline_target_slot,
                           timestamp_to_string(current_baseline_ts).c_str(), current_baseline_ts);
                } else {
                    printf("Warning: no aligned baseline <= target slot. Falling back to earliest aligned >= slot: %s (%ld)\n",
                           timestamp_to_string(current_baseline_ts).c_str(), current_baseline_ts);
                }
                break;
            }

            if (candidate_prev > 0 && !prev_within_limit) {
                printf("Aligned previous baseline exists but is too old: %s (%ld), target slot: %s (%ld). Waiting 30s...\n",
                       timestamp_to_string(candidate_prev).c_str(), candidate_prev,
                       timestamp_to_string(baseline_target_slot).c_str(), baseline_target_slot);
            } else {
                printf("No valid aligned baseline cache found near slot %s. Waiting 30s...\n",
                       timestamp_to_string(baseline_target_slot).c_str());
            }
            this_thread::sleep_for(chrono::seconds(30));
        }
    }

    time_t last_processed_upd = current_baseline_ts - 1;
    if (state.baseline_ts == current_baseline_ts &&
        state.last_processed_upd_ts >= current_baseline_ts - 1) {
        last_processed_upd = state.last_processed_upd_ts;
    }

    printf("Starting online mode at baseline: %s (%ld), last UPD: %s (%ld)\n",
           timestamp_to_string(current_baseline_ts).c_str(), current_baseline_ts,
           timestamp_to_string(last_processed_upd).c_str(), last_processed_upd);

    BaselineCollectorFilter baseline_filter =
        load_rib_missing_collectors_for_baseline(raw_data_dir, current_baseline_ts);

    g_ringbuffer_manager = make_unique<LockFreeRingBufferManager>();
    if (!g_ringbuffer_manager->init_producer()) {
        fprintf(stderr, "Failed to initialize RingBuffer producer\n");
        return 1;
    }
    printf("RingBuffer producer initialized successfully\n");

    string cache_path = processed_data_dir + "/" + format_processed_dir_timestamp(current_baseline_ts) + "/rib_records_cache.bin";
    if (!stream_rib_cache_to_ringbuffer(cache_path)) {
        fprintf(stderr, "Failed to stream initial baseline cache to RingBuffer\n");
        return 1;
    }

    const time_t warmup_seconds = 8 * 60 * 60;


    if (!write_signal_to_ringbuffer(current_baseline_ts, STAGE_END_AF)) {
        fprintf(stderr, "Failed to send RIB end signal\n");
        return 1;
    }

    write_incremental_state({current_baseline_ts, last_processed_upd});

    while (true) {
        bool processed_any = false;
        while (process_next_upd_batch(raw_data_dir, current_baseline_ts, last_processed_upd, LLONG_MAX, baseline_filter)) {
            processed_any = true;
            write_incremental_state({current_baseline_ts, last_processed_upd});
        }

        time_t latest_ready_baseline = find_latest_complete_baseline(processed_data_dir, current_baseline_ts);
        if (latest_ready_baseline > current_baseline_ts) {
            // Strong consistency: flush all UPD with ts < next baseline before switching.
            while (process_next_upd_batch(raw_data_dir, current_baseline_ts, last_processed_upd, latest_ready_baseline, baseline_filter)) {
                processed_any = true;
                write_incremental_state({current_baseline_ts, last_processed_upd});
            }
            printf("Switching baseline from %s (%ld) to %s (%ld)\n",
                   timestamp_to_string(current_baseline_ts).c_str(), current_baseline_ts,
                   timestamp_to_string(latest_ready_baseline).c_str(), latest_ready_baseline);
            if (!write_signal_to_ringbuffer(latest_ready_baseline, BASELINE_SWITCH_AF)) {
                fprintf(stderr, "Failed to send baseline switch signal\n");
                this_thread::sleep_for(chrono::seconds(5));
                continue;
            }
            string next_cache = processed_data_dir + "/" + format_processed_dir_timestamp(latest_ready_baseline) + "/rib_records_cache.bin";
            if (!check_rib_cache(next_cache) || !stream_rib_cache_to_ringbuffer(next_cache)) {
                fprintf(stderr, "Failed to load new baseline cache: %s\n", next_cache.c_str());
                this_thread::sleep_for(chrono::seconds(10));
                continue;
            }
            BaselineCollectorFilter next_baseline_filter =
                load_rib_missing_collectors_for_baseline(raw_data_dir, latest_ready_baseline);
            if (warmup_seconds > 0) {
                if (!process_warmup_upd_batches(raw_data_dir, latest_ready_baseline, warmup_seconds, next_baseline_filter)) {
                    fprintf(stderr, "Failed warmup during baseline switch\n");
                    this_thread::sleep_for(chrono::seconds(10));
                    continue;
                }
            }
            if (!write_signal_to_ringbuffer(latest_ready_baseline, STAGE_END_AF)) {
                fprintf(stderr, "Failed to send RIB end signal for new baseline\n");
                this_thread::sleep_for(chrono::seconds(5));
                continue;
            }
            current_baseline_ts = latest_ready_baseline;
            last_processed_upd = current_baseline_ts - 1;
            baseline_filter = std::move(next_baseline_filter);
            write_incremental_state({current_baseline_ts, last_processed_upd});
            printf("Baseline switch completed. New UPD replay starts from %s (%ld).\n",
                   timestamp_to_string(current_baseline_ts).c_str(), current_baseline_ts);
            continue;
        }

        if (processed_any) {
            this_thread::sleep_for(chrono::seconds(1));
        } else {
            this_thread::sleep_for(chrono::seconds(30));
        }
    }
}
