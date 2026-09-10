#pragma once
#include <getopt.h>
#include <unordered_map>
#include <regex>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <memory>
#include <arpa/inet.h>  // Add inet_ntop support
#include "lockfree_ringbuffer.hpp"
#include "bgpstream_parser.hpp"  // Introduce new BGPStream parser

using namespace std;

constexpr auto RETRY_INTERVAL_US = chrono::milliseconds(20);

// ====================== Global Variables ======================
extern std::unique_ptr<LockFreeRingBufferManager> g_ringbuffer_manager;
inline std::regex upd_file_regexp("updates\\.([0-9]{8})\\.([0-9]{4})-([0-9]{2})");
inline std::regex rib_file_regexp("bview\\.([0-9]{8})\\.([0-9]{4})-([0-9]{2})");
inline std::regex upd_file_regexp_new("upd\\.(ris|rv)\\.([0-9]{8})\\.([0-9]{4})\\..+");
inline std::regex rib_file_regexp_new("rib\\.(ris|rv)\\.([0-9]{8})\\.([0-9]{4})\\..+");

// ====================== BGP Processing Functions ======================

// Extract the route prefix with the mask length into a BGP record
inline bool extract_prefix_info(bgpstream_elem_t* elem, BGPRecord& record) {
    char prefix_buf[INET6_ADDRSTRLEN];
    
    // IPv4
    if (elem->prefix.address.version == BGPSTREAM_ADDR_VERSION_IPV4) {
        bgpstream_ipv4_pfx_t& ipv4 = elem->prefix.bs_ipv4;
        // record.addr_family = 4; 
        // record.prefix_len = ipv4.mask_len;
        // // Store to IPv4Prefix structure
        // record.prefix.ipv4 = ipv4.address.addr.s_addr;
        record.prefix = Prefix(4, ipv4.address.addr.s_addr, ipv4.mask_len);
    } else if (elem->prefix.address.version == BGPSTREAM_ADDR_VERSION_IPV6) {
        // IPv6
        bgpstream_ipv6_pfx_t& ipv6 = elem->prefix.bs_ipv6;
        record.prefix = Prefix(6, ipv6.address.addr.s6_addr, ipv6.mask_len);
    } else {
        // Unsupported address family
        return false;
    }
    return true;
}

// Parse AS segments and get internal AS links and get new ASNs should link with other segments
inline vector<uint32_t> parse_as_segment(bgpstream_as_path_seg_t* seg, 
        const vector<uint32_t>& bound_asns, ASPair* pairs, uint8_t& count) {
    vector<uint32_t> new_bound_asns;
    
    // lambda function to insert AS pair into the record
    auto insert_as_pair = [&](const uint32_t& from_as, const uint32_t& to_as) {
        if (count < AS_PAIR_SIZE) {
            pairs[count].from_as = from_as;
            pairs[count].to_as = to_as;
            count++;
        }
    };
    
    // The input bound_asns are always `to_asn` because the AS PATH is stored in reverse order
    if (seg->type == BGPSTREAM_AS_PATH_SEG_ASN) {
        bgpstream_as_path_seg_asn_t* asn_seg = (bgpstream_as_path_seg_asn_t*)seg;
        for(auto asn : bound_asns) {
            insert_as_pair(asn_seg->asn, asn);
        }
        new_bound_asns.push_back(asn_seg->asn);
    } else if (seg->type == BGPSTREAM_AS_PATH_SEG_SET || seg->type == BGPSTREAM_AS_PATH_SEG_CONFED_SET) {
        bgpstream_as_path_seg_set_t* set_seg = (bgpstream_as_path_seg_set_t*)seg;
        for (int i = 0; i < set_seg->asn_cnt; ++i) {
            for(auto asn : bound_asns) {
                insert_as_pair(set_seg->asn[i], asn);
            }
            new_bound_asns.push_back(set_seg->asn[i]);
        }
    } else if (seg->type == BGPSTREAM_AS_PATH_SEG_CONFED_SEQ) {
        // The most complex case: a sequence of ASNs
        // We need to generate all possible AS pairs within this segment with *RIGHT ORDER*
        bgpstream_as_path_seg_set_t* seq_seg = (bgpstream_as_path_seg_set_t*)seg;
        // internal links
        for (int i = seq_seg->asn_cnt - 1; i > 0; --i) {
            insert_as_pair(seq_seg->asn[i], seq_seg->asn[i - 1]);
        }
        new_bound_asns.push_back(seq_seg->asn[seq_seg->asn_cnt - 1]);
        // outter links
        for(auto asn : bound_asns) {
            insert_as_pair(seq_seg->asn[0], asn);
        }
    }
    return new_bound_asns;
}

// Extract the as_path information into a pairs buffer
// Returns true on success, false on failure (e.g. invalid path)
inline bool extract_as_path_info(bgpstream_elem_t* elem, ASPair* pairs, uint8_t& count) {
    count = 0;
    
    unique_ptr<bgpstream_as_path_iter_t> path_iter(new bgpstream_as_path_iter_t);
    bgpstream_as_path_iter_reset(path_iter.get());
    bgpstream_as_path_seg_t* cur_seg = nullptr;
    vector<uint32_t> bound_asns; // Boundary ASNs of current segment
    uint32_t hash_val = 0;

    cur_seg = bgpstream_as_path_get_next_seg(elem->as_path, path_iter.get());
    if (cur_seg) {
        hash_val = bgpstream_as_path_seg_hash(cur_seg);
        bound_asns = parse_as_segment(cur_seg, bound_asns, pairs, count);

        // Traverse all segments of AS path, construct all possible AS Pairs
        while ((cur_seg = bgpstream_as_path_get_next_seg(elem->as_path, path_iter.get())) != nullptr) {
            uint32_t cur_hash = bgpstream_as_path_seg_hash(cur_seg);
            if (cur_hash == hash_val) {
                continue;
            }
            hash_val = cur_hash;
            bound_asns = parse_as_segment(cur_seg, bound_asns, pairs, count);
        }
    }
    return true;
}

inline bool write_one_record_to_ringbuffer(const BGPRecord& record) {
    int retry_count = 0;
    while (!g_ringbuffer_manager->write(record)) {
        retry_count++;
        this_thread::sleep_for(RETRY_INTERVAL_US);
        if (retry_count > 1000) {
            return false;
        }
    }
    return true;
}

inline bool write_signal_to_ringbuffer(const time_t& timestamp = 0, uint8_t signal_type = STAGE_END_AF) {
    int retry_count = 0;
    // Calculate alignment requirements, build padding records
    size_t alignment_records = g_ringbuffer_manager->get_write_alignment_records_needed();
    size_t padding_num = alignment_records == 0 ? BATCH_SIZE : alignment_records;
    vector<BGPRecord> padding_records(padding_num);
    
    // Set the last record as stage end signal
    padding_records[padding_num - 1].signal = signal_type;
    padding_records[padding_num - 1].timestamp = timestamp;

    // Use batch write to fill the entire batch to ensure memory alignment
    while (!g_ringbuffer_manager->write_batch(padding_records.data(), padding_num)) {
        retry_count++;
        this_thread::sleep_for(RETRY_INTERVAL_US);
        if (retry_count > 1000) {
            return false;
        }
    }
    return true;
}

inline bool write_batch_to_ringbuffer(const std::vector<BGPRecord>& records, bool need_align = false) {    
    if (records.empty()) return true;
    size_t total_records = records.size();
    
    if (!need_align) {
        int retry_count = 0;    
        // Directly write the entire batch without alignment
        while (!g_ringbuffer_manager->write_batch(records.data(), total_records)) {
            retry_count++;
            this_thread::sleep_for(RETRY_INTERVAL_US);

            if (retry_count % 10000 == 0) {
                fprintf(stderr, "Warning: RingBuffer full, retrying... (attempt %d)\n", retry_count);
            }
        }
    } else {
        // Calculate alignment requirements, build padding records
        size_t written = 0;
        size_t alignment_records = g_ringbuffer_manager->get_write_alignment_records_needed();
        size_t align_num = alignment_records == 0 ? BATCH_SIZE : alignment_records;
        while (written < total_records) {
            size_t remaining = total_records - written;
            size_t current_batch_size = min(remaining, align_num);
            int retry_count = 0;
            bool success = false;
            
            while (!success) {
                success = g_ringbuffer_manager->write_batch(
                    records.data() + written, 
                    current_batch_size
                );
                if (!success) {
                    retry_count++;
                    this_thread::sleep_for(chrono::milliseconds(10));
                    if (retry_count % 10000 == 0) {
                        fprintf(stderr, "Warning: RingBuffer full (aligned write), retrying... (attempt %d)\n", retry_count);
                    }
                }
            }
            
            written += current_batch_size;
            align_num = BATCH_SIZE; // After the first aligned batch, use standard batch size
        }
    }
    return true; // Add missing return statement
}

// Check whether a BGP element carries RFC 7999 blackhole community.
inline bool has_blackhole_community(bgpstream_elem_t* elem) {
    if (!elem || !elem->communities) return false;
    const bgpstream_community_t* comm;
    int set_size = bgpstream_community_set_size(elem->communities);
    for (int i = 0; i < set_size; ++i) {
        comm = bgpstream_community_set_get(elem->communities, i);
        if (comm->asn == 65535 && comm->value == 666) return true;
    }
    return false;
}

inline bool process_bgp_element(const time_t& timestamp, bgpstream_elem_t* elem, BGPRecord& bgp_record) {    
    // Extract prefix information
    if (!extract_prefix_info(elem, bgp_record)) {
        return false; // Skip invalid prefix, continue processing
    }

    bgp_record.timestamp = timestamp;
    bgp_record.peer_asn = elem->peer_asn;

    // Keep withdrawal events so downstream logic can detect complete-withdraw -> new-origin transitions.
    if (elem->type == BGPSTREAM_ELEM_TYPE_WITHDRAWAL) {
        bgp_record.route_event = ROUTE_EVENT_WITHDRAW;
        bgp_record.as_pair_count = 0;
        bgp_record.origin_as = 0;
        return true;
    }

    bgp_record.route_event = ROUTE_EVENT_ANNOUNCE;

    // Flag routes that carry a blackhole community (e.g. RFC 7999 65535:666).
    if (has_blackhole_community(elem)) {
        bgp_record.community_label = COMMUNITY_BLACKHOLE;
    }
    
    uint32_t origin_asn = 0;
    int origin_result = bgpstream_as_path_get_origin_val(elem->as_path, &origin_asn);
    if (origin_result == 0) {
        bgp_record.origin_as = origin_asn;
    } else {
        return false;
    }

    // Extract AS path information
    if (!extract_as_path_info(elem, bgp_record.as_pairs, bgp_record.as_pair_count)) {
        return false; // Skip records where AS path extraction failed
    }
    return true;
}

// Unified BGP parser initialization function - reduce code duplication
inline std::unique_ptr<BGPStreamParser> initialize_parser(const string& file_path, const bool& is_rib) {
    try {
        // Use C++ style BGPStream parser
        auto parser = is_rib ? 
            BGPStreamParserBuilder::for_rib_file(file_path).build() :
            BGPStreamParserBuilder::for_update_file(file_path).build();

        if (!parser->initialize()) {
            fprintf(stderr, "ERROR: Failed to initialize BGPStream parser for file: %s\n", file_path.c_str());
            return nullptr;
        }

        return parser;
    } catch (const exception& e) {
        fprintf(stderr, "ERROR: Failed to create BGPStream parser for file %s: %s\n", file_path.c_str(), e.what());
        return nullptr;
    }
}

inline int parse_one_file_to_container(const string& file_path, const time_t& fallback_timestamp, 
                                    const bool& is_rib, std::vector<BGPRecord>& container) {
    // Use unified parser initialization function
    auto parser = initialize_parser(file_path, is_rib);
    if (!parser) {
        return -1;
    }

    int element_count = 0;
    int record_count = 0;

    // Parse each record and add directly to container
    record_count = parser->parse_records([&](bgpstream_record_t* record) -> bool {
        const time_t rec_timestamp = record->time_sec != 0 ? record->time_sec : fallback_timestamp;
        bgpstream_elem_t* elem = nullptr;
        int read_res;
        while ((read_res = bgpstream_record_get_next_elem(record, &elem))) {
            BGPRecord bgp_record{};
            if (process_bgp_element(rec_timestamp, elem, bgp_record)) {
                container.push_back(bgp_record);
                element_count++;
            }
        }
        return true; // Continue processing next record
    });
    
    if (record_count < 0) {
        fprintf(stderr, "ERROR: BGPStream record parsing failed for file: %s\n", file_path.c_str());
        return -1;
    }
    
    return element_count; // Return the number of elements processed
}

inline int parse_one_file_to_ringbuffer(const string& file_path, const time_t& fallback_timestamp, const bool& is_rib) {    
    // Use unified parser initialization function
    auto parser = initialize_parser(file_path, is_rib);
    if (!parser) {
        return -1;
    }

    int element_count = 0;
    int record_count = 0;
    int failed_batches = 0;
    
    size_t alignment_records = g_ringbuffer_manager->get_write_alignment_records_needed();
    size_t current_threshold = (alignment_records == 0) ? BATCH_SIZE : alignment_records;
    
    // Batch buffer to reduce RingBuffer write operations
    std::vector<BGPRecord> batch_buffer;
    batch_buffer.reserve(BATCH_SIZE);
    
    auto flush_batch = [&]() -> void {
        if (batch_buffer.empty()) return;
        write_batch_to_ringbuffer(batch_buffer, false);
        batch_buffer.clear();
        current_threshold = BATCH_SIZE; // After the first aligned batch, use standard batch size
    };

    // Parse each record and batch write to RingBuffer
    record_count = parser->parse_records([&](bgpstream_record_t* record) -> bool {
        // Determine record timestamp, prefer record's own timestamp
        const time_t rec_timestamp = record->time_sec != 0 ? record->time_sec : fallback_timestamp;
        bgpstream_elem_t* elem = nullptr;
        int read_res;
        while ((read_res = bgpstream_record_get_next_elem(record, &elem))) {
            BGPRecord bgp_record{};
            if (process_bgp_element(rec_timestamp, elem, bgp_record)) {
                batch_buffer.push_back(bgp_record);
                element_count++;
                
                // Use different batch size strategies
                if (batch_buffer.size() == current_threshold) {
                    flush_batch();
                }
            }
        }
        return true; // Continue processing next record
    });
    
    // Write the last batch of data
    if (!batch_buffer.empty()) {
        flush_batch();
    }
        
    if (record_count < 0) {
        fprintf(stderr, "ERROR: BGPStream record parsing failed for file: %s\n", file_path.c_str());
        return -1;
    }
    
    return element_count; // Return the number of processed elements
}

// Extract timestamp and RRC from filename (in UTC)
inline int extract_timestamp_rrc(const string &filename, time_t &timestamp, int &rrc_id, const int& file_type) {
    smatch match;

    if ((file_type == 0 && regex_search(filename, match, rib_file_regexp)) ||
        (file_type == 1 && regex_search(filename, match, upd_file_regexp))) {

        string date_str = match[1].str();
        string time_str = match[2].str();
        
        rrc_id = stoi(match[3].str());
        
        // Parse date and time
        struct tm tm_time = {};
        int year = stoi(date_str.substr(0, 4));
        int month = stoi(date_str.substr(4, 2));
        int day = stoi(date_str.substr(6, 2));
        int hour = stoi(time_str.substr(0, 2));
        int minute = stoi(time_str.substr(2, 2));
        
        tm_time.tm_year = year - 1900;
        tm_time.tm_mon = month - 1;
        tm_time.tm_mday = day;
        tm_time.tm_hour = hour;
        tm_time.tm_min = minute;
        tm_time.tm_sec = 0;
        
        timestamp = timegm(&tm_time);  // Use timegm to handle UTC time, not mktime for local time
        return 0;
    }

    // New downloader naming support:
    //  - upd.ris.<YYYYmmdd>.<HHMM>.<collector>
    //  - upd.rv.<YYYYmmdd>.<HHMM>.<collector>
    //  - rib.ris.<YYYYmmdd>.<HHMM>.<collector>
    //  - rib.rv.<YYYYmmdd>.<HHMM>.<collector>
    if ((file_type == 0 && regex_search(filename, match, rib_file_regexp_new)) ||
        (file_type == 1 && regex_search(filename, match, upd_file_regexp_new))) {

        string source = match[1].str();
        string date_str = match[2].str();
        string time_str = match[3].str();

        // Keep old numeric field semantics for compatibility where needed.
        rrc_id = (source == "ris") ? 0 : 1;

        struct tm tm_time = {};
        int year = stoi(date_str.substr(0, 4));
        int month = stoi(date_str.substr(4, 2));
        int day = stoi(date_str.substr(6, 2));
        int hour = stoi(time_str.substr(0, 2));
        int minute = stoi(time_str.substr(2, 2));

        tm_time.tm_year = year - 1900;
        tm_time.tm_mon = month - 1;
        tm_time.tm_mday = day;
        tm_time.tm_hour = hour;
        tm_time.tm_min = minute;
        tm_time.tm_sec = 0;

        timestamp = timegm(&tm_time);
        return 0;
    }

    return -1; // Failed to parse
}
