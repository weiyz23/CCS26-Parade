#ifndef TOPOLOGY_ANALYZER_HPP
#define TOPOLOGY_ANALYZER_HPP

#include <cstdio>
#include <thread>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <stack>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <limits>
#include <ctime>
#include <cmath>
#include <functional>
#include <bitset>
#include <sys/stat.h>  // for mkdir
#include <filesystem>
#include <cstdarg>
#include <algorithm>

#include "lockfree_ringbuffer.hpp"
#include "propagation_graph.hpp"
#include "my_queue.hpp"
#include <climits>

using namespace std;

constexpr size_t DETECT_INTERVAL = 20;
constexpr size_t MAX_BITSET_SIZE = 2048;
constexpr size_t MAX_SCC_COUNT = MAX_BITSET_SIZE - 1; // Reserve one bit for computed flag
constexpr time_t ANOMALY_BASELINE_RETENTION_SECONDS = 30 * 24 * 60 * 60;

const char* TEST_LOG_DIR = "./test_log/";

/*
 * Hash function for pair<Prefix, uint32_t>
 */
struct pair_hash {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ (hash2 << 1);  // Simple combination to avoid collisions
    }
};

/*
 * One anomaly record to be written to the CSV output.
 */
struct ASAnomalyRecord {
    time_t timestamp;
    Prefix prefix;
    uint32_t asn;              // The suspicious ASN.
    uint32_t level;            // Output level used by downstream consumers.
    uint32_t topology_level;   // Raw topological level in the SCC DAG.
    uint32_t origin_as;        // Route origin ASN observed with the triggering link.
    uint8_t is_origin_asn;     // Whether the suspicious ASN matches the route origin ASN.
    uint8_t is_blackhole_route; // Whether this anomaly was triggered by a blackhole-community route.
    uint8_t is_ip_leasing;     // Whether this anomaly matches IP leasing pattern.
    string upstream_asns_str;  // Colon-separated upstream ASNs.
    string affected_asns_str;  // Colon-separated affected downstream ASNs.
};

/*
 * BGP Anomaly Detector
 * - Independent anomaly detection thread, real-time analysis of prefix-AS link changes
 * - Supports snapshot comparison graph anomaly detection
 * - Batch aggregation processing to improve detection efficiency
 */

// Prefix anomaly detection event structure - supports batch aggregation
struct PrefixAnomalyBatch {
    // New prefix AS links (with timestamp and Origin AS)
    unordered_map<Prefix, vector<ASPairWithTimeAndOrigin>> new_links;
    // Prefixes that carried a blackhole community in this batch.
    unordered_set<Prefix> blackhole_prefixes;
    // Prefixes tagged as IP leasing in this batch.
    unordered_set<Prefix> ip_leasing_prefixes;
    time_t batch_timestamp;
    uint64_t batch_id;

    PrefixAnomalyBatch() : batch_timestamp(0), batch_id(0) {}

    PrefixAnomalyBatch(time_t ts, uint64_t id) : batch_timestamp(ts), batch_id(id) {}

    PrefixAnomalyBatch(time_t ts, uint64_t id,
                        const unordered_map<Prefix, vector<ASPairWithTimeAndOrigin>>& links)
            : batch_timestamp(ts), batch_id(id), new_links(links) {}

    // Move constructor form to avoid large batch copies
    PrefixAnomalyBatch(time_t ts, uint64_t id,
                        unordered_map<Prefix, vector<ASPairWithTimeAndOrigin>>&& links)
            : batch_timestamp(ts), batch_id(id), new_links(std::move(links)) {}

    void addASLink(const Prefix& prefix, const ASPairWithTimeAndOrigin& link_with_ts) {
        new_links[prefix].emplace_back(link_with_ts);
    }

    void addBatchedASLink(const Prefix& prefix, const vector<ASPairWithTimeAndOrigin>& links_with_ts) {
        auto& vec = new_links[prefix];
        vec.insert(vec.end(), links_with_ts.begin(), links_with_ts.end());
    }

    bool isEmpty() const {
        return new_links.empty();
    }

    size_t getTotalLinkCount() const {
        size_t count = 0;
        for (const auto& [prefix, links] : new_links) {
            count += links.size();
        }
        return count;
    }
};

// Batch timestamp comparator
struct BatchTimestampComparator {
    bool operator()(const PrefixAnomalyBatch& a, const PrefixAnomalyBatch& b) const {
        return a.batch_timestamp > b.batch_timestamp; // Min heap: smaller timestamp has higher priority
    }
};

class TopologyAnalyzer {
private:
    const PrefixPropGraph& graph_;
    PriorityMyQueue<PrefixAnomalyBatch, BatchTimestampComparator> anomaly_queue_;
    size_t num_threads_;
    vector<thread> detection_threads_;
    atomic<bool> running_{true};
    atomic<uint64_t> in_flight_batches_{0};

    // CSV output related
    string csv_filename_;
    vector<ASAnomalyRecord> anomaly_records_;
    mutable mutex records_mutex_;  // Protect record list
    unordered_map<pair<Prefix, uint32_t>, time_t, pair_hash> recorded_asns_;  // prefix-ASN -> last emitted ts
    mutable mutex recorded_asns_mutex_;  // Protect deduplication set

    // Data merging and scheduled detection related members
    mutable mutex accumulated_data_mutex_;
    unordered_map<Prefix, unordered_map<ASPair, pair<time_t, uint32_t>>> accumulated_links_;
    unordered_set<Prefix> accumulated_blackhole_prefixes_;
    unordered_set<Prefix> accumulated_ip_leasing_prefixes_;
    time_t last_detection_time_{0};
    mutable mutex time_mutex_;

    // Added: Prefix AS Link history records for centrality calculation
    // Sharded Map to reduce lock contention
    static constexpr size_t NUM_SHARDS = 16;
    struct HistoryShard {
        mutable mutex mtx;
        unordered_map<Prefix, pair<unordered_set<ASPair>, time_t>> history;
        // Add padding to avoid false sharing
        char padding[64]; 
    };
    vector<unique_ptr<HistoryShard>> history_shards_;
    static constexpr time_t HISTORY_EXPIRY_SECONDS = 2 * 60 * 60; // 2 hours expiry time

    // Get shard index for prefix
    size_t getShardIndex(const Prefix& prefix) const {
        return std::hash<Prefix>{}(prefix) % NUM_SHARDS;
    }

    void detectionLoop() {
        PrefixAnomalyBatch batch;
        while (running_.load(memory_order_relaxed) || anomaly_queue_.size() > 0) {
            if (anomaly_queue_.dequeue(batch, chrono::milliseconds(100))) {
                in_flight_batches_.fetch_add(1, memory_order_relaxed);
                processAnomalyBatch(batch);
                in_flight_batches_.fetch_sub(1, memory_order_relaxed);
            }
        }
    }

    void processAnomalyBatch(const PrefixAnomalyBatch& batch) {
        {
            lock_guard<mutex> lock(accumulated_data_mutex_);

            // Merge prefix's new AS links into accumulated data (record minimum timestamp and Origin AS)
            for (const auto& [prefix, new_links] : batch.new_links) {
                auto& accumulated_links = accumulated_links_[prefix];
                for (const auto& link_with_ts : new_links) {
                    auto it = accumulated_links.find(link_with_ts.as_pair);
                    if (it == accumulated_links.end()) {
                        // New AS Pair, insert directly
                        accumulated_links[link_with_ts.as_pair] = {link_with_ts.timestamp, link_with_ts.origin_as};
                    } else {
                        // Existing AS Pair, keep minimum timestamp, but update Origin AS (if timestamp is same or newer)
                        if (link_with_ts.timestamp < it->second.first) {
                            it->second = {link_with_ts.timestamp, link_with_ts.origin_as};
                        }
                    }
                }
            }

            accumulated_blackhole_prefixes_.insert(
                batch.blackhole_prefixes.begin(), batch.blackhole_prefixes.end());
            accumulated_ip_leasing_prefixes_.insert(
                batch.ip_leasing_prefixes.begin(), batch.ip_leasing_prefixes.end());
        }

        // Check if anomaly detection needs to be executed (based on time interval)
        time_t current_time = batch.batch_timestamp;
        bool trigger_detection = false;
        {
            lock_guard<mutex> lock(time_mutex_);
            if (current_time >= last_detection_time_ + DETECT_INTERVAL) {
                last_detection_time_ = current_time;
                trigger_detection = true;
            }
        }
        
        if (trigger_detection) {
            executeAnomalyDetection(current_time);
        }
    }

    // Update prefix history records and clean up expired data
    void updatePrefixHistory(const unordered_map<Prefix, unordered_map<ASPair, pair<time_t, uint32_t>>>& links,
                            time_t current_time) {        
        for (const auto& [prefix, links_map] : links) {
            size_t idx = getShardIndex(prefix);
            auto& shard = *history_shards_[idx];
            {
                lock_guard<mutex> lock(shard.mtx);
                auto& [as_links, last_access] = shard.history[prefix];
                for (const auto& [as_pair, ts_origin] : links_map) {
                    as_links.insert(as_pair);
                }
                last_access = current_time;
            }
        }

        // Lazy cleanup of expired data, check only one shard per call
        static atomic<size_t> cleanup_idx{0};
        size_t idx = cleanup_idx.fetch_add(1, memory_order_relaxed) % NUM_SHARDS;
        cleanupExpiredHistoryShard(idx, current_time);
    }
    
    // Clean up expired history records for specified shard
    void cleanupExpiredHistoryShard(size_t idx, time_t current_time) {
        auto& shard = *history_shards_[idx];
        lock_guard<mutex> lock(shard.mtx);
        for (auto it = shard.history.begin(); it != shard.history.end(); ) {
            if (current_time - it->second.second > HISTORY_EXPIRY_SECONDS) {
                it = shard.history.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Check all newly generated prefix-AS links, execute anomaly detection
    void executeAnomalyDetection(time_t timestamp) {
        unordered_map<Prefix, unordered_map<ASPair, pair<time_t, uint32_t>>> links_to_check;
        unordered_set<Prefix> blackhole_prefixes;
        unordered_set<Prefix> ip_leasing_prefixes;

        // Get accumulated data and clear cache
        {
            lock_guard<mutex> lock(accumulated_data_mutex_);
            links_to_check = std::move(accumulated_links_);
            accumulated_links_.clear();
            blackhole_prefixes = std::move(accumulated_blackhole_prefixes_);
            accumulated_blackhole_prefixes_.clear();
            ip_leasing_prefixes = std::move(accumulated_ip_leasing_prefixes_);
            accumulated_ip_leasing_prefixes_.clear();
        }

        // Update prefix history records and clean up expired data
        updatePrefixHistory(links_to_check, timestamp);

        // Execute topology analysis, record details of newly appeared AS Links
        vector<ASAnomalyRecord> local_records;
        analyzeCriticalAS(links_to_check, blackhole_prefixes, ip_leasing_prefixes, local_records);
        
        if (!local_records.empty()) {
            lock_guard<mutex> lock(records_mutex_);
            anomaly_records_.insert(anomaly_records_.end(), local_records.begin(), local_records.end());
        }
    }

    // AS topology analysis function: build AS graph and record critical anomalous AS Links and origin AS for each prefix
    void analyzeCriticalAS(const unordered_map<Prefix, 
        unordered_map<ASPair, pair<time_t, uint32_t>>>& prefix_events,
        const unordered_set<Prefix>& blackhole_prefixes,
        const unordered_set<Prefix>& ip_leasing_prefixes,
        vector<ASAnomalyRecord>& records) {

        if (prefix_events.empty()) return;

        // Parse all prefixes, find AS Links with highest centrality for them
        for (const auto& [prefix, as_pairs_with_ts] : prefix_events) {
            if (as_pairs_with_ts.empty()) continue;

            // Get historical AS links for this prefix
            unordered_set<ASPair> historical_links;
            {
                size_t idx = getShardIndex(prefix);
                auto& shard = *history_shards_[idx];
                lock_guard<mutex> lock(shard.mtx);
                auto it = shard.history.find(prefix);
                if (it != shard.history.end()) {
                    historical_links = it->second.first;
                }
            }

            // Merge historical and new links to build complete graph
            unordered_set<ASPair> all_links = historical_links;
            for (const auto& [as_pair, ts_origin] : as_pairs_with_ts) {
                all_links.insert(as_pair);
            }

            // Use AS level scoring to find critical AS and corresponding origin AS
            unordered_map<uint32_t, tuple<time_t, uint32_t, uint32_t>> as_info; // asn -> (timestamp, level, origin_as)
            unordered_map<uint32_t, vector<uint32_t>> asn_upstream;
            unordered_map<uint32_t, vector<uint32_t>> asn_affected_downstream;
            unordered_map<uint32_t, uint32_t> as_origins;
            unordered_set<ASPair> to_record;
            findMostCriticalAS(all_links, as_pairs_with_ts, as_info, asn_upstream, asn_affected_downstream);

            // Derive a semantic level from graph distance to origin on the merged prefix graph.
            unordered_map<uint32_t, vector<uint32_t>> merged_adj;
            for (const auto& as_pair : all_links) {
                merged_adj[as_pair.from_as].push_back(as_pair.to_as);
                if (merged_adj.find(as_pair.to_as) == merged_adj.end()) {
                    merged_adj[as_pair.to_as] = {};
                }
            }

            unordered_map<uint32_t, unordered_map<uint32_t, uint32_t>> origin_dist_cache;
            auto get_origin_distance_map = [&](uint32_t origin_as) -> const unordered_map<uint32_t, uint32_t>& {
                auto it = origin_dist_cache.find(origin_as);
                if (it != origin_dist_cache.end()) {
                    return it->second;
                }

                unordered_map<uint32_t, uint32_t> dist;
                queue<uint32_t> bfs;
                dist[origin_as] = 0;
                bfs.push(origin_as);

                while (!bfs.empty()) {
                    uint32_t u = bfs.front();
                    bfs.pop();
                    auto adj_it = merged_adj.find(u);
                    if (adj_it == merged_adj.end()) {
                        continue;
                    }
                    for (uint32_t nxt : adj_it->second) {
                        if (dist.find(nxt) == dist.end()) {
                            dist[nxt] = dist[u] + 1;
                            bfs.push(nxt);
                        }
                    }
                }

                auto [inserted_it, inserted] = origin_dist_cache.emplace(origin_as, std::move(dist));
                return inserted_it->second;
            };

            unordered_map<uint32_t, uint32_t> semantic_level_by_asn;
            for (const auto& [asn, info] : as_info) {
                uint32_t topology_level = get<1>(info);
                uint32_t origin_as = get<2>(info);
                uint32_t semantic_level = topology_level;
                if (origin_as != 0) {
                    const auto& dist_map = get_origin_distance_map(origin_as);
                    auto d_it = dist_map.find(asn);
                    if (d_it != dist_map.end()) {
                        semantic_level = d_it->second + 1;
                    } else if (asn == origin_as) {
                        semantic_level = 1;
                    }
                }
                semantic_level_by_asn[asn] = semantic_level;
            }

            auto asn_list_to_string = [](const vector<uint32_t>& asns) -> string {
                string output;
                for (size_t i = 0; i < asns.size(); ++i) {
                    if (i > 0) output += ":";
                    output += to_string(asns[i]);
                }
                return output;
            };

            // After deduplication, record critical AS for subsequent anomaly detection
            {
                lock_guard<mutex> lock(recorded_asns_mutex_);
                uint8_t is_bh = static_cast<uint8_t>(blackhole_prefixes.count(prefix) > 0);
                uint8_t is_ip_leasing = static_cast<uint8_t>(ip_leasing_prefixes.count(prefix) > 0);
                for (const auto& [asn, info] : as_info) {
                    pair<Prefix, uint32_t> key = {prefix, asn};
                    time_t ts = get<0>(info);
                    auto it = recorded_asns_.find(key);
                    if (it != recorded_asns_.end() && ts <= it->second + ANOMALY_BASELINE_RETENTION_SECONDS) {
                        continue;
                    }

                    recorded_asns_[key] = ts;
                    const vector<uint32_t> empty_list;
                    const auto& ups = asn_upstream.count(asn) ? asn_upstream[asn] : empty_list;
                    const auto& down = asn_affected_downstream.count(asn) ? asn_affected_downstream[asn] : empty_list;
                    uint32_t topology_level = get<1>(info);
                    uint32_t origin_as = get<2>(info);
                    uint32_t semantic_level = semantic_level_by_asn.count(asn)
                        ? semantic_level_by_asn[asn]
                        : topology_level;
                    records.push_back({
                        ts,
                        prefix,
                        asn,
                        semantic_level,
                        topology_level,
                        origin_as,
                        static_cast<uint8_t>(asn == origin_as),
                        is_bh,
                        is_ip_leasing,
                        asn_list_to_string(ups),
                        asn_list_to_string(down),
                    });
                }
            }
        }
    }

    // Helper function: reconstruct AS topology graph based on AS Links, calculate level of each AS, and decide which AS to record
    void findMostCriticalAS(const unordered_set<ASPair>& as_links, const unordered_map<ASPair, pair<time_t, uint32_t>>& as_pairs_with_ts,
        unordered_map<uint32_t, tuple<time_t, uint32_t, uint32_t>>& as_info,
        unordered_map<uint32_t, vector<uint32_t>>& asn_upstream,
        unordered_map<uint32_t, vector<uint32_t>>& asn_affected_downstream) {
        if (as_links.empty()) return;

        // Build directed graph: from_as -> to_as
        unordered_map<uint32_t, vector<uint32_t>> adj;
        unordered_map<uint32_t, vector<uint32_t>> rev_adj; // Reverse graph: to_as -> from_as
        unordered_map<uint32_t, int> indegree;
        unordered_set<uint32_t> nodes;

        for (const auto& as_pair : as_links) {
            adj[as_pair.from_as].push_back(as_pair.to_as);
            rev_adj[as_pair.to_as].push_back(as_pair.from_as);
            indegree[as_pair.to_as]++;
            if (indegree.find(as_pair.from_as) == indegree.end()) indegree[as_pair.from_as] = 0;
            nodes.insert(as_pair.from_as);
            nodes.insert(as_pair.to_as);
        }

        // Ensure all nodes have entries in adj and rev_adj
        for (uint32_t node : nodes) {
            adj[node];
            rev_adj[node];
        }

        vector<uint32_t> node_list(nodes.begin(), nodes.end());
        size_t N = node_list.size();
        if (N == 0) return;

        // Tarjan algorithm to compute strongly connected components
        vector<vector<uint32_t>> sccs;
        unordered_map<uint32_t, int> node_to_scc;
        tarjanSCC(adj, node_list, sccs, node_to_scc);

        size_t S = sccs.size();
        if (S == 0) return;

        // Build SCC graph (DAG)
        vector<vector<int>> scc_adj(S);
        vector<vector<int>> scc_rev_adj(S);
        vector<int> scc_nodes(S);
        for (size_t i = 0; i < S; ++i) scc_nodes[i] = i;

        for (const auto& as_pair : as_links) {
            int scc_u = node_to_scc[as_pair.from_as];
            int scc_v = node_to_scc[as_pair.to_as];
            if (scc_u != scc_v) {
                scc_adj[scc_u].push_back(scc_v);
                scc_rev_adj[scc_v].push_back(scc_u);
            }
        }

        // Calculate SCC levels (using standard Kahn algorithm topological sort)
        vector<int> scc_level(S, 0);

        queue<int> q;
        vector<int> scc_indegree(S, 0);
        for (int scc = 0; scc < S; ++scc) {
            scc_indegree[scc] = scc_rev_adj[scc].size();
            if (scc_indegree[scc] == 0) {
                scc_level[scc] = 1;
                q.push(scc);
            }
        }

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : scc_adj[u]) {
                scc_level[v] = max(scc_level[v], scc_level[u] + 1);
                scc_indegree[v]--;
                if (scc_indegree[v] == 0) {
                    q.push(v);
                }
            }
        }

        // Calculate SCC level count
        unordered_map<int, int> scc_level_count;
        for (int l : scc_level) {
            if (l > 0) scc_level_count[l]++;
        }

        unordered_map<int, int> scc_cumul_level_count;
        int cum = 0;
        int max_scc_level = 0;
        for (auto& p : scc_level_count) max_scc_level = max(max_scc_level, p.first);
        for (int l = 1; l <= max_scc_level; ++l) {
            if (scc_level_count.count(l)) cum += scc_level_count[l];
            scc_cumul_level_count[l] = cum;
        }

        // For each level 1 SCC, compute coverage in its reachable subgraph
        auto find_origin_as = [&](uint32_t from_as, uint32_t to_as) -> pair<time_t, uint32_t> {
            ASPair p = {from_as, to_as};
            auto it = as_pairs_with_ts.find(p);
            return (it != as_pairs_with_ts.end()) ? it->second : pair<time_t, uint32_t>{0, 0};
        };

        // Cache downstream ASNs per SCC to avoid repeated BFS for nodes in the same SCC.
        unordered_map<int, vector<uint32_t>> downstream_asns_cache;

        auto get_downstream_asns_for_scc = [&](int current_scc) -> const vector<uint32_t>& {
            auto it = downstream_asns_cache.find(current_scc);
            if (it != downstream_asns_cache.end()) {
                return it->second;
            }

            unordered_set<int> visited_scc;
            queue<int> bfs;
            for (int succ_scc : scc_adj[current_scc]) {
                if (visited_scc.insert(succ_scc).second) {
                    bfs.push(succ_scc);
                }
            }

            while (!bfs.empty()) {
                int u = bfs.front();
                bfs.pop();
                for (int v : scc_adj[u]) {
                    if (visited_scc.insert(v).second) {
                        bfs.push(v);
                    }
                }
            }

            unordered_set<uint32_t> affected_set;
            for (int scc_id : visited_scc) {
                for (uint32_t asn : sccs[scc_id]) {
                    affected_set.insert(asn);
                }
            }

            vector<uint32_t> affected_sorted(affected_set.begin(), affected_set.end());
            sort(affected_sorted.begin(), affected_sorted.end());
            auto [inserted_it, inserted] = downstream_asns_cache.emplace(current_scc, std::move(affected_sorted));
            return inserted_it->second;
        };

        auto collect_downstream_asns = [&](int current_scc, uint32_t current_asn) -> vector<uint32_t> {
            const auto& cached_asns = get_downstream_asns_for_scc(current_scc);
            if (cached_asns.empty()) {
                return {};
            }

            if (!binary_search(cached_asns.begin(), cached_asns.end(), current_asn)) {
                return cached_asns;
            }

            vector<uint32_t> filtered;
            filtered.reserve(cached_asns.size() - 1);
            for (uint32_t asn : cached_asns) {
                if (asn != current_asn) {
                    filtered.push_back(asn);
                }
            }
            return filtered;
        };

        auto process_scc = [&](int current_scc, const vector<double>& coverage, unordered_map<uint32_t, vector<uint32_t>>& asn_upstream) -> void {
            vector<uint32_t> upstream_asns;
            // Prioritize internal links if SCC has multiple nodes
            if (sccs[current_scc].size() > 1) {
                for (uint32_t node : sccs[current_scc]) {
                    if (adj.count(node)) {
                        for (uint32_t neighbor : adj[node]) {
                            if (node_to_scc[neighbor] == current_scc) {
                                auto [timestamp, origin_as] = find_origin_as(node, neighbor);
                                if (origin_as != 0) {
                                    as_info[node] = {timestamp, (uint32_t)scc_level[current_scc], origin_as};
                                    asn_upstream[node] = upstream_asns;
                                    asn_affected_downstream[node] = collect_downstream_asns(current_scc, node);
                                }
                            }
                        }
                    }
                    if (rev_adj.count(node)) {
                        for (uint32_t neighbor : rev_adj[node]) {
                            if (node_to_scc[neighbor] == current_scc) {
                                auto [timestamp, origin_as] = find_origin_as(neighbor, node);
                                if (origin_as != 0) {
                                    as_info[node] = {timestamp, (uint32_t)scc_level[current_scc], origin_as};
                                    asn_upstream[node] = upstream_asns;
                                    asn_affected_downstream[node] = collect_downstream_asns(current_scc, node);
                                }
                            }
                        }
                    }
                }
                return;
            }

            // If no internal links, check neighbors in adjacent SCCs
            double max_neighbor_cov = -1.0;
            int best_neighbor_scc = -1;
            bool is_predecessor = false;

            for (int pred_scc : scc_rev_adj[current_scc]) {
                if (coverage[pred_scc] > max_neighbor_cov) {
                    max_neighbor_cov = coverage[pred_scc];
                    best_neighbor_scc = pred_scc;
                    is_predecessor = true;
                }
            }
            for (int succ_scc : scc_adj[current_scc]) {
                if (coverage[succ_scc] > max_neighbor_cov) {
                    max_neighbor_cov = coverage[succ_scc];
                    best_neighbor_scc = succ_scc;
                    is_predecessor = false;
                }
            }

            if (best_neighbor_scc != -1) {
                upstream_asns = sccs[best_neighbor_scc];  // Collect all ASNs in the best neighbor SCC
                const auto& neighbors = is_predecessor ? rev_adj : adj;
                const auto& current_nodes = sccs[current_scc];

                for (uint32_t node : current_nodes) {
                    if (neighbors.find(node) == neighbors.end()) continue;
                    for (uint32_t neighbor : neighbors.at(node)) {
                        if (node_to_scc.count(neighbor) && node_to_scc[neighbor] == best_neighbor_scc) {
                            auto [timestamp, origin_as] = find_origin_as(
                                is_predecessor ? neighbor : node,
                                is_predecessor ? node : neighbor
                            );
                            if (origin_as != 0) {
                                as_info[node] = {timestamp, (uint32_t)scc_level[current_scc], origin_as};
                                asn_upstream[node] = upstream_asns;
                                asn_affected_downstream[node] = collect_downstream_asns(current_scc, node);
                            }
                        }
                    }
                }
            }
        };

        // Find all level 1 SCCs
        vector<int> level1_sccs;
        for (size_t scc = 0; scc < S; ++scc) {
            if (scc_level[scc] == 1) {
                level1_sccs.push_back(scc);
            }
        }

        // For each level 1 SCC, compute coverage in its reachable subgraph
        for (int root_scc : level1_sccs) {
            // Find all SCCs reachable from root_scc
            unordered_set<int> reachable;
            function<void(int)> dfs = [&](int u) {
                if (reachable.count(u)) return;
                reachable.insert(u);
                for (int v : scc_adj[u]) dfs(v);
            };
            dfs(root_scc);

            // Compute coverage for reachable SCCs using global graph with mask
            vector<double> sub_coverage(S, 0.0); // Global size, but only fill reachable
            if (S < 256) {
                solveReachabilityMasked<256>(S, scc_adj, scc_rev_adj, scc_level, scc_cumul_level_count, S, reachable, sub_coverage);
            } else if (S < 512) {
                solveReachabilityMasked<512>(S, scc_adj, scc_rev_adj, scc_level, scc_cumul_level_count, S, reachable, sub_coverage);
            } else if (S < 1024) {
                solveReachabilityMasked<1024>(S, scc_adj, scc_rev_adj, scc_level, scc_cumul_level_count, S, reachable, sub_coverage);
            } else {
                // For large S, limit to top 1023 reachable SCCs sorted by level
                vector<pair<int, int>> valid_sccs; // <level, scc_id>
                for (int scc : reachable) {
                    if (scc_level[scc] > 0) valid_sccs.push_back({scc_level[scc], scc});
                }
                size_t sub_S = min((size_t)1023, valid_sccs.size());
                if (valid_sccs.size() > sub_S) {
                    std::partial_sort(valid_sccs.begin(), valid_sccs.begin() + sub_S, valid_sccs.end());
                }
                unordered_set<int> sub_reachable;
                for (size_t i = 0; i < sub_S; ++i) sub_reachable.insert(valid_sccs[i].second);
                solveReachabilityMasked<1024>(S, scc_adj, scc_rev_adj, scc_level, scc_cumul_level_count, S, sub_reachable, sub_coverage);
            }

            // Find max coverage SCC in reachable set
            double max_cov = 0;
            int crit_scc = -1;
            for (int scc : reachable) {
                if (scc_level[scc] <= 6 && sub_coverage[scc] > max_cov) {
                    max_cov = sub_coverage[scc];
                    crit_scc = scc;
                }
            }

            // Process the critical SCC and its upstream SCCs within 2 levels
            if (max_cov >= 1 && reachable.size() > 2) {
                const int MAX_DEPTH = 1;
                queue<pair<int, int>> q; // <scc_id, depth>
                unordered_set<int> visited;
                q.push({crit_scc, 0});
                visited.insert(crit_scc);

                while (!q.empty()) {
                    auto [u, depth] = q.front(); q.pop();
                    process_scc(u, sub_coverage, asn_upstream);

                    if (depth < MAX_DEPTH && scc_level[u] <= scc_level[crit_scc]) {
                        for (int v : scc_rev_adj[u]) {
                            if (visited.find(v) == visited.end() && scc_level[v] < scc_level[u]) {
                                visited.insert(v);
                                q.push({v, depth + 1});
                            }
                        }
                    }
                }
            }
        }
    }

    // Template function: for solving reachability problems of specific size, using mask to compute only active SCCs
    template<size_t N>
    void solveReachabilityMasked(size_t S, 
                          const vector<vector<int>>& scc_adj,
                          const vector<vector<int>>& scc_rev_adj,
                          const vector<int>& scc_level,
                          const unordered_map<int, int>& scc_cumul_level_count,
                          size_t total_scc_count,
                          const unordered_set<int>& active,
                          vector<double>& scc_coverage) {
        
        static constexpr size_t COMPUTED_FLAG = N - 1;

        // The active subgraph may contain large global SCC IDs even when it
        // has fewer than N nodes. Pack those IDs into the available data bits;
        // the final bit remains reserved for the computed flag.
        if (active.size() > COMPUTED_FLAG) {
            throw length_error("Active SCC count exceeds reachability bitset capacity");
        }
        vector<size_t> bit_index(S);
        size_t next_bit = 0;
        for (int scc : active) {
            if (scc < 0 || static_cast<size_t>(scc) >= S) {
                throw out_of_range("Active SCC index is outside the graph");
            }
            bit_index[scc] = next_bit++;
        }
        
        // Bitset version - use last bit as computed flag
        vector<bitset<N>> downstream_reach(S);
        vector<bitset<N>> upstream_reach(S);

        // Compute downstream reachability
        function<void(size_t)> compute_downstream = [&](size_t scc) {
            if (downstream_reach[scc][COMPUTED_FLAG]) return;   // Check last bit
            downstream_reach[scc][COMPUTED_FLAG] = 1;           // Set last bit as computed flag
            downstream_reach[scc][bit_index[scc]] = 1;          // Include self
            for (int v : scc_adj[scc]) {
               if (active.count(v) && scc_level[v] >= scc_level[scc]) {
                    compute_downstream(v);
                    downstream_reach[scc] |= downstream_reach[v];
                }
            }
        };

        // Compute upstream reachability
        function<void(size_t)> compute_upstream = [&](size_t scc) {
            if (upstream_reach[scc][COMPUTED_FLAG]) return;   // Check last bit
            upstream_reach[scc][COMPUTED_FLAG] = 1;           // Set last bit as computed flag
            upstream_reach[scc][bit_index[scc]] = 1;          // Include self
            for (int v : scc_rev_adj[scc]) {
                if (active.count(v) && scc_level[v] <= scc_level[scc]) {
                    compute_upstream(v);
                    upstream_reach[scc] |= upstream_reach[v];
                }
            }
        };

        for (size_t scc = 0; scc < S; ++scc) {
            if (active.count(scc) && scc_level[scc] > 0) {  
                compute_downstream(scc);
                compute_upstream(scc);
            }
        }
        
        // Calculate coverage
        for (size_t scc = 0; scc < S; ++scc) {
            if (active.count(scc) && scc_level[scc] > 0) {                
                size_t upstream_reach_count = upstream_reach[scc].count() - 1; 
                size_t downstream_reach_count = downstream_reach[scc].count() - 1;

                int my_level = scc_level[scc];
                int upstream_total = scc_cumul_level_count.at(my_level);
                int prev_level_count = (my_level <= 1) ? 0 : scc_cumul_level_count.at(my_level - 1);
                int downstream_total = total_scc_count - prev_level_count;

                double upstream_cov = (upstream_total == 0) ? 1 : static_cast<double>(upstream_reach_count) / upstream_total;
                double downstream_cov = (downstream_total == 0) ? 0 : static_cast<double>(downstream_reach_count) / downstream_total;
                double coverage = log2(my_level + 1) * pow(upstream_cov, 3) * pow(downstream_cov, 2);
                scc_coverage[scc] = coverage;
            }
        }
    }

    // Tarjan algorithm implementation for strongly connected components
    void tarjanSCC(const unordered_map<uint32_t, vector<uint32_t>>& adj, const vector<uint32_t>& node_list,
                   vector<vector<uint32_t>>& sccs, unordered_map<uint32_t, int>& node_to_scc) {
        int time = 0;
        vector<int> disc(node_list.size(), -1);
        vector<int> low(node_list.size(), -1);
        vector<bool> in_stack(node_list.size(), false);
        stack<uint32_t> stk;
        unordered_map<uint32_t, int> node_index;
        for (size_t i = 0; i < node_list.size(); ++i) node_index[node_list[i]] = i;

        function<void(uint32_t)> dfs = [&](uint32_t u) {
            int idx = node_index[u];
            disc[idx] = low[idx] = time++;
            stk.push(u);
            in_stack[idx] = true;

            for (uint32_t v : adj.at(u)) {
                int vidx = node_index[v];
                if (disc[vidx] == -1) {
                    dfs(v);
                    low[idx] = min(low[idx], low[vidx]);
                } else if (in_stack[vidx]) {
                    low[idx] = min(low[idx], disc[vidx]);
                }
            }

            if (low[idx] == disc[idx]) {
                vector<uint32_t> scc;
                while (true) {
                    uint32_t v = stk.top(); stk.pop();
                    int vidx = node_index[v];
                    in_stack[vidx] = false;
                    scc.push_back(v);
                    node_to_scc[v] = sccs.size();
                    if (v == u) break;
                }
                sccs.push_back(scc);
            }
        };

        for (uint32_t node : node_list) {
            if (disc[node_index[node]] == -1) {
                dfs(node);
            }
        }
    }

public:
    explicit TopologyAnalyzer(const PrefixPropGraph& graph, size_t num_threads = 8,
        size_t expected_prefixes = 1200000,
        const char* csv_filename = nullptr
    )
        : graph_(graph), num_threads_(num_threads), anomaly_queue_(4096) {
        
        // Initialize history shards
        history_shards_.reserve(NUM_SHARDS);
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            history_shards_.push_back(make_unique<HistoryShard>());
        }

        last_detection_time_ = 0; // Initialize detection time

        // Initialize CSV filename
        if (csv_filename) {
            csv_filename_ = csv_filename;
        }

        // Start detection threads
        detection_threads_.reserve(num_threads_);
        for (size_t i = 0; i < num_threads_; ++i) {
            detection_threads_.emplace_back(&TopologyAnalyzer::detectionLoop, this);
        }
    }

    ~TopologyAnalyzer() {
        shutdown();
    }

    // Submit new prefix-AS link batch for anomaly detection
    void submitPrefixLinkBatch(PrefixAnomalyBatch&& batch) {
        if (!batch.isEmpty()) {
            anomaly_queue_.enqueue(std::move(batch));
        }
    }

    bool isIdle() const {
        return anomaly_queue_.size() == 0 && in_flight_batches_.load(memory_order_relaxed) == 0;
    }

    void waitUntilIdle() {
        while (!isIdle()) {
            this_thread::sleep_for(chrono::milliseconds(1));
        }
    }

    void flushPending(time_t timestamp) {
        waitUntilIdle();
        executeAnomalyDetection(timestamp);
    }

    void shutdown() {
        if (running_.exchange(false, memory_order_relaxed)) {
            anomaly_queue_.shutdown();
            for (auto& t : detection_threads_) {
                if (t.joinable()) {
                    t.join();
                }
            }

            // Write to CSV file
            if (!csv_filename_.empty()) {
                lock_guard<mutex> lock(records_mutex_);
                // Sort by timestamp
                sort(anomaly_records_.begin(), anomaly_records_.end(), [](const ASAnomalyRecord& a, const ASAnomalyRecord& b) {
                    return a.timestamp < b.timestamp;
                });
                // Write CSV
                FILE* csv_file = fopen(csv_filename_.c_str(), "w");
                if (csv_file) {
                    fprintf(csv_file, "timestamp,prefix,asn,level,topology_level,origin,is_origin_asn,is_blackhole_route,is_ip_leasing,upstream_asns,affected_asns\n");
                    for (const auto& record : anomaly_records_) {
                        char prefix_str[64];
                        prefixToString(record.prefix, prefix_str, sizeof(prefix_str));
                        fprintf(
                            csv_file,
                            "%ld,%s,%u,%u,%u,%u,%u,%u,%u,%s,%s\n",
                            record.timestamp,
                            prefix_str,
                            record.asn,
                            record.level,
                            record.topology_level,
                            record.origin_as,
                            record.is_origin_asn,
                            static_cast<unsigned>(record.is_blackhole_route),
                            static_cast<unsigned>(record.is_ip_leasing),
                            record.upstream_asns_str.c_str(),
                            record.affected_asns_str.c_str()
                        );
                    }
                    fclose(csv_file);
                    fprintf(stdout, "Anomaly records written to %s\n", csv_filename_.c_str());
                } else {
                    fprintf(stderr, "Warning: Failed to open CSV file: %s\n", csv_filename_.c_str());
                }
            }
        }
    }

    struct AnomalyStats {
        uint64_t primary_anomalies;
        uint64_t secondary_anomalies;
        uint64_t total_anomalies;
        uint64_t batches_processed;
        uint64_t total_links_processed;
        size_t queue_size;
    };


    // Calculate memory usage of snapshot storage
    size_t getMemoryUsage() const {
        size_t memory = 0;
        
        // Calculate prefix history memory usage
        for (const auto& shard_ptr : history_shards_) {
            lock_guard<mutex> lock(shard_ptr->mtx);
            for (const auto& [prefix, history] : shard_ptr->history) {
                memory += sizeof(Prefix);  // Prefix key
                memory += sizeof(pair<unordered_set<ASPair>, time_t>);  // History record structure
                memory += history.first.size() * sizeof(ASPair);  // AS link set
            }
             memory += (shard_ptr->history.size()) * 64; // Hash table overhead per element
        }

        // Calculate anomaly_records_ memory usage
        {
            lock_guard<mutex> lock(records_mutex_);
            memory += anomaly_records_.size() * sizeof(ASAnomalyRecord);
        }

        // Calculate recorded_asns_ memory usage
        {
            lock_guard<mutex> lock(recorded_asns_mutex_);
            memory += recorded_asns_.size() * (sizeof(Prefix) + sizeof(uint32_t) + 16);  // Rough estimate of overhead per element
        }
        
        return memory;
    }
};

#endif // TOPOLOGY_ANALYZER_HPP
