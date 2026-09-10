#ifndef FINE_GRAINED_CONCURRENT_GRAPH_HPP
#define FINE_GRAINED_CONCURRENT_GRAPH_HPP

#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <array>
#include <thread>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <vector>
#include <ctime>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <stdio.h>
#include "helpers.hpp"

using namespace std;

inline time_t getCurrentTimestamp() {
    return time(nullptr);
}

/**
 * Attribute information of AS edges
 * Store all prefixes corresponding to this edge and their timestamps and Origin AS
 */
struct ASEdgeInfo {
    // Mapping from prefix to timestamp and Origin AS
    unordered_map<Prefix, pair<time_t, uint32_t>> prefix_last_ts_and_origin;
    
    // Batch update prefix function
    void updatePrefixes(unordered_map<Prefix, pair<time_t, uint32_t>>&& prefixes) {
        if (prefix_last_ts_and_origin.bucket_count() < prefix_last_ts_and_origin.size() + prefixes.size()) {
            prefix_last_ts_and_origin.reserve(prefix_last_ts_and_origin.size() + prefixes.size());
        }
        
        // High-performance merge: process new keys first, then optimize conflicting keys
        prefix_last_ts_and_origin.merge(prefixes);
        
        // Only need to process when there are conflicts (unmerged elements remain in prefixes)
        if (!prefixes.empty()) {
            for (auto& [prefix, new_pair] : prefixes) {
                auto& old_pair = prefix_last_ts_and_origin[prefix];
                if (new_pair.first > old_pair.first) {
                    old_pair = new_pair; 
                }
            }
        }
    }
    
    // Batch update and return new prefixes with their timestamps and Origin AS (for upper layer to carry timestamps directly)
    vector<pair<Prefix, pair<time_t, uint32_t>>> updatePrefixesAndGetNew(unordered_map<Prefix, pair<time_t, uint32_t>>&& prefixes) {
        vector<pair<Prefix, pair<time_t, uint32_t>>> new_items;
        new_items.reserve(prefixes.size());

        if (prefix_last_ts_and_origin.bucket_count() < prefix_last_ts_and_origin.size() + prefixes.size()) {
            prefix_last_ts_and_origin.reserve(prefix_last_ts_and_origin.size() + prefixes.size());
        }

        for (auto& [prefix, new_pair] : prefixes) {
            auto [iter, inserted] = prefix_last_ts_and_origin.insert({prefix, new_pair});        
            if (inserted) {
                // Since records basically arrive in ascending time order, when inserting, the earliest timestamp can be considered for detection
                new_items.emplace_back(prefix, new_pair);
            } else if (new_pair.first > iter->second.first) {
                // But for updates to existing entries, the latest timestamp and Origin AS need to be used
                iter->second = new_pair;
            }
        }
        return new_items;
    }

    // Check if it contains the specified prefix
    bool hasPrefix(const Prefix& prefix) const {
        return prefix_last_ts_and_origin.find(prefix) != prefix_last_ts_and_origin.end();
    }
    
    bool isEmpty() const {
        return prefix_last_ts_and_origin.empty();
    }
    
    size_t getTotalPrefixCount() const {
        return prefix_last_ts_and_origin.size();
    }
};

class PrefixPropGraph {
public:
    PrefixPropGraph(size_t expected_prefixes_ipv4 = 1000000, 
                                size_t expected_prefixes_ipv6 = 100000,
                                size_t expected_as_count = 100000, 
                                size_t expected_edges = 500000) {
                
        // Initialize AS neighbor storage
        global_as_neighbors_.reserve(expected_as_count);
    }

    // Copy constructor (for creating backup graph)
    PrefixPropGraph(const PrefixPropGraph& other) {
        
        // Copy all shard data
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            shared_lock<shared_mutex> lock(other.shards_[i].rw_mutex);
            shards_[i].edges = other.shards_[i].edges;
        }
        
        // Copy AS neighbor data
        {
            shared_lock<shared_mutex> lock(other.as_neighbors_mutex_);
            global_as_neighbors_ = other.global_as_neighbors_;
        }
    }

public:
    // Core function: batch update and return newly added AS links
    auto updateASLinksAndGetNew(
        unordered_map<ASPair, unordered_map<Prefix, pair<time_t, uint32_t>>>&& updates) {
        
        unordered_map<Prefix, vector<ASPairWithTimeAndOrigin>> new_links;
        
        if (updates.empty()) return new_links;
        
        // Organize data by shards
        array<unordered_map<ASPair, unordered_map<Prefix, pair<time_t, uint32_t>>>, NUM_SHARDS> shard_updates;        
        for (auto& [as_pair, prefix_map] : updates) {
            size_t shard_idx = getShardIndex(as_pair);
            shard_updates[shard_idx][as_pair] = std::move(prefix_map);
        }
        
        // Process all shards serially and collect newly added links
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            if (!shard_updates[i].empty()) {
                auto& shard = shards_[i];
                
                unique_lock<shared_mutex> lock(shard.rw_mutex);
                
                for (auto& [as_pair, prefix_map] : shard_updates[i]) {
                    auto& edge_info = shard.edges[as_pair];
                    // Newly added prefixes with timestamps and origins for one ASPair
                    auto new_items = edge_info.updatePrefixesAndGetNew(std::move(prefix_map));                    

                    // Newly added ASPairWithTimeAndOrigin entries
                    for (const auto& item : new_items) {
                        const auto& prefix = item.first;
                        const auto& [ts, origin] = item.second;
                        new_links[prefix].push_back(ASPairWithTimeAndOrigin(as_pair, ts, origin));
                    }
                }
            }
        }
        return new_links;
    }

    // Batch update AS links (without returning newly added links, for RIB phase)
    void updateASLinks(unordered_map<ASPair, unordered_map<Prefix, pair<time_t, uint32_t>>>&& updates) {
        if (updates.empty()) return;
        
        // Organize data by shards
        array<unordered_map<ASPair, unordered_map<Prefix, pair<time_t, uint32_t>>>, NUM_SHARDS> shard_updates;        
        for (auto& [as_pair, prefix_map] : updates) {
            size_t shard_idx = getShardIndex(as_pair);
            shard_updates[shard_idx][as_pair] = std::move(prefix_map);
        }
        
        // Process all shards serially
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            if (!shard_updates[i].empty()) {
                auto& shard = shards_[i];
                
                unique_lock<shared_mutex> lock(shard.rw_mutex);
                
                for (auto& [as_pair, prefix_map] : shard_updates[i]) {
                    auto& edge_info = shard.edges[as_pair];
                    edge_info.updatePrefixes(std::move(prefix_map));
                }
            }
        }
    }

    // Batch update AS neighbor relationships
    void updateASNeighbors(unordered_map<uint32_t, unordered_set<uint32_t>>&& new_neighbors) {
        if (new_neighbors.empty()) return;
        unique_lock<shared_mutex> lock(as_neighbors_mutex_);
        for (auto& [as_num, neighbors] : new_neighbors) {
            auto& dst = global_as_neighbors_[as_num];
            if (dst.empty()) {
                dst = std::move(neighbors);
            } else {
                dst.merge(neighbors);
            }
        }
    }
 
    // Read interface: get edge information
    const ASEdgeInfo* getEdgeInfo(uint32_t from_as, uint32_t to_as) const {
        ASPair as_pair(from_as, to_as);
        size_t shard_idx = getShardIndex(as_pair);
        
        auto& shard = shards_[shard_idx];
        shared_lock<shared_mutex> lock(shard.rw_mutex);
        
        auto it = shard.edges.find(as_pair);
        
        return (it != shard.edges.end()) ? &it->second : nullptr;
    }
    
    // Read interface: get AS neighbors
    const unordered_set<uint32_t>* getASNeighbors(uint32_t as_num) const {
        shared_lock<shared_mutex> lock(as_neighbors_mutex_);
        
        auto it = global_as_neighbors_.find(as_num);
        return (it != global_as_neighbors_.end()) ? &it->second : nullptr;
    }
    
    // Check if prefix contains specific AS pair
    bool hasPrefixASPair(const Prefix& prefix, const ASPair& as_pair) const {
        size_t shard_idx = getShardIndex(as_pair);
        
        auto& shard = shards_[shard_idx];
        shared_lock<shared_mutex> lock(shard.rw_mutex);
        
        auto it = shard.edges.find(as_pair);
        if (it != shard.edges.end()) {
            bool found = it->second.hasPrefix(prefix);
            return found;
        }
        
        return false;
    }

    // Get all edge sets for the prefix
    set<ASPair> getPrefixEdges(const Prefix& prefix) const {
        set<ASPair> result;
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            auto& shard = shards_[i];
            shared_lock<shared_mutex> lock(shard.rw_mutex);
            for (const auto& [as_pair, edge_info] : shard.edges) {
                bool found;
                found = edge_info.prefix_last_ts_and_origin.find(prefix) != edge_info.prefix_last_ts_and_origin.end();
                if (found) {
                    result.insert(as_pair);
                }
            }
        }
        return result;
    }

    // Statistics interface
    struct GlobalStats {
        size_t total_edges;
        size_t total_ases;
    };
    
    GlobalStats getGlobalStats() const {
        GlobalStats stats = {0};
        
        // Count edge information for all shards
        for (const auto& shard : shards_) {
            shared_lock<shared_mutex> lock(shard.rw_mutex);
            stats.total_edges += shard.edges.size();
        }
        
        // Count global AS neighbors
        {
            shared_lock<shared_mutex> lock(as_neighbors_mutex_);
            stats.total_ases = global_as_neighbors_.size();
        }
        
        return stats;
    }
    
    size_t getMemoryUsage() const {
        size_t total = 0;
        
        // Count memory usage for all shards
        for (const auto& shard : shards_) {
            shared_lock<shared_mutex> lock(shard.rw_mutex);
            
            total += shard.edges.size() * sizeof(pair<ASPair, ASEdgeInfo>);
            for (const auto& [as_pair, edge_info] : shard.edges) {
                total += edge_info.prefix_last_ts_and_origin.size() * (sizeof(Prefix) + sizeof(pair<time_t, uint32_t>));
            }
        }
        
        // AS neighbor memory
        {
            shared_lock<shared_mutex> lock(as_neighbors_mutex_);
            total += global_as_neighbors_.size() * sizeof(pair<uint32_t, unordered_set<uint32_t>>);
            for (const auto& [as_num, neighbors] : global_as_neighbors_) {
                total += neighbors.size() * sizeof(uint32_t);
            }
        }
        
        return total;
    }

private:
    // Internal helper function: get shard index
    size_t getShardIndex(const ASPair& as_pair) const {
        size_t hash1 = hash<uint32_t>{}(as_pair.from_as);
        size_t hash2 = hash<uint32_t>{}(as_pair.to_as);
        return (hash1 ^ (hash2 << 1)) % NUM_SHARDS;
    }

    // Internal structure: shard
    struct Shard {
        unordered_map<ASPair, ASEdgeInfo> edges;
        mutable shared_mutex rw_mutex;
        
        Shard() {
            edges.reserve(2000);  // Reserve space for each shard
        }
    };

    // Member variables
    static constexpr size_t NUM_SHARDS = 256;
    Shard shards_[NUM_SHARDS];
    
    // AS neighbor relationship storage
    unordered_map<uint32_t, unordered_set<uint32_t>> global_as_neighbors_;
    mutable shared_mutex as_neighbors_mutex_;
};

#endif // FINE_GRAINED_CONCURRENT_GRAPH_HPP
