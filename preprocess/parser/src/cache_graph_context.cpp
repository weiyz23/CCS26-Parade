#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "helpers.hpp"

namespace {

constexpr std::array<char, 8> MAGIC{'P', 'G', 'C', 'T', 'X', '0', '1', '\0'};

template <typename T>
void read_exact(std::ifstream& input, T& value, const char* label) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error(std::string("truncated ") + label);
    }
}

template <typename T>
void write_exact(std::ofstream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: cache_graph_context <rib_records_cache.bin> "
                     "<prefix_profile_map.bin> <graph_context.bin>\n";
        return 1;
    }

    try {
        std::ifstream mapping(argv[2], std::ios::binary);
        if (!mapping) {
            throw std::runtime_error(std::string("cannot open mapping: ") + argv[2]);
        }
        uint64_t mapping_count = 0;
        read_exact(mapping, mapping_count, "mapping header");
        std::unordered_map<Prefix, int32_t> profile_by_prefix;
        profile_by_prefix.reserve(static_cast<size_t>(mapping_count * 1.15));
        int32_t maximum_profile = -1;
        for (uint64_t index = 0; index < mapping_count; ++index) {
            Prefix prefix{};
            int32_t profile_id = -1;
            mapping.read(reinterpret_cast<char*>(prefix.addr), sizeof(prefix.addr));
            read_exact(mapping, profile_id, "mapping record");
            if (!mapping) {
                throw std::runtime_error("truncated prefix bytes in mapping record");
            }
            if (profile_id >= 0) {
                profile_by_prefix.emplace(prefix, profile_id);
                maximum_profile = std::max(maximum_profile, profile_id);
            }
        }
        const uint64_t profile_count = static_cast<uint64_t>(maximum_profile) + 1;
        std::vector<std::unordered_set<uint32_t>> profile_origins(profile_count);

        std::ifstream cache(argv[1], std::ios::binary);
        if (!cache) {
            throw std::runtime_error(std::string("cannot open cache: ") + argv[1]);
        }
        uint64_t record_count = 0;
        read_exact(cache, record_count, "cache header");
        const uintmax_t expected_bytes = sizeof(uint64_t) + record_count * sizeof(BGPRecord);
        const uintmax_t actual_bytes = std::filesystem::file_size(argv[1]);
        if (expected_bytes != actual_bytes) {
            throw std::runtime_error(
                "cache size mismatch: expected=" + std::to_string(expected_bytes) +
                " actual=" + std::to_string(actual_bytes));
        }

        std::unordered_set<uint64_t> edges;
        edges.reserve(800000);
        constexpr size_t CHUNK_RECORDS = 1u << 16;
        std::vector<BGPRecord> records(CHUNK_RECORDS);
        uint64_t processed = 0;
        while (processed < record_count) {
            const size_t wanted = static_cast<size_t>(
                std::min<uint64_t>(CHUNK_RECORDS, record_count - processed));
            cache.read(
                reinterpret_cast<char*>(records.data()),
                static_cast<std::streamsize>(wanted * sizeof(BGPRecord)));
            if (cache.gcount() != static_cast<std::streamsize>(wanted * sizeof(BGPRecord))) {
                throw std::runtime_error("truncated cache record block");
            }
            for (size_t index = 0; index < wanted; ++index) {
                const BGPRecord& record = records[index];
                const auto match = profile_by_prefix.find(record.prefix);
                const uint8_t pair_count = std::min<uint8_t>(record.as_pair_count, AS_PAIR_SIZE);
                if (match != profile_by_prefix.end()) {
                    auto& origins = profile_origins[static_cast<size_t>(match->second)];
                    if (record.origin_as != 0) {
                        origins.insert(record.origin_as);
                    } else {
                        // Older parser caches left origin_as unset.  Its path
                        // extractor orients each link away from the origin, so
                        // the origin is a source: it occurs as from_as and
                        // never as to_as.  For example, 1.0.0.0/24 yields
                        // AS13335 as this endpoint at every collector.
                        bool found_source = false;
                        for (uint8_t pair_index = 0; pair_index < pair_count; ++pair_index) {
                            const uint32_t candidate = record.as_pairs[pair_index].from_as;
                            if (candidate == 0) {
                                continue;
                            }
                            bool appears_as_target = false;
                            for (uint8_t other = 0; other < pair_count; ++other) {
                                if (record.as_pairs[other].to_as == candidate) {
                                    appears_as_target = true;
                                    break;
                                }
                            }
                            if (!appears_as_target) {
                                origins.insert(candidate);
                                found_source = true;
                            }
                        }
                        if (!found_source && pair_count > 0) {
                            const uint32_t fallback = record.as_pairs[0].from_as;
                            if (fallback != 0) {
                                origins.insert(fallback);
                            }
                        }
                    }
                }
                for (uint8_t pair_index = 0; pair_index < pair_count; ++pair_index) {
                    const uint32_t first = record.as_pairs[pair_index].from_as;
                    const uint32_t second = record.as_pairs[pair_index].to_as;
                    if (first == 0 || second == 0 || first == second) {
                        continue;
                    }
                    const uint32_t left = std::min(first, second);
                    const uint32_t right = std::max(first, second);
                    edges.insert((static_cast<uint64_t>(left) << 32) | right);
                }
            }
            processed += wanted;
            if (processed % (10u * 1000u * 1000u) < CHUNK_RECORDS) {
                std::cerr << "[cache-context] records=" << processed << "/" << record_count
                          << " edges=" << edges.size() << '\n';
            }
        }

        std::vector<uint64_t> ordered_edges(edges.begin(), edges.end());
        std::sort(ordered_edges.begin(), ordered_edges.end());
        std::ofstream output(argv[3], std::ios::binary);
        if (!output) {
            throw std::runtime_error(std::string("cannot open output: ") + argv[3]);
        }
        output.write(MAGIC.data(), MAGIC.size());
        const uint64_t edge_count = ordered_edges.size();
        write_exact(output, edge_count);
        for (const uint64_t encoded : ordered_edges) {
            const uint32_t left = static_cast<uint32_t>(encoded >> 32);
            const uint32_t right = static_cast<uint32_t>(encoded);
            write_exact(output, left);
            write_exact(output, right);
        }
        write_exact(output, profile_count);
        uint64_t profiles_with_origins = 0;
        for (auto& origins : profile_origins) {
            std::vector<uint32_t> ordered(origins.begin(), origins.end());
            std::sort(ordered.begin(), ordered.end());
            const uint32_t origin_count = static_cast<uint32_t>(ordered.size());
            write_exact(output, origin_count);
            if (origin_count > 0) {
                ++profiles_with_origins;
                output.write(
                    reinterpret_cast<const char*>(ordered.data()),
                    static_cast<std::streamsize>(ordered.size() * sizeof(uint32_t)));
            }
        }
        if (profiles_with_origins == 0) {
            throw std::runtime_error(
                "no profile origins recovered; cache/mapping format mismatch");
        }
        output.flush();
        if (!output) {
            throw std::runtime_error("failed writing graph context");
        }
        std::cout << "[cache-context] records=" << record_count
                  << " profiles=" << profile_count
                  << " profiles_with_origins=" << profiles_with_origins
                  << " edges=" << edge_count << '\n';
    } catch (const std::exception& error) {
        std::cerr << "cache_graph_context: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
