#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: reference_summary <reference_paths.bin> <summary.json>\n";
        return 1;
    }

    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open input: " << argv[1] << '\n';
        return 1;
    }
    uint64_t total = 0;
    input.read(reinterpret_cast<char*>(&total), sizeof(total));
    if (!input) {
        std::cerr << "Invalid compact reference header\n";
        return 1;
    }

    std::unordered_set<uint32_t> ases;
    std::unordered_set<uint64_t> edges;
    ases.reserve(100000);
    edges.reserve(700000);
    std::array<uint32_t, 255> path{};

    for (uint64_t index = 0; index < total; ++index) {
        int32_t prefix_id = -1;
        uint8_t path_len = 0;
        input.read(reinterpret_cast<char*>(&prefix_id), sizeof(prefix_id));
        input.read(reinterpret_cast<char*>(&path_len), sizeof(path_len));
        input.read(reinterpret_cast<char*>(path.data()),
                   static_cast<std::streamsize>(path_len) * sizeof(uint32_t));
        if (!input) {
            std::cerr << "Truncated compact reference at record " << index << '\n';
            return 1;
        }
        for (uint8_t pos = 0; pos < path_len; ++pos) {
            ases.insert(path[pos]);
            if (pos == 0 || path[pos - 1] == path[pos]) {
                continue;
            }
            const uint32_t left = std::min(path[pos - 1], path[pos]);
            const uint32_t right = std::max(path[pos - 1], path[pos]);
            edges.insert((static_cast<uint64_t>(left) << 32) | right);
        }
    }

    std::ofstream output(argv[2]);
    if (!output) {
        std::cerr << "Cannot open output: " << argv[2] << '\n';
        return 1;
    }
    output << "{\n"
           << "  \"paths\": " << total << ",\n"
           << "  \"asns\": " << ases.size() << ",\n"
           << "  \"edges\": " << edges.size() << "\n"
           << "}\n";
    return 0;
}
