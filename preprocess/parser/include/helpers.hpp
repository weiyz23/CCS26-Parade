#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <sys/mman.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <functional>
#include <stdexcept>

using namespace std;

// AS pair structure, representing an AS Link
struct ASPair {
    uint32_t from_as;    // Source AS (4 bytes)
    uint32_t to_as;      // Target AS (4 bytes)

    ASPair() : from_as(0), to_as(0) {}
    ASPair(uint32_t from, uint32_t to) : from_as(from), to_as(to) {}

    bool operator==(const ASPair& other) const {
        return from_as == other.from_as && to_as == other.to_as;
    }

    bool operator!=(const ASPair& other) const {
        return !(*this == other);
    }
    
    bool operator<(const ASPair& other) const {
        if (from_as != other.from_as) {
            return from_as < other.from_as;
        }
        return to_as < other.to_as;
    }
};


/*
 * AS Pair with timestamp and origin AS
 */
struct ASPairWithTimeAndOrigin {
    ASPair as_pair;
    time_t timestamp;
    uint32_t origin_as;

    ASPairWithTimeAndOrigin(ASPair p, time_t ts, uint32_t origin) : as_pair(p), timestamp(ts), origin_as(origin) {}
};

// Standard library compatible hash function definition
namespace std {
    template<>
    struct hash<ASPair> {
        size_t operator()(const ASPair& pair) const {
            return hash<uint32_t>()(pair.from_as) ^ (hash<uint32_t>()(pair.to_as) << 1);
        }
    };
}

// IP prefix struct, memory usage 16 bytes (using last 2 bytes to store family and mask_len)
struct Prefix {
    uint8_t addr[16];

    Prefix() {}

    Prefix(uint8_t fam, uint32_t ipv4_addr, uint8_t len) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        uint64_t* ptr = reinterpret_cast<uint64_t*>(addr);
        ptr[0] = static_cast<uint64_t>(ipv4_addr); 
        // [addr8...addr13, addr14, addr15] -> 0 | (fam<<48) | (len<<56)
        ptr[1] = (static_cast<uint64_t>(len) << 56) | (static_cast<uint64_t>(fam) << 48);
#else
        // General safe path: big-endian or unknown architecture
        memset(addr + 4, 0, 10);
        memcpy(addr, &ipv4_addr, 4); 
        addr[14] = fam;
        addr[15] = len;
#endif
    }

    Prefix(uint8_t fam, const uint8_t* ipv6_addr, uint8_t len) {
        // For IPv6, memcpy is still the most stable
        memcpy(addr, ipv6_addr, 14); 
        addr[14] = fam;
        addr[15] = len;
    }

    uint8_t get_family() const { return addr[14]; }
    uint8_t get_mask_len() const { return addr[15]; }

    bool operator==(const Prefix& other) const {
        const uint64_t* p1 = reinterpret_cast<const uint64_t*>(addr);
        const uint64_t* p2 = reinterpret_cast<const uint64_t*>(other.addr);
        return p1[0] == p2[0] && p1[1] == p2[1];
    }

    bool operator<(const Prefix& other) const {
        return memcmp(addr, other.addr, 16) < 0;
    }
};

namespace std {
    template<>
    struct hash<Prefix> {
        size_t operator()(const Prefix& p) const {
            if (p.get_family() == 4) {
                uint32_t ip;
                memcpy(&ip, p.addr, 4);
                uint8_t len = p.get_mask_len();
                size_t h = hash<uint32_t>()(ip);
                h ^= hash<uint8_t>()(len) + 0x9e3779b9 + (h << 6) + (h >> 2);
                return h;
            }
            uint64_t buf[2];
            memcpy(buf, p.addr, 16);
            size_t h1 = hash<uint64_t>()(buf[0]);
            size_t h2 = hash<uint64_t>()(buf[1]);
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2)); 
        }
    };
}

bool prefixToString(const Prefix& prefix, char* buffer, size_t buffer_size) {
    if (prefix.get_family() == 4) {
        if (buffer_size < INET_ADDRSTRLEN + 4) return false;
        struct in_addr addr;
        memcpy(&addr, prefix.addr, 4);
        inet_ntop(AF_INET, &addr, buffer, buffer_size);
        sprintf(buffer + strlen(buffer), "/%d", prefix.get_mask_len());
        return true;
    } else if (prefix.get_family() == 6) {
        if (buffer_size < INET6_ADDRSTRLEN + 4) return false;
        struct in6_addr addr6;
        memcpy(&addr6.s6_addr, prefix.addr, 14);
        memset(addr6.s6_addr + 14, 0, 2);
        inet_ntop(AF_INET6, &addr6, buffer, buffer_size);
        sprintf(buffer + strlen(buffer), "/%d", prefix.get_mask_len());
        return true;
    }
    return false;
}

// Convert from string to Prefix, supports IPv4 "a.b.c.d/len" and IPv6 "addr/len"
Prefix stringToPrefix(const char* str) {
    char ip_str[INET6_ADDRSTRLEN];
    int len;
    if (sscanf(str, "%[^/]/%d", ip_str, &len) != 2) {
        throw std::invalid_argument("Invalid prefix format");
    }
    if (strchr(ip_str, ':')) {
        // IPv6
        struct in6_addr addr6;
        if (inet_pton(AF_INET6, ip_str, &addr6) != 1) {
            throw std::invalid_argument("Invalid IPv6 address");
        }
        return Prefix(6, addr6.s6_addr, len);
    } else {
        // IPv4
        struct in_addr addr;
        if (inet_pton(AF_INET, ip_str, &addr) != 1) {
            throw std::invalid_argument("Invalid IPv4 address");
        }
        return Prefix(4, addr.s_addr, len);
    }
}

// Record layout constants shared by parser and monitor.
constexpr size_t TIME_T_SIZE = sizeof(time_t);
constexpr size_t BGP_RECORD_META_SIZE = sizeof(time_t) + sizeof(uint32_t) + sizeof(uint32_t) + 4;
constexpr size_t AS_PAIR_SIZE = (256 - sizeof(Prefix) - BGP_RECORD_META_SIZE) / sizeof(ASPair);
// Signal bit flags carried by BGPRecord::signal.
constexpr uint8_t STAGE_END_AF = (1u << 0); // End-of-phase marker.
constexpr uint8_t BASELINE_SWITCH_AF = (1u << 1); // Baseline switch marker.
constexpr uint8_t UPD_WINDOW_END_AF = (1u << 2); // End-of-update-window marker.

// Community label values for BGPRecord::community_label.
// A non-zero value indicates a well-known community was detected on the route.
constexpr uint8_t COMMUNITY_NONE      = 0; // No special community.
constexpr uint8_t COMMUNITY_BLACKHOLE = 1; // RFC 7999 BLACKHOLE community (65535:666).

// Route event types carried by BGPRecord::route_event.
constexpr uint8_t ROUTE_EVENT_ANNOUNCE = 0;
constexpr uint8_t ROUTE_EVENT_WITHDRAW  = 1;

// Training sample policy for mixed RIB+UPD dataset construction.
constexpr time_t TRAINING_UPD_LOOKBACK_SEC = 14 * 24 * 3600;
constexpr float TRAINING_RIB_SAMPLE_WEIGHT = 1.0f;
constexpr float TRAINING_UPD_SAMPLE_WEIGHT = 0.2f;

// Fixed-size record exchanged through the lock-free ring buffer.
struct BGPRecord {
    // Network prefix.
    Prefix prefix;
    // Record timestamp in seconds.
    time_t timestamp;
    // Origin AS (4 bytes)
    uint32_t origin_as;
    // Peer ASN that emitted this element.
    uint32_t peer_asn;
    // Control signal for phase/window boundaries.
    uint8_t signal;
    // Number of valid entries in as_pairs.
    uint8_t as_pair_count;
    // Well-known community tag for this route (see COMMUNITY_* constants).
    uint8_t community_label;
    // Route event type for update semantics (see ROUTE_EVENT_* constants).
    uint8_t route_event;
    // AS links extracted from the AS path.
    ASPair as_pairs[AS_PAIR_SIZE];
    
    // Zero-initialize all fields.
    BGPRecord() : timestamp(0), origin_as(0), peer_asn(0), prefix{}, signal(0), as_pair_count(0), community_label(COMMUNITY_NONE), route_event(ROUTE_EVENT_ANNOUNCE), as_pairs{} {}
};

static_assert(sizeof(BGPRecord) == 256, "BGPRecord must remain 256 bytes");

// Parser phase carried by work batches.
enum class ProcessingPhase {
    RIB_PHASE,      // Building baseline graph from RIB snapshots.
    UPD_PHASE       // Processing update stream and anomaly windows.
};

#endif // HELPERS_HPP
