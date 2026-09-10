#include <cstdio>
#include <thread>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <memory>
#include <limits>
#include <algorithm>
#include <arpa/inet.h>
#include <getopt.h>
#include <fstream>
#include <cctype>
#include <cstdlib>

#include "lockfree_ringbuffer.hpp"
#include "propagation_graph.hpp"
#include "topology_analyzer.hpp"
#include "my_queue.hpp"

using namespace std;

/*
 * Fine-grained concurrent BGP Consumer implementation
 * - Uses sharded lock mechanism for concurrent-safe graph, significantly reducing lock contention
 * - Supports higher concurrency and better performance scalability
 * - Main thread: reads BGP records from RingBuffer
 * - Worker thread pool: processes BGP records and updates graph structure
 * - Monitoring thread: periodically outputs statistics and detects anomalies
 */

constexpr size_t CHECK_FREQ = 50; // Check frequency
constexpr size_t DEFAULT_WORKER_THREAD_COUNT = 16; // Default worker thread count
constexpr size_t DEFAULT_ANOMALY_THREAD_COUNT = 12; // Default anomaly detection thread count
constexpr auto RETRY_INTERVAL_BUSY_MS = chrono::microseconds(10); // Busy wait retry interval time
constexpr auto RETRY_INTERVAL_IDLE_MS = chrono::microseconds(1); // Idle wait retry interval time
constexpr time_t DEFAULT_IP_LEASING_MIN_QUIET_SECONDS = 12 * 60 * 60; // 12h quiet period after withdrawal.
constexpr time_t DEFAULT_IP_LEASING_STATE_RETENTION_SECONDS = 180LL * 24 * 60 * 60; // 180d retention for prefix state.
// Global log file pointer
FILE* g_monitor_log = nullptr;

// Prefix information structure
struct PrefixInfo {
    bool is_ipv4;
    union {
        uint32_t ipv4_addr;
        uint8_t ipv6_addr[16];
    };
    uint8_t prefix_len;
    string prefix_str;
};

// Work task structure - use fixed-size array for better performance
struct WorkBatch {
    array<BGPRecord, BATCH_SIZE> records;  // Fixed-size array
    size_t actual_size;  // Actual number of valid records
    uint64_t batch_id;
    chrono::high_resolution_clock::time_point enqueue_time;
    ProcessingPhase phase;  // Processing phase
    
    WorkBatch() : actual_size(0), batch_id(0), phase(ProcessingPhase::RIB_PHASE) {}
    
    // Construct from external array
    WorkBatch(const BGPRecord* data, size_t size, uint64_t id, ProcessingPhase batch_phase) 
        : actual_size(size), batch_id(id), phase(batch_phase), enqueue_time(chrono::high_resolution_clock::now()) {
        if (size > 0 && size <= BATCH_SIZE) {
            memcpy(records.data(), data, size * sizeof(BGPRecord));
        }
    }
    
    // Get span view of valid data
    const BGPRecord* data() const { return records.data(); }
    size_t size() const { return actual_size; }
    bool empty() const { return actual_size == 0; }
};

// Fine-grained concurrent graph updater
class AsyncGraphUpdater {
private:
    struct PrefixLeaseState {
        unordered_map<uint32_t, uint32_t> peer_to_origin;
        unordered_map<uint32_t, uint32_t> origin_refcnt;
        unordered_set<uint32_t> active_origins;
        unordered_set<uint32_t> origins_before_empty;
        time_t last_withdraw_ts = 0;
        time_t last_empty_ts = 0;
        time_t last_activity_ts = 0;
        bool complete_withdraw_seen = false;
    };

    PrefixPropGraph& graph_;
    MyQueue<WorkBatch> work_queue_;
    vector<thread> worker_threads_;
    atomic<bool> running_{true};
    atomic<uint64_t> processed_batches_{0};
    atomic<uint64_t> processed_records_{0};
    atomic<size_t> total_processing_time_ns_{0};
    atomic<uint64_t> active_workers_{0};
    
    // Anomaly detection support
    TopologyAnalyzer* anomaly_processor_{nullptr};
    mutable mutex lease_state_mutex_;
    unordered_map<Prefix, PrefixLeaseState> lease_state_;
    atomic<uint64_t> lease_cleanup_tick_{0};

    void cleanupLeaseState(time_t now_ts) {
        if (now_ts <= 0) return;
        lock_guard<mutex> lock(lease_state_mutex_);
        for (auto it = lease_state_.begin(); it != lease_state_.end(); ) {
            if (it->second.last_activity_ts > 0 &&
                now_ts - it->second.last_activity_ts > DEFAULT_IP_LEASING_STATE_RETENTION_SECONDS) {
                it = lease_state_.erase(it);
            } else {
                ++it;
            }
        }
    }

    unordered_set<Prefix> detectIpLeasingPrefixes(const WorkBatch& batch) {
        unordered_set<Prefix> tagged_prefixes;
        const size_t batch_size = batch.size();
        if (batch_size == 0) return tagged_prefixes;

        lock_guard<mutex> lock(lease_state_mutex_);
        for (size_t i = 0; i < batch_size; ++i) {
            const BGPRecord& record = batch.data()[i];
            auto& state = lease_state_[record.prefix];
            state.last_activity_ts = max(state.last_activity_ts, record.timestamp);

            if (record.peer_asn == 0) {
                continue;
            }

            if (record.route_event == ROUTE_EVENT_WITHDRAW) {
                auto it_peer = state.peer_to_origin.find(record.peer_asn);
                if (it_peer == state.peer_to_origin.end()) {
                    continue;
                }

                const auto origins_snapshot = state.active_origins;
                uint32_t old_origin = it_peer->second;
                state.peer_to_origin.erase(it_peer);

                auto it_cnt = state.origin_refcnt.find(old_origin);
                if (it_cnt != state.origin_refcnt.end()) {
                    if (it_cnt->second > 1) {
                        --it_cnt->second;
                    } else {
                        state.origin_refcnt.erase(it_cnt);
                        state.active_origins.erase(old_origin);
                    }
                }

                if (state.peer_to_origin.empty()) {
                    state.complete_withdraw_seen = true;
                    state.last_empty_ts = record.timestamp;
                    state.last_withdraw_ts = record.timestamp;
                    state.origins_before_empty = origins_snapshot;
                }

                if (record.timestamp >= state.last_withdraw_ts) {
                    state.last_withdraw_ts = record.timestamp;
                }
                continue;
            }

            if (record.route_event != ROUTE_EVENT_ANNOUNCE ||
                record.origin_as == 0) {
                continue;
            }

            bool was_empty = state.peer_to_origin.empty();

            auto it_peer = state.peer_to_origin.find(record.peer_asn);
            if (it_peer != state.peer_to_origin.end()) {
                uint32_t old_origin = it_peer->second;
                if (old_origin != record.origin_as) {
                    auto it_cnt = state.origin_refcnt.find(old_origin);
                    if (it_cnt != state.origin_refcnt.end()) {
                        if (it_cnt->second > 1) {
                            --it_cnt->second;
                        } else {
                            state.origin_refcnt.erase(it_cnt);
                            state.active_origins.erase(old_origin);
                        }
                    }
                    it_peer->second = record.origin_as;
                    state.origin_refcnt[record.origin_as] += 1;
                    state.active_origins.insert(record.origin_as);
                }
            } else {
                state.peer_to_origin[record.peer_asn] = record.origin_as;
                state.origin_refcnt[record.origin_as] += 1;
                state.active_origins.insert(record.origin_as);
            }

            if (was_empty) {
                bool quiet_ok = state.complete_withdraw_seen &&
                                state.last_empty_ts > 0 &&
                                (record.timestamp - state.last_empty_ts >= DEFAULT_IP_LEASING_MIN_QUIET_SECONDS);
                bool origin_switched = state.origins_before_empty.find(record.origin_as) == state.origins_before_empty.end();
                if (quiet_ok && origin_switched) {
                    tagged_prefixes.insert(record.prefix);
                }
                state.complete_withdraw_seen = false;
                state.origins_before_empty.clear();
            }
        }

        return tagged_prefixes;
    }
    
    // Worker thread function
    void workerLoop(int worker_id) {
        WorkBatch batch;
        while (running_.load(memory_order_relaxed) || work_queue_.size() > 0) {
            if (work_queue_.dequeue(batch)) {
                active_workers_.fetch_add(1, memory_order_relaxed);
                auto start_time = chrono::high_resolution_clock::now();
                
                processBGPRecordsBatch(batch);
                
                auto end_time = chrono::high_resolution_clock::now();
                auto processing_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
                auto queue_wait_time = chrono::duration_cast<chrono::milliseconds>(start_time - batch.enqueue_time);
                
                // Update statistics
                processed_batches_.fetch_add(1, memory_order_relaxed);
                processed_records_.fetch_add(batch.size(), memory_order_relaxed);
                total_processing_time_ns_.fetch_add(static_cast<size_t>(processing_time.count()), memory_order_relaxed);
                active_workers_.fetch_sub(1, memory_order_relaxed);
            }
        }
    }
    
    // Batch process BGP records
    void processBGPRecordsBatch(const WorkBatch& batch) {
        if (batch.empty()) return;
        
        size_t batch_size = batch.size();
        
        // Pre-aggregation data structures
        unordered_map<ASPair, unordered_map<Prefix, pair<time_t, uint32_t>>> updates;
        unordered_map<uint32_t, unordered_set<uint32_t>> as_neighbors; 
        updates.reserve(2 * batch_size);
        
        // Process each record and aggregate data
        for (size_t i = 0; i < batch_size; ++i) {
            const BGPRecord& record = batch.data()[i];
            if (record.as_pair_count == 0) continue;
            uint32_t origin_as = record.origin_as;

            for (int j = 0; j < record.as_pair_count; j++) {
                const ASPair& as_pair = record.as_pairs[j];
                as_neighbors[as_pair.from_as].insert(as_pair.to_as);

                auto& ts_origin_ref = updates[as_pair][record.prefix];
                if (ts_origin_ref.first == 0) {
                    ts_origin_ref = {record.timestamp, origin_as};
                } else {
                    ts_origin_ref.first = max(ts_origin_ref.first, record.timestamp);
                    ts_origin_ref.second = origin_as;
                }
            }
        }

        if (!as_neighbors.empty()) {
            graph_.updateASNeighbors(move(as_neighbors));
        }

        // Update leasing state from both phases so UPD starts with warmed active-peer view.
        unordered_set<Prefix> ip_leasing_prefixes = detectIpLeasingPrefixes(batch);

        // Decide whether to perform anomaly detection based on phase
        if (batch.phase == ProcessingPhase::UPD_PHASE) {
            // Collect prefixes carrying a blackhole community in this batch.
            unordered_set<Prefix> blackhole_prefixes;
            for (size_t i = 0; i < batch_size; ++i) {
                const BGPRecord& record = batch.data()[i];
                if (record.community_label == COMMUNITY_BLACKHOLE) {
                    blackhole_prefixes.insert(record.prefix);
                }
            }

            // Batch update and get new links
            time_t default_ts = batch.data()[batch_size - 1].timestamp;
            auto new_links = graph_.updateASLinksAndGetNew(move(updates));
            PrefixAnomalyBatch anomaly_batch(default_ts, batch.batch_id,
                move(new_links)); 
            anomaly_batch.blackhole_prefixes = move(blackhole_prefixes);
            anomaly_batch.ip_leasing_prefixes = move(ip_leasing_prefixes);

            // Amortize retention cleanup to keep per-prefix state bounded.
            if ((lease_cleanup_tick_.fetch_add(1, memory_order_relaxed) & 0x3ffu) == 0) {
                cleanupLeaseState(default_ts);
            }
            
            if (anomaly_processor_ && !anomaly_batch.isEmpty()) {
                anomaly_processor_->submitPrefixLinkBatch(move(anomaly_batch));
            }
        } else {
            // RIB phase: only update graph structure, no anomaly detection
            graph_.updateASLinks(move(updates));
        }
    }

public:
    explicit AsyncGraphUpdater(PrefixPropGraph& graph, int num_workers = 8, TopologyAnalyzer* anomaly_processor = nullptr) 
        : graph_(graph), anomaly_processor_(anomaly_processor) {
        
        // Start worker threads
        for (int i = 0; i < num_workers; i++) {
            worker_threads_.emplace_back(&AsyncGraphUpdater::workerLoop, this, i);
        }
        fprintf(g_monitor_log, "AsyncGraphUpdater started with %d worker threads\n", num_workers);
    }
    
    ~AsyncGraphUpdater() {
        shutdown();
    }
    
    void submitBatch(const BGPRecord* data, size_t size, const ProcessingPhase& phase) {
        static atomic<uint64_t> batch_counter{0};
        uint64_t batch_id = batch_counter.fetch_add(1, memory_order_relaxed);
        
        WorkBatch batch(data, size, batch_id, phase);
        work_queue_.enqueue(move(batch));
    }
    
    void shutdown() {
        if (running_.exchange(false, memory_order_relaxed)) {
            fprintf(g_monitor_log, "Shutting down AsyncGraphUpdater...\n");
            
            work_queue_.shutdown();
            
            for (auto& thread : worker_threads_) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            
            fprintf(g_monitor_log, "AsyncGraphUpdater shutdown complete\n");
        }
    }
    
    struct UpdaterStats {
        uint64_t processed_batches;
        uint64_t processed_records;
        double avg_processing_time_us;
        typename MyQueue<WorkBatch>::QueueStats queue_stats;
    };
    
    UpdaterStats getStats() const {
        auto queue_stats = work_queue_.getStats();
        uint64_t batches = processed_batches_.load(memory_order_relaxed);
        size_t total_processing_ns = total_processing_time_ns_.load(memory_order_relaxed);
        
        return {
            batches,
            processed_records_.load(memory_order_relaxed),
            batches > 0 ? (double)total_processing_ns / batches / 1000.0 : 0.0,
            queue_stats
        };
    }
    
    bool isIdle() const {
        return work_queue_.size() == 0 && active_workers_.load(memory_order_relaxed) == 0;
    }

    void waitUntilIdle() const {
        while (!isIdle()) {
            this_thread::sleep_for(chrono::microseconds(100));
        }
    }
};

size_t read_aligned_batch_concurrent(LockFreeRingBufferManager& manager, BGPRecord* buffer) {
    // Get alignment information
    size_t alignment_records = manager.get_read_alignment_records_needed();
    size_t target_read_size = (alignment_records == 0) ? BATCH_SIZE : alignment_records;
    
    size_t records_read = manager.read_batch(buffer, target_read_size);
    
    return records_read;
}

// Control markers are synthetic records emitted by the producer signal path.
// Guarding this shape avoids misclassifying legacy/dirty data as control flags.
static inline bool is_control_marker_record(const BGPRecord& rec) {
    if (rec.signal == 0) {
        return false;
    }
    return rec.origin_as == 0 && rec.peer_asn == 0 && rec.as_pair_count == 0;
}

// Display help information
void print_help(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  -p, --parser-threads NUM     Number of parser threads (default: %zu)\n", DEFAULT_WORKER_THREAD_COUNT);
    printf("  -a, --anomaly-threads NUM    Number of anomaly detection threads (default: %zu)\n", DEFAULT_ANOMALY_THREAD_COUNT);
    printf("  -l, --monitor-log FILE      Monitor output log file (default: monitor.log)\n");
    printf("  -c, --csv-filename FILE      CSV output file for anomaly links (default: anomaly_links.csv)\n");
    printf("  -h, --help           Show this help message\n");
    printf("\nDescription:\n");
    printf("  Fine-Grained Concurrent BGP Consumer with configurable parser threads.\n");
    printf("  Uses fine-grained sharded locking for superior concurrency performance.\n");
    printf("\nExamples:\n");
    printf("  %s                    # Use default %zu parser threads and %zu anomaly threads\n", program_name, 
        DEFAULT_WORKER_THREAD_COUNT, DEFAULT_ANOMALY_THREAD_COUNT);
    printf("  %s -p 8               # Use 8 parser threads\n", program_name);
    printf("  %s --parser-threads 16       # Use 16 parser threads\n", program_name);
    printf("  %s -a 4               # Use 4 anomaly detection threads\n", program_name);
    printf("  %s -l ./log/monitor.log     # Specify log files\n", program_name);
    printf("\n");
}

// Parse command line arguments
struct Config {
    size_t parser_threads;
    size_t anomaly_threads;
    string monitor_log_file;
    string csv_filename;
    
    Config() : parser_threads(DEFAULT_WORKER_THREAD_COUNT), anomaly_threads(DEFAULT_ANOMALY_THREAD_COUNT),
               monitor_log_file("monitor.log"), csv_filename("anomaly_links.csv") {}
};

Config parse_arguments(int argc, char* argv[]) {
    Config config;
    
    static struct option long_options[] = {
        {"parser-threads", required_argument, 0, 'p'},
        {"anomaly-threads", required_argument, 0, 'a'},
        {"monitor-log", required_argument, 0, 'l'},
        {"csv-filename", required_argument, 0, 'c'},
        {"processed-data", required_argument, 0, 'P'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "p:a:l:c:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'p': {
                int workers = atoi(optarg);
                if (workers <= 0 || workers > 128) {
                    fprintf(stderr, "Error: Parser threads must be between 1 and 128\n");
                    exit(1);
                }
                config.parser_threads = static_cast<size_t>(workers);
                break;
            }
            case 'a': {
                int threads = atoi(optarg);
                if (threads <= 0 || threads > 64) {
                    fprintf(stderr, "Error: Anomaly threads must be between 1 and 64\n");
                    exit(1);
                }
                config.anomaly_threads = static_cast<size_t>(threads);
                break;
            }
            case 'l': {
                config.monitor_log_file = optarg;
                break;
            }
            case 'c': {
                config.csv_filename = optarg;
                break;
            }
            case 'h':
                print_help(argv[0]);
                exit(0);
            case '?':
                fprintf(stderr, "Error: Unknown option or missing argument\n");
                print_help(argv[0]);
                exit(1);
            default:
                fprintf(stderr, "Error: Unexpected option\n");
                exit(1);
        }
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    Config config = parse_arguments(argc, argv);
    
    // Initialize monitor log file
    g_monitor_log = fopen(config.monitor_log_file.c_str(), "w");
    if (!g_monitor_log) {
        fprintf(stderr, "Error: Failed to open monitor log file: %s\n", config.monitor_log_file.c_str());
        exit(1);
    }
    
    LockFreeRingBufferManager manager;
    
    fprintf(g_monitor_log, "Initializing Fine-Grained Concurrent BGP Consumer...\n");
    fprintf(g_monitor_log, "Configuration: %zu parser threads, %zu anomaly threads\n", config.parser_threads, config.anomaly_threads);
    
    // Use fine-grained concurrent-safe graph structure
    PrefixPropGraph graph(1000000, 200000, 100000, 800000);
    
    // Wait for producer
    fprintf(g_monitor_log, "Fine-Grained Consumer waiting for producer...\n");
    while (!manager.init_consumer()) {
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    
    // Start anomaly detection processor
    string full_csv_path = config.csv_filename;
    TopologyAnalyzer anomaly_detector(graph, config.anomaly_threads, 1200000, full_csv_path.c_str());
    // Start fine-grained asynchronous graph updater
    AsyncGraphUpdater graph_updater(graph, config.parser_threads, &anomaly_detector);
    
    fprintf(g_monitor_log, "Fine-Grained Consumer started with %zu parser threads.\n", config.parser_threads);
    fprintf(g_monitor_log, "================================================================================\n");
    
    // Main loop
    uint64_t record_count = 0;
    int empty_reads = 0;
    uint64_t last_record_count = 0;
    BGPRecord record = {};
    auto start_time = chrono::high_resolution_clock::now();
    auto last_stats_time = start_time;
    

    ProcessingPhase current_phase = ProcessingPhase::RIB_PHASE;
    fprintf(g_monitor_log, "Concurrent Consumer initialized in RIB processing phase\n");
    
    // Use fixed-size array instead of vector to avoid dynamic resize overhead
    auto read_buffer = make_unique<array<BGPRecord, BATCH_SIZE>>();
        
    const auto stats_print_interval = chrono::seconds(3);
    int batch_count = 0;
    
    while (true) {
        size_t records_read = read_aligned_batch_concurrent(manager, read_buffer->data());
        if (records_read > 0) {
            // Check whether the batch contains phase-end markers (signal flags on the last record)
            const BGPRecord& last_record = (*read_buffer)[records_read - 1];
            const uint8_t signal_flags =
                is_control_marker_record(last_record) ? last_record.signal : 0;
            bool contains_stage_end = (signal_flags & STAGE_END_AF) != 0;
            bool contains_upd_window_end = (signal_flags & UPD_WINDOW_END_AF) != 0;
            
            record_count += records_read;
            graph_updater.submitBatch(read_buffer->data(), records_read, current_phase);

            if (contains_upd_window_end && current_phase == ProcessingPhase::UPD_PHASE) {
                graph_updater.waitUntilIdle();
                anomaly_detector.flushPending(last_record.timestamp);
                empty_reads = 0;
            }

            // If it contains stage end signal, switch phase after submitting batch
            if (contains_stage_end) {
                if (current_phase == ProcessingPhase::RIB_PHASE) {
                    current_phase = ProcessingPhase::UPD_PHASE;
                    fprintf(g_monitor_log, "=== RIB END SIGNAL RECEIVED - Switching to UPDATE phase ===\n");
                } else {
                    current_phase = ProcessingPhase::RIB_PHASE; // Reset for potential future use
                    fprintf(g_monitor_log, "=== UPDATE END SIGNAL RECEIVED - All processing completed ===\n");
                }
            }

            batch_count++;
            empty_reads = 0;
            
            if (batch_count % CHECK_FREQ == 0) {
                auto current_time = chrono::high_resolution_clock::now();
                
                if (current_time - last_stats_time >= stats_print_interval) {
                    auto elapsed = chrono::duration_cast<chrono::milliseconds>(current_time - last_stats_time);
                    uint64_t records_this_period = record_count - last_record_count;
                    
                    auto updater_stats = graph_updater.getStats();
                    fprintf(g_monitor_log, "=== Performance Stats (Elapsed: %.1fs) ===\n", elapsed.count() / 1000.0);
                    fprintf(g_monitor_log, "Records: %llu (+%llu), Rate: %.1f rec/s\n",
                           static_cast<unsigned long long>(record_count),
                           static_cast<unsigned long long>(records_this_period),
                           (double)records_this_period / elapsed.count() * 1000.0);
                    fprintf(g_monitor_log, "Phase: %s\n",
                           (current_phase == ProcessingPhase::RIB_PHASE ? "RIB" : "UPDATE"));
                    fprintf(g_monitor_log, "Updater: %lu batches, avg %.1fμs/batch, queue: %zu\n",
                           updater_stats.processed_batches, updater_stats.avg_processing_time_us,
                           updater_stats.queue_stats.current_size);
                                        
                    last_stats_time = current_time;
                    last_record_count = record_count;
                }
            }
        } else {
            empty_reads++;
            if (manager.is_finished()) {
                this_thread::sleep_for(RETRY_INTERVAL_IDLE_MS);
                if (empty_reads > 50) {
                    fprintf(g_monitor_log, "Producer finished, breaking...\n");
                    break;
                }
            } else {
                this_thread::sleep_for(RETRY_INTERVAL_BUSY_MS);
            }
        }
    }
    
    // Wait for processing to complete
    fprintf(g_monitor_log, "Waiting for all fine-grained async tasks to complete...\n");
    while (!graph_updater.isIdle()) {
        this_thread::sleep_for(RETRY_INTERVAL_BUSY_MS);
        auto stats = graph_updater.getStats();
    }
    anomaly_detector.waitUntilIdle();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    fprintf(g_monitor_log, "================================================================================\n");
    fprintf(g_monitor_log, "=== Concurrent Consumer Final Report ===\n");
    fprintf(g_monitor_log, "Total records processed: %llu\n",
           static_cast<unsigned long long>(record_count));
    fprintf(g_monitor_log, "Final phase: %s\n", (current_phase == ProcessingPhase::RIB_PHASE ? "RIB" : "UPDATE"));
    fprintf(g_monitor_log, "Total processing time: %.2f seconds\n", total_duration / 1000.0);
    if (total_duration > 0) {
        fprintf(g_monitor_log, "Overall average rate: %.1f records/sec\n", 
               (double)record_count / total_duration * 1000.0);
    }
    
    auto final_updater_stats = graph_updater.getStats();
    // Anomaly detection processor final statistics are managed by independent threads
    fprintf(g_monitor_log, "\n=== Graph Updater Final Statistics ===\n");
    fprintf(g_monitor_log, "Processed: %lu batches, %lu records\n", 
           final_updater_stats.processed_batches, final_updater_stats.processed_records);
    fprintf(g_monitor_log, "Average processing time: %.1fμs/batch\n", 
           final_updater_stats.avg_processing_time_us);
    fprintf(g_monitor_log, "Queue max size: %zu, avg wait: %.1fμs\n", 
           final_updater_stats.queue_stats.max_size_reached,
           final_updater_stats.queue_stats.avg_wait_time_us);
    
    fprintf(g_monitor_log, "\n=== Anomaly Detection Final Statistics ===\n");
    fprintf(g_monitor_log, "Anomaly detection running in independent thread\n");
    fprintf(g_monitor_log, "Detection method: Prefix-specific (prefix, from_as, to_as) link monitoring\n");
    
    auto final_graph_stats = graph.getGlobalStats();
    
    // Print memory usage statistics for graph data structure
    size_t graph_memory_mb = graph.getMemoryUsage() / 1024 / 1024;
    size_t snapshot_memory_mb = anomaly_detector.getMemoryUsage() / 1024 / 1024;
    size_t total_memory_mb = graph_memory_mb + snapshot_memory_mb;
    fprintf(g_monitor_log, "\n=== Memory Usage Statistics ===\n");
    fprintf(g_monitor_log, "Total AS edges: %zu, Total ASes: %zu\n", final_graph_stats.total_edges, final_graph_stats.total_ases);
    fprintf(g_monitor_log, "Graph memory usage: %zu MB\n", graph_memory_mb);
    fprintf(g_monitor_log, "Snapshot memory usage: %zu MB\n", snapshot_memory_mb);
    fprintf(g_monitor_log, "Total memory usage: %zu MB\n", total_memory_mb);
    
    // Shut down all asynchronous components
    graph_updater.shutdown();

    fprintf(g_monitor_log, "Shutting down Anomaly Detector (finishing queue processing and saving results)...\n");
    anomaly_detector.shutdown();
    fprintf(g_monitor_log, "Fine-Grained Consumer finished.\n");
    
    // Close log
    if (g_monitor_log) {
        fclose(g_monitor_log);
        g_monitor_log = nullptr;
    }

    // All externally visible work is complete at this point: asynchronous
    // workers are stopped, anomaly records are persisted, and the log is
    // closed.  Explicitly release the shared ring-buffer mappings, then avoid
    // spending minutes destructing the very large in-memory graph one node at
    // a time.  The operating system reclaims the remaining private memory.
    manager.cleanup();
    fflush(nullptr);
    std::_Exit(EXIT_SUCCESS);
}
