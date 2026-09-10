#ifndef LOCKFREE_RINGBUFFER_H
#define LOCKFREE_RINGBUFFER_H

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <cstdio>
#include <atomic>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "helpers.hpp"


// RingBuffer configuration - use constexpr to ensure compile-time constants
constexpr size_t TOTAL_BUFFER_SIZE = 128 * 1024 * 1024; // 128 MB
constexpr size_t RECORD_SIZE = sizeof(BGPRecord);
// CAPACITY = 128*1024*1024 / 256 = 524288 entries
constexpr size_t BUFFER_CAPACITY = TOTAL_BUFFER_SIZE / RECORD_SIZE;

// 2MB huge page configuration constants
constexpr size_t HUGEPAGE_2MB = 2 * 1024 * 1024; // 2MB
constexpr size_t HUGEPAGE_SHIFT_2MB = 21; // log2(2MB)
constexpr size_t BATCH_SIZE = HUGEPAGE_2MB / RECORD_SIZE;

// Control information structure (regular shared memory)
struct RingBufferControl {
    std::atomic<size_t> read_pos{0};
    std::atomic<size_t> write_pos{0};
    std::atomic<bool> finished{false};
    std::atomic<size_t> total_written{0};
    std::atomic<size_t> total_read{0};
    alignas(64) std::atomic<size_t> read_cache{0};
    alignas(64) std::atomic<size_t> write_cache{0};
};

// Data cache structure (huge page shared memory)
struct RingBufferData {
    BGPRecord data[BUFFER_CAPACITY];
};

// Huge page utility functions - use dynamic configuration
namespace hugepage_utils {
    // Check if 2MB huge pages are available
    inline bool is_2mb_hugepage_available() {
        std::ifstream free_pages("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages");
        if (!free_pages.is_open()) return false;
        
        size_t free_count;
        free_pages >> free_count;
        return free_count > 0;
    }
    
    // Get the number of available 2MB huge pages
    inline size_t get_available_2mb_hugepages() {
        std::ifstream free_pages("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages");
        if (!free_pages.is_open()) return 0;
        
        size_t free_count;
        free_pages >> free_count;
        return free_count;
    }
    
    // Calculate the number of required 2MB huge pages
    inline size_t pages_needed_2mb() {
        return (TOTAL_BUFFER_SIZE + HUGEPAGE_2MB - 1) / HUGEPAGE_2MB;
    }
    
    // Get buffer size aligned to 2MB
    inline size_t get_aligned_buffer_size() {
        return ((TOTAL_BUFFER_SIZE + HUGEPAGE_2MB - 1) / HUGEPAGE_2MB) * HUGEPAGE_2MB;
    }
    
    // Check if 2MB hugetlbfs is mounted
    inline bool is_2mb_hugetlbfs_mounted() {
        struct stat st;
        return (stat("/mnt/huge2mb", &st) == 0 && S_ISDIR(st.st_mode));
    }
    
    // Print huge page status information
    inline void print_hugepage_info() {
        printf("=== Hugepage Information ===\n");
        printf("Target 2MB hugepage: %s\n", is_2mb_hugepage_available() ? "Available" : "Not Available");
        printf("2MB hugetlbfs mounted: %s\n", is_2mb_hugetlbfs_mounted() ? "Yes (/mnt/huge2mb)" : "No");
        printf("Buffer size: %zu bytes (%zu MB)\n", sizeof(RingBufferData), TOTAL_BUFFER_SIZE / (1024 * 1024));
        printf("2MB aligned buffer size: %zu bytes\n", get_aligned_buffer_size());
        printf("2MB pages needed: %zu\n", pages_needed_2mb());
        printf("Available 2MB hugepages: %zu\n", get_available_2mb_hugepages());
        printf("=============================\n");
    }
}

class LockFreeRingBufferManager {
private:
    int ctrl_fd; // Control block file descriptor
    int data_fd; // Data block file descriptor
    RingBufferControl* control; // Control block pointer
    RingBufferData* data;       // Data block pointer
    bool using_hugepages;       // Whether to use huge pages
    size_t data_size;           // Data block size

public:
    LockFreeRingBufferManager() : ctrl_fd(-1), data_fd(-1), control(nullptr), data(nullptr), using_hugepages(false), data_size(0) {}
    ~LockFreeRingBufferManager() { cleanup(); }

    static void print_size_info() {
        printf("=== LockFree RingBuffer Size Information ===\n");
        printf("BGPRecord size: %zu bytes\n", sizeof(BGPRecord));
        printf("Buffer capacity: %zu records\n", BUFFER_CAPACITY);
        printf("Total buffer size: %zu bytes (%zu MB)\n", 
               TOTAL_BUFFER_SIZE, TOTAL_BUFFER_SIZE / (1024*1024));
        printf("Control struct size: %zu bytes\n", sizeof(RingBufferControl));
        printf("Data struct size: %zu bytes\n", sizeof(RingBufferData));
        printf("=============================================\n");
        hugepage_utils::print_hugepage_info();
    }

    bool init_producer() {
        print_size_info();
        
        // Control block uses POSIX shared memory (regular pages)
        ctrl_fd = shm_open("/bgp_ringbuffer_ctrl", O_CREAT | O_RDWR, 0666);
        if (ctrl_fd == -1) { return false; }
        if (ftruncate(ctrl_fd, sizeof(RingBufferControl)) == -1) { return false; }
        control = (RingBufferControl*)mmap(0, sizeof(RingBufferControl), PROT_READ | PROT_WRITE, MAP_SHARED, ctrl_fd, 0);
        if (control == MAP_FAILED) { return false; }
        
        // placement new to initialize atomic variables
        new (&control->read_pos) std::atomic<size_t>(0);
        new (&control->write_pos) std::atomic<size_t>(0);
        new (&control->finished) std::atomic<bool>(false);
        new (&control->total_written) std::atomic<size_t>(0);
        new (&control->total_read) std::atomic<size_t>(0);
        new (&control->read_cache) std::atomic<size_t>(0);
        new (&control->write_cache) std::atomic<size_t>(0);

        // Data block: try to use hugepage filesystem, otherwise use regular file
        using_hugepages = hugepage_utils::is_2mb_hugepage_available() && hugepage_utils::is_2mb_hugetlbfs_mounted();
        
        if (using_hugepages) {
            data_size = hugepage_utils::get_aligned_buffer_size();
            data_fd = open("/mnt/huge2mb/bgp_ringbuffer_data", O_CREAT | O_RDWR, 0666);
            if (data_fd == -1) {
                using_hugepages = false;
            } else {
                if (ftruncate(data_fd, data_size) == -1) {
                    unlink("/mnt/huge2mb/bgp_ringbuffer_data");
                    using_hugepages = false;
                } else {
                    data = (RingBufferData*)mmap(0, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, data_fd, 0);
                    if (data == MAP_FAILED) {
                        close(data_fd);
                        unlink("/mnt/huge2mb/bgp_ringbuffer_data");
                        using_hugepages = false;
                    }
                }
            }
        }
        
        if (!using_hugepages) {
            // Use regular file mapping
            data_size = sizeof(RingBufferData);
            data_fd = shm_open("/bgp_ringbuffer_data", O_CREAT | O_RDWR, 0666);
            if (data_fd == -1) { return false; }
            if (ftruncate(data_fd, data_size) == -1) { return false; }
            data = (RingBufferData*)mmap(0, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, data_fd, 0);
            if (data == MAP_FAILED) { return false; }
        }
        return true;
    }

    bool init_consumer() {
        // Control block uses POSIX shared memory
        ctrl_fd = shm_open("/bgp_ringbuffer_ctrl", O_RDWR, 0666);
        if (ctrl_fd == -1) { return false; }
        control = (RingBufferControl*)mmap(0, sizeof(RingBufferControl), PROT_READ | PROT_WRITE, MAP_SHARED, ctrl_fd, 0);
        if (control == MAP_FAILED) { return false; }

        // Data block: try hugepage file first, then regular shared memory
        data_fd = open("/mnt/huge2mb/bgp_ringbuffer_data", O_RDWR, 0666);
        if (data_fd != -1) {
            // hugepage file exists
            struct stat st;
            if (fstat(data_fd, &st) == 0) {
                data_size = st.st_size;
                using_hugepages = true;
                data = (RingBufferData*)mmap(0, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, data_fd, 0);
                if (data == MAP_FAILED) {
                    close(data_fd);
                    data_fd = -1;
                    using_hugepages = false;
                }
            } else {
                close(data_fd);
                data_fd = -1;
                using_hugepages = false;
            }
        }
        
        if (data_fd == -1) {
            // Connect to regular POSIX shared memory
            data_fd = shm_open("/bgp_ringbuffer_data", O_RDWR, 0666);
            if (data_fd == -1) { return false; }
            struct stat st;
            if (fstat(data_fd, &st) == -1) { return false; }
            data_size = st.st_size;
            using_hugepages = false;
            data = (RingBufferData*)mmap(0, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, data_fd, 0);
            if (data == MAP_FAILED) { return false; }
        }
        return true;
    }

    bool write(const BGPRecord& record) {
        if (!control || !data) return false;
        size_t current_write = control->write_pos.load(std::memory_order_relaxed);
        size_t next_write = (current_write + 1) % BUFFER_CAPACITY;
        size_t cached_read = control->read_cache.load(std::memory_order_relaxed);
        if (next_write == cached_read) {
            cached_read = control->read_pos.load(std::memory_order_acquire);
            control->read_cache.store(cached_read, std::memory_order_relaxed);
            if (next_write == cached_read) return false;
        }
        data->data[current_write] = record;
        control->write_pos.store(next_write, std::memory_order_release);
        control->total_written.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool write_batch(const BGPRecord* records, size_t count) {
        if (!control || !data || count == 0 || count > BATCH_SIZE) return false;
        
        size_t current_write = control->write_pos.load(std::memory_order_relaxed);
        size_t cached_read = control->read_cache.load(std::memory_order_relaxed);
        
        // Check if there is enough space to write the entire batch
        size_t space_available;
        if (current_write >= cached_read) {
            space_available = BUFFER_CAPACITY - (current_write - cached_read) - 1;
        } else {
            space_available = cached_read - current_write - 1;
        }
        
        if (space_available < count) {
            // Refresh read pointer cache and check again
            cached_read = control->read_pos.load(std::memory_order_acquire);
            control->read_cache.store(cached_read, std::memory_order_relaxed);
            
            if (current_write >= cached_read) {
                space_available = BUFFER_CAPACITY - (current_write - cached_read) - 1;
            } else {
                space_available = cached_read - current_write - 1;
            }
            
            if (space_available < count) {
                return false; // Insufficient space
            }
        }
        
        // Use memcpy for efficient batch copying
        if (current_write + count <= BUFFER_CAPACITY) {
            // Single copy, no wraparound needed
            memcpy(&data->data[current_write], records, count * sizeof(BGPRecord));
        } else {
            // Wraparound case needed
            size_t first_part = BUFFER_CAPACITY - current_write;
            size_t second_part = count - first_part;
            
            memcpy(&data->data[current_write], records, first_part * sizeof(BGPRecord));
            memcpy(&data->data[0], &records[first_part], second_part * sizeof(BGPRecord));
        }
        
        size_t next_write = (current_write + count) % BUFFER_CAPACITY;
        control->write_pos.store(next_write, std::memory_order_release);
        control->total_written.fetch_add(count, std::memory_order_relaxed);
        return true;
    }

    bool read(BGPRecord& record) {
        if (!control || !data) return false;
        size_t current_read = control->read_pos.load(std::memory_order_relaxed);
        size_t cached_write = control->write_cache.load(std::memory_order_relaxed);
        if (current_read == cached_write) {
            cached_write = control->write_pos.load(std::memory_order_acquire);
            control->write_cache.store(cached_write, std::memory_order_relaxed);
            if (current_read == cached_write) {
                if (control->finished.load(std::memory_order_acquire)) return false;
                return false;
            }
        }
        record = data->data[current_read];
        size_t next_read = (current_read + 1) % BUFFER_CAPACITY;
        control->read_pos.store(next_read, std::memory_order_release);
        control->total_read.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Note: this is a best-effort batch read, may read less than max_count
    size_t read_batch(BGPRecord* buffer, size_t max_count) {
        if (!control || !data || max_count == 0) return 0;
        
        size_t current_read = control->read_pos.load(std::memory_order_relaxed);
        size_t cached_write = control->write_cache.load(std::memory_order_relaxed);
        
        // Calculate available data to read
        size_t available;
        if (cached_write >= current_read) {
            available = cached_write - current_read;
        } else {
            available = BUFFER_CAPACITY - current_read + cached_write;
        }
        
        if (available == 0) {
            // Refresh write pointer cache and check again
            cached_write = control->write_pos.load(std::memory_order_acquire);
            control->write_cache.store(cached_write, std::memory_order_relaxed);
            
            if (cached_write >= current_read) {
                available = cached_write - current_read;
            } else {
                available = BUFFER_CAPACITY - current_read + cached_write;
            }
            
            if (available == 0) {
                return 0; // No data available to read
            }
        }
        
        size_t to_read = std::min(available, max_count);
        
        // Use memcpy for efficient batch copying
        if (current_read + to_read <= BUFFER_CAPACITY) {
            // Single copy, no wraparound needed
            memcpy(buffer, &data->data[current_read], to_read * sizeof(BGPRecord));
        } else {
            // Wraparound case needed
            size_t first_part = BUFFER_CAPACITY - current_read;
            size_t second_part = to_read - first_part;
            
            memcpy(buffer, &data->data[current_read], first_part * sizeof(BGPRecord));
            memcpy(&buffer[first_part], &data->data[0], second_part * sizeof(BGPRecord));
        }
        
        size_t next_read = (current_read + to_read) % BUFFER_CAPACITY;
        control->read_pos.store(next_read, std::memory_order_release);
        control->total_read.fetch_add(to_read, std::memory_order_relaxed);
        return to_read;
    }

    void set_finished() {
        if (control) control->finished.store(true, std::memory_order_release);
    }
    bool is_finished() {
        return control ? control->finished.load(std::memory_order_acquire) : false;
    }

    size_t get_buffer_usage() {
        if (!control) return 0;
        size_t write_pos = control->write_pos.load(std::memory_order_relaxed);
        size_t read_pos = control->read_pos.load(std::memory_order_relaxed);
        if (write_pos >= read_pos) return write_pos - read_pos;
        else return BUFFER_CAPACITY - read_pos + write_pos;
    }

    // Get the number of records needed to align write pointer to BATCH_SIZE boundary
    // Returns 0 if already aligned, or the number of records needed to write to reach alignment
    size_t get_write_alignment_records_needed() const {
        size_t current_write = control->write_pos.load(std::memory_order_relaxed);
        return get_alignment_records_needed(current_write);
    }

    // Get the number of records needed to align read pointer to BATCH_SIZE boundary
    size_t get_read_alignment_records_needed() const {
        size_t current_read = control->read_pos.load(std::memory_order_relaxed);
        return get_alignment_records_needed(current_read);
    }

private:
    // Generic alignment calculation function - calculate the number of records needed to align specified position to BATCH_SIZE boundary
    static size_t get_alignment_records_needed(size_t position) {
        size_t current_batch_offset = position % BATCH_SIZE;
        return current_batch_offset == 0 ? 0 : BATCH_SIZE - current_batch_offset;
    }

public:

    void cleanup() {
        if (control) {
            control->read_pos.~atomic();
            control->write_pos.~atomic();
            control->finished.~atomic();
            control->total_written.~atomic();
            control->total_read.~atomic();
            control->read_cache.~atomic();
            control->write_cache.~atomic();
            munmap(control, sizeof(RingBufferControl));
            control = nullptr;
        }
        if (data && data_size > 0) {
            munmap(data, data_size);
            data = nullptr;
        }
        if (ctrl_fd != -1) { close(ctrl_fd); ctrl_fd = -1; }
        if (data_fd != -1) { close(data_fd); data_fd = -1; }
        data_size = 0;
        using_hugepages = false;
    }
};

#endif // LOCKFREE_RINGBUFFER_H