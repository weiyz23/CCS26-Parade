#ifndef MY_QUEUE_HPP
#define MY_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>

using namespace std;

/*
 * High-performance thread-safe work queue
 * - Supports blocking enqueue and dequeue operations
 * - Provides detailed statistics
 * - Supports graceful shutdown
 */
template<typename T>
class MyQueue {
private:
    size_t capacity_;
    mutable mutex mtx_;
    condition_variable cv_;
    condition_variable cv_not_full_;
    queue<T> queue_;
    atomic<bool> shutdown_{false};
    atomic<size_t> total_enqueued_{0};
    atomic<size_t> total_dequeued_{0};
    atomic<size_t> max_queue_size_{0};
    atomic<size_t> total_wait_time_ns_{0};

public:

    MyQueue(size_t capacity = 64) : capacity_(capacity) {}

    void enqueue(T&& item) {
        {
            unique_lock<mutex> lock(mtx_);
            // Wait for queue to have space, instead of discarding data
            cv_not_full_.wait(lock, [this] {
                return queue_.size() < capacity_ || shutdown_.load(memory_order_relaxed);
            });

            if (shutdown_.load(memory_order_relaxed)) {
                return;
            }

            queue_.push(std::move(item));
            total_enqueued_.fetch_add(1, memory_order_relaxed);

            size_t current_size = queue_.size();
            size_t current_max = max_queue_size_.load(memory_order_relaxed);
            while (current_size > current_max &&
                   !max_queue_size_.compare_exchange_weak(current_max, current_size, memory_order_relaxed)) {
            }
        }
        cv_.notify_one();
    }

    bool dequeue(T& item, const chrono::nanoseconds& timeout = chrono::milliseconds(50)) {
        auto start_time = chrono::high_resolution_clock::now();
        unique_lock<mutex> lock(mtx_);

        bool success = cv_.wait_for(lock, timeout, [this] {
            return !queue_.empty() || shutdown_.load(memory_order_relaxed);
        });

        if (success && !queue_.empty()) {
            item = std::move(queue_.front());
            queue_.pop();
            total_dequeued_.fetch_add(1, memory_order_relaxed);

            // Notify producer that queue has space
            cv_not_full_.notify_one();

            auto end_time = chrono::high_resolution_clock::now();
            auto wait_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
            total_wait_time_ns_.fetch_add(static_cast<size_t>(wait_time.count()), memory_order_relaxed);

            return true;
        }
        return false;
    }

    void shutdown() {
        {
            lock_guard<mutex> lock(mtx_);
            shutdown_.store(true, memory_order_relaxed);
        }
        cv_.notify_all();
        cv_not_full_.notify_all(); // Notify all waiting producers
    }

    bool is_shutdown() const {
        return shutdown_.load(memory_order_relaxed);
    }

    size_t size() const {
        lock_guard<mutex> lock(mtx_);
        return queue_.size();
    }

    struct QueueStats {
        size_t current_size;
        size_t total_enqueued;
        size_t total_dequeued;
        size_t max_size_reached;
        size_t pending;
        double avg_wait_time_us;
    };

    QueueStats getStats() const {
        lock_guard<mutex> lock(mtx_);
        size_t dequeued = total_dequeued_.load(memory_order_relaxed);
        size_t total_wait_ns = total_wait_time_ns_.load(memory_order_relaxed);
        return {
            queue_.size(),
            total_enqueued_.load(memory_order_relaxed),
            dequeued,
            max_queue_size_.load(memory_order_relaxed),
            total_enqueued_.load(memory_order_relaxed) - dequeued,
            dequeued > 0 ? (double)total_wait_ns / dequeued / 1000.0 : 0.0
        };
    }
};

/*
 * High-performance thread-safe priority queue
 * - Supports blocking enqueue and dequeue operations, sorted by priority
 * - Provides detailed statistics
 * - Supports graceful shutdown
 */
template<typename T, typename Comparator = less<T>>
class PriorityMyQueue {
private:
    size_t capacity_;
    mutable mutex mtx_;
    condition_variable cv_;
    condition_variable cv_not_full_;
    priority_queue<T, vector<T>, Comparator> pq_;
    atomic<bool> shutdown_{false};
    atomic<size_t> total_enqueued_{0};
    atomic<size_t> total_dequeued_{0};
    atomic<size_t> max_queue_size_{0};
    atomic<size_t> total_wait_time_ns_{0};

public:

    PriorityMyQueue(size_t capacity = 64) : capacity_(capacity) {}

    void enqueue(T&& item) {
        {
            unique_lock<mutex> lock(mtx_);
            // Wait for queue to have space, instead of discarding data
            cv_not_full_.wait(lock, [this] {
                return pq_.size() < capacity_ || shutdown_.load(memory_order_relaxed);
            });

            if (shutdown_.load(memory_order_relaxed)) {
                return;
            }

            pq_.push(std::move(item));
            total_enqueued_.fetch_add(1, memory_order_relaxed);

            size_t current_size = pq_.size();
            size_t current_max = max_queue_size_.load(memory_order_relaxed);
            while (current_size > current_max &&
                   !max_queue_size_.compare_exchange_weak(current_max, current_size, memory_order_relaxed)) {
            }
        }
        cv_.notify_one();
    }

    bool dequeue(T& item, const chrono::nanoseconds& timeout = chrono::milliseconds(50)) {
        auto start_time = chrono::high_resolution_clock::now();
        unique_lock<mutex> lock(mtx_);

        bool success = cv_.wait_for(lock, timeout, [this] {
            return !pq_.empty() || shutdown_.load(memory_order_relaxed);
        });

        if (success && !pq_.empty()) {
            item = std::move(pq_.top());
            pq_.pop();
            total_dequeued_.fetch_add(1, memory_order_relaxed);

            // Notify producer that queue has space
            cv_not_full_.notify_one();

            auto end_time = chrono::high_resolution_clock::now();
            auto wait_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
            total_wait_time_ns_.fetch_add(static_cast<size_t>(wait_time.count()), memory_order_relaxed);

            return true;
        }
        return false;
    }

    void shutdown() {
        {
            lock_guard<mutex> lock(mtx_);
            shutdown_.store(true, memory_order_relaxed);
        }
        cv_.notify_all();
        cv_not_full_.notify_all(); // Notify all waiting producers
    }

    bool is_shutdown() const {
        return shutdown_.load(memory_order_relaxed);
    }

    size_t size() const {
        lock_guard<mutex> lock(mtx_);
        return pq_.size();
    }

    struct QueueStats {
        size_t current_size;
        size_t total_enqueued;
        size_t total_dequeued;
        size_t max_size_reached;
        size_t pending;
        double avg_wait_time_us;
    };

    QueueStats getStats() const {
        lock_guard<mutex> lock(mtx_);
        size_t dequeued = total_dequeued_.load(memory_order_relaxed);
        size_t total_wait_ns = total_wait_time_ns_.load(memory_order_relaxed);
        return {
            pq_.size(),
            total_enqueued_.load(memory_order_relaxed),
            dequeued,
            max_queue_size_.load(memory_order_relaxed),
            total_enqueued_.load(memory_order_relaxed) - dequeued,
            dequeued > 0 ? (double)total_wait_ns / dequeued / 1000.0 : 0.0
        };
    }
};

#endif // MY_QUEUE_HPP