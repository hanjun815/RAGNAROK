/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file ThreadsafeQueue.hpp
 * @brief Header file for the ThreadsafeQueue class.
 * @author Simon Lynen
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_THREADSAFE_THREADSAFEQUEUE_HPP_
#define INCLUDE_OKVIS_THREADSAFE_THREADSAFEQUEUE_HPP_

#include <atomic>
#include <pthread.h>
#include <queue>
#include <sys/time.h>

#include <glog/logging.h>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// \brief Namespace for helper classes for threadsafe operation.
namespace threadsafe {

/**
 * @brief Class that implements a threadsafe FIFO queue.
 * @tparam QueueType Datatype that is safed in the queue.
 */
template<typename QueueType, class Allocator = std::allocator<QueueType>>
class Queue {

 public:

  /// \brief Notify all waiting threads. Only used in destructor and when shutting down.
  virtual void NotifyAll() const final {
    pthread_cond_broadcast(&condition_empty_);
    pthread_cond_broadcast(&condition_full_);
  }

  /// \brief Constructor.
  Queue() {
    shutdown_ = false;
    pthread_mutex_init(&mutex_, nullptr);
    pthread_cond_init(&condition_empty_, nullptr);
    pthread_cond_init(&condition_full_, nullptr);
  }

  /// \brief Destructor.
  virtual ~Queue() {
    shutdown_ = true;
    NotifyAll();
    pthread_mutex_destroy(&mutex_);
    pthread_cond_destroy(&condition_empty_);
    pthread_cond_destroy(&condition_full_);
  }

  /// \brief Tell the queue shut down. This will notify all threads to wake up.
  virtual void Shutdown() final {
    pthread_mutex_lock(&mutex_);
    shutdown_ = true;
    NotifyAll();
    pthread_mutex_unlock(&mutex_);
  }

  /// \brief Tell the queue to resume after a shutdown request.
  virtual void Resume() final {
    pthread_mutex_lock(&mutex_);
    shutdown_ = false;
    NotifyAll();
    pthread_mutex_unlock(&mutex_);
  }

  /// \brief Push non-blocking to the queue.
  void Push(const QueueType& value) {
    PushNonBlocking(value);
  }

  /// \brief Push to the queue.
  void PushNonBlocking(const QueueType& value) {
    pthread_mutex_lock(&mutex_);
    queue_.push(value);
    pthread_cond_signal(&condition_empty_);  // Signal that data is available.
    pthread_mutex_unlock(&mutex_);
  }

  /// \brief Return the size of the queue.
  virtual size_t Size() const final {
    pthread_mutex_lock(&mutex_);
    size_t size = queue_.size();
    pthread_mutex_unlock(&mutex_);
    return size;
  }

  /// \brief Return true if the queue is empty.
  virtual bool Empty() const final {
    pthread_mutex_lock(&mutex_);
    bool empty = queue_.empty();
    pthread_mutex_unlock(&mutex_);
    return empty;
  }

  /// \brief Push to the queue if the size is less than max_queue_size, else block.
  /// \param[in] value New entry in queue.
  /// \param[in] max_queue_size Maximum queue size.
  /// \return False if shutdown is requested.
  bool PushBlockingIfFull(const QueueType& value, size_t max_queue_size) {
    pthread_mutex_lock(&mutex_);
    while (queue_.size() >= max_queue_size && !shutdown_) {
      pthread_cond_wait(&condition_full_, &mutex_);
    }
    if (shutdown_) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }
    queue_.push(value);
    pthread_cond_signal(&condition_empty_);  // Signal that data is available.
    pthread_mutex_unlock(&mutex_);
    return true;
  }

  /// \brief Push to the queue. If full, drop the oldest entry.
  /// \param[in] value New entry in queue.
  /// \param[in] max_queue_size Maximum queue size.
  /// \return True if oldest was dropped because queue was full.
  bool PushNonBlockingDroppingIfFull(const QueueType& value,
                                     size_t max_queue_size) {
    pthread_mutex_lock(&mutex_);
    bool result = false;
    if (queue_.size() >= max_queue_size) {
      queue_.pop();
      result = true;
    }
    queue_.push(value);
    pthread_cond_signal(&condition_empty_);  // Signal that data is available.
    pthread_mutex_unlock(&mutex_);
    return result;
  }

  /**
   * @brief Get the oldest entry still in the queue. Blocking if queue is empty.
   * @param[out] value Oldest entry in queue.
   * @return False if shutdown is requested.
   */
  bool Pop(QueueType* value) {
    return PopBlocking(value);
  }

  /**
   * @brief Get the oldest entry still in the queue. Blocking if queue is empty.
   * @param[out] value Oldest entry in queue.
   * @return False if shutdown is requested.
   */
  bool PopBlocking(QueueType* value) {
    CHECK_NOTNULL(value);
    pthread_mutex_lock(&mutex_);
    while (queue_.empty() && !shutdown_) {
      pthread_cond_wait(&condition_empty_, &mutex_);
    }
    if (shutdown_) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }
    QueueType _value = queue_.front();
    queue_.pop();
    pthread_cond_signal(&condition_full_);  // Notify that space is available.
    pthread_mutex_unlock(&mutex_);
    *value = _value;
    return true;
  }

  /**
   * @brief Get the oldest entry still in the queue. If queue is empty value is not altered.
   * @param[out] value Oldest entry in queue if queue was not empty.
   * @return True if queue was not empty.
   */
  bool PopNonBlocking(QueueType* value) {
    CHECK_NOTNULL(value);
    pthread_mutex_lock(&mutex_);
    if (queue_.empty()) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }
    *value = queue_.front();
    queue_.pop();
    pthread_mutex_unlock(&mutex_);
    return true;
  }

  /**
   * @brief Get the oldest entry still in the queue. If the queue is empty wait for a given
   *        amount of time. If during this time an entry was pushed alter the value. If the
   *        queue is still empty, the value is not altered and it will return false
   * @param[out] value Oldest entry in queue if queue was not empty.
   * @param timeout_nanoseconds Maximum amount of time to wait for an entry if queue is empty.
   * @return True if value was updated. False if queue was empty and no new entry was pushed
   *         during the given timeout.
   */
  bool PopTimeout(QueueType* value, int64_t timeout_nanoseconds) {
    CHECK_NOTNULL(value);
    pthread_mutex_lock(&mutex_);
    if (queue_.empty()) {
      struct timeval tv;
      struct timespec ts;
      gettimeofday(&tv, nullptr);
      ts.tv_sec = tv.tv_sec;
      ts.tv_nsec = tv.tv_usec * 1e3 + timeout_nanoseconds;
      pthread_cond_timedwait(&condition_empty_, &mutex_, &ts);
    }
    if (queue_.empty()) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }
    QueueType _value = queue_.front();
    queue_.pop();
    pthread_cond_signal(&condition_full_);  // Notify that space is available.
    pthread_mutex_unlock(&mutex_);
    *value = _value;
    return true;
  }

  /**
   * @brief Get a copy of the front / oldest element in the queue. If queue is empty
   *        value is not altered. The queue itself is not changed, i.e. the returned
   *        element is still in the queue.
   * @param[out] value Oldest entry in queue if queue was not empty.
   * @return True if queue was not empty and value was updated.
   */
  bool getCopyOfFront(QueueType* value) {
    CHECK_NOTNULL(value);
    pthread_mutex_lock(&mutex_);
    if (queue_.empty()) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }
    // COPY the value.
    *value = queue_.front();
    pthread_mutex_unlock(&mutex_);
    return true;
  }

  /**
   * @brief Get a copy of the front / oldest element in the queue. Blocking if queue is empty.
   *        The queue itself is not changed, i.e. the returned element is still in the queue.
   * @param value Oldest entry in the queue.
   * @return False if shutdown is requested.
   */
  bool getCopyOfFrontBlocking(QueueType* value) {
    CHECK_NOTNULL(value);
    pthread_mutex_lock(&mutex_);
    while (queue_.empty() && !shutdown_) {
      pthread_cond_wait(&condition_empty_, &mutex_);
    }
    if (shutdown_) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }
    *value = queue_.front();
    pthread_mutex_unlock(&mutex_);
    return true;
  }

  /**
   * @brief Get a copy of the back / newest element in the queue. If queue is empty
   *        value is not altered. The queue itself is not changed, i.e. the returned
   *        element is still in the queue.
   * @param[out] value Newest entry in queue if queue was not empty.
   * @return True if queue was not empty and value was updated.
   */
  bool getCopyOfBack(QueueType* value) {
    CHECK_NOTNULL(value);
    pthread_mutex_lock(&mutex_);
    if (queue_.empty()) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }
    // COPY the value.
    *value = queue_.back();
    pthread_mutex_unlock(&mutex_);
    return true;
  }


  mutable pthread_mutex_t mutex_;                ///< The queue mutex.

  /// \brief Condition variable to wait and signal that queue is not empty.
  mutable pthread_cond_t condition_empty_;

  /// \brief Condition variable to wait and signal when an element is popped.
  mutable pthread_cond_t condition_full_;

  std::queue<QueueType,
      std::deque<QueueType, Allocator>> queue_;  ///< Actual queue.
  std::atomic_bool shutdown_;                    ///< Flag if shutdown is requested.

};

}  // namespace threadsafe

}  // namespace okvis

#endif  // INCLUDE_OKVIS_THREADSAFE_THREADSAFEQUEUE_HPP_
