#ifndef OMTALK_MUTATOR_MUTEX_H
#define OMTALK_MUTATOR_MUTEX_H

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <omtalk/Util/Annotations.h>

namespace omtalk::gc {

/// Private mutex data
struct MutatorMutexData {
  /// 0 indicates shared state
  /// 1 indicates exclusive state
  std::uint8_t state;
  std::uintptr_t count;
};

/// Provides functionality similiar to a reader/writer lock.  Mutators all have
/// shared access to the mutex.  When exclusive access is requested all mutator
/// threads should advance to the next GC safepoint and yield.
class OMTALK_MUTEX_CAPABILITY MutatorMutex {
public:
  MutatorMutex() noexcept {}

  ~MutatorMutex() noexcept {}

  /// Attach a running mutator
  void lockShared() noexcept OMTALK_ACQUIRE_SHARED() {
    std::unique_lock lock(mutex);
    auto check = [this] { return data.state == 0; };
    yieldCV.wait(lock, check);
    data.count++;
  }

  /// Detach a running mutator
  void unlockShared() noexcept OMTALK_RELEASE_SHARED() {
    std::unique_lock lock(mutex);
    data.count--;
    if (data.count == 0) {
      requestCV.notify_all();
    }
  }

  /// Yield shared access and block until shared access is regained.
  bool yield() noexcept {
    if (!requested()) {
      return false;
    }
    std::unique_lock lock(mutex);
    data.count--;
    if (data.count == 0) {
      requestCV.notify_all();
    }
    auto check = [this] { return data.state == 0; };
    yieldCV.wait(lock, check);
    data.count++;
    return true;
  }

  /// Returns true if access was requested.
  bool requested() noexcept {
    std::scoped_lock lock(mutex);
    return data.state;
  }

  /// Return the attached count
  std::uintptr_t count() noexcept {
    std::scoped_lock lock(mutex);
    return data.count;
  }

  /// Acquire exclusive access, causing all mutators to pause.  Will block until
  /// all mutators are paused.
  void lock() noexcept OMTALK_ACQUIRE() {
    std::unique_lock lock(mutex);
    data.state = 1;
    requestCV.wait(lock, [this] { return data.count == 0; });
  }

  /// Release exclusive access, causing all mutators to resume.
  void unlock() noexcept OMTALK_RELEASE() {
    std::scoped_lock lock(mutex);
    data.state = 0;
    yieldCV.notify_all();
  }

  MutatorMutexData *getData() { return &data; }

private:
  std::mutex mutex;
  MutatorMutexData data = {0, 0};
  std::condition_variable yieldCV;
  std::condition_variable requestCV;
};

/// Requests to pause all mutator threads.
class OMTALK_SCOPED_CAPABILITY MutatorLock {
public:
  MutatorLock(MutatorMutex &mutex) noexcept OMTALK_ACQUIRE(mutex)
      : mutex(mutex) {
    lock();
  }

  ~MutatorLock() noexcept OMTALK_RELEASE() { unlock(); }

  void lock() noexcept OMTALK_ACQUIRE() { mutex.lock(); }

  void unlock() noexcept OMTALK_RELEASE() { mutex.unlock(); }

private:
  MutatorMutex &mutex;
};

} // namespace omtalk::gc

#endif