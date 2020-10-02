#ifndef OMTALK_UTIL_MUTEX_H
#define OMTALK_UTIL_MUTEX_H

/// Contains annotated wrappers around regular standard library types.

#include <mutex>
#include <omtalk/Util/Annotations.h>

namespace omtalk {

/// Wrapper around std::mutex.
class OMTALK_MUTEX_CAPABILITY Mutex {
public:
  using UnderlyingMutexType = std::mutex;

  Mutex() = default;

  Mutex(const Mutex &) = delete;

  Mutex &operator=(const Mutex &) = delete;

  ~Mutex() = default;

  void lock() OMTALK_ACQUIRE() { m.lock(); }

  bool try_lock() noexcept OMTALK_TRY_ACQUIRE(true) { return m.try_lock(); }

  void unlock() noexcept OMTALK_RELEASE() { m.unlock(); }

  UnderlyingMutexType &getUnderlyingMutex() { return m; }

private:
  UnderlyingMutexType m;
};

/// Wrapper around a std::scoped_lock;
class OMTALK_SCOPED_CAPABILITY Lock {
public:
  using MutexType = Mutex;

  using UnderlyingLockType =
      std::unique_lock<typename Mutex::UnderlyingMutexType>;

  explicit Lock(MutexType &m) OMTALK_ACQUIRE(m) : l(m.getUnderlyingMutex()) {}

  explicit Lock(Mutex &m, std::adopt_lock_t adopt) OMTALK_REQUIRES(m)
      : l(m.getUnderlyingMutex(), adopt) {}

  Lock(const Lock &) = delete;

  Lock &operator=(const Lock &) = delete;

  ~Lock() OMTALK_RELEASE() {}

  UnderlyingLockType &getUnderlyingLock() { return l; }

private:
  UnderlyingLockType l;
};

/// Wrapper around std::unique_lock
class OMTALK_SCOPED_CAPABILITY UniqueLock {
public:
  using MutexType = Mutex;

  using UnderlyingLockType = std::unique_lock<Mutex::UnderlyingMutexType>;

  UniqueLock() noexcept : l() {}

  explicit UniqueLock(MutexType &m) OMTALK_ACQUIRE(m)
      : l(m.getUnderlyingMutex()) {}

  UniqueLock(MutexType &m, std::defer_lock_t defer) noexcept OMTALK_EXCLUDES(m)
      : l(m.getUnderlyingMutex(), defer) {}

  UniqueLock(MutexType &m, std::try_to_lock_t try_to)
      : l(m.getUnderlyingMutex(), try_to) {}

  UniqueLock(MutexType &m, std::adopt_lock_t adopt)
      : l(m.getUnderlyingMutex(), adopt) {}

  ~UniqueLock() OMTALK_RELEASE() {}

  UniqueLock(const UniqueLock &) = delete;

  UniqueLock &operator=(const UniqueLock &) = delete;

  void lock() OMTALK_ACQUIRE() { l.lock(); }

  bool try_lock() OMTALK_TRY_ACQUIRE(true) { return l.try_lock(); }

  void unlock() OMTALK_RELEASE() { return l.unlock(); }

  UnderlyingLockType &getUnderlyingLock() { return l; }

private:
  UnderlyingLockType l;
};

// Wrapper around std::condition_variable
class ConditionVariable {
public:
  using UnderlyingConditionVariableType = std::condition_variable;

  constexpr ConditionVariable() noexcept = default;

  ~ConditionVariable() = default;

  ConditionVariable(const ConditionVariable &) = delete;

  ConditionVariable &operator=(const ConditionVariable &) = delete;

  void notifyAll() noexcept { cv.notify_all(); }

  void notifyOne() noexcept { cv.notify_one(); }

  void wait(UniqueLock &l) noexcept { cv.wait(l.getUnderlyingLock()); }

  template <class Predicate>
  void wait(UniqueLock &l, Predicate p) {
    cv.wait(l.getUnderlyingLock(), p);
  }

  UnderlyingConditionVariableType &getUnderlyingConditionVariable() {
    return cv;
  }

private:
  UnderlyingConditionVariableType cv;
};

} // namespace omtalk

#endif
