#ifndef OMTALK_MUTATORLOCK_H
#define OMTALK_MUTATORLOCK_H

#include <omtalk/MemoryManager.h>
#include <omtalk/Util/Annotations.h>

namespace omtalk::gc {

/// Requests to pause all mutator threads.
template <typename S>
class OMTALK_SCOPED_CAPABILITY MutatorLock {
public:
  MutatorLock(MemoryManager<S> &mm) noexcept OMTALK_ACQUIRE(mm) : mm(mm) {
    lock();
  }

  MutatorLock(const MutatorLock &) = delete;
  
  MutatorLock &operator=(const MutatorLock &) = delete;

  ~MutatorLock() noexcept OMTALK_RELEASE() { unlock(); }

  void lock() noexcept OMTALK_ACQUIRE() { mm.pauseMutators(); }

  void unlock() noexcept OMTALK_RELEASE() { mm.unpauseMutators(); }

private:

  MemoryManager<S> &mm;
};

} // namespace omtalk::gc

#endif
