#ifndef OMTALK_MEMORYMANAGER_H
#define OMTALK_MEMORYMANAGER_H

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <omtalk/GlobalCollector.h>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>
#include <omtalk/Util/Atomic.h>
#include <omtalk/Util/Bytes.h>
#include <omtalk/Util/IntrusiveList.h>
#include <omtalk/Util/Math.h>
#include <sys/mman.h>
#include <thread>
#include <vector>

namespace omtalk::gc {

//===----------------------------------------------------------------------===//
// AllocationBuffer
//===----------------------------------------------------------------------===//

class AllocationBuffer final {
public:
  AllocationBuffer() = default;

  AllocationBuffer(std::byte *begin, std::byte *end) : begin(begin), end(end) {
    assert(begin <= end);
    assert(aligned(begin, OBJECT_ALIGNMENT));
  }

  Ref<void> tryAllocate(std::size_t size) {

    assert(aligned(size, OBJECT_ALIGNMENT));

    if (size > available()) {
      return nullptr;
    }
    auto allocation = Ref<void>(begin);
    begin += size;
    return allocation;
  }

  std::size_t available() const { return end - begin; }

  bool empty() const { return available() == 0; }

  std::byte *begin = nullptr;

  std::byte *end = nullptr;
};

//===----------------------------------------------------------------------===//
// MemoryManagerConfig
//===----------------------------------------------------------------------===//

struct MemoryManagerConfig {
  unsigned gcWorkerThreads = 1;
  unsigned initialMemory = mebibytes(1);
};

constexpr MemoryManagerConfig DEFAULT_MEMORY_MANAGER_CONFIG;

//===----------------------------------------------------------------------===//
// MemoryManager
//===----------------------------------------------------------------------===//

template <typename S>
class Context;

template <typename S>
using ContextList = IntrusiveList<Context<S>>;

template <typename S>
using ContextListNode = typename ContextList<S>::Node;

template <typename S>
class MemoryManager;

template <typename S>
struct MemoryManagerBuilder final {
  friend MemoryManager<S>;

  MemoryManagerBuilder() {}

  MemoryManager<S> build() { return MemoryManager<S>(std::move(*this)); }

  MemoryManagerBuilder &
  withRootWalker(std::unique_ptr<RootWalker<S>> &&rootWalker) {
    this->rootWalker = std::move(rootWalker);
    return *this;
  }

  MemoryManagerBuilder &withConfig(MemoryManagerConfig &config) {
    this->config = config;
    return *this;
  }

private:
  std::unique_ptr<RootWalker<S>> rootWalker;
  MemoryManagerConfig config;
};

template <typename S>
class MemoryManager final {
public:
  friend Context<S>;

  explicit MemoryManager(MemoryManagerBuilder<S> &&builder);

  ~MemoryManager();

  RootWalker<S> &getRootWalker() { return *rootWalker; }

  RegionManager &getRegionManager() { return regionManager; }

  GlobalCollector<S> &getGlobalCollector() { return globalCollector; }

  void setFreeList(FreeList list) { freeList = list; }

  /// Get the number of contexts currently attached to the MM
  unsigned getContextCount() { return contextCount.load(); }

  /// Get the number of threads which have access to the MM.  All threads must
  /// yield access to the MM before a GC can happen.
  unsigned getContextAccessCount() { return contextAccessCount.load(); }

  /// Get the current size of the heap in bytes.
  std::size_t getHeapSize() noexcept { return regionManager.getHeapSize(); }

  /// Signal to other threads that this thread wants exclusive access. Returns
  /// false if another thread has already requested it.  This will yield to
  /// another thread.
  bool requestExclusive(Context<S> &context);

  /// Remove request for exclusive access.  Must be called from the context
  /// which already hold exclusive access.
  void releaseExclusive(Context<S> &context);

  /// Returns if a thread has requested exclusive access
  bool exclusiveRequested();

  /// Refresh the allocation buffer associated with a thread.  May cause tax
  /// paying or garbage collection work to be done.
  bool refreshBuffer(Context<S> &context, std::size_t minimumSize);

  // Return the old region.  Remove it from the alloc list and put it into the
  // region list.
  void flushBuffer(Context<S> &context);

  /// Check if another thread is attempting to garbage collect.  Will yield
  /// access to the memory manager so another thread can collect.
  bool yieldForGC(Context<S> &context);

  /// Perform a global garbage collection.  This will wait for all attached
  /// threads to reach GC safe points.
  void collect(Context<S> &context);

  /// Start a global collection if one is not already occuring.
  void kickoff(Context<S> &context);

  /// Returns if we are in the marking phase of the GC.
  bool isMarking(const Context<S> &context) const;

  /// Enable the marking phase.
  void enableMarking();

  /// Disable the marking phase.
  void disableMarking();

private:
  /// Attach a context to the context list. Gives access to the context.
  void attach(Context<S> &context);

  /// Remove a context from the context list. Removes access from the context.
  void detach(Context<S> &context);

  /// Pause the context while waiting for the GC to complete.  If this
  /// context is the last active context, perform the GC.
  void waitOrGC(Context<S> &context);

  /// Perform a stop the world garbage collection.  All mutator threads must be
  /// paused.
  void performGC(Context<S> &context);

  MemoryManagerConfig config;
  RegionManager regionManager;
  GlobalCollector<S> globalCollector;
  std::unique_ptr<RootWalker<S>> rootWalker;
  ContextList<S> contexts;

  // If exclusive access is held, this points to the context
  std::mutex yieldForGcMutex;
  std::condition_variable yieldForGcCv;

  std::atomic<Context<S> *> exclusiveContext = nullptr;
  std::atomic<unsigned> contextCount = 0;
  std::atomic<unsigned> contextAccessCount = 0;

  std::atomic<bool> inMarkingPhase = false;

  /// Guards access to the global free list
  std::mutex freeListMutex;
  FreeList freeList;
};

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

template <typename S>
class Context final {
public:
  friend MemoryManager<S>;

  Context(MemoryManager<S> &memoryManager)
      : memoryManager(&memoryManager),
        gcContext(&memoryManager.getGlobalCollector()) {
    memoryManager.attach(*this);
  }

  ~Context() { memoryManager->detach(*this); }

  ContextListNode<S> &getListNode() noexcept { return listNode; }

  const ContextListNode<S> &getListNode() const noexcept { return listNode; }

  MemoryManager<S> *getCollector() { return memoryManager; }

  AllocationBuffer &buffer() { return ab; }

  GlobalCollectorContext<S> &getCollectorContext() noexcept {
    return gcContext;
  }

  // GC Notification

  /// If another thread has requested a collection, allow it to proceed.  All
  /// active contexts must yield before a collection can happen.
  bool yieldForGC();

  /// Perform a global garbage collection.  This will wait for all attached
  /// threads to reach GC safe points and yield.
  void collect();

  /// Refresh the allocation buffer associated with a thread.  May cause tax
  /// paying or garbage collection work to be done.
  bool refreshBuffer(std::size_t minimumSize);

  /// Return true if we are in the marking phase of the GC.
  bool isMarking() const;

private:
  MemoryManager<S> *memoryManager;
  GlobalCollectorContext<S> gcContext;
  ContextListNode<S> listNode;
  AllocationBuffer ab;
};

//===----------------------------------------------------------------------===//
// MemoryManager Inlines
//===----------------------------------------------------------------------===//

template <typename S>
MemoryManager<S>::MemoryManager(MemoryManagerBuilder<S> &&builder)
    : config(builder.config), globalCollector(this),
      rootWalker(std::move(builder.rootWalker)) {
  // round up the initial heap size to the number of regions.
  auto regionCount = ceilingDivide(config.initialMemory, REGION_SIZE);
  for (std::size_t i = 0; i < regionCount; i++) {
    regionManager.allocateEmptyRegion();
  }
}

template <typename S>
MemoryManager<S>::~MemoryManager() {}

template <typename S>
void MemoryManager<S>::attach(Context<S> &cx) {
  std::scoped_lock<std::mutex> lock(yieldForGcMutex);
  contextCount++;
  contextAccessCount++;
  contexts.push_front(&cx);
}

template <typename S>
void MemoryManager<S>::detach(Context<S> &cx) {
  std::scoped_lock<std::mutex> lock(yieldForGcMutex);
  contextCount--;
  contextAccessCount--;
  contexts.remove(&cx);
}

template <typename S>
void MemoryManager<S>::flushBuffer(Context<S> &context) {
  auto &buffer = context.buffer();
  if (buffer.begin != nullptr) {
    auto *region = Region::get(static_cast<void *>(buffer.begin));
    region->setFreeSpacePointer(buffer.begin);
    auto &allocRegionList = regionManager.getAllocateRegions();
    allocRegionList.remove(region);
    auto &regionList = regionManager.getRegions();
    regionList.push_front(region);
  }
  buffer.begin = nullptr;
  buffer.end = nullptr;
}

template <typename S>
bool MemoryManager<S>::refreshBuffer(Context<S> &context,
                                     std::size_t minimumSize) {
  {
    std::scoped_lock lock(regionManager);

    flushBuffer(context);

    auto *region = regionManager.getEmptyOrNewRegion();
    if (region != nullptr) {
      regionManager.getAllocateRegions().push_front(region);
      context.buffer().begin = region->heapBegin();
      context.buffer().end = region->heapEnd();
      return true;
    }
  }

  // Failed to allocate
  return false;
}

template <typename S>
bool MemoryManager<S>::exclusiveRequested() {
  return exclusiveContext != nullptr;
}

template <typename S>
void MemoryManager<S>::releaseExclusive(Context<S> &context) {
  contextAccessCount = contextCount.load();
  exclusiveContext = nullptr;
}

template <typename S>
void MemoryManager<S>::waitOrGC(Context<S> &context) {

  std::unique_lock yieldLock(yieldForGcMutex);
  contextAccessCount--;
  // If we are not the last thread, wait
  if (contextAccessCount != 0) {
    yieldForGcCv.wait(yieldLock, [this] { return exclusiveRequested(); });
  } else {
    globalCollector.collect(context.getCollectorContext());

    // Must remove exclusive request before waking up other threads
    releaseExclusive(context);

    // Wake up other threads
    yieldLock.unlock();
    yieldForGcCv.notify_all();
  }
}

template <typename S>
bool MemoryManager<S>::yieldForGC(Context<S> &context) {
  if (exclusiveRequested()) {
    waitOrGC(context);
    return true;
  }
  return false;
}

template <typename S>
void MemoryManager<S>::collect(Context<S> &context) {
  // If no other thread has requested exclusive, take it
  Context<S> *expected = nullptr;
  exclusiveContext.compare_exchange_strong(expected, &context);
  waitOrGC(context);
}

template <typename S>
void MemoryManager<S>::kickoff(Context<S> &context) {
  globalCollector.kickoff(context.getCollectorContext());
}

template <typename S>
bool MemoryManager<S>::isMarking(const Context<S> &context) const {
  return inMarkingPhase;
}

template <typename S>
void MemoryManager<S>::enableMarking() {
  inMarkingPhase = true;
}

template <typename S>
void MemoryManager<S>::disableMarking() {
  inMarkingPhase = false;
}

//===----------------------------------------------------------------------===//
// Context Inlines
//===----------------------------------------------------------------------===//

template <typename S>
bool Context<S>::yieldForGC() {
  return memoryManager->yieldForGC(*this);
}

template <typename S>
void Context<S>::collect() {
  memoryManager->collect(*this);
}

template <typename S>
bool Context<S>::refreshBuffer(std::size_t minimumSize) {
  return memoryManager->refreshBuffer(*this, minimumSize);
}

template <typename S>
bool Context<S>::isMarking() const {
  return memoryManager->isMarking(*this);
}

} // namespace omtalk::gc

#endif
