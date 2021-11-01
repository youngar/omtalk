#ifndef OM_GC_MEMORYMANAGER_H
#define OM_GC_MEMORYMANAGER_H

#include <ab/Util/Atomic.h>
#include <ab/Util/Bytes.h>
#include <ab/Util/IntrusiveList.h>
#include <ab/Util/Math.h>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <om/GC/GlobalCollector.h>
#include <om/GC/Heap.h>
#include <om/GC/MutatorMutex.h>
#include <om/GC/Ref.h>
#include <om/GC/Scheme.h>
#include <sys/mman.h>
#include <thread>
#include <vector>

namespace om::gc {

//===----------------------------------------------------------------------===//
// AllocationBuffer
//===----------------------------------------------------------------------===//

class AllocationBuffer final {
public:
  AllocationBuffer() = default;

  AllocationBuffer(std::byte *begin, std::byte *end) : begin(begin), end(end) {
    assert(begin <= end);
    assert(ab::aligned(begin, OBJECT_ALIGNMENT));
  }

  Ref<void> tryAllocate(std::size_t size) noexcept {

    assert(ab::aligned(size, OBJECT_ALIGNMENT));

    if (size > available()) {
      return nullptr;
    }
    auto allocation = Ref<void>(begin);
    begin += size;
    return allocation;
  }

  std::size_t available() const noexcept { return end - begin; }

  bool empty() const noexcept { return available() == 0; }

  std::byte *begin = nullptr;

  std::byte *end = nullptr;
};

//===----------------------------------------------------------------------===//
// MemoryManagerConfig
//===----------------------------------------------------------------------===//

struct MemoryManagerConfig {
  unsigned gcWorkerThreads = 1;
  unsigned initialMemory = ab::mebibytes(1);
};

constexpr MemoryManagerConfig DEFAULT_MEMORY_MANAGER_CONFIG;

//===----------------------------------------------------------------------===//
// MemoryManager
//===----------------------------------------------------------------------===//

template <typename S>
class Context;

template <typename S>
using ContextList = ab::IntrusiveList<Context<S>>;

template <typename S>
using ContextListNode = typename ContextList<S>::Node;

template <typename S>
class MemoryManager;

template <typename S>
class AuxMemoryManagerData {};

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

  MemoryManagerBuilder &withAuxData(AuxMemoryManagerData<S> &&auxData) {
    this->auxData = std::move(auxData);
    return *this;
  }

private:
  std::unique_ptr<RootWalker<S>> rootWalker;
  MemoryManagerConfig config;
  AuxMemoryManagerData<S> auxData;
};

template <typename S>
class MemoryManager final {
public:
  friend Context<S>;

  explicit MemoryManager(MemoryManagerBuilder<S> &&builder);

  ~MemoryManager();

  RootWalker<S> &getRootWalker() noexcept { return *rootWalker; }

  RegionManager &getRegionManager() noexcept { return regionManager; }

  GlobalCollector<S> &getGlobalCollector() noexcept { return globalCollector; }

  AuxMemoryManagerData<S> &getAuxData() noexcept { return auxData; }

  const AuxMemoryManagerData<S> &getAuxData() const noexcept { return auxData; }

  void setFreeList(FreeList list) { freeList = list; }

  /// Get the list of contexts
  ContextList<S> &getContexts() { return contexts; }

  /// Get the number of contexts currently attached to the MM
  unsigned getContextCount() {
    std::scoped_lock lock(contextMutex);
    return contextCount;
  }

  /// Get the number of threads which have access to the MM.  All threads must
  /// yield access to the MM before a GC can happen.
  unsigned getContextAccessCount() { return mutatorMutex.count(); }

  /// Get the current size of the heap in bytes.
  std::size_t getHeapSize() noexcept { return regionManager.getHeapSize(); }

  /// Release exclusive access.  All mutator threads will unpause.
  void releaseExclusive();

  /// Acquire exclusive access. All mutator threads will pause.
  void acquireExclusive();

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

  AuxMemoryManagerData<S> auxData;

  MemoryManagerConfig config;
  RegionManager regionManager;
  GlobalCollector<S> globalCollector;
  std::unique_ptr<RootWalker<S>> rootWalker;

  // Context List
  std::mutex contextMutex;
  std::size_t contextCount = 0;
  ContextList<S> contexts;

  // Mutator
  MutatorMutex mutatorMutex;

  std::atomic<bool> inMarkingPhase = false;

  /// Guards access to the global free list
  std::mutex freeListMutex;
  FreeList freeList;
};

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

template <typename S>
class AuxContextData {};

template <typename S>
class Context final {
public:
  friend MemoryManager<S>;

  explicit Context(MemoryManager<S> &memoryManager)
      : memoryManager(&memoryManager),
        gcContext(&memoryManager.getGlobalCollector()) {
    memoryManager.attach(*this);
  }

  Context(const Context &) = delete;

  Context(const Context &&) = delete;

  ~Context() { memoryManager->detach(*this); }

  ContextListNode<S> &getListNode() noexcept { return listNode; }

  const ContextListNode<S> &getListNode() const noexcept { return listNode; }

  MemoryManager<S> *getCollector() { return memoryManager; }

  AllocationBuffer &buffer() { return ab; }

  GlobalCollectorContext<S> &getCollectorContext() noexcept {
    return gcContext;
  }

  AuxContextData<S> &getAuxData() noexcept { return auxData; }

  const AuxContextData<S> &getAuxData() const noexcept { return auxData; }

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
  AuxContextData<S> auxData;
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
    : auxData(std::move(builder.auxData)), config(builder.config),
      globalCollector(this), rootWalker(std::move(builder.rootWalker)) {
  // round up the initial heap size to the number of regions.
  auto regionCount = ab::ceilingDivide(config.initialMemory, REGION_SIZE);
  for (std::size_t i = 0; i < regionCount; i++) {
    regionManager.allocateEmptyRegion();
  }
}

template <typename S>
MemoryManager<S>::~MemoryManager() {}

template <typename S>
void MemoryManager<S>::attach(Context<S> &cx) {
  std::scoped_lock lock(contextMutex);
  mutatorMutex.attach();
  contextCount++;
  contexts.push_front(&cx);
}

template <typename S>
void MemoryManager<S>::detach(Context<S> &cx) {
  std::scoped_lock lock(contextMutex);
  mutatorMutex.detach();
  contextCount--;
  contexts.remove(&cx);
}

template <typename S>
void MemoryManager<S>::flushBuffer(Context<S> &context) {
  auto &buffer = context.buffer();
  if (buffer.begin != nullptr) {
    auto *region = Region::get(buffer.end - 1);
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
  return mutatorMutex.requested();
}

template <typename S>
void MemoryManager<S>::acquireExclusive() {
  mutatorMutex.lock();
}

template <typename S>
void MemoryManager<S>::releaseExclusive() {
  mutatorMutex.unlock();
}

template <typename S>
bool MemoryManager<S>::yieldForGC(Context<S> &context) {
  return mutatorMutex.yield();
}

template <typename S>
void MemoryManager<S>::collect(Context<S> &context) {
  mutatorMutex.detach();
  mutatorMutex.lock();
  globalCollector.collect(context.getCollectorContext());
  mutatorMutex.unlock();
  mutatorMutex.attach();
}

template <typename S>
void MemoryManager<S>::kickoff(Context<S> &context) {
  mutatorMutex.detach();
  mutatorMutex.lock();
  globalCollector.kickoff(context.getCollectorContext());
  mutatorMutex.unlock();
  mutatorMutex.attach();
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

} // namespace om::gc

#endif
