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
#include <omtalk/Util/IntrusiveList.h>
#include <omtalk/WorkStack.h>
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

struct MemoryManagerConfig {};

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

  explicit MemoryManager(MemoryManagerBuilder<S> &&builder)
      : config(builder.config), globalCollector(this),
        rootWalker(std::move(builder.rootWalker)) {}

  ~MemoryManager();

  RootWalker<S> &getRootWalker() { return *rootWalker; }

  RegionManager &getRegionManager() { return regionManager; }

  void setFreeList(FreeList list) { freeList = list; }

  unsigned getContextAccessCount() { return contextAccessCount; }

  /// Get the mutex used to ensure all contexts yield before performing a
  /// garbage collection.
  std::mutex &getYieldForGcMutex() { return yieldForGcMutex; }

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

  /// Check if another thread is attempting to garbage collect.  Will yield
  /// access to the memory manager so another thread can collect.
  bool yieldForGC(Context<S> &context);

  /// Perform a global garbage collection.  This will wait for all attached
  /// threads to reach GC safe points.
  void collect(Context<S> &context);

private:
  /// Attach a context to the context list. Gives access to the context.
  void attach(Context<S> &context);

  /// Remove a context from the context list. Removes access from the context.
  void detach(Context<S> &context);

  /// Pause the context while waiting for the GC to complete.  If this
  /// context is the last active context, perform the GC.  The yield mutex must
  /// be held when this function is called.
  void waitOrGC(Context<S> &context, std::unique_lock<std::mutex> &yieldLock);

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
  Context<S> *exclusiveContext = nullptr;
  std::atomic<unsigned> contextCount = 0;
  std::atomic<unsigned> contextAccessCount = 0;

  FreeList freeList;
};

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

template <typename S>
class Context final {
public:
  friend MemoryManager<S>;

  Context(MemoryManager<S> &memoryManager) : memoryManager(&memoryManager) {
    memoryManager.attach(*this);
  }

  ~Context() { memoryManager->detach(*this); }

  ContextListNode<S> &getListNode() noexcept { return listNode; }

  const ContextListNode<S> &getListNode() const noexcept { return listNode; }

  MemoryManager<S> *getCollector() { return memoryManager; }

  AllocationBuffer &buffer() { return ab; }

  // GC Notification

  /// If another thread is requesting exclusive access, relinquish access
  /// to the other thread.
  bool yieldForGC();

  /// Perform a global garbage collection.  This will wait for all attached
  /// threads to reach GC safe points.
  void collect();

  /// Refresh the allocation buffer associated with a thread.  May cause tax
  /// paying or garbage collection work to be done.
  bool refreshBuffer(std::size_t minimumSize);

private:
  MemoryManager<S> *memoryManager;
  ContextListNode<S> listNode;
  AllocationBuffer ab;
};

//===----------------------------------------------------------------------===//
// MemoryManager Inlines
//===----------------------------------------------------------------------===//

template <typename S>
MemoryManager<S>::~MemoryManager() {}

template <typename S>
void MemoryManager<S>::attach(Context<S> &cx) {
  std::lock_guard<std::mutex> lock(yieldForGcMutex);
  contextCount++;
  contextAccessCount++;
  contexts.insert(&cx);
}

template <typename S>
void MemoryManager<S>::detach(Context<S> &cx) {
  std::lock_guard<std::mutex> lock(yieldForGcMutex);
  contextCount--;
  contextAccessCount--;
  contexts.remove(&cx);
}

template <typename S>
bool MemoryManager<S>::refreshBuffer(Context<S> &context,
                                     std::size_t minimumSize) {
  // search the free list for an entry at least as big
  FreeBlock *block = freeList.firstFit(minimumSize);
  if (block != nullptr) {
    context.buffer().begin = reinterpret_cast<std::byte *>(block);
    context.buffer().end = block->end();
    return true;
  }

  // Collect and try again
  collect(context);
  block = freeList.firstFit(minimumSize);
  if (block != nullptr) {
    context.buffer().begin = reinterpret_cast<std::byte *>(block);
    context.buffer().end = block->end();
    return true;
  }

  // Get a new region
  Region *region = regionManager.allocateRegion();
  if (region != nullptr) {
    context.buffer().begin = region->heapBegin();
    context.buffer().end = region->heapEnd();
    return true;
  }

  // Failed to allocate
  return false;
}

template <typename S>
bool MemoryManager<S>::exclusiveRequested() {
  return mem::load<SEQ_CST>(&exclusiveContext) != nullptr;
}

template <typename S>
void MemoryManager<S>::releaseExclusive(Context<S> &context) {
  mem::store<SEQ_CST>(&exclusiveContext, static_cast<Context<S> *>(nullptr));
}

template <typename S>
void MemoryManager<S>::waitOrGC(Context<S> &context,
                                std::unique_lock<std::mutex> &yieldLock) {

  contextAccessCount--;

  // If we are not the last thread, wait
  if (contextAccessCount != 0) {
    yieldForGcCv.wait(yieldLock);
    contextAccessCount++;
  } else {
    globalCollector.collect();

    // Must remove exclusive request before waking up other threads
    releaseExclusive(context);

    contextAccessCount++;

    // Wake up other threads
    yieldLock.unlock();
    yieldForGcCv.notify_all();
  }
}

template <typename S>
bool MemoryManager<S>::yieldForGC(Context<S> &context) {
  if (exclusiveRequested()) {
    std::unique_lock yieldLock(yieldForGcMutex);
    waitOrGC(context, yieldLock);
    return true;
  }
  return false;
}

template <typename S>
void MemoryManager<S>::collect(Context<S> &context) {
  std::unique_lock yieldLock(yieldForGcMutex);
  // If no other thread has requested exclusive, take it
  mem::compareExchange<SEQ_CST, SEQ_CST>(
      &exclusiveContext, static_cast<Context<S> *>(nullptr), &context);
  waitOrGC(context, yieldLock);
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

} // namespace omtalk::gc

#endif
