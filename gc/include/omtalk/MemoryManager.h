#ifndef OMTALK_MEMORYMANAGER_H
#define OMTALK_MEMORYMANAGER_H

#include <cassert>
#include <cstdint>
#include <memory>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>
#include <omtalk/Util/IntrusiveList.h>
#include <omtalk/WorkStack.h>
#include <sys/mman.h>
#include <vector>

#include <omtalk/Scheme.h>

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
      : config(builder.config), rootWalker(std::move(builder.rootWalker)) {}

  ~MemoryManager();

  RootWalker<S> &getRootWalker() { return *rootWalker; }

  bool refreshBuffer(Context<S> &cx, std::size_t minimumSize) {
    // search the free list for an entry at least as big
    FreeBlock *block = freeList.firstFit(minimumSize);
    if (block != nullptr) {
      cx.buffer().begin = block->begin();
      cx.buffer().end = block->end();
      return true;
    }

    // Get a new region
    Region *region = regionManager.allocateRegion();
    if (region != nullptr) {
      cx.buffer().begin = region->heapBegin();
      cx.buffer().end = region->heapEnd();
      return true;
    }

    // Failed to allocate
    return false;
  }

private:
  void attach(Context<S> &cx);

  void detach(Context<S> &cx);

  MemoryManagerConfig config;
  RegionManager regionManager;
  ContextList<S> contexts;
  FreeList freeList;
  std::unique_ptr<RootWalker<S>> rootWalker;
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

private:
  MemoryManager<S> *memoryManager;
  ContextListNode<S> listNode;
  AllocationBuffer ab;
};

//===----------------------------------------------------------------------===//
// MemoryManager Inlines
//===----------------------------------------------------------------------===//

template <typename S>
inline MemoryManager<S>::~MemoryManager() {}

template <typename S>
inline void MemoryManager<S>::attach(Context<S> &cx) {
  contexts.insert(&cx);
}

template <typename S>
inline void MemoryManager<S>::detach(Context<S> &cx) {
  contexts.remove(&cx);
}

} // namespace omtalk::gc

#endif
