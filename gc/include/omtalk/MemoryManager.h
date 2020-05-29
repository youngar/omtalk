#ifndef OMTALK_MEMORYMANAGER_H
#define OMTALK_MEMORYMANAGER_H

#include <cassert>
#include <cstdint>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>
#include <omtalk/Util/IntrusiveList.h>
#include <omtalk/WorkStack.h>
#include <sys/mman.h>
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

constexpr MemoryManagerConfig DEFAULT_COLLECTOR_CONFIG;

//===----------------------------------------------------------------------===//
// MemoryManager
//===----------------------------------------------------------------------===//

class Context;
using ContextList = IntrusiveList<Context>;
using ContextListNode = ContextList::Node;

class MemoryManager final {
public:
  friend Context;

  MemoryManager(const MemoryManagerConfig &MemoryManagerConfig =
                    DEFAULT_COLLECTOR_CONFIG);

  ~MemoryManager();

  bool refreshBuffer(Context &cx, std::size_t minimumSize);

private:
  void attach(Context &cx);

  void detach(Context &cx);

  RegionManager regionManager;
  ContextList contexts;
  FreeList freeList;
};

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

class Context final {
public:
  friend MemoryManager;

  Context(MemoryManager &memoryManager) : memoryManager(&memoryManager) {
    memoryManager.attach(*this);
  }

  ~Context() { memoryManager->detach(*this); }

  ContextListNode &getListNode() noexcept { return listNode; }

  const ContextListNode &getListNode() const noexcept { return listNode; }

  MemoryManager *getCollector() { return memoryManager; }
  AllocationBuffer &buffer() { return ab; }

private:
  MemoryManager *memoryManager;
  ContextListNode listNode;
  AllocationBuffer ab;
};

//===----------------------------------------------------------------------===//
// MemoryManager Inlines
//===----------------------------------------------------------------------===//

inline MemoryManager::MemoryManager(
    const MemoryManagerConfig &MemoryManagerConfig) {}

inline MemoryManager::~MemoryManager() {}

inline void MemoryManager::attach(Context &cx) { contexts.insert(&cx); }

inline void MemoryManager::detach(Context &cx) { contexts.remove(&cx); }

} // namespace omtalk::gc

#endif
