#ifndef OMTALK_GC_GC_H_
#define OMTALK_GC_GC_H_

#include <cassert>
#include <cstdint>
#include <omtalk/Heap.h>
#include <omtalk/IntrusiveList.h>
#include <omtalk/Ref.h>
#include <omtalk/WorkStack.h>
#include <sys/mman.h>
#include <vector>

namespace omtalk::gc {

//===----------------------------------------------------------------------===//
// Marking
//===----------------------------------------------------------------------===//

class Marking {
public:
  void mark(Ref<void> ref) {
    auto region = Region::get(ref);
    if (region->mark(ref)) {
      stack_.push(ref);
    }
  }

  void handleWorkUnits() {
    while (stack_.more()) {
      scan(stack_.pop().get());
    }
  }

  void scan(Ref<>) {
    // TODO: something, anything?
    assert(0);
  }

private:
  WorkStack stack_;
};

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

  template <typename T = void>
  Ref<T> allocate(std::size_t size = sizeof(T)) {

    assert(aligned(size, OBJECT_ALIGNMENT));

    if (size > available()) {
      return Ref<T>(nullptr);
    }

    Ref<T> ref(new(begin) T());
    begin += size;
    return ref;
  }

  std::size_t available() const { return end - begin; }

  bool empty() const { return available() == 0; }

  std::byte *begin = nullptr;

  std::byte *end = nullptr;
};

//===----------------------------------------------------------------------===//
// CollectorConfig
//===----------------------------------------------------------------------===//

struct CollectorConfig {};

constexpr CollectorConfig DEFAULT_COLLECTOR_CONFIG;

//===----------------------------------------------------------------------===//
// Collector
//===----------------------------------------------------------------------===//

class CollectorContext;
using CollectorContextList = IntrusiveList<CollectorContext>;
using CollectorContextListNode = CollectorContextList::Node;

class Collector final {
public:
  friend CollectorContext;

  Collector(const CollectorConfig &CollectorConfig = DEFAULT_COLLECTOR_CONFIG);

  ~Collector();

private:
  void attach(CollectorContext *cx);

  void detach(CollectorContext *cx);

  bool refreshBuffer(CollectorContext *cx, std::size_t minimumSize);

  RegionManager regionManager;
  CollectorContextList contexts;
  FreeList freeList;
};

//===----------------------------------------------------------------------===//
// CollectorContext
//===----------------------------------------------------------------------===//

class CollectorContext final {
public:
  friend Collector;

  CollectorContext(Collector &collector) : collector(&collector) {
    collector.attach(this);
  }

  ~CollectorContext() { collector->detach(this); }

  CollectorContextListNode &getListNode() noexcept { return listNode; }

  const CollectorContextListNode &getListNode() const noexcept {
    return listNode;
  }

  AllocationBuffer &buffer() { return ab; }

  template <typename T = void>
  Ref<T> allocate(std::size_t size = sizeof(T));

  bool refreshBuffer(std::size_t minimumSize);

private:
  Collector *collector;
  CollectorContextListNode listNode;
  AllocationBuffer ab;
};

//===----------------------------------------------------------------------===//
// Collector Inlines
//===----------------------------------------------------------------------===//

inline Collector::Collector(const CollectorConfig &CollectorConfig) {}

inline Collector::~Collector() {}

inline void Collector::attach(CollectorContext *cx) { contexts.insert(cx); }

inline void Collector::detach(CollectorContext *cx) { contexts.remove(cx); }

//===----------------------------------------------------------------------===//
// CollectorContext Inlines
//===----------------------------------------------------------------------===//

template <typename T = void>
Ref<T> CollectorContext::allocate(std::size_t size) {

  Ref<T> ref = buffer().allocate<T>(size);
  if (ref != nullptr) {
    return ref;
  }

  refreshBuffer(size);

  return ab.allocate<T>(size);
}

} // namespace omtalk::gc

#endif // OMTALK_GC_GC_H_
