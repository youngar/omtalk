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

class Marking {
public:
  void mark(Ref<void> ref) {
    auto region = Region::get(ref);
    if (region->mark(ref)) {
      stack_.push(ref);
    }
  }

  void scanWorkUnits() {
    while (stack_.more()) {
      scan(stack_.pop().get());
    }
  }

  void scan(Ref<> ref) {
    // TODO: something, anything?
    assert(0);
  }

private:
  WorkStack stack_;
};

struct CollectorConfig {};

constexpr CollectorConfig DEFAULT_COLLECTOR_CONFIG;

class CollectorContext;

using CollectorContextList = IntrusiveList<CollectorContext>;
using CollectorContextListNode = CollectorContextList::Node;

class Collector final {
public:
  friend CollectorContext;

  Collector(const CollectorConfig &CollectorConfig = DEFAULT_COLLECTOR_CONFIG);

  ~Collector();

  CollectorContext &createCollectorContext();

private:
  void attach(CollectorContext *cx);

  void detach(CollectorContext *cx);

  RegionManager regionManager;
  CollectorContextList contexts;
};

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

    Ref<T> ref(begin);
    begin += size;
    return ref;
  }

  std::size_t available() const { return end - begin; }

  bool empty() const { return available() == 0; }

private:
  std::byte *begin = nullptr;
  std::byte *end = nullptr;
};

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

private:
  Collector *collector;
  CollectorContextListNode listNode;
  AllocationBuffer ab;
};

inline Collector::Collector(const CollectorConfig &CollectorConfig) {}

inline Collector::~Collector() {}

inline CollectorContext &Collector::createCollectorContext() {
  Region *region = regionManager.allocateRegion();
};

inline void Collector::attach(CollectorContext *cx) { contexts.insert(cx); }

inline void Collector::detach(CollectorContext *cx) { contexts.remove(cx); }

} // namespace omtalk::gc

#endif // OMTALK_GC_GC_H_
