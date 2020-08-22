#ifndef OMTALK_GLOBALCOLLECTOR_H
#define OMTALK_GLOBALCOLLECTOR_H

#include <omtalk/Handle.h>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>
#include <stack>

namespace omtalk::gc {

template <typename S>
class MemoryManager;

//===----------------------------------------------------------------------===//
// Work Stack
//===----------------------------------------------------------------------===//

template <typename S>
struct WorkItem {
public:
  WorkItem(ObjectProxy<S> target) : target(target) {}

  ObjectProxy<S> target;
};

template <typename S>
class WorkStack {
public:
  WorkStack() = default;

  void push(WorkItem<S> ref) { data.push(ref); }

  WorkItem<S> pop() {
    auto ref = data.top();
    data.pop();
    return ref;
  }

  WorkItem<S> top() { return data.top(); }

  bool more() const { return !data.empty(); }

private:
  std::stack<WorkItem<S>> data;
};

//===----------------------------------------------------------------------===//
// Global Collector Scheme -- Default
//===----------------------------------------------------------------------===//

/// Interface for the global collector scheme.
class AbstractGlobalCollector {
public:
  virtual void collect() = 0;

  virtual ~AbstractGlobalCollector() = default;
};

template <typename S>
class GlobalCollectorContext;

/// Default implementation of global collection.
template <typename S>
class GlobalCollector : public AbstractGlobalCollector {
public:
  using Context = GlobalCollectorContext<S>;

  GlobalCollector(MemoryManager<S> *memoryManager)
      : memoryManager(memoryManager) {}

  ~GlobalCollector(){};

  virtual void collect() noexcept override;

  WorkStack<S> &getStack() { return stack; }

private:
  void setup(Context &context) noexcept;

  void scanRoots(Context &context) noexcept;

  void completeScanning(Context &context) noexcept;

  void sweep(Context &context) noexcept;

  MemoryManager<S> *memoryManager;
  WorkStack<S> stack;
};

/// Default context of the marking scheme.
template <typename S>
class GlobalCollectorContext {
public:
  explicit GlobalCollectorContext(GlobalCollector<S> *collector)
      : collector(collector) {}

  GlobalCollector<S> &getCollector() { return *collector; }

private:
  GlobalCollector<S> *collector;
};

//===----------------------------------------------------------------------===//
// Mark Functor -- default
//===----------------------------------------------------------------------===//

template <typename S>
struct Mark {
  void operator()(GlobalCollectorContext<S> context,
                  ObjectProxy<S> target) noexcept {
    auto ref = target.asRef();
    auto region = Region::get(ref);
    std::cout << "!!! mark: " << ref;
    if (region->mark(ref)) {
      std::cout << " pushed ";
      context.getCollector().getStack().push(target);
    }
    std::cout << std::endl;
  }
};

template <typename S>
void mark(GlobalCollectorContext<S> &context, ObjectProxy<S> target) noexcept {
  Mark<S>()(context, target);
}

template <typename S>
void mark(GlobalCollectorContext<S> &context, Ref<> target) noexcept {
  Mark<S>()(context, ObjectProxy<S>(target));
}

//===----------------------------------------------------------------------===//
// Scan Functor -- default
//===----------------------------------------------------------------------===//

template <typename S>
class ScanVisitor {
public:
  // Mark any kind of slot proxy
  template <typename SlotProxyT>
  void visit(SlotProxyT slot, GlobalCollectorContext<S> &context) {
    mark<S>(context, proxy::load<RELAXED>(slot));
  }
};

template <typename S>
struct Scan {
  void operator()(GlobalCollectorContext<S> &context,
                  ObjectProxy<S> target) const noexcept {
    ScanVisitor<S> visitor;
    walk<S>(context, target, visitor);
  }
};

template <typename S>
void scan(GlobalCollectorContext<S> &context, ObjectProxy<S> target) noexcept {
  return Scan<S>()(context, target);
}

//===----------------------------------------------------------------------===//
// Object Scavenging
//===----------------------------------------------------------------------===//

template <typename S>
class ScavengeVisitor {
public:
  template <typename SlotProxyT>
  void visit(GlobalCollectorContext<S> &context,
             const SlotProxyT slot) const noexcept {
    auto ref = slot.load();
    scavenge(context, slot);
  }
};

template <typename S>
struct Scavenge {
  void operator()(GlobalCollectorContext<S> &context,
                  ObjectProxy<S> target) const noexcept {
    ScavengeVisitor<S> scavenger;
    target.walk(context, scavenger);
  }
};

template <typename S>
void scavenge(GlobalCollectorContext<S> &context,
              ObjectProxy<S> target) noexcept {
  Scavenge<S>()(context, target);
}

//===----------------------------------------------------------------------===//
// Global Collector Inlines
//===----------------------------------------------------------------------===//

template <typename S>
void GlobalCollector<S>::collect() noexcept {
  GlobalCollectorContext<S> context(this);
  std::cout << "@@@ GC Start\n";
  setup(context);
  scanRoots(context);
  std::cout << "@@@ GC Marking\n";
  completeScanning(context);
  std::cout << "@@@ GC Sweep\n";
  sweep(context);
  std::cout << "@@@ GC End\n";
}

template <typename S>
void GlobalCollector<S>::setup(Context &context) noexcept {
  // assert that the workstack is empty
  memoryManager->getRegionManager().clearMarkMaps();
}

template <typename S>
void GlobalCollector<S>::scanRoots(Context &context) noexcept {
  ScanVisitor<S> visitor;
  memoryManager->getRootWalker().walk(context, visitor);
}

template <typename S>
void GlobalCollector<S>::completeScanning(Context &context) noexcept {
  while (stack.more()) {
    auto item = stack.pop();
    std::cout << "!!! Scan: " << item.target.asRef() << std::endl;
    scan<S>(context, item.target);
  }
}

template <typename S>
void GlobalCollector<S>::sweep(Context &context) noexcept {
  FreeList freeList;

  for (auto &region : memoryManager->getRegionManager()) {
    std::byte *address = region.heapBegin();
    for (const auto object : RegionMarkedObjects<S>(region)) {
      std::byte *objectAddress = reinterpret<std::byte>(object.asRef()).get();
      std::cout << "!!! region: " << region.heapBegin() << std::endl;
      if (address < objectAddress) {
        std::size_t size = objectAddress - address;
        std::cout << "!!! add to freelist: " << address << " size: " << size
                  << std::endl;
        freeList.add(address, size);
      }
      address = objectAddress + object.getSize();
      (void)object;
    }
    if (address != region.heapEnd()) {
      std::size_t size = region.heapEnd() - address;
      std::cout << "!!! add region tail to free list " << address
                << " size: " << size << std::endl;
    }
  }
  memoryManager->setFreeList(freeList);
}

} // namespace omtalk::gc

#endif