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

//===----------------------------------------------------------------------===//
// Scan Functor -- default
//===----------------------------------------------------------------------===//

template <typename S>
class ScanVisitor {
public:
  // Mark any kind of slot proxy
  template <typename SlotProxyT>
  void visit(GlobalCollectorContext<S> &context, SlotProxyT slot) {
    mark<S>(context, slot.load());
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
  setup(context);
  scanRoots(context);
  completeScanning(context);
  sweep(context);
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
    std::cout << "completeScanning: " << item.target.asRef() << std::endl;
    scan<S>(context, item.target);
  }
}

template <typename S>
void GlobalCollector<S>::sweep(Context &context) noexcept {}

} // namespace omtalk::gc

#endif