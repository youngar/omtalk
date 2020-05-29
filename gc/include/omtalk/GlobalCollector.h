#ifndef OMTALK_GLOBALCOLLECTOR_H
#define OMTALK_GLOBALCOLLECTOR_H

#include <omtalk/Scheme.h>

namespace omtalk::gc {

//===----------------------------------------------------------------------===//
// Work Stack
//===----------------------------------------------------------------------===//

template <typename S>
class WorkItem {
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
  virtual void collect();

  virtual ~AbstractGlobalCollector() = 0;
};

template <typename S>
class GlobalCollectorContext;

/// Default implementation of global collection.
template <typename S>
class GlobalCollector : public AbstractGlobalCollector {
public:
  using Context = GlobalCollectorContext<S>;

  GlobalCollector(MemoryManager mm) = default;

  virtual void collect() noexcept override;

private:
  void setup(Context &cx) noexcept;

  void scanRoots(Context &cx) noexcept;

  void completeScanning(Context &cx) noexcept;

  MemoryManager *memoryManager;
  WorkStack<S> stack;
};

/// Default context of the marking scheme.
template <typename S>
class GlobalCollectorContext {
public:
  explicit GlobalCollectorContext(GlobalCollector<S> &collector)
      : collector(collector) {}

  GlobalCollector *collector;
};

//===----------------------------------------------------------------------===//
// Mark Functor -- default
//===----------------------------------------------------------------------===//

template <typename S>
struct Mark {
  void operator()(GlobalCollectorContext<S> cx, Ref<void> target) noexcept {
    auto region = Region::get(ref);
    if (region->mark(ref)) {
      cx.marking->stack.push(getObjectProxy(ref));
    }
  }
};

template <typename S>
void mark(GlobalCollectorContext<S> &cx, Ref<void> target) noexcept {
  Mark<S>()(cx, target);
}

//===----------------------------------------------------------------------===//
// Scan Functor -- default
//===----------------------------------------------------------------------===//

template <typename S>
class ScanVisitor {
public:
  template <typename SlotProxyT>
  void visit(GlobalCollectorContext<S> &cx, SlotProxyT slot) {
    mark<S>(cx, Ref<void>(slot.load()));
  }
};

template <typename S>
struct Scan {
  void operator()(GlobalCollectorContext<S> &cx,
                  ObjectProxy<S> target) const noexcept {
    ScanVisitor<S> visitor;
    walk(cx, target, visitor);
  }
};

template <typename S>
void scan(GlobalCollectorContext<S> &cx, ObjectProxy<S> target) noexcept {
  return Scan<S>()(cx, ref);
}

template <typename S>
void scan(GlobalCollectorContext<S> cx, Ref<void> target) noexcept {
  return scan<S>(cx, getProxy(target));
}

//===----------------------------------------------------------------------===//
// Object Scavenging
//===----------------------------------------------------------------------===//

template <typename S>
class ScavengeVisitor {
public:
  template <typename SlotProxyT>
  void visit(Context &cx, const SlotProxyT slot) const noexcept {
    auto ref = slot.load();
    scavenge(cx, slot);
  }
};

template <typename S>
struct Scavenge {
  void operator()(Context &cx, ObjectProxy<S> target) const noexcept {
    ScavengeVisitor scavenger;
    target.walk(cx, scavenger);
  }
};

template <typename S>
void scavenge(Context &cx, ObjectProxy<S> target) noexcept {
  Scavenge<S>()(cx, target);
}

//===----------------------------------------------------------------------===//
// Global Collector Inlines
//===----------------------------------------------------------------------===//

template <typename S>
void GlobalCollector<S>::collect() noexcept {
  GlobCollectorContext<S> cx(*this);
  setup(cx);
  scanRoots(cx);
  completeScanning(cx);
  sweep(cx);
}

template <typename S>
void GlobalCollector<S>::setup(Context &cx) noexcept {
  cx.collector->memoryManager.regionManager.clearMarkMaps();
}

template <typename S>
void GlobalCollector<S>::scanRoots(Context &cx) noexcept {
  ScanVisitor<S> visitor;
  RootWalker<S> rootWalker;
  rootWalker.walk(cx, visitor);
}

template <typename S>
void GlobalCollector<S>::completeScanning(Context &cx) noexcept {
  while (stack.more()) {
    auto item = stack.pop();
    scan<S>(cx, getProxy(item.target));
  }
}

template <typename S>
void GlobalCollector<S>::sweep(Context &cx) noexcept {
  while (stack.more()) {
    auto item = stack.pop();
    scan<S>(cx, item);
  }
}

} // namespace omtalk::gc

#endif