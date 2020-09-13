#ifndef OMTALK_MARKING_H
#define OMTALK_MARKING_H

#include <iostream>
#include <mutex>
#include <omtalk/GlobalCollector.h>
#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>
#include <omtalk/Workstack.h>
#include <stack>

namespace omtalk::gc {

template <typename S>
class GlobalCollector;

template <typename S>
class GlobalCollectorContext;

//===----------------------------------------------------------------------===//
// Mark
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
void mark(GlobalCollectorContext<S> &context, Ref<void> target) noexcept {
  Mark<S>()(context, ObjectProxy<S>(target));
}

//===----------------------------------------------------------------------===//
// Scan
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
    auto ref = target.asRef();
    auto region = Region::get(ref);
    region->addLiveObjectCount(1);
    region->addLiveDataSize(target.getSize());
    walk<S>(context, target, visitor);
  }
};

template <typename S>
void scan(GlobalCollectorContext<S> &context, ObjectProxy<S> target) noexcept {
  return Scan<S>()(context, target);
}

} // namespace omtalk::gc

#endif