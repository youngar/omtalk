#ifndef OM_GC_RECORD_H
#define OM_GC_RECORD_H

#include <om/GC/Heap.h>

namespace om::gc {

/// Record object statistics
template <typename S>
struct Record {
  void operator()(GlobalCollectorContext<S> context,
                  ObjectProxy<S> target) noexcept {
    auto ref = target.asRef();
    auto region = Region::get(ref);
    region->addLiveObjectCount(1);
    region->addLiveDataSize(target.getSize());
  }
};

template <typename S>
void record(GlobalCollectorContext<S> &context,
            ObjectProxy<S> target) noexcept {
  Record<S>()(context, target);
}

} // namespace om::gc

#endif