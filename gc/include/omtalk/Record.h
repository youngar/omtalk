#ifndef OMTALK_RECORD_H
#define OMTALK_RECORD_H

#include <omtalk/Heap.h>

namespace omtalk::gc {

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

} // namespace omtalk::gc

#endif