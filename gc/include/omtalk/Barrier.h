#ifndef OMTALK_BARRIER_H
#define OMTALK_BARRIER_H

#include <omtalk/Mark.h>
#include <omtalk/MemoryManager.h>
#include <omtalk/Scheme.h>

namespace omtalk::gc {

/// Called after an object has been allocated and initialized.
template <typename S, typename ObjectProxyT>
void allocateBarrier(Context<S> &context, ObjectProxyT object) {
  // Newly allocated objects must be marked black and scanned by the GC.
  if (context.writeBarrierEnabled()) {
    mark<S>(context.getCollectorContext(), object);
  }
}

template <typename S, typename ObjectProxyT, typename SlotProxyT>
auto load(Context<S> &context, ObjectProxyT &object, SlotProxyT &slot) {
  // if the slot points to an evactuate region, copy the object into the current
  // allocation region.
  auto result = slot.load();
  if (context.loadBarrierEnabled()) {
    auto region = Region::get(result);
    if (region.isEvacuating()) {
      // auto newObject =
      OMTALK_ASSERT_UNREACHABLE();
    }
  }
  return result;
}

template <typename S, typename ObjectProxyT, typename SlotProxyT,
          typename ValueT>
auto store(Context<S> &context, ObjectProxyT object, SlotProxyT &slot,
           ValueT value) {
  // mark the value which is being replaced
  if (context.writeBarrierEnabled()) {
    mark<S>(context.getCollectorContext(), slot.load());
  }
  auto result = slot.store(value);
  return result;
}

} // namespace omtalk::gc

#endif