#ifndef OMTALK_GC_SCAVENGE_H_
#define OMTALK_GC_SCAVENGE_H_

#include <omtalk/Copy.h>
#include <omtalk/Forward.h>
#include <omtalk/Scheme.h>

namespace omtalk::gc {

//===----------------------------------------------------------------------===//
// CopyForward
//===----------------------------------------------------------------------===//

/// Default copy forward implemenation.  Assumes that it is safe to memcpy the
/// object and that the object size will not change after copy forward.
template <typename S>
class CopyForward {
public:
  CopyForwardResult operator()(GlobalCollectorContext<S> &context,
                               ObjectProxy<S> from, std::byte *to,
                               std::byte *end) const noexcept {
    std::cout << "!!! forward " << from.asRef().get() << " to " << to
              << std::endl;
    auto forwardedSize = from.getForwardedSize();
    if (forwardedSize > (end - to)) {
      return CopyForwardResult::fail(to);
    }
    memcpy(to, from.asRef().get(), from.getSize());
    from.forward(to);
    return CopyForwardResult::success(from + forwardedSize, to + forwardedSize);
  }
};

template <typename S>
CopyForwardResult copyForward(GlobalCollectorContext<S> &context,
                              ObjectProxy<S> from, std::byte *to,
                              std::byte *end) {
  return CopyForward<S>()(context, from, to, end);
}

template <typename S>
CopyForwardResult copyForward(GlobalCollectorContext<S> &context,
                              Region &fromRegion, std::byte *fromBegin,
                              std::byte *fromEnd, std::byte *toBegin,
                              std::byte toEnd) {
  CopyForwardResult result = CopyForwardResult::fail(fromBegin, toBegin);
  for (const auto object : RegionMarkedObjects<S>(fromBegin, fromEnd)) {
    if (object.isForwarded()) {
      result = copyForward<S>(context, object, toBegin, toEnd);
      toBegin = result.getTo();
      if (!result) {
        break;
      }
    }
  }
  return result;
}

template <typename S>
CopyForwardResult copyForward(GlobalCollectorContext<S> &context, Region &from,
                              Region &to) {
  return copyForward<S>(context, from, from.heapBegin(), from.heapEnd(),
                        to.heapBegin(), to.heapEnd());
}

//===----------------------------------------------------------------------===//
// Evacuate
//===----------------------------------------------------------------------===//

/// Evacuate the object a slot points to and fix up the slot.  This will not
/// detect if the region becomes empty. Does not fix up the evacuated object.
template <typename S, typename SlotProxyT>
CopyForwardResult evacuate(GlobalCollectorContext<S> &context,
                           SlotProxyT slot) {
  auto to = collector.getEvacuateTo();
  auto end = collector.getEvacuateEnd();
  auto result = copyForward(context, slot.load(), to, end);
  if (!result) {
    // if we ran out of space in the current region, grab a new region to copy
    // in to.
    auto collector = context.getCollector();
    auto regionManager = collector.getRegionManager();
    auto evacuateRegion = collector.getEvacuateRegion();
    regionManager.addRegion(region);
    auto newRegion = regionManager.getEmptyOrNewRegion();
    collector.setEvacuateRegion(newRegion);
    result = copyForward(context, slot.load(), newRegion.heapBegin(),
                         newRegion.heapEnd());
  }
  collector.setEvacuateAddress(result.get());
  fixupSlot(slot, to);
  return result;
}

/// Evacuate an entire region into another region.  Will fixup all region
/// slots.
template <typename S>
CopyForwardResult evacuate(GlobalCollectorContext<S> &context, Region &from,
                           Region &to) {
  auto collector = context.getCollector();
  from.setEvacuated();
  auto result = copyForward<S>(context, from, to);
  fixup<S>(context, to, to.heapBegin(), result.get());
  from.setEvacuated(false);
  getRegionManager.addFreeRegion(region);
  return result;
}

} // namespace omtalk::gc

#endif