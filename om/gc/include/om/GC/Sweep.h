#ifndef OM_GC_SWEEP_H
#define OM_GC_SWEEP_H

#include <om/GC/GlobalCollector.h>
#include <om/GC/Scheme.h>

namespace om::gc {

template <typename S>
class GlobalCollectorContext;

/// Sweep a region with a valid mark map, and add free chunks to the free list.
template <typename S>
void sweep(GlobalCollectorContext<S> &context, Region &region,
           FreeList &freeList) {
  std::byte *address = region.heapBegin();
  for (const auto object : RegionMarkedObjects<S>(region)) {
    std::byte *objectAddress = reinterpret<std::byte>(object.asRef()).get();
    if (address < objectAddress) {
      std::size_t size = objectAddress - address;
      freeList.add(address, size);
    }
    address = objectAddress + object.getSize();
  }
  if (address != region.heapEnd()) {
    std::size_t size = region.heapEnd() - address;
    freeList.add(address, size);
  }
}

} // namespace om::gc

#endif