#ifndef OMTALK_SWEEP_H
#define OMTALK_SWEEP_H

#include <omtalk/Scheme.h>

namespace omtalk::gc {

/// Sweep a region with a valid mark map, and add free chunks to the free list.
template <typename S>
void sweep(GlobalCollectorContext<S> &context, Region &region,
           FreeList &freeList) {
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
  }
  if (address != region.heapEnd()) {
    std::size_t size = region.heapEnd() - address;
    std::cout << "!!! add region tail to free list " << address
              << " size: " << size << std::endl;
  }
}

} // namespace omtalk::gc

#endif