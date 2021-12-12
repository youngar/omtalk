#ifndef FORWARDINGTABLE_H_
#define FORWARDINGTABLE_H_

#include <cstddef>
#include <cstdint>

namespace omtalk {

class FowardingEntry {
public:
  ForwardingEntry(std::uintptr_t from, std::uintptr_t to) noexcept {

  }

  std::uintptr_t data;
};

class FowardingTable {
public:
  FowardingTable(std::size_t capacity) : entries(nullptr), size(0) {
    size = capacity * LOAD_FACTOR;
    entries = new ForwardingEntry[size];
    OMTALK_ASSERT(entries != nullptr);
  }

  ForwardingTable(Region *region) : ForwardingTable(region->liveObjectCount) {}

  ForwardingEntry insert(HeapIndex index)
      : private : static constexpr std::size_t LOAD_FACTOR = 2;

  ForwardingEntry *entries;
  std::size_t size;
};

} // namespace omtalk

#endif // FORWARDINGTABLE_H_
