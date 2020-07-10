#ifndef OMTALK_GC_HEAP_
#define OMTALK_GC_HEAP_

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <omtalk/Ref.h>
#include <omtalk/Util/Assert.h>
#include <omtalk/Util/BitArray.h>
#include <omtalk/Util/IntrusiveList.h>
#include <type_traits>
#include <vector>
#include <stdlib.h>

namespace omtalk::gc {

//===----------------------------------------------------------------------===//
// Heap Constants
//===----------------------------------------------------------------------===//

constexpr std::size_t MIN_OBJECT_SIZE = 16;
constexpr std::size_t OBJECT_ALIGNMENT = 8;

// region size is 512kib
constexpr std::size_t REGION_SIZE_LOG2 = 19;
constexpr std::size_t REGION_SIZE = std::size_t(1) << REGION_SIZE_LOG2;
constexpr std::size_t REGION_ALIGNMENT = REGION_SIZE;
constexpr std::size_t REGION_NSLOTS = REGION_SIZE / OBJECT_ALIGNMENT;

constexpr std::uintptr_t REGION_INDEX_MASK = REGION_ALIGNMENT - 1;
constexpr std::uintptr_t REGION_ADDRESS_MASK = ~REGION_INDEX_MASK;

// region map size is 4kib (nbits = 32768, 1/128 or 0.78% of the heap)
constexpr std::size_t REGION_MAP_NBITS = REGION_NSLOTS;
constexpr std::size_t REGION_MAP_NCHUNKS = REGION_MAP_NBITS / BITCHUNK_NBITS;

//===----------------------------------------------------------------------===//
// FreeList
//===----------------------------------------------------------------------===//

class alignas(OBJECT_ALIGNMENT) FreeBlock {
public:
  FreeBlock() = delete;

  FreeBlock(std::size_t size, FreeBlock *next) : size(size), next(next) {}

  std::size_t getSize() const noexcept { return size; }

  FreeBlock *getNext() const noexcept { return next; }

  std::byte *begin() noexcept { return reinterpret_cast<std::byte *>(this); }

  std::byte *end() noexcept { return begin() + getSize(); }

  void link(FreeBlock *freeBlock) noexcept {
    freeBlock->next = next;
    next = freeBlock;
  }

private:
  std::size_t size = 0;
  FreeBlock *next = nullptr;
};

static_assert(sizeof(FreeBlock) == MIN_OBJECT_SIZE);

class FreeList {
public:
  void addFreeBlockNoCheck(FreeBlock *freeBlock) noexcept {
    freeList->link(freeBlock);
  }

  void addFreeBlock(FreeBlock *freeBlock) noexcept {
    if (freeList) {
      addFreeBlockNoCheck(freeBlock);
    } else {
      freeList = freeBlock;
    }
  }

  FreeBlock *firstFit(std::size_t size) {
    FreeBlock **block = &freeList;
    while (*block != nullptr) {
      if (size <= (*block)->getSize()) {
        // remove block from the free list and return it
        FreeBlock *firstFit = *block;
        *block = (*block)->getNext();
        return firstFit;
      }
    }

    return nullptr;
  }

private:
  FreeBlock *freeList = nullptr;
};

//===----------------------------------------------------------------------===//
// HeapIndex
//===----------------------------------------------------------------------===//

/// An index into the data portion of a region.
///
enum class HeapIndex : std::uintptr_t {};

constexpr std::byte *toAddr(HeapIndex index, std::byte *base) {
  return base + (std::uintptr_t(index) * OBJECT_ALIGNMENT);
}

constexpr HeapIndex toHeapIndex(std::uintptr_t addr) {
  return HeapIndex((addr & REGION_INDEX_MASK) / OBJECT_ALIGNMENT);
}

//===----------------------------------------------------------------------===//
// RegionMap
//===----------------------------------------------------------------------===//

class RegionMap {
public:
  friend struct RegionMapChecks;

  RegionMap() { clear(); }

  void clear() noexcept { data.clear(); }

  bool mark(HeapIndex index) { return data.set(std::size_t(index)); }

  bool unmark(HeapIndex index) { return data.unset(std::size_t(index)); }

  bool marked(HeapIndex index) const { return data.get(std::size_t(index)); }

  bool unmarked(HeapIndex index) const { return !data.get(std::size_t(index)); }

private:
  BitArray<REGION_MAP_NBITS> data;
};

struct RegionMapChecks {
  static_assert(std::is_trivially_destructible_v<RegionMap>);
  // static_assert(sizeof(RegionMap) == (REGION_MAP_NCHUNKS *
  // sizeof(BitChunk)));
  static_assert(
      check_size<RegionMap, (REGION_MAP_NCHUNKS * sizeof(BitChunk))>());
};

//===----------------------------------------------------------------------===//
// Region
//===----------------------------------------------------------------------===//

class Region;
using RegionList = IntrusiveList<Region>;
using RegionListNode = RegionList::Node;

enum class RegionMode : std::uintptr_t {};

class alignas(REGION_ALIGNMENT) Region {
public:
  friend class RegionChecks;

  static Region *allocate() {
    auto ptr = aligned_alloc(REGION_ALIGNMENT, REGION_SIZE);
    std::cout << "ptr " << ptr << std::endl;
    std::cout << "region_size " << REGION_SIZE << std::endl;
    std::cout << "sizeof(Region) " << sizeof(Region) << std::endl;
    std::cout << "sizeof(RegionMap) " << sizeof(RegionMap) << std::endl;
    std::cout << "offsetof(Region, data) " << offsetof(Region, data)
              << std::endl;

    if (ptr == nullptr) {
      return nullptr;
    }
    return new (ptr) Region();
  }

  template <typename T>
  static Region *get(Ref<T> ref) {
    return reinterpret_cast<Region *>(ref.toAddr() & REGION_ADDRESS_MASK);
  }

  void kill() noexcept {
    this->~Region();
    std::free(this);
  }

  /// Remove this region from the RegionList
  void unlink() { getListNode().clear(); }

  std::byte *heapBegin() noexcept {
    return reinterpret_cast<std::byte *>(data);
  }

  const std::byte *heapBegin() const noexcept {
    return reinterpret_cast<const std::byte *>(data);
  }

  std::byte *heapEnd() noexcept {
    return reinterpret_cast<std::byte *>(this) + REGION_SIZE;
  }

  const std::byte *heapEnd() const noexcept {
    return reinterpret_cast<const std::byte *>(this) + REGION_SIZE;
  }

  void clearMarkMap() noexcept { markMap.clear(); }

  bool inRange(Ref<> ref) const {
    return (heapBegin() <= ref.get()) && (ref.get() < heapEnd());
  }

  bool mark(Ref<> ref) {
    assert(inRange(ref));
    return markMap.mark(toIndex(ref));
  }

  HeapIndex toIndex(Ref<> ref) {
    return HeapIndex((ref.toAddr() & REGION_INDEX_MASK) / OBJECT_ALIGNMENT);
  }

  template <typename T = void>
  constexpr Ref<T> toRef(HeapIndex index) {
    return Ref<T>::fromAddr((std::uintptr_t(index) * OBJECT_ALIGNMENT) +
                            std::uintptr_t(this));
  }

  /// Intrusive list

  RegionListNode &getListNode() noexcept { return listNode; }

  const RegionListNode &getListNode() const noexcept { return listNode; }

private:
  Region() {}

  ~Region() { unlink(); }

  std::size_t flags;
  RegionListNode listNode;

  // Order is important
  RegionMap markMap;

  // trailing data must be last
  alignas(OBJECT_ALIGNMENT) std::byte data[0];
};

class RegionChecks {
  static_assert((sizeof(Region) % OBJECT_ALIGNMENT) == 0);
  static_assert((offsetof(Region, data) % OBJECT_ALIGNMENT) == 0);
  static_assert(check_size<Region, REGION_SIZE>());
  static_assert(sizeof(Region) <= REGION_SIZE);
};

//===----------------------------------------------------------------------===//
// RegionManager
//===----------------------------------------------------------------------===//

using RegionTable = std::vector<Region *>;

class RegionManager {
public:
  ~RegionManager() {
    auto i = regions.begin();
    auto e = regions.end();
    while (i != e) {
      auto region = i++;
      region->kill();
    }
  }

  Region *allocateRegion() {
    Region *region = Region::allocate();
    if (region == nullptr) {
      return nullptr;
    }

    regions.insert(region);
    return region;
  }

  void freeRegion(Region *region) { std::free(region); }

  void clearMarkMaps() noexcept {
    for (auto &region : regions)
      region.clearMarkMap();
  }

private:
  RegionList regions;
};

} // namespace omtalk::gc

#endif // OMTALK_GC_HEAP_
