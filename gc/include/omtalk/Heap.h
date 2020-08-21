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
#include <stdlib.h>
#include <type_traits>
#include <vector>

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
  FreeBlock dummy;
  FreeBlock *freeList = &dummy;
  FreeBlock *lastBlock = &dummy;
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
  
  typedef BitArray<REGION_MAP_NBITS>::Iterator Iterator;

  typedef BitArray<REGION_MAP_NBITS>::ConstIterator ConstIterator;

  Iterator begin() { return data.begin(); }

  ConstIterator cbegin() { return data.cbegin(); }

  Iterator end() { return data.end(); }

  ConstIterator cend() { return data.cend(); }

private:
  BitArray<REGION_MAP_NBITS> data;
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

  bool unmark(Ref<> ref) {
    assert(inRange(ref));
    return markMap.unmark(toIndex(ref));
  }

  bool marked(Ref<> ref) {
    assert(inRange(ref));
    return markMap.marked(toIndex(ref));
  }

  RegionMap &getMarkMap() { return markMap; }

  const RegionMap &getMarkMap() const { return regionMap; }

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

  // std::size_t flags;
  RegionListNode listNode;

  // Order is important
  RegionMap markMap;

  // trailing data must be last
  alignas(OBJECT_ALIGNMENT) std::byte data[];
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

  typedef RegionList::Iterator Iterator;

  typedef RegionList::ConstIterator ConstIterator;

  Iterator begin() const { return regions.begin(); }

  Iterator end() const { return regions.end(); }

  ConstIterator cbegin() const { return regions.cbegin(); }

  ConstIterator cend() const { return regions.cend(); }

private:
  RegionList regions;
};

} // namespace omtalk::gc

#endif // OMTALK_GC_HEAP_
