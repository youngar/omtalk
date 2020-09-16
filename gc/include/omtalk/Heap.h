#ifndef OMTALK_GC_HEAP_
#define OMTALK_GC_HEAP_

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>
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
  FreeBlock() = default;

  static FreeBlock *create(void *address, std::size_t size = 0,
                           FreeBlock *next = nullptr) noexcept {
    return new (address) FreeBlock(size, next);
  }

  std::size_t getSize() const noexcept { return size; }

  FreeBlock *getNext() const noexcept { return next; }

  bool hasNext() const noexcept { return next != nullptr; }

  /// Return the beginning of the bytes in the free block, located immediately
  /// after this header.
  std::byte *begin() noexcept {
    return reinterpret_cast<std::byte *>(this + 1);
  }

  /// Return the end of the bytes in the free block
  std::byte *end() noexcept {
    return reinterpret_cast<std::byte *>(this) + getSize();
  }

  /// Create a new FreeBlock at the address and link it to this free block.
  /// Returns the newly created FreeBlock
  FreeBlock *link(void *address, std::size_t size = 0) noexcept {
    return link(FreeBlock::create(address, size));
  }

  FreeBlock *link(FreeBlock *freeBlock) noexcept {
    next = freeBlock;
    return freeBlock;
  }

private:
  FreeBlock(std::size_t size, FreeBlock *next = nullptr)
      : size(size), next(next) {
    assert(size > sizeof(FreeBlock));
  }

  std::size_t size = 0;
  FreeBlock *next = nullptr;
};

class FreeList {
public:
  void add(void *address, std::size_t size) noexcept {
    assert(size > sizeof(FreeBlock));
    last = last->link(address, size);
  }

  /// Return the first block which is at least as large as size.  Removes the
  /// free block from the free list.
  FreeBlock *firstFit(std::size_t size) {
    FreeBlock *freeBlock = &dummy;
    while (freeBlock->hasNext()) {
      auto result = freeBlock->getNext();
      if (result->getSize() >= size) {
        freeBlock->link(result->getNext());
        return result;
      }
    }

    return nullptr;
  }

private:
  FreeBlock dummy;
  FreeBlock *last = &dummy;
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

  using Iterator = BitArray<REGION_MAP_NBITS>::Iterator;

  using ConstIterator = BitArray<REGION_MAP_NBITS>::ConstIterator;

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

enum class RegionMode { NONE, EVACUATING_FROM, EVACUATING_TO };

class Region;

using RegionList = IntrusiveList<Region>;
using RegionListNode = RegionList::Node;

enum class RegionMode : std::uintptr_t {};

class alignas(REGION_ALIGNMENT) Region {
public:
  friend class RegionChecks;
  friend class RegionManager;

  template <typename T>
  static Region *get(Ref<T> ref) {
    return reinterpret_cast<Region *>(ref.toAddr() & REGION_ADDRESS_MASK);
  }

  static Region *containing(std::uintptr_t address) noexcept {
    return reinterpret_cast<Region *>(address & REGION_ADDRESS_MASK);
  }
  template <typename T>
  static Region *containing(T *ptr) {
    return containing(std::uintptr_t(ptr));
  }

  template <typename T>
  static Region *containing(Ref<T> ref) {
    return containing(ref.toAddr());
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

  bool mark(Ref<void> ref) {
    assert(inRange(ref));
    return markMap.mark(toIndex(ref));
  }

  bool unmark(Ref<void> ref) {
    assert(inRange(ref));
    return markMap.unmark(toIndex(ref));
  }

  bool marked(Ref<void> ref) {
    assert(inRange(ref));
    return markMap.marked(toIndex(ref));
  }

  bool unmarked(Ref<void> ref) {
    assert(inRange(ref));
    return markMap.unmarked(toIndex(ref));
  }

  RegionMap &getMarkMap() { return markMap; }

  const RegionMap &getMarkMap() const { return markMap; }

  HeapIndex toIndex(Ref<void> ref) {
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

  Region *next() const noexcept { return listNode.next; }

  Region *prev() const noexcept { return listNode.prev; }

private:
  Region() {}

  ~Region() { unlink(); }

  // std::size_t flags;
  RegionListNode listNode;

  // Order is important
  RegionMap markMap;

  // trailing data must be last
  alignas(OBJECT_ALIGNMENT) std::byte data[];
}; // namespace omtalk::gc

/// Iterate the live objects in a Region.  Requires the region to have a valid
/// MarkMap
template <typename S>
class RegionMarkedObjectsIterator {
public:
  RegionMarkedObjectsIterator(Region &region, std::byte *address) noexcept
      : region(region), address(address) {}

  RegionMarkedObjectsIterator(Region &region) noexcept
      : RegionMarkedObjectsIterator(region, region.heapBegin()) {}

  ObjectProxy<S> operator*() const noexcept { return ObjectProxy<S>(address); }

  RegionMarkedObjectsIterator &operator++() noexcept {
    assert(address < region.heapEnd());
    address += ObjectProxy<S>(address).getSize();
    while ((address < region.heapEnd()) && region.unmarked(address)) {
      address += OBJECT_ALIGNMENT;
    }
    return *this;
  }

  bool operator!=(const RegionMarkedObjectsIterator &other) const noexcept {
    return address != other.address;
  }

  Region &region;
  std::byte *address = nullptr;
};

/// Range Adaptor for Regions: iterates marked objects.
template <typename S>
class RegionMarkedObjects {
public:
  RegionMarkedObjects(Region &region) : region(region) {}

  auto begin() const noexcept { return RegionMarkedObjectsIterator<S>(region); }

  auto end() const noexcept {
    return RegionMarkedObjectsIterator<S>(region, region.heapEnd());
  }

private:
  Region &region;
};

//===----------------------------------------------------------------------===//
// RegionManager
//===----------------------------------------------------------------------===//

#if 0

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
    std::lock_guard regionGuard(regionsMutex);
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

  using Iterator = RegionList::Iterator;

  using ConstIterator = RegionList::ConstIterator;

  Iterator begin() const { return regions.begin(); }

  Iterator end() const { return regions.end(); }

  ConstIterator cbegin() const { return regions.cbegin(); }

  ConstIterator cend() const { return regions.cend(); }

private:
  std::mutex regionsMutex;
  RegionList regions;
};

#endif

//===----------------------------------------------------------------------===//
// Inlines
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RegionSet
//===----------------------------------------------------------------------===//

template <typename T>

template <RegionMode MODE>
class RegionSet {
public:
  using Iterator = RegionList::Iterator;

  using Iterator =

      RegionSet(){};

  /// @group Iteration
  /// @{

  auto begin() noexcept { return elements.begin(); }

  auto begin() const noexcept { return elements.begin(); }

  auto end() noexcept { return elements.end(); }

  auto end() const noexcept { return elements.end(); }

  auto cbegin() const noexcept { return elements.cbegin(); }

  auto cend() const noexcept { return elements.cend(); }

  bool contains(const Region *region) const noexcept {
    return region->mode == MODE;
  }

  void insert(Region *region) noexcept {
    assert(region->getMode() == none);
    assert(region->next() == nullptr);
    assert(region->prev() == nullptr);
    region->setMode(MODE);
    elements.insert(region);
  }

  void remove(Region *region) noexcept {
    assert(region->getMode() == MODE);
    elements.remove(region);
    region->setMode(RegionMode::NONE);
  } 

  /// @}

private:
  RegionList elements;
};

using FromSet = RegionSet<RegionMode::EVACUATING_FROM>;
using ToSet = RegionSet<RegionMode::EVACUATING_TO>;

} // namespace omtalk::gc

#endif // OMTALK_GC_HEAP_
