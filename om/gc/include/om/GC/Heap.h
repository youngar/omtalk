#ifndef OM_GC_GC_HEAP_H
#define OM_GC_GC_HEAP_H

#include <ab/Util/Assert.h>
#include <ab/Util/BitArray.h>
#include <ab/Util/IntrusiveList.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <om/GC/ForwardingMap.h>
#include <om/GC/Ref.h>
#include <om/GC/Scheme.h>
#include <stdlib.h>
#include <type_traits>
#include <vector>

namespace om::gc {

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
constexpr std::size_t REGION_MAP_NCHUNKS =
    REGION_MAP_NBITS / ab::BITCHUNK_NBITS;

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
    AB_ASSERT(size > sizeof(FreeBlock));
  }

  std::size_t size = 0;
  FreeBlock *next = nullptr;
};

static_assert(sizeof(FreeBlock) == MIN_OBJECT_SIZE);

class FreeList {
public:
  void add(void *address, std::size_t size) noexcept {
    AB_ASSERT(size > sizeof(FreeBlock));
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

  /// Get the number of marked bits in the bitmap
  unsigned count() const { return data.count(); }

  using Iterator = ab::BitArray<REGION_MAP_NBITS>::Iterator;

  using ConstIterator = ab::BitArray<REGION_MAP_NBITS>::ConstIterator;

  Iterator begin() { return data.begin(); }

  ConstIterator cbegin() { return data.cbegin(); }

  Iterator end() { return data.end(); }

  ConstIterator cend() { return data.cend(); }

private:
  ab::BitArray<REGION_MAP_NBITS> data;
};

//===----------------------------------------------------------------------===//
// Region
//===----------------------------------------------------------------------===//

class Region;

using RegionList = ab::IntrusiveList<Region>;
using RegionListNode = RegionList::Node;

enum class RegionMode : std::uintptr_t {};

class alignas(REGION_ALIGNMENT) Region {
public:
  friend class RegionChecks;

  static Region *allocate() {
    auto ptr = aligned_alloc(REGION_ALIGNMENT, REGION_SIZE);
    if (ptr == nullptr) {
      return nullptr;
    }
    return new (ptr) Region();
  }

  template <typename T>
  static Region *get(Ref<T> ref) {
    return reinterpret_cast<Region *>(ref.toAddr() & REGION_ADDRESS_MASK);
  }

  static Region *get(void *ref) { return get(Ref(ref)); }

  void kill() noexcept {
    this->~Region();
    std::free(this);
  }

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

  /// Get a pointer to the unsed area of the region.  The region should be used
  /// from low address to high address.
  std::byte *getFree() const noexcept { return free; }

  /// Set the address of the unused area of the region.
  void setFreeSpacePointer(std::byte *freeMark) noexcept { free = freeMark; }

  /// Get the size of the allocatable part of a region
  constexpr std::size_t size() noexcept {
    return REGION_SIZE - offsetof(Region, data);
  }

  void clearMarkMap() noexcept { markMap.clear(); }

  bool inRange(Ref<> ref) const {
    return (heapBegin() <= ref.get()) && (ref.get() < heapEnd());
  }

  bool mark(Ref<void> ref) {
    AB_ASSERT(inRange(ref));
    return markMap.mark(toIndex(ref));
  }

  bool unmark(Ref<void> ref) {
    AB_ASSERT(inRange(ref));
    return markMap.unmark(toIndex(ref));
  }

  bool marked(Ref<void> ref) {
    AB_ASSERT(inRange(ref));
    return markMap.marked(toIndex(ref));
  }

  bool unmarked(Ref<void> ref) {
    AB_ASSERT(inRange(ref));
    return markMap.unmarked(toIndex(ref));
  }

  /// Return the number of live objects in this region.  This count is only
  /// accurate at the end of the marking phase.
  unsigned getLiveObjectCount() const noexcept { return liveObjectCount; }

  void clearLiveObjectCount() noexcept { liveObjectCount = 0; }

  void addLiveObjectCount(std::size_t objectCount) noexcept {
    liveObjectCount += objectCount;
  }

  /// Get the amount of live data in this region.  This count is only accurate
  /// at the end of the marking phase.
  std::size_t getLiveDataSize() const noexcept { return liveDataSize; }

  void clearLiveDataSize() noexcept { liveDataSize = 0; }

  void addLiveDataSize(std::size_t dataSize) noexcept {
    liveDataSize += dataSize;
  }

  /// Clear all statistics held in this region.  This is typically called by the
  /// gc before it recalculates statistics during a garbage collection.
  void clearStatistics() noexcept {
    clearLiveDataSize();
    clearLiveObjectCount();
  }

  /// Records if this region is in the set of regions to be evacuated during a
  /// garbage collection.  This value is only valid during a garbage collection.
  bool isEvacuating() { return evacuating; }

  void setEvacuating(bool value = true) { evacuating = value; }

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

  ForwardingMap &getForwardingMap() noexcept { return forwardingMap; }

  /// Intrusive list
  /// @{
  RegionListNode &getListNode() noexcept { return listNode; }
  const RegionListNode &getListNode() const noexcept { return listNode; }
  /// @}

private:
  Region() {}

  ~Region() {}

  /// The size of live data in the region.  Used and updated by the garbage
  /// collector.
  std::atomic<std::size_t> liveDataSize = 0;

  /// The number of live object in this region.  Used and updated by the garbage
  /// collector.
  std::atomic<std::size_t> liveObjectCount = 0;

  /// True if this region is marked for evacuation.
  std::atomic<bool> evacuating = false;

  /// Pointer to the unused area in a region.
  std::atomic<std::byte *> free;

  ForwardingMap forwardingMap;

  RegionListNode listNode;

  // Order is important
  RegionMap markMap;

  // trailing data must be last
  alignas(OBJECT_ALIGNMENT) std::byte data[];
}; // namespace om::gc

/// Iterate the live objects in a Region by walking the MarkMap.  Requires
/// the region to have a valid MarkMap.
///
/// Can be used with a range based for loop with:
///   for (auto object : RegionMarkedObjects(region))
template <typename S>
class RegionMarkedObjectsIterator {
public:
  RegionMarkedObjectsIterator(Region &region, std::byte *address) noexcept
      : region(region), address(address) {}

  RegionMarkedObjectsIterator(Region &region) noexcept
      : RegionMarkedObjectsIterator(region, region.heapBegin()) {}

  ObjectProxy<S> operator*() const noexcept { return ObjectProxy<S>(address); }

  RegionMarkedObjectsIterator &operator++() noexcept {
    AB_ASSERT(address < region.heapEnd());
    // TODO
    // When we are forwarding, we are destroying the object by installing a
    // forwarding header as we walk the markmap.  This means we cannot read the
    // size of an object at this point, as it will be an invalid object.
    //
    // On the other hand, if the thing using this iterator is touching the
    // object, grabbing the object size may increase the speed of the iterator.
    // This works well with sweeping since sweep needs the object size i.e.
    // sweeping is from the end of a live object to the start of the next.
    //
    // address += ObjectProxy<S>(address).getSize();
    address += OBJECT_ALIGNMENT;
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
  RegionMarkedObjects(Region &region, std::byte *begin, std::byte *end)
      : region(region), begin_(begin), end_(end) {}

  RegionMarkedObjects(Region &region)
      : region(region), begin_(region.heapBegin()), end_(region.heapEnd()) {}

  auto begin() const noexcept {
    return RegionMarkedObjectsIterator<S>(region, begin_);
  }

  auto end() const noexcept {
    return RegionMarkedObjectsIterator<S>(region, end_);
  }

private:
  Region &region;
  std::byte *begin_;
  std::byte *end_;
};

/// Iterate the live objects in a Region by assuming that the next Object begins
/// immediately after the current object.  Using this iterator requires that
/// objects in the region are fully compacted.
///
/// It is possible to continuously increase the end address mark in the
/// iterator as it is used.  This is useful for iterating objects in a region as
/// they are copied in via a cheney style scavenging algorithm.
template <typename S>
class ContiguousObjectsIterator {
public:
  ContiguousObjectsIterator(std::byte *begin) noexcept : scan(begin) {}

  ObjectProxy<S> operator*() const noexcept { return ObjectProxy<S>(scan); }

  ContiguousObjectsIterator &operator++() noexcept {
    scan += ObjectProxy<S>(scan).getSize();
    return *this;
  }

  bool operator!=(const ContiguousObjectsIterator &other) const noexcept {
    return scan != other.scan;
  }

  std::byte *scan;
};

/// Region adapter for iterating contiguous objects
template <typename S>
class RegionContiguousObjects {
public:
  RegionContiguousObjects(Region &region)
      : RegionContiguousObjects(region, region.heapBegin(), region.getFree()) {}

  RegionContiguousObjects(Region &region, std::byte *begin)
      : RegionContiguousObjects(region, begin, region.getFree()) {}

  RegionContiguousObjects(Region &region, std::byte *begin, std::byte *end)
      : begin_(begin), end_(end) {}

  ContiguousObjectsIterator<S> begin() {
    return ContiguousObjectsIterator<S>(begin_);
  }

  ContiguousObjectsIterator<S> end() {
    return ContiguousObjectsIterator<S>(end_);
  }

private:
  std::byte *begin_;
  std::byte *end_;
};

//===----------------------------------------------------------------------===//
// RegionManager
//===----------------------------------------------------------------------===//

/// Manages allocation of regions.  Provides a centralized location to manage
/// region lists.
class RegionManager {
public:
  RegionManager() = default;

  ~RegionManager() {
    auto i = regions.begin();
    auto e = regions.end();
    while (i != e) {
      auto region = i++;
      region->kill();
    }
  }

  /// Get the size of the heap in bytes.
  std::size_t getHeapSize() noexcept { return regionCount * REGION_SIZE; }

  Region *allocateEmptyRegion() {
    auto *region = allocateRegion();
    regions.remove(region);
    emptyRegions.push_front(region);
    return region;
  }

  Region *allocateRegion() {
    Region *region = Region::allocate();
    if (region == nullptr) {
      return nullptr;
    }
    regionCount++;
    region->clearMarkMap();
    region->clearStatistics();
    regions.push_front(region);
    return region;
  }

  void addRegion(Region *region) { regions.push_front(region); }

  Region *getEmptyRegion() {
    auto i = emptyRegions.begin();
    if (i == emptyRegions.end()) {
      return nullptr;
    }
    auto region = &*i;
    emptyRegions.remove(region);
    return region;
  }

  Region *getEmptyOrNewRegion() {
    auto region = getEmptyRegion();
    if (!region) {
      region = allocateRegion();
    }
    return region;
  }

  void freeRegion(Region *region) {
    regionCount--;
    std::free(region);
  }

  /// Get regular regions which are not being allocated out of or evacuated.
  RegionList &getRegions() { return regions; }

  /// Get the list of regions being evacuated.  This list is empty outisde of a
  /// garbage collection.
  RegionList &getEvacuateRegions() { return evacuateRegions; }

  /// Get the list of regions being allocated from.
  RegionList &getAllocateRegions() noexcept { return allocateRegions; }

  /// Get the list of empt regions.  Regions in this list are evually reused for
  /// allocation, or copied to this region during a garbage collection.
  RegionList &getEmptyRegions() { return emptyRegions; }

  void lock() { regionsMutex.lock(); }

  void unlock() { regionsMutex.unlock(); }

  bool try_lock() { return regionsMutex.try_lock(); }

private:
  /// The number of regions currently allocated
  std::size_t regionCount = 0;

  /// Protects all the region lists.
  std::mutex regionsMutex;

  /// List of live regions.
  RegionList regions;

  /// Regions with no objects allocated out of them.
  RegionList emptyRegions;

  /// Regions with active allocation.
  RegionList allocateRegions;

  /// Regions being evacuated from.
  RegionList evacuateRegions;
};

} // namespace om::gc

#endif // OM_GC_GC_HEAP_H
