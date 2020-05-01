#ifndef OMTALK_GC_HEAP_
#define OMTALK_GC_HEAP_

#include <cassert>
#include <cstdlib>
#include <omtalk/BitArray.h>
#include <omtalk/Ref.h>
#include <type_traits>
#include <vector>

namespace omtalk::gc {

// region size is 512kib
constexpr std::size_t REGION_SIZE_LOG2 = 19;
constexpr std::size_t REGION_SIZE = std::size_t(1) << REGION_SIZE_LOG2;
constexpr std::size_t REGION_ALIGNMENT = REGION_SIZE;

constexpr std::uintptr_t REGION_INDEX_MASK = REGION_ALIGNMENT - 1;
constexpr std::uintptr_t REGION_ADDRESS_MASK = ~REGION_INDEX_MASK;

constexpr std::size_t MIN_OBJECT_SIZE = 16;
constexpr std::size_t OBJECT_ALIGNMENT = 8;

// region map size is 4kib (nbits = 32768, 1/128 or 0.78% of the heap)
constexpr std::size_t REGION_MAP_NBITS = REGION_SIZE / OBJECT_ALIGNMENT;
constexpr std::size_t REGION_SLOTS = (REGION_SIZE - 16) / OBJECT_ALIGNMENT;

class alignas(OBJECT_ALIGNMENT) FreeBlock {
public:
  FreeBlock() = delete;

  FreeBlock(std::size_t size, FreeBlock *next) : size(size), next(next) {}

  std::size_t getSize() const noexcept { return size; }

  FreeBlock *getNext() const noexcept { return next; }

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
    if(freeList) {
      addFreeBlockNoCheck(freeBlock);
    } else {
      freeList = freeBlock;
    }
  }

private:
  FreeBlock *freeList = nullptr;
};

/// An index into the data portion of a region.
///
enum class HeapIndex : std::uintptr_t {};

class RegionMap {
public:
  RegionMap() { clear(); }

  void clear() noexcept { data.clear(); }

  bool mark(HeapIndex index) { return data.set(std::size_t(index)); }

  bool unmark(HeapIndex index) { return data.unset(std::size_t(index)); }

  bool marked(HeapIndex index) const { return data.get(std::size_t(index)); }

  bool unmarked(HeapIndex index) const { return !data.get(std::size_t(index)); }

private:
  BitArray<REGION_MAP_NBITS> data;
};

static_assert(std::is_trivially_destructible_v<RegionMap>);

enum class RegionMode : std::uintptr_t {};

class alignas(REGION_ALIGNMENT) Region {
public:
  friend class RegionChecks;

  static Region *allocate() {
    auto ptr = std::aligned_alloc(REGION_ALIGNMENT, REGION_SIZE);
    if (ptr == nullptr) {
      return nullptr;
    }
    return new (ptr) Region();
  }

  static Region *get(Ref<auto> ref) {
    return reinterpret_cast<Region *>(ref.toAddr() & REGION_ADDRESS_MASK);
  }

  std::byte *heapBegin() noexcept { return &data[0]; }

  const std::byte *heapBegin() const noexcept { return &data[0]; }

  std::byte *heapEnd() noexcept {
    return reinterpret_cast<std::byte *>(this) + REGION_SIZE;
  }

  const std::byte *heapEnd() const noexcept {
    return reinterpret_cast<const std::byte *>(this) + REGION_SIZE;
  }

  void clearMarkMap() noexcept { markMap.clear(); }

  Region *getNextRegion() const noexcept { return nextRegion; }

  void setNextRegion(Region *region) noexcept { nextRegion = region; }

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


private:
  Region() = default;

  std::size_t flags;
  Region *nextRegion;
  RegionMap markMap;
  alignas(OBJECT_ALIGNMENT) std::byte data[0];
};

class RegionChecks {
  static_assert(std::is_trivially_destructible_v<Region>);
  static_assert((sizeof(Region) % OBJECT_ALIGNMENT) == 0);
  static_assert((offsetof(Region, data) % OBJECT_ALIGNMENT) == 0);
};

using RegionTable = std::vector<Region *>;

class RegionManager {
public:
  Region *popRegion(Region **regionList) {
    Region *region = *regionList;
    if (region != nullptr) {
      *regionList = region->getNextRegion();
    }
    return region;
  }

  void pushRegion(Region **regionList, Region *region) {
    Region *nextRegion = *regionList;
    if (nextRegion != nullptr) {
      region->setNextRegion(nextRegion);
      *regionList = region;
    }
  }

  Region *getFreeRegion() { popRegion(&freeRegions); }

  Region *allocateRegion() {
    Region *region = Region::allocate();
    pushRegion(&usedRegions, region);
    return region;
  }

  void freeRegion(Region *region) { std::free(region); }

private:
  Region *freeRegions = nullptr;
  Region *usedRegions = nullptr;
};

} // namespace omtalk::gc

#endif // OMTALK_GC_HEAP_
