#ifndef OMTALK_REGIONMANAGER_H_
#define OMTALK_REGIONMANAGER_H_

#include <mutex>
#include <omtalk/ArenaManager.h>
#include <queue>
#include <unordered_set>
#include <vector>

namespace omtalk::gc {

using RegionSet = std::unordered_set<Region *>;
using RegionList = std::vector<Region *>;

/// Tracks free lists 
class ArenaFreeList {
public:
  std::uintptr_t address;
  BitArray32 free;
};

class RegionHeap {
public:
  void *allocate() noexcept {
    auto 
  }

  void free(void *region) noexcept {
    auto arena = arenaContaining(region);
    auto index = (region - arena) / REGION_SIZE;
    auto &map = data[arena];
    map.set(index);
    if (map.all()) {
      arenaManager.free(arena);
      data.remove(arena);
    }
  }

private:
  ArenaManager manager;
  std::map<std::uintptr_t, BitArray32> data;
};

class CachingRegionHeap {
public:

};

class RegionManager {
public:
  RegionManager() = default;

  ~RegionManager() noexcept;

  Region *allocate() noexcept;

  /// Free a region. Resilient against thread races.
  void free(Region *region) noexcept;

  /// True if the address is within the managed heap.
  bool managed(void *addr) const noexcept { return arenaManager.managed(addr); }

  std::vector<Region *> &getRegions() noexcept { return activeRegions; }

  const std::vector<Region *> &getRegions() const noexcept {
    return activeregions;
  }

  void clearMarkMaps() const noexcept {
    for (auto region : activeRegions) {
      region->getMarkMap().clear();
    }
  }

private:
  /// commit the backing memory for a region.
  void commit(void *region) noexcept;

  /// Decommit the backing memory for a region. Return region memory back to the
  /// OS. Memory range is still reserved.
  void decommit(void *region) noexcept;

  std::mutex mutex;

  /// The current bump allocation pointer, for allocating regions out of an
  /// arena.
  char *alloc = nullptr;

  /// The arena manager allocates arenas, from which we sub-allocate regions.
  ArenaManager arenaManager;

  /// The number of active regions managed by this manager.
  std::vector<Region *> activeRegions;

  /// A cached set of regions that have been freed but not decommitted.
  RegionList cachedRegions;

  /// A list of regions that have been freed and decommitted.
  RegionList freeRegions;

  FromSet fromSet;
  toSet toSet;
  
};

} // namespace omtalk::gc

inline omtalk::gc::RegionManager::~RegionManager() noexcept {
  for (auto region : activeRegions) {
    decommit(region);
  }
  for (auto region : cachedRegions) {
    decommit(region);
  }
}

inline omtalk::gc::Region *omtalk::gc::RegionManager::allocate() noexcept {
  std::lock_guard guard(mutex);

  // Try to take a committed region from the cache. Fast path.
  if (cachedRegions.size() != 0) {
    auto region = cachedRegions.back();
    cachedRegions.pop_back();
    activeRegions.push_back(region);
    assert(omtalk::aligned(region, REGION_SIZE));
    assert(managed(region));
    return region;
  }

  // Try to use a reserved but uncommitted region from the free list. Temporary
  // hack.
  if (freeRegions.size() != 0) {
    auto region = freeRegions.back();
    freeRegions.pop_back();
    commit(region);
    activeRegions.push_back(region);
    assert(omtalk::aligned(region, REGION_SIZE));
    assert(managed(region));
    return region;
  }

  // Grow memory.

  // First step, ensure we have an arena to allocate from.
  if (alloc == nullptr) {
    alloc = reinterpret_cast<char *>(arenaManager.allocate());
    assert(omtalk::aligned(alloc, ARENA_SIZE));
  }

  assert(alloc != nullptr);
  assert(omtalk::aligned(alloc, REGION_SIZE));

  // Allocate and commit a new region.
  auto region = reinterpret_cast<Region *>(alloc);
  assert(omtalk::aligned(region, REGION_SIZE));

  alloc += REGION_SIZE;
  if (omtalk::aligned(alloc, ARENA_SIZE)) {
    alloc = nullptr; // Arena filled.
  }

  commit(region);
  activeRegions.push_back(region);
  assert(managed(region));
  return region;
}

inline void omtalk::gc::RegionManager::free(Region *region) noexcept {
  assert(omtalk::aligned(region, REGION_SIZE));
  assert(managed(region));
  std::lock_guard guard(mutex);
  --activeRegions;
  cachedRegions.push_back(region);
}

inline void omtalk::gc::RegionManager::commit(void *region) noexcept {
  assert(aligned(region, REGION_SIZE));
  assert(managed(region));
  auto rc = mprotect(region, REGION_SIZE, PROT_READ | PROT_WRITE);
  (void)rc;
  assert(rc == 0);
  // madvise(region, REGION_SIZE, MADV_WILLNEED);
}

inline void omtalk::gc::RegionManager::decommit(void *region) noexcept {
  auto rc = mprotect(region, REGION_SIZE, PROT_NONE);
  (void)rc;
  assert(rc == 0);
  madvise(region, REGION_SIZE, MADV_DONTNEED);
}

#endif // OMTALK_REGIONMANAGER_H_
