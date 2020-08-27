#ifndef OMTALK_REGIONALLOCATOR_H_
#define OMTALK_REGIONALLOCATOR_H_

#include <mutex>
#include <omtalk/RegionManager.h>
#include <queue>
#include <vector>

namespace omtalk::gc {

class RegionManager {
public:
  RegionManager() = default;

  void *allocate() noexcept;

  /// Free a region. Resilient against thread races.
  void free(Region *region) noexcept;

  /// True if the address is within the managed heap.
  bool managed(void *addr) const noexcept { return arenaManager.managed(addr); }

private:
  /// commit the backing memory for a region.
  bool commit(void *region) noexcept;

  /// decommit the backing memory for a region.
  void decommit(void *region) noexcept;

  std::mutex mutex;

  /// The arena we are currently allocating regions from.
  void *arena = nullptr;

  /// The current bump allocation pointer in the active region.
  void *alloc = nullptr;

  /// The arena manager allocates arenas, which we sub-allocate arenas from.
  ArenaManager arenaManager;

  /// A cached set of regions that have been freed.
  std::vector<void *> cache;
};

} // namespace omtalk::gc

void *omtalk::RegionManager::allocate() noexcept {
  std::lock_guard guard(mutex);

  if (cache.size() != 0) {
    auto region = cache.pop();
    assert(omtalk::aligned(region, REGION_SIZE));
    cache.pop_back();
    return region;
  }

  if (alloc == nullptr) {
    alloc = arenaManager.allocate();
    if (alloc == nullptr) {
        return nullptr;
    }
  }

  auto region = alloc;
  alloc += 
}

/// Free an entire arena
void omtalk::RegionManager::free(void *arena) noexcept {
  std::lock_guard guard(mutex);

  assert(aligned(arena, ARENA_SIZE));
  auto rc = madvise(arena, ARENA_SIZE, MADV_DONTNEED);
  munmap(arena, ARENA_SIZE);
}

/// Commit memory for use.
static void omtalk::RegionManager::commit(void *region) {
  assert(aligned(region, REGION_SIZE));
  assert(managed(region));
  auto ptr = mprotect(region, REGION_SIZE, PROT_READ, PROT_WRITE);
  madvise(region, REGION_SIZE, MADV_WILLNEED);
}

/// Return region memory back to the OS (memory range is still reserved).
static void omtalk::RegionManager::decommit(void *region) {
  mprotect(region, REGION_SIZE, PROT_NONE);
  madvise(region, REGION_SIZE, MADV_DONTNEED);
}

#endif // OMTALK_REGIONALLOCATOR_H_
