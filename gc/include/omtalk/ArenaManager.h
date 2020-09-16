#ifndef OMTALK_ARENAALLOCATOR_H_
#define OMTALK_ARENAALLOCATOR_H_

#include <deque>
#include <omtalk/ArenaMap.h>
#include <set>

namespace omtalk::gc {

/// A collection of arenas.
class ArenaCache {
public:
  ArenaCache() = default;

  ArenaCache(const ArenaCache &) = default;

  ArenaCache(ArenaCache &&) = default;

  void push(void *arena) noexcept {
    arenas.push_back(arena);
    std::push_heap(arenas.begin(), arenas.end(), std::greater<void *>());
  }

  void *pop() noexcept {
    assert(0 < arenas.size());
    auto x = arenas.front();
    arenas.pop_front();
    return x;
  }

  std::size_t size() const noexcept { return arenas.size(); }

  auto begin() noexcept { return arenas.begin(); }

  auto begin() const noexcept { return arenas.begin(); }

  auto end() noexcept { return arenas.end(); }

  auto end() const noexcept { return arenas.end(); }

  bool empty() const noexcept { return arenas.empty(); }

private:
  std::deque<void *> arenas;
};

/// Arena manager is not thread safe, so access must be externally synchonized.
class ArenaManager {
public:
  ArenaManager() = default;

  ArenaManager(const ArenaManager &) = delete;

  ArenaManager(ArenaManager &&) = delete;

  /// On destruction, releases all cached regions. There must be no active
  /// regions.
  ~ArenaManager() noexcept;

  /// True if the address is within a GC arena. MT-safe.
  bool managed(void *addr) const noexcept { return map.managed(addr); }

  /// Return a new reserved arena. MT-unsafe.
  void *allocate() noexcept;

  /// Unreserve an arena. MT-unsafe.
  void free(void *arena) noexcept;

  const std::set<void *> &getArenas() const noexcept { return arenas; }

  const ArenaCache &getCache() const noexcept { return cache; }

private:
  /// We cache up to 1/4 of the active heap size.
  /// What this means is, if total heap usage decreases by more than 1/5th from
  /// the peak, we begin to unreserve pages and return address ranges to the
  /// process.
  static constexpr std::size_t CACHE_RATIO_DIVISOR = 4;

  /// Reserve an arena at an arena-aligned hint. If successful, the
  /// result will be an arena-aligned pointer and arena-sized block of memory.
  /// We try to allocate at the hint address, but if unsuccessful, will allocate
  /// anywhere else. On out-of-memory conditions, will return nullptr on
  /// failure.
  void *reserve() noexcept;

  /// Try to reserve an arena sized chunk at arena alignment.
  /// Ensures the allocation is
  void *reserveSlow() noexcept;

  /// Try to reserve an arena sized chunk at arena alignment.
  /// May fail if the result is unaligned.
  void *reserveFast() noexcept;

  /// Release an arena reservation.
  void unreserve(void *arena) noexcept;

  /// A hint for the next arena allocation. We try to allocate regions
  /// contiguously, so this is a pointer off the end of the last allocated
  /// arena, and will be arena-aligned.
  void *hint = nullptr;

  ArenaMap map;
  ArenaCache cache;
  std::set<void *> arenas;
};

} // namespace omtalk::gc

#endif // OMTALK_ARENAALLOCATOR_H_
