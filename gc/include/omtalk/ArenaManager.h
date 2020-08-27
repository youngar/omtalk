#ifndef OMTALK_ARENAALLOCATOR_H_
#define OMTALK_ARENAALLOCATOR_H_

#include <omtalk/ArenaMap.h>
#include <deque>

namespace omtalk::gc {

/// Arena manager is not thread safe, so access must be externally synchonized.
class ArenaManager {
public:
  ArenaManager() {}

  ArenaManager(const ArenaManager &) = delete;

  ArenaManager(ArenaManager &&) = default;

  /// True if the address is within a GC arena. MT-safe.
  bool managed(void *addr) const noexcept { return map.managed(addr); }

  /// Return a new reserved arena. MT-unsafe.
  void *allocate() noexcept;

  /// Unreserve an arena. MT-unsafe.
  void *free() noexcept;

private:
  /// We cache up to 1/4 of the active heap size.
  /// What this means is, if total heap usage decreases by more than 1/5th from
  /// the peak, we begin to return regions to the OS.
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
  void *reserveFast(void *hint) noexcept;

  /// Release an arena reservation.
  void unreserve(void *arena) noexcept;

  /// A hint for the next arena allocation. We try to allocate regions
  /// contiguously, so this is a pointer off the end of the last allocated
  /// arena, and will be arena-aligned.
  void *hint;

  /// A massive bitmap indicated which addresses in the process are reserved by
  /// the GC. Used to filter out off-heap pointers in the GC.
  ArenaMap map;

  /// A list of inactive reserved arenas.
  std::deque<void *> cache;

  std::size_t arenaCount = 0;
};

} // namespace omtalk::gc

#endif // OMTALK_ARENAALLOCATOR_H_
