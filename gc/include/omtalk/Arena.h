#ifndef OMTALK_ARENA_H_
#define OMTALK_ARENA_H_

#include <cstddef>
#include <cstdint>

namespace omtalk::gc {

/// The GC divides the entire address space into fixed sized, fixed alignment
/// chunks, called arenas. We can calculate the arena address for any arbitrary
/// address in the process.

constexpr std::size_t MAX_ADDR = 0xFFFFFFFF;
constexpr std::size_t ARENA_SIZE_LOG2 = 26; // 64 mebibytes.
constexpr std::size_t ARENA_SIZE = 1 << ARENA_SIZE_LOG2;

/// The total number of arenas in the address space.
constexpr std::size_t ARENA_COUNT = MAX_ADDR / ARENA_SIZE;

constexpr std::uintptr_t ARENA_MASK = ~std::uintptr_t(ARENA_SIZE);

/// Get the arena address for an arbitrary pointer by rounding it down
/// to arena-alignment.
inline void *arenaPtr(void *ptr) noexcept {
  auto p = reinterpret_cast<std::uintptr_t>(ptr);
  auto a = p & ARENA_MASK;
  return reinterpret_cast<void *>(a);
}

/// Get the arena index for an arbitrary pointer.
inline std::uintptr_t arenaIndex(void *ptr) noexcept {
  auto p = reinterpret_cast<std::uintptr_t>(ptr);
  auto a = p & ARENA_MASK;
  return a / ARENA_SIZE;
}

} // namespace omtalk::gc

#endif // OMTALK_ARENA_H_
