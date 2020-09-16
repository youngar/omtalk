#ifndef OMTALK_ARENA_H_
#define OMTALK_ARENA_H_

#include <cstddef>
#include <cstdint>

namespace omtalk::gc {

/// The GC divides the entire address space of the process into large, fixed
/// sized chunks called arenas. Arenas are aligned to their size (aka naturally
/// aligned). We can calculate the arena address for any arbitrary address in
/// the process, by rounding the address down to the nearest arena start
/// address. Regions are allocated out of arenas.

constexpr std::size_t MAX_ADDR = 2ul << 48;
constexpr std::size_t ARENA_SIZE_LOG2 = 26; // 64 mebibytes.
constexpr std::size_t ARENA_SIZE = 1 << ARENA_SIZE_LOG2;
constexpr std::size_t ARENA_COUNT = MAX_ADDR / ARENA_SIZE;
constexpr std::uintptr_t ARENA_MASK = ~std::uintptr_t(ARENA_SIZE);

/// Get the base arena address for an arbitrary pointer, by rounding that
/// pointer down to arena-alignment.
inline void *arenaContaining(void *ptr) noexcept {
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
