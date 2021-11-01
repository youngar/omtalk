#ifndef AB_UTIL_BOX_H
#define AB_UTIL_BOX_H

#include <cstdint>

namespace ab {

constexpr std::uint64_t BOX_MAX = 20;

constexpr std::uint64_t box_int(std::uint64_t value) {
  return (value << 1) | 1;
}

constexpr std::uint64_t unbox_int(std::uint64_t value) { return (value >> 1); }

constexpr std::uint64_t box_ref(void *value) { return (std::uint64_t)value; }

constexpr std::uint64_t box_ref(std::uint64_t value) { return value; }

constexpr void *unbox_ref(std::uint64_t value) { return (void *)value; }

} // namespace ab

#endif // AB_UTIL_BOX_H