#ifndef AB_UTIL_BIT_H
#define AB_UTIL_BIT_H

#include <cstdint>

namespace ab {

/// Prefetch the memory pointed to by x.
inline void prefetch(void *x) noexcept { __builtin_prefetch(x); }

/// Return the number of bits set to 1 in a value.
constexpr int popcount(unsigned int value) noexcept {
  return __builtin_popcount(value);
}

constexpr int popcount(unsigned long value) noexcept {
  return __builtin_popcountl(value);
}

constexpr int popcount(unsigned long long value) noexcept {
  return __builtin_popcountll(value);
}

/// Return the index of the lowest set bit. If value is 0 returns 0.
constexpr int findFirstSet(int value) noexcept { return __builtin_ffs(value); }

constexpr int findFirstSet(long value) noexcept {
  return __builtin_ffsl(value);
}

constexpr int findFirstSet(long long value) noexcept {
  return __builtin_ffsll(value);
}

constexpr int findFirstSet(short value) noexcept {
  return __builtin_ffs(int(value)) - 16;
}

constexpr int findFirstSet(signed char value) noexcept {
  return __builtin_ffs(unsigned(value)) - 24;
}

constexpr int findFirstSet(unsigned value) noexcept {
  return findFirstSet(signed(value));
}

constexpr int findFirstSet(unsigned long value) noexcept {
  return findFirstSet((long)value);
}

constexpr int findFirstSet(unsigned long long value) noexcept {
  return findFirstSet((long long)value);
}

constexpr int findFirstSet(unsigned short value) noexcept {
  return findFirstSet((short)value);
}

constexpr int findFirstSet(unsigned char value) noexcept {
  return findFirstSet((signed char)value);
}

/// Return the number of leading 0 bits.
constexpr int countLeadingZeros(unsigned int value) noexcept {
  return value == 0 ? sizeof(unsigned int) * 8 : __builtin_clz(value);
}

constexpr int countLeadingZeros(unsigned long value) noexcept {
  return value == 0 ? sizeof(unsigned long) * 8 : __builtin_clzl(value);
}

constexpr int countLeadingZeros(unsigned long long value) noexcept {
  return value == 0 ? sizeof(unsigned long long) * 8 : __builtin_clzll(value);
}

// Return the number of trailing 0 bits.
constexpr int countTrailingZeros(unsigned int value) noexcept {
  return value == 0 ? sizeof(unsigned int) * 8 : __builtin_ctz(value);
}

constexpr int countTrailingZeros(unsigned long value) noexcept {
  return value == 0 ? sizeof(unsigned long) * 8 : __builtin_ctzl(value);
}

constexpr int countTrailingZeros(unsigned long long value) noexcept {
  return value == 0 ? sizeof(unsigned long long) * 8 : __builtin_ctzll(value);
}

/// Smear all 1 bits to the right. All bits to the right of the highest 1 bit
/// are set to 1.
constexpr unsigned long smear(unsigned long value) noexcept {
  return value == 0 ? 0 : SIZE_MAX >> countLeadingZeros(value);
#if 0
  // alternative implementation
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  value |= value >> 32;
  return value;
#endif
}

} // namespace ab

#endif // AB_UTIL_BIT_H