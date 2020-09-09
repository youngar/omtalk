#ifndef OMTALK_UTIL_BIT_H
#define OMTALK_UTIL_BIT_H

namespace omtalk {

/// Return the number of bits set to 1 in a value.
template <typename T>
constexpr unsigned popcount(T value) noexcept {
  return __builtin_popcount(value);
}

} // namespace omtalk

#endif