#ifndef AB_UTIL_MATH_H
#define AB_UTIL_MATH_H

namespace ab {

/// Divide a value by a divisor and round up.
template <typename T, typename U>
auto ceilingDivide(T value, U divisor) -> decltype(value + divisor) {
  return (value + divisor - 1) / divisor;
}

} // namespace ab

#endif // AB_UTIL_MATH_H