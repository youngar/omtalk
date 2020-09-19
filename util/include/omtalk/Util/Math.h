#ifndef OMTALK_UTIL_MATH_H
#define OMTALK_UTIL_MATH_H

namespace omtalk {

/// Divide a value by a divisor and round up.
template <typename T, typename U>
auto ceilingDivide(T value, U divisor) -> decltype(value + divisor) {
  return (value + divisor - 1) / divisor;
}

} // namespace omtalk

#endif