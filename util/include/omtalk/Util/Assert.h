#ifndef OMTALK_GC_ASSERT
#define OMRTALK_GC_ASSERT

#include <cstdint>
#include <type_traits>

namespace omtalk {

template <std::size_t actual, std::size_t expected>
struct check_equal : std::true_type {
  static_assert(actual == expected);
};

// template <typename T, typename U>
// struct check_size : check_equal<sizeof(T), sizeof(T)> {};

template <typename T, std::size_t expected>
struct check_size : check_equal<sizeof(T), expected> {};

} // namespace omtalk

#endif