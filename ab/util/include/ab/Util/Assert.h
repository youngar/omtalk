#ifndef AB_UTIL_ASSERT_H
#define AB_UTIL_ASSERT_H

#include <ab/Util/CppUtil.h>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <type_traits>

namespace ab {

/// Crash the process. This call should result in a process signal that can be
/// caught by a native debugger.
[[noreturn]] inline void trap() noexcept { __builtin_trap(); }

/// Output an error message to stderr and trap.
/// Message format:
/// ```
///   <file>:<line>: <message>
///       in: <function>
///       note: <note>
/// ```
[[noreturn]] inline void fail(const char *location, const char *function,
                              const char *message, const char *note) {
  std::stringstream str;

  str << location << ": Error: " << message << std::endl;
  str << "\tin: " << function << std::endl;

  if (note != nullptr) {
    str << "\tnote: " << note << std::endl;
  }

  std::cerr << str.str() << std::endl;
  trap();
}

/// Check condition, fail if false.
inline void check(bool value, const char *location, const char *function,
                  const char *message, const char *note) {
  if (!value) {
    fail(location, function, message, note);
  }
}

} // namespace ab

/// Assert that x is true.
#define AB_ASSERT(x)                                                           \
  ::ab::check((x), AB_LOCATION_STR(), AB_FUNCTION_STR(), "Assertion Failed",   \
              AB_STRINGIFY(x))

/// Assert that x is true. Report with message on failure.
#define AB_ASSERT_MSG(x, message)                                              \
  ::ab::check((x), AB_LOCATION_STR(), AB_FUNCTION_STR(), (message),            \
              AB_STRINGIFY(x))

/// Unconditional crash. No-return.
#define AB_ASSERT_UNREACHABLE()                                                \
  ::ab::fail(AB_LOCATION_STR(), AB_FUNCTION_STR(),                             \
             "Unreachable statement reached", nullptr)

/// Unconditional crash with message. No-return.
#define AB_ASSERT_UNREACHABLE_MSG(message)                                     \
  ::ab::fail(AB_LOCATION_STR(), AB_FUNCTION_STR(), (message), nullptr)

/// Unconditional crash. No-return.
#define AB_ASSERT_UNIMPLEMENTED()                                              \
  ::ab::fail(AB_LOCATION_STR(), AB_FUNCTION_STR(),                             \
             "Unimplemented function called", nullptr)

namespace ab {

template <std::size_t actual, std::size_t expected>
struct check_equal : std::true_type {
  static_assert(actual == expected);
};

template <typename T, std::size_t expected>
struct check_size : check_equal<sizeof(T), expected> {};

} // namespace ab

#endif // AB_UTIL_ASSERT_H