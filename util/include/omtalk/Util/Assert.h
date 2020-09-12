#ifndef OMTALK_GC_ASSERT_H
#define OMTALK_GC_ASSERT_H

#include <cstdint>
#include <iostream>
#include <omtalk/Util/CppUtil.h>
#include <sstream>
#include <type_traits>

namespace omtalk {

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

} // namespace omtalk

/// Designate a statement as unreachable. This pseudo-function acts as a hint to
/// the compiler that a point in the program cannot be reached. It is undefined
/// behaviour to execute this function. If you just want to trigger a crash, use
/// OMTALK_ASSERT_UNREACHABLE instead.
#define OMTALK_UNREACHABLE() __builtin_unreachable();

/// Assert that x is true.
#define OMTALK_ASSERT(x)                                                       \
  ::omtalk::check((x), OMTALK_LOCATION_STR(), OMTALK_FUNCTION_STR(),           \
                  "Assertion Failed", OMTALK_STRINGIFY(x))

/// Assert that x is true. Report with message on failure.
#define OMTALK_ASSERT_MSG(x, message)                                          \
  ::omtalk::check((x), OMTALK_LOCATION_STR(), OMTALK_FUNCTION_STR(),           \
                  (message), OMTALK_STRINGIFY(x))

/// Unconditional crash. No-return.
#define OMTALK_ASSERT_UNREACHABLE()                                            \
  ::omtalk::fail(OMTALK_LOCATION_STR(), OMTALK_FUNCTION_STR(),                 \
                 "Unreachable statement reached", nullptr)

/// Unconditional crash with message. No-return.
#define OMTALK_ASSERT_UNREACHABLE_MSG(message)                                 \
  ::omtalk::fail(OMTALK_LOCATION_STR(), OMTALK_FUNCTION_STR(), (message),      \
                 nullptr)

/// Unconditional crash. No-return.
#define OMTALK_ASSERT_UNIMPLEMENTED()                                          \
  ::omtalk::fail(OMTALK_LOCATION_STR(), OMTALK_FUNCTION_STR(),                 \
                 "Unimplemented function called", nullptr)

namespace omtalk {

template <std::size_t actual, std::size_t expected>
struct check_equal : std::true_type {
  static_assert(actual == expected);
};

template <typename T, std::size_t expected>
struct check_size : check_equal<sizeof(T), expected> {};

} // namespace omtalk

#endif