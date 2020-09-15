#ifndef OMTALK_UTIL_ANNOTATIONS_H
#define OMTALK_UTIL_ANNOTATIONS_H

// GCC-like compiler
#if defined(__GNUC__) || defined(__clang__)
#define OMTALK_LIKELY(x) __builtin_expect(!!(x), 1)
#define OMTALK_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define OMTALK_ALWAYS_INLINE __attribute__((always_inline))
#define OMTALK_UNREACHABLE() __builtin_unreachable()
#else
// Unknown compiler
#define OMTALK_LIKELY(x) x
#define OMTALK_UNLIKELY(x) x
#define OMTALK_ALWAYS_INLINE __forceinline
#define OMTALK_UNREACHABLE()
#endif

#endif