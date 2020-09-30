#ifndef OMTALK_UTIL_ANNOTATIONS_H
#define OMTALK_UTIL_ANNOTATIONS_H

/// OMTALK_ATTRIBUTE(x)
#define OMTALK_ATTRIBUTE(x) __attribute__((x))

/// OMTALK_HAS_ATTRIBUTE(x)
#ifdef __has_attribute
#define OMTALK_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define OMTALK_HAS_ATTRIBUTE(x) 0
#endif

/// OMTALK_EXPECT(x)
#ifdef __builtin__expect__
#define OMTALK_EXPECT(expr, expected) __builtin__expect__(expr, expected)
#else
#define OMTALK_EXPECT(expr, expected) long(expr)
#endif

/// OMTALK_LIKELY(expr)
#define OMTALK_LIKELY(expr) OMTALK_EXPECT(!!(expr), 1)

/// OMTALK_UNLIKELY(expr)
#define OMTALK_UNLIKELY(expr) OMTALK_EXPECT(!!(expr), 0)

/// OMTALK_ALWAYS_INLINE()
#if OMTALK_HAS_ATTRIBUTE(always_inline)
#define OMTALK_ALWAYS_INLINE OMTALK_ATTRIBUTE(always_inline)
#else
#define OMTALK_ALWAYS_INLINE
#endif

/// OMTALK_UNREACHABLE
/// Designate a statement as unreachable. This pseudo-function acts as a hint to
/// the compiler that a point in the program cannot be reached. It is undefined
/// behaviour to execute this function. If you just want to trigger a crash, use
/// OMTALK_ASSERT_UNREACHABLE() instead.
#ifdef __builtin_unreachable__
#define OMTALK_UNREACHABLE() __builtin_unreachable__()
#else
#define OMTALK_UNREACHABLE()
#endif

//===----------------------------------------------------------------------===//
// Thread Safety Annotations
//===----------------------------------------------------------------------===//

// https://clang.llvm.org/docs/ThreadSafetyAnalysis.html

// LIBCPP only adds thread safety annotations to the standard library if
// _LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS is defined.  If this is not defined
// then we will generate a lot of false warning about improper use of mutex when
// we use annotations.  This means we have to disable all thread safety
// annotations if they are disabled in libc++.
#if defined(_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS)
#define OMTALK_THREAD_ANNOTATION(x) OMTALK_ATTRIBUTE(x)
#else
#define OMTALK_THREAD_ANNOTATION(x)
#endif

/// OMTALK_CAPABILITY(x)
#if OMTALK_HAS_ATTRIBUTE(capability)
#define OMTALK_CAPABILITY(x) OMTALK_THREAD_ANNOTATION(capability(x))
#else
#define OMTALK_CAPABILITY(x)
#endif

/// OMTALK_MUTEX_CAPABILITY
/// Mark a class as having the mutex capability for thread safety annotations.
#define OMTALK_MUTEX_CAPABILITY OMTALK_CAPABILITY("mutex")

/// OMTALK_SCOPED_CAPABILITY
#if OMTALK_HAS_ATTRIBUTE(scoped_lockable)
#define OMTALK_SCOPED_CAPABILITY OMTALK_THREAD_ANNOTATION(scoped_lockable)
#else
#define OMTALK_SCOPED_CAPABILITY
#endif

/// OMTALK_GUARDED_BY(x)
#if OMTALK_HAS_ATTRIBUTE(guarded_by)
#define OMTALK_GUARDED_BY(x) OMTALK_THREAD_ANNOTATION(guarded_by(x))
#else
#define OMTALK_GUARDED_BY(x)
#endif

/// OMTALK_PT_GUARDED_BY(x)
#if OMTALK_HAS_ATTRIBUTE(pt_guarded_by)
#define OMTALK_PT_GUARDED_BY(x) OMTALK_THREAD_ANNOTATION(pt_guarded_by(x))
#else
#define OMTALK_PT_GUARDED_BY(x)
#endif

/// OMTALK_ACQUIRED_BEFORE(...)
#if OMTALK_HAS_ATTRIBUTE(acquired_before)
#define OMTALK_ACQUIRED_BEFORE(...)                                            \
  OMTALK_THREAD_ANNOTATION(acquired_before(__VA_ARGS__))
#else
#define OMTALK_ACQUIRED_BEFORE(...)
#endif

/// OMTALK_ACQUIRED_AFTER(...)
#if OMTALK_HAS_ATTRIBUTE(acquired_after)
#define OMTALK_ACQUIRED_AFTER(...)                                             \
  OMTALK_THREAD_ANNOTATION(acquired_after(__VA_ARGS__))
#else
#define OMTALK_ACQUIRED_AFTER(...)
#endif

/// OMTALK_REQUIRES(...)
#if OMTALK_HAS_ATTRIBUTE(requires_capability)
#define OMTALK_REQUIRES(...)                                                   \
  OMTALK_THREAD_ANNOTATION(requires_capability(__VA_ARGS__))
#else
#define OMTALK_REQUIRES(...)
#endif

/// OMTALK_REQUIRES_SHARED(...)
#if OMTALK_HAS_ATTRIBUTE(requires_shared_capability)
#define OMTALK_REQUIRES_SHARED(...)                                            \
  OMTALK_THREAD_ANNOTATION(requires_shared_capability(__VA_ARGS__))
#else
#define OMTALK_REQUIRES_SHARED(...)
#endif

/// OMTALK_ACQUIRE(...)
#if OMTALK_HAS_ATTRIBUTE(acquire_capability)
#define OMTALK_ACQUIRE(...)                                                    \
  OMTALK_THREAD_ANNOTATION(acquire_capability(__VA_ARGS__))
#else
#define OMTALK_ACQUIRE(...)
#endif

/// OMTALK_ACQUIRE_SHARED(...)
#if OMTALK_HAS_ATTRIBUTE(acquire_shared_capability)
#define OMTALK_ACQUIRE_SHARED(...)                                             \
  OMTALK_THREAD_ANNOTATION(acquire_shared_capability(__VA_ARGS__))
#else
#define OMTALK_ACQUIRE_SHARED(...)
#endif

/// OMTALK_RELEASE(...)
#if OMTALK_HAS_ATTRIBUTE(release_capability)
#define OMTALK_RELEASE(...)                                                    \
  OMTALK_THREAD_ANNOTATION(release_capability(__VA_ARGS__))
#else
#error asdfafa
#define OMTALK_RELEASE(...)
#endif

/// OMTALK_RELEASE_SHARED(...)
#if OMTALK_HAS_ATTRIBUTE(release_shared_capability)
#define OMTALK_RELEASE_SHARED(...)                                             \
  OMTALK_THREAD_ANNOTATION(release_shared_capability(__VA_ARGS__))
#else
#define OMTALK_RELEASE_SHARED(...)
#endif

/// OMTALK_RELEASE_GENERIC(...)
#if OMTALK_HAS_ATTRIBUTE(release_generic_capability)
#define OMTALK_RELEASE_GENERIC(...)                                            \
  OMTALK_THREAD_ANNOTATION(release_generic_capability(__VA_ARGS__))
#else
#define OMTALK_RELEASE_GENERIC(...)
#endif

/// OMTALK_TRY_ACQUIRE(...)
#if OMTALK_HAS_ATTRIBUTE(try_acquire_capability)
#define OMTALK_TRY_ACQUIRE(...)                                                \
  OMTALK_THREAD_ANNOTATION(try_acquire_capability(__VA_ARGS__))
#else
#define OMTALK_TRY_ACQUIRE(...)
#endif

/// OMTALK_TRY_ACQUIRE_SHARED(..)
#if OMTALK_HAS_ATTRIBUTE(try_acquire_shared_capability)
#define OMTALK_TRY_ACQUIRE_SHARED(...)                                         \
  OMTALK_THREAD_ANNOTATION(try_acquire_shared_capability(__VA_ARGS__))
#else
#define OMTALK_TRY_ACQUIRE_SHARED(...)
#endif

/// OMTALK_EXCLUDES(...)
#if OMTALK_HAS_ATTRIBUTE(locks_excluded)
#define OMTALK_EXCLUDES(...)                                                   \
  OMTALK_THREAD_ANNOTATION(locks_excluded(__VA_ARGS__))
#else
#define OMTALK_EXCLUDES(...)
#endif

/// OMTALK_ASSERT_CAPABILITY(x)
#if OMTALK_HAS_ATTRIBUTE(assert_capability)
#define OMTALK_ASSERT_CAPABILITY(x)                                            \
  OMTALK_THREAD_ANNOTATION(assert_capability(x))
#else
#define OMTALK_ASSERT_CAPABILITY(x)
#endif

/// OMTALK_ASSERT_SHARED_CAPABILITY(x)
#if OMTALK_HAS_ATTRIBUTE(assert_shared_capability)
#define OMTALK_ASSERT_SHARED_CAPABILITY(x)                                     \
  OMTALK_THREAD_ANNOTATION(assert_shared_capability(x))
#else
#define OMTALK_ASSERT_SHARED_CAPABILITY(x)
#endif

/// OMTALK_RETURN_CAPABILITY(x)
#if OMTALK_HAS_ATTRIBUTE(lock_returned)
#define OMTALK_RETURN_CAPABILITY(x) OMTALK_THREAD_ANNOTATION(lock_returned(x))
#else
#define OMTALK_RETURN_CAPABILITY(x)
#endif

/// OMTALK_NO_THREAD_SAFETY_ANALYSIS
#if OMTALK_HAS_ATTRIBUTE(no_thread_safety_analysis)
#define OMTALK_NO_THREAD_SAFETY_ANALYSIS                                       \
  OMTALK_THREAD_ANNOTATION(no_thread_safety_analysis)
#else
#define OMTALK_NO_THREAD_SAFETY_ANALYSIS
#endif

#endif
