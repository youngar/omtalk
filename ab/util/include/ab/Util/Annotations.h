#ifndef AB_UTIL_ANNOTATIONS_H
#define AB_UTIL_ANNOTATIONS_H

/// AB_ATTRIBUTE(x)
#define AB_ATTRIBUTE(x) __attribute__((x))

/// AB_HAS_ATTRIBUTE(x)
#ifdef __has_attribute
#define AB_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define AB_HAS_ATTRIBUTE(x) 0
#endif

/// AB_EXPECT(x)
#ifdef __builtin__expect__
#define AB_EXPECT(expr, expected) __builtin__expect__(expr, expected)
#else
#define AB_EXPECT(expr, expected) long(expr)
#endif

/// AB_LIKELY(expr)
#define AB_LIKELY(expr) AB_EXPECT(!!(expr), 1)

/// AB_UNLIKELY(expr)
#define AB_UNLIKELY(expr) AB_EXPECT(!!(expr), 0)

/// AB_ALWAYS_INLINE()
#if AB_HAS_ATTRIBUTE(always_inline)
#define AB_ALWAYS_INLINE AB_ATTRIBUTE(always_inline)
#else
#define AB_ALWAYS_INLINE
#endif

/// AB_UNREACHABLE
/// Designate a statement as unreachable. This pseudo-function acts as a hint to
/// the compiler that a point in the program cannot be reached. It is undefined
/// behaviour to execute this function. If you just want to trigger a crash, use
/// AB_ASSERT_UNREACHABLE() instead.
#ifdef __builtin_unreachable__
#define AB_UNREACHABLE() __builtin_unreachable__()
#else
#define AB_UNREACHABLE()
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
#define AB_THREAD_ANNOTATION(x) AB_ATTRIBUTE(x)
#else
#define AB_THREAD_ANNOTATION(x)
#endif

/// AB_CAPABILITY(x)
#if AB_HAS_ATTRIBUTE(capability)
#define AB_CAPABILITY(x) AB_THREAD_ANNOTATION(capability(x))
#else
#define AB_CAPABILITY(x)
#endif

/// AB_MUTEX_CAPABILITY
/// Mark a class as having the mutex capability for thread safety annotations.
#define AB_MUTEX_CAPABILITY AB_CAPABILITY("mutex")

/// AB_SCOPED_CAPABILITY
#if AB_HAS_ATTRIBUTE(scoped_lockable)
#define AB_SCOPED_CAPABILITY AB_THREAD_ANNOTATION(scoped_lockable)
#else
#define AB_SCOPED_CAPABILITY
#endif

/// AB_GUARDED_BY(x)
#if AB_HAS_ATTRIBUTE(guarded_by)
#define AB_GUARDED_BY(x) AB_THREAD_ANNOTATION(guarded_by(x))
#else
#define AB_GUARDED_BY(x)
#endif

/// AB_PT_GUARDED_BY(x)
#if AB_HAS_ATTRIBUTE(pt_guarded_by)
#define AB_PT_GUARDED_BY(x) AB_THREAD_ANNOTATION(pt_guarded_by(x))
#else
#define AB_PT_GUARDED_BY(x)
#endif

/// AB_ACQUIRED_BEFORE(...)
#if AB_HAS_ATTRIBUTE(acquired_before)
#define AB_ACQUIRED_BEFORE(...)                                                \
  AB_THREAD_ANNOTATION(acquired_before(__VA_ARGS__))
#else
#define AB_ACQUIRED_BEFORE(...)
#endif

/// AB_ACQUIRED_AFTER(...)
#if AB_HAS_ATTRIBUTE(acquired_after)
#define AB_ACQUIRED_AFTER(...) AB_THREAD_ANNOTATION(acquired_after(__VA_ARGS__))
#else
#define AB_ACQUIRED_AFTER(...)
#endif

/// AB_REQUIRES(...)
#if AB_HAS_ATTRIBUTE(requires_capability)
#define AB_REQUIRES(...) AB_THREAD_ANNOTATION(requires_capability(__VA_ARGS__))
#else
#define AB_REQUIRES(...)
#endif

/// AB_REQUIRES_SHARED(...)
#if AB_HAS_ATTRIBUTE(requires_shared_capability)
#define AB_REQUIRES_SHARED(...)                                                \
  AB_THREAD_ANNOTATION(requires_shared_capability(__VA_ARGS__))
#else
#define AB_REQUIRES_SHARED(...)
#endif

/// AB_ACQUIRE(...)
#if AB_HAS_ATTRIBUTE(acquire_capability)
#define AB_ACQUIRE(...) AB_THREAD_ANNOTATION(acquire_capability(__VA_ARGS__))
#else
#define AB_ACQUIRE(...)
#endif

/// AB_ACQUIRE_SHARED(...)
#if AB_HAS_ATTRIBUTE(acquire_shared_capability)
#define AB_ACQUIRE_SHARED(...)                                                 \
  AB_THREAD_ANNOTATION(acquire_shared_capability(__VA_ARGS__))
#else
#define AB_ACQUIRE_SHARED(...)
#endif

/// AB_RELEASE(...)
#if AB_HAS_ATTRIBUTE(release_capability)
#define AB_RELEASE(...) AB_THREAD_ANNOTATION(release_capability(__VA_ARGS__))
#else
#error asdfafa
#define AB_RELEASE(...)
#endif

/// AB_RELEASE_SHARED(...)
#if AB_HAS_ATTRIBUTE(release_shared_capability)
#define AB_RELEASE_SHARED(...)                                                 \
  AB_THREAD_ANNOTATION(release_shared_capability(__VA_ARGS__))
#else
#define AB_RELEASE_SHARED(...)
#endif

/// AB_RELEASE_GENERIC(...)
#if AB_HAS_ATTRIBUTE(release_generic_capability)
#define AB_RELEASE_GENERIC(...)                                                \
  AB_THREAD_ANNOTATION(release_generic_capability(__VA_ARGS__))
#else
#define AB_RELEASE_GENERIC(...)
#endif

/// AB_TRY_ACQUIRE(...)
#if AB_HAS_ATTRIBUTE(try_acquire_capability)
#define AB_TRY_ACQUIRE(...)                                                    \
  AB_THREAD_ANNOTATION(try_acquire_capability(__VA_ARGS__))
#else
#define AB_TRY_ACQUIRE(...)
#endif

/// AB_TRY_ACQUIRE_SHARED(..)
#if AB_HAS_ATTRIBUTE(try_acquire_shared_capability)
#define AB_TRY_ACQUIRE_SHARED(...)                                             \
  AB_THREAD_ANNOTATION(try_acquire_shared_capability(__VA_ARGS__))
#else
#define AB_TRY_ACQUIRE_SHARED(...)
#endif

/// AB_EXCLUDES(...)
#if AB_HAS_ATTRIBUTE(locks_excluded)
#define AB_EXCLUDES(...) AB_THREAD_ANNOTATION(locks_excluded(__VA_ARGS__))
#else
#define AB_EXCLUDES(...)
#endif

/// AB_ASSERT_CAPABILITY(x)
#if AB_HAS_ATTRIBUTE(assert_capability)
#define AB_ASSERT_CAPABILITY(x) AB_THREAD_ANNOTATION(assert_capability(x))
#else
#define AB_ASSERT_CAPABILITY(x)
#endif

/// AB_ASSERT_SHARED_CAPABILITY(x)
#if AB_HAS_ATTRIBUTE(assert_shared_capability)
#define AB_ASSERT_SHARED_CAPABILITY(x)                                         \
  AB_THREAD_ANNOTATION(assert_shared_capability(x))
#else
#define AB_ASSERT_SHARED_CAPABILITY(x)
#endif

/// AB_RETURN_CAPABILITY(x)
#if AB_HAS_ATTRIBUTE(lock_returned)
#define AB_RETURN_CAPABILITY(x) AB_THREAD_ANNOTATION(lock_returned(x))
#else
#define AB_RETURN_CAPABILITY(x)
#endif

/// AB_NO_THREAD_SAFETY_ANALYSIS
#if AB_HAS_ATTRIBUTE(no_thread_safety_analysis)
#define AB_NO_THREAD_SAFETY_ANALYSIS                                           \
  AB_THREAD_ANNOTATION(no_thread_safety_analysis)
#else
#define AB_NO_THREAD_SAFETY_ANALYSIS
#endif

#endif // AB_UTIL_ANNOTATIONS_H