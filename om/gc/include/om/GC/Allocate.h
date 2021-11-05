#ifndef OM_GC_ALLOCATE_H
#define OM_GC_ALLOCATE_H

#include <cstddef>
#include <om/GC/Barrier.h>
#include <om/GC/Ref.h>
#include <type_traits>

namespace om::gc {

template <typename S>
class Context;

//===----------------------------------------------------------------------===//
// BasicInit
//===----------------------------------------------------------------------===//

template <typename T>
class BasicInit {
public:
  void operator()(Ref<T> target) {
    // no-op.
  }
};

//===----------------------------------------------------------------------===//
// Tax
//===----------------------------------------------------------------------===//

class Tax {
public:
  Tax() = default;

  std::size_t amount = 0;

  constexpr operator bool() const { return amount != 0; }
};

template <typename S>
inline void pay(Context<S> &cx, const Tax &tax) {
  // for now, there is no tax paying.
}

//===----------------------------------------------------------------------===//
// AllocationResult
//===----------------------------------------------------------------------===//

class AllocationResult {
public:
  Ref<void> allocation = nullptr;
  Tax tax;
};

/// Every object size must be a multiple of the BASE_OBJECT_ALIGNMENT_FACTOR
constexpr std::size_t BASE_OBJECT_ALIGNMENT_FACTOR = 8;

/// Every object size must be at least this large.
constexpr std::size_t MINIMUM_OBJECT_SIZE = 32;

//===----------------------------------------------------------------------===//
// Internal Byte Allocators
//===----------------------------------------------------------------------===//

/// Fast-path byte allocator. Will NOT collect. Memory is NOT zeroed.
template <typename S>
Ref<void> allocateBytesFast(Context<S> &cx, std::size_t size) noexcept {
  return cx.buffer().tryAllocate(size);
}

/// Fast-path byte allocator. Will NOT collect. Memory is zeroed.
template <typename S>
Ref<void> allocateBytesZeroFast(Context<S> &cx, std::size_t size) noexcept {
  return cx.buffer().tryAllocate(size);
}

/// Slow-path byte allocator. MAY collect. Memory is NOT zeroed.
template <typename S>
AllocationResult allocateBytesSlow(Context<S> &cx, std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  auto allocation = allocateBytesFast(cx, size);
  return {allocation, Tax()};
}

/// Slow-path byte allocator. MAY collect. Memory IS zeroed.
template <typename S>
AllocationResult allocateBytesZeroSlow(Context<S> &cx,
                                       std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  auto allocation = allocateBytesFast(cx, size);
  return {allocation, Tax()};
}

/// Slow-path byte allocator. WILL NOT collect. Memory is NOT zeroed.
template <typename S>
Ref<void> allocateBytesNoCollectSlow(Context<S> &cx,
                                     std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  return allocateBytesFast(cx, size);
}

/// Slow-path byte allocator. Will NOT collect. Memory IS zeroed.
template <typename S>
Ref<void> allocateBytesZeroNoCollectSlow(Context<S> &cx,
                                         std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  return allocateBytesFast(cx, size);
}

//===----------------------------------------------------------------------===//
// Internal Object Allocators
//===----------------------------------------------------------------------===//

/// Fast-path object allocator. Will NOT collect. Memory is NOT zeroed.
template <typename T = void, typename S, typename Init, typename... Args>
Ref<T> allocateFast(Context<S> &cx, std::size_t size, Init &&init,
                    Args &&...args) noexcept {
  auto object = cast<T>(allocateBytesFast(cx, size));

  if (object) {
    init(object, std::forward<Args>(args)...);
    allocateBarrier(cx, object);
  }
  return object;
}

/// Fast-path object allocator. Will NOT collect. Memory IS zeroed.
template <typename T = void, typename S, typename Init, typename... Args>
Ref<T> allocateZeroFast(Context<S> &cx, std::size_t size, Init &&init,
                        Args &&...args) noexcept {
  auto object = cast<T>(allocateBytesZeroFast(cx, size));
  if (object) {
    init(object, std::forward<Args>(args)...);
    allocateBarrier(cx, object);
  }
  return object;
}

/// Slow-path object allocator. MAY collect. Memory is NOT zeroed.
template <typename T = void, typename S, typename Init, typename... Args>
Ref<T> allocateSlow(Context<S> &cx, std::size_t size, Init &&init,
                    Args &&...args) noexcept {
  auto [allocation, tax] = allocateBytesSlow(cx, size);
  auto object = cast<T>(allocation);
  if (object) {
    init(object, std::forward<Args>(args)...);
    allocateBarrier(cx, object);
    if (tax) {
      pay<S>(cx, tax);
    }
  }
  return object;
}

/// Slow-path object allocator. MAY collect. Memory IS zeroed.
template <typename T = void, typename S, typename Init, typename... Args>
Ref<T> allocateZeroSlow(Context<S> &cx, std::size_t size, Init &&init,
                        Args &&...args) noexcept {
  auto [allocation, tax] = allocateBytesZeroSlow(cx, size);
  auto object = cast<T>(allocation);
  if (object) {
    init(object, std::forward<Args>(args)...);
    allocateBarrier(cx, object);
    if (tax) {
      pay<S>(cx, tax);
    }
  }
  return object;
}

/// Slow-path object allocator. Will NOT collect. Memory is NOT zeroed.
template <typename T = void, typename S, typename Init, typename... Args>
Ref<T> allocateNoCollectSlow(Context<S> &cx, std::size_t size, Init &&init,
                             Args &&...args) noexcept {
  auto object = cast<T>(allocateBytesNoCollectSlow(cx, size));
  if (object) {
    init(object, std::forward<Args>(args)...);
    allocateBarrier(cx, object);
  }
  return object;
}

/// Slow-path object allocator. Will NOT collect. Memory IS zeroed.
template <typename T = void, typename S, typename Init, typename... Args>
Ref<T> allocateZeroNoCollectSlow(Context<S> &cx, std::size_t size, Init &&init,
                                 Args &&...args) noexcept {
  auto object = cast<T>(allocateBytesZeroNoCollectSlow(cx, size));
  if (object) {
    init(object, std::forward<Args>(args)...);
    allocateBarrier(cx, object);
  }
  return object;
}

//===----------------------------------------------------------------------===//
// General Purpose Object Allocators
//===----------------------------------------------------------------------===//

/// Allocate an object and initialize it.  May cause a garbage collection.
template <typename T = void, typename S, typename Init, typename... Args>
Ref<T> allocate(Context<S> &cx, std::size_t size, Init &&init,
                Args &&...args) noexcept {
  auto object = allocateFast<T>(cx, size, std::forward<Init>(init),
                                std::forward<Args>(args)...);
  if (object) {
    return object;
  }

  return allocateSlow<T>(cx, size, std::forward<Init>(init),
                         std::forward<Args>(args)...);
}

/// Allocate an object and initialize it.  May cause garbage collection. The
/// underlying memory will be initialized to zero.
template <typename T = void, typename S, typename Init, typename... Args>
Ref<T> allocateZero(Context<S> &cx, std::size_t size, Init &&init,
                    Args &&...args) noexcept {
  auto object = allocateZeroFast<T>(cx, size, std::forward<Init>(init),
                                    std::forward<Args>(args)...);
  if (object) {
    return object;
  }

  return allocateZeroSlow<T>(cx, size, std::forward<Init>(init),
                             std::forward<Args>(args)...);
}

/// Allocate an object and initalize it.  Will not garbage collect.
template <typename T = void, typename S, typename Init, typename... Args>
Ref<T> allocateNoCollect(Context<S> &cx, std::size_t size, Init &&init,
                         Args &&...args) noexcept {
  auto object = allocateFast<T>(cx, size, std::forward<Init>(init),
                                std::forward<Args>(args)...);
  if (object) {
    return object;
  }

  return allocateNoCollectSlow<T>(cx, size, std::forward<Init>(init),
                                  std::forward<Args>(args)...);
}

/// Allocate an object and initalize it.  Will not garbage collect. The
/// underlying memory will be initialized to zero.
template <typename T = void, typename S, typename Init, typename... Args>
Ref<T> allocateZeroNoCollect(Context<S> &cx, std::size_t size, Init &&init,
                             Args &&...args) noexcept {
  auto object = allocateZeroFast<T>(cx, size, std::forward<Init>(init),
                                    std::forward<Args>(args)...);
  if (object) {
    return object;
  }

  return allocateZeroNoCollectSlow<T>(cx, size, std::forward<Init>(init),
                                      std::forward<Args>(args)...);
}

} // namespace om::gc

#endif // OM_GC_ALLOCATE_H