#ifndef OMTALK_GC_ALLOCATE_H_
#define OMTALK_GC_ALLOCATE_H_

#include <cstddef>
#include <cstdint>
#include <omtalk/MemoryManager.h>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>
#include <omtalk/Util/Bytes.h>

namespace omtalk::gc {

class Context;

//===----------------------------------------------------------------------===//
// BaseInit
//===----------------------------------------------------------------------===//

template <typename T>
class BasicInit {
public:
  void operator()(Ref<T> target) {} // no-op.
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

inline void pay(Context &cx, const Tax &tax) {
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
inline Ref<void> allocateBytesFast(Context &cx,
                                   std::size_t size) noexcept {
  return cx.buffer().tryAllocate(size);
}

/// Fast-path byte allocator. Will NOT collect. Memory is zeroed.
inline Ref<void> allocateBytesZeroFast(Context &cx,
                                       std::size_t size) noexcept {
  return cx.buffer().tryAllocate(size);
}

/// Slow-path byte allocator. MAY collect. Memory is NOT zeroed.
AllocationResult allocateBytesSlow(Context &cx,
                                   std::size_t size) noexcept;

/// Slow-path byte allocator. MAY collect. Memory IS zeroed.
AllocationResult allocateBytesZeroSlow(Context &cx,
                                       std::size_t size) noexcept;

/// Slow-path byte allocator. WILL NOT collect. Memory is NOT zeroed.
Ref<void> allocateBytesNoCollectSlow(Context &cx,
                                     std::size_t size) noexcept;

/// Slow-path byte allocator. Will NOT collect. Memory IS zeroed.
Ref<void> allocateBytesZeroNoCollectSlow(Context &cx,
                                         std::size_t size) noexcept;

//===----------------------------------------------------------------------===//
// Internal Object Allocators
//===----------------------------------------------------------------------===//

/// Fast-path object allocator. Will NOT collect. Memory is NOT zeroed.
template <typename T = void, typename Init, typename... Args>
Ref<T> allocateFast(Context &cx, std::size_t size, Init &&init,
                    Args &&... args) noexcept {
  auto object = allocateBytesFast(cx, size).cast<T>();
  if (object) {
    init(object, std::forward<Args>(args)...);
  }
  return object;
}

/// /// Fast-path object allocator. Will NOT collect. Memory IS zeroed.
template <typename T = void, typename Init, typename... Args>
Ref<T> allocateZeroFast(Context &cx, std::size_t size, Init &&init,
                        Args &&... args) noexcept {
  auto object = allocateBytesZeroFast(cx, size).cast<T>();
  if (object) {
    init(object, std::forward<Args>(args)...);
  }
  return object;
}

/// Slow-path object allocator. MAY collect. Memory is NOT zeroed.
template <typename T = void, typename Init, typename... Args>
Ref<T> allocateSlow(Context &cx, std::size_t size, Init &&init,
                    Args &&... args) noexcept {
  auto [allocation, tax] = allocateBytesSlow(cx, size);
  auto object = allocation.cast<T>();
  if (object) {
    init(object, std::forward<Args>(args)...);
    if (tax) {
      pay(cx, tax);
    }
  }
  return object;
}

/// Slow-path object allocator. MAY collect. Memory IS zeroed.
template <typename T = void, typename Init, typename... Args>
Ref<T> allocateZeroSlow(Context &cx, std::size_t size, Init &&init,
                        Args &&... args) noexcept {
  auto [allocation, tax] = allocateBytesZeroSlow(cx, size);
  auto object = allocation.cast<T>();
  if (object) {
    init(object, std::forward<Args>(args)...);
    if (tax) {
      pay(cx, tax);
    }
  }
  return object;
}

/// Slow-path object allocator. Will NOT collect. Memory is NOT zeroed.
template <typename T = void, typename Init, typename... Args>
Ref<T> allocateNoCollectSlow(Context &cx, std::size_t size,
                             Init &&init, Args &&... args) noexcept {
  auto object = allocateBytesNoCollectSlow(cx, size).cast<T>();
  if (object) {
    init(object, std::forward<Args>(args)...);
  }
  return object;
}

/// Slow-path object allocator. Will NOT collect. Memory IS zeroed.
template <typename T = void, typename Init, typename... Args>
Ref<T> allocateZeroNoCollectSlow(Context &cx, std::size_t size,
                                 Init &&init, Args &&... args) noexcept {
  auto object = allocateBytesZeroNoCollectSlow(cx, size).cast<T>();
  if (object) {
    init(object, std::forward<Args>(args)...);
  }
  return object;
}

//===----------------------------------------------------------------------===//
// General Porpoise Object Allocators
//===----------------------------------------------------------------------===//

/// Allocate an object and initialize it.  May cause a garbage collection.
template <typename T = void, typename Init, typename... Args>
Ref<T> allocate(Context &cx, std::size_t size, Init &&init,
                Args &&... args) noexcept {
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
template <typename T = void, typename Init, typename... Args>
Ref<T> allocateZero(Context &cx, std::size_t size, Init &&init,
                    Args &&... args) noexcept {
  auto object = allocateZeroFast<T>(cx, size, std::forward<Init>(init),
                                    std::forward<Args>(args)...);
  if (object) {
    return object;
  }

  return allocateZeroSlow<T>(cx, size, std::forward<Init>(init),
                             std::forward<Args>(args)...);
}

/// Allocate an object and initalize it.  Will not garbage collect.
template <typename T = void, typename Init, typename... Args>
Ref<T> allocateNoCollect(Context &cx, std::size_t size, Init &&init,
                         Args &&... args) noexcept {
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
template <typename T = void, typename Init, typename... Args>
Ref<T> allocateZeroNoCollect(Context &cx, std::size_t size,
                             Init &&init, Args &&... args) noexcept {
  auto object = allocateZeroFast<T>(cx, size, std::forward<Init>(init),
                                    std::forward<Args>(args)...);
  if (object) {
    return object;
  }

  return allocateZeroNoCollectSlow<T>(cx, size, std::forward<Init>(init),
                                      std::forward<Args>(args)...);
}

} // namespace omtalk::gc

#endif
