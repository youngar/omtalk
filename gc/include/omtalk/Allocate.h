#ifndef OMTALK_GC_ALLOCATE_H_
#define OMTALK_GC_ALLOCATE_H_

#include <cstddef>
#include <cstdint>
#include <omtalk/Heap.h>
#include <omtalk/MemoryManager.h>
#include <omtalk/Ref.h>
#include <omtalk/Util/Bytes.h>

namespace omtalk::gc {

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
  auto allocation = allocateBytesFast<S>(cx, size);
  return {allocation, Tax()};
}

/// Slow-path byte allocator. MAY collect. Memory IS zeroed.
template <typename S>
AllocationResult allocateBytesZeroSlow(Context<S> &cx,
                                       std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  auto allocation = allocateBytesFast<S>(cx, size);
  return {allocation, Tax()};
}

/// Slow-path byte allocator. WILL NOT collect. Memory is NOT zeroed.
template <typename S>
Ref<void> allocateBytesNoCollectSlow(Context<S> &cx,
                                     std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  return allocateBytesFast<S>(cx, size);
}

/// Slow-path byte allocator. Will NOT collect. Memory IS zeroed.
template <typename S>
Ref<void> allocateBytesZeroNoCollectSlow(Context<S> &cx,
                                         std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  return allocateBytesFast<S>(cx, size);
}

//===----------------------------------------------------------------------===//
// Internal Object Allocators
//===----------------------------------------------------------------------===//

/// Fast-path object allocator. Will NOT collect. Memory is NOT zeroed.
template <typename S, typename T = void, typename Init, typename... Args>
Ref<T> allocateFast(Context<S> &cx, std::size_t size, Init &&init,
                    Args &&... args) noexcept {
  auto object = cast<T>(allocateBytesFast<S>(cx, size));

  if (object) {
    init(object, std::forward<Args>(args)...);
  }
  return object;
}

/// Fast-path object allocator. Will NOT collect. Memory IS zeroed.
template <typename S, typename T = void, typename Init, typename... Args>
Ref<T> allocateZeroFast(Context<S> &cx, std::size_t size, Init &&init,
                        Args &&... args) noexcept {
  auto object = cast<T>(allocateBytesZeroFast<S>(cx, size));
  if (object) {
    init(object, std::forward<Args>(args)...);
  }
  return object;
}

/// Slow-path object allocator. MAY collect. Memory is NOT zeroed.
template <typename S, typename T = void, typename Init, typename... Args>
Ref<T> allocateSlow(Context<S> &cx, std::size_t size, Init &&init,
                    Args &&... args) noexcept {
  auto [allocation, tax] = allocateBytesSlow<S>(cx, size);
  auto object = cast<T>(allocation);
  if (object) {
    init(object, std::forward<Args>(args)...);
    if (tax) {
      pay<S>(cx, tax);
    }
  }
  return object;
}

/// Slow-path object allocator. MAY collect. Memory IS zeroed.
template <typename S, typename T = void, typename Init, typename... Args>
Ref<T> allocateZeroSlow(Context<S> &cx, std::size_t size, Init &&init,
                        Args &&... args) noexcept {
  auto [allocation, tax] = allocateBytesZeroSlow<S>(cx, size);
  auto object = cast<T>(allocation);
  if (object) {
    init(object, std::forward<Args>(args)...);
    if (tax) {
      pay<S>(cx, tax);
    }
  }
  return object;
}

/// Slow-path object allocator. Will NOT collect. Memory is NOT zeroed.
template <typename S, typename T = void, typename Init, typename... Args>
Ref<T> allocateNoCollectSlow(Context<S> &cx, std::size_t size, Init &&init,
                             Args &&... args) noexcept {
  auto object = cast<T>(allocateBytesNoCollectSlow<S>(cx, size));
  if (object) {
    init(object, std::forward<Args>(args)...);
  }
  return object;
}

/// Slow-path object allocator. Will NOT collect. Memory IS zeroed.
template <typename S, typename T = void, typename Init, typename... Args>
Ref<T> allocateZeroNoCollectSlow(Context<S> &cx, std::size_t size, Init &&init,
                                 Args &&... args) noexcept {
  auto object = cast<T>(allocateBytesZeroNoCollectSlow<S>(cx, size));
  if (object) {
    init(object, std::forward<Args>(args)...);
  }
  return object;
}

//===----------------------------------------------------------------------===//
// General Purpose Object Allocators
//===----------------------------------------------------------------------===//

/// Allocate an object and initialize it.  May cause a garbage collection.
template <typename S, typename T = void, typename Init, typename... Args>
Ref<T> allocate(Context<S> &cx, std::size_t size, Init &&init,
                Args &&... args) noexcept {
  auto object = allocateFast<S, T>(cx, size, std::forward<Init>(init),
                                   std::forward<Args>(args)...);
  if (object) {
    return object;
  }

  return allocateSlow<S, T>(cx, size, std::forward<Init>(init),
                            std::forward<Args>(args)...);
}

/// Allocate an object and initialize it.  May cause garbage collection. The
/// underlying memory will be initialized to zero.
template <typename S, typename T = void, typename Init, typename... Args>
Ref<T> allocateZero(Context<S> &cx, std::size_t size, Init &&init,
                    Args &&... args) noexcept {
  auto object = allocateZeroFast<S, T>(cx, size, std::forward<Init>(init),
                                       std::forward<Args>(args)...);
  if (object) {
    return object;
  }

  return allocateZeroSlow<S, T>(cx, size, std::forward<Init>(init),
                                std::forward<Args>(args)...);
}

/// Allocate an object and initalize it.  Will not garbage collect.
template <typename S, typename T = void, typename Init, typename... Args>
Ref<T> allocateNoCollect(Context<S> &cx, std::size_t size, Init &&init,
                         Args &&... args) noexcept {
  auto object = allocateFast<S, T>(cx, size, std::forward<Init>(init),
                                   std::forward<Args>(args)...);
  if (object) {
    return object;
  }

  return allocateNoCollectSlow<S, T>(cx, size, std::forward<Init>(init),
                                     std::forward<Args>(args)...);
}

/// Allocate an object and initalize it.  Will not garbage collect. The
/// underlying memory will be initialized to zero.
template <typename S, typename T = void, typename Init, typename... Args>
Ref<T> allocateZeroNoCollect(Context<S> &cx, std::size_t size, Init &&init,
                             Args &&... args) noexcept {
  auto object = allocateZeroFast<S, T>(cx, size, std::forward<Init>(init),
                                       std::forward<Args>(args)...);
  if (object) {
    return object;
  }

  return allocateZeroNoCollectSlow<S, T>(cx, size, std::forward<Init>(init),
                                         std::forward<Args>(args)...);
}

} // namespace omtalk::gc

#endif
