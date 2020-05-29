#include <omtalk/Allocate.h>
#include <omtalk/MemoryManager.h>

using namespace omtalk::gc;

AllocationResult omtalk::gc::allocateBytesSlow(Context &cx,
                                               std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  auto allocation = allocateBytesFast(cx, size);
  return {allocation, Tax()};
}

AllocationResult omtalk::gc::allocateBytesZeroSlow(Context &cx,
                                                   std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  auto allocation = allocateBytesFast(cx, size);
  return {allocation, Tax()};
}

Ref<void> omtalk::gc::allocateBytesNoCollectSlow(Context &cx,
                                                 std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  return allocateBytesFast(cx, size);
}

Ref<void>
omtalk::gc::allocateBytesZeroNoCollectSlow(Context &cx,
                                           std::size_t size) noexcept {
  cx.getCollector()->refreshBuffer(cx, size);
  return allocateBytesFast(cx, size);
}