#ifndef OM_OM_SLOTPROXY_H
#define OM_OM_SLOTPROXY_H

#include <ab/Util/Atomic.h>
#include <cstdint>
#include <om/GC/Ref.h>

namespace om::om {

// A handle to an object slot. The slot holds a reference to an object.
// Provides an API to load from, or store to, the target object slot.
class SlotProxy {
public:
  static SlotProxy fromAddr(std::uintptr_t addr) noexcept {
    return SlotProxy(reinterpret_cast<void **>(addr));
  }

  static SlotProxy fromPtr(void *ptr) noexcept {
    return SlotProxy(reinterpret_cast<void **>(ptr));
  }

  explicit SlotProxy(void **target) noexcept : target(target) {}

  /// Load from the underlying object slot.
  template <ab::MemoryOrder M>
  gc::Ref<void> load() const noexcept {
    return gc::Ref<void>::fromPtr(ab::mem::load<M>(target));
  }

  /// Store to the underlying object slot.
  template <ab::MemoryOrder M>
  void store(gc::Ref<void> value) const noexcept {
    ab::mem::store<M>(target, value.get());
  }

private:
  void **target;
};

} // namespace om::om

#endif // OM_OM_SLOTPROXY_H
