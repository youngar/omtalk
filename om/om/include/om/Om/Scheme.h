#ifndef OMTALK_OM_SCHEME_H
#define OMTALK_OM_SCHEME_H

#include <cstdint>
#include <om/GC/Ref.h>
#include <om/Om/Layout.h>
#include <om/Om/Object.h>
#include <om/Om/ObjectHeader.h>

namespace om::om {

// A handle to an object slot. The slot holds a reference to an object.
class SlotProxy {
public:
  static SlotProxy fromAddr(std::uintptr_t addr) noexcept {
    return SlotProxy(reinterpret_cast<gc::Ref<void> *>(addr));
  }

  static SlotProxy fromPtr(void *ptr) noexcept {
    return SlotProxy(reinterpret_cast<gc::Ref<void> *>(ptr));
  }

  explicit SlotProxy(gc::Ref<void> *target) noexcept : target(target) {}

  /// Load from the underlying object slot.
  template <ab::MemoryOrder M>
  gc::Ref<void> load() const noexcept {
    return ab::mem::load<M>(target);
  }

  /// Store to the underlying object slot.
  template <ab::MemoryOrder M>
  void store(gc::Ref<void> value) const noexcept {
    ab::mem::store<M>(target, value);
  }

private:
  gc::Ref<void> *target;
};

class ObjectProxy {
public:
  template <typename T>
  ObjectProxy(gc::Ref<T> target) : target(target) {}

  SlotProxy slot(unsigned offset) {
    return SlotProxy::fromPtr(&target->data + offset);
  }

private:
  gc::Ref<Object> target;
};

class Scheme {
  using ObjectProxy = ObjectProxy;
};

} // namespace om::om

#endif // OMTALK_OM_SCHEME_H
