#ifndef OM_GC_BARRIER_H
#define OM_GC_BARRIER_H

#include <om/GC/ForwardingMap.h>
#include <om/GC/Heap.h>
#include <om/GC/Mark.h>
#include <om/GC/MemoryManager.h>
#include <om/GC/Scheme.h>

namespace om::gc {

/// Called after an object has been allocated and initialized.
template <typename S, typename ObjectProxyT>
void allocateBarrier(Context<S> &context, ObjectProxyT object) {
  // Newly allocated objects must be marked black and scanned by the GC.
  if (context.isMarking()) {
    mark<S>(context.getCollectorContext(), object);
  }
}

template <typename T = void, typename S, typename SlotProxyT>
auto load(Context<S> &context, SlotProxyT slot) {

  // if the slot points to an evactuate region, copy the object into the current
  // allocation region.
  Ref<T> address = ab::proxy::load<ab::SEQ_CST>(slot);
  auto *region = Region::get(address);
  if (region->isEvacuating()) {
    ForwardingMap &map = region->getForwardingMap();
    auto &entry = map[address.get()];
    if (entry.tryLock()) {
      // Copy the object to the local allocation buffer.
      GlobalCollectorContext<S> &gcContext = context.getCollectorContext();
      auto &buffer = context.buffer();
      auto *to = buffer.begin;
      auto available = buffer.available();
      auto result = copy(gcContext, ObjectProxy<S>(address), to, available);

      if (!result) {
        // get a new allocation buffer
        auto copySize = getCopySize<S>(address);
        context.refreshBuffer(copySize);
        // Refreshing the buffer may block the thread until the end of the
        // garbage collection cycle.
        to = buffer.begin;
        available = buffer.available();
        result = copy(gcContext, ObjectProxy<S>(address), to, available);
      }

      entry.set(to);
      address = Ref<T>::fromPtr(to);
    } else {
      // Object is being or has already been copied. entry.get() will wait for
      // the object to be copied.
      address = Ref<T>::fromPtr(entry.get());
    }
    // Update the slot with the object's new address
    ab::proxy::store<ab::SEQ_CST>(slot, address);
  }

  // Mark the object for concurrent marking
  if (context.isMarking()) {
    mark(context.getCollectorContext(), address);
  }

  return address;
}

template <typename S, typename ObjectProxyT, typename SlotProxyT,
          typename ValueT>
void store(Context<S> &context, ObjectProxyT object, SlotProxyT slot,
           ValueT value) {
  ab::proxy::store<ab::SEQ_CST>(slot, value);
}

template <typename S, typename T>
auto load(Context<S> &context, Handle<T> &handle) noexcept {
  return load(context, handle.proxy());
}

template <typename S, typename T, typename V>
auto store(Context<S> &context, Handle<T> &handle, V value) noexcept {
  return store(context, handle.proxy());
}

} // namespace om::gc

#endif