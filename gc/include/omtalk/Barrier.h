#ifndef OMTALK_BARRIER_H
#define OMTALK_BARRIER_H

#include <omtalk/ForwardingMap.h>
#include <omtalk/Heap.h>
#include <omtalk/Mark.h>
#include <omtalk/MemoryManager.h>
#include <omtalk/Scheme.h>

namespace omtalk::gc {

/// Called after an object has been allocated and initialized.
template <typename S, typename ObjectProxyT>
void allocateBarrier(Context<S> &context, ObjectProxyT object)
    OMTALK_REQUIRES_CONTEXT(context) {
  // Newly allocated objects must be marked black and scanned by the GC.
  if (context.isMarking()) {
    mark<S>(context.getCollectorContext(), object);
  }
}

template <typename S, typename SlotProxyT>
auto load(Context<S> &context, SlotProxyT slot)
    OMTALK_REQUIRES_CONTEXT(context) {

  // if the slot points to an evactuate region, copy the object into the current
  // allocation region.
  Ref<void> address = proxy::load<SEQ_CST>(slot);
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
      address = to;
    } else {
      // Object is being or has already been copied. entry.get() will wait for
      // the object to be copied.
      address = entry.get();
    }
    // Update the slot with the object's new address
    proxy::store<SEQ_CST>(slot, address);
  }

  // Mark the object for concurrent marking
  if (context.isMarking()) {
    mark<S>(context.getCollectorContext(), address);
  }

  return address;
}

template <typename S, typename ObjectProxyT, typename SlotProxyT,
          typename ValueT>
void store(Context<S> &context, ObjectProxyT object, SlotProxyT slot,
           ValueT value) noexcept OMTALK_REQUIRES_CONTEXT(context) {
  proxy::store<SEQ_CST>(slot, value);
}

template <typename S, typename T>
auto load(Context<S> &context, Handle<T> &handle) noexcept
    OMTALK_REQUIRES_CONTEXT(context) {
  return load(context, handle.proxy());
}

template <typename S, typename T, typename V>
auto store(Context<S> &context, Handle<T> &handle, V value) noexcept
    OMTALK_REQUIRES_CONTEXT(context) {
  return store(context, handle.proxy());
}

} // namespace omtalk::gc

#endif