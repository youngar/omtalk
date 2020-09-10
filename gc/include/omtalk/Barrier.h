#ifndef OMTALK_BARRIER_H
#define OMTALK_BARRIER_H

#include <omtalk/Mark.h>
#include <omtalk/MemoryManager.h>
#include <omtalk/Scheme.h>

namespace omtalk::gc {

template <typename S, typename ObjectProxyT, typename SlotProxyT>
void preLoadBarrier(Context<S> &cx, ObjectProxyT &object, SlotProxyT &slot) {}

template <typename S, typename ObjectProxyT, typename SlotProxyT>
void postLoadBarrier(Context<S> &cx, ObjectProxyT &object, SlotProxyT &slot) {}

template <typename S, typename ObjectProxyT, typename SlotProxyT,
          typename ValueT>
void preStoreBarrier(Context<S> &cx, ObjectProxyT object, SlotProxyT &slot,
                     ValueT value) {}

template <typename S, typename ObjectProxyT, typename SlotProxyT,
          typename ValueT>
void postStoreBarrier(Context<S> &cx, ObjectProxyT object, SlotProxyT &slot,
                      ValueT value) {}

/// Called after an object has been allocated and initialized.
template <typename S, typename ObjectProxyT>
void allocateBarrier(Context<S> &cx, ObjectProxyT object) {
  // Newly allocated objects must be marked black and scanned by the GC.
  mark<S>(cx.getCollectorContext(), object);
}

template <typename S, typename ObjectProxyT, typename SlotProxyT>
auto load(Context<S> &cx, ObjectProxyT &object, SlotProxyT &slot) {
  preLoadBarrier(cx, object, slot);
  auto result = slot.load();
  postLoadBarrier(cx);
  return result;
}

template <typename S, typename ObjectProxyT, typename SlotProxyT,
          typename ValueT>
auto store(Context<S> &cx, ObjectProxyT object, SlotProxyT &slot,
           ValueT value) {
  preStoreBarrier(cx);
  auto result = slot.store(value);
  postStoreBarrier(cx);
  return result;
}

} // namespace omtalk::gc

#endif