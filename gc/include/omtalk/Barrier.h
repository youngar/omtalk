#ifndef OMTALK_BARRIER_H_
#define OMTALK_BARRIER_H_

#include <omtalk/Tracing.h>

namespace omtalk::gc {

template <typename ObjectProxyT, typename SlotProxyT>
void preLoadBarrier(Context &cx, ObjectProxyT &object,
                    SlotProxyT &slot) {}

template <typename ObjectProxyT, typename SlotProxyT>
void postLoadBarrier(Context &cx, ObjectProxyT &object,
                     SlotProxyT &slot) {}

template <typename ObjectProxyT, typename SlotProxyT, typename ValueT>
void preStoreBarrier(Context &cx, ObjectProxyT object,
                     SlotProxyT &slot, ValueT value) {}

template <typename ObjectProxyT, typename SlotProxyT, typename ValueT>
void postStoreBarrier(Context &cx, ObjectProxyT object,
                      SlotProxyT &slot, ValueT value) {}

template <typename ObjectProxyT, typename SlotProxyT>
auto load(Context &cx, ObjectProxyT &object, SlotProxyT &slot) {
  preLoadBarrier(cx, object, slot);
  auto result = slot.load();
  postLoadBarrier(cx);
  return result;
}

template <typename ObjectProxyT, typename SlotProxyT, typename ValueT>
auto store(Context &cx, ObjectProxyT object, SlotProxyT &slot,
           ValueT value) {
  preStoreBarrier(cx);
  auto result = slot.store(value);
  postStoreBarrier(cx);
  return result;
}

} // namespace omtalk::gc

#endif // OMTALK_BARRIER_H_
