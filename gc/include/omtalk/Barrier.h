#ifndef OMTALK_BARRIER_H_
#define OMTALK_BARRIER_H_

#include <omtalk/Tracing.h>

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

#endif // OMTALK_BARRIER_H_
