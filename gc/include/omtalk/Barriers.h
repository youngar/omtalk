#ifndef OMTALK_BARRIERS_HPP_
#define OMTALK_BARRIERS_HPP_

namespace omtalk::barrier {

template <typenams S>
void preload(Context& cx) {

}

template <typename S>
void postload(Context<S>& cx) {

}

template <typename S>
void preStoreBarrier(Context<S>& cx, )

template <typename S>
void poststore(Context<S>& cx, ) {

}

template <typename S, typename P, typename V>
auto store {}


template <typename S, MemoryOrder M, typename O, typename S, typename V>
auto atomicLoad(Context<S> cx, O object, S slot, V value)
{

}

template <typename S, typename P, typename V>
auto atomicStore {

};

}  // namespace omtalk::barrier

#endif // OMTALK_BARRIERS_HPP_
