#ifndef OMTALK_UTIL_ATOMIC_H_
#define OMTALK_UTIL_ATOMIC_H_

#include <type_traits>

namespace omtalk {

// https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html#g_t_005f_005fatomic-Builtins

enum MemoryOrder : int {
  RELAXED = __ATOMIC_RELAXED,
  CONSUME = __ATOMIC_CONSUME,
  ACQUIRE = __ATOMIC_ACQUIRE,
  RELEASE = __ATOMIC_RELEASE,
  ACQ_REL = __ATOMIC_ACQ_REL,
  SEQ_CST = __ATOMIC_SEQ_CST,
};

template <typename T>
T atomicLoad(T *addr, MemoryOrder order = SEQ_CST) {
  return __atomic_load_n(addr, int(order));
}

template <typename T>
void atomicStore(T *addr, T value, MemoryOrder order = SEQ_CST) {
  __atomic_store_n(addr, value, int(order));
}

template <typename T>
T atomicExchange(T *addr, T value, MemoryOrder order = SEQ_CST) {
  return __atomic_exchange_n(addr, value, int(order));
}

template <typename T>
bool atomicCompareExchange(T *addr, T expected, T desired,
                           MemoryOrder succ = SEQ_CST,
                           MemoryOrder fail = RELAXED) {
  return __atomic_compare_exchange_n(addr, &expected, desired, int(succ),
                                     int(fail), false);
}

} // namespace omtalk

#endif // OMTALK_UTIL_ATOMIC_H_
