#ifndef OMTALK_UTIL_ATOMIC_H_
#define OMTALK_UTIL_ATOMIC_H_

#include <type_traits>

namespace omtalk {

// https://gcc.gnu.org/wiki/Atomic/GCCMM/AtomicSync
// https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html#g_t_005f_005fatomic-Builtins

/// Strong enum for memory order.
enum class MemoryOrder : int {
  RELAXED = __ATOMIC_RELAXED,
  CONSUME = __ATOMIC_CONSUME,
  ACQUIRE = __ATOMIC_ACQUIRE,
  RELEASE = __ATOMIC_RELEASE,
  ACQ_REL = __ATOMIC_ACQ_REL,
  SEQ_CST = __ATOMIC_SEQ_CST,
};

// short forms.
constexpr auto RELAXED = MemoryOrder::RELAXED;
constexpr auto CONSUME = MemoryOrder::CONSUME;
constexpr auto ACQUIRE = MemoryOrder::ACQUIRE;
constexpr auto RELEASE = MemoryOrder::RELEASE;
constexpr auto ACQ_REL = MemoryOrder::ACQ_REL;
constexpr auto SEQ_CST = MemoryOrder::SEQ_CST;

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

namespace mem {

template <MemoryOrder M, typename T>
T load(T *addr) {
  return __atomic_load_n(addr, int(M));
}

template <MemoryOrder M, typename T>
void store(T *addr, T value) {
  __atomic_store_n(addr, value, int(M));
}

template <MemoryOrder M, typename T>
T exchange(T *addr, T value) {
  return __atomic_exchange_n(addr, value, int(M));
}

template <MemoryOrder S, MemoryOrder F, typename T>
bool compareExchange(T *addr, T expected, T desired) {
  return __atomic_compare_exchange_n(addr, &expected, desired, int(S), int(F),
                                     false);
}

} // namespace mem

namespace proxy {

/// Call target.load<M>(). Useful when target is a templated type.
template <MemoryOrder M, typename T>
auto load(T&& target) noexcept {
  return target.template load<M>();
}

/// Call target.store<M>(value). Useful when target is a templated type.
template <MemoryOrder M, typename T, typename U>
auto store(T&& target, U&& value) noexcept {
  return target.template store<M>(std::forward<U>(value));
}

/// Call target.exchange<M>(value). Useful when target is a templated type.
template <MemoryOrder M, typename T, typename U>
auto exchange(T& target, U& value) noexcept {
  return target.template exchange<M>(value);
}

template <MemoryOrder M, typename T, typename U, typename V>
auto compareExchange(T& target, U& expected, V& desired) noexcept {
  return target.template compareExchange<M>(expected, desired);
}

} // namespace proxy

} // namespace omtalk

#endif // OMTALK_UTIL_ATOMIC_H_
