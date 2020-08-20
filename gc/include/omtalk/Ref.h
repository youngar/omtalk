#ifndef OMTALK_GC_REF_HPP_
#define OMTALK_GC_REF_HPP_

#include <cstdint>
#include <omtalk/Util/Atomic.h>
#include <ostream>
#include <type_traits>

namespace omtalk::gc {

/// A GC Reference. Put simply, a Ref<int> is a pointer to a heap-allocated int.
/// It will quack like a pointer. Why? To _highlight_ bare references into the
/// heap, which are unsafe dangerous things.
template <typename T = void>
class Ref final {
public:
  static Ref<T> fromAddr(std::uintptr_t addr) noexcept {
    return Ref<T>(reinterpret_cast<T *>(addr));
  }

  Ref() = default;

  constexpr Ref(std::nullptr_t) : value_(nullptr) {}

  constexpr Ref(T *value) : value_(value) {}

  constexpr Ref(const Ref &other) = default;

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  explicit constexpr Ref(const Ref<U> &other) : value_(other.get()) {}

  constexpr T *get() const noexcept { return value_; }

  constexpr T load() const noexcept { return *value_; }

  std::uintptr_t toAddr() const noexcept {
    return reinterpret_cast<std::uintptr_t>(value_);
  }

  constexpr T *operator->() const noexcept { return value_; }

  constexpr T &operator*() const noexcept { return *value_; }

  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<T *, U *>>>
  constexpr operator Ref<U>() const {
    return Ref<U>(get());
  }

  constexpr bool operator==(std::nullptr_t) const { return value_ == nullptr; }

  constexpr bool operator!=(std::nullptr_t) const { return value_ != nullptr; }

  template <typename U,
            typename = decltype(std::declval<T *>() == std::declval<U *>())>
  constexpr bool operator==(Ref<U> rhs) const {
    return value_ == rhs.get();
  }

  template <typename U,
            typename = decltype(std::declval<T *>() != std::declval<U *>())>
  constexpr bool operator!=(Ref<U> rhs) const {
    return value_ != rhs.get();
  }

  /// Static cast from T to U.
  template <typename U>
  Ref<U> cast() const {
    return Ref<U>(static_cast<U * const>(value_));
  }

  template <typename U>
  Ref<U> reinterpret() const {
    return Ref<U>(reinterpret_cast<U *>(value_));
  }

  constexpr operator bool() const { return value_ != nullptr; }

  friend Ref<T> atomicLoad(Ref<T> *addr, MemoryOrder order = SEQ_CST) {
    return Ref<T>(atomicLoad(&addr->value_, order));
  }

  friend void atomicStore(Ref<T> *addr, Ref<T> value,
                          MemoryOrder order = SEQ_CST) {
    atomicStore(&addr->value_, value.get(), order);
  }

  friend Ref<T> atomicExchange(Ref<T> *addr, Ref<T> value,
                               MemoryOrder order = SEQ_CST) {
    return Ref<T>(atomicExchange(&addr->value_, value.get(), order));
  }

  friend bool atomicCompareExchange(Ref<T> *addr, Ref<T> expected,
                                    Ref<T> desired, MemoryOrder succ = SEQ_CST,
                                    MemoryOrder fail = RELAXED) {
    return atomicCompareExchange(&addr->value_, expected.get(), desired.get(),
                                 succ, fail);
  }

private:
  T *value_;
};

template <>
class Ref<void> final {
public:
  Ref() = default;

  constexpr Ref(std::nullptr_t) : value_(nullptr) {}

  constexpr Ref(void *value) : value_(value) {}

  constexpr Ref(const Ref<void> &other) = default;

  template <typename U>
  explicit constexpr Ref(Ref<U> other) : value_(other.get()) {}

  constexpr void *get() const noexcept { return value_; }

  std::uintptr_t toAddr() const noexcept {
    return reinterpret_cast<std::uintptr_t>(value_);
  }

  constexpr bool operator==(std::nullptr_t) const { return value_ == nullptr; }

  constexpr bool operator!=(std::nullptr_t) const { return value_ != nullptr; }

  template <typename U>
  constexpr bool operator==(Ref<U> rhs) const {
    return value_ == rhs.get();
  }

  template <typename U>
  constexpr bool operator!=(Ref<U> rhs) const {
    return value_ != rhs.get();
  }

  template <typename U>
  Ref<U> cast() const {
    return Ref<U>(static_cast<U *>(value_));
  }

  template <typename U>
  Ref<U> reinterpret() const {
    return Ref<U>(reinterpret_cast<U *>(value_));
  }

  constexpr operator bool() const { return value_ != nullptr; }

  friend Ref<void> atomicLoad(Ref<void> *addr, MemoryOrder order = SEQ_CST) {
    return Ref<void>(atomicLoad(&addr->value_, order));
  }

  friend void atomicStore(Ref<void> *addr, Ref<void> value,
                          MemoryOrder order = SEQ_CST) {
    atomicStore(&addr->value_, value.get(), order);
  }

  friend Ref<void> atomicExchange(Ref<void> *addr, Ref<void> value,
                                  MemoryOrder order = SEQ_CST) {
    return Ref<void>(atomicExchange(&addr->value_, value.get(), order));
  }

  friend bool atomicCompareExchange(Ref<void> *addr, Ref<void> expected,
                                    Ref<void> desired,
                                    MemoryOrder succ = SEQ_CST,
                                    MemoryOrder fail = RELAXED) {
    return atomicCompareExchange(&addr->value_, expected.get(), desired.get(),
                                 succ, fail);
  }

private:
  void *value_;
};

template <typename T>
std::ostream &operator<<(std::ostream &out, const Ref<T> &ref) {
  return out << "(Ref " << ref.get() << ")";
}

template <typename T, typename U>
constexpr Ref<T> cast(Ref<U> x) {
  return x.template cast<T>();
}

} // namespace omtalk::gc

#endif // OMTALK_GC_REF_HPP_
