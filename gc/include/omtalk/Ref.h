#ifndef OMTALK_GC_REF_HPP_
#define OMTALK_GC_REF_HPP_

#include <cstdint>
#include <omtalk/Util/Atomic.h>
#include <ostream>
#include <type_traits>

namespace omtalk::gc {

class RefProxy;

/// A GC Reference. Put simply, a Ref<int> is a pointer to a heap-allocated int.
/// It will quack like a pointer. Why? To _highlight_ bare references into the
/// heap, which are unsafe dangerous things.
template <typename T = void>
class Ref final {
public:
  static Ref<T> fromAddr(std::uintptr_t addr) noexcept {
    return Ref<T>(reinterpret_cast<T *>(addr));
  }

  Ref() noexcept = default;

  constexpr Ref(std::nullptr_t) : value_(nullptr) {}

  constexpr Ref(T *value) : value_(value) {}

  constexpr Ref(const Ref &other) = default;

  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U, T>>>
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
    return Ref<U>(static_cast<U *const>(value_));
  }

  template <typename U>
  Ref<U> reinterpret() const {
    return Ref<U>(reinterpret_cast<U *>(value_));
  }

  constexpr operator bool() const { return value_ != nullptr; }

  /// @group Memory Operations
  /// @{

  template <MemoryOrder M>
  Ref<T> load() const noexcept {
    return Ref<T>(mem::load<M>(&value_));
  }

  template <MemoryOrder M>
  void store(Ref<T> value) noexcept {
    mem::store<M>(&value_, value.get());
  }

  template <MemoryOrder M>
  Ref<T> exchange(Ref<T> value) noexcept {
    return Ref<T>(mem::exchange<M>(&value_, value.get()));
  }

  template <MemoryOrder S, MemoryOrder F = RELAXED>
  bool cas(Ref<T> expected, Ref<T> desired) noexcept {
    return mem::cas<S, F>(&value_, expected.get(), desired.get());
  }

  /// @}

  RefProxy proxy() noexcept;

private:
  T *value_;
};

template <>
class Ref<void> final {
public:
  Ref() noexcept = default;

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

  /// @group Memory Operations
  /// @{

  template <MemoryOrder M>
  Ref<void> load() const noexcept {
    return mem::load<M>(&value_);
  }

  template <MemoryOrder M>
  void store(Ref<void> value) noexcept {
    mem::store<M>(&value_, value.get());
  }

  template <MemoryOrder M>
  Ref<void> exchange(Ref<void> value) noexcept {
    return Ref<void>(mem::exchange<M>(&value_, value.get()));
  }

  template <MemoryOrder S, MemoryOrder F = RELAXED>
  bool compareExchange(Ref<void> expected, Ref<void> desired) noexcept {
    return mem::compareExchange<S, F>(&value_, expected.get(), desired.get());
  }

  /// @}

  RefProxy proxy() noexcept;

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

/// A type-erasing stand-in for a `Ref<T>`. Forwards all calls along to target.
class RefProxy {
public:
  template <typename T>
  explicit RefProxy(Ref<T> *target)
      : target(reinterpret_cast<Ref<void> *>(target)) {}

  /// Load from target Ref.
  template <MemoryOrder M>
  Ref<void> load() const noexcept {
    return target->load<M>();
  }

  /// Store to target Ref.
  template <MemoryOrder M>
  void store(Ref<void> value) const noexcept {
    return target->store<M>(value);
  }

  /// Exchange the held value for another.
  template <MemoryOrder M>
  Ref<void> exchange(Ref<void> value) const noexcept {
    return target->exchange<M>(value);
  }

  /// Compare-and-swap the held value.
  template <MemoryOrder S, MemoryOrder F = RELAXED>
  bool compareExchange(Ref<void> expected, Ref<void> value) const noexcept {
    return target->compareExchange<S, F>(expected, value);
  }

  /// Get the underlying target we are proxying.
  Ref<void> *get() const noexcept { return target; }

private:
  Ref<void> *target;
};

template <typename T>
RefProxy Ref<T>::proxy() noexcept {
  return RefProxy(this);
}

inline RefProxy Ref<void>::proxy() noexcept { return RefProxy(this); }

} // namespace omtalk::gc

#endif // OMTALK_GC_REF_HPP_
