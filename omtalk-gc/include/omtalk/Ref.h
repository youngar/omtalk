#ifndef OMTALK_GC_REF_HPP_
#define OMTALK_GC_REF_HPP_

#include <cstdint>
#include <type_traits>

namespace omtalk::gc {

template <typename T = void>
using PrimitiveRef = T *;

template <typename T = void>
class Ref final {
public:
  static Ref<T> fromAddr(std::uintptr_t addr) noexcept {
    return Ref<T>(reinterpret_cast<T *>(addr));
  }

  Ref() = default;

  constexpr Ref(T *value) : value_(value) {}

  constexpr Ref(const Ref &other) = default;

  template <typename U>
  constexpr Ref(const Ref<U> &other) : value_(other.get()) {}

  constexpr T *get() const noexcept { return value_; }

  std::uintptr_t toAddr() const noexcept {
    return reinterpret_cast<std::uintptr_t>(value_);
  }

  constexpr T *operator->() const noexcept { return value_; }

  constexpr T &operator*() const noexcept { return value_; }

  template <typename U,
            typename std::enable_if_t<std::is_convertible_v<T *, U *>>>
  constexpr operator Ref<U>() const {
    return Ref<U>(get());
  }

  constexpr bool operator==(nullptr_t) const { return value_ == nullptr; }

  constexpr bool operator!=(nullptr_t) const { return value_ != nullptr; }

private:
  PrimitiveRef<T> value_;
};

template <>
class Ref<void> final {
public:
  Ref() = default;

  constexpr Ref(void *value) : value_(value) {}

  constexpr Ref(const Ref<void> &other) = default;

  template <typename U>
  constexpr Ref(Ref<U> other) : value_(other.get()) {}

  constexpr void *get() const noexcept { return value_; }

  std::uintptr_t toAddr() const noexcept {
    return reinterpret_cast<std::uintptr_t>(value_);
  }

  constexpr bool operator==(nullptr_t) const { return value_ == nullptr; }

  constexpr bool operator!=(nullptr_t) const { return value_ != nullptr; }

private:
  PrimitiveRef<void> value_;
};

} // namespace omtalk::gc

#endif // OMTALK_GC_REF_HPP_
