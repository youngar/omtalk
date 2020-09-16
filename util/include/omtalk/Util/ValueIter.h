#ifndef OMTALK_UTIL_VALUEITER_H
#define OMTALK_UTIL_VALUEITER_H

namespace omtalk {

/// A value generator.  Will produce values starting from the initial value and
/// increasing.
template <typename T>
class ValueIter {
public:
  using difference_type = T;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::forward_iterator_tag;

  constexpr explicit ValueIter(T x) noexcept : x(x) {}

  ValueIter &operator++() noexcept {
    ++x;
    return *this;
  }

  ValueIter operator++(int) const noexcept {
    auto copy = *this;
    ++(*this);
    return copy;
  }

  ValueIter &operator--() noexcept {
    --x;
    return *this;
  }

  ValueIter operator--(int) const noexcept {
    auto copy = *this;
    --(*this);
    return copy;
  }

  constexpr bool operator==(const ValueIter &rhs) const noexcept {
    return x == rhs.x;
  }

  constexpr bool operator!=(const ValueIter &rhs) const noexcept {
    return x != rhs.x;
  }

  constexpr T operator*() const noexcept { return x; }

  constexpr T operator-(const ValueIter &rhs) const noexcept {
    return x - rhs.x;
  }

private:
  T x;
};

template <typename T>
auto toIter(T x) -> ValueIter<T> {
  return ValueIter<T>(x);
}

} // namespace omtalk

#endif