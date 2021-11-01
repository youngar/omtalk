#ifndef AB_UTIL_EYTZINGER_H
#define AB_UTIL_EYTZINGER_H

#include <ab/Util/Assert.h>
#include <ab/Util/Bit.h>
#include <cstddef>
#include <iterator>
#include <utility>

namespace ab {

/// A compact form of a binary search tree.  The number of elements must be
/// calculated up front, and the keys must be inserted in order.  In general
/// this tree provides faster lookup than binary searching a sorted vector,
/// while using the same amount of space.
///
/// This tree works by sorting an array in an eytzinger layout, which stores the
/// tree in an array in row-order.  Branching to the left child becomes
/// `(2i +1)`, and branching to the right child is `(2i + 2)`, where `i` is the
/// current index.
///
///         i= 1 2 3 4 5 6 7
/// eytzinger= 4 2 6 1 3 5 7
///
template <typename K, typename V>
class EytzingerTree {
public:
  using ElementTy = std::pair<K, V>;

  class Iterator;

  class ConstIterator;

  /// Construct a tree with no elements.
  explicit EytzingerTree() noexcept : EytzingerTree(0) {}

  /// Construct a tree with `size` elements.
  explicit EytzingerTree(unsigned size) noexcept { alloc(size); }

  ~EytzingerTree() noexcept { clear(); }

  /// Insert all keys from the iterator into the tree.  There must be enough
  /// elements to fill the tree.
  template <typename I>
  void populate(I i) noexcept {
    eytzinger(i);
  }

  V &at(const K &key) noexcept {
    auto i = find(key);
    return data_[i].second;
  }

  const V &at(const K &key) const noexcept {
    auto i = find(key);
    return data_[i].second;
  }

  V &operator[](const K &key) noexcept { return at(key); }

  const V &operator[](const K &key) const noexcept {}

  Iterator begin() noexcept { return Iterator(&nth(0)); }

  ConstIterator begin() const noexcept { return Iterator(&nth(0)); }

  Iterator end() noexcept { return Iterator(&nth(size())); }

  ConstIterator end() const noexcept { return Iterator(&nth(size())); }

  ConstIterator cbegin() const noexcept { return Iterator(&nth(0)); }

  ConstIterator cend() const noexcept { return Iterator(&nth(size())); }

  std::size_t size() const noexcept { return size_ - 1; }

  template <typename Iter>
  void rebuild(Iter first, Iter last) noexcept {
    std::size_t size = std::distance(first, last);
    rebuild(first, size);
  }

  template <typename Iter>
  void rebuild(Iter first, std::size_t size) noexcept {
    clear();
    alloc(size);
    populate(first);
  }

  void clear() noexcept {
    delete[] data_;
    size_ = 1;
    data_ = nullptr;
  }

private:
  static constexpr unsigned BLOCK_SIZE = 16;

  ElementTy &nth(std::size_t n) noexcept { return data_[n + 1]; }

  const ElementTy &nth(std::size_t n) const noexcept { return data_[n + 1]; }

  void alloc(std::size_t size) noexcept {
    // First element is ignored for alignment
    if (size == 0) {
      size_ = 1;
      data_ = nullptr;
      return;
    }
    size_ = size + 1;
    data_ = new (std::nothrow) ElementTy[size_];
    AB_ASSERT(data_ != nullptr);
  }

  std::size_t lower_bound(const K &key) const noexcept {
    std::size_t i = 1;
    while (i <= size_) {
      prefetch(data_ + i * BLOCK_SIZE);
      i = 2 * i + (data_[i].first < key);
    }
    return i >> findFirstSet(~i);
  }

  std::size_t find(const K &key) const noexcept {
    unsigned i = 1;
    while (data_[i].first != key) {
      prefetch(data_ + i * BLOCK_SIZE);
      i = 2 * i + (data_[i].first < key);
    }
    return i;
  }

  template <typename Iter>
  void eytzinger(Iter &i, std::size_t k = 1) {
    if (k < size_) {
      eytzinger(i, 2 * k);
      data_[k].first = *i;
      ++i;
      eytzinger(i, 2 * k + 1);
    }
  }

  std::size_t size_;
  ElementTy *data_;
};

template <typename K, typename V>
class EytzingerTree<K, V>::Iterator {
public:
  Iterator(ElementTy *p) : p_(p) {}

  ElementTy &operator*() const noexcept { return *p_; }

  Iterator &operator++() noexcept {
    p_++;
    return *this;
  }

  bool operator!=(const Iterator &other) const noexcept {
    return p_ != other.p_;
  }

private:
  ElementTy *p_;
};

template <typename K, typename V>
class EytzingerTree<K, V>::ConstIterator {
public:
  ConstIterator(ElementTy *p) : p_(p) {}

  const ElementTy &operator*() const noexcept { return *p_; }

  ConstIterator &operator++() noexcept {
    p_++;
    return *this;
  }

  bool operator!=(const Iterator &other) const noexcept {
    return p_ != other.p_;
  }

private:
  const ElementTy *p_;
};

} // namespace ab

#endif // AB_UTIL_EYTZINGER_H