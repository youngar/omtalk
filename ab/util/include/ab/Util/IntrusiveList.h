#ifndef AB_UTIL_INTRUSIVELIST_H
#define AB_UTIL_INTRUSIVELIST_H

#include <cstddef>
#include <iterator>

namespace ab {

template <typename T>
class IntrusiveListNode;

template <typename T, typename Accessor>
class IntrusiveList;

template <typename T, typename Accessor>
class IntrusiveListIterator;

template <typename T, typename Accessor>
class IntrusiveListConstIterator;

/// Stores the next and previous pointers for elements in an IntrusiveList. Each
/// element in an intrusive list must contain an IntrusiveListNode<T>. The Node
/// is obtained via an accessor class. See
template <typename T>
class IntrusiveListNode {
public:
  IntrusiveListNode() noexcept : prev(nullptr), next(nullptr) {}

  IntrusiveListNode(T *p, T *n) noexcept : prev(p), next(n) {}

  /// assign previous, next to node.
  void assign(T *p, T *n) noexcept {
    prev = p;
    next = n;
  }

  // deactivate the node, and clear the next/prev pointers.
  void clear() noexcept {
    prev = nullptr;
    next = nullptr;
  }

  T *prev;
  T *next;
};

/// The IntrusiveListNodeAccessor is the default accessor used to obtain an
/// InstrusiveListNode from a list element. This template can be specialized to
/// set the default accessor for a type.
template <typename T>
struct IntrusiveListNodeAccessor {
  using Node = IntrusiveListNode<T>;

  /// Obtain the IntrusiveListNode from an element. By default, calls
  /// element.node().
  static Node &node(T &element) noexcept { return element.getListNode(); }

  /// Obtain a constant node from an element.
  static const Node &node(const T &element) noexcept {
    return element.getListNode();
  }
};

/// Simple bidirectional iterator for the elements in an intrusive list.
template <typename T, typename Accessor = IntrusiveListNodeAccessor<T>>
class IntrusiveListIterator {
public:
  using difference_type = std::size_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::bidirectional_iterator_tag;

  IntrusiveListIterator() noexcept : _current(nullptr) {}

  explicit IntrusiveListIterator(T *root) noexcept : _current(root) {}

  IntrusiveListIterator(const IntrusiveListIterator<T, Accessor> &rhs) noexcept
      : _current(rhs.current()) {}

  T &operator*() const noexcept { return *_current; }

  T *operator->() const noexcept { return _current; }

  IntrusiveListIterator<T, Accessor> operator+(difference_type rhs) noexcept {
    auto copy = *this;
    for (difference_type i = 0; i < rhs; i++) {
      ++copy;
    }
    return copy;
  }

  IntrusiveListIterator<T, Accessor> &operator++() noexcept {
    _current = Accessor::node(*_current).next;
    return *this;
  }

  IntrusiveListIterator<T, Accessor> operator++(int) noexcept {
    IntrusiveListIterator<T, Accessor> copy = *this;
    ++(*this);
    return copy;
  }

  IntrusiveListIterator<T, Accessor> &operator--() noexcept {
    _current = Accessor::node(*_current).prev;
    return *this;
  }

  IntrusiveListIterator<T, Accessor> operator--(int) noexcept {
    IntrusiveListIterator<T, Accessor> copy = *this;
    _current = Accessor::node(*_current).prev;
    return copy;
  }

  bool
  operator==(const IntrusiveListIterator<T, Accessor> &rhs) const noexcept {
    return _current == rhs._current;
  }

  bool
  operator!=(const IntrusiveListIterator<T, Accessor> &rhs) const noexcept {
    return _current != rhs._current;
  }

  IntrusiveListIterator<T, Accessor> &
  operator=(const IntrusiveListIterator<T, Accessor> &rhs) noexcept {
    _current = rhs._current;
    return *this;
  }

  T *current() const noexcept { return _current; }

private:
  T *_current;
};

/// Simple bidirectional iterator for the elements in a constant intrusive list.
template <typename T, typename Accessor = IntrusiveListNodeAccessor<T>>
class IntrusiveListConstIterator {
public:
  using difference_type = std::size_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::bidirectional_iterator_tag;

  IntrusiveListConstIterator() noexcept : _current(nullptr) {}

  explicit IntrusiveListConstIterator(T *root) noexcept : _current(root) {}

  IntrusiveListConstIterator(
      const IntrusiveListConstIterator<T, Accessor> &rhs) noexcept
      : _current(rhs.current()) {}

  IntrusiveListConstIterator(
      const IntrusiveListIterator<T, Accessor> &rhs) noexcept
      : _current(rhs.current()) {}

  const T &operator*() const noexcept { return *_current; }

  const T *operator->() const noexcept { return _current; }

  IntrusiveListConstIterator<T, Accessor>
  operator+(difference_type rhs) noexcept {
    auto copy = *this;
    for (difference_type i = 0; i < rhs; i++) {
      ++copy;
    }
    return copy;
  }

  IntrusiveListConstIterator<T, Accessor> &operator++() noexcept {
    _current = Accessor::node(*_current).next;
    return *this;
  }

  IntrusiveListConstIterator<T, Accessor> operator++(int) noexcept {
    IntrusiveListConstIterator<T, Accessor> copy = *this;
    ++this;
    return copy;
  }

  IntrusiveListConstIterator<T, Accessor> &operator--() noexcept {
    _current = Accessor::node(*_current).prev;
    return *this;
  }

  IntrusiveListConstIterator<T, Accessor> operator--(int) noexcept {
    IntrusiveListConstIterator<T, Accessor> copy = *this;
    _current = Accessor::node(*_current).prev;
    return copy;
  }

  bool
  operator==(const IntrusiveListIterator<T, Accessor> &rhs) const noexcept {
    return _current == rhs.current();
  }

  bool
  operator!=(const IntrusiveListIterator<T, Accessor> &rhs) const noexcept {
    return _current != rhs.current();
  }

  bool operator==(
      const IntrusiveListConstIterator<T, Accessor> &rhs) const noexcept {
    return _current == rhs._current;
  }

  bool operator!=(
      const IntrusiveListConstIterator<T, Accessor> &rhs) const noexcept {
    return _current != rhs._current;
  }

  IntrusiveListConstIterator<T, Accessor> &
  operator=(const IntrusiveListConstIterator<T, Accessor> &rhs) noexcept {
    _current = rhs._current;
    return *this;
  }

  const T *current() const noexcept { return _current; }

private:
  const T *_current;
};

template <typename T, typename A>
bool operator==(const IntrusiveListIterator<T, A> &lhs,
                const IntrusiveListConstIterator<T, A> &rhs) noexcept {
  return rhs.current() == lhs.current();
}

template <typename T, typename A>
bool operator!=(const IntrusiveListIterator<T, A> &lhs,
                const IntrusiveListConstIterator<T, A> &rhs) noexcept {
  return rhs.current() != lhs.current();
}

/// A doubly linked linear list, where the list node is embedded in the element.
///
/// To use an intrusive list, the element type T must store an
/// IntrusiveListNode<T>. The default accessor will use T's node()
/// member-function to access the list node.
///
/// The Intrusive list provides two mechanisms for overriding the default node
/// accessor:
///  1. Specialize the IntrusiveListNodeAccessor<T> template.
///  2. Override the Accessor template parameter in the list.
///
template <typename T, typename Accessor = IntrusiveListNodeAccessor<T>>
class IntrusiveList {
public:
  using Node = IntrusiveListNode<T>;
  using Iterator = IntrusiveListIterator<T, Accessor>;
  using ConstIterator = IntrusiveListConstIterator<T, Accessor>;

  explicit IntrusiveList() noexcept : _root(nullptr) {}

  /// Get the front element of the list.
  T &front() const noexcept { return *_root; }

  /// Get the back element of the list.
  T &back() const noexcept {
    Iterator i = begin();
    while ((i + 1) != end()) {
      ++i;
    }
    return *i;
  }

  /// Add element to the head of the list. Constant time.
  void push_front(T *element) noexcept {
    Accessor::node(*element).assign(nullptr, _root);
    if (_root) {
      Accessor::node(*_root).prev = element;
    }
    _root = element;
  }

  /// Remove element from the list. Removing an element invalidates any
  /// iterators. Constant time.
  void remove(T *element) noexcept {
    Node &node = Accessor::node(*element);
    if (element == _root) {
      if (node.next != nullptr) {
        Accessor::node(*node.next).prev = nullptr;
        _root = node.next;
      } else {
        _root = nullptr;
      }
    } else {
      Accessor::node(*node.prev).next = node.next;
      if (node.next != nullptr) {
        Accessor::node(*node.next).prev = node.prev;
      }
    }
    node.clear();
  }

  /// Move elements from the other list to the start of this list. Does not
  /// invalidate any iterators.
  void splice(IntrusiveList &other) noexcept {

    if (other._root == nullptr) {
      return;
    }

    if (_root == nullptr) {
      _root = other._root;
      other._root = nullptr;
      return;
    }

    Node &otherBack = Accessor::node(other.back());
    otherBack.next = _root;
    _root = other._root;
    other._root = nullptr;
  }

  std::size_t size() noexcept { return std::distance(begin(), end()); }

  Iterator begin() const noexcept { return Iterator(_root); }

  Iterator end() const noexcept { return Iterator(nullptr); }

  ConstIterator cbegin() const noexcept { return ConstIterator(_root); }

  ConstIterator cend() const noexcept { return ConstIterator(nullptr); }

  bool empty() const noexcept { return _root == nullptr; }

private:
  T *_root;
};

} // namespace ab

#endif // AB_UTIL_INTRUSIVELIST_H
