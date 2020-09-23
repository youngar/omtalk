#ifndef OMTALK_HANDLE_H
#define OMTALK_HANDLE_H

#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>
#include <vector>

namespace omtalk::gc {

/// Handles are a helper mechanism to assist language writers with keeping
/// references to on heap objects from C++ code. Handles are thin wrappers
/// around pointers that are tracked by a HandleScope.  A HandleScope allows the
/// language to provide a list of on stack references for the purposes of root
/// walking.
///
/// The garbage collector is not automatically aware of HandleScopes, and must
/// be manually walked during root walking.

/// TODO: create a handle which is guaranteed to point into old space.
/// It will not need to be scavenged during a local GC.

class HandleBase;

template <typename T>
class Handle;

/// Stack allocated Handle manager.  When a HandleScope goes out of scope,
/// all Handles created by it, or an inner scope, become invalid.  When a Handle
/// becomes invalid, the garbage collector will not find the Handle, and the
/// object could be free'd or moved.
class HandleScope {
  friend class HandleBase;
  template <typename T>
  friend class Handle;

public:
  HandleScope createScope() { return HandleScope(*this); }

  ~HandleScope() { data->resize(oldSize); }

  std::size_t handleCount() { return data->size(); }

protected:
  HandleScope() {}

  HandleScope(HandleScope &other)
      : oldSize(other.data->size()), data(other.data) {}

  HandleScope(std::vector<HandleBase *> *data)
      : oldSize(data->size()), data(data) {}

  void attach(HandleBase *handle) { data->push_back(handle); }

  std::size_t oldSize;
  std::vector<HandleBase *> *data;
};

/// The outer most HandleScope.  A RootHandleScope should be scanned as a part
/// of the root set.  This will scan all interior HandleScopes.
class RootHandleScope : public HandleScope {
public:
  using Iterator = std::vector<HandleBase *>::iterator;
  using ConstIterator = std::vector<HandleBase *>::const_iterator;

  RootHandleScope() : HandleScope() {
    oldSize = 0;
    data = &handles;
  }

  Iterator begin() noexcept { return handles.begin(); }

  ConstIterator begin() const noexcept { return handles.begin(); }

  Iterator end() noexcept { return handles.end(); }

  ConstIterator end() const noexcept { return handles.end(); }

  ConstIterator cbegin() const noexcept { return handles.begin(); }

  ConstIterator cend() noexcept { return handles.end(); }

  template <typename V, typename... Ts>
  void walk(V &visitor, Ts &&... xs);

private:
  std::vector<HandleBase *> handles;
};

/// A container class holding a single Ref<void>.
/// Implements the Slot concept.
class HandleBase {
public:
  template <MemoryOrder M>
  Ref<void> load() const noexcept {
    return value.load<M>();
  }

  template <MemoryOrder M>
  void store(Ref<void> x) noexcept {
    value.store<M>(x);
  }

  template <typename V, typename... Ts>
  void walk(V &visitor, Ts &&... xs) {
    visitor.visit(value.proxy(), std::forward<Ts>(xs)...);
  }

protected:
  HandleBase(Ref<void> value) : value(value) {}

  Ref<void> value;
};

template <typename V, typename... Ts>
void RootHandleScope::walk(V &visitor, Ts &&... xs) {
  for (auto handle : handles) {
    handle->walk(visitor, std::forward<Ts>(xs)...);
  }
}

/// GC safe object pointer.  Handles are tracked by their HandleScope, and are
/// traced during garbage collection.  This ensures that the object pointed to
/// by a Handle is not collected, and the Handle will always point to a
/// valid object in case the object moves.
template <typename T = void>
class Handle final : public HandleBase {
public:
  Handle(HandleScope &scope, std::nullptr_t) : HandleBase(nullptr) {
    scope.attach(this);
  }

  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U *, T *>>>
  Handle(HandleScope &scope, U *value) : HandleBase(value) {
    scope.attach(this);
  }

  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U *, T *>>>
  Handle(HandleScope &scope, Ref<U> value) : HandleBase(value) {
    scope.attach(this);
  }

  T &operator*() const noexcept { return *value.reinterpret<T>(); }

  T *operator->() const noexcept { return value.reinterpret<T>().get(); }

  Ref<T> get() const noexcept { return value.reinterpret<T>(); }

  Handle<T> &operator=(const Handle<T> &other) noexcept {
    value = other.value;
    return *this;
  }

  Handle<T> &operator=(const Ref<T> &other) noexcept {
    value = other;
    return *this;
  }
};

template <>
class Handle<void> final : public HandleBase {
public:
  Handle(HandleScope &scope, std::nullptr_t) : HandleBase(nullptr) {}

  Handle(HandleScope &scope, void *value) : HandleBase(value) {}

  Handle(HandleScope &scope, Ref<void> value) : HandleBase(value) {}

  Ref<void> get() const noexcept { return value; }

  Handle<void> &operator=(const Handle<void> &other) noexcept {
    value = other.value;
    return *this;
  }

  Handle<void> &operator=(const Ref<void> &other) noexcept {
    value = other;
    return *this;
  }
};

} // namespace omtalk::gc

#endif