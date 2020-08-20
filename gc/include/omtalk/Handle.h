#ifndef OMTALK_GC_HANDLE_HPP_
#define OMTALK_GC_HANDLE_HPP_

#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>
#include <vector>

namespace omtalk::gc {

///
/// TODO: create a handle which is guaranteed to point into old space.
/// It will not need to be scavenged during a local GC.
///

class HandleBase;

template <typename T>
class Handle;

/// Stack allocated Handle manager.  When HandleScope goes out of scope,
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
/// of the root set.  This will scan all interier HandleScopes.
class RootHandleScope : public HandleScope {
public:
  using Iterator = std::vector<HandleBase *>::iterator;
  using ConstIterator = std::vector<HandleBase *>::const_iterator;

  RootHandleScope() : HandleScope() {
    oldSize = 0;
    data = &handleData;
  }

  Iterator begin() noexcept { return handleData.begin(); }

  ConstIterator begin() const noexcept { return handleData.begin(); }

  Iterator end() noexcept { return handleData.end(); }

  ConstIterator end() const noexcept { return handleData.end(); }

  ConstIterator cbegin() const noexcept { return handleData.begin(); }

  ConstIterator cend() noexcept { return handleData.end(); }

private:
  std::vector<HandleBase *> handleData;
};

class HandleBase {
public:
  Ref<void> load() const noexcept { return value; }

  void store(Ref<void> address) noexcept { value = address; }

protected:
  HandleBase(Ref<void> value) : value(value) {}

  Ref<void> value;
};

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

  T load() const noexcept { return *value.reinterpret<T>(); }

  Ref<T> get() const noexcept { return value.reinterpret<T>(); }
};

template <>
class Handle<void> final : public HandleBase {
public:
  Handle(HandleScope &scope, std::nullptr_t) : HandleBase(nullptr) {}

  Handle(HandleScope &scope, void *value) : HandleBase(value) {}

  Handle(HandleScope &scope, Ref<void> value) : HandleBase(value) {}

  Ref<void> get() const noexcept { return value; }
};

/// Handle proxy.  Allows the garbage collector to scan handles and return
/// language specific ObjectProxy
template <typename S>
class HandleProxy {
public:
  HandleProxy(HandleBase *handle) : handle(handle) {}

  ObjectProxy<S> load() const noexcept {
    return ObjectProxy<S>(handle->load());
  }

  void store(ObjectProxy<S> object) const noexcept {
    handle->store(object->asRef());
  }

  Ref<void> loadRef() const noexcept { return handle->load(); }

  void storeRef(Ref<void> ref) const noexcept { return handle->store(ref); }

private:
  HandleBase *handle;
};

} // namespace omtalk::gc

#endif // OMTALK_GC_HANDLE_HPP_
