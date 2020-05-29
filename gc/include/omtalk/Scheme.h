#ifndef OMTALK_SCHEME_H
#define OMTALK_SCHEME_H

#include <cstdint>

namespace omtalk::gc {

//===----------------------------------------------------------------------===//
// Scheme Type Accessors
//===----------------------------------------------------------------------===//

template <typename S>
using Object = typename S::Object;

template <typename S>
using ObjectRef = typename S::ObjectRef;

template <typename S>
using ObjectProxy = typename S::ObjectProxy;

template <typename S>
using RootWalker = typename S::RootWalker;

//===----------------------------------------------------------------------===//
// Object Proxy Construction
//===----------------------------------------------------------------------===//

/// overrideable functor for converting an address to an object proxy.
template <typename S>
struct GetProxy {
  ObjectProxy<S> operator()(Ref<void> target) const noexcept {
    return ObjectProxy<S>(target);
  }
};

/// Convert a heap address to an object proxy.
template <typename S>
ObjectProxy<S> getProxy(Ref<void> target) noexcept {
  return GetProxy<S>()(target);
}

//===----------------------------------------------------------------------===//
// Generic Get Object Size Function
//===----------------------------------------------------------------------===//

/// Overrideable functor which gets the size of an object at a given object.
template <typename S>
struct GetSize {
  std::size_t operator()(Ref<void> target) const noexcept {
    return getProxy<S>(target).getSize();
  }
};

/// Get the size of an object located at a given address.
template <typename S>
std::size_t getSize(Ref<void> target) noexcept {
  return GetSize<S>()(target);
}

//===----------------------------------------------------------------------===//
// Generic Walk Function
//===----------------------------------------------------------------------===//

/// Overrideable functor which walks the slots of an object.
template <typename S>
struct Walk {
  template <typename ContextT, typename VisitorT>
  void operator()(ContextT &cx, ObjectProxy<S> target,
                  VisitorT &visitor) const noexcept {
    target.walk(cx, visitor);
  }
};

/// Walk the slots of a function
template <typename S, typename ContextT, typename VisitorT>
void walk(ContextT &cx, ObjectProxy<S> target, VisitorT &visitor) noexcept {
  Walk<S>()(cx, target, visitor);
}

} // namespace omtalk::gc

#endif