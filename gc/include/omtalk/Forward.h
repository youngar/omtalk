#ifndef OMTALK_FORWARD_H
#define OMTALK_FORWARD_H

#include <omtalk/Scheme.h>
#include <omtalk/Ref.h>

namespace omtalk::gc {

/// Represents a forwarded object. A forwarded object is one that used to be
/// at this address but has moved to another address.  This record is left
/// behind to provide a forwarding address to the object's new location.
class ForwardedObject {
public:
  /// Place a forwarded object at an address.  This will forward to a new
  /// address.
  static Ref<ForwardedObject> create(Ref<void> address, Ref<void> to) noexcept {
    return new (address.get()) ForwardedObject(to);
  }

  /// Get a ForwardedObject which already exists at an address.
  static Ref<ForwardedObject> at(Ref<void> address) noexcept {
    return address.reinterpret<ForwardedObject>();
  }

  /// Get the forwarding address.  This is the address that the object has moved
  /// to.
  Ref<void> getForwardedAddress() const noexcept { return to; }

private:
  ForwardedObject(Ref<void> to) : to(to) {}
  Ref<void> to;
};

template<typename S>
void forward(ObjectProxy<S> from, ObjectProxy<S> to) {

}

} // namespace omtalk::gc

#endif