#ifndef OM_GC_FORWARD_H
#define OM_GC_FORWARD_H

#include <om/GC/Ref.h>
#include <om/GC/Scheme.h>

namespace om::gc {

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

template <typename S>
void isForwarded(GlobalCollectorContext<S> &context, ObjectProxy<S> from) {
  auto fromRegion = Region::get(from);
  fromRegion.getForwardingMap().insert
}

template <typename S>
void forward(GlobalCollectorContext<S> &context, ObjectProxy<S> from,
             ObjectProxy<S> to) {
  auto fromRegion = Region::get(from);
  fromRegion.getForwardingMap().insert(from, to);
}

template <typename S>
struct Forward {
  void operator()(GlobalCollectorContext<S> context, ObjectProxy<S> from,
                  std::byte *to, std::size_t size) const noexcept {
    std::size_t copySize = from.getCopySize();
    assert(copySize = from.getSize());
    if (copySize > size) {
      return CopyResult::fail();
    }
    std::memcpy(to, from.asRef().get(), from.getSize());
    return CopyResult::success(copySize);
  }
};

template <typename S>
void copy(GlobalCollectorContext<S> &context, ObjectProxy<S> from,
          std::byte *to, std::size_t size) noexcept {
  return Copy<S>()(context, from, to, size);
}

} // namespace om::gc

#endif