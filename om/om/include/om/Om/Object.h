#ifndef OM_OM_OBJECT_H
#define OM_OM_OBJECT_H

#include <cstdint>
#include <om/Om/ObjectHeader.h>

namespace om::om {

/// This is the most generic representation of objects,
/// providing basically no information.
struct Object {
  Object() = delete;

  ObjectType type() const noexcept { return header.type(); }

  template <typename T>
  T &as() {
    assert(header.type() == T::TYPE);
    return reinterpret_cast<T &>(*this);
  }

  ObjectHeader header;
  std::uint8_t data[0];
};

} // namespace om::om

#endif // OM_OM_OBJECT_H
