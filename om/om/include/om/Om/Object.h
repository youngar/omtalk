#ifndef OM_OM_OBJECT_H
#define OM_OM_OBJECT_H

#include <cstdint>
#include <om/Om/ObjectHeader.h>

namespace om::om {

/// This is the most generic representation of objects,
/// providing basically no information about the data.
struct Object {
  ObjectHeader header;
  std::uint8_t data[0];
};

} // namespace om::om

#endif // OM_OM_OBJECT_H
