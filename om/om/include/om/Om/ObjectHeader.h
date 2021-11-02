#ifndef OM_OM_OBJECTHEADER_H
#define OM_OM_OBJECTHEADER_H

#include <cstdint>

namespace om::om {

enum class ObjectType {
  STRUCT,
  ARRAY,
};

struct ObjectHeader {
  ObjectType type() const { return ObjectType::STRUCT; }

  std::uintptr_t value;
};

} // namespace om::om

#endif // OM_OM_OBJECTHEADER_H
