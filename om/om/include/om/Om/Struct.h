#ifndef OM_OM_STRUCTTYPE_H
#define OM_OM_STRUCTTYPE_H

#include <cstddef>
#include <cstdint>
#include <om/Om/ObjectHeader.h>
#include <om/Om/StructLayout.h>
#include <om/Om/Type.h>

namespace om::om {

struct Struct {
  static const ObjectType TYPE = ObjectType::STRUCT;

  StructLayout &layout() const noexcept {
    return header.layout()->as<StructLayout>();
  }

  std::size_t size() { return layout().instanceSize; }

  ObjectHeader header;
  std::uint8_t data[];
};

} // namespace om::om

#endif
