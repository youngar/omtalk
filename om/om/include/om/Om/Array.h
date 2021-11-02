#ifndef OM_OM_ARRAY_H
#define OM_OM_ARRAY_H

#include <cstdint>
#include <om/Om/ArrayLayout.h>
#include <om/Om/ObjectHeader.h>

namespace om::om {

struct Array {
  static const ObjectType TYPE = ObjectType::ARRAY;

  static std::size_t allocSize(Type type, std::size_t length) noexcept {
    return getSize(type) * length;
  }

  ArrayLayout &layout() const noexcept {
    return header.layout()->as<ArrayLayout>();
  }

  Type elementType() const noexcept { return layout().elementType; }

  std::size_t size() const noexcept { return allocSize(elementType(), length); }

  ObjectHeader header;
  std::size_t length;
  std::uint8_t data;
};

} // namespace om::om

#endif