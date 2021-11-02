#ifndef OM_OM_ARRAYLAYOUT_H
#define OM_OM_ARRAYLAYOUT_H

#include <cstddef>
#include <om/Om/Layout.h>
#include <om/Om/Type.h>

namespace om::om {

struct ArrayLayout {
  static const ObjectType TYPE = ObjectType::ARRAY_LAYOUT;

  static std::size_t allocSize() noexcept { return sizeof(ArrayLayout); }

  std::size_t size() const noexcept { return ArrayLayout::allocSize(); }

  ObjectHeader header;
  Type elementType;
};

} // namespace om::om

#endif