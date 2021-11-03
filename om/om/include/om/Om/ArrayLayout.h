#ifndef OM_OM_ARRAYLAYOUT_H
#define OM_OM_ARRAYLAYOUT_H

#include <cstddef>
#include <om/Om/Layout.h>
#include <om/Om/Type.h>

namespace om::om {

struct ArrayLayout {
  static const ObjectType TYPE = ObjectType::ARRAY_LAYOUT;

  static std::size_t allocSize() noexcept { return sizeof(ArrayLayout); }

  ArrayLayout(gc::Ref<Object> layout, Type elementType)
      : header(TYPE, layout), elementType(elementType) {}

  std::size_t size() const noexcept { return ArrayLayout::allocSize(); }

  template <typename ContextT, typename VisitorT>
  void walk(ContextT &context, VisitorT &visitor) noexcept {
    header.walk(context, visitor);
  }

  ObjectHeader header;
  Type elementType;
};

} // namespace om::om

#endif