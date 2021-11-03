#ifndef OM_OM_ARRAY_H
#define OM_OM_ARRAY_H

#include <cstdint>
#include <om/Om/ArrayLayout.h>
#include <om/Om/ObjectHeader.h>
#include <om/Om/SlotProxy.h>

namespace om::om {

struct Array {
  static const ObjectType TYPE = ObjectType::ARRAY;

  static std::size_t allocSize(Type type, std::size_t length) noexcept {
    return getSize(type) * length;
  }

  explicit Array(gc::Ref<ArrayLayout> layout, std::size_t length)
      : header(TYPE, layout.reinterpret<Object>()), length(length) {}

  ArrayLayout &layout() const noexcept {
    return header.layout()->as<ArrayLayout>();
  }

  Type elementType() const noexcept { return layout().elementType; }

  std::size_t size() const noexcept { return allocSize(elementType(), length); }

  template <typename ContextT, typename VisitorT>
  void walk(ContextT &context, VisitorT &visitor) noexcept {
    header.walk(context, visitor);
    if (elementType() == Type::ref) {
      std::uintptr_t *ptr = data;
      std::uintptr_t *end = data + length;
      while (ptr < end) {
        visitor.visit(context, SlotProxy::fromPtr(ptr));
        ++ptr;
      }
    }
  }

  ObjectHeader header;
  std::size_t length;
  std::uintptr_t data[0];
};

} // namespace om::om

#endif