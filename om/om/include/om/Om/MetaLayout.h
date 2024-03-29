#ifndef OM_OM_METALAYOUT_H
#define OM_OM_METALAYOUT_H

#include <om/Om/ObjectHeader.h>

namespace om::om {

struct MetaLayout {
  static const ObjectType TYPE = ObjectType::META_LAYOUT;

  static std::size_t allocSize() noexcept { return sizeof(MetaLayout); }

  MetaLayout() : header(TYPE, gc::Ref<Object>::fromPtr(this)) {}

  std::size_t size() const noexcept { return MetaLayout::allocSize(); }

  template <typename ContextT, typename VisitorT>
  void walk(ContextT &context, VisitorT &visitor) noexcept {
    header.walk(context, visitor);
  }

  ObjectHeader header;
};

} // namespace om::om

#endif // OM_OM_METALAYOUT_H
