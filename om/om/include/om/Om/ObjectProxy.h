#ifndef OM_OM_OBJECTPROXY_H
#define OM_OM_OBJECTPROXY_H

#include <cstdint>
#include <om/Om/Array.h>
#include <om/Om/ArrayLayout.h>
#include <om/Om/Layout.h>
#include <om/Om/MetaLayout.h>
#include <om/Om/Object.h>
#include <om/Om/Struct.h>
#include <om/Om/StructLayout.h>
#include <vector>

namespace om::om {

struct ObjectProxy {
  constexpr explicit ObjectProxy(gc::Ref<void> target) noexcept
      : target(target.reinterpret<Object>()) {}

  gc::Ref<void> asRef() const noexcept { return target.reinterpret<void>(); }

  constexpr operator void *() const noexcept { return target.get(); }

  std::size_t getSize() const noexcept {
    switch (target->type()) {
    case ObjectType::STRUCT:
      return target->as<Struct>().size();
    case ObjectType::ARRAY:
      return target->as<Array>().size();
    case ObjectType::STRUCT_LAYOUT:
      return target->as<StructLayout>().size();
    case ObjectType::ARRAY_LAYOUT:
      return target->as<ArrayLayout>().size();
    case ObjectType::META_LAYOUT:
      return target->as<MetaLayout>().size();
    }
  }

  template <typename VisitorT, typename... Args>
  void walk(VisitorT &visitor, Args... args) const noexcept {
    switch (target->type()) {
    case ObjectType::STRUCT:
      return target->as<Struct>().walk(visitor, args...);
    case ObjectType::ARRAY:
      return target->as<Array>().walk(visitor, args...);
    case ObjectType::STRUCT_LAYOUT:
      return target->as<StructLayout>().walk(visitor, args...);
    case ObjectType::ARRAY_LAYOUT:
      return target->as<ArrayLayout>().walk(visitor, args...);
    case ObjectType::META_LAYOUT:
      return target->as<MetaLayout>().walk(visitor, args...);
    }
  }

private:
  gc::Ref<Object> target;
};

} // namespace om::om

#endif