#ifndef OM_OM_OBJECTMODEL_H
#define OM_OM_OBJECTMODEL_H

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

/// This class groups methods for working with objects that
/// are polymorphic / powered by reflection.
struct ObjectModel {
  static std::size_t size(gc::Ref<Object> obj) {
    switch (obj->type()) {
    case ObjectType::STRUCT:
      return obj->as<Struct>().size();
    case ObjectType::ARRAY:
      return obj->as<Array>().size();
    case ObjectType::STRUCT_LAYOUT:
      return obj->as<StructLayout>().size();
    case ObjectType::ARRAY_LAYOUT:
      return obj->as<ArrayLayout>().size();
    case ObjectType::META_LAYOUT:
      return obj->as<MetaLayout>().size();
    }
  }

  template <typename C, typename V>
  void walk(C &cx, gc::Ref<Object> obj, V &visitor) noexcept {
    switch (obj->type()) {
    case ObjectType::STRUCT:
      return obj->as<Struct>().walk(cx, visitor);
    case ObjectType::ARRAY:
      return obj->as<Array>().walk(cx, visitor);
    case ObjectType::STRUCT_LAYOUT:
      return obj->as<StructLayout>().walk(cx, visitor);
    case ObjectType::ARRAY_LAYOUT:
      return obj->as<ArrayLayout>().walk(cx, visitor);
    case ObjectType::META_LAYOUT:
      return obj->as<MetaLayout>().walk(cx, visitor);
    }
  }
};

} // namespace om::om

#endif