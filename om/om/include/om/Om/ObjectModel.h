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
};

} // namespace om::om

#endif