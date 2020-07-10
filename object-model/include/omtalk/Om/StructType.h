#ifndef OMTALK_OM_STRUCTTYPE_H
#define OMTALK_OM_STRUCTTYPE_H

#include <omtalk/Om/ObjectModel.h>

namespace omtalk {

class StructShapeBuilder {
public:
  StructShapeBuilder() = default;
  StructShapeBuilder &i8(Id id) {
    i8Slots().push_back(id);
    return *this;
  }
  StructShapeBuilder &i16(Id id) {
    i16Slots().push_back(id);
    return *this;
  }
  StructShapeBuilder &i32(Id id) {
    i32Slots().push_back(id);
    return *this;
  }
  StructShapeBuilder &i64(Id id) {
    i64Slots().push_back(id);
    return *this;
  }
  StructShapeBuilder &f32(Id id) {
    f32Slots().push_back(id);
    return *this;
  }
  StructShapeBuilder &f64(Id id) {
    f64Slots().push_back(id);
    return *this;
  }
  StructShapeBuilder &ref(Id id) {
    refSlots().push_back(id);
    return *this;
  }
  StructShapeBuilder &ply(Id id) {
    refSlots().push_back(id);
    return *this;
  }
  StructShape *build() {
    auto offset = 0;
    for (auto)
      for (id : plySlots) {
      }
  }
  std::size_t nslots() const {
    auto count = std::size_t(0);
    for (auto &slots : table) {
      count += slots.size();
    }
    return count;
  }
  std::vector<Id> &i8Slots() noexcept { return table[int(CoreType::I8)]; }
  std::vector<Id> &i16Slots() noexcept { return table[int(CoreType::I16)]; }
  std::vector<Id> &i32Slots() noexcept { return table[int(CoreType::I32)]; }
  std::vector<Id> &i64Slots() noexcept { return table[int(CoreType::I64)]; }
  std::vector<Id> &f32Slots() noexcept { return table[int(CoreType::f32)]; }
  std::vector<Id> &f64Slots() noexcept { return table[int(CoreType::f64)]; }
  std::vector<Id> &refSlots() noexcept { return table[int(CoreType::REF)]; }
  std::vector<Id> &plySlots() noexcept { return table[int(CoreType::PLY)]; }
  std::array<std::vector<Id, CORE_TYPE_COUNT>> table;
};

} // namespace omtalk

#endif
