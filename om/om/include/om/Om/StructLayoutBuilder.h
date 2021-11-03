#ifndef OM_OM_STRUCTBUILDER_H
#define OM_OM_STRUCTBUILDER_H

#include <ab/Util/Bytes.h>
#include <om/Om/ObjectModel.h>
#include <om/Om/StructLayout.h>
#include <om/Om/Type.h>

namespace om::om {

/// A simple struct builder that adds padding to ensure alignment,
/// but respects the order of fields.
class StructLayoutBuilder {
public:
  struct Field {
    std::size_t offset;
    Type type;
  };

  StructLayoutBuilder() : offset(0), fields() {}

  template <Type T>
  StructLayoutBuilder &add() {
    offset = align(offset, alignment<T>);
    fields.push_back({offset, T});
    offset = offset + size<T>;
    return *this;
  }

  StructLayoutBuilder &i8() { return add<Type::i8>(); }
  StructLayoutBuilder &i16() { return add<Type::i16>(); }
  StructLayoutBuilder &i32() { return add<Type::i32>(); }
  StructLayoutBuilder &i64() { return add<Type::i64>(); }
  StructLayoutBuilder &f32() { return add<Type::f32>(); }
  StructLayoutBuilder &f64() { return add<Type::f64>(); }
  StructLayoutBuilder &ref() { return add<Type::ref>(); }
  StructLayoutBuilder &ply() { return add<Type::ply>(); }

  const std::vector<Field> &build() { return fields; }

private:
  std::size_t offset;
  std::vector<Field> fields;
};

} // namespace om::om

#endif
