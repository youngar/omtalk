#ifndef OM_OM_STRUCTBUILDER_H
#define OM_OM_STRUCTBUILDER_H

#include <om/Om/ObjectModel.h>
#include <om/Om/Type.h>
#include <om/Util/Bytes.h>

namespace om::om {

/// A simple struct builder that adds padding to ensure alignment,
/// but respects the order of fields.
class StructBuilder {
public:
  struct Field {
    std::size_t offset;
    Type type;
  };

  StructBuilder() : offset(0), fields() {}

  template <Type T>
  StructBuilder &add() {
    offset = align(offset, alignment<T>);
    fields.push_back({offset, T});
    offset = offset + size<T>;
    return *this;
  }

  StructBuilder &i8() { return add<Type::i8>(); }
  StructBuilder &i16() { return add<Type::i16>(); }
  StructBuilder &i32() { return add<Type::i32>(); }
  StructBuilder &i64() { return add<Type::i64>(); }
  StructBuilder &f32() { return add<Type::f32>(); }
  StructBuilder &f64() { return add<Type::f64>(); }
  StructBuilder &ref() { return add<Type::ref>(); }
  StructBuilder &ply() { return add<Type::ply>(); }

private:
  std::size_t offset;
  std::vector<Field> fields;
};

} // namespace om::om

#endif
