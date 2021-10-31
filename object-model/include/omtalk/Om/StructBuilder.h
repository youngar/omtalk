#ifndef OMTALK_OM_STRUCTBUILDER_H
#define OMTALK_OM_STRUCTBUILDER_H

#include <omtalk/Om/CoreType.h>
#include <omtalk/Om/ObjectModel.h>

namespace omtalk::om {

/// A simple struct builder that adds padding to ensure alignment,
/// but respects the order of fields.
class StructBuilder {
public:
  struct Field {
    std::size_t offset;
    Tag type;
  };

  StructBuilder() : offset(0), fields() {}

  template <typename T>
  StructBuilder &add() {
    static_assert(is_known_type<T>, "T must be a known type");
    offset = align(offset, std::alignment_of<T>);
    fields.push_back({offset, T});
    offset = offset + core_type::prop::size<T>;
    return *this;
  }

  StructBuilder &i8() { return add<Tag::i8>(); }
  StructBuilder &i16() { return add<Tag::i16>(); }
  StructBuilder &i32() { return add<Tag::i32>(); }
  StructBuilder &i64() { return add<Tag::i64>(); }
  StructBuilder &f32() { return add<Tag::f32>(); }
  StructBuilder &f64() { return add<Tag::f64>(); }
  StructBuilder &ref() { return add<Tag::ref>(); }
  StructBuilder &ply() { return add<Tag::ply>(); }

private:
  std::size_t offset;
  std::vector<Field> fields;
};

} // namespace omtalk::om

#endif
