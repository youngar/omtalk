#ifndef OM_OM_OBJECTHEADER_H
#define OM_OM_OBJECTHEADER_H

#include <cstdint>
#include <om/GC/Heap.h>
#include <om/GC/Ref.h>

namespace om::om {

enum class ObjectType : std::uint8_t {
  STRUCT = 0b001,
  ARRAY = 0b010,
  STRUCT_LAYOUT = 0b101,
  ARRAY_LAYOUT = 0b110,
  META_LAYOUT = 0b111,
};

struct Object;
struct Struct;
struct Array;

// struct Layout;
struct StructLayout;
struct ArrayLayout;
struct LayoutLayout;
class ObjectHeaderProxy;

class ObjectHeader {
public:
  static constexpr std::uintptr_t TYPE_MASK = 0b111;
  static constexpr std::uintptr_t LAYOUT_MASK = ~TYPE_MASK;

  ObjectHeader(ObjectType type, gc::Ref<Object> layout)
      : value(layout.toAddr() | std::uint8_t(type)) {}

  gc::Ref<Object> layout() const noexcept {
    return gc::Ref<Object>::fromAddr(value & LAYOUT_MASK);
  }

  ObjectType type() const noexcept { return ObjectType(value & TYPE_MASK); }

  inline ObjectHeaderProxy proxy() noexcept;

  template <typename C, typename V>
  void walk(C &cx, V &visitor) noexcept;

private:
  friend class ObjectHeaderProxy;

  std::uintptr_t value;
};

static_assert(sizeof(ObjectHeader) == 8);
static_assert(sizeof(ObjectHeader) <= gc::MIN_OBJECT_SIZE);

/// A load/store adapter for the layout pointer in the
/// object header.
class ObjectHeaderProxy {
public:
  ObjectHeaderProxy(ObjectHeader *target) noexcept : target(target) {}

  template <ab::MemoryOrder M>
  gc::Ref<void> load() const noexcept {
    auto value = ab::mem::load<M>(&target->value) & ObjectHeader::LAYOUT_MASK;
    return gc::Ref<void>::fromAddr(value);
  }

  template <ab::MemoryOrder M>
  void store(gc::Ref<void> value) const noexcept {
    auto type = ab::mem::load<M>(&target->value) & ObjectHeader::TYPE_MASK;
    ab::mem::store<M>(&target->value, value.toAddr() | type);
  }

private:
  ObjectHeader *target;
};

inline ObjectHeaderProxy ObjectHeader::proxy() noexcept {
  return ObjectHeaderProxy(this);
}

template <typename C, typename V>
void ObjectHeader::walk(C &cx, V &visitor) noexcept {
  visitor.visit(cx, proxy());
}

} // namespace om::om

#endif // OM_OM_OBJECTHEADER_H
