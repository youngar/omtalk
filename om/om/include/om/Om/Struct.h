#ifndef OM_OM_STRUCTTYPE_H
#define OM_OM_STRUCTTYPE_H

#include <cstddef>
#include <cstdint>
#include <om/Om/ObjectHeader.h>
#include <om/Om/SlotProxy.h>
#include <om/Om/StructLayout.h>
#include <om/Om/Type.h>

namespace om::om {

class StructSlotSpan {
public:
  class Iterator {
  public:
    Iterator(const SlotDecl *decl, std::uint8_t *slot)
        : decl(decl), slot(slot) {}

    SlotProxy operator*() const noexcept { return SlotProxy::fromPtr(slot); }

    bool operator==(const Iterator &rhs) const { return decl == rhs.decl; }

    bool operator!=(const Iterator &rhs) const { return decl != rhs.decl; }

  public:
    const SlotDecl *decl;
    std::uint8_t *slot;
  };

  StructSlotSpan(SlotDeclSpan decls, std::uint8_t *base)
      : decls(decls), base(base) {}

  Iterator begin() const noexcept { return Iterator(decls.begin(), base); }

  Iterator end() const noexcept { return Iterator(decls.end(), nullptr); }

private:
  const SlotDeclSpan decls;
  std::uint8_t *base;
};

struct Struct {
  static const ObjectType TYPE = ObjectType::STRUCT;

  /// RAW initialization. No write barriers!
  explicit Struct(gc::Ref<StructLayout> layout)
      : header(TYPE, layout.reinterpret<Object>()) {}

  StructLayout &layout() const noexcept {
    return header.layout()->as<StructLayout>();
  }

  std::size_t size() { return layout().instanceSize; }

  StructSlotSpan slots() noexcept;

  template <typename VisitorT, typename... Args>
  void walk(VisitorT &visitor, Args &&...args) {
    header.walk(visitor, std::forward<Args>(args)...);
    for (auto decl : layout().slotDeclSpan()) {
      if (decl.type == Type::ref) {
        visitor.visit(SlotProxy::fromPtr(data + decl.offset),
                      std::forward<Args>(args)...);
      }
    }
  }

  ObjectHeader header;
  std::uint8_t data[];
};

} // namespace om::om

#endif
