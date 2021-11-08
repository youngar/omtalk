#ifndef OM_OM_STRUCTLAYOUT_H
#define OM_OM_STRUCTLAYOUT_H

#include <om/Om/ObjectHeader.h>
#include <om/Om/Type.h>

#include <vector>

namespace om::om {

class SlotLayoutBuilder;

struct SlotDecl {
  std::size_t offset;
  Type type;
};

/// A non-owning view over a contiguous span of slot declarations.
class SlotDeclSpan {
public:
  SlotDeclSpan(const SlotDecl *ptr, std::size_t length)
      : ptr(ptr), length(length) {}

  const SlotDecl *begin() const noexcept { return ptr; }

  const SlotDecl *end() const noexcept { return ptr + length; }

private:
  const SlotDecl *ptr;
  const std::size_t length;
};

struct StructLayout {
  friend class StructLayoutBuilder;

  static const ObjectType TYPE = ObjectType::STRUCT_LAYOUT;

  static std::size_t allocSize(std::size_t slotDeclCount) noexcept {
    return sizeof(StructLayout) + sizeof(SlotDecl) * slotDeclCount;
  }

  /// The size of this object in bytes.
  std::size_t size() const noexcept { return allocSize(slotDeclCount); }

  template <typename Visitor, typename... Args>
  void walk(Visitor &visitor, Args... args) noexcept {
    header.walk(visitor, args...);
  }

  /// A non-owning view over the slots declaration embedded in the tail of this
  /// layout object.
  SlotDeclSpan slotDeclSpan() const noexcept {
    return SlotDeclSpan(slotDecls, slotDeclCount);
  }

  ObjectHeader header;
  std::size_t instanceSize;
  std::size_t slotDeclCount;
  SlotDecl slotDecls[0];

private:
  /// Raw initialization.
  StructLayout(gc::Ref<Object> layout, std::size_t instanceSize,
               const std::vector<SlotDecl> &slotDecls) noexcept;
};

} // namespace om::om

#endif // OM_OM_STRUCTLAYOUT_H
