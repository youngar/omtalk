#ifndef OM_OM_STRUCTLAYOUT_H
#define OM_OM_STRUCTLAYOUT_H

#include <om/Om/ObjectHeader.h>
#include <om/Om/Type.h>

namespace om::om {

struct SlotDecl {
  std::uint32_t offset;
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
  static const ObjectType TYPE = ObjectType::STRUCT_LAYOUT;

  static std::size_t allocSize(std::size_t instanceSlotCount) noexcept {
    return sizeof(StructLayout) + sizeof(SlotDecl) * instanceSlotCount;
  }

  /// The size of this object in bytes.
  std::size_t size() const noexcept { return allocSize(slotDeclCount); }

  template <typename ContextT, typename VisitorT>
  void walk(ContextT &context, VisitorT &visitor) noexcept {
    header.walk(context, visitor);
  }

  /// A non-owning view over the slots declaration embedded in the tail of this
  /// layout object.
  SlotDeclSpan slotDeclSpan() const noexcept {
    return SlotDeclSpan(&slotDecls[0], slotDeclCount);
  }

  ObjectHeader header;
  std::uint32_t instanceSize;
  std::uint32_t slotDeclCount;
  SlotDecl slotDecls[0];
};

} // namespace om::om

#endif // OM_OM_STRUCTLAYOUT_H
