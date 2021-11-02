#ifndef OM_OM_STRUCTLAYOUT_H
#define OM_OM_STRUCTLAYOUT_H

namespace om::om {

struct SlotDescription {
  std::uint32_t offset;
  Type type;
};

struct StructLayout {
  static const ObjectType TYPE = ObjectType::STRUCT_LAYOUT;

  static constexpr std::size_t
  allocSize(std::size_t instanceSlotCount) noexcept {
    return sizeof(StructLayout) + sizeof(SlotDescription) * instanceSlotCount;
  }

  /// The size of this object in bytes.
  std::size_t size() const noexcept { return allocSize(instanceSlotCount); }

  ObjectHeader header;
  std::uint32_t instanceSize;
  std::uint32_t instanceSlotCount;
  SlotDescription instanceSlotDescriptions[0];
};

} // namespace om::om

#endif // OM_OM_STRUCTLAYOUT_H
