#ifndef OM_OM_LAYOUT_H
#define OM_OM_LAYOUT_H

#include <cstdint>
#include <om/Om/Object.h>
#include <om/Om/ObjectHeader.h>
#include <om/Om/Type.h>

namespace om::om {

/// A special layout that describes the shape of layouts.
/// Rather than an object, MetaLayout is a static global.
struct MetaLayout {};

struct SlotDescription {
  std::size_t offset;
  Type type;
};

/// Layouts describe the shape of other objects.
struct Layout {
  static constexpr std::size_t allocSize(std::size_t length) noexcept {
    return sizeof(Layout) + sizeof(SlotDescription) * length;
  }

  std::size_t allocSize() const noexcept { return Layout::allocSize(length); }

  ObjectHeader header;
  std::size_t length;
  SlotDescription slots[0];
};

} // namespace om::om

#endif // OM_OM_LAYOUT_H
