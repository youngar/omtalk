#include <ab/Util/Algorithm.h>
#include <om/Om/StructLayout.h>

om::om::StructLayout::StructLayout(
    gc::Ref<Object> meta, std::size_t instanceSize,
    const std::vector<SlotDecl> &slotDecls) noexcept
    : header(TYPE, meta), instanceSize(instanceSize),
      slotDeclCount(slotDecls.size()) {
  ab::copy(slotDecls, this->slotDecls);
}
