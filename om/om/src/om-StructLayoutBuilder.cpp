#include <om/GC/Allocate.h>
#include <om/Om/Context.h>
#include <om/Om/MemoryManager.h>
#include <om/Om/StructLayoutBuilder.h>

auto om::om::StructLayoutBuilder::build(Context &context) const noexcept
    -> gc::Ref<StructLayout> {
  return gc::allocateZero<StructLayout>(
      context, StructLayout::allocSize(slotDecls.size()),
      [&](gc::Ref<void> allocation) {
        auto meta = context.getCollector()->getAuxData().global.metaLayout;

        new StructLayout(meta.reinterpret<Object>(), instanceSize(), slotDecls);
      });
}
