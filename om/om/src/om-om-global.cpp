#include <om/Om/Allocate.h>
#include <om/Om/Barrier.h>
#include <om/Om/Context.h>
#include <om/Om/Global.h>

om::om::Global::Global() : metaLayout(nullptr) {}

void om::om::Global::init(Context &context) noexcept {
  metaLayout = allocateMetaLayout(context);
}

auto om::om::Global::loadMetaLayout(Context &context) noexcept
    -> gc::Ref<MetaLayout> {
  return gc::load<MetaLayout>(context, metaLayout);
}
