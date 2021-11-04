#include <om/Om/Barrier.h>
#include <om/Om/Context.h>
#include <om/Om/Global.h>

om::om::Global::Global() : metaLayout(nullptr) {}

auto om::om::Global::loadMetaLayout(Context &context) noexcept
    -> gc::Ref<MetaLayout> {
  return gc::load(context, metaLayout.proxy()).reinterpret<MetaLayout>();
}
