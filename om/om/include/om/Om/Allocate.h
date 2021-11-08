#ifndef OM_OM_ALLOC_H
#define OM_OM_ALLOC_H

#include <om/GC/Allocate.h>
#include <om/Om/Barrier.h>
#include <om/Om/Context.h>
#include <om/Om/Scheme.h>

namespace om::om {

inline gc::Ref<MetaLayout> allocateMetaLayout(Context &context) noexcept {
  return gc::allocateZero(context, MetaLayout::allocSize(),
                          [=](auto object) { new (object.get()) MetaLayout(); })
      .reinterpret<MetaLayout>();
}

inline gc::Ref<Struct> allocateStruct(Context &context,
                                      gc::Ref<StructLayout> layout) noexcept {
  std::size_t size = layout->instanceSize;

  // fast path
  {
    auto ref = gc::allocateBytesZeroFast(context, size);
    if (ref) {
      auto object = gc::Ref<Struct>::fromPtr(new (ref.get()) Struct(layout));
      gc::allocateBarrier(context, object);
      return object;
    }
  }

  // slow path
  {
    gc::Handle layoutHandle(context.getAuxData().rootScope, layout);
    auto [ref, tax] = gc::allocateBytesZeroSlow(context, size);
    if (ref) {
      auto object = gc::Ref<Struct>::fromPtr(new (ref.get()) Struct(layout));
      allocateBarrier(context, object);
      if (tax) {
        gc::pay(context, tax);
      }
      return object;
    }
  }

  return nullptr;
}

} // namespace om::om

#endif // OM_OM_ALLOC_H
