#ifndef OM_OM_BARRIER_H
#define OM_OM_BARRIER_H

#include <om/GC/Barrier.h>
#include <om/GC/Ref.h>
#include <om/Om/Context.h>
#include <om/Om/Object.h>
#include <om/Om/ObjectProxy.h>
#include <om/Om/Scheme.h>
#include <om/Om/SlotProxy.h>

namespace om::om {
namespace barrier {

/// Basic offset-based store operation.
inline void store(Context &context, gc::Ref<Object> object, std::size_t offset,
                  gc::Ref<void> value) {
  gc::store(context, ObjectProxy(object),
            SlotProxy::fromPtr(object->data + offset), value);
}

/// Basic offset-based load operation.
inline gc::Ref<void> load(Context &context, gc::Ref<Object> object,
                          std::size_t offset) {
  return gc::load(context, SlotProxy::fromPtr(object->data + offset));
}

} // namespace barrier
} // namespace om::om

#endif
