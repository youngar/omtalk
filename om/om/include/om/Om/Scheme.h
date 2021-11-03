#ifndef OMTALK_OM_SCHEME_H
#define OMTALK_OM_SCHEME_H

#include <cstdint>
#include <om/GC/Ref.h>
#include <om/Om/Layout.h>
#include <om/Om/Object.h>
#include <om/Om/ObjectHeader.h>

namespace om::om {

class ObjectProxy {
public:
  template <typename T>
  ObjectProxy(gc::Ref<T> target) : target(target) {}

  SlotProxy slot(unsigned offset) {
    return SlotProxy::fromPtr(&target->data + offset);
  }

private:
  gc::Ref<Object> target;
};

class Scheme {
  using ObjectProxy = ObjectProxy;
};

} // namespace om::om

#endif // OMTALK_OM_SCHEME_H
