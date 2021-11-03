#ifndef OMTALK_OM_SCHEME_H
#define OMTALK_OM_SCHEME_H

#include <cstdint>
#include <om/GC/Ref.h>
#include <om/Om/Layout.h>
#include <om/Om/Object.h>
#include <om/Om/ObjectHeader.h>
#include <om/Om/ObjectProxy.h>

namespace om::om {

struct Scheme {
  using ObjectProxy = ObjectProxy;
};

} // namespace om::om

#endif // OMTALK_OM_SCHEME_H
