#ifndef OM_OM_H_
#define OM_OM_H_

#include <om/NamespaceAliases.h>

namespace om {

class ObjectProxy;

/// The GC Scheme.
struct Scheme {
  using ObjectProxy = om::ObjectProxy;
};

} // namespace om

#endif // OM_OM_H_
