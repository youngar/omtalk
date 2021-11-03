#ifndef OM_OM_CONTEXT_H
#define OM_OM_CONTEXT_H

#include <om/GC/Handle.h>
#include <om/GC/MemoryManager.h>
#include <om/Om/Scheme.h>

namespace om::gc {

/// om-specific extensions to the gc context.
template <>
class AuxContextData<om::Scheme> {
public:
  gc::RootHandleScope rootScope;
};

} // namespace om::gc

namespace om::om {

using Context = gc::Context<Scheme>;

} // namespace om::om

#endif // OM_OM_CONTEXT_H
