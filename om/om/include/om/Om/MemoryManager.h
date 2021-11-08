#ifndef OM_OM_MEMORYMANAGER_H
#define OM_OM_MEMORYMANAGER_H

#include <om/GC/MemoryManager.h>
#include <om/Om/Context.h>
#include <om/Om/Global.h>
#include <om/Om/Scheme.h>

namespace om::gc {

template <>
class AuxMemoryManagerData<om::Scheme> {
public:
  om::Global global;
};

template <>
class RootWalker<om::Scheme> {
  template <typename VisitorT>
  void walk(GlobalCollectorContext<om::Scheme> &context,
            VisitorT &visitor) noexcept {
    auto &mm = context.getMemoryManager();
    mm.getAuxData().global.walk(context, visitor);
  }
};

} // namespace om::gc

namespace om::om {

using RootWalker = gc::RootWalker<Scheme>;
using MemoryManagerBuilder = gc::MemoryManagerBuilder<Scheme>;
using MemoryManager = gc::MemoryManager<Scheme>;

std::unique_ptr<RootWalker> makeRootWalker();
MemoryManager makeMemoryManager();

} // namespace om::om

#endif // OM_OM_MEMORYMANAGER_H
