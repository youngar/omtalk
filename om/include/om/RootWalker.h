#ifndef OM_ROOTWALKER_H_
#define OM_ROOTWALKER_H_

#include <om/Scheme.h>
#include <omtalk/MemoryManager.h>

namespace om {

template <>
struct gc::RootWalker<om::Scheme> {
public:
  template <typename ContextT, typename VisitorT>
  void walk(ContextT &cx, VisitorT &visitor) {
    for (auto &context : mm.getContexts()) {
      auto &scope = context.getAuxData().rootScope;
      for (auto *handle : scope) {
        std::cout << "!!! rootwalker: handle " << handle << std::endl;
        handle->walk(visitor, cx);
      }
    }
  }

  template <template <typename VisitorT>>
};

} // namespace om

#endif // OM_ROOTWALKER_H_
